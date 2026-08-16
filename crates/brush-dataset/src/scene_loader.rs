use std::sync::Arc;

use brush_async::Actor;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::{
    config::LoadDatasetConfig,
    scene::{Scene, SceneBatch, sample_to_packed_data, view_to_sample_image},
};

/// Shared cache of GPU-ready scene batches. Each slot holds at most one
/// batch; once the running total passes `budget_bytes`, new batches bypass
/// the cache and just get re-decoded + re-packed on every visit.
///
/// Caching the packed batch (instead of the decoded `DynamicImage`) skips
/// the per-hit decode → premultiply → repack work: a cache hit is now a
/// single copy of the already-packed `[H, W]` u32 buffer.
struct BatchCache {
    slots: Vec<Option<Arc<SceneBatch>>>,
    used_bytes: u64,
    budget_bytes: u64,
}

impl BatchCache {
    fn new(n_views: usize, budget_bytes: u64) -> Self {
        Self {
            slots: vec![None; n_views],
            used_bytes: 0,
            budget_bytes,
        }
    }

    fn get(&self, index: usize) -> Option<Arc<SceneBatch>> {
        self.slots[index].clone()
    }

    fn insert(&mut self, index: usize, batch: Arc<SceneBatch>) -> bool {
        if self.slots[index].is_some() {
            return true;
        }
        // Track exact bytes: rounding to whole MB let sub-MB images slip in
        // for free and bypass the budget entirely.
        let size_bytes: u64 = batch
            .img_packed
            .as_bytes()
            .len()
            .try_into()
            .expect("shouldn't exceed ~18 Exabytes...");
        if self.used_bytes + size_bytes < self.budget_bytes {
            self.slots[index] = Some(batch);
            self.used_bytes += size_bytes;
            true
        } else {
            false
        }
    }
}

/// A cache that begins preparing scene batches immediately and can be handed
/// to a later [`SceneLoader`]. Progressive training uses this to overlap the
/// full-resolution decode/resize/pack work with its lower-resolution phase
/// without changing the later loader's seeded view order.
pub struct SceneLoaderCache {
    cache: Arc<Mutex<BatchCache>>,
    // Keep the exact scene identity alongside the index-keyed cache so a
    // same-sized but different scene cannot accidentally consume its batches.
    views: Arc<Vec<crate::scene::SceneView>>,
    // Owns the prefetch actor. Dropping the cache cancels unfinished work.
    _actor: Actor,
}

impl SceneLoaderCache {
    pub fn prefetch(scene: &Scene, config: &LoadDatasetConfig) -> Self {
        let views = scene.views.clone();
        let cache = Arc::new(Mutex::new(BatchCache::new(
            views.len(),
            config.max_scene_batch_cache_size,
        )));
        let actor = Actor::new("dataloader-prefetch");
        let actor_views = views.clone();
        let actor_cache = cache.clone();
        actor
            .run(move || prefetch_batches(actor_views, actor_cache))
            .detach();
        Self {
            cache,
            views,
            _actor: actor,
        }
    }
}

pub struct SceneLoader {
    rx: mpsc::Receiver<SceneBatch>,
    // Owns the loader actor threads. Dropping cancels them; their
    // senders then drop, the channel closes, and `next_batch` returns.
    _actors: Vec<Actor>,
}

impl SceneLoader {
    pub fn new(scene: &Scene, seed: u64, config: &LoadDatasetConfig) -> Self {
        Self::new_with_mode(scene, seed, config, false, None)
    }

    pub fn new_with_cache(
        scene: &Scene,
        seed: u64,
        config: &LoadDatasetConfig,
        cache: &SceneLoaderCache,
    ) -> Self {
        Self::new_with_mode(scene, seed, config, false, Some(cache))
    }

    /// Deterministic host ordering uses one producer task so async completion
    /// order cannot reshuffle otherwise seeded samples.
    pub fn new_deterministic(scene: &Scene, seed: u64, config: &LoadDatasetConfig) -> Self {
        Self::new_with_mode(scene, seed, config, true, None)
    }

    pub fn new_deterministic_with_cache(
        scene: &Scene,
        seed: u64,
        config: &LoadDatasetConfig,
        cache: &SceneLoaderCache,
    ) -> Self {
        Self::new_with_mode(scene, seed, config, true, Some(cache))
    }

    fn new_with_mode(
        scene: &Scene,
        seed: u64,
        config: &LoadDatasetConfig,
        deterministic: bool,
        prefetched_cache: Option<&SceneLoaderCache>,
    ) -> Self {
        // Prefetch buffer: at most 4 batches ahead of the trainer.
        // Two tasks per actor share this buffer so one task's I/O can
        // overlap with the other's decode + GPU upload.
        let (tx, rx) = mpsc::channel(4);

        // Fan out only as many loaders as we have real parallelism.
        // Wasm shares one JS event loop, so extra actors just add
        // contention without overlapping I/O.
        let n_actors = if deterministic || cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism().map_or(8, |p| p.get())
        };
        let tasks_per_actor = if deterministic { 1 } else { 2 };

        let views = scene.views.clone();
        let cache = if let Some(prefetched_cache) = prefetched_cache {
            assert!(
                Arc::ptr_eq(&prefetched_cache.views, &views),
                "prefetched cache must belong to the same scene"
            );
            prefetched_cache.cache.clone()
        } else {
            Arc::new(Mutex::new(BatchCache::new(
                views.len(),
                config.max_scene_batch_cache_size,
            )))
        };

        let mut task_idx: u64 = 0;
        let actors: Vec<Actor> = (0..n_actors)
            .map(|i| {
                let actor = Actor::new(&format!("dataloader-{i}"));
                for _ in 0..tasks_per_actor {
                    let views = views.clone();
                    let cache = cache.clone();
                    let tx = tx.clone();
                    let task_seed = seed.wrapping_add(task_idx);
                    task_idx += 1;
                    actor
                        .run(move || run_loader(views, cache, tx, task_seed))
                        .detach();
                }
                actor
            })
            .collect();

        Self {
            rx,
            _actors: actors,
        }
    }

    pub async fn next_batch(&mut self) -> SceneBatch {
        self.rx
            .recv()
            .await
            .expect("Scene loader channel closed unexpectedly")
    }
}

async fn run_loader(
    views: Arc<Vec<crate::scene::SceneView>>,
    cache: Arc<Mutex<BatchCache>>,
    tx: mpsc::Sender<SceneBatch>,
    seed: u64,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut shuffled: Vec<usize> = Vec::new();

    loop {
        if shuffled.is_empty() {
            shuffled = (0..views.len()).collect();
            shuffled.shuffle(&mut rng);
        }
        let index = shuffled.pop().expect("Need at least one view in dataset");
        let view = &views[index];

        let batch = if let Some(batch) = cache.lock().await.get(index) {
            batch
        } else {
            let raw = view
                .image
                .load()
                .await
                .expect("Scene loader failed to load an image");
            let sample = view_to_sample_image(raw, view.image.alpha_mode());
            let (img_packed, has_alpha) = sample_to_packed_data(sample);
            let batch = Arc::new(SceneBatch {
                img_packed,
                has_alpha,
                alpha_mode: view.image.alpha_mode(),
                camera: view.camera,
            });
            cache.lock().await.insert(index, batch.clone());
            batch
        };

        // The channel takes an owned batch; clone the packed buffer out of
        // the shared cache entry.
        if tx.send(batch.as_ref().clone()).await.is_err() {
            break;
        }
        brush_async::yield_now().await;
    }
}

async fn prefetch_batches(views: Arc<Vec<crate::scene::SceneView>>, cache: Arc<Mutex<BatchCache>>) {
    for (index, view) in views.iter().enumerate() {
        if cache.lock().await.get(index).is_some() {
            continue;
        }
        let raw = match view.image.load().await {
            Ok(raw) => raw,
            Err(error) => {
                log::warn!(
                    "Scene loader prefetch failed for {}: {error}",
                    view.image.img_name()
                );
                continue;
            }
        };
        let sample = view_to_sample_image(raw, view.image.alpha_mode());
        let (img_packed, has_alpha) = sample_to_packed_data(sample);
        let batch = Arc::new(SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: view.image.alpha_mode(),
            camera: view.camera,
        });
        if !cache.lock().await.insert(index, batch) {
            break;
        }
        brush_async::yield_now().await;
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use brush_render::camera::Camera;
    use brush_vfs::BrushVfs;
    use image::{Rgb, RgbImage};
    use std::{
        fs,
        path::PathBuf,
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{load_image::LoadImage, scene::SceneView};

    fn test_config() -> LoadDatasetConfig {
        LoadDatasetConfig {
            max_frames: None,
            max_resolution: 16,
            eval_split_every: None,
            subsample_frames: None,
            subsample_points: None,
            alpha_mode: None,
            max_scene_batch_cache_size: 1 << 20,
        }
    }

    #[tokio::test]
    async fn prefetched_cache_is_scene_bound_and_survives_source_removal() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("brush-scene-prefetch-{unique}"));
        fs::create_dir(&root).unwrap();
        let paths = [PathBuf::from("red.png"), PathBuf::from("green.png")];
        for (path, color) in paths.iter().zip([Rgb([255, 0, 0]), Rgb([0, 255, 0])]) {
            RgbImage::from_pixel(16, 8, color)
                .save(root.join(path))
                .unwrap();
        }

        let vfs = Arc::new(BrushVfs::from_path(&root).await.unwrap());
        let scene = Scene::new(
            paths
                .iter()
                .map(|path| SceneView {
                    image: LoadImage::new(vfs.clone(), path.clone(), None, 16, None),
                    camera: Camera::default(),
                })
                .collect(),
        );
        let config = test_config();
        let cache = SceneLoaderCache::prefetch(&scene, &config);

        for _ in 0..10_000 {
            if cache.cache.lock().await.slots.iter().all(Option::is_some) {
                break;
            }
            brush_async::yield_now().await;
        }
        assert!(cache.cache.lock().await.slots.iter().all(Option::is_some));

        let same_size_different_scene = Scene::new(scene.views.as_ref().clone());
        let wrong_scene_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SceneLoader::new_deterministic_with_cache(
                &same_size_different_scene,
                0,
                &config,
                &cache,
            )
        }));
        assert!(wrong_scene_result.is_err());

        for path in &paths {
            fs::remove_file(root.join(path)).unwrap();
        }
        let mut loader = SceneLoader::new_deterministic_with_cache(&scene, 0, &config, &cache);
        let first = loader.next_batch().await;
        let second = loader.next_batch().await;
        assert_eq!(first.img_size(), [8, 16]);
        assert_eq!(second.img_size(), [8, 16]);

        drop(loader);
        drop(cache);
        fs::remove_dir(root).unwrap();
    }
}
