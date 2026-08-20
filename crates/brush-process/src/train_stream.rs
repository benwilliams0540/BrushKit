use crate::{
    Emitter,
    config::TrainStreamConfig,
    message::{
        INITIALIZER_FIELD_LOG_SCALES, INITIALIZER_FIELD_MEANS, INITIALIZER_FIELD_OPACITY,
        INITIALIZER_FIELD_ROTATIONS, INITIALIZER_FIELD_SH, InitializerReport, InitializerRoute,
        ProcessMessage, TelemetryBoundary, TrainMessage,
    },
    progressive_sh::{PROGRESSIVE_SH_MODE_INTERVAL, ShTransitionReason},
    slot::SlotSender,
    wait_for_device,
};
use anyhow::Context;
use brush_dataset::{
    load_dataset,
    scene::Scene,
    scene_loader::{SceneLoader, SceneLoaderCache},
};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::sh::sh_coeffs_for_degree;
use brush_rerun::visualize_tools::VisualizeTools;
use brush_serde::SplatData;
use brush_train::{
    RandomSplatsConfig, create_random_splats,
    eval::eval_stats,
    lod::{compute_pup_scores, decimate_to_count},
    msg::RefineStats,
    to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use brush_vfs::BrushVfs;
use burn::module::AutodiffModule;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{AutoCompiler, WgpuRuntime};
use rand::SeedableRng;
use std::{path::PathBuf, sync::Arc};

#[allow(unused)]
use std::path::Path;

use tracing::{Instrument, trace_span};
use web_time::{Duration, Instant};

#[allow(clippy::large_stack_frames)]
pub(crate) async fn train_stream(
    vfs: Arc<BrushVfs>,
    train_stream_config: TrainStreamConfig,
    emitter: &Emitter,
    slot: SlotSender<Splats>,
) -> anyhow::Result<()> {
    log::info!("Start of training stream");

    let visualize = VisualizeTools::new(train_stream_config.rerun_config.rerun_enabled).await;

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
            config: Box::new(train_stream_config.clone()),
        }))
        .await;

    let process_config = &train_stream_config.process_config;
    let progressive_sh_schedule = train_stream_config
        .host_runtime
        .progressive_sh_schedule
        .clone();
    if let Some(schedule) = &progressive_sh_schedule {
        anyhow::ensure!(
            schedule.maximum_degree() == train_stream_config.model_config.sh_degree,
            "progressive SH maximum degree does not match configured model degree"
        );
        anyhow::ensure!(
            schedule.total_iterations() == train_stream_config.train_config.total_train_iters,
            "progressive SH total iterations do not match training configuration"
        );
        anyhow::ensure!(
            process_config.start_iter == 0,
            "progressive SH optimizer-state resume is unsupported; refusing to reset optimizer state"
        );
    }
    log::info!("Using seed {}", process_config.seed);

    let wgpu_device = wait_for_device().await;
    // Splats live on the inner (non-autodiff) device between steps; each
    // training step lifts them via [`lift_splats_to_autodiff`] then strips
    // back via `.valid()`. Going through `Module::train()` would hit
    // burn-dispatch's `from_inner` checkpointing bug.
    let device: burn::tensor::Device = wgpu_device.clone().into();
    device.seed(process_config.seed);
    let mut rng = match train_stream_config.host_runtime.deterministic_seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_seed([process_config.seed as u8; 32]),
    };

    log::info!("Loading dataset");
    if train_stream_config.host_runtime.phase0_telemetry {
        emitter
            .emit(ProcessMessage::TrainMessage(
                TrainMessage::TelemetryBoundary {
                    boundary: TelemetryBoundary::DatasetLoadStarted,
                },
            ))
            .await;
    }
    let load_result = load_dataset(
        vfs.clone(),
        &train_stream_config.load_config,
        train_stream_config
            .host_runtime
            .explicit_initializer_path
            .as_deref(),
    )
    .instrument(trace_span!("Load dataset"))
    .await?;
    if train_stream_config.host_runtime.phase0_telemetry {
        emitter
            .emit(ProcessMessage::TrainMessage(
                TrainMessage::TelemetryBoundary {
                    boundary: TelemetryBoundary::DatasetLoadFinished,
                },
            ))
            .await;
    }

    // Emit any warnings from dataset loading.
    for warning in load_result.warnings {
        emitter
            .emit(ProcessMessage::Warning {
                error: anyhow::anyhow!("{warning}"),
            })
            .await;
    }

    let dataset = load_result.dataset;

    log::info!("Log scene to rerun");
    if let Err(error) = visualize.log_scene(
        &dataset.train,
        train_stream_config.rerun_config.rerun_max_img_size,
    ) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    let num_eval_views = dataset.eval.as_ref().map_or(0, |s| s.views.len());
    if let Err(error) = visualize.send_default_blueprint(num_eval_views) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    log::info!("Dataset loaded");
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
            dataset: dataset.clone(),
        }))
        .await;

    log::info!("Loading initial splats if any.");
    let estimated_up = dataset.estimate_up();
    if train_stream_config.host_runtime.phase0_telemetry {
        emitter
            .emit(ProcessMessage::TrainMessage(
                TrainMessage::TelemetryBoundary {
                    boundary: TelemetryBoundary::TrainerInitializationStarted,
                },
            ))
            .await;
    }

    // Convert SplatData to Splats using KNN initialization
    let (up_axis, init_splats) = if let Some(msg) = load_result.init_splat {
        if let Some(schedule) = &progressive_sh_schedule
            && process_config.start_iter > 0
        {
            let checkpoint = msg.meta.progressive_sh_checkpoint.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "progressive SH resume requires compatible checkpoint schedule identity"
                )
            })?;
            schedule
                .validate_checkpoint(checkpoint, process_config.start_iter)
                .map_err(anyhow::Error::msg)?;
        }
        // Use loaded splats with KNN init
        let render_mode = train_stream_config
            .train_config
            .render_mode
            .or(msg.meta.render_mode)
            .unwrap_or(SplatRenderMode::Default);
        let max_splats = train_stream_config.train_config.max_splats as usize;
        let original = msg.data.num_splats();
        let strong_report = train_stream_config
            .host_runtime
            .require_strong_initializer
            .then(|| {
                validate_strong_initializer(
                    &msg.data,
                    load_result.init_splat_path.clone(),
                    max_splats,
                    train_stream_config.model_config.sh_degree,
                )
            })
            .transpose()?;
        let data = if strong_report.is_some() {
            msg.data
        } else {
            msg.data.subsample(max_splats)
        };
        if data.num_splats() < original {
            emitter
                .emit(ProcessMessage::Warning {
                    error: anyhow::anyhow!(
                        "Initial point cloud has {original} points, exceeding --max-splats ({max_splats}). Subsampled to {}; the remaining points were discarded. Raise --max-splats to keep more.",
                        data.num_splats()
                    ),
                })
                .await;
        }
        let splats = to_init_splats(data, render_mode, &device);
        if train_stream_config.host_runtime.phase0_telemetry {
            let report = strong_report.unwrap_or_else(|| InitializerReport {
                route: if train_stream_config
                    .host_runtime
                    .explicit_initializer_path
                    .is_some()
                {
                    InitializerRoute::ExplicitPly
                } else if load_result.init_splat_path.is_some() {
                    InitializerRoute::ImplicitPly
                } else {
                    InitializerRoute::DatasetSparse
                },
                path: load_result.init_splat_path.clone(),
                primitive_count: splats.num_splats(),
                fields_consumed: 0,
                rejected_count: original.saturating_sub(splats.num_splats() as usize) as u32,
                supplied_sh_degree: Some(splats.sh_degree()),
                configured_sh_degree: train_stream_config.model_config.sh_degree,
            });
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::Initializer {
                    report,
                }))
                .await;
        }
        (msg.meta.up_axis, splats)
    } else {
        if train_stream_config.host_runtime.require_strong_initializer {
            anyhow::bail!("required strong initializer was not found");
        }
        // Default: just use random splats
        let render_mode = train_stream_config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);
        log::info!("Starting with random splat config.");
        let cameras: Vec<_> = dataset.train.views.iter().map(|v| v.camera).collect();
        let config = RandomSplatsConfig::new();
        let scene_scale = train_stream_config.train_config.random_init_scene_scale;
        let splats = create_random_splats(
            &config,
            &cameras,
            scene_scale,
            &mut rng,
            render_mode,
            &device,
        );
        if train_stream_config.host_runtime.phase0_telemetry {
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::Initializer {
                    report: InitializerReport {
                        route: InitializerRoute::Random,
                        path: None,
                        primitive_count: splats.num_splats(),
                        fields_consumed: 0,
                        rejected_count: 0,
                        supplied_sh_degree: None,
                        configured_sh_degree: train_stream_config.model_config.sh_degree,
                    },
                }))
                .await;
        }
        (None, splats)
    };

    let init_splats = init_splats.with_sh_degree(train_stream_config.model_config.sh_degree);

    // If the metadata has an up axis prefer that, otherwise estimate the up direction.
    let up_axis = up_axis.or(Some(estimated_up));

    // The trainer owns its working `splats` locally and publishes a
    // clone to the `Slot` after every modification (train
    // step, refine, LOD decimation).
    let mut splats: Splats = init_splats.clone();
    #[cfg(feature = "trainer-diagnostics")]
    if train_stream_config.host_runtime.phase0_telemetry {
        emit_trainer_finiteness_diagnostic(&splats, 0, "post_import").await;
    }
    slot.set(0, splats.clone());
    emitter
        .emit(ProcessMessage::SplatsUpdated {
            up_axis,
            frame: 0,
            total_frames: 1,
            num_splats: init_splats.num_splats(),
            sh_degree: init_splats.sh_degree(),
        })
        .await;

    emitter.emit(ProcessMessage::DoneLoading).await;

    // Start with memory cleared out.
    let client = WgpuRuntime::<AutoCompiler>::client(wgpu_device);
    client.memory_cleanup();

    let mut eval_scene = dataset.eval;

    let mut train_duration = Duration::from_secs(0);
    let dataloader_seed = train_stream_config
        .host_runtime
        .deterministic_seed
        .unwrap_or(42);
    let make_dataloader = |scene: &Scene, cache: Option<&SceneLoaderCache>| match (
        train_stream_config
            .host_runtime
            .deterministic_seed
            .is_some(),
        cache,
    ) {
        (true, Some(cache)) => SceneLoader::new_deterministic_with_cache(
            scene,
            dataloader_seed,
            &train_stream_config.load_config,
            cache,
        ),
        (true, None) => {
            SceneLoader::new_deterministic(scene, dataloader_seed, &train_stream_config.load_config)
        }
        (false, Some(cache)) => SceneLoader::new_with_cache(
            scene,
            dataloader_seed,
            &train_stream_config.load_config,
            cache,
        ),
        (false, None) => SceneLoader::new(scene, dataloader_seed, &train_stream_config.load_config),
    };
    let progressive_start_percent = train_stream_config
        .host_runtime
        .progressive_resolution_start_percent;
    let progressive_switch_iteration = train_stream_config
        .host_runtime
        .progressive_resolution_switch_iteration;
    let progressive_resolution_enabled = progressive_start_percent < 100
        && progressive_switch_iteration > process_config.start_iter
        && progressive_switch_iteration < train_stream_config.train_config.total_train_iters;
    let progressive_scene = progressive_resolution_enabled.then(|| {
        dataset
            .train
            .clone()
            .with_image_scale(progressive_start_percent as f32 / 100.0)
    });
    let mut dataloader =
        make_dataloader(progressive_scene.as_ref().unwrap_or(&dataset.train), None);
    let full_resolution_cache = progressive_resolution_enabled.then(|| {
        log::info!("Progressive resolution: prefetching the full-resolution batch cache");
        SceneLoaderCache::prefetch(&dataset.train, &train_stream_config.load_config)
    });
    if progressive_resolution_enabled {
        log::info!(
            "Progressive resolution: {progressive_start_percent}% through iteration {}, then 100%",
            progressive_switch_iteration.saturating_sub(1)
        );
    }
    let bounds = get_splat_bounds(init_splats.clone(), BOUND_PERCENTILE).await;

    // Per-train-view (world center, focal-px at native res) for the
    // Mip-Splatting 3D filter (always on).
    let mut view_cams: Vec<(glam::Vec3, f32)> = Vec::with_capacity(dataset.train.views.len());
    for view in dataset.train.views.iter() {
        let (w, h) = view.image.dimensions().await.unwrap_or((1, 1));
        let focal = view.camera.focal(glam::uvec2(w, h)).x;
        view_cams.push((view.camera.position, focal));
    }

    let mut trainer = SplatTrainer::new(&train_stream_config.train_config, &device, bounds);
    trainer.set_view_cams(view_cams.clone());
    trainer.set_deterministic_seed(train_stream_config.host_runtime.deterministic_seed);
    trainer.set_phase0_host_telemetry(train_stream_config.host_runtime.phase0_telemetry);
    if let Some(schedule) = &progressive_sh_schedule {
        let active_degree = schedule.active_degree_at(process_config.start_iter);
        trainer.set_active_sh_degree(
            active_degree,
            schedule.mode() == PROGRESSIVE_SH_MODE_INTERVAL,
        );
        emitter
            .emit(ProcessMessage::TrainMessage(
                TrainMessage::ShDegreeChanged {
                    zero_based_iteration: process_config.start_iter,
                    maximum_degree: schedule.maximum_degree(),
                    active_degree,
                    initial_degree: schedule.initial_degree(),
                    step_interval: schedule.step_interval(),
                    schedule_mode: schedule.mode(),
                    schedule_identity: schedule.identity(),
                    reason: if process_config.start_iter > 0 {
                        ShTransitionReason::Resume
                    } else if schedule.mode() == PROGRESSIVE_SH_MODE_INTERVAL {
                        ShTransitionReason::Initial
                    } else {
                        ShTransitionReason::Disabled
                    },
                },
            ))
            .await;
    }
    if train_stream_config.host_runtime.phase0_telemetry {
        emitter
            .emit(ProcessMessage::TrainMessage(
                TrainMessage::TelemetryBoundary {
                    boundary: TelemetryBoundary::TrainerInitializationFinished,
                },
            ))
            .await;
    }

    // Get the dataset name from the base path (if available) for interpolation.
    let dataset_name = vfs
        .base_path()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "dataset".to_owned());

    // Interpolate {dataset} in the export path.
    let export_path_str = train_stream_config
        .process_config
        .export_path
        .replace("{dataset}", &dataset_name);

    // Resolve relative to the dataset's parent directory if available, otherwise CWD.
    let base_path = vfs
        .base_path()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let export_path = base_path.join(&export_path_str);
    // Normalize path components
    let export_path: PathBuf = export_path.components().collect();
    let sh_degree = init_splats.sh_degree();

    let training_steps = train_stream_config.train_config.total_train_iters;
    let lod_levels = train_stream_config.train_config.lod_levels;
    let lod_refine_steps = train_stream_config.train_config.lod_refine_steps;
    let mut current_lod: u32 = 0;

    let process_config = &train_stream_config.process_config;

    log::info!("Start training loop.");
    for iter in process_config.start_iter..train_stream_config.train_config.total_iters() {
        if iter != process_config.start_iter
            && let Some(schedule) = &progressive_sh_schedule
            && let Some(active_degree) = schedule.transition_degree_at(iter)
        {
            trainer.set_active_sh_degree(
                active_degree,
                schedule.mode() == PROGRESSIVE_SH_MODE_INTERVAL,
            );
            emitter
                .emit(ProcessMessage::TrainMessage(
                    TrainMessage::ShDegreeChanged {
                        zero_based_iteration: iter,
                        maximum_degree: schedule.maximum_degree(),
                        active_degree,
                        initial_degree: schedule.initial_degree(),
                        step_interval: schedule.step_interval(),
                        schedule_mode: schedule.mode(),
                        schedule_identity: schedule.identity(),
                        reason: ShTransitionReason::Scheduled,
                    },
                ))
                .await;
        }
        if progressive_resolution_enabled && iter == progressive_switch_iteration {
            dataloader = make_dataloader(&dataset.train, full_resolution_cache.as_ref());
            log::info!("Progressive resolution: switched to 100% at iteration {iter}");
        }
        let target_lod = if lod_levels == 0 || iter < training_steps {
            0u32
        } else {
            ((iter - training_steps) / lod_refine_steps + 1).min(lod_levels)
        };

        if target_lod > current_lod {
            #[cfg(not(target_family = "wasm"))]
            {
                let (name, exp_iter, exp_total) = if current_lod == 0 {
                    (process_config.export_name.clone(), iter, training_steps)
                } else {
                    let lod_name = process_config
                        .export_name
                        .replace(".ply", &format!("_lod{current_lod}.ply"));
                    (lod_name, lod_refine_steps, lod_refine_steps)
                };
                if train_stream_config.host_runtime.phase0_telemetry {
                    emitter
                        .emit(ProcessMessage::TrainMessage(
                            TrainMessage::CheckpointExportStarted { iter: exp_iter },
                        ))
                        .await;
                }
                let checkpoint_metadata = progressive_sh_schedule
                    .as_ref()
                    .map(|schedule| schedule.checkpoint_metadata(exp_iter));
                let res = export_checkpoint(
                    splats.clone(),
                    &export_path,
                    &name,
                    exp_iter,
                    exp_total,
                    checkpoint_metadata,
                )
                .await
                .with_context(|| "Export at LOD boundary failed");

                match res {
                    Ok((path, _report)) => {
                        emitter
                            .emit(ProcessMessage::TrainMessage(
                                TrainMessage::CheckpointExported {
                                    iter: exp_iter,
                                    path,
                                },
                            ))
                            .await;
                    }
                    Err(error) => {
                        emitter.emit(ProcessMessage::Warning { error }).await;
                    }
                }
            }

            current_lod = target_lod;
            let lod_keep_pct = train_stream_config.train_config.lod_decimation_keep;
            let lod_img_pct = train_stream_config.train_config.lod_image_scale;

            log::info!("LOD {current_lod}/{lod_levels}: Decimating (keep {lod_keep_pct}%)");

            let before = splats.num_splats();
            let target_count = (before as f32 * lod_keep_pct as f32 / 100.0).max(1.0) as u32;

            log::info!("LOD {current_lod}/{lod_levels}: Computing sensitivity scores...");
            let scores = compute_pup_scores(splats.clone(), &dataset.train, &device).await;
            splats = decimate_to_count(splats, &scores, target_count).await;
            slot.set(0, splats.clone());

            let after = splats.num_splats();
            log::info!("LOD {current_lod}/{lod_levels}: {before} -> {after} splats");

            let client = WgpuRuntime::<AutoCompiler>::client(wgpu_device);
            client.memory_cleanup();

            let cumulative_scale = (lod_img_pct as f32 / 100.0).powi(current_lod as i32);
            dataloader = if lod_img_pct < 100 {
                let lod_scene = dataset.train.clone().with_image_scale(cumulative_scale);
                make_dataloader(&lod_scene, None)
            } else {
                make_dataloader(&dataset.train, None)
            };

            let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
            trainer = SplatTrainer::new(&train_stream_config.train_config, &device, bounds);
            trainer.set_view_cams(view_cams.clone());
            trainer.set_deterministic_seed(train_stream_config.host_runtime.deterministic_seed);
            trainer.set_phase0_host_telemetry(train_stream_config.host_runtime.phase0_telemetry);
            if let Some(schedule) = &progressive_sh_schedule {
                trainer.set_active_sh_degree(
                    schedule.active_degree_at(iter),
                    schedule.mode() == PROGRESSIVE_SH_MODE_INTERVAL,
                );
            }

            log::info!(
                "LOD {current_lod}/{lod_levels}: Training for {lod_refine_steps} steps (image scale {:.0}%)",
                cumulative_scale * 100.0
            );
        }

        let step_time = Instant::now();

        let data_wait_start = Instant::now();
        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("Wait for next data batch"))
            .await;
        let data_wait_duration = data_wait_start.elapsed();

        // Lift splats onto the autodiff graph for this step, run training,
        // then strip back to inner so the viewer slot sees plain splats.
        let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats.clone());
        let (new_diff_splats, stats) = trainer.step(batch, diff_splats).await;
        splats = new_diff_splats.valid();
        slot.set(0, splats.clone());
        #[cfg(feature = "trainer-diagnostics")]
        if train_stream_config.host_runtime.phase0_telemetry {
            emit_trainer_finiteness_diagnostic(&splats, iter + 1, "post_optimizer").await;
        }

        // Phase-local iteration for refine gating
        let phase_iter = if current_lod == 0 {
            iter
        } else {
            (iter - training_steps) % lod_refine_steps
        };
        let phase_total = if current_lod == 0 {
            training_steps
        } else {
            lod_refine_steps
        };
        let phase_progress = (phase_iter as f32 / phase_total as f32).clamp(0.0, 1.0);

        let refine_start = Instant::now();
        let did_refine = phase_iter > 0
            && phase_iter.is_multiple_of(train_stream_config.train_config.refine_every)
            && phase_progress <= 0.95;
        let refine = if did_refine {
            let (new_splats, refine_stats) = trainer.refine(iter, splats).await;
            splats = new_splats;
            slot.set(0, splats.clone());
            #[cfg(feature = "trainer-diagnostics")]
            if train_stream_config.host_runtime.phase0_telemetry {
                emit_trainer_finiteness_diagnostic(&splats, iter + 1, "post_refine").await;
            }
            refine_stats
        } else {
            RefineStats {
                num_added: 0,
                num_split_oversized: 0,
                num_split_high_grad: 0,
                num_pruned: 0,
                num_pruned_non_finite: 0,
                total_splats: splats.num_splats(),
            }
        };
        let refine_dur = refine_start.elapsed();

        // We just finished iter 'iter', now starting iter + 1.
        let iter = iter + 1;
        let is_last_step = iter == train_stream_config.train_config.total_iters();

        let step_dur = step_time.elapsed();
        train_duration += step_dur;

        // Do evals. We skip this for LODs as it'd be confusing for rerun, but, could
        // revisit this.
        if current_lod == 0
            && (iter % process_config.eval_every == 0 || iter == training_steps)
            && let Some(eval_scene) = eval_scene.as_mut()
        {
            let save_path = train_stream_config
                .process_config
                .eval_save_to_disk
                .then(|| export_path.clone());

            let eval = run_eval(
                &device,
                emitter,
                &visualize,
                splats.clone(),
                iter,
                eval_scene,
                save_path,
                train_stream_config.rerun_config.rerun_max_img_size,
            )
            .await
            .with_context(|| format!("Failed evaluation at iteration {iter}"));

            if let Err(error) = eval {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
        }

        // Export checkpoints
        #[cfg(not(target_family = "wasm"))]
        {
            let should_export = if current_lod == 0 {
                iter % process_config.export_every == 0 || (is_last_step && lod_levels == 0)
            } else {
                is_last_step
            };
            if should_export {
                let (name, exp_iter, exp_total) = if current_lod == 0 {
                    (process_config.export_name.clone(), iter, training_steps)
                } else {
                    let lod_name = process_config
                        .export_name
                        .replace(".ply", &format!("_lod{current_lod}.ply"));
                    (lod_name, lod_refine_steps, lod_refine_steps)
                };
                if train_stream_config.host_runtime.phase0_telemetry {
                    emitter
                        .emit(ProcessMessage::TrainMessage(
                            TrainMessage::CheckpointExportStarted { iter: exp_iter },
                        ))
                        .await;
                }
                if is_last_step && let Some(schedule) = &progressive_sh_schedule {
                    anyhow::ensure!(
                        trainer.active_sh_degree() == Some(schedule.maximum_degree()),
                        "terminal active SH degree does not match configured/export degree"
                    );
                }
                let checkpoint_metadata = progressive_sh_schedule
                    .as_ref()
                    .map(|schedule| schedule.checkpoint_metadata(exp_iter));
                let res = export_checkpoint(
                    splats.clone(),
                    &export_path,
                    &name,
                    exp_iter,
                    exp_total,
                    checkpoint_metadata,
                )
                .await
                .with_context(|| format!("Export at iteration {iter} failed"));

                match res {
                    Ok((path, report)) => {
                        if is_last_step {
                            if report.pruned_non_finite_count > 0 {
                                log::warn!(
                                    "Terminal export compacted {} non-finite Gaussians (transforms={}, sh={}, opacity={})",
                                    report.pruned_non_finite_count,
                                    report.transforms_non_finite_row_count,
                                    report.sh_coeffs_non_finite_row_count,
                                    report.opacity_non_finite_row_count,
                                );
                            }
                            emitter
                                .emit(ProcessMessage::TrainMessage(
                                    TrainMessage::TerminalCompaction {
                                        iter: exp_iter,
                                        report,
                                    },
                                ))
                                .await;
                        }
                        emitter
                            .emit(ProcessMessage::TrainMessage(
                                TrainMessage::CheckpointExported {
                                    iter: exp_iter,
                                    path,
                                },
                            ))
                            .await;
                    }
                    Err(error) => {
                        emitter.emit(ProcessMessage::Warning { error }).await;
                    }
                }
            }
        }

        // --- Rerun logging ---
        {
            let rerun_config = &train_stream_config.rerun_config;
            visualize
                .log_splat_stats(iter, refine.total_splats)
                .unwrap();

            if let Some(every) = rerun_config.rerun_log_splats_every
                && (iter.is_multiple_of(every) || is_last_step)
            {
                visualize.log_splats(iter, splats.clone()).await.unwrap();
            }

            if iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step {
                visualize
                    .log_train_stats(iter, &stats, step_dur)
                    .await
                    .unwrap();
            }

            // The memory query goes through the compute server and stalls
            // behind all queued GPU work — keep it off the hot path unless
            // rerun is actually recording, and then only on the stats cadence.
            if rerun_config.rerun_enabled
                && (iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step)
            {
                visualize.log_memory(
                    iter,
                    &WgpuRuntime::<AutoCompiler>::client(wgpu_device).memory_usage()?,
                )?;
            }

            if refine.num_added > 0 {
                visualize
                    .log_refine_stats(iter, &refine, refine_dur)
                    .unwrap();
            }

            // Distribution stats need a GPU read-back, so sample them on a
            // coarser cadence than the per-refine stats.
            if iter.is_multiple_of(rerun_config.rerun_log_distribution_every) || is_last_step {
                visualize
                    .log_splat_distribution_stats(iter, splats.clone())
                    .await
                    .unwrap();
            }
        }

        if refine.num_added > 0 || (did_refine && train_stream_config.host_runtime.phase0_telemetry)
        {
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                    cur_splat_count: refine.total_splats,
                    iter,
                    num_added: refine.num_added,
                    num_split_oversized: refine.num_split_oversized,
                    num_split_high_grad: refine.num_split_high_grad,
                    num_pruned: refine.num_pruned,
                    num_pruned_non_finite: refine.num_pruned_non_finite,
                    duration: refine_dur,
                }))
                .await;
        }

        const UPDATE_EVERY: u32 = 5;
        let phase0_sample = iter == 1
            || matches!(iter, 50 | 51 | 100 | 101)
            || iter.is_multiple_of(10)
            || is_last_step;
        if iter % UPDATE_EVERY == 0
            || is_last_step
            || (train_stream_config.host_runtime.phase0_telemetry && phase0_sample)
        {
            emitter
                .emit(ProcessMessage::SplatsUpdated {
                    up_axis: None,
                    frame: 0,
                    total_frames: 1,
                    num_splats: refine.total_splats,
                    sh_degree,
                })
                .await;

            let lod_progress = if current_lod > 0 {
                Some((current_lod, lod_levels))
            } else {
                None
            };

            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter,
                    total_elapsed: train_duration,
                    step_duration: step_dur,
                    data_wait_duration,
                    forward_duration: stats.forward_duration,
                    loss_duration: stats.loss_duration,
                    backward_duration: stats.backward_duration,
                    optimizer_duration: stats.optimizer_duration,
                    optimizer_transforms_duration: stats.optimizer_transforms_duration,
                    optimizer_sh_coeffs_duration: stats.optimizer_sh_coeffs_duration,
                    optimizer_opacity_duration: stats.optimizer_opacity_duration,
                    render_before_count_readback_duration: stats
                        .render_before_count_readback_duration,
                    render_count_readback_duration: stats.render_count_readback_duration,
                    render_after_count_readback_duration: stats
                        .render_after_count_readback_duration,
                    live_splat_count: refine.total_splats,
                    lod_progress,
                }))
                .await;
        }

        brush_async::yield_now().await;
    }

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::DoneTraining))
        .await;

    Ok(())
}

fn validate_strong_initializer(
    data: &SplatData,
    path: Option<PathBuf>,
    max_splats: usize,
    configured_sh_degree: u32,
) -> anyhow::Result<InitializerReport> {
    let count = data.num_splats();
    anyhow::ensure!(
        count > 0,
        "required strong initializer contains no primitives"
    );
    anyhow::ensure!(
        count <= max_splats,
        "required strong initializer contains {count} primitives, exceeding max_splats ({max_splats})"
    );
    anyhow::ensure!(
        data.means.len() == count * 3,
        "required strong initializer means have an invalid element count"
    );

    let rotations = data
        .rotations
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("required strong initializer is missing rotations"))?;
    let log_scales = data.log_scales.as_ref().ok_or_else(|| {
        anyhow::anyhow!("required strong initializer is missing anisotropic log scales")
    })?;
    let opacities = data
        .raw_opacities
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("required strong initializer is missing opacity"))?;
    let sh = data.sh_coeffs.as_ref().ok_or_else(|| {
        anyhow::anyhow!("required strong initializer is missing DC/RGB coefficients")
    })?;

    anyhow::ensure!(
        rotations.len() == count * 4,
        "required strong initializer rotations have an invalid element count"
    );
    anyhow::ensure!(
        log_scales.len() == count * 3,
        "required strong initializer log scales have an invalid element count"
    );
    anyhow::ensure!(
        opacities.len() == count,
        "required strong initializer opacity has an invalid element count"
    );
    anyhow::ensure!(
        sh.len().is_multiple_of(count * 3),
        "required strong initializer SH coefficients have an invalid element count"
    );

    let coefficients_per_channel = sh.len() / (count * 3);
    let supplied_sh_degree = (0..=4)
        .find(|degree| sh_coeffs_for_degree(*degree) as usize == coefficients_per_channel)
        .ok_or_else(|| anyhow::anyhow!(
            "required strong initializer has incompatible SH coefficient count ({coefficients_per_channel} per channel)"
        ))?;
    anyhow::ensure!(
        supplied_sh_degree <= configured_sh_degree,
        "required strong initializer SH degree {supplied_sh_degree} exceeds configured degree {configured_sh_degree}; truncation is not permitted"
    );

    let non_finite = data
        .means
        .iter()
        .chain(rotations)
        .chain(log_scales)
        .chain(opacities)
        .chain(sh)
        .filter(|value| !value.is_finite())
        .count();
    anyhow::ensure!(
        non_finite == 0,
        "required strong initializer contains {non_finite} non-finite values"
    );

    Ok(InitializerReport {
        route: InitializerRoute::ExplicitStrong,
        path,
        primitive_count: count as u32,
        fields_consumed: INITIALIZER_FIELD_MEANS
            | INITIALIZER_FIELD_ROTATIONS
            | INITIALIZER_FIELD_LOG_SCALES
            | INITIALIZER_FIELD_OPACITY
            | INITIALIZER_FIELD_SH,
        rejected_count: 0,
        supplied_sh_degree: Some(supplied_sh_degree),
        configured_sh_degree,
    })
}

#[cfg(test)]
mod strong_initializer_tests {
    use super::*;

    fn strong_data(coefficients_per_channel: usize) -> SplatData {
        SplatData {
            means: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            rotations: Some(vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]),
            log_scales: Some(vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
            sh_coeffs: Some(vec![0.25; 2 * coefficients_per_channel * 3]),
            raw_opacities: Some(vec![-0.5, 0.5]),
        }
    }

    #[test]
    fn degree_zero_strong_initializer_is_valid_and_preserved_for_degree_three() {
        let data = strong_data(1);
        let before_rotations = data.rotations.clone();
        let before_scales = data.log_scales.clone();
        let before_sh = data.sh_coeffs.clone();
        let before_opacity = data.raw_opacities.clone();

        let report =
            validate_strong_initializer(&data, Some(PathBuf::from("strong-init.ply")), 2, 3)
                .unwrap();

        assert_eq!(report.primitive_count, 2);
        assert_eq!(report.supplied_sh_degree, Some(0));
        assert_eq!(report.configured_sh_degree, 3);
        assert_eq!(report.rejected_count, 0);
        assert_eq!(data.rotations, before_rotations);
        assert_eq!(data.log_scales, before_scales);
        assert_eq!(data.sh_coeffs, before_sh);
        assert_eq!(data.raw_opacities, before_opacity);
    }

    #[test]
    fn strong_initializer_rejects_truncation_non_finite_and_budget_overflow() {
        let degree_one = strong_data(4);
        let error = validate_strong_initializer(&degree_one, None, 2, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("truncation is not permitted"));

        let mut non_finite = strong_data(1);
        non_finite.log_scales.as_mut().unwrap()[2] = f32::NAN;
        let error = validate_strong_initializer(&non_finite, None, 2, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-finite"));

        let error = validate_strong_initializer(&strong_data(1), None, 1, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeding max_splats"));
    }

    #[test]
    fn strong_initializer_requires_every_full_gaussian_field() {
        let mut data = strong_data(1);
        data.rotations = None;
        assert!(
            validate_strong_initializer(&data, None, 2, 3)
                .unwrap_err()
                .to_string()
                .contains("missing rotations")
        );

        let mut data = strong_data(1);
        data.log_scales = None;
        assert!(
            validate_strong_initializer(&data, None, 2, 3)
                .unwrap_err()
                .to_string()
                .contains("anisotropic log scales")
        );

        let mut data = strong_data(1);
        data.raw_opacities = None;
        assert!(
            validate_strong_initializer(&data, None, 2, 3)
                .unwrap_err()
                .to_string()
                .contains("missing opacity")
        );

        let mut data = strong_data(1);
        data.sh_coeffs = None;
        assert!(
            validate_strong_initializer(&data, None, 2, 3)
                .unwrap_err()
                .to_string()
                .contains("missing DC/RGB")
        );
    }
}

async fn run_eval(
    device: &burn::tensor::Device,
    emitter: &Emitter,
    visualize: &VisualizeTools,
    splats: Splats,
    iter: u32,
    eval_scene: &Scene,
    save_path: Option<PathBuf>,
    rerun_max_img_size: u32,
) -> Result<(), anyhow::Error> {
    if eval_scene.views.is_empty() {
        return Ok(());
    }

    let mut psnr = 0.0;
    let mut ssim = 0.0;
    let mut count = 0;
    log::info!("Running evaluation for iteration {iter}");

    for (i, view) in eval_scene.views.iter().enumerate() {
        brush_async::yield_now().await;

        let eval_img = view.image.load().await?;
        let sample = eval_stats(
            splats.clone(),
            &view.camera,
            eval_img,
            view.image.alpha_mode(),
            device,
        )
        .await
        .context("Failed to run eval for sample.")?;

        count += 1;
        psnr += sample.psnr.clone().into_scalar_async::<f32>().await?;
        ssim += sample.ssim.clone().into_scalar_async::<f32>().await?;

        #[cfg(not(target_family = "wasm"))]
        if let Some(path) = &save_path {
            let img_name = view.image.img_name();
            let path = path
                .join(format!("eval_{iter}"))
                .join(format!("{img_name}.png"));
            sample.save_to_disk(&path).await?;
        }

        #[cfg(target_family = "wasm")]
        let _ = save_path;

        visualize
            .log_eval_sample(iter, i as u32, sample, rerun_max_img_size)
            .await?;
    }
    psnr /= count as f32;
    ssim /= count as f32;
    visualize.log_eval_stats(iter, psnr, ssim)?;
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::EvalResult {
            iter,
            avg_psnr: psnr,
            avg_ssim: ssim,
        }))
        .await;

    Ok(())
}

// TODO: Want to support this on WASM somehow. Maybe have user pick a file once,
// and write to it repeatedly?
#[cfg(not(target_family = "wasm"))]
async fn export_checkpoint(
    splats: Splats,
    export_path: &Path,
    export_name: &str,
    iter: u32,
    total_steps: u32,
    progressive_sh_metadata: Option<brush_serde::ProgressiveShCheckpointMetadata>,
) -> Result<(PathBuf, brush_serde::SplatExportValidationReport), anyhow::Error> {
    tokio::fs::create_dir_all(&export_path)
        .await
        .with_context(|| format!("Creating export directory {}", export_path.display()))?;
    let digits = ((total_steps as f64).log10().floor() as usize) + 1;
    let export_name = export_name.replace("{iter}", &format!("{iter:0digits$}"));
    let (splat_data, report) =
        brush_serde::splat_to_ply_with_report_and_metadata(splats, progressive_sh_metadata)
            .await
            .context("Serializing splat data")?;
    let output_path = export_path.join(&export_name);
    tokio::fs::write(&output_path, splat_data)
        .await
        .context(format!("Failed to export ply {export_path:?}"))?;
    Ok((output_path, report))
}

#[cfg(feature = "trainer-diagnostics")]
async fn emit_trainer_finiteness_diagnostic(
    splats: &Splats,
    iteration: u32,
    boundary: &'static str,
) {
    let transforms_tensor = splats.transforms.val();
    let sh_coeffs_tensor = splats.sh_coeffs.val();
    let opacities_tensor = splats.raw_opacities.val();
    let vectors: Vec<Vec<f32>> = burn::tensor::Transaction::default()
        .register(transforms_tensor)
        .register(sh_coeffs_tensor)
        .register(opacities_tensor)
        .execute_async()
        .await
        .expect("trainer diagnostic tensor readback failed")
        .into_iter()
        .map(|data| {
            data.into_vec::<f32>()
                .expect("trainer diagnostic tensor conversion failed")
        })
        .collect();
    let [transforms, sh_coeffs, opacities]: [Vec<f32>; 3] = vectors
        .try_into()
        .expect("trainer diagnostic transaction result mismatch");
    let row_count = splats.num_splats() as usize;
    let transform_stride = 10;
    let sh_stride = sh_coeffs.len() / row_count.max(1);

    fn invalid_rows_and_columns(values: &[f32], stride: usize) -> (Vec<usize>, Vec<u32>) {
        let mut rows = Vec::new();
        let mut columns = vec![0; stride];
        for (row_index, row) in values.chunks_exact(stride).enumerate() {
            let mut row_invalid = false;
            for (column_index, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    columns[column_index] += 1;
                    row_invalid = true;
                }
            }
            if row_invalid {
                rows.push(row_index);
            }
        }
        (rows, columns)
    }

    let (transform_rows, transform_columns) =
        invalid_rows_and_columns(&transforms, transform_stride);
    let (sh_rows, _sh_columns) = invalid_rows_and_columns(&sh_coeffs, sh_stride);
    let (opacity_rows, _opacity_columns) = invalid_rows_and_columns(&opacities, 1);
    let mut invalid_rows = transform_rows
        .iter()
        .chain(&sh_rows)
        .chain(&opacity_rows)
        .copied()
        .collect::<Vec<_>>();
    invalid_rows.sort_unstable();
    invalid_rows.dedup();

    let transform_names = [
        "mean_x",
        "mean_y",
        "mean_z",
        "rotation_0",
        "rotation_1",
        "rotation_2",
        "rotation_3",
        "log_scale_0",
        "log_scale_1",
        "log_scale_2",
    ];
    let transform_column_non_finite = transform_names
        .iter()
        .zip(transform_columns)
        .map(|(name, count)| serde_json::json!({"name": name, "count": count}))
        .collect::<Vec<_>>();
    let transform_ranges = transform_names
        .iter()
        .enumerate()
        .map(|(column_index, name)| {
            let finite = transforms
                .chunks_exact(transform_stride)
                .filter_map(|row| row[column_index].is_finite().then_some(row[column_index]))
                .collect::<Vec<_>>();
            let minimum = finite.iter().copied().reduce(f32::min);
            let maximum = finite.iter().copied().reduce(f32::max);
            serde_json::json!({"name": name, "minimum": minimum, "maximum": maximum})
        })
        .collect::<Vec<_>>();
    let diagnostic = serde_json::json!({
        "schemaVersion": 1,
        "iteration": iteration,
        "boundary": boundary,
        "primitiveCount": row_count,
        "nonFiniteRowCount": invalid_rows.len(),
        "nonFiniteRows": invalid_rows,
        "transformNonFiniteRows": transform_rows,
        "shNonFiniteRows": sh_rows,
        "opacityNonFiniteRows": opacity_rows,
        "transformColumnNonFiniteCounts": transform_column_non_finite,
        "transformFiniteRanges": transform_ranges,
    });
    let selected_boundary = boundary != "post_optimizer"
        || matches!(iteration, 1 | 199 | 200 | 201 | 202 | 300)
        || !diagnostic["nonFiniteRows"]
            .as_array()
            .expect("diagnostic rows must be an array")
            .is_empty();
    if selected_boundary {
        let line = format!("BRUSHKIT_TRAINER_FINITE_DIAGNOSTIC {diagnostic}");
        eprintln!("{line}");
        if let Some(path) = std::env::var_os("BRUSHKIT_TRAINER_DIAGNOSTIC_PATH") {
            use std::io::Write as _;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("opening trainer diagnostic output failed");
            writeln!(file, "{line}").expect("writing trainer diagnostic output failed");
        }
    }
}
