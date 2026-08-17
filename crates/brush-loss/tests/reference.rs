//! Smoke + invariant tests for the loss kernels.
//!
//! GT lives as `[H, W]` u32 packing `[r g b a]` u8. We feed deterministic u8
//! data through `image_loss` and check structural properties (`SSIM(x, x) ≈ 1`,
//! output range, backward produces finite gradients). Bit-exact reference
//! matching is covered by the integration training tests in `brush-bench-test`.

use brush_loss::{ImageLossConfig, image_loss};
use burn::tensor::{Device, Int, Tensor, TensorData};
use glam::Vec3;
use wasm_bindgen_test::wasm_bindgen_test;

// Manual A/B comparisons use these predeclared bounds if byte identity fails.
// The current and candidate kernels perform the same f32 operations, so any
// larger deviation is a correctness failure rather than an accepted speedup.
const GRADIENT_AB_ABS_TOLERANCE: f32 = 2.0e-6;
const GRADIENT_AB_REL_TOLERANCE: f32 = 2.0e-5;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn pack_rgba(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|p| {
            u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16 | u32::from(p[3]) << 24
        })
        .collect()
}

/// Deterministic u8 pattern (avoids RNG so the test is reproducible across
/// machines). Returns `H*W*4` RGBA bytes.
fn make_pattern(h: usize, w: usize, scale: u32, offset: u32) -> Vec<u8> {
    (0..h * w * 4)
        .map(|i| ((i as u32 * scale + offset) % 251) as u8)
        .collect()
}

fn pred_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<3> {
    let rgb: Vec<f32> = bytes
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]].map(|b| b as f32 / 255.0))
        .collect();
    Tensor::<1>::from_floats(rgb.as_slice(), device).reshape([h, w, 3])
}

fn gt_packed_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<2, Int> {
    // Bit-reinterpret the u32 packing as i32 so the dispatch int_from_data
    // path doesn't reject magnitudes > i32::MAX.
    let packed: Vec<i32> = pack_rgba(bytes).into_iter().map(|x| x as i32).collect();
    Tensor::from_data(TensorData::new(packed, [h, w]), device)
}

fn ssim_only_cfg() -> ImageLossConfig {
    ImageLossConfig {
        l1_weight: 0.0,
        ssim_weight: 1.0,
        composite_bg: None,
        mask: false,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_identical_inputs_is_one() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes = make_pattern(h, w, 11, 13);
    let pred = pred_from_bytes(&bytes, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = image_loss(pred, gt, ssim_only_cfg());
    let mean: f32 = map
        .into_data_async()
        .await
        .expect("readback")
        .iter::<f32>()
        .sum::<f32>()
        / (h * w * 3) as f32;
    // Identical inputs SSIM saturates at 1; allow a sub-ULP roundoff.
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "SSIM(x, x) should be 1, got {mean}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_in_clamp_range() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes_a = make_pattern(h, w, 7, 19);
    let bytes_b = make_pattern(h, w, 13, 7);
    let pred = pred_from_bytes(&bytes_a, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let data: Vec<f32> = image_loss(pred, gt, ssim_only_cfg())
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("vec");
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (-1.0..=1.0).contains(&min) && (-1.0..=1.0).contains(&max),
        "SSIM out of [-1, 1]: min={min} max={max}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn image_loss_backward_runs() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (32, 48);
    let bytes_a = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let pred = pred_from_bytes(&bytes_a, h, w, &device).require_grad();
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let map = image_loss(
        pred.clone(),
        gt,
        ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: None,
            mask: false,
        },
    );
    let grads = map.mean().backward();
    let grad = pred.grad(&grads).expect("pred should have a gradient");
    let data: Vec<f32> = grad
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("vec");
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs > 0.0,
        "backward should produce non-zero gradients, got all zeros"
    );
    assert!(
        data.iter().all(|v| v.is_finite()),
        "gradients should be finite"
    );
}

async fn gradient_data(
    h: usize,
    w: usize,
    pred_bytes: &[u8],
    gt_bytes: &[u8],
    cfg: ImageLossConfig,
    rgba_pred: bool,
) -> Vec<f32> {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let pred = if rgba_pred {
        let rgba: Vec<f32> = pred_bytes.iter().map(|b| *b as f32 / 255.0).collect();
        Tensor::<1>::from_floats(rgba.as_slice(), &device).reshape([h, w, 4])
    } else {
        pred_from_bytes(pred_bytes, h, w, &device)
    }
    .require_grad();
    let gt = gt_packed_from_bytes(gt_bytes, h, w, &device);
    let map = image_loss(pred.clone(), gt, cfg);
    let grads = map.mean().backward();
    pred.grad(&grads)
        .expect("pred should have a gradient")
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("vec")
}

fn assert_finite_nonzero(data: &[f32], label: &str) {
    assert!(
        data.iter().all(|v| v.is_finite()),
        "{label} gradients should be finite"
    );
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs > GRADIENT_AB_ABS_TOLERANCE,
        "{label} gradients should be non-zero, max_abs={max_abs}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn rgb_backward_handles_halo_and_non_multiple_tile_edges() {
    let (h, w) = (37, 53);
    let pred_bytes = make_pattern(h, w, 5, 1);
    let gt_bytes = make_pattern(h, w, 7, 11);
    let data = gradient_data(
        h,
        w,
        &pred_bytes,
        &gt_bytes,
        ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: None,
            mask: false,
        },
        false,
    )
    .await;
    assert_eq!(data.len(), h * w * 3);
    assert_finite_nonzero(&data, "awkward RGB");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn rgba_mask_composite_backward_regression() {
    let (h, w) = (19, 23);
    let pred_bytes = make_pattern(h, w, 13, 17);
    let gt_bytes = make_pattern(h, w, 29, 3);
    let data = gradient_data(
        h,
        w,
        &pred_bytes,
        &gt_bytes,
        ImageLossConfig {
            l1_weight: 0.65,
            ssim_weight: -0.35,
            composite_bg: Some(Vec3::new(0.13, 0.47, 0.79)),
            mask: true,
        },
        true,
    )
    .await;
    assert_eq!(data.len(), h * w * 4);
    assert_finite_nonzero(&data, "RGBA mask/composite");
    assert!(
        data.chunks_exact(4).any(|pixel| pixel[3].abs() > 0.0),
        "alpha-match gradients should be non-zero"
    );
}

/// Writes the complete awkward-RGB gradient as little-endian f32 bits for a
/// manual feature-off/feature-on comparison. This is intentionally ignored in
/// normal tests; set `BRUSH_LOSS_GRADIENT_DUMP` to an evidence path.
#[ignore = "manual A/B gradient evidence helper"]
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn dump_rgb_gradient_for_tile_ab() {
    let output = std::env::var("BRUSH_LOSS_GRADIENT_DUMP")
        .expect("set BRUSH_LOSS_GRADIENT_DUMP to an output path");
    let (h, w) = (37, 53);
    let pred_bytes = make_pattern(h, w, 5, 1);
    let gt_bytes = make_pattern(h, w, 7, 11);
    let data = gradient_data(
        h,
        w,
        &pred_bytes,
        &gt_bytes,
        ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: None,
            mask: false,
        },
        false,
    )
    .await;
    assert_finite_nonzero(&data, "A/B RGB");

    let mut bytes = Vec::with_capacity(data.len() * size_of::<f32>());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&output, bytes).expect("write gradient evidence");
    eprintln!(
        "wrote {output}; abs_tol={GRADIENT_AB_ABS_TOLERANCE}; rel_tol={GRADIENT_AB_REL_TOLERANCE}"
    );
}

/// Profiler-off host-side wall proxy for one RGB image-loss forward/backward
/// at the frozen workload's 1080-pixel maximum dimension. This includes Burn
/// dispatch and readback, so it is an A/B screen rather than device evidence.
#[ignore = "manual profiler-off image-loss A/B microbenchmark"]
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn benchmark_rgb_image_loss_wall() {
    use std::time::Instant;

    const WARMUPS: usize = 4;
    const SAMPLES: usize = 12;
    const H: usize = 810;
    const W: usize = 1080;
    #[cfg(feature = "image-loss-bwd-tile-16")]
    const TILE: &str = "16x16";
    #[cfg(not(feature = "image-loss-bwd-tile-16"))]
    const TILE: &str = "8x8";

    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let pred_bytes = make_pattern(H, W, 5, 1);
    let gt_bytes = make_pattern(H, W, 7, 11);
    let base_pred = pred_from_bytes(&pred_bytes, H, W, &device);
    let gt = gt_packed_from_bytes(&gt_bytes, H, W, &device);
    let cfg = ImageLossConfig {
        l1_weight: 0.8,
        ssim_weight: -0.2,
        composite_bg: None,
        mask: false,
    };

    let mut samples = Vec::with_capacity(SAMPLES);
    for sample in 0..(WARMUPS + SAMPLES) {
        let pred = base_pred.clone().require_grad();
        let start = Instant::now();
        let map = image_loss(pred.clone(), gt.clone(), cfg);
        let grads = map.mean().backward();
        let grad = pred.grad(&grads).expect("pred should have a gradient");
        let data: Vec<f32> = grad
            .into_data_async()
            .await
            .expect("readback")
            .to_vec()
            .expect("vec");
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs > 0.0 && max_abs.is_finite());
        std::hint::black_box(max_abs);

        if sample >= WARMUPS {
            samples.push(elapsed_us);
            eprintln!(
                "image_loss_wall tile={TILE} sample={} elapsed_us={elapsed_us:.3}",
                sample - WARMUPS
            );
        }
    }
    samples.sort_by(f64::total_cmp);
    let median = (samples[SAMPLES / 2 - 1] + samples[SAMPLES / 2]) * 0.5;
    eprintln!(
        "image_loss_wall_summary tile={TILE} h={H} w={W} samples={SAMPLES} median_us={median:.3}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn alpha_match_via_4ch_pred() {
    // Feeding 4-channel `pred` makes the kernel emit `|pred.a - gt.a|`
    // into the alpha channel of the loss map.
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (16, 24);
    let bytes = make_pattern(h, w, 17, 5);
    let rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
        .reshape([h, w, 4])
        .require_grad();
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = image_loss(
        pred,
        gt,
        ImageLossConfig {
            l1_weight: 1.0,
            ssim_weight: 0.0,
            composite_bg: None,
            mask: false,
        },
    );
    let _grads = map.mean().backward();
}
