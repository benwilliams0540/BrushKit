use burn::tensor::Tensor;
use web_time::Duration;

#[derive(Clone)]
pub struct RefineStats {
    pub num_added: u32,
    /// Subset of `num_added` from force-splitting oversized (on-screen) splats.
    pub num_split_oversized: u32,
    /// Subset of `num_added` that came from gradient-driven sampling.
    pub num_split_high_grad: u32,
    pub num_pruned: u32,
    /// Subset of `num_pruned` whose params went non-finite (NaN/Inf).
    pub num_pruned_non_finite: u32,
    pub total_splats: u32,
}

#[derive(Clone)]
pub struct TrainStepStats {
    pub num_visible: u32,
    pub lr_mean: f64,
    pub lr_rotation: f64,
    pub lr_scale: f64,
    pub lr_coeffs: f64,
    pub lr_opac: f64,
    /// Host-observed duration around render submission/completion as already
    /// awaited by the trainer. This does not add synchronization.
    pub forward_duration: Duration,
    /// Host-observed duration for loss and SSIM graph construction.
    pub loss_duration: Duration,
    /// Host-observed duration around the existing backward await.
    pub backward_duration: Duration,
    /// Host-observed duration for optimizer command construction.
    pub optimizer_duration: Duration,
    /// Host-observed duration for the transforms optimizer call. Present only
    /// when the host explicitly enables optimizer-substage telemetry.
    pub optimizer_transforms_duration: Option<Duration>,
    /// Host-observed duration for the SH-coefficients optimizer call. Present
    /// only when the host explicitly enables optimizer-substage telemetry.
    pub optimizer_sh_coeffs_duration: Option<Duration>,
    /// Host-observed duration for the opacity optimizer call. Present only
    /// when the host explicitly enables optimizer-substage telemetry.
    pub optimizer_opacity_duration: Option<Duration>,
    // Non-autodiff inner tensor; consumers read the scalar lazily so disabled
    // logging doesn't force a GPU readback.
    pub loss: Tensor<1>,
}
