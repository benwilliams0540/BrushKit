use clap::{Args, Parser};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::progressive_sh::ProgressiveShSchedule;

/// Host-only controls used by additive native ABIs. These are intentionally
/// excluded from CLI/args-file serialization so existing callers retain their
/// exact defaults and behavior.
#[derive(Clone, Debug, Default)]
pub struct HostRuntimeConfig {
    /// When present, every host-controlled stochastic path uses this seed.
    /// `None` preserves legacy random behavior.
    pub deterministic_seed: Option<u64>,
    /// Dataset-root-relative initializer selected explicitly by the host.
    pub explicit_initializer_path: Option<PathBuf>,
    /// Reject fallback/defaulted fields and require a complete Gaussian input.
    pub require_strong_initializer: bool,
    /// Emit the low-overhead Phase 0 event stream.
    pub phase0_telemetry: bool,
    /// Percentage of `load_config.max_resolution` used before the explicit
    /// progressive-resolution switch. `100` preserves the fixed-resolution path.
    pub progressive_resolution_start_percent: u32,
    /// Zero-based training iteration that switches from the reduced image scale
    /// to the configured full resolution. `0` disables the transition.
    pub progressive_resolution_switch_iteration: u32,
    /// Present only for the additive V4 callable ABI. `None` leaves the
    /// released fixed-degree path byte-for-behavior unchanged.
    pub progressive_sh_schedule: Option<ProgressiveShSchedule>,
}

#[derive(Clone, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ProcessConfig {
    /// Random seed.
    #[arg(long, help_heading = "Process options", default_value = "42")]
    pub seed: u64,
    /// Iteration to resume from
    #[arg(long, help_heading = "Process options", default_value = "0")]
    pub start_iter: u32,
    /// Eval every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "1000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub eval_every: u32,
    /// Save the rendered eval images to disk. Uses export-path for the file location.
    #[arg(long, help_heading = "Process options", default_value = "false")]
    pub eval_save_to_disk: bool,
    /// Export every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "5000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub export_every: u32,
    /// Location to put exported files. Supports {dataset} interpolation for the dataset
    /// folder name. Path is relative to the dataset's parent directory (or CWD if unavailable).
    /// Use "./{dataset}/" to export inside the dataset folder.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "./{dataset}_exports/"
    )]
    pub export_path: String,
    /// Filename of exported ply file
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "export_{iter}.ply"
    )]
    pub export_name: String,
}

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainStreamConfig {
    #[clap(flatten)]
    #[serde(flatten)]
    pub train_config: brush_train::config::TrainConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub model_config: brush_dataset::config::ModelConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub load_config: brush_dataset::config::LoadDatasetConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub process_config: ProcessConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub rerun_config: brush_rerun::RerunConfig,
    #[arg(skip)]
    #[serde(skip)]
    pub host_runtime: HostRuntimeConfig,
}

impl Default for TrainStreamConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
