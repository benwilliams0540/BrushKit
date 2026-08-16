use std::path::PathBuf;

use brush_vfs::DataSource;
use glam::Vec3;

use crate::config::TrainStreamConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryBoundary {
    DatasetLoadStarted,
    DatasetLoadFinished,
    TrainerInitializationStarted,
    TrainerInitializationFinished,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializerRoute {
    Random,
    DatasetSparse,
    ImplicitPly,
    ExplicitPly,
    ExplicitStrong,
}

#[derive(Clone, Debug)]
pub struct InitializerReport {
    pub route: InitializerRoute,
    pub path: Option<PathBuf>,
    pub primitive_count: u32,
    pub fields_consumed: u32,
    pub rejected_count: u32,
    pub supplied_sh_degree: Option<u32>,
    pub configured_sh_degree: u32,
}

pub const INITIALIZER_FIELD_MEANS: u32 = 1 << 0;
pub const INITIALIZER_FIELD_ROTATIONS: u32 = 1 << 1;
pub const INITIALIZER_FIELD_LOG_SCALES: u32 = 1 << 2;
pub const INITIALIZER_FIELD_OPACITY: u32 = 1 << 3;
pub const INITIALIZER_FIELD_SH: u32 = 1 << 4;

pub enum TrainMessage {
    /// Training configuration - sent at the start of training.
    TrainConfig {
        config: Box<TrainStreamConfig>,
    },
    /// Loaded a dataset to train on.
    Dataset {
        dataset: brush_dataset::Dataset,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    TrainStep {
        iter: u32,
        total_elapsed: web_time::Duration,
        step_duration: web_time::Duration,
        data_wait_duration: web_time::Duration,
        forward_duration: web_time::Duration,
        loss_duration: web_time::Duration,
        backward_duration: web_time::Duration,
        optimizer_duration: web_time::Duration,
        optimizer_transforms_duration: Option<web_time::Duration>,
        optimizer_sh_coeffs_duration: Option<web_time::Duration>,
        optimizer_opacity_duration: Option<web_time::Duration>,
        render_before_count_readback_duration: Option<web_time::Duration>,
        render_count_readback_duration: Option<web_time::Duration>,
        render_after_count_readback_duration: Option<web_time::Duration>,
        live_splat_count: u32,
        /// If in LOD phase: `(current_lod_1_based, total_lod_levels)`.
        lod_progress: Option<(u32, u32)>,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    RefineStep {
        cur_splat_count: u32,
        iter: u32,
        num_added: u32,
        num_split_oversized: u32,
        num_split_high_grad: u32,
        num_pruned: u32,
        num_pruned_non_finite: u32,
        duration: web_time::Duration,
    },
    TelemetryBoundary {
        boundary: TelemetryBoundary,
    },
    Initializer {
        report: InitializerReport,
    },
    /// Eval was run successfully with these results.
    #[allow(unused)]
    EvalResult {
        iter: u32,
        avg_psnr: f32,
        avg_ssim: f32,
    },
    /// A checkpoint was exported successfully.
    #[allow(unused)]
    CheckpointExportStarted {
        iter: u32,
    },
    /// A checkpoint was exported successfully.
    #[allow(unused)]
    CheckpointExported {
        iter: u32,
        path: PathBuf,
    },
    /// Exhaustive validation/compaction of the terminal export population.
    TerminalCompaction {
        iter: u32,
        report: brush_serde::SplatExportValidationReport,
    },
    DoneTraining,
}

pub enum ProcessMessage {
    /// A new process is starting (before we know what type)
    NewProcess,
    /// Source has been loaded, contains the display name and type
    StartLoading {
        name: String,
        source: DataSource,
        training: bool,
        /// The base directory path if available.
        base_path: Option<PathBuf>,
    },
    /// Notification that splats have been updated.
    SplatsUpdated {
        up_axis: Option<Vec3>,
        frame: u32,
        total_frames: u32,
        num_splats: u32,
        sh_degree: u32,
    },
    TrainMessage(TrainMessage),
    /// Some warning occurred during the process, but the process can continue.
    Warning {
        error: anyhow::Error,
    },
    /// Splat, or dataset and initial splat, are done loading.
    #[allow(unused)]
    DoneLoading,
}
