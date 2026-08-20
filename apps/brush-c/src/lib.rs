// brush-c is a native-only FFI shim. The crate compiles to an empty stub on wasm.
#![cfg(not(target_family = "wasm"))]

use brush_process::burn_init_setup;
use brush_process::config::{HostRuntimeConfig, TrainStreamConfig};
use brush_process::message::{InitializerRoute, TelemetryBoundary, TrainMessage};
use brush_process::progressive_sh::{
    PROGRESSIVE_SH_MODE_DISABLED, PROGRESSIVE_SH_MODE_INTERVAL, ProgressiveShSchedule,
    ShTransitionReason,
};
use brush_process::{DataSource, create_process, message::ProcessMessage};
use brush_render::gaussian_splats::SplatRenderMode;
#[cfg(brushkit_cubecl_gpu_profile)]
use burn_cubecl::cubecl::config::{
    CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel,
};
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainExitCode {
    Success = 0,
    Error = 1,
    Cancelled = 2,
}

pub const BRUSH_ABI_VERSION_V2: u32 = 2;
pub const BRUSH_ABI_VERSION_V3: u32 = 3;
pub const BRUSH_ABI_VERSION_V4: u32 = 4;
pub const BRUSH_CAPABILITY_DATASET_BOUNDARIES_V2: u64 = 1 << 0;
pub const BRUSH_CAPABILITY_INITIALIZER_AUDIT_V2: u64 = 1 << 1;
pub const BRUSH_CAPABILITY_STEP_TIMINGS_V2: u64 = 1 << 2;
pub const BRUSH_CAPABILITY_REFINEMENT_COUNTS_V2: u64 = 1 << 3;
pub const BRUSH_CAPABILITY_OPERATION_TIMINGS_V2: u64 = 1 << 4;
pub const BRUSH_CAPABILITY_ADAPTER_IDENTITY_V2: u64 = 1 << 5;
pub const BRUSH_CAPABILITY_GPU_MEMORY_V2: u64 = 1 << 6;
pub const BRUSH_CAPABILITY_GPU_COMMAND_TIMING_V2: u64 = 1 << 7;
pub const BRUSH_CAPABILITY_CPU_WAIT_TIMING_V2: u64 = 1 << 8;
pub const BRUSH_CAPABILITY_TERMINAL_COMPACTION_V2: u64 = 1 << 9;
pub const BRUSH_CAPABILITY_PROGRESSIVE_SH_V4: u64 = 1 << 10;
pub const BRUSH_RENDER_MODE_DEFAULT_V2: u32 = 0;
pub const BRUSH_RENDER_MODE_MIP_V2: u32 = 1;
pub const BRUSH_SH_POLICY_PRESERVE_AND_ZERO_PAD_V2: u32 = 0;
pub const BRUSH_INSTRUMENTATION_DISABLED_V2: u32 = 0;
pub const BRUSH_INSTRUMENTATION_PHASE0_V2: u32 = 1;
pub const BRUSH_INITIALIZER_FIELD_MEANS_V2: u32 = 1 << 0;
pub const BRUSH_INITIALIZER_FIELD_ROTATIONS_V2: u32 = 1 << 1;
pub const BRUSH_INITIALIZER_FIELD_LOG_SCALES_V2: u32 = 1 << 2;
pub const BRUSH_INITIALIZER_FIELD_OPACITY_V2: u32 = 1 << 3;
pub const BRUSH_INITIALIZER_FIELD_SH_V2: u32 = 1 << 4;

const BRUSH_AVAILABLE_CAPABILITIES_V2: u64 = BRUSH_CAPABILITY_DATASET_BOUNDARIES_V2
    | BRUSH_CAPABILITY_INITIALIZER_AUDIT_V2
    | BRUSH_CAPABILITY_STEP_TIMINGS_V2
    | BRUSH_CAPABILITY_REFINEMENT_COUNTS_V2
    | BRUSH_CAPABILITY_OPERATION_TIMINGS_V2
    | BRUSH_CAPABILITY_CPU_WAIT_TIMING_V2
    | BRUSH_CAPABILITY_TERMINAL_COMPACTION_V2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushEventKindV2 {
    Capabilities = 0,
    Configuration = 1,
    DatasetLoadStarted = 2,
    DatasetLoadFinished = 3,
    Initializer = 4,
    TrainerInitializationStarted = 5,
    TrainerInitializationFinished = 6,
    Step = 7,
    Refinement = 8,
    CheckpointExportStarted = 9,
    CheckpointExported = 10,
    Terminal = 11,
    TerminalCompaction = 12,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushErrorCodeV2 {
    None = 0,
    InvalidArgument = 1,
    UnsupportedAbi = 2,
    Dataset = 3,
    Initializer = 4,
    Training = 5,
    Cancelled = 6,
    Panic = 7,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushInitializerRouteV2 {
    Random = 0,
    DatasetSparse = 1,
    ImplicitPly = 2,
    ExplicitPly = 3,
    ExplicitStrong = 4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptionsV2 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub seed: u64,
    pub render_mode: u32,
    pub sh_degree: u32,
    pub sh_policy: u32,
    pub total_train_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub max_splats: u32,
    pub export_every: u32,
    pub output_path: *const c_char,
    pub export_name: *const c_char,
    pub initializer_path: *const c_char,
    pub initializer_required: u8,
    pub instrumentation_level: u32,
}

/// Additive callable ABI for one explicit progressive-resolution transition.
/// V2 remains available and retains its fixed-resolution behavior.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptionsV3 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub seed: u64,
    pub render_mode: u32,
    pub sh_degree: u32,
    pub sh_policy: u32,
    pub total_train_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub max_splats: u32,
    pub export_every: u32,
    pub output_path: *const c_char,
    pub export_name: *const c_char,
    pub initializer_path: *const c_char,
    pub initializer_required: u8,
    pub instrumentation_level: u32,
    pub progressive_resolution_start_percent: u32,
    pub progressive_resolution_switch_iteration: u32,
}

/// Additive callable ABI for the one predeclared progressive-SH schedule.
/// V2 and V3 remain available and retain their fixed-SH behavior.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptionsV4 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub seed: u64,
    pub render_mode: u32,
    pub sh_degree: u32,
    pub sh_policy: u32,
    pub total_train_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub max_splats: u32,
    pub export_every: u32,
    pub output_path: *const c_char,
    pub export_name: *const c_char,
    pub initializer_path: *const c_char,
    pub initializer_required: u8,
    pub instrumentation_level: u32,
    pub progressive_resolution_start_percent: u32,
    pub progressive_resolution_switch_iteration: u32,
    pub sh_schedule_mode: u32,
    pub initial_sh_degree: u32,
    pub sh_degree_step_interval: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BrushEventV2 {
    pub struct_size: u32,
    pub kind: BrushEventKindV2,
    pub timestamp_ns: u64,
    pub iteration: u32,
    pub error_code: BrushErrorCodeV2,
    pub capability_flags: u64,
    pub seed: u64,
    pub render_mode: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub max_splats: u32,
    pub export_every: u32,
    pub instrumentation_level: u32,
    pub initializer_route: BrushInitializerRouteV2,
    pub primitive_count: u32,
    pub initial_primitive_count: u32,
    pub final_primitive_count: u32,
    pub fields_consumed: u32,
    pub rejected_count: u32,
    pub supplied_sh_degree: i32,
    pub configured_sh_degree: u32,
    pub added_count: u32,
    pub split_count: u32,
    pub split_oversized_count: u32,
    pub split_high_gradient_count: u32,
    pub clone_count: u32,
    pub pruned_count: u32,
    pub pruned_non_finite_count: u32,
    pub net_growth: i64,
    pub duration_ns: u64,
    pub data_wait_ns: u64,
    pub forward_ns: u64,
    pub loss_and_ssim_ns: u64,
    pub backward_ns: u64,
    pub optimizer_ns: u64,
    pub densification_and_compaction_ns: u64,
    /// Callback-scoped initializer path, checkpoint path, or failure detail.
    pub text: *const c_char,
}

impl BrushEventV2 {
    fn new(kind: BrushEventKindV2, timestamp_ns: u64) -> Self {
        Self {
            struct_size: mem::size_of::<Self>() as u32,
            kind,
            timestamp_ns,
            iteration: 0,
            error_code: BrushErrorCodeV2::None,
            capability_flags: 0,
            seed: 0,
            render_mode: 0,
            refine_every: 0,
            max_resolution: 0,
            max_splats: 0,
            export_every: 0,
            instrumentation_level: 0,
            initializer_route: BrushInitializerRouteV2::Random,
            primitive_count: 0,
            initial_primitive_count: 0,
            final_primitive_count: 0,
            fields_consumed: 0,
            rejected_count: 0,
            supplied_sh_degree: -1,
            configured_sh_degree: 0,
            added_count: 0,
            split_count: 0,
            split_oversized_count: 0,
            split_high_gradient_count: 0,
            clone_count: 0,
            pruned_count: 0,
            pruned_non_finite_count: 0,
            net_growth: 0,
            duration_ns: 0,
            data_wait_ns: 0,
            forward_ns: 0,
            loss_and_ssim_ns: 0,
            backward_ns: 0,
            optimizer_ns: 0,
            densification_and_compaction_ns: 0,
            text: ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushEventKindV4 {
    Capabilities = 0,
    Configuration = 1,
    DatasetLoadStarted = 2,
    DatasetLoadFinished = 3,
    Initializer = 4,
    TrainerInitializationStarted = 5,
    TrainerInitializationFinished = 6,
    Step = 7,
    Refinement = 8,
    CheckpointExportStarted = 9,
    CheckpointExported = 10,
    Terminal = 11,
    TerminalCompaction = 12,
    ActiveSHDegreeChanged = 13,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushSHScheduleModeV4 {
    Disabled = 0,
    Interval = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushSHTransitionReasonV4 {
    None = 0,
    Disabled = 1,
    Initial = 2,
    Scheduled = 3,
    Resume = 4,
}

/// V4 wraps the complete V2 evidence payload and adds typed progressive-SH
/// identity. `zero_based_iteration` is intentionally separate from the V2
/// payload's historically human-step-oriented `iteration` field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BrushEventV4 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub kind: BrushEventKindV4,
    pub base: BrushEventV2,
    pub maximum_sh_degree: u32,
    pub active_sh_degree: u32,
    pub initial_sh_degree: u32,
    pub sh_degree_step_interval: u32,
    pub sh_schedule_mode: BrushSHScheduleModeV4,
    pub zero_based_iteration: u32,
    pub transition_reason: BrushSHTransitionReasonV4,
    pub schedule_identity: u64,
}

impl BrushEventV4 {
    fn from_v2(kind: BrushEventKindV4, base: BrushEventV2, state: &EventJobState) -> Self {
        Self {
            struct_size: mem::size_of::<Self>() as u32,
            abi_version: BRUSH_ABI_VERSION_V4,
            kind,
            base,
            maximum_sh_degree: state.maximum_sh_degree,
            active_sh_degree: state.active_sh_degree,
            initial_sh_degree: state.initial_sh_degree,
            sh_degree_step_interval: state.sh_degree_step_interval,
            sh_schedule_mode: match state.sh_schedule_mode {
                PROGRESSIVE_SH_MODE_DISABLED => BrushSHScheduleModeV4::Disabled,
                PROGRESSIVE_SH_MODE_INTERVAL => BrushSHScheduleModeV4::Interval,
                _ => unreachable!("validated schedule mode"),
            },
            zero_based_iteration: state.zero_based_iteration,
            transition_reason: BrushSHTransitionReasonV4::None,
            schedule_identity: state.schedule_identity,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BrushNativeIdentityV2 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub build_revision: *const c_char,
    pub crate_version: *const c_char,
    pub graphics_backend: *const c_char,
    pub adapter_name: *const c_char,
    pub adapter_identity_available: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressMessageKind {
    NewProcess = 0,
    Training = 1,
    CheckpointExported = 2,
    DoneTraining = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProgressMessage {
    pub kind: ProgressMessageKind,
    pub iter: u32,
    pub path: *const c_char,
}

impl ProgressMessage {
    fn new(kind: ProgressMessageKind) -> Self {
        Self {
            kind,
            iter: 0,
            path: ptr::null(),
        }
    }
}

#[repr(C)]
pub struct BrushJob {
    _private: [u8; 0],
}

#[repr(C)]
pub struct BrushJobV2 {
    _private: [u8; 0],
}

struct BrushJobState {
    cancellation_requested: Arc<AtomicBool>,
    completion: Mutex<JobCompletion>,
    completion_changed: Condvar,
}

struct JobCompletion {
    handle: Option<JoinHandle<TrainExitCode>>,
    result: Option<TrainExitCode>,
    joining: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptions {
    pub total_train_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub max_splats: u32,
    pub export_every: u32,
    pub output_path: *const c_char,
    pub export_name: *const c_char,
}

struct OwnedTrainOptions {
    total_train_steps: u32,
    refine_every: u32,
    max_resolution: u32,
    max_splats: u32,
    export_every: u32,
    output_path: Option<String>,
    export_name: Option<String>,
}

struct OwnedTrainOptionsV2 {
    seed: u64,
    render_mode: u32,
    sh_degree: u32,
    total_train_steps: u32,
    refine_every: u32,
    max_resolution: u32,
    max_splats: u32,
    export_every: u32,
    output_path: Option<String>,
    export_name: Option<String>,
    initializer_path: Option<PathBuf>,
    initializer_required: bool,
    instrumentation_level: u32,
}

struct OwnedTrainOptionsV3 {
    base: OwnedTrainOptionsV2,
    progressive_resolution_start_percent: u32,
    progressive_resolution_switch_iteration: u32,
}

struct OwnedTrainOptionsV4 {
    base: OwnedTrainOptionsV3,
    sh_schedule: ProgressiveShSchedule,
}

impl OwnedTrainOptions {
    /// # Safety
    ///
    /// `options` must either be null or point to a valid `TrainOptions` value. If
    /// `output_path` or `export_name` is not null, it must be a valid
    /// null-terminated C string.
    unsafe fn copy_from(options: *const TrainOptions) -> Option<Self> {
        if options.is_null() {
            return None;
        }

        // SAFETY: The caller guarantees `options` points to a valid TrainOptions value.
        let options = unsafe { *options };
        let output_path = if options.output_path.is_null() {
            None
        } else {
            // SAFETY: The caller guarantees `output_path` is a valid C string when non-null.
            Some(
                unsafe { CStr::from_ptr(options.output_path) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        let export_name = if options.export_name.is_null() {
            None
        } else {
            // SAFETY: The caller guarantees `export_name` is a valid C string when non-null.
            Some(
                unsafe { CStr::from_ptr(options.export_name) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };

        Some(Self {
            total_train_steps: options.total_train_steps,
            refine_every: options.refine_every,
            max_resolution: options.max_resolution,
            max_splats: options.max_splats,
            export_every: options.export_every,
            output_path,
            export_name,
        })
    }

    fn into_train_stream_config(self) -> TrainStreamConfig {
        let mut process_args = TrainStreamConfig::default();
        if let Some(output_path) = self.output_path {
            process_args.process_config.export_path = output_path;
        }
        if let Some(export_name) = self.export_name {
            process_args.process_config.export_name = export_name;
        }
        process_args.train_config.total_train_iters = self.total_train_steps;
        process_args.train_config.refine_every = self.refine_every;
        if self.max_splats > 0 {
            process_args.train_config.max_splats = self.max_splats;
        }
        process_args.load_config.max_resolution = self.max_resolution;
        process_args.process_config.export_every = self.export_every;
        process_args.process_config.eval_save_to_disk = true;
        process_args
    }
}

impl OwnedTrainOptionsV2 {
    /// Copy and validate the complete V2 input before the worker starts. The
    /// integer discriminants are deliberate: invalid C enum values can be
    /// rejected without constructing an invalid Rust enum.
    unsafe fn copy_from(options: *const TrainOptionsV2) -> Result<Self, String> {
        if options.is_null() {
            return Err("V2 options pointer is null".to_owned());
        }
        // SAFETY: caller promises readable storage; struct_size is checked
        // immediately after the copy and V2 has no accepted smaller layout.
        let options = unsafe { *options };
        if options.struct_size != mem::size_of::<TrainOptionsV2>() as u32 {
            return Err(format!(
                "V2 options struct_size {} does not match {}",
                options.struct_size,
                mem::size_of::<TrainOptionsV2>()
            ));
        }
        if options.abi_version != BRUSH_ABI_VERSION_V2 {
            return Err(format!(
                "unsupported callable ABI version {}; expected {}",
                options.abi_version, BRUSH_ABI_VERSION_V2
            ));
        }
        if options.render_mode > 1 {
            return Err(format!("invalid render_mode {}", options.render_mode));
        }
        if options.sh_degree > 4 {
            return Err(format!("invalid sh_degree {}", options.sh_degree));
        }
        if options.sh_policy != 0 {
            return Err(format!("unsupported sh_policy {}", options.sh_policy));
        }
        if options.initializer_required > 1 {
            return Err("initializer_required must be 0 or 1".to_owned());
        }
        if options.instrumentation_level > 1 {
            return Err(format!(
                "unsupported instrumentation_level {}",
                options.instrumentation_level
            ));
        }
        if options.total_train_steps == 0
            || options.refine_every == 0
            || options.max_resolution == 0
            || options.max_splats == 0
            || options.export_every == 0
        {
            return Err(
                "total_train_steps, refine_every, max_resolution, max_splats, and export_every must be non-zero"
                    .to_owned(),
            );
        }

        // SAFETY: the V2 caller contract covers every non-null string pointer.
        let output_path = unsafe { copy_optional_utf8(options.output_path, "output_path") }?;
        // SAFETY: the V2 caller contract covers every non-null string pointer.
        let export_name = unsafe { copy_optional_utf8(options.export_name, "export_name") }?;
        // SAFETY: the V2 caller contract covers every non-null string pointer.
        let initializer_path =
            unsafe { copy_optional_utf8(options.initializer_path, "initializer_path") }?
                .map(PathBuf::from);
        let initializer_required = options.initializer_required == 1;
        if initializer_required && initializer_path.is_none() {
            return Err("initializer_required is set but initializer_path is null".to_owned());
        }

        Ok(Self {
            seed: options.seed,
            render_mode: options.render_mode,
            sh_degree: options.sh_degree,
            total_train_steps: options.total_train_steps,
            refine_every: options.refine_every,
            max_resolution: options.max_resolution,
            max_splats: options.max_splats,
            export_every: options.export_every,
            output_path,
            export_name,
            initializer_path,
            initializer_required,
            instrumentation_level: options.instrumentation_level,
        })
    }

    fn into_train_stream_config(self) -> TrainStreamConfig {
        let mut config = TrainStreamConfig::default();
        config.process_config.seed = self.seed;
        if let Some(output_path) = self.output_path {
            config.process_config.export_path = output_path;
        }
        if let Some(export_name) = self.export_name {
            config.process_config.export_name = export_name;
        }
        config.process_config.export_every = self.export_every;
        config.process_config.eval_save_to_disk = true;
        config.train_config.render_mode = Some(match self.render_mode {
            0 => SplatRenderMode::Default,
            1 => SplatRenderMode::Mip,
            _ => unreachable!("validated render mode"),
        });
        config.train_config.total_train_iters = self.total_train_steps;
        config.train_config.refine_every = self.refine_every;
        config.train_config.max_splats = self.max_splats;
        config.load_config.max_resolution = self.max_resolution;
        config.model_config.sh_degree = self.sh_degree;
        config.host_runtime = HostRuntimeConfig {
            deterministic_seed: Some(self.seed),
            explicit_initializer_path: self.initializer_path,
            require_strong_initializer: self.initializer_required,
            phase0_telemetry: self.instrumentation_level == 1,
            progressive_resolution_start_percent: 100,
            progressive_resolution_switch_iteration: 0,
            progressive_sh_schedule: None,
        };
        config
    }
}

impl OwnedTrainOptionsV3 {
    /// Copy and validate the complete V3 input before the worker starts.
    unsafe fn copy_from(options: *const TrainOptionsV3) -> Result<Self, String> {
        if options.is_null() {
            return Err("V3 options pointer is null".to_owned());
        }
        // SAFETY: caller promises readable V3 storage; exact size is checked
        // immediately after the copy.
        let options = unsafe { *options };
        if options.struct_size != mem::size_of::<TrainOptionsV3>() as u32 {
            return Err(format!(
                "V3 options struct_size {} does not match {}",
                options.struct_size,
                mem::size_of::<TrainOptionsV3>()
            ));
        }
        if options.abi_version != BRUSH_ABI_VERSION_V3 {
            return Err(format!(
                "unsupported callable ABI version {}; expected {}",
                options.abi_version, BRUSH_ABI_VERSION_V3
            ));
        }
        if !(1..=100).contains(&options.progressive_resolution_start_percent) {
            return Err("progressive_resolution_start_percent must be in 1...100".to_owned());
        }
        if options.progressive_resolution_start_percent < 100 {
            if options.progressive_resolution_switch_iteration == 0
                || options.progressive_resolution_switch_iteration >= options.total_train_steps
            {
                return Err(
                    "progressive_resolution_switch_iteration must be between 1 and total_train_steps - 1"
                        .to_owned(),
                );
            }
        } else if options.progressive_resolution_switch_iteration != 0 {
            return Err(
                "progressive_resolution_switch_iteration must be 0 when start percent is 100"
                    .to_owned(),
            );
        }

        let v2 = TrainOptionsV2 {
            struct_size: mem::size_of::<TrainOptionsV2>() as u32,
            abi_version: BRUSH_ABI_VERSION_V2,
            seed: options.seed,
            render_mode: options.render_mode,
            sh_degree: options.sh_degree,
            sh_policy: options.sh_policy,
            total_train_steps: options.total_train_steps,
            refine_every: options.refine_every,
            max_resolution: options.max_resolution,
            max_splats: options.max_splats,
            export_every: options.export_every,
            output_path: options.output_path,
            export_name: options.export_name,
            initializer_path: options.initializer_path,
            initializer_required: options.initializer_required,
            instrumentation_level: options.instrumentation_level,
        };
        // SAFETY: `v2` is complete local storage and all C-string pointers are
        // covered by the V3 caller contract.
        let base = unsafe { OwnedTrainOptionsV2::copy_from(&v2) }?;
        Ok(Self {
            base,
            progressive_resolution_start_percent: options.progressive_resolution_start_percent,
            progressive_resolution_switch_iteration: options
                .progressive_resolution_switch_iteration,
        })
    }

    fn into_train_stream_config(self) -> TrainStreamConfig {
        let mut config = self.base.into_train_stream_config();
        config.host_runtime.progressive_resolution_start_percent =
            self.progressive_resolution_start_percent;
        config.host_runtime.progressive_resolution_switch_iteration =
            self.progressive_resolution_switch_iteration;
        config
    }
}

impl OwnedTrainOptionsV4 {
    /// Copy and validate the complete V4 input before the worker starts.
    unsafe fn copy_from(options: *const TrainOptionsV4) -> Result<Self, String> {
        if options.is_null() {
            return Err("V4 options pointer is null".to_owned());
        }
        // SAFETY: caller promises readable V4 storage; exact size is checked
        // immediately after the copy.
        let options = unsafe { *options };
        if options.struct_size != mem::size_of::<TrainOptionsV4>() as u32 {
            return Err(format!(
                "V4 options struct_size {} does not match {}",
                options.struct_size,
                mem::size_of::<TrainOptionsV4>()
            ));
        }
        if options.abi_version != BRUSH_ABI_VERSION_V4 {
            return Err(format!(
                "unsupported callable ABI version {}; expected {}",
                options.abi_version, BRUSH_ABI_VERSION_V4
            ));
        }

        let v3 = TrainOptionsV3 {
            struct_size: mem::size_of::<TrainOptionsV3>() as u32,
            abi_version: BRUSH_ABI_VERSION_V3,
            seed: options.seed,
            render_mode: options.render_mode,
            sh_degree: options.sh_degree,
            sh_policy: options.sh_policy,
            total_train_steps: options.total_train_steps,
            refine_every: options.refine_every,
            max_resolution: options.max_resolution,
            max_splats: options.max_splats,
            export_every: options.export_every,
            output_path: options.output_path,
            export_name: options.export_name,
            initializer_path: options.initializer_path,
            initializer_required: options.initializer_required,
            instrumentation_level: options.instrumentation_level,
            progressive_resolution_start_percent: options.progressive_resolution_start_percent,
            progressive_resolution_switch_iteration: options
                .progressive_resolution_switch_iteration,
        };
        // SAFETY: `v3` is complete local storage and all string pointers are
        // covered by the V4 caller contract.
        let base = unsafe { OwnedTrainOptionsV3::copy_from(&v3) }?;
        let sh_schedule = ProgressiveShSchedule::new(
            options.sh_schedule_mode,
            options.sh_degree,
            options.initial_sh_degree,
            options.sh_degree_step_interval,
            options.total_train_steps,
        )?;

        match options.sh_schedule_mode {
            PROGRESSIVE_SH_MODE_DISABLED => {}
            PROGRESSIVE_SH_MODE_INTERVAL => {
                if options.sh_degree != 3
                    || options.initial_sh_degree != 0
                    || options.sh_degree_step_interval != 150
                    || options.total_train_steps != 600
                {
                    return Err(
                        "V4 interval mode supports only the predeclared SH3 schedule: initial degree 0, 150-step interval, 600 total steps"
                            .to_owned(),
                    );
                }
            }
            _ => unreachable!("schedule constructor rejected unknown mode"),
        }

        Ok(Self { base, sh_schedule })
    }

    fn schedule(&self) -> &ProgressiveShSchedule {
        &self.sh_schedule
    }

    fn into_train_stream_config(self) -> TrainStreamConfig {
        let progressive_enabled = self.sh_schedule.mode() == PROGRESSIVE_SH_MODE_INTERVAL;
        let mut config = self.base.into_train_stream_config();
        if progressive_enabled {
            config.host_runtime.progressive_sh_schedule = Some(self.sh_schedule);
        }
        config
    }
}

unsafe fn copy_optional_utf8(value: *const c_char, field: &str) -> Result<Option<String>, String> {
    if value.is_null() {
        return Ok(None);
    }
    // SAFETY: the FFI caller guarantees non-null pointers reference a
    // null-terminated string for the duration of this call.
    let value = unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|_utf8_error| format!("{field} is not valid UTF-8"))?;
    Ok(Some(value.to_owned()))
}

pub type ProgressCallback =
    extern "C" fn(progress_message: ProgressMessage, user_data: *mut c_void);
pub type ProgressCallbackV2 = extern "C" fn(event: BrushEventV2, user_data: *mut c_void);
pub type ProgressCallbackV4 = extern "C" fn(event: BrushEventV4, user_data: *mut c_void);

struct EventJobState {
    started: Instant,
    initial_primitive_count: u32,
    last_primitive_count: u32,
    terminal_exported_primitive_count: Option<u32>,
    maximum_sh_degree: u32,
    active_sh_degree: u32,
    initial_sh_degree: u32,
    sh_degree_step_interval: u32,
    sh_schedule_mode: u32,
    schedule_identity: u64,
    zero_based_iteration: u32,
}

impl EventJobState {
    fn v2(started: Instant) -> Self {
        Self {
            started,
            initial_primitive_count: 0,
            last_primitive_count: 0,
            terminal_exported_primitive_count: None,
            maximum_sh_degree: 0,
            active_sh_degree: 0,
            initial_sh_degree: 0,
            sh_degree_step_interval: 0,
            sh_schedule_mode: PROGRESSIVE_SH_MODE_DISABLED,
            schedule_identity: 0,
            zero_based_iteration: 0,
        }
    }

    fn v4(started: Instant, schedule: &ProgressiveShSchedule) -> Self {
        Self {
            started,
            initial_primitive_count: 0,
            last_primitive_count: 0,
            terminal_exported_primitive_count: None,
            maximum_sh_degree: schedule.maximum_degree(),
            active_sh_degree: schedule.active_degree_at(0),
            initial_sh_degree: schedule.initial_degree(),
            sh_degree_step_interval: schedule.step_interval(),
            sh_schedule_mode: schedule.mode(),
            schedule_identity: schedule.identity(),
            zero_based_iteration: 0,
        }
    }
}

enum JobCallback {
    Legacy {
        callback: ProgressCallback,
        user_data_address: usize,
    },
    V2 {
        callback: ProgressCallbackV2,
        user_data_address: usize,
        state: EventJobState,
    },
    V4 {
        callback: ProgressCallbackV4,
        user_data_address: usize,
        state: EventJobState,
    },
}

static SETUP: OnceCell<()> = OnceCell::const_new();
static V2_TRAINING_LOCK: Mutex<()> = Mutex::new(());

const BUILD_REVISION_C: &[u8] = concat!(env!("BRUSHKIT_BUILD_REVISION"), "\0").as_bytes();
const BUILD_PROVENANCE_C: &[u8] = concat!(env!("BRUSHKIT_BUILD_PROVENANCE"), "\0").as_bytes();
const CRATE_VERSION_C: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
#[cfg(target_vendor = "apple")]
const GRAPHICS_BACKEND_C: &[u8] = b"Metal\0";
#[cfg(not(target_vendor = "apple"))]
const GRAPHICS_BACKEND_C: &[u8] = b"Auto\0";
const ADAPTER_UNAVAILABLE_C: &[u8] =
    b"unavailable: default Burn setup does not expose the bound adapter identity\0";

#[unsafe(no_mangle)]
pub extern "C" fn brush_get_abi_version() -> u32 {
    BRUSH_ABI_VERSION_V2
}

/// Returns a stable, process-lifetime, semicolon-delimited provenance record.
/// The record is generated from compile-time state and includes the engine
/// revision, crate version, ABI versions, enabled Apple feature set, profiler
/// state, Rust toolchain, and target triple.
#[unsafe(no_mangle)]
pub extern "C" fn brush_get_build_provenance_v1() -> *const c_char {
    BUILD_PROVENANCE_C.as_ptr().cast()
}

/// Copies stable, process-lifetime native identity pointers into `identity`.
/// Adapter identity is explicitly unavailable on the default Burn setup; this
/// is reported instead of inferring a device name.
///
/// # Safety
/// `identity` must point to writable storage of exactly `struct_size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_get_native_identity_v2(
    identity: *mut BrushNativeIdentityV2,
    struct_size: u32,
) -> bool {
    if identity.is_null() || struct_size != mem::size_of::<BrushNativeIdentityV2>() as u32 {
        return false;
    }
    // SAFETY: validated non-null and exact V2 storage size above.
    unsafe {
        identity.write(BrushNativeIdentityV2 {
            struct_size,
            abi_version: BRUSH_ABI_VERSION_V2,
            build_revision: BUILD_REVISION_C.as_ptr().cast(),
            crate_version: CRATE_VERSION_C.as_ptr().cast(),
            graphics_backend: GRAPHICS_BACKEND_C.as_ptr().cast(),
            adapter_name: ADAPTER_UNAVAILABLE_C.as_ptr().cast(),
            adapter_identity_available: false,
        });
    }
    true
}

/// Starts a Brush training job and returns an opaque job handle.
///
/// The returned handle must be passed to `brush_job_release` exactly once.
/// Call `brush_job_wait` to block until completion, and `brush_job_cancel` to
/// request cooperative cancellation.
///
/// # Safety
///
/// - `dataset_path` must point to a valid, null-terminated C string.
/// - `options` must point to a valid `TrainOptions` value.
/// - `options.output_path` and `options.export_name`, when non-null, must point
///   to valid C strings.
/// - `user_data` is passed back to `progress_callback` on the worker thread and
///   must remain valid until the job reaches a terminal state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_train_start(
    dataset_path: *const c_char,
    options: *const TrainOptions,
    progress_callback: ProgressCallback,
    user_data: *mut c_void,
) -> *mut BrushJob {
    if dataset_path.is_null() || options.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: Checked non-null above; caller guarantees a valid C string.
    let dataset_path = unsafe { CStr::from_ptr(dataset_path) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: Checked non-null above; caller guarantees options and output_path validity.
    let Some(train_options) = (unsafe { OwnedTrainOptions::copy_from(options) }) else {
        return ptr::null_mut();
    };

    let cancellation_requested = Arc::new(AtomicBool::new(false));
    let worker_cancellation = Arc::clone(&cancellation_requested);
    let user_data_address = user_data as usize;

    let Ok(handle) = std::thread::Builder::new()
        .name("brush-c-train".to_owned())
        .spawn(move || {
            run_training_job(
                dataset_path,
                train_options,
                JobCallback::Legacy {
                    callback: progress_callback,
                    user_data_address,
                },
                &worker_cancellation,
                false,
            )
        })
    else {
        return ptr::null_mut();
    };

    let state = Box::new(BrushJobState {
        cancellation_requested,
        completion: Mutex::new(JobCompletion {
            handle: Some(handle),
            result: None,
            joining: false,
        }),
        completion_changed: Condvar::new(),
    });
    Box::into_raw(state).cast::<BrushJob>()
}

/// Starts a callable ABI V2 training job. V2 jobs serialize their seeded
/// interaction with the shared Burn device so overlapping jobs cannot race the
/// device RNG. Use `brush_job_retain_v2` before handing a job to another owner.
///
/// # Safety
/// All non-null string pointers must remain readable for this call. `user_data`
/// must remain valid until a terminal callback or until the final retained job
/// handle has been released after waiting.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_train_start_v2(
    dataset_path: *const c_char,
    options: *const TrainOptionsV2,
    progress_callback: ProgressCallbackV2,
    user_data: *mut c_void,
) -> *mut BrushJobV2 {
    let started = Instant::now();
    let dataset_path = if dataset_path.is_null() {
        Err("dataset_path pointer is null".to_owned())
    } else {
        // SAFETY: caller guarantees a null-terminated C string.
        unsafe { CStr::from_ptr(dataset_path) }
            .to_str()
            .map(str::to_owned)
            .map_err(|_utf8_error| "dataset_path is not valid UTF-8".to_owned())
    };
    // SAFETY: forwarded V2 pointer contract.
    let options = unsafe { OwnedTrainOptionsV2::copy_from(options) };
    let (dataset_path, train_options) = match (dataset_path, options) {
        (Ok(dataset_path), Ok(options)) => (dataset_path, options),
        (dataset_path, options) => {
            let detail = dataset_path.err().or_else(|| options.err()).unwrap();
            let mut event = BrushEventV2::new(BrushEventKindV2::Terminal, 0);
            event.error_code = if detail.contains("ABI version") {
                BrushErrorCodeV2::UnsupportedAbi
            } else {
                BrushErrorCodeV2::InvalidArgument
            };
            emit_v2_with_text(progress_callback, user_data as usize, event, &detail);
            return ptr::null_mut();
        }
    };

    let cancellation_requested = Arc::new(AtomicBool::new(false));
    let worker_cancellation = Arc::clone(&cancellation_requested);
    let user_data_address = user_data as usize;
    let Ok(handle) = std::thread::Builder::new()
        .name("brush-c-train-v2".to_owned())
        .spawn(move || {
            run_training_job_v2(
                dataset_path,
                train_options,
                JobCallback::V2 {
                    callback: progress_callback,
                    user_data_address,
                    state: EventJobState::v2(started),
                },
                &worker_cancellation,
            )
        })
    else {
        let mut event = BrushEventV2::new(BrushEventKindV2::Terminal, 0);
        event.error_code = BrushErrorCodeV2::Training;
        emit_v2_with_text(
            progress_callback,
            user_data_address,
            event,
            "failed to create training worker",
        );
        return ptr::null_mut();
    };

    let state = Arc::new(BrushJobState {
        cancellation_requested,
        completion: Mutex::new(JobCompletion {
            handle: Some(handle),
            result: None,
            joining: false,
        }),
        completion_changed: Condvar::new(),
    });
    Box::into_raw(Box::new(state)).cast::<BrushJobV2>()
}

/// Starts a callable ABI V3 training job. V3 adds one explicit
/// progressive-resolution transition while retaining the V2 event and job
/// ownership contracts.
///
/// # Safety
/// All non-null string pointers must remain readable for this call. `user_data`
/// must remain valid until a terminal callback or until the final retained job
/// handle has been released after waiting.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_train_start_v3(
    dataset_path: *const c_char,
    options: *const TrainOptionsV3,
    progress_callback: ProgressCallbackV2,
    user_data: *mut c_void,
) -> *mut BrushJobV2 {
    let started = Instant::now();
    let dataset_path = if dataset_path.is_null() {
        Err("dataset_path pointer is null".to_owned())
    } else {
        // SAFETY: caller guarantees a null-terminated C string.
        unsafe { CStr::from_ptr(dataset_path) }
            .to_str()
            .map(str::to_owned)
            .map_err(|_utf8_error| "dataset_path is not valid UTF-8".to_owned())
    };
    // SAFETY: forwarded V3 pointer contract.
    let options = unsafe { OwnedTrainOptionsV3::copy_from(options) };
    let (dataset_path, train_options) = match (dataset_path, options) {
        (Ok(dataset_path), Ok(options)) => (dataset_path, options),
        (dataset_path, options) => {
            let detail = dataset_path.err().or_else(|| options.err()).unwrap();
            let mut event = BrushEventV2::new(BrushEventKindV2::Terminal, 0);
            event.error_code = if detail.contains("ABI version") {
                BrushErrorCodeV2::UnsupportedAbi
            } else {
                BrushErrorCodeV2::InvalidArgument
            };
            emit_v2_with_text(progress_callback, user_data as usize, event, &detail);
            return ptr::null_mut();
        }
    };

    let cancellation_requested = Arc::new(AtomicBool::new(false));
    let worker_cancellation = Arc::clone(&cancellation_requested);
    let user_data_address = user_data as usize;
    let Ok(handle) = std::thread::Builder::new()
        .name("brush-c-train-v3".to_owned())
        .spawn(move || {
            run_training_job_v3(
                dataset_path,
                train_options,
                JobCallback::V2 {
                    callback: progress_callback,
                    user_data_address,
                    state: EventJobState::v2(started),
                },
                &worker_cancellation,
            )
        })
    else {
        let mut event = BrushEventV2::new(BrushEventKindV2::Terminal, 0);
        event.error_code = BrushErrorCodeV2::Training;
        emit_v2_with_text(
            progress_callback,
            user_data_address,
            event,
            "failed to create training worker",
        );
        return ptr::null_mut();
    };

    let state = Arc::new(BrushJobState {
        cancellation_requested,
        completion: Mutex::new(JobCompletion {
            handle: Some(handle),
            result: None,
            joining: false,
        }),
        completion_changed: Condvar::new(),
    });
    Box::into_raw(Box::new(state)).cast::<BrushJobV2>()
}

/// Starts a callable ABI V4 training job. V4 adds the default-disabled,
/// predeclared progressive-SH schedule and typed V4 events while reusing the
/// retained V2 job ownership functions.
///
/// # Safety
/// All non-null string pointers must remain readable for this call. `user_data`
/// must remain valid until a terminal callback or until the final retained job
/// handle has been released after waiting.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_train_start_v4(
    dataset_path: *const c_char,
    options: *const TrainOptionsV4,
    progress_callback: ProgressCallbackV4,
    user_data: *mut c_void,
) -> *mut BrushJobV2 {
    let started = Instant::now();
    let dataset_path = if dataset_path.is_null() {
        Err("dataset_path pointer is null".to_owned())
    } else {
        // SAFETY: caller guarantees a null-terminated C string.
        unsafe { CStr::from_ptr(dataset_path) }
            .to_str()
            .map(str::to_owned)
            .map_err(|_utf8_error| "dataset_path is not valid UTF-8".to_owned())
    };
    // SAFETY: forwarded V4 pointer contract.
    let options = unsafe { OwnedTrainOptionsV4::copy_from(options) };
    let (dataset_path, train_options) = match (dataset_path, options) {
        (Ok(dataset_path), Ok(options)) => (dataset_path, options),
        (dataset_path, options) => {
            let detail = dataset_path.err().or_else(|| options.err()).unwrap();
            let mut base = BrushEventV2::new(BrushEventKindV2::Terminal, 0);
            base.error_code = if detail.contains("ABI version") {
                BrushErrorCodeV2::UnsupportedAbi
            } else {
                BrushErrorCodeV2::InvalidArgument
            };
            let state = EventJobState::v2(started);
            let event = BrushEventV4::from_v2(BrushEventKindV4::Terminal, base, &state);
            emit_v4_with_text(progress_callback, user_data as usize, event, &detail);
            return ptr::null_mut();
        }
    };

    let event_state = EventJobState::v4(started, train_options.schedule());
    let cancellation_requested = Arc::new(AtomicBool::new(false));
    let worker_cancellation = Arc::clone(&cancellation_requested);
    let user_data_address = user_data as usize;
    let Ok(handle) = std::thread::Builder::new()
        .name("brush-c-train-v4".to_owned())
        .spawn(move || {
            run_training_job_v4(
                dataset_path,
                train_options,
                JobCallback::V4 {
                    callback: progress_callback,
                    user_data_address,
                    state: event_state,
                },
                &worker_cancellation,
            )
        })
    else {
        let mut base = BrushEventV2::new(BrushEventKindV2::Terminal, 0);
        base.error_code = BrushErrorCodeV2::Training;
        let state = EventJobState::v2(started);
        let event = BrushEventV4::from_v2(BrushEventKindV4::Terminal, base, &state);
        emit_v4_with_text(
            progress_callback,
            user_data_address,
            event,
            "failed to create training worker",
        );
        return ptr::null_mut();
    };

    let state = Arc::new(BrushJobState {
        cancellation_requested,
        completion: Mutex::new(JobCompletion {
            handle: Some(handle),
            result: None,
            joining: false,
        }),
        completion_changed: Condvar::new(),
    });
    Box::into_raw(Box::new(state)).cast::<BrushJobV2>()
}

/// Creates an independently releasable reference to a V2 job.
///
/// # Safety
/// `job` must be a live V2 handle. The returned handle must be released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_job_retain_v2(job: *mut BrushJobV2) -> *mut BrushJobV2 {
    // SAFETY: this function's contract requires a live V2 handle.
    let Some(state) = (unsafe { job_state_v2(job) }) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(Arc::clone(state))).cast::<BrushJobV2>()
}

/// Requests cooperative cancellation through a retained V2 job handle.
///
/// # Safety
/// `job` must be a live V2 handle. Null is accepted and returns false.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_job_cancel_v2(job: *mut BrushJobV2) -> bool {
    // SAFETY: this function's contract requires a live V2 handle.
    let Some(state) = (unsafe { job_state_v2(job) }) else {
        return false;
    };
    state.cancellation_requested.store(true, Ordering::SeqCst);
    true
}

/// Waits for a retained V2 job. Multiple retained handles may wait safely.
///
/// # Safety
/// `job` must be a live V2 handle. Null returns `TrainExitCode::Error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_job_wait_v2(job: *mut BrushJobV2) -> TrainExitCode {
    // SAFETY: this function's contract requires a live V2 handle.
    let Some(state) = (unsafe { job_state_v2(job) }) else {
        return TrainExitCode::Error;
    };
    wait_for_state(state)
}

/// Releases one retained V2 handle. The final release cancels and joins an
/// unfinished worker before freeing callback-visible state.
///
/// # Safety
/// `job` must be a live V2 handle and may not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_job_release_v2(job: *mut BrushJobV2) {
    if job.is_null() {
        return;
    }
    // SAFETY: V2 handles are Box<Arc<BrushJobState>> allocations.
    let state = unsafe { Box::from_raw(job.cast::<Arc<BrushJobState>>()) };
    if Arc::strong_count(&state) == 1 {
        state.cancellation_requested.store(true, Ordering::SeqCst);
        let _ = wait_for_state(&state);
    }
}

/// Requests cooperative cancellation for a running Brush training job.
///
/// # Safety
///
/// `job` must be a handle returned by `brush_train_start` that has not been
/// released yet. Passing null is safe and returns `false`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_job_cancel(job: *mut BrushJob) -> bool {
    let Some(state) = job_state(job) else {
        return false;
    };
    state.cancellation_requested.store(true, Ordering::SeqCst);
    true
}

/// Waits for a Brush training job to finish and returns its terminal status.
///
/// # Safety
///
/// `job` must be a handle returned by `brush_train_start` that has not been
/// released yet. Passing null is safe and returns `TrainExitCode::Error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_job_wait(job: *mut BrushJob) -> TrainExitCode {
    let Some(state) = job_state(job) else {
        return TrainExitCode::Error;
    };
    wait_for_state(state)
}

/// Releases a Brush training job handle.
///
/// If the job is still running, release requests cancellation and waits for the
/// worker to finish before freeing the handle.
///
/// # Safety
///
/// `job` must be a handle returned by `brush_train_start` and must not be used
/// after this call. Passing null is safe.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_job_release(job: *mut BrushJob) {
    if job.is_null() {
        return;
    }

    // SAFETY: The caller guarantees this handle came from Box::into_raw in brush_train_start.
    let state = unsafe { Box::from_raw(job.cast::<BrushJobState>()) };
    state.cancellation_requested.store(true, Ordering::SeqCst);
    let _ = wait_for_state(&state);
}

/// Trains a model from a dataset and saves the result.
///
/// This compatibility wrapper blocks the current thread until training is
/// complete. New native integrations should use `brush_train_start`,
/// `brush_job_cancel`, `brush_job_wait`, and `brush_job_release`.
///
/// # Safety
///
/// The caller must uphold several invariants. Passing `null` for `dataset_path`
/// or `options` is safe and will result in an error code, but if they are
/// non-null, they must be valid.
///
/// - If `dataset_path` is not null, it must point to a valid, null-terminated C
///   string.
/// - If `options` is not null, it must point to a valid `TrainOptions` struct.
///   Its `output_path` and `export_name` must be valid, null-terminated C
///   strings if not null.
/// - The `user_data` pointer is passed to `progress_callback` and must remain
///   valid for the duration of this function call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_and_save(
    dataset_path: *const c_char,
    options: *const TrainOptions,
    progress_callback: ProgressCallback,
    user_data: *mut c_void,
) -> TrainExitCode {
    // SAFETY: This wrapper forwards the caller's FFI contract to the async job API.
    let job = unsafe { brush_train_start(dataset_path, options, progress_callback, user_data) };
    if job.is_null() {
        return TrainExitCode::Error;
    }
    // SAFETY: `job` was returned by brush_train_start and has not been released.
    let status = unsafe { brush_job_wait(job) };
    // SAFETY: `job` was returned by brush_train_start and is no longer needed.
    unsafe { brush_job_release(job) };
    status
}

/// Blocking convenience wrapper for callable ABI V2.
///
/// # Safety
/// This forwards the complete `brush_train_start_v2` pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_and_save_v2(
    dataset_path: *const c_char,
    options: *const TrainOptionsV2,
    progress_callback: ProgressCallbackV2,
    user_data: *mut c_void,
) -> TrainExitCode {
    // SAFETY: forwards the caller's V2 FFI contract.
    let job = unsafe { brush_train_start_v2(dataset_path, options, progress_callback, user_data) };
    if job.is_null() {
        return TrainExitCode::Error;
    }
    // SAFETY: job is live until the matching release below.
    let status = unsafe { brush_job_wait_v2(job) };
    // SAFETY: release the one owner returned by start.
    unsafe { brush_job_release_v2(job) };
    status
}

/// Blocking convenience wrapper for callable ABI V4.
///
/// # Safety
/// This forwards the complete `brush_train_start_v4` pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_and_save_v4(
    dataset_path: *const c_char,
    options: *const TrainOptionsV4,
    progress_callback: ProgressCallbackV4,
    user_data: *mut c_void,
) -> TrainExitCode {
    // SAFETY: forwards the caller's V4 FFI contract.
    let job = unsafe { brush_train_start_v4(dataset_path, options, progress_callback, user_data) };
    if job.is_null() {
        return TrainExitCode::Error;
    }
    // SAFETY: job is live until the matching release below.
    let status = unsafe { brush_job_wait_v2(job) };
    // SAFETY: release the one owner returned by start.
    unsafe { brush_job_release_v2(job) };
    status
}

fn job_state<'a>(job: *mut BrushJob) -> Option<&'a BrushJobState> {
    if job.is_null() {
        return None;
    }
    // SAFETY: Callers pass handles returned by brush_train_start.
    Some(unsafe { &*job.cast::<BrushJobState>() })
}

unsafe fn job_state_v2<'a>(job: *mut BrushJobV2) -> Option<&'a Arc<BrushJobState>> {
    if job.is_null() {
        return None;
    }
    // SAFETY: V2 callers pass a live Box<Arc<BrushJobState>> handle.
    Some(unsafe { &*job.cast::<Arc<BrushJobState>>() })
}

fn wait_for_state(state: &BrushJobState) -> TrainExitCode {
    let mut completion = state
        .completion
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    loop {
        if let Some(result) = completion.result {
            return result;
        }
        if !completion.joining
            && let Some(handle) = completion.handle.take()
        {
            completion.joining = true;
            drop(completion);
            let result = handle.join().unwrap_or(TrainExitCode::Error);
            completion = state
                .completion
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            completion.result = Some(result);
            completion.joining = false;
            state.completion_changed.notify_all();
            return result;
        }
        completion = state
            .completion_changed
            .wait(completion)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
}

fn run_training_job(
    dataset_path: String,
    train_options: OwnedTrainOptions,
    callback: JobCallback,
    cancellation_requested: &AtomicBool,
    serialize_v2: bool,
) -> TrainExitCode {
    run_training_job_with_config(
        dataset_path,
        train_options.into_train_stream_config(),
        callback,
        cancellation_requested,
        serialize_v2,
    )
}

fn run_training_job_v2(
    dataset_path: String,
    train_options: OwnedTrainOptionsV2,
    callback: JobCallback,
    cancellation_requested: &AtomicBool,
) -> TrainExitCode {
    run_training_job_with_config(
        dataset_path,
        train_options.into_train_stream_config(),
        callback,
        cancellation_requested,
        true,
    )
}

fn run_training_job_v3(
    dataset_path: String,
    train_options: OwnedTrainOptionsV3,
    callback: JobCallback,
    cancellation_requested: &AtomicBool,
) -> TrainExitCode {
    run_training_job_with_config(
        dataset_path,
        train_options.into_train_stream_config(),
        callback,
        cancellation_requested,
        true,
    )
}

fn run_training_job_v4(
    dataset_path: String,
    train_options: OwnedTrainOptionsV4,
    callback: JobCallback,
    cancellation_requested: &AtomicBool,
) -> TrainExitCode {
    run_training_job_with_config(
        dataset_path,
        train_options.into_train_stream_config(),
        callback,
        cancellation_requested,
        true,
    )
}

fn run_training_job_with_config(
    dataset_path: String,
    process_args: TrainStreamConfig,
    mut callback: JobCallback,
    cancellation_requested: &AtomicBool,
    serialize_v2: bool,
) -> TrainExitCode {
    if matches!(callback, JobCallback::V2 { .. } | JobCallback::V4 { .. }) {
        emit_v2_capabilities(&callback);
        emit_v2_configuration(&callback, &process_args);
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _training_guard = serialize_v2.then(|| {
            V2_TRAINING_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        configure_cubecl_gpu_profile(process_args.host_runtime.phase0_telemetry);
        let source = DataSource::Path(dataset_path);
        let mut process = create_process(source, async move |_| Some(process_args));

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
            .block_on(async {
                SETUP
                    .get_or_init(async move || {
                        burn_init_setup().await;
                    })
                    .await;

                while let Some(message_result) = process.stream.next().await {
                    if cancellation_requested.load(Ordering::SeqCst) {
                        return Ok(TrainExitCode::Cancelled);
                    }
                    match message_result {
                        Ok(message) => {
                            emit_progress_message(message, &mut callback);
                        }
                        Err(error) => {
                            return Err(error.to_string());
                        }
                    }
                    if cancellation_requested.load(Ordering::SeqCst) {
                        return Ok(TrainExitCode::Cancelled);
                    }
                }

                if cancellation_requested.load(Ordering::SeqCst) {
                    Ok(TrainExitCode::Cancelled)
                } else {
                    Ok(TrainExitCode::Success)
                }
            })
    }));

    let (status, detail, error_code) = match result {
        Ok(Ok(status)) => {
            let code = if status == TrainExitCode::Cancelled {
                BrushErrorCodeV2::Cancelled
            } else {
                BrushErrorCodeV2::None
            };
            (status, None, code)
        }
        Ok(Err(detail)) => {
            let code = if detail.contains("initializer") || detail.contains("SH degree") {
                BrushErrorCodeV2::Initializer
            } else if detail.contains("dataset") || detail.contains("Format") {
                BrushErrorCodeV2::Dataset
            } else {
                BrushErrorCodeV2::Training
            };
            (TrainExitCode::Error, Some(detail), code)
        }
        Err(_) => (
            TrainExitCode::Error,
            Some("native training panic was contained at the FFI boundary".to_owned()),
            BrushErrorCodeV2::Panic,
        ),
    };
    emit_v2_terminal(&callback, status, error_code, detail.as_deref());
    status
}

#[cfg(brushkit_cubecl_gpu_profile)]
fn configure_cubecl_gpu_profile(phase0_telemetry_requested: bool) {
    use std::sync::Once;

    if !phase0_telemetry_requested {
        return;
    }

    static CONFIGURE_ONCE: Once = Once::new();
    CONFIGURE_ONCE.call_once(|| {
        let mut config = CubeClRuntimeConfig::default();
        config.profiling.logger.level = ProfilingLogLevel::Basic;
        config.profiling.logger.stdout = true;
        CubeClRuntimeConfig::set(config);
        println!("BRUSHKIT_CUBECL_GPU_PROFILE=basic_stdout_device_if_supported");
    });
}

#[cfg(not(brushkit_cubecl_gpu_profile))]
fn configure_cubecl_gpu_profile(_phase0_telemetry_requested: bool) {}

fn emit_progress_message(message: ProcessMessage, callback: &mut JobCallback) {
    if matches!(callback, JobCallback::V2 { .. } | JobCallback::V4 { .. }) {
        emit_progress_message_v2(&message, callback);
        return;
    }
    let JobCallback::Legacy {
        callback: progress_callback,
        user_data_address,
    } = callback
    else {
        unreachable!();
    };
    let user_data = *user_data_address as *mut c_void;
    match message {
        ProcessMessage::NewProcess => {
            progress_callback(
                ProgressMessage::new(ProgressMessageKind::NewProcess),
                user_data,
            );
        }
        ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter, .. }) => {
            progress_callback(
                ProgressMessage {
                    kind: ProgressMessageKind::Training,
                    iter,
                    path: ptr::null(),
                },
                user_data,
            );
        }
        ProcessMessage::TrainMessage(TrainMessage::CheckpointExported { iter, path }) => {
            let path_string = path.to_string_lossy();
            let path_cstring = CString::new(path_string.as_bytes()).ok();
            progress_callback(
                ProgressMessage {
                    kind: ProgressMessageKind::CheckpointExported,
                    iter,
                    path: path_cstring
                        .as_ref()
                        .map_or(ptr::null(), |path| path.as_ptr()),
                },
                user_data,
            );
        }
        ProcessMessage::TrainMessage(TrainMessage::DoneTraining) => {
            progress_callback(
                ProgressMessage::new(ProgressMessageKind::DoneTraining),
                user_data,
            );
        }
        _ => {}
    }
}

fn duration_ns(duration: impl Into<std::time::Duration>) -> u64 {
    duration.into().as_nanos().min(u64::MAX as u128) as u64
}

fn v2_timestamp(callback: &JobCallback) -> u64 {
    match callback {
        JobCallback::V2 { state, .. } | JobCallback::V4 { state, .. } => {
            duration_ns(state.started.elapsed())
        }
        JobCallback::Legacy { .. } => 0,
    }
}

fn event_state(callback: &JobCallback) -> Option<&EventJobState> {
    match callback {
        JobCallback::V2 { state, .. } | JobCallback::V4 { state, .. } => Some(state),
        JobCallback::Legacy { .. } => None,
    }
}

fn event_state_mut(callback: &mut JobCallback) -> Option<&mut EventJobState> {
    match callback {
        JobCallback::V2 { state, .. } | JobCallback::V4 { state, .. } => Some(state),
        JobCallback::Legacy { .. } => None,
    }
}

fn emit_v2_with_text(
    callback: ProgressCallbackV2,
    user_data_address: usize,
    mut event: BrushEventV2,
    text: &str,
) {
    let text = CString::new(text.replace('\0', "�")).ok();
    event.text = text.as_ref().map_or(ptr::null(), |text| text.as_ptr());
    callback(event, user_data_address as *mut c_void);
}

fn emit_v4_with_text(
    callback: ProgressCallbackV4,
    user_data_address: usize,
    mut event: BrushEventV4,
    text: &str,
) {
    let text = CString::new(text.replace('\0', "�")).ok();
    event.base.text = text.as_ref().map_or(ptr::null(), |text| text.as_ptr());
    callback(event, user_data_address as *mut c_void);
}

fn v4_kind(kind: BrushEventKindV2) -> BrushEventKindV4 {
    match kind {
        BrushEventKindV2::Capabilities => BrushEventKindV4::Capabilities,
        BrushEventKindV2::Configuration => BrushEventKindV4::Configuration,
        BrushEventKindV2::DatasetLoadStarted => BrushEventKindV4::DatasetLoadStarted,
        BrushEventKindV2::DatasetLoadFinished => BrushEventKindV4::DatasetLoadFinished,
        BrushEventKindV2::Initializer => BrushEventKindV4::Initializer,
        BrushEventKindV2::TrainerInitializationStarted => {
            BrushEventKindV4::TrainerInitializationStarted
        }
        BrushEventKindV2::TrainerInitializationFinished => {
            BrushEventKindV4::TrainerInitializationFinished
        }
        BrushEventKindV2::Step => BrushEventKindV4::Step,
        BrushEventKindV2::Refinement => BrushEventKindV4::Refinement,
        BrushEventKindV2::CheckpointExportStarted => BrushEventKindV4::CheckpointExportStarted,
        BrushEventKindV2::CheckpointExported => BrushEventKindV4::CheckpointExported,
        BrushEventKindV2::Terminal => BrushEventKindV4::Terminal,
        BrushEventKindV2::TerminalCompaction => BrushEventKindV4::TerminalCompaction,
    }
}

fn emit_v2(callback: &JobCallback, event: BrushEventV2, text: Option<&str>) {
    match callback {
        JobCallback::V2 {
            callback,
            user_data_address,
            ..
        } => {
            if let Some(text) = text {
                emit_v2_with_text(*callback, *user_data_address, event, text);
            } else {
                callback(event, *user_data_address as *mut c_void);
            }
        }
        JobCallback::V4 {
            callback,
            user_data_address,
            state,
        } => {
            let kind = v4_kind(event.kind);
            let event = BrushEventV4::from_v2(kind, event, state);
            if let Some(text) = text {
                emit_v4_with_text(*callback, *user_data_address, event, text);
            } else {
                callback(event, *user_data_address as *mut c_void);
            }
        }
        JobCallback::Legacy { .. } => {}
    }
}

fn emit_v2_capabilities(callback: &JobCallback) {
    let mut event = BrushEventV2::new(BrushEventKindV2::Capabilities, v2_timestamp(callback));
    event.capability_flags = BRUSH_AVAILABLE_CAPABILITIES_V2;
    if matches!(callback, JobCallback::V4 { .. }) {
        event.capability_flags |= BRUSH_CAPABILITY_PROGRESSIVE_SH_V4;
    }
    emit_v2(
        callback,
        event,
        Some(
            "unavailable: adapter identity is not exposed by default Burn setup; GPU allocated memory would stall the compute server; GPU command timing is not exposed without added synchronization; operation timings are host-observed around existing work; clone is not distinct from split; residual GPU scheduling and floating-point nondeterminism may remain",
        ),
    );
}

fn emit_v2_configuration(callback: &JobCallback, config: &TrainStreamConfig) {
    let mut event = BrushEventV2::new(BrushEventKindV2::Configuration, v2_timestamp(callback));
    event.seed = config.process_config.seed;
    event.render_mode = match config.train_config.render_mode {
        Some(SplatRenderMode::Mip) => 1,
        _ => 0,
    };
    event.iteration = config.train_config.total_train_iters;
    event.refine_every = config.train_config.refine_every;
    event.max_resolution = config.load_config.max_resolution;
    event.max_splats = config.train_config.max_splats;
    event.export_every = config.process_config.export_every;
    event.configured_sh_degree = config.model_config.sh_degree;
    event.instrumentation_level = u32::from(config.host_runtime.phase0_telemetry);
    emit_v2(callback, event, None);
}

fn emit_progress_message_v2(message: &ProcessMessage, callback: &mut JobCallback) {
    let timestamp = v2_timestamp(callback);
    let mut event_and_text: Option<(BrushEventV2, Option<String>)> = None;
    match message {
        ProcessMessage::TrainMessage(TrainMessage::TelemetryBoundary { boundary }) => {
            let kind = match boundary {
                TelemetryBoundary::DatasetLoadStarted => BrushEventKindV2::DatasetLoadStarted,
                TelemetryBoundary::DatasetLoadFinished => BrushEventKindV2::DatasetLoadFinished,
                TelemetryBoundary::TrainerInitializationStarted => {
                    BrushEventKindV2::TrainerInitializationStarted
                }
                TelemetryBoundary::TrainerInitializationFinished => {
                    BrushEventKindV2::TrainerInitializationFinished
                }
            };
            event_and_text = Some((BrushEventV2::new(kind, timestamp), None));
        }
        ProcessMessage::TrainMessage(TrainMessage::Initializer { report }) => {
            let mut event = BrushEventV2::new(BrushEventKindV2::Initializer, timestamp);
            event.primitive_count = report.primitive_count;
            event.initial_primitive_count = report.primitive_count;
            event.fields_consumed = report.fields_consumed;
            event.rejected_count = report.rejected_count;
            event.supplied_sh_degree = report.supplied_sh_degree.map_or(-1, |degree| degree as i32);
            event.configured_sh_degree = report.configured_sh_degree;
            event.initializer_route = match report.route {
                InitializerRoute::Random => BrushInitializerRouteV2::Random,
                InitializerRoute::DatasetSparse => BrushInitializerRouteV2::DatasetSparse,
                InitializerRoute::ImplicitPly => BrushInitializerRouteV2::ImplicitPly,
                InitializerRoute::ExplicitPly => BrushInitializerRouteV2::ExplicitPly,
                InitializerRoute::ExplicitStrong => BrushInitializerRouteV2::ExplicitStrong,
            };
            if let Some(state) = event_state_mut(callback) {
                state.initial_primitive_count = report.primitive_count;
                state.last_primitive_count = report.primitive_count;
                state.terminal_exported_primitive_count = None;
            }
            event_and_text = Some((
                event,
                report
                    .path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
            ));
        }
        ProcessMessage::TrainMessage(TrainMessage::TrainStep {
            iter,
            step_duration,
            data_wait_duration,
            forward_duration,
            loss_duration,
            backward_duration,
            optimizer_duration,
            optimizer_transforms_duration,
            optimizer_sh_coeffs_duration,
            optimizer_opacity_duration,
            render_before_count_readback_duration,
            render_count_readback_duration,
            render_after_count_readback_duration,
            live_splat_count,
            ..
        }) => {
            let mut event = BrushEventV2::new(BrushEventKindV2::Step, timestamp);
            event.iteration = *iter;
            event.primitive_count = *live_splat_count;
            event.duration_ns = duration_ns(*step_duration);
            event.data_wait_ns = duration_ns(*data_wait_duration);
            event.forward_ns = duration_ns(*forward_duration);
            event.loss_and_ssim_ns = duration_ns(*loss_duration);
            event.backward_ns = duration_ns(*backward_duration);
            event.optimizer_ns = duration_ns(*optimizer_duration);
            if let Some(state) = event_state_mut(callback) {
                state.last_primitive_count = *live_splat_count;
                state.zero_based_iteration = iter.saturating_sub(1);
            }
            let optimizer_substage_text = optimizer_transforms_duration
                .as_ref()
                .zip(optimizer_sh_coeffs_duration.as_ref())
                .zip(optimizer_opacity_duration.as_ref())
                .map(|((transforms, sh_coeffs), opacity)| {
                    format!(
                        "optimizer_substages transforms_ns={} sh_coeffs_ns={} opacity_ns={}",
                        duration_ns(*transforms),
                        duration_ns(*sh_coeffs),
                        duration_ns(*opacity),
                    )
                });
            let render_host_text = render_before_count_readback_duration
                .as_ref()
                .zip(render_count_readback_duration.as_ref())
                .zip(render_after_count_readback_duration.as_ref())
                .map(|((before, readback), after)| {
                    format!(
                        "render_host before_count_readback_ns={} count_readback_ns={} after_count_readback_ns={}",
                        duration_ns(*before),
                        duration_ns(*readback),
                        duration_ns(*after),
                    )
                });
            let telemetry_text = match (optimizer_substage_text, render_host_text) {
                (Some(optimizer), Some(render)) => Some(format!("{optimizer}; {render}")),
                (Some(optimizer), None) => Some(optimizer),
                (None, Some(render)) => Some(render),
                (None, None) => None,
            };
            event_and_text = Some((event, telemetry_text));
        }
        ProcessMessage::TrainMessage(TrainMessage::RefineStep {
            cur_splat_count,
            iter,
            num_added,
            num_split_oversized,
            num_split_high_grad,
            num_pruned,
            num_pruned_non_finite,
            duration,
            ..
        }) => {
            let mut event = BrushEventV2::new(BrushEventKindV2::Refinement, timestamp);
            event.iteration = *iter;
            event.primitive_count = *cur_splat_count;
            event.added_count = *num_added;
            // Brush's current refiner duplicates selected Gaussians and then
            // perturbs/shrinks them; there is no genuinely distinct clone op.
            event.split_count = *num_added;
            event.split_oversized_count = *num_split_oversized;
            event.split_high_gradient_count = *num_split_high_grad;
            event.clone_count = 0;
            event.pruned_count = *num_pruned;
            event.pruned_non_finite_count = *num_pruned_non_finite;
            event.net_growth = i64::from(*num_added) - i64::from(*num_pruned);
            event.densification_and_compaction_ns = duration_ns(*duration);
            if let Some(state) = event_state_mut(callback) {
                state.last_primitive_count = *cur_splat_count;
            }
            event_and_text = Some((event, None));
        }
        ProcessMessage::TrainMessage(TrainMessage::CheckpointExported { iter, path }) => {
            let mut event = BrushEventV2::new(BrushEventKindV2::CheckpointExported, timestamp);
            event.iteration = *iter;
            event_and_text = Some((event, Some(path.to_string_lossy().into_owned())));
        }
        ProcessMessage::TrainMessage(TrainMessage::CheckpointExportStarted { iter }) => {
            let mut event = BrushEventV2::new(BrushEventKindV2::CheckpointExportStarted, timestamp);
            event.iteration = *iter;
            event_and_text = Some((event, None));
        }
        ProcessMessage::TrainMessage(TrainMessage::TerminalCompaction { iter, report }) => {
            let mut event = BrushEventV2::new(BrushEventKindV2::TerminalCompaction, timestamp);
            event.iteration = *iter;
            event.primitive_count = report.exported_splat_count;
            event.pruned_count = report.pruned_non_finite_count;
            event.pruned_non_finite_count = report.pruned_non_finite_count;
            event.net_growth = -i64::from(report.pruned_non_finite_count);
            event.densification_and_compaction_ns = duration_ns(report.validation_duration);
            if let Some(state) = event_state_mut(callback) {
                state.last_primitive_count = report.exported_splat_count;
                state.terminal_exported_primitive_count = Some(report.exported_splat_count);
            }
            event_and_text = Some((
                event,
                Some(format!(
                    "terminal_export_validation source={} exported={} transforms_non_finite_rows={} sh_non_finite_rows={} opacity_non_finite_rows={}",
                    report.original_splat_count,
                    report.exported_splat_count,
                    report.transforms_non_finite_row_count,
                    report.sh_coeffs_non_finite_row_count,
                    report.opacity_non_finite_row_count,
                )),
            ));
        }
        ProcessMessage::TrainMessage(TrainMessage::ShDegreeChanged {
            zero_based_iteration,
            maximum_degree,
            active_degree,
            initial_degree,
            step_interval,
            schedule_mode,
            schedule_identity,
            reason,
        }) => {
            if matches!(callback, JobCallback::V4 { .. }) {
                let timestamp = v2_timestamp(callback);
                let state = event_state_mut(callback).expect("V4 callback has event state");
                state.maximum_sh_degree = *maximum_degree;
                state.active_sh_degree = *active_degree;
                state.initial_sh_degree = *initial_degree;
                state.sh_degree_step_interval = *step_interval;
                state.sh_schedule_mode = *schedule_mode;
                state.schedule_identity = *schedule_identity;
                state.zero_based_iteration = *zero_based_iteration;
                let mut base = BrushEventV2::new(BrushEventKindV2::Configuration, timestamp);
                base.iteration = *zero_based_iteration;
                base.configured_sh_degree = *maximum_degree;
                let JobCallback::V4 {
                    callback,
                    user_data_address,
                    state,
                } = callback
                else {
                    unreachable!();
                };
                let mut event =
                    BrushEventV4::from_v2(BrushEventKindV4::ActiveSHDegreeChanged, base, state);
                event.zero_based_iteration = *zero_based_iteration;
                event.transition_reason = match reason {
                    ShTransitionReason::Disabled => BrushSHTransitionReasonV4::Disabled,
                    ShTransitionReason::Initial => BrushSHTransitionReasonV4::Initial,
                    ShTransitionReason::Scheduled => BrushSHTransitionReasonV4::Scheduled,
                    ShTransitionReason::Resume => BrushSHTransitionReasonV4::Resume,
                };
                callback(event, *user_data_address as *mut c_void);
            }
        }
        _ => {}
    }

    if let Some((event, text)) = event_and_text {
        emit_v2(callback, event, text.as_deref());
    }
}

fn emit_v2_terminal(
    callback: &JobCallback,
    status: TrainExitCode,
    error_code: BrushErrorCodeV2,
    detail: Option<&str>,
) {
    let mut event = BrushEventV2::new(BrushEventKindV2::Terminal, v2_timestamp(callback));
    event.error_code = error_code;
    if let Some(state) = event_state(callback) {
        event.initial_primitive_count = state.initial_primitive_count;
        event.final_primitive_count = state
            .terminal_exported_primitive_count
            .unwrap_or(state.last_primitive_count);
    }
    if status == TrainExitCode::Cancelled && event.error_code == BrushErrorCodeV2::None {
        event.error_code = BrushErrorCodeV2::Cancelled;
    }
    emit_v2(callback, event, detail);
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_process::message::InitializerReport;
    use brush_serde::SplatExportValidationReport;

    extern "C" fn collect_event(event: BrushEventV2, user_data: *mut c_void) {
        // SAFETY: the test keeps this Vec alive for every synchronous callback.
        unsafe { &mut *user_data.cast::<Vec<BrushEventV2>>() }.push(event);
    }

    extern "C" fn collect_event_v4(event: BrushEventV4, user_data: *mut c_void) {
        // SAFETY: the test keeps this Vec alive for every synchronous callback.
        unsafe { &mut *user_data.cast::<Vec<BrushEventV4>>() }.push(event);
    }

    fn v4_options(mode: u32) -> TrainOptionsV4 {
        let enabled = mode == PROGRESSIVE_SH_MODE_INTERVAL;
        TrainOptionsV4 {
            struct_size: mem::size_of::<TrainOptionsV4>() as u32,
            abi_version: BRUSH_ABI_VERSION_V4,
            seed: 0,
            render_mode: BRUSH_RENDER_MODE_DEFAULT_V2,
            sh_degree: 3,
            sh_policy: BRUSH_SH_POLICY_PRESERVE_AND_ZERO_PAD_V2,
            total_train_steps: if enabled { 600 } else { 10 },
            refine_every: 200,
            max_resolution: 720,
            max_splats: 500_000,
            export_every: if enabled { 600 } else { 10 },
            output_path: ptr::null(),
            export_name: ptr::null(),
            initializer_path: ptr::null(),
            initializer_required: 0,
            instrumentation_level: BRUSH_INSTRUMENTATION_DISABLED_V2,
            progressive_resolution_start_percent: 100,
            progressive_resolution_switch_iteration: 0,
            sh_schedule_mode: mode,
            initial_sh_degree: if enabled { 0 } else { 3 },
            sh_degree_step_interval: if enabled { 150 } else { 0 },
        }
    }

    #[test]
    fn released_layouts_are_unchanged_and_v4_is_explicit() {
        assert_eq!(mem::size_of::<TrainOptionsV2>(), 80);
        assert_eq!(mem::size_of::<TrainOptionsV3>(), 88);
        assert_eq!(mem::size_of::<BrushEventV2>(), 200);
        assert_eq!(mem::size_of::<TrainOptionsV4>(), 104);
        assert_eq!(mem::size_of::<BrushEventV4>(), 256);
        assert_eq!(brush_get_abi_version(), BRUSH_ABI_VERSION_V2);
    }

    #[test]
    fn v4_accepts_only_disabled_or_the_predeclared_schedule() {
        let disabled = v4_options(PROGRESSIVE_SH_MODE_DISABLED);
        // SAFETY: local exact V4 storage and null optional strings.
        let disabled = unsafe { OwnedTrainOptionsV4::copy_from(&disabled) }.unwrap();
        assert_eq!(disabled.schedule().mode(), PROGRESSIVE_SH_MODE_DISABLED);
        assert!(
            disabled
                .into_train_stream_config()
                .host_runtime
                .progressive_sh_schedule
                .is_none(),
            "disabled V4 must use the existing fixed-SH engine path"
        );

        let enabled = v4_options(PROGRESSIVE_SH_MODE_INTERVAL);
        // SAFETY: local exact V4 storage and null optional strings.
        let enabled = unsafe { OwnedTrainOptionsV4::copy_from(&enabled) }.unwrap();
        assert_eq!(
            enabled.schedule().transitions(),
            &[(0, 0), (150, 1), (300, 2), (450, 3)]
        );

        let mut invalid = v4_options(PROGRESSIVE_SH_MODE_INTERVAL);
        invalid.sh_degree_step_interval = 149;
        // SAFETY: local exact V4 storage and null optional strings.
        let error = unsafe { OwnedTrainOptionsV4::copy_from(&invalid) }
            .err()
            .unwrap();
        assert!(error.contains("predeclared SH3 schedule"));
        let mut invalid = v4_options(99);
        // Avoid the helper's disabled field defaults disguising the mode error.
        invalid.initial_sh_degree = 0;
        invalid.sh_degree_step_interval = 150;
        // SAFETY: local exact V4 storage and null optional strings.
        let error = unsafe { OwnedTrainOptionsV4::copy_from(&invalid) }
            .err()
            .unwrap();
        assert!(error.contains("unsupported progressive SH schedule mode"));
        let mut malformed = v4_options(PROGRESSIVE_SH_MODE_INTERVAL);
        malformed.struct_size -= 1;
        // SAFETY: local readable storage; malformed size is intentionally rejected.
        let error = unsafe { OwnedTrainOptionsV4::copy_from(&malformed) }
            .err()
            .unwrap();
        assert!(error.contains("struct_size"));
    }

    #[test]
    fn v4_transition_events_are_typed_and_ordered_before_their_steps() {
        let schedule =
            ProgressiveShSchedule::new(PROGRESSIVE_SH_MODE_INTERVAL, 3, 0, 150, 600).unwrap();
        let mut events = Vec::<BrushEventV4>::new();
        let mut callback = JobCallback::V4 {
            callback: collect_event_v4,
            user_data_address: std::ptr::from_mut(&mut events) as usize,
            state: EventJobState::v4(Instant::now(), &schedule),
        };
        for (iteration, degree) in schedule.transitions() {
            emit_progress_message_v2(
                &ProcessMessage::TrainMessage(TrainMessage::ShDegreeChanged {
                    zero_based_iteration: *iteration,
                    maximum_degree: 3,
                    active_degree: *degree,
                    initial_degree: 0,
                    step_interval: 150,
                    schedule_mode: PROGRESSIVE_SH_MODE_INTERVAL,
                    schedule_identity: schedule.identity(),
                    reason: if *iteration == 0 {
                        ShTransitionReason::Initial
                    } else {
                        ShTransitionReason::Scheduled
                    },
                }),
                &mut callback,
            );
            emit_progress_message_v2(
                &ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter: *iteration + 1,
                    total_elapsed: std::time::Duration::ZERO,
                    step_duration: std::time::Duration::ZERO,
                    data_wait_duration: std::time::Duration::ZERO,
                    forward_duration: std::time::Duration::ZERO,
                    loss_duration: std::time::Duration::ZERO,
                    backward_duration: std::time::Duration::ZERO,
                    optimizer_duration: std::time::Duration::ZERO,
                    optimizer_transforms_duration: None,
                    optimizer_sh_coeffs_duration: None,
                    optimizer_opacity_duration: None,
                    render_before_count_readback_duration: None,
                    render_count_readback_duration: None,
                    render_after_count_readback_duration: None,
                    live_splat_count: 1,
                    lod_progress: None,
                }),
                &mut callback,
            );
        }
        assert_eq!(events.len(), 8);
        for (index, (iteration, degree)) in schedule.transitions().iter().enumerate() {
            let transition = events[index * 2];
            let step = events[index * 2 + 1];
            assert_eq!(transition.kind, BrushEventKindV4::ActiveSHDegreeChanged);
            assert_eq!(transition.zero_based_iteration, *iteration);
            assert_eq!(transition.active_sh_degree, *degree);
            assert_eq!(transition.maximum_sh_degree, 3);
            assert_eq!(transition.schedule_identity, schedule.identity());
            assert_eq!(step.kind, BrushEventKindV4::Step);
            assert_eq!(step.base.iteration, *iteration + 1);
            assert_eq!(step.active_sh_degree, *degree);
        }
    }

    #[test]
    fn terminal_count_uses_export_compaction_after_delayed_step() {
        let mut events: Vec<BrushEventV2> = Vec::new();
        let mut callback = JobCallback::V2 {
            callback: collect_event,
            user_data_address: std::ptr::from_mut(&mut events) as usize,
            state: EventJobState::v2(Instant::now()),
        };

        emit_progress_message_v2(
            &ProcessMessage::TrainMessage(TrainMessage::Initializer {
                report: InitializerReport {
                    route: InitializerRoute::ExplicitStrong,
                    path: None,
                    primitive_count: 11_169,
                    fields_consumed: 0,
                    rejected_count: 0,
                    supplied_sh_degree: Some(0),
                    configured_sh_degree: 3,
                },
            }),
            &mut callback,
        );
        emit_progress_message_v2(
            &ProcessMessage::TrainMessage(TrainMessage::TerminalCompaction {
                iter: 300,
                report: SplatExportValidationReport {
                    original_splat_count: 13_209,
                    exported_splat_count: 13_198,
                    pruned_non_finite_count: 11,
                    transforms_non_finite_row_count: 11,
                    sh_coeffs_non_finite_row_count: 0,
                    opacity_non_finite_row_count: 11,
                    validation_duration: std::time::Duration::ZERO,
                },
            }),
            &mut callback,
        );
        emit_progress_message_v2(
            &ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                iter: 300,
                total_elapsed: std::time::Duration::ZERO,
                step_duration: std::time::Duration::ZERO,
                data_wait_duration: std::time::Duration::ZERO,
                forward_duration: std::time::Duration::ZERO,
                loss_duration: std::time::Duration::ZERO,
                backward_duration: std::time::Duration::ZERO,
                optimizer_duration: std::time::Duration::ZERO,
                optimizer_transforms_duration: None,
                optimizer_sh_coeffs_duration: None,
                optimizer_opacity_duration: None,
                render_before_count_readback_duration: None,
                render_count_readback_duration: None,
                render_after_count_readback_duration: None,
                live_splat_count: 13_209,
                lod_progress: None,
            }),
            &mut callback,
        );
        emit_v2_terminal(
            &callback,
            TrainExitCode::Success,
            BrushErrorCodeV2::None,
            None,
        );

        let compaction = events
            .iter()
            .find(|event| event.kind == BrushEventKindV2::TerminalCompaction)
            .expect("missing terminal compaction event");
        assert_eq!(compaction.primitive_count, 13_198);
        assert_eq!(compaction.pruned_non_finite_count, 11);
        let terminal = events.last().expect("missing terminal event");
        assert_eq!(terminal.kind, BrushEventKindV2::Terminal);
        assert_eq!(terminal.initial_primitive_count, 11_169);
        assert_eq!(terminal.final_primitive_count, 13_198);
    }
}
