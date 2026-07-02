// brush-c is a native-only FFI shim. The crate compiles to an empty stub on wasm.
#![cfg(not(target_family = "wasm"))]

use brush_process::burn_init_setup;
use brush_process::config::TrainStreamConfig;
use brush_process::message::TrainMessage;
use brush_process::{DataSource, create_process, message::ProcessMessage};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainExitCode {
    Success = 0,
    Error = 1,
    Cancelled = 2,
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
    fn new(kind: ProgressMessageKind) -> ProgressMessage {
        ProgressMessage {
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

struct BrushJobState {
    cancellation_requested: Arc<AtomicBool>,
    handle: Mutex<Option<JoinHandle<TrainExitCode>>>,
    result: Mutex<Option<TrainExitCode>>,
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

impl OwnedTrainOptions {
    /// # Safety
    ///
    /// `options` must either be null or point to a valid `TrainOptions` value. If
    /// `output_path` or `export_name` is not null, it must be a valid
    /// null-terminated C string.
    unsafe fn copy_from(options: *const TrainOptions) -> Option<OwnedTrainOptions> {
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

        Some(OwnedTrainOptions {
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

pub type ProgressCallback =
    extern "C" fn(progress_message: ProgressMessage, user_data: *mut c_void);

static SETUP: OnceCell<()> = OnceCell::const_new();

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
                progress_callback,
                user_data_address,
                worker_cancellation,
            )
        })
    else {
        return ptr::null_mut();
    };

    let state = Box::new(BrushJobState {
        cancellation_requested,
        handle: Mutex::new(Some(handle)),
        result: Mutex::new(None),
    });
    Box::into_raw(state).cast::<BrushJob>()
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

fn job_state<'a>(job: *mut BrushJob) -> Option<&'a BrushJobState> {
    if job.is_null() {
        return None;
    }
    // SAFETY: Callers pass handles returned by brush_train_start.
    Some(unsafe { &*job.cast::<BrushJobState>() })
}

fn wait_for_state(state: &BrushJobState) -> TrainExitCode {
    if let Some(result) = *state
        .result
        .lock()
        .expect("Brush job result mutex poisoned")
    {
        return result;
    }

    let handle = state
        .handle
        .lock()
        .expect("Brush job handle mutex poisoned")
        .take();
    let result = match handle {
        Some(handle) => handle.join().unwrap_or(TrainExitCode::Error),
        None => TrainExitCode::Error,
    };

    *state
        .result
        .lock()
        .expect("Brush job result mutex poisoned") = Some(result);
    result
}

fn run_training_job(
    dataset_path: String,
    train_options: OwnedTrainOptions,
    progress_callback: ProgressCallback,
    user_data_address: usize,
    cancellation_requested: Arc<AtomicBool>,
) -> TrainExitCode {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let source = DataSource::Path(dataset_path);
        let process_args = train_options.into_train_stream_config();
        let mut process = create_process(source, async move |_| Some(process_args));
        let user_data = user_data_address as *mut c_void;

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
                        return TrainExitCode::Cancelled;
                    }
                    match message_result {
                        Ok(message) => {
                            emit_progress_message(message, progress_callback, user_data);
                        }
                        Err(_) => {
                            return TrainExitCode::Error;
                        }
                    }
                    if cancellation_requested.load(Ordering::SeqCst) {
                        return TrainExitCode::Cancelled;
                    }
                }

                if cancellation_requested.load(Ordering::SeqCst) {
                    TrainExitCode::Cancelled
                } else {
                    TrainExitCode::Success
                }
            })
    }));

    result.unwrap_or(TrainExitCode::Error)
}

fn emit_progress_message(
    message: ProcessMessage,
    progress_callback: ProgressCallback,
    user_data: *mut c_void,
) {
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
