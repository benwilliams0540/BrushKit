#![cfg(not(target_family = "wasm"))]

use std::ffi::{CStr, CString, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use brush_c::{
    ProgressMessage, ProgressMessageKind, TrainExitCode, TrainOptions, brush_job_cancel,
    brush_job_release, brush_job_wait, brush_train_start, train_and_save,
};

#[repr(C)]
struct CallbackState {
    call_count: AtomicUsize,
    training_count: AtomicUsize,
    checkpoint_count: AtomicUsize,
    checkpoint_iter: AtomicUsize,
    finished_called: AtomicBool,
    checkpoint_path: Mutex<Option<String>>,
}

impl CallbackState {
    fn new() -> CallbackState {
        CallbackState {
            call_count: AtomicUsize::new(0),
            training_count: AtomicUsize::new(0),
            checkpoint_count: AtomicUsize::new(0),
            checkpoint_iter: AtomicUsize::new(0),
            finished_called: AtomicBool::new(false),
            checkpoint_path: Mutex::new(None),
        }
    }
}

extern "C" fn test_progress_callback(process_message: ProgressMessage, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: user_data is a pointer to a CallbackState struct.
    let state = unsafe { (user_data as *const CallbackState).as_ref().unwrap() };
    state.call_count.fetch_add(1, Ordering::SeqCst);

    match process_message.kind {
        ProgressMessageKind::NewProcess => {
            println!("FFI Test: Training starting...");
        }
        ProgressMessageKind::Training => {
            state.training_count.fetch_add(1, Ordering::SeqCst);
            println!("FFI Test: Training iteration: {}", process_message.iter);
        }
        ProgressMessageKind::CheckpointExported => {
            state.checkpoint_count.fetch_add(1, Ordering::SeqCst);
            state
                .checkpoint_iter
                .store(process_message.iter as usize, Ordering::SeqCst);
            if !process_message.path.is_null() {
                // SAFETY: Checkpoint path pointers are valid for the duration of this callback.
                let path = unsafe { CStr::from_ptr(process_message.path) }
                    .to_string_lossy()
                    .into_owned();
                *state.checkpoint_path.lock().unwrap() = Some(path);
            }
        }
        ProgressMessageKind::DoneTraining => {
            println!("FFI Test: Training finished!");
            state.finished_called.store(true, Ordering::SeqCst);
        }
    }
}

fn test_dataset_path() -> CString {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dataset_path = Path::new(manifest_dir)
        .join("tests")
        .join("data")
        .join("test_dataset");
    CString::new(dataset_path.to_str().unwrap()).unwrap()
}

fn train_options(
    output_path: &CString,
    export_name: &CString,
    total_train_steps: u32,
) -> TrainOptions {
    TrainOptions {
        total_train_steps,
        refine_every: 5,
        max_resolution: 50,
        max_splats: 1000,
        export_every: total_train_steps,
        output_path: output_path.as_ptr(),
        export_name: export_name.as_ptr(),
    }
}

fn export_name_template() -> CString {
    CString::new("component-0_{iter}.ply").unwrap()
}

fn output_files(output_path: &str) -> Vec<PathBuf> {
    fs::read_dir(output_path)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect()
}

#[test]
fn test_train_and_save_ffi_short() {
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();
    let export_name_cstr = export_name_template();
    let dataset_path_cstr = test_dataset_path();

    let mut callback_state = CallbackState::new();
    let options = train_options(&output_path_cstr, &export_name_cstr, 10);

    // SAFETY: paths are valid, user_data is valid for lifetime of callback_state.
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Success));
    assert!(callback_state.call_count.load(Ordering::SeqCst) > 2);
    assert!(callback_state.training_count.load(Ordering::SeqCst) > 0);
    assert!(callback_state.finished_called.load(Ordering::SeqCst));
    assert_eq!(callback_state.checkpoint_count.load(Ordering::SeqCst), 1);
    assert_eq!(callback_state.checkpoint_iter.load(Ordering::SeqCst), 10);

    let checkpoint_path = callback_state
        .checkpoint_path
        .lock()
        .unwrap()
        .clone()
        .expect("missing checkpoint path");
    assert!(
        Path::new(&checkpoint_path).exists(),
        "checkpoint path does not exist: {checkpoint_path}"
    );
    assert!(
        checkpoint_path.ends_with("component-0_10.ply"),
        "checkpoint path did not use requested export name: {checkpoint_path}"
    );
    assert!(
        !output_files(output_path).is_empty(),
        "No output file was created"
    );
}

#[test]
fn test_brush_job_start_wait_release() {
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_job_test_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();
    let export_name_cstr = export_name_template();
    let dataset_path_cstr = test_dataset_path();

    let mut callback_state = CallbackState::new();
    let options = train_options(&output_path_cstr, &export_name_cstr, 10);

    // SAFETY: paths are valid, user_data is valid until after wait completes.
    let job = unsafe {
        brush_train_start(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };
    assert!(!job.is_null());

    // SAFETY: job came from brush_train_start and has not been released.
    let status = unsafe { brush_job_wait(job) };
    // SAFETY: job came from brush_train_start and is no longer needed.
    unsafe { brush_job_release(job) };

    assert!(matches!(status, TrainExitCode::Success));
    assert!(callback_state.finished_called.load(Ordering::SeqCst));
    assert_eq!(callback_state.checkpoint_count.load(Ordering::SeqCst), 1);
    assert!(
        !output_files(output_path).is_empty(),
        "No output file was created"
    );
}

#[test]
fn test_brush_job_cancel() {
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_job_cancel_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();
    let export_name_cstr = export_name_template();
    let dataset_path_cstr = test_dataset_path();

    let mut callback_state = CallbackState::new();
    let options = train_options(&output_path_cstr, &export_name_cstr, 500);

    // SAFETY: paths are valid, user_data is valid until after wait completes.
    let job = unsafe {
        brush_train_start(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };
    assert!(!job.is_null());

    // SAFETY: job came from brush_train_start and has not been released.
    assert!(unsafe { brush_job_cancel(job) });
    // SAFETY: job came from brush_train_start and has not been released.
    let status = unsafe { brush_job_wait(job) };
    // SAFETY: job came from brush_train_start and is no longer needed.
    unsafe { brush_job_release(job) };

    assert!(matches!(status, TrainExitCode::Cancelled));
}

#[test]
fn test_train_and_save_ffi_invalid_path() {
    let invalid_dataset_path = "/path/that/does/not/exist/and/should/fail";
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_invalid_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();
    let export_name_cstr = export_name_template();

    let dataset_path_cstr = CString::new(invalid_dataset_path).unwrap();
    let mut callback_state = CallbackState::new();
    let options = train_options(&output_path_cstr, &export_name_cstr, 10);

    // SAFETY: The paths are valid, and the callback state is alive for the duration of the call.
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Error));
}

#[test]
fn test_train_and_save_ffi_null_options() {
    let dataset_path_cstr = test_dataset_path();

    let mut callback_state = CallbackState::new();

    // SAFETY: The path is valid, and the callback state is alive for the duration of the call.
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            std::ptr::null(),
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Error));
}

#[test]
fn test_train_and_save_ffi_null_dataset() {
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_invalid_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();
    let export_name_cstr = export_name_template();

    let options = train_options(&output_path_cstr, &export_name_cstr, 10);

    // SAFETY: The paths are valid, and the callback state is null.
    let status_null_dataset = unsafe {
        train_and_save(
            std::ptr::null(),
            &options,
            test_progress_callback,
            std::ptr::null_mut(),
        )
    };

    assert!(matches!(status_null_dataset, TrainExitCode::Error));
}
