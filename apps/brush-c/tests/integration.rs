#![cfg(not(target_family = "wasm"))]

use std::ffi::{CStr, CString, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use brush_c::{
    BRUSH_ABI_VERSION_V2, BRUSH_ABI_VERSION_V3, BRUSH_ABI_VERSION_V4,
    BRUSH_CAPABILITY_INITIALIZER_AUDIT_V2, BRUSH_CAPABILITY_TERMINAL_COMPACTION_V2,
    BrushEventKindV2, BrushEventKindV4, BrushEventV2, BrushEventV4, BrushInitializerRouteV2,
    BrushNativeIdentityV2, ProgressMessage, ProgressMessageKind, TrainExitCode, TrainOptions,
    TrainOptionsV2, TrainOptionsV3, TrainOptionsV4, brush_get_abi_version,
    brush_get_build_provenance_v1, brush_get_native_identity_v2, brush_job_cancel,
    brush_job_release, brush_job_release_v2, brush_job_retain_v2, brush_job_wait,
    brush_job_wait_v2, brush_train_start, brush_train_start_v2, brush_train_start_v3,
    train_and_save, train_and_save_v4,
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
    fn new() -> Self {
        Self {
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

#[derive(Default)]
struct V2CallbackState {
    events: Mutex<Vec<(BrushEventV2, Option<String>)>>,
}

extern "C" fn test_v2_callback(event: BrushEventV2, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: tests keep the callback state alive through wait/release.
    let state = unsafe { &*user_data.cast::<V2CallbackState>() };
    let text = (!event.text.is_null()).then(|| {
        // SAFETY: V2 text is callback-scoped and copied before returning.
        unsafe { CStr::from_ptr(event.text) }
            .to_string_lossy()
            .into_owned()
    });
    state.events.lock().unwrap().push((event, text));
}

#[derive(Default)]
struct V4CallbackState {
    events: Mutex<Vec<(BrushEventV4, Option<String>)>>,
}

extern "C" fn test_v4_callback(event: BrushEventV4, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: tests keep the callback state alive through wait/release.
    let state = unsafe { &*user_data.cast::<V4CallbackState>() };
    let text = (!event.base.text.is_null()).then(|| {
        // SAFETY: V4 text is callback-scoped and copied before returning.
        unsafe { CStr::from_ptr(event.base.text) }
            .to_string_lossy()
            .into_owned()
    });
    state.events.lock().unwrap().push((event, text));
}

fn v2_options(
    output_path: &CString,
    export_name: &CString,
    initializer_path: &CString,
    instrumentation_level: u32,
) -> TrainOptionsV2 {
    TrainOptionsV2 {
        struct_size: std::mem::size_of::<TrainOptionsV2>() as u32,
        abi_version: BRUSH_ABI_VERSION_V2,
        seed: 0x0123_4567_89ab_cdef,
        render_mode: 1,
        sh_degree: 3,
        sh_policy: 0,
        total_train_steps: 10,
        refine_every: 200,
        max_resolution: 50,
        max_splats: 1000,
        export_every: 10,
        output_path: output_path.as_ptr(),
        export_name: export_name.as_ptr(),
        initializer_path: initializer_path.as_ptr(),
        initializer_required: 1,
        instrumentation_level,
    }
}

fn v3_options(
    output_path: &CString,
    export_name: &CString,
    initializer_path: &CString,
) -> TrainOptionsV3 {
    let v2 = v2_options(output_path, export_name, initializer_path, 1);
    TrainOptionsV3 {
        struct_size: std::mem::size_of::<TrainOptionsV3>() as u32,
        abi_version: BRUSH_ABI_VERSION_V3,
        seed: v2.seed,
        render_mode: v2.render_mode,
        sh_degree: v2.sh_degree,
        sh_policy: v2.sh_policy,
        total_train_steps: v2.total_train_steps,
        refine_every: v2.refine_every,
        max_resolution: v2.max_resolution,
        max_splats: v2.max_splats,
        export_every: v2.export_every,
        output_path: v2.output_path,
        export_name: v2.export_name,
        initializer_path: v2.initializer_path,
        initializer_required: v2.initializer_required,
        instrumentation_level: v2.instrumentation_level,
        progressive_resolution_start_percent: 50,
        progressive_resolution_switch_iteration: 5,
    }
}

fn v4_options(
    output_path: &CString,
    export_name: &CString,
    initializer_path: &CString,
) -> TrainOptionsV4 {
    let v2 = v2_options(output_path, export_name, initializer_path, 0);
    TrainOptionsV4 {
        struct_size: std::mem::size_of::<TrainOptionsV4>() as u32,
        abi_version: BRUSH_ABI_VERSION_V4,
        seed: v2.seed,
        render_mode: v2.render_mode,
        sh_degree: v2.sh_degree,
        sh_policy: v2.sh_policy,
        total_train_steps: v2.total_train_steps,
        refine_every: v2.refine_every,
        max_resolution: v2.max_resolution,
        max_splats: v2.max_splats,
        export_every: v2.export_every,
        output_path: v2.output_path,
        export_name: v2.export_name,
        initializer_path: v2.initializer_path,
        initializer_required: v2.initializer_required,
        instrumentation_level: v2.instrumentation_level,
        progressive_resolution_start_percent: 100,
        progressive_resolution_switch_iteration: 0,
        sh_schedule_mode: 0,
        initial_sh_degree: 3,
        sh_degree_step_interval: 0,
    }
}

#[test]
fn test_v2_abi_identity_is_explicit() {
    assert_eq!(brush_get_abi_version(), 2);
    let mut identity = BrushNativeIdentityV2 {
        struct_size: 0,
        abi_version: 0,
        build_revision: std::ptr::null(),
        crate_version: std::ptr::null(),
        graphics_backend: std::ptr::null(),
        adapter_name: std::ptr::null(),
        adapter_identity_available: true,
    };
    // SAFETY: identity is exact writable V2 storage.
    assert!(unsafe {
        brush_get_native_identity_v2(
            &mut identity,
            std::mem::size_of::<BrushNativeIdentityV2>() as u32,
        )
    });
    assert_eq!(identity.abi_version, 2);
    assert!(!identity.build_revision.is_null());
    assert!(!identity.crate_version.is_null());
    assert!(!identity.graphics_backend.is_null());
    assert!(!identity.adapter_name.is_null());
    assert!(!identity.adapter_identity_available);

    let provenance = brush_get_build_provenance_v1();
    assert!(!provenance.is_null());
    // SAFETY: provenance is a process-lifetime, null-terminated static string.
    let provenance = unsafe { CStr::from_ptr(provenance) }.to_str().unwrap();
    assert!(provenance.starts_with("format=brushkit-build-provenance-v1;"));
    assert!(provenance.contains(";callable_abi=2;additive_train_abi=4;"));
    assert!(provenance.contains(concat!(";crate_version=", env!("CARGO_PKG_VERSION"), ";")));
    assert!(provenance.contains(";cubecl_gpu_profile=off;"));
    assert!(provenance.contains(";rustc=rustc "));
    assert!(provenance.contains(";target="));
    if cfg!(feature = "image-loss-bwd-tile-16") {
        assert!(provenance.contains(";apple_features=image-loss-bwd-tile-16;"));
    } else {
        assert!(provenance.contains(";apple_features=none;"));
    }
}

#[test]
fn test_v2_rejects_wrong_abi_synchronously_with_detail() {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();
    let mut options = v2_options(&output_path, &export_name, &initializer, 1);
    options.abi_version = 99;
    let mut state = V2CallbackState::default();
    // SAFETY: pointers are valid for this synchronous validation call.
    let job = unsafe {
        brush_train_start_v2(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert!(job.is_null());
    let events = state.events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0.kind, BrushEventKindV2::Terminal);
    assert!(events[0].1.as_deref().unwrap().contains("ABI version"));
}

#[test]
fn test_v3_progressive_resolution_job_preserves_v2_event_and_job_contracts() {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_v3_progressive_")
        .tempdir()
        .unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();
    let options = v3_options(&output_path, &export_name, &initializer);
    let mut state = V2CallbackState::default();

    // SAFETY: callback state and all option strings outlive wait and release.
    let job = unsafe {
        brush_train_start_v3(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert!(!job.is_null());
    // SAFETY: V3 returns the retained V2 job ownership contract.
    let status = unsafe { brush_job_wait_v2(job) };
    // SAFETY: matching release for start.
    unsafe { brush_job_release_v2(job) };
    assert_eq!(status, TrainExitCode::Success);

    let events = state.events.lock().unwrap();
    assert_eq!(
        events.first().unwrap().0.kind,
        BrushEventKindV2::Capabilities
    );
    assert_eq!(
        events.get(1).unwrap().0.kind,
        BrushEventKindV2::Configuration
    );
    assert_eq!(events.last().unwrap().0.kind, BrushEventKindV2::Terminal);
    assert!(temp_dir.path().join("component-0_10.ply").is_file());
}

#[test]
fn test_v4_disabled_uses_the_existing_fixed_sh3_path() {
    let dataset_path = test_dataset_path();
    let v3_dir = tempfile::Builder::new()
        .prefix("ffi_v3_fixed_sh3_")
        .tempdir()
        .unwrap();
    let v4_dir = tempfile::Builder::new()
        .prefix("ffi_v4_disabled_sh3_")
        .tempdir()
        .unwrap();
    let v3_path = CString::new(v3_dir.path().to_str().unwrap()).unwrap();
    let v4_path = CString::new(v4_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();

    let mut v3 = v3_options(&v3_path, &export_name, &initializer);
    v3.total_train_steps = 1;
    v3.export_every = 1;
    v3.instrumentation_level = 0;
    v3.progressive_resolution_start_percent = 100;
    v3.progressive_resolution_switch_iteration = 0;
    let mut v3_state = V2CallbackState::default();
    // SAFETY: callback state and strings outlive wait and release.
    let v3_job = unsafe {
        brush_train_start_v3(
            dataset_path.as_ptr(),
            &v3,
            test_v2_callback,
            std::ptr::from_mut(&mut v3_state).cast(),
        )
    };
    assert!(!v3_job.is_null());
    // SAFETY: V3 uses the retained V2 ownership contract.
    assert_eq!(unsafe { brush_job_wait_v2(v3_job) }, TrainExitCode::Success);
    // SAFETY: matching release for start.
    unsafe { brush_job_release_v2(v3_job) };

    let mut v4 = v4_options(&v4_path, &export_name, &initializer);
    v4.total_train_steps = 1;
    v4.export_every = 1;
    let mut v4_state = V4CallbackState::default();
    // SAFETY: callback state and strings outlive the blocking call.
    let status = unsafe {
        train_and_save_v4(
            dataset_path.as_ptr(),
            &v4,
            test_v4_callback,
            std::ptr::from_mut(&mut v4_state).cast(),
        )
    };
    assert_eq!(status, TrainExitCode::Success);

    let v3_bytes = fs::read(v3_dir.path().join("component-0_1.ply")).unwrap();
    let v4_bytes = fs::read(v4_dir.path().join("component-0_1.ply")).unwrap();
    assert_eq!(
        v4_bytes, v3_bytes,
        "disabled V4 must preserve the exact fixed-SH3 output path"
    );
    let events = v4_state.events.lock().unwrap();
    assert_eq!(
        events.first().unwrap().0.kind,
        BrushEventKindV4::Capabilities
    );
    assert_eq!(
        events.get(1).unwrap().0.kind,
        BrushEventKindV4::Configuration
    );
    assert_eq!(events.last().unwrap().0.kind, BrushEventKindV4::Terminal);
    assert!(
        !events
            .iter()
            .any(|event| event.0.kind == BrushEventKindV4::ActiveSHDegreeChanged)
    );
    assert!(
        events
            .iter()
            .all(|event| { event.0.maximum_sh_degree == 3 && event.0.active_sh_degree == 3 })
    );
}

#[test]
fn test_v4_predeclared_schedule_emits_exact_transitions_and_sh3_checkpoints() {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_v4_progressive_sh_")
        .tempdir()
        .unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();
    let mut options = v4_options(&output_path, &export_name, &initializer);
    options.total_train_steps = 600;
    options.refine_every = 200;
    options.export_every = 150;
    options.sh_schedule_mode = 1;
    options.initial_sh_degree = 0;
    options.sh_degree_step_interval = 150;
    let mut state = V4CallbackState::default();

    // SAFETY: callback state and all strings outlive the blocking call.
    let status = unsafe {
        train_and_save_v4(
            dataset_path.as_ptr(),
            &options,
            test_v4_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert_eq!(status, TrainExitCode::Success);

    let events = state.events.lock().unwrap();
    let transitions: Vec<_> = events
        .iter()
        .filter(|event| event.0.kind == BrushEventKindV4::ActiveSHDegreeChanged)
        .map(|event| {
            (
                event.0.zero_based_iteration,
                event.0.active_sh_degree,
                event.0.schedule_identity,
            )
        })
        .collect();
    assert_eq!(transitions.len(), 4);
    assert_eq!(
        transitions
            .iter()
            .map(|(iteration, degree, _)| (*iteration, *degree))
            .collect::<Vec<_>>(),
        vec![(0, 0), (150, 1), (300, 2), (450, 3)]
    );
    let schedule_identity = transitions[0].2;
    assert_ne!(schedule_identity, 0);
    assert!(
        transitions
            .iter()
            .all(|transition| transition.2 == schedule_identity)
    );
    for (iteration, degree, _) in &transitions {
        let transition_index = events
            .iter()
            .position(|event| {
                event.0.kind == BrushEventKindV4::ActiveSHDegreeChanged
                    && event.0.zero_based_iteration == *iteration
            })
            .unwrap();
        let step_index = events
            .iter()
            .position(|event| {
                event.0.kind == BrushEventKindV4::Step && event.0.base.iteration > *iteration
            })
            .unwrap();
        assert!(transition_index < step_index);
        assert_eq!(events[step_index].0.active_sh_degree, *degree);
    }
    let terminal = events.last().unwrap().0;
    assert_eq!(terminal.kind, BrushEventKindV4::Terminal);
    assert_eq!(terminal.maximum_sh_degree, 3);
    assert_eq!(terminal.active_sh_degree, 3);
    assert_eq!(terminal.schedule_identity, schedule_identity);
    drop(events);

    for (completed, active) in [(150, 0), (300, 1), (450, 2), (600, 3)] {
        let bytes = fs::read(temp_dir.path().join(format!("component-0_{completed}.ply"))).unwrap();
        let ply = String::from_utf8_lossy(&bytes);
        assert_eq!(ply.matches("property float f_rest_").count(), 45);
        assert!(ply.contains("BrushKitProgressiveSH: version=1;"));
        assert!(ply.contains(&format!(";completed={completed};active={active}")));
        assert!(ply.contains(&format!(";identity={schedule_identity:016x};")));
    }
}

#[test]
fn test_v2_strong_initializer_and_phase0_events() {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_v2_strong_")
        .tempdir()
        .unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();
    let options = v2_options(&output_path, &export_name, &initializer, 1);
    let mut state = V2CallbackState::default();

    // SAFETY: all callback-owned data remains alive through wait and release.
    let job = unsafe {
        brush_train_start_v2(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert!(!job.is_null());
    // SAFETY: retain produces a separately releasable handle.
    let retained = unsafe { brush_job_retain_v2(job) };
    assert!(!retained.is_null());
    let retained_address = retained as usize;
    let waiter = std::thread::spawn(move || {
        // SAFETY: retained handle remains owned by this thread until release.
        let retained = retained_address as *mut brush_c::BrushJobV2;
        // SAFETY: retained is the live handle transferred to this thread.
        let status = unsafe { brush_job_wait_v2(retained) };
        // SAFETY: matching release for retain.
        unsafe { brush_job_release_v2(retained) };
        status
    });
    // SAFETY: original handle remains live.
    let status = unsafe { brush_job_wait_v2(job) };
    assert_eq!(waiter.join().unwrap(), TrainExitCode::Success);
    // SAFETY: matching release for start.
    unsafe { brush_job_release_v2(job) };
    assert_eq!(
        status,
        TrainExitCode::Success,
        "Fast Preview evidence run failed"
    );

    let events = state.events.lock().unwrap();
    let kinds: Vec<_> = events.iter().map(|event| event.0.kind).collect();
    assert_eq!(kinds.first(), Some(&BrushEventKindV2::Capabilities));
    assert_eq!(kinds.get(1), Some(&BrushEventKindV2::Configuration));
    assert_eq!(kinds.last(), Some(&BrushEventKindV2::Terminal));
    assert!(events[0].0.capability_flags & BRUSH_CAPABILITY_INITIALIZER_AUDIT_V2 != 0);
    assert!(events[0].0.capability_flags & BRUSH_CAPABILITY_TERMINAL_COMPACTION_V2 != 0);

    let configuration = events
        .iter()
        .find(|event| event.0.kind == BrushEventKindV2::Configuration)
        .unwrap()
        .0;
    assert_eq!(configuration.seed, options.seed);
    assert_eq!(configuration.render_mode, 1);
    assert_eq!(configuration.iteration, 10);
    assert_eq!(configuration.refine_every, 200);
    assert_eq!(configuration.max_resolution, 50);
    assert_eq!(configuration.max_splats, 1000);
    assert_eq!(configuration.export_every, 10);
    assert_eq!(configuration.configured_sh_degree, 3);

    let initializer_event = events
        .iter()
        .find(|event| event.0.kind == BrushEventKindV2::Initializer)
        .unwrap();
    assert_eq!(
        initializer_event.0.initializer_route,
        BrushInitializerRouteV2::ExplicitStrong
    );
    assert_eq!(initializer_event.0.primitive_count, 3);
    assert_eq!(initializer_event.0.fields_consumed, 0b1_1111);
    assert_eq!(initializer_event.0.rejected_count, 0);
    assert_eq!(initializer_event.0.supplied_sh_degree, 0);
    assert_eq!(initializer_event.0.configured_sh_degree, 3);
    assert_eq!(initializer_event.1.as_deref(), Some("strong-init.ply"));

    let step_iterations: Vec<_> = events
        .iter()
        .filter(|event| event.0.kind == BrushEventKindV2::Step)
        .map(|event| event.0.iteration)
        .collect();
    assert!(step_iterations.contains(&1));
    assert!(step_iterations.contains(&10));
    assert!(
        events
            .iter()
            .filter(|event| { event.0.kind == BrushEventKindV2::Step })
            .all(|event| {
                event.1.as_deref().is_some_and(|text| {
                    text.starts_with("optimizer_substages transforms_ns=")
                        && text.contains(" sh_coeffs_ns=")
                        && text.contains(" opacity_ns=")
                        && text.contains("; render_host before_count_readback_ns=")
                        && text.contains(" count_readback_ns=")
                        && text.contains(" after_count_readback_ns=")
                })
            })
    );
    assert!(events.iter().any(|event| {
        event.0.kind == BrushEventKindV2::CheckpointExportStarted && event.0.iteration == 10
    }));
    assert!(events.iter().any(|event| {
        event.0.kind == BrushEventKindV2::CheckpointExported && event.0.iteration == 10
    }));
    let terminal_compaction = events
        .iter()
        .find(|event| event.0.kind == BrushEventKindV2::TerminalCompaction)
        .expect("missing terminal export validation telemetry");
    assert_eq!(terminal_compaction.0.iteration, 10);
    assert_eq!(terminal_compaction.0.primitive_count, 3);
    assert_eq!(terminal_compaction.0.pruned_non_finite_count, 0);
    assert!(
        terminal_compaction
            .1
            .as_deref()
            .is_some_and(|text| text.contains("terminal_export_validation source=3 exported=3"))
    );
    let terminal = events.last().unwrap().0;
    assert_eq!(terminal.initial_primitive_count, 3);
    assert_eq!(terminal.final_primitive_count, 3);
}

#[test]
fn test_v2_phase0_host_telemetry_is_opt_in() {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_v2_optimizer_telemetry_off_")
        .tempdir()
        .unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();
    let mut options = v2_options(&output_path, &export_name, &initializer, 0);
    options.total_train_steps = 1;
    options.export_every = 1;
    let mut state = V2CallbackState::default();

    // SAFETY: callback state and all option strings outlive the blocking call.
    let status = unsafe {
        brush_c::train_and_save_v2(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert_eq!(status, TrainExitCode::Success);

    let events = state.events.lock().unwrap();
    let steps: Vec<_> = events
        .iter()
        .filter(|event| event.0.kind == BrushEventKindV2::Step)
        .collect();
    assert!(!steps.is_empty());
    assert!(steps.iter().all(|event| event.1.is_none()));
}

#[test]
fn test_v2_required_strong_initializer_cannot_fall_back() {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    // The legacy fixture is positions/color only and lacks rotation/scale.
    let initializer = CString::new("init.ply").unwrap();
    let options = v2_options(&output_path, &export_name, &initializer, 1);
    let mut state = V2CallbackState::default();
    // SAFETY: callback state remains alive through completion.
    let status = unsafe {
        brush_c::train_and_save_v2(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert_eq!(status, TrainExitCode::Error);
    let events = state.events.lock().unwrap();
    let terminal = events.last().unwrap();
    assert_eq!(terminal.0.kind, BrushEventKindV2::Terminal);
    assert!(terminal.1.as_deref().unwrap().contains("missing rotations"));
    assert!(output_files(temp_dir.path().to_str().unwrap()).is_empty());
}

#[test]
fn test_v2_refinement_counts_are_coherent() {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_v2_refine_")
        .tempdir()
        .unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();
    let mut options = v2_options(&output_path, &export_name, &initializer, 1);
    options.refine_every = 5;
    let mut state = V2CallbackState::default();

    // SAFETY: callback state and all option strings outlive the blocking call.
    let status = unsafe {
        brush_c::train_and_save_v2(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert_eq!(
        status,
        TrainExitCode::Success,
        "Fast Preview evidence run failed"
    );

    let events = state.events.lock().unwrap();
    let refinements: Vec<_> = events
        .iter()
        .filter(|event| event.0.kind == BrushEventKindV2::Refinement)
        .map(|event| event.0)
        .collect();
    assert_eq!(refinements.len(), 1);
    let refinement = refinements[0];
    assert_eq!(refinement.split_count, refinement.added_count);
    assert_eq!(refinement.clone_count, 0);
    assert!(refinement.split_oversized_count <= refinement.split_count);
    assert!(refinement.split_high_gradient_count <= refinement.split_count);
    assert!(refinement.pruned_non_finite_count <= refinement.pruned_count);
    assert_eq!(
        refinement.net_growth,
        i64::from(refinement.added_count) - i64::from(refinement.pruned_count)
    );

    let terminal = events.last().unwrap().0;
    let cumulative_net_growth: i64 = refinements.iter().map(|event| event.net_growth).sum();
    assert_eq!(
        i64::from(terminal.final_primitive_count),
        i64::from(terminal.initial_primitive_count) + cumulative_net_growth
    );
}

struct Phase0RunEvidence {
    elapsed: Duration,
    output: Vec<u8>,
    final_count: u32,
}

fn run_phase0_evidence(instrumentation_level: u32, total_steps: u32) -> Phase0RunEvidence {
    let dataset_path = test_dataset_path();
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_v2_phase0_evidence_")
        .tempdir()
        .unwrap();
    let output_path = CString::new(temp_dir.path().to_str().unwrap()).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("strong-init.ply").unwrap();
    let mut options = v2_options(
        &output_path,
        &export_name,
        &initializer,
        instrumentation_level,
    );
    options.total_train_steps = total_steps;
    options.refine_every = 200;
    options.max_resolution = 1080;
    options.max_splats = 500_000;
    options.export_every = total_steps;
    let mut state = V2CallbackState::default();

    let started = Instant::now();
    // SAFETY: callback state and all option strings outlive the blocking call.
    let status = unsafe {
        brush_c::train_and_save_v2(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    let elapsed = started.elapsed();
    assert_eq!(
        status,
        TrainExitCode::Success,
        "Fast Preview evidence run failed"
    );

    let output = fs::read(
        temp_dir
            .path()
            .join(format!("component-0_{total_steps}.ply")),
    )
    .unwrap();
    let events = state.events.lock().unwrap();
    let final_count = events
        .last()
        .filter(|event| event.0.kind == BrushEventKindV2::Terminal)
        .map_or(0, |event| event.0.final_primitive_count);
    Phase0RunEvidence {
        elapsed,
        output,
        final_count,
    }
}

fn median_duration(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn assert_numeric_outputs_equivalent(reference: &[u8], candidate: &[u8]) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (reference, candidate) = runtime.block_on(async {
        let reference = brush_serde::load_splat_from_ply(reference, None)
            .await
            .unwrap()
            .data;
        let candidate = brush_serde::load_splat_from_ply(candidate, None)
            .await
            .unwrap()
            .data;
        (reference, candidate)
    });
    assert_eq!(
        reference.num_splats(),
        candidate.num_splats(),
        "exported Gaussian count changed"
    );

    let compare = |name: &str, reference: &[f32], candidate: &[f32]| {
        assert_eq!(reference.len(), candidate.len(), "{name} length changed");
        let max_abs = reference
            .iter()
            .zip(candidate)
            .map(|(reference, candidate)| (reference - candidate).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-3,
            "{name} exceeded frozen absolute tolerance: {max_abs}"
        );
    };
    compare("means", &reference.means, &candidate.means);
    compare(
        "rotations",
        reference.rotations.as_deref().unwrap(),
        candidate.rotations.as_deref().unwrap(),
    );
    compare(
        "log_scales",
        reference.log_scales.as_deref().unwrap(),
        candidate.log_scales.as_deref().unwrap(),
    );
    compare(
        "SH coefficients",
        reference.sh_coeffs.as_deref().unwrap(),
        candidate.sh_coeffs.as_deref().unwrap(),
    );
    compare(
        "opacity",
        reference.raw_opacities.as_deref().unwrap(),
        candidate.raw_opacities.as_deref().unwrap(),
    );
}

#[test]
#[ignore = "manual external Variant-D trainer diagnosis"]
fn diagnose_external_variant_d_fixture() {
    let dataset_path = std::env::var("BRUSHKIT_VARIANT_D_DATASET")
        .expect("BRUSHKIT_VARIANT_D_DATASET must name the read-only trainer leaf");
    let output_path = std::env::var("BRUSHKIT_VARIANT_D_OUTPUT")
        .expect("BRUSHKIT_VARIANT_D_OUTPUT must name a dedicated derived output directory");
    fs::create_dir_all(&output_path).unwrap();

    let dataset_path = CString::new(dataset_path).unwrap();
    let output_path = CString::new(output_path).unwrap();
    let export_name = export_name_template();
    let initializer = CString::new("initializers/variant-d.ply").unwrap();
    let mut options = v3_options(&output_path, &export_name, &initializer);
    options.seed = 0;
    options.total_train_steps = 300;
    options.refine_every = 200;
    options.max_resolution = 1080;
    options.max_splats = 500_000;
    options.export_every = 300;
    options.instrumentation_level = 1;
    options.progressive_resolution_start_percent = 50;
    options.progressive_resolution_switch_iteration = 200;
    let mut state = V2CallbackState::default();

    // SAFETY: callback state and all option strings outlive wait and release.
    let job = unsafe {
        brush_train_start_v3(
            dataset_path.as_ptr(),
            &options,
            test_v2_callback,
            std::ptr::from_mut(&mut state).cast(),
        )
    };
    assert!(!job.is_null(), "Variant-D diagnostic job failed to start");
    // SAFETY: job is a live V3 handle using the V2 lifecycle contract.
    let status = unsafe { brush_job_wait_v2(job) };
    // SAFETY: the completed handle has not previously been released.
    unsafe { brush_job_release_v2(job) };
    assert_eq!(status, TrainExitCode::Success);

    let events = state.events.lock().unwrap();
    let initializer = events
        .iter()
        .find(|event| event.0.kind == BrushEventKindV2::Initializer)
        .expect("missing initializer event")
        .0;
    let refinement_net: i64 = events
        .iter()
        .filter(|event| event.0.kind == BrushEventKindV2::Refinement)
        .map(|event| event.0.net_growth)
        .sum();
    let terminal_compaction = events
        .iter()
        .find(|event| event.0.kind == BrushEventKindV2::TerminalCompaction)
        .expect("missing terminal compaction event")
        .0;
    let terminal = events
        .last()
        .filter(|event| event.0.kind == BrushEventKindV2::Terminal)
        .expect("missing terminal event")
        .0;
    println!(
        "BRUSHKIT_VARIANT_D_ACCOUNTING initial={} refinement_net={} terminal_pruned_non_finite={} exported={} terminal_final={}",
        initializer.primitive_count,
        refinement_net,
        terminal_compaction.pruned_non_finite_count,
        terminal_compaction.primitive_count,
        terminal.final_primitive_count,
    );
    let expected_exported =
        i64::from(initializer.primitive_count) + refinement_net + terminal_compaction.net_growth;
    assert_eq!(
        expected_exported,
        i64::from(terminal_compaction.primitive_count),
        "initial + refinement net + terminal compaction net must equal exported count"
    );
    assert_eq!(
        terminal.final_primitive_count, terminal_compaction.primitive_count,
        "terminal final count must equal the exhaustively validated export population"
    );
    assert!(
        Path::new(output_path.to_str().unwrap())
            .join("component-0_300.ply")
            .is_file()
    );
}

#[test]
#[ignore = "manual Fast Preview instrumentation overhead and determinism evidence"]
fn validate_v2_phase0_overhead_and_determinism() {
    // Warm the shared Metal/Burn setup and shader cache outside the samples.
    let _ = run_phase0_evidence(0, 10);

    let mut uninstrumented = Vec::new();
    let mut instrumented = Vec::new();
    for _ in 0..3 {
        uninstrumented.push(run_phase0_evidence(0, 300));
        instrumented.push(run_phase0_evidence(1, 300));
    }

    let baseline_is_byte_deterministic = uninstrumented
        .iter()
        .skip(1)
        .all(|run| run.output == uninstrumented[0].output);
    for run in &instrumented {
        assert_numeric_outputs_equivalent(&uninstrumented[0].output, &run.output);
        if baseline_is_byte_deterministic {
            assert_eq!(
                run.output, uninstrumented[0].output,
                "instrumentation changed bytes after baseline byte determinism was proven"
            );
        }
    }

    let expected_count = instrumented[0].final_count;
    assert!(expected_count > 0);
    assert!(
        instrumented
            .iter()
            .all(|run| run.final_count == expected_count)
    );

    let mut uninstrumented_times: Vec<_> = uninstrumented.iter().map(|run| run.elapsed).collect();
    let mut instrumented_times: Vec<_> = instrumented.iter().map(|run| run.elapsed).collect();
    let baseline_median = median_duration(&mut uninstrumented_times);
    let instrumented_median = median_duration(&mut instrumented_times);
    let overhead = instrumented_median.as_secs_f64() / baseline_median.as_secs_f64() - 1.0;
    println!(
        "phase0 evidence: baseline_byte_deterministic={baseline_is_byte_deterministic} baseline_median={baseline_median:?} instrumented_median={instrumented_median:?} overhead={:.3}% final_count={expected_count}",
        overhead * 100.0
    );
    assert!(
        overhead < 0.05,
        "Phase 0 instrumentation overhead {:.3}% exceeded 5%",
        overhead * 100.0
    );
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
