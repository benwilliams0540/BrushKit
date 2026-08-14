# BrushKit callable ABI V2: Splats integration handoff

This is a local-only handoff for a later, separately authorized release and
consumer-integration task. BrushKit was integrated against the read-only Splats
authority at commit `8b3145bcb20a1de3b31ee1b9d0caf3c044241d71`. No Splats,
ColmapKit, or Memo files were changed.

## Publication gate

The production `Package.swift` and `Sources/BrushKit/BrushKit.swift` still expose
the remotely resolvable `brushkit-ffi-v0.2.1` contract. Do not add production
Swift calls to V2 until an immutable three-slice asset is available.

In the later authorized release/pin task, perform these as one ordered change:

1. Rebuild the three slices from the chosen final BrushKit source revision.
2. Reserve and publish the immutable asset as `brushkit-ffi-v0.3.0`; do not
   reuse or replace that tag or asset after publication.
3. Download the published archive, recompute its SHA-256 and SwiftPM checksum,
   and compare both with the uploaded bytes.
4. Atomically update BrushKit's binary-target URL/checksum and add the
   production Swift V2 facade. Until both are available together, keep the
   v0.2.1 URL/checksum and existing facade unchanged.
5. Pin Splats to that immutable BrushKit revision and run its package, Simulator,
   and later coordinated physical-device gates separately.

The local candidate built from native source commit
`182f34cdf77a81d1b64c7b87ef7d304d40dbe3fc` is:

- `artifacts/BrushKitFFI.xcframework`
- `artifacts/BrushKitFFI.xcframework.zip`
- SHA-256 and SwiftPM checksum:
  `1278877be34eea61cf0e99af5f8fff74a7d98a4a2222ea5f641ebe30fa47da45`

These local files are validation evidence, not published dependencies.

## Splats-owned request changes

At the authority commit, `SplatTrainingBenchmarkConfiguration.randomSeed` is
recorded as evidence but does not reach `SplatNativeTrainingRequest` or native
BrushKit. Thread it into the request/configuration without changing the frozen
Fast Preview cost settings. Add an explicit initializer relative path and a
required-strong flag; do not infer Variant D from a filename.

Map the expanded Splats request to `TrainOptionsV2` exactly:

| V2 field | Splats source / fixed value |
| --- | --- |
| `struct_size` | `MemoryLayout<TrainOptionsV2>.size` |
| `abi_version` | `BRUSH_ABI_VERSION_V2` (`2`) |
| `seed` | `SplatTrainingBenchmarkConfiguration.randomSeed` |
| `render_mode` | `renderMode == "mip" ? BRUSH_RENDER_MODE_MIP_V2 : BRUSH_RENDER_MODE_DEFAULT_V2` after rejecting unknown strings |
| `sh_degree` | Explicit training degree; use `3` for the frozen Phase 0 workload |
| `sh_policy` | `BRUSH_SH_POLICY_PRESERVE_AND_ZERO_PAD_V2` (`0`) |
| `total_train_steps` | `configuration.iterations` after checked `UInt32` conversion |
| `refine_every` | `configuration.effectiveRefineEvery` after checked conversion |
| `max_resolution` | `configuration.maxResolution` after checked conversion |
| `max_splats` | `configuration.maxSplats` after checked conversion |
| `export_every` | `configuration.exportEvery` after checked conversion |
| `output_path` | `request.exportDirectoryURL.path` |
| `export_name` | `configuration.exportNameTemplate` |
| `initializer_path` | Dataset-root-relative staged `.ply` path; Variant D must supply it |
| `initializer_required` | `1` only for the explicit required-strong route, otherwise `0` |
| `instrumentation_level` | `.phase0Full` to `BRUSH_INSTRUMENTATION_PHASE0_V2` (`1`); normal/legacy runs to disabled (`0`) |

Do not reuse the existing `splatBrushKitUInt32` clamping helper for the new V2
facade. Reject zero, negative, or overflowing cost settings before crossing the
ABI so persisted evidence describes the values actually used.

## Strong initializer staging

BrushKit consumes a full Gaussian PLY; it does not manufacture Splats' proposed
dense surface-aware RGB prior. The later Splats lane must own or call the
producer, validate its provenance, and stage the result under
`request.trainerLeafURL`. Extend `SplatNativeTrainerLeafStager` so a rebuild of
the trainer leaf cannot discard that file. Pass only a relative path such as
`initializers/variant-d.ply`; absolute paths and traversal are rejected.

Required-strong validation accepts degree-0 DC/RGB as well as higher SH bands.
Every primitive must provide finite:

- `x`, `y`, `z` means;
- `rot_0...rot_3` rotation;
- anisotropic `scale_0...scale_2` log scales;
- `opacity` in Brush's raw-opacity representation; and
- `f_dc_0...f_dc_2`, plus complete higher-band coefficients when supplied.

The primitive count must be nonzero and no greater than `max_splats`. Supplied
coefficients are preserved. Missing higher bands are zero-padded only up to the
explicit configured degree. A coefficient layout incompatible with a complete
degree, or a supplied degree above the configured degree, is an error; V2 never
silently truncates it. Missing, malformed, non-finite, over-budget, or incomplete
required input terminates with `BrushErrorCodeV2_Initializer`; it never falls
back to sparse initialization. Accept the route only after an initializer event
reports `BrushInitializerRouteV2_ExplicitStrong`, all five field bits, zero
rejections, the expected primitive count/path, and the expected supplied and
configured SH degrees.

## Swift facade ownership

The V2 facade should check `brush_get_abi_version() == 2` and query
`brush_get_native_identity_v2` before starting. Copy every callback-scoped
`text` C string synchronously inside the callback.

Each independently executing waiter or cancellation owner must receive its own
handle from `brush_job_retain_v2`. Each retained handle is released exactly
once with `brush_job_release_v2`. Keep the callback context alive until the
terminal callback and all waits have finished; final release cancels and joins
an unfinished worker. Never unlock a store and then call through a borrowed raw
handle that another task can release. The local reference implementation is
`smoke/BrushKitV2Smoke/main.swift`.

## Event-to-evidence mapping

Preserve callback order and monotonic `timestamp_ns`; copy events into
Splats-owned `Sendable` values before dispatching:

- `Capabilities`: persist flags and the unavailable-reason `text`.
- `Configuration`: assert the seed and every cost-bearing setting match the
  benchmark configuration before accepting the run.
- `DatasetLoadStarted/Finished`: delimit data loading.
- `Initializer`: persist route, path, counts, field mask, rejected count, and SH
  degrees.
- `TrainerInitializationStarted/Finished`: delimit native trainer setup.
- `Step`: use iteration/timestamp/live count and the host-observed operation
  durations. First-step latency is step 1 timestamp minus trainer-init-finished.
- `Refinement`: persist iteration, before/after count, added/split subcounts,
  prune/non-finite prune counts, net growth, and duration. `clone_count` is zero
  because current Brush has no genuinely distinct clone operation.
- `CheckpointExportStarted/CheckpointExported`: delimit exports and continue
  using `SplatTrainingOutputValidator` on the completed artifact.
- `Terminal`: finish only after consuming the final count/status/error detail.

The capability event intentionally reports these limits rather than fabricating
values: bound Metal adapter identity is unavailable from the default Burn
setup; querying GPU allocation would stall the compute server; GPU command
timing would require added synchronization. Operation timings are host-observed
around existing work. CPU wait timing is available. Residual GPU scheduling and
floating-point nondeterminism may remain even though V2 controls every
host-owned stochastic path.

## Consumer acceptance and remaining dependencies

Before enabling production use, port the assertions in
`smoke/BrushKitV2Smoke/main.swift`, including retained concurrent waits, exact
configuration, explicit-strong audit, step/export order, terminal status, and
final PLY validation. For deterministic comparisons, first test whether the
chosen baseline is byte-identical. If it is not, require exact
configuration/count/event equivalence and the frozen numeric output criteria;
do not claim byte determinism.

BrushKit V2 removes Splats' `strongInitializationUnavailable` trainer-input
gate only after a real dense initializer producer is staged and audited. Variant
D still depends on the separate capture and ColmapKit tracked-pose bounded
reconstruction work required by Variant C. Do not remove that earlier guard or
substitute current ColmapKit silently. Physical execution on “iPad (2)” is a
later coordinated integration/device pass, not evidence from this handoff.
