# BrushKit physical-device correctness harness

This local XCTest harness calls the BrushKit V3 C ABI directly on iOS. It exists so an unreleased native candidate can be measured without weakening a downstream consumer's pinned-build lifecycle guard.

The test freezes the retained tracked-pose workload at seed 0, SH3, 600 steps, mip rendering, refinement every 200 steps, 1080 maximum resolution, a 500,000-Gaussian cap, and 50% to 100% progressive resolution at step 200. Before training it verifies the exact retained trainer-leaf digest, component hashes, file/byte counts, and the downstream invariant fingerprint. It records native identity, population accounting, step/resource samples, thermal and battery state, memory, and an exhaustive binary PLY validation. The bundled fixture is intentionally ignored because it is derived benchmark data.

Stage an iOS XCFramework and trainer leaf:

```sh
ln -sfn /absolute/path/to/BrushKitFFI.xcframework Vendor/BrushKitFFI.xcframework
ln -sfn /absolute/path/to/tracked-pose/trainer-leaf Fixture/pinhole-fullres-component-0
```

Generate and build:

```sh
tuist generate --path device-harness/BrushKitDeviceBenchmark --no-open
xcodebuild build-for-testing \
  -workspace BrushKitDeviceBenchmark.xcworkspace \
  -scheme BrushKitDeviceBenchmarkTests \
  -destination 'platform=iOS,id=<device-udid>' \
  -derivedDataPath /private/tmp/brushkit-device-benchmark-derived \
  -allowProvisioningUpdates COMPILER_INDEX_STORE_ENABLE=NO
```

Run one fresh-process observation:

```sh
xcodebuild test-without-building \
  -workspace BrushKitDeviceBenchmark.xcworkspace \
  -scheme BrushKitDeviceBenchmarkTests \
  -destination 'platform=iOS,id=<device-udid>' \
  -derivedDataPath /private/tmp/brushkit-device-benchmark-derived \
  -only-testing:'BrushKitDeviceBenchmarkTests/BrushKitDeviceBenchmarkTests/testTrackedPoseFrozen600TrainerLeaf' \
  -allowProvisioningUpdates \
  -resultBundlePath /private/tmp/brushkit-device-benchmark.xcresult
```

Export the attached JSON and PLY:

```sh
xcrun xcresulttool export attachments \
  --path /private/tmp/brushkit-device-benchmark.xcresult \
  --output-path /absolute/path/to/evidence
```

`instrumentation_level` is Phase 0, matching the downstream failure control. CubeCL's synchronization-heavy GPU profiler remains off unless the linked native artifact itself was compiled with `BRUSHKIT_CUBECL_GPU_PROFILE=1`. The test rejects any native revision other than the audited local candidate, so it cannot silently exercise the released binary.
