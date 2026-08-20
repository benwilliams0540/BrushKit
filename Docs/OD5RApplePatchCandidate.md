# OD5R Apple Patch Candidate

Date: 2026-08-19

## Boundary and source identity

- Branch: `fix/od5r-terminal-nonfinite-accounting`
- Audited engine source: `b77dc41172c339ad8792413c22f75360e7a9b0ce`
- Mip scale-floor repair: `c27e45625103316795204091d0ff5d06f46cb424`
- Terminal-accounting repair: `6fd3ba9792b91ed903389d3a135f09119fa68154`
- Published `brushkit-ffi-v0.3.1` remains immutable at `b7422ae582eff2d76c4ec8ee329e08ce7a44dbbf`.
- No remote state, release, package consumer, Splats source, or ColmapKit source was changed.

The candidate keeps callable ABI 2, additive training ABI 3, every v0.2.1 entry point,
and the v0.3.1 public header/module map. It changes native trainer behavior and therefore
requires a new immutable patch release before downstream acceptance.

## Local Apple artifact

Artifact:

`artifacts/od5r-patch-b77dc411/BrushKitFFI.xcframework.zip`

| Fact | Value |
| --- | --- |
| ZIP bytes | 629,667,700 |
| ZIP SHA-256 | `362bd05e31a207002653b6611daf49d5fe85093d1e107e0d352acb4e2cca4c72` |
| SwiftPM checksum | `362bd05e31a207002653b6611daf49d5fe85093d1e107e0d352acb4e2cca4c72` |
| Slices | macOS arm64, iOS arm64, iOS Simulator arm64 |
| Deployment targets | macOS 13.0, iOS 18.0, iOS Simulator 18.0 |
| Cargo feature | `image-loss-bwd-tile-16` |
| CubeCL GPU profiler | off; diagnostic activation marker absent from all slices |
| Embedded source | `b77dc41172c339ad8792413c22f75360e7a9b0ce` in all slices |
| Header SHA-256 | `751df7dc6d46c1442cc9018211001af057f6ebf57bdd096ced0088661a9a8967` |
| Module-map SHA-256 | `21b16a4c992eec366b9cfe8a309d5ff8d20811a3e2ba58a6bf712062250ecee6` |

The archive passes `unzip -t`. All three libraries are arm64. C compile/link checks pass
for both the v0.2.1 legacy surface and the current V2/V3/job/provenance surface on all
three targets; the macOS binaries also run. The macOS identity smoke reports ABI 2,
version 0.3.1, Metal, the exact source SHA, tile16, and profiler-off provenance. Linked
macOS dependencies are only QuartzCore, Metal, Foundation, CoreFoundation, Objective-C,
iconv, and libSystem; there is no Homebrew or other host-only runtime dependency. The
static XCFramework is not signed, as expected; the dedicated physical-device test host
was Apple Development signed.

Build recipe:

```sh
env -u BRUSHKIT_CUBECL_GPU_PROFILE \
  BRUSHKIT_ARTIFACT_OUTPUT_ROOT="$PWD/artifacts/od5r-patch-b77dc411" \
  BRUSHKIT_BUILD_REVISION=b77dc41172c339ad8792413c22f75360e7a9b0ce \
  BRUSHKIT_CARGO_FEATURES=image-loss-bwd-tile-16 \
  ./scripts/build_brushkit_xcframework.sh
```

The complete mechanical build provenance is inside
`BrushKitFFI.xcframework/BrushKitFFI.provenance.txt`.

## Validation

Passed checks:

```sh
cargo fmt --all -- --check
cargo test -p brush-render mip_scale_floor_has_finite_gradients_for_od5r_small_scales -- --nocapture
cargo test -p brush-serde terminal_export_compacts_non_finite_rows_and_reports_properties -- --nocapture
cargo test -p brush-c --lib terminal_count_uses_export_compaction_after_delayed_step -- --nocapture
cargo test -p brush-c --test integration
cargo clippy -p brush-c --tests -- -D warnings -A clippy::redundant-clone
git diff --check
```

The GPU-backed tests require host Metal access. Strict Clippy without the narrow allow
reports only two `redundant_clone` style findings at
`crates/brush-render/src/gaussian_splats.rs:99` and `:109`. They do not change math or
correctness. They were not edited after packaging because that would change the source
identity and invalidate the completed slice and device proof.

Fresh local controls from this exact source also pass:

| Control | Initial | Refinement net | Terminal prune | Exported/final | PLY SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| OD5R | 11,169 | +2,040 | 0 | 13,209 | `1fcb7832bc7f5b3dfea9577620807878cab858062ae97c2c62acdbcb991113c5` |
| OD3R | 6,484 | +1,278 | 0 | 7,762 | `310bc5015fe2faac8a12f1a4277bac569db79f4c986579ffa22956b01acc993b` |

Both outputs are exhaustively finite. Byte hashes differ from earlier retained local runs,
consistent with the already-recorded residual GPU nondeterminism; population, event,
schema, and finiteness invariants are exact.

## Exact tracked-pose 600-step physical proof

Authoritative retained input:

`/Users/brw/Developer/apps/gsplat/build/codex-artifacts/tracked-pose-frozen600-v1/retrieved/run-1-failed/trainer-leaf`

The harness checks all values before calling BrushKit:

| Input fact | Recomputed value |
| --- | --- |
| Files / bytes | 110 / 358,272,040 |
| Canonical directory SHA-256 | `764504ec6d7fcc1c3b6cb296897d5f27ff6496e320d89f1c5f6efd5c58bf8c2d` |
| `cameras.bin` | `fb77970aa4904809074c6367a67c7e54fc524f1894fd7db6eb8fb3d0227e0d10` |
| `images.bin` | `b87a16ffbb523f3eb3663e03d554a2403e6796c6a28953544bfa2defe3e299b1` |
| `points3D.bin` | `14fb2891d9bc6795e45b7f29333963b8d708617412a94935e80c6867bf6ddd7b` |
| Frozen-invariant fingerprint | `8f7201b9ff6ed9813da4a3ac1abde5f4b74bbc5e84a4da76de6783345708ae94` |

The historical control checkpoint claims dataset SHA-256 `ae9332c46bef2c48ca8c66dea787950095952b5dbb897293a9381ee7956118cf`,
which is not reproducible from the retained post-run leaf. An earlier local digest
`fdb043aaf593671239f65baf17b5b3a8636bab3d37181400652d0aea6dd9da93`
was also invalid: its path algorithm dropped the first character of every relative path
when the directory URL ended in `/`. The corrected component-based algorithm produces
`764504ec...` identically from the authoritative host leaf, packaged test fixture, and
physical device. Individual component hashes and the full file/byte inventory agree.

Physical device: paired `iPad (2)`, hardware `iPad13,6` (M1), iPadOS 27.0. The harness
required nominal thermal state before launch and exact ABI/source/backend identity. It ran
seed 0, Mip, SH3, 600 steps, refinement every 200, 1080 maximum resolution, 500,000
Gaussian cap, 50% to 100% progressive resolution at step 200, and terminal export at 600.

| Population boundary | Count |
| --- | ---: |
| Initial | 5,641 |
| Refinement at 201 | +1,183 to 6,824 |
| Refinement at 401 | +1,515 to 8,339 |
| Terminal compaction at 600 | +0 / -0; non-finite prune 0 |
| Accounted final | 8,339 |
| Terminal final | 8,339 |
| Exported | 8,339 |

The delayed step-600 event still occurs after terminal compaction, reproducing the released
failure ordering, but it no longer overwrites the terminal count. Trainer wall time was
31.189534334 seconds. Thermal state was nominal at start, every 25-step sample, terminal,
and end. Battery was 100%, full/charging, Low Power Mode off. Peak resident memory was
1,043,447,808 bytes. The bound backend was confirmed as Metal; the exact adapter name is
unavailable because the current default Burn setup does not expose it.

The attached PLY is 1,969,582 bytes, has 8,339 rows and 59 float properties, and contains
zero non-finite values across all 492,001 values. Its SHA-256 is
`66b049994857ecfb6462066c01488365060d9d028e82fb3663c74f51f253a4b4`.
The released-0.3.1 control PLY is finite but has only 8,336 rows, SHA-256
`cf69f22179770ec21883091f9c54296811c02f82cc4c606a517a699203417f53`.
The candidate's zero terminal pruning and exact population arithmetic prove that the repair
preserved the three rows rather than masking them with compaction. GPU nondeterminism and
the repaired gradient math make row-wise numerical identity with the old control an invalid
criterion; schema and finite numeric ranges remain structurally consistent.

Raw evidence:

- Passing result: `artifacts/od5r-patch-b77dc411/evidence/physical/tracked-pose-frozen600-b77-run2.xcresult`
- Exported attachments and manifest: `artifacts/od5r-patch-b77dc411/evidence/physical/run2-attachments`
- Three stopped preflight bundles (no training): the sibling
  `tracked-pose-frozen600-b77.xcresult`, `tracked-pose-frozen600-b77-preflight2.xcresult`,
  and `tracked-pose-frozen600-b77-run1.xcresult` records.

Independent output audit:

```sh
python3 scripts/audit_ply_finiteness.py \
  artifacts/od5r-patch-b77dc411/evidence/physical/run2-attachments/2EDCF8C3-E5AE-49DD-BE3F-C5FB5AD5CB15
```

## Remaining publication and integration decision

All local correctness, Apple packaging, ABI, and physical engine gates are closed for source
`b77dc411...`. Publication is not authorized by this task. The smallest next decision is
whether to authorize a new immutable patch release from this exact source and feature set.
After publication, Splats must pin that immutable package and rerun its formal tracked-pose
acceptance. This proof is engine correctness evidence only; it is not a product-quality or
tracked-pose-geometry acceptance claim.
