# OD5R Trainer Non-Finite Investigation

Date: 2026-08-18

## Scope and release boundary

- Investigation base: `c7dfd58ee242afe9b4d411ac72079907a856e1f9`
- Immutable published release remains unchanged: `brushkit-ffi-v0.3.1` / `b7422ae582eff2d76c4ec8ee329e08ce7a44dbbf`
- Local branch: `fix/od5r-terminal-nonfinite-accounting`
- Math repair: `c27e45625103316795204091d0ff5d06f46cb424`
- Accounting repair: `6fd3ba9792b91ed903389d3a135f09119fa68154`
- No Splats or ColmapKit source, package pin, release asset, tag, or remote state was changed.

## Read-only source evidence

The failing OD5R evidence is rooted at:

`/Users/brw/Developer/apps/gsplat/build/codex-artifacts/speedy-splats-overnight/phase3-v031-d-revisit-feature-preflight/retrieved/OD5R-v031-feature480-failed`

| File | SHA-256 | Rows | Finiteness |
| --- | --- | ---: | --- |
| `trainer-leaves/pinhole-fullres-component-0/initializers/variant-d.ply` | `012cc15b14ae5eebd0dac649a75133e8fccff2225e99fa22a64035b4ea7d3b5e` | 11,169 | All 14 float properties finite |
| `generations/brush-component-0-fullres-1080-300-500000/component-0_300.ply` | `5c9cce1923abc4168c2b13a746f72cf28230c3d7e71776f303d9c33344ab4972` | 13,198 | All 59 exported float properties finite after terminal compaction |
| `generations/brush-component-0-fullres-1080-300-500000/brush-v2-evidence.json` | `35fdb052997e6e5e0254a70efe52742033b0028d5104b307a0e158617b04ae9e` | — | Reports the original inconsistent accounting |

The preserved valid OD3R initializer and terminal export hashes are respectively
`4a7d7fb6d9e552aad47b72d4a83e58f98c31d6b40b6e8a0ba620b67d657f67a7`
and `8ca16543587f039b9fdaea1d676b6a3a8477c0dc03acad3af5901d895c12efa7`.

## First-invalid boundary

Default-off trainer diagnostics reproduced the exact OD5R configuration, seed, 300 steps,
progressive-resolution transition, refinement cadence, Mip mode, SH degree, tile16 feature,
and 500,000 Gaussian ceiling.

1. The initializer, post-import parameters, optimizer steps 1 through 201, and the post-step-201 refinement population of 13,209 were fully finite.
2. Before the transform optimizer at iteration 202, all three log-scale gradients became infinite for rows `209, 1796, 1990, 2315, 2520, 3325, 3688, 3800, 4090`. Means and rotations were still finite. The rows were visible and their raw transforms and Mip scale floors were finite.
3. The transform optimizer then converted those infinite gradients into non-finite log-scale parameters. Row `70` first joined at iteration 232 and row `5879` at iteration 267.
4. The complete internal terminal invalid set was `70, 209, 1796, 1990, 2315, 2520, 3325, 3688, 3800, 4090, 5879`; every row was an original initializer index, not a newly split child.
5. Terminal export correctly removed those 11 internal rows, which is why the saved PLY itself is finite.

The pre-fix diagnostic hashes are:

- `/private/tmp/brushkit-od5r-gradient-details.20260818/trainer-finiteness.jsonl`: `5f418634a13cb3688b0d1114779d567498a6a16446353ef21074593d4c418c92`
- `/private/tmp/brushkit-od5r-gradient-details.20260818/transform-gradients.jsonl`: `e8d285acf3d81a9898d25f33e51c9a35606355d197b3ad5ab041c34ca64c6997`

## Ownership and repairs

BrushKit owns both defects.

The training defect was in Mip-Splatting's differentiable opacity compensation. The direct
`sqrt(det(s²) / det(s² + f²))` quotient makes the generic division backward square a determinant
near `1e-20`; that intermediate underflows even though the mathematically reduced gradient is
finite. The focused OD5R regression reproduced `[-inf, -inf, -inf]` without the trainer. The
repair evaluates the same determinant ratio in log space and locks forward scale/opacity values
and analytical nonzero gradients to explicit tolerances. Loss math and training policy are unchanged.

The accounting defect was independent. Terminal compaction emitted the correct 13,198 export count,
then the delayed step-300 event overwrote the callback's last count with the pre-compaction 13,209
live count. The callback now remembers the exhaustive export population separately, and a synthetic
event-order test locks the terminal count to it.

## Before and after

| Run | Initial | Refinement net | Terminal non-finite prune | Exported | Terminal final |
| --- | ---: | ---: | ---: | ---: | ---: |
| OD5R v0.3.1 evidence | 11,169 | +2,040 | -11 | 13,198 | 13,209 (incorrect) |
| OD5R repaired local run | 11,169 | +2,040 | 0 | 13,209 | 13,209 |
| OD3R repaired control | 6,484 | +1,278 | 0 | 7,762 | 7,762 |

The repaired OD5R run had no non-finite gradient or parameter checkpoint through step 300.
Its output was 13,209 fully finite rows, SHA-256
`df4bda779fba80a6113fd7a6a03947f4d4b9660ed3ceb598d47e9ad8f21a0ebe`.
The repaired OD3R control had 7,762 fully finite rows, SHA-256
`c7e596b35b3bc13f7ace4719381c30f35872446d57a9a466b2dc250af91c1f2e`.

## Reproduction and audit

Audit a source or exported PLY without modifying it:

```sh
python3 scripts/audit_ply_finiteness.py \
  --rows 70,209,1796,1990,2315,2520,3325,3688,3800,4090,5879 \
  /absolute/path/to/variant-d.ply
```

Run the focused math and accounting regressions:

```sh
cargo test -p brush-render mip_scale_floor_has_finite_gradients_for_od5r_small_scales -- --nocapture
cargo test -p brush-c --lib terminal_count_uses_export_compaction_after_delayed_step -- --nocapture
```

Run a frozen external Variant-D diagnosis (manual and ignored by default):

```sh
BRUSHKIT_VARIANT_D_DATASET=/absolute/path/to/trainer-leaves/pinhole-fullres-component-0 \
BRUSHKIT_VARIANT_D_OUTPUT=/private/tmp/brushkit-variant-d-output \
BRUSHKIT_TRAINER_DIAGNOSTIC_PATH=/private/tmp/brushkit-variant-d-output/trainer-finiteness.jsonl \
BRUSHKIT_TRAINER_GRADIENT_DIAGNOSTIC_PATH=/private/tmp/brushkit-variant-d-output/transform-gradients.jsonl \
cargo test -p brush-c --test integration \
  --features image-loss-bwd-tile-16,trainer-diagnostics \
  diagnose_external_variant_d_fixture -- --ignored --nocapture
```

The `trainer-diagnostics` feature is not enabled by default and is absent from the published
v0.3.1 artifact. It adds readbacks and must not be used for performance measurements.

## ABI and package verdict

The repair changes no public C type, symbol, capability bit, ABI version, Swift API, default tile
policy, or package identity. Callable ABI 2, additive training ABI 3, and all v0.2.1 entry points
remain intact. A new native package candidate is warranted before downstream product acceptance,
because the math repair changes shipped trainer behavior. It should be a new immutable patch release;
the existing v0.3.1 tag and asset must not move. This investigation does not authorize that release.
