#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_name="BrushKitFFI"
targets="${BRUSHKIT_TARGETS:-${BRUSHKIT_TARGET:-aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim}}"
artifact_root="${BRUSHKIT_ARTIFACT_OUTPUT_ROOT:-$repo_root/artifacts}"
build_root="$repo_root/target/brushkit-ffi"
include_dir="$build_root/include"
output_dir="$artifact_root/$artifact_name.xcframework"
zip_path="$artifact_root/$artifact_name.xcframework.zip"
cargo_features="${BRUSHKIT_CARGO_FEATURES:-}"
build_revision="${BRUSHKIT_BUILD_REVISION:-unknown}"
workspace_version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$repo_root/Cargo.toml" | head -1)"
cubecl_gpu_profile="off"
if [[ "${BRUSHKIT_CUBECL_GPU_PROFILE:-}" == "1" ]]; then
  cubecl_gpu_profile="on"
fi
rustc_version="$(rustc --version)"
cargo_version="$(cargo --version)"
cbindgen_version="$(cbindgen --version)"
xcode_version="$(xcodebuild -version | paste -sd ' ' -)"

if ! command -v cbindgen >/dev/null 2>&1; then
  echo "error: cbindgen is required. Install it with: cargo install cbindgen --locked" >&2
  exit 1
fi

mkdir -p "$include_dir" "$artifact_root"

cbindgen "$repo_root/apps/brush-c" \
  --config "$repo_root/apps/brush-c/cbindgen.toml" \
  --output "$include_dir/brush_c.h"

cat > "$include_dir/module.modulemap" <<'MODULEMAP'
module BrushKitFFI {
  header "brush_c.h"
  export *
}
MODULEMAP

xcframework_args=()
for target in $targets; do
  build_dir="$build_root/$target"
  source_lib="$repo_root/target/$target/release/libbrush_c.a"
  stripped_lib="$build_dir/libbrush_c.a"

  cargo_args=(build -p brush-c --release --target "$target")
  if [[ -n "$cargo_features" ]]; then
    cargo_args+=(--features "$cargo_features")
  fi
  cargo "${cargo_args[@]}"

  mkdir -p "$build_dir"
  cp "$source_lib" "$stripped_lib"
  if xcrun strip -S -x "$stripped_lib" 2>/dev/null; then
    xcrun ranlib "$stripped_lib"
  else
    echo "warning: strip failed for $stripped_lib; packaging unstripped archive" >&2
  fi

  xcframework_args+=(
    -library "$stripped_lib"
    -headers "$include_dir"
  )
done

rm -rf "$output_dir"
xcodebuild -create-xcframework \
  "${xcframework_args[@]}" \
  -output "$output_dir"

cp "$repo_root/LICENSE" "$output_dir/LICENSE"

printf '%s\n' \
  'format=brushkit-xcframework-provenance-v1' \
  "engine_source_sha=$build_revision" \
  "crate_version=$workspace_version" \
  'callable_abi=2' \
  'additive_train_abi=4' \
  "apple_features=${cargo_features:-none}" \
  "cubecl_gpu_profile=$cubecl_gpu_profile" \
  "rustc=$rustc_version" \
  "cargo=$cargo_version" \
  "cbindgen=$cbindgen_version" \
  "xcode=$xcode_version" \
  "targets=$targets" \
  > "$output_dir/BrushKitFFI.provenance.txt"

du -sh "$output_dir"

rm -f "$zip_path"
ditto -c -k --sequesterRsrc --keepParent "$output_dir" "$zip_path"
swift package compute-checksum "$zip_path"
