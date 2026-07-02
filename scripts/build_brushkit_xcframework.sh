#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${BRUSHKIT_TARGET:-aarch64-apple-darwin}"
artifact_name="BrushKitFFI"
build_dir="$repo_root/target/brushkit-ffi/$target"
include_dir="$build_dir/include"
source_lib="$repo_root/target/$target/release/libbrush_c.a"
stripped_lib="$build_dir/libbrush_c.a"
output_dir="$repo_root/artifacts/$artifact_name.xcframework"

if ! command -v cbindgen >/dev/null 2>&1; then
  echo "error: cbindgen is required. Install it with: cargo install cbindgen --locked" >&2
  exit 1
fi

mkdir -p "$include_dir"

cbindgen "$repo_root/apps/brush-c" \
  --config "$repo_root/apps/brush-c/cbindgen.toml" \
  --output "$include_dir/brush_c.h"

cat > "$include_dir/module.modulemap" <<'MODULEMAP'
module BrushKitFFI {
  header "brush_c.h"
  export *
}
MODULEMAP

cargo build -p brush-c --release --target "$target"

cp "$source_lib" "$stripped_lib"
if xcrun strip -S -x "$stripped_lib" 2>/dev/null; then
  xcrun ranlib "$stripped_lib"
else
  echo "warning: strip failed for $stripped_lib; packaging unstripped archive" >&2
fi

rm -rf "$output_dir"
xcodebuild -create-xcframework \
  -library "$stripped_lib" \
  -headers "$include_dir" \
  -output "$output_dir"

du -sh "$output_dir"
