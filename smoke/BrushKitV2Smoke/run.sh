#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
xcframework="${1:-$repo_root/artifacts/BrushKitFFI.xcframework}"
dataset="${2:-$repo_root/apps/brush-c/tests/data/test_dataset}"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/brushkit-v2-swift.XXXXXX")"
trap 'rm -rf "$smoke_root"' EXIT

macos_library="$(find "$xcframework" -path '*macos*' -name libbrush_c.a -print -quit)"
if [[ -z "$macos_library" ]]; then
  echo "error: macOS arm64 library not found in $xcframework" >&2
  exit 1
fi
headers="$(dirname "$macos_library")/Headers"
binary="$repo_root/target/brushkit-ffi/BrushKitV2Smoke"
mkdir -p "$(dirname "$binary")"

xcrun swiftc \
  "$repo_root/smoke/BrushKitV2Smoke/main.swift" \
  -I "$headers" \
  "$macos_library" \
  -framework QuartzCore \
  -framework Metal \
  -framework Foundation \
  -framework CoreFoundation \
  -lobjc -liconv -lc -lm \
  -o "$binary"

"$binary" "$dataset" "$smoke_root/output"
