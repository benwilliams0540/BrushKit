#!/usr/bin/env python3

"""Audit binary little-endian float PLY rows without rewriting the source file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys


def _display_number(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value


def _read_header(data: bytes) -> tuple[list[str], int]:
    for marker in (b"end_header\n", b"end_header\r\n"):
        marker_offset = data.find(marker)
        if marker_offset >= 0:
            header_end = marker_offset + len(marker)
            return data[:header_end].decode("ascii").splitlines(), header_end
    raise ValueError("PLY header has no end_header marker")


def audit(path: pathlib.Path, selected_rows: set[int]) -> dict[str, object]:
    data = path.read_bytes()
    header, payload_offset = _read_header(data)
    if not header or header[0] != "ply":
        raise ValueError("not a PLY file")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary_little_endian 1.0 PLY files are supported")

    vertex_lines = [line for line in header if line.startswith("element vertex ")]
    if len(vertex_lines) != 1:
        raise ValueError(f"expected one vertex declaration, found {len(vertex_lines)}")
    vertex_count = int(vertex_lines[0].split()[-1])
    property_lines = [line for line in header if line.startswith("property ")]
    unsupported = [line for line in property_lines if not line.startswith("property float ")]
    if unsupported:
        raise ValueError(f"unsupported non-float vertex properties: {unsupported}")
    properties = [line.split()[-1] for line in property_lines]
    row_stride = len(properties) * 4
    expected_payload_bytes = vertex_count * row_stride
    payload = data[payload_offset:]
    if len(payload) != expected_payload_bytes:
        raise ValueError(
            f"payload size mismatch: expected {expected_payload_bytes}, found {len(payload)}"
        )

    minima = [math.inf] * len(properties)
    maxima = [-math.inf] * len(properties)
    finite_counts = [0] * len(properties)
    non_finite_counts = [0] * len(properties)
    invalid_rows: list[dict[str, object]] = []
    selected_row_values: list[dict[str, object]] = []

    row_struct = struct.Struct(f"<{len(properties)}f")
    for row_index, row in enumerate(row_struct.iter_unpack(payload)):
        invalid_values = []
        for property_index, value in enumerate(row):
            if math.isfinite(value):
                finite_counts[property_index] += 1
                minima[property_index] = min(minima[property_index], value)
                maxima[property_index] = max(maxima[property_index], value)
            else:
                non_finite_counts[property_index] += 1
                invalid_values.append(
                    {
                        "property": properties[property_index],
                        "propertyIndex": property_index,
                        "value": _display_number(value),
                    }
                )
        if invalid_values:
            invalid_rows.append({"row": row_index, "values": invalid_values})
        if row_index in selected_rows:
            selected_row_values.append(
                {
                    "row": row_index,
                    "values": {
                        name: _display_number(value)
                        for name, value in zip(properties, row, strict=True)
                    },
                }
            )

    property_stats = []
    for index, name in enumerate(properties):
        finite_count = finite_counts[index]
        property_stats.append(
            {
                "name": name,
                "finiteCount": finite_count,
                "nonFiniteCount": non_finite_counts[index],
                "minimum": minima[index] if finite_count else None,
                "maximum": maxima[index] if finite_count else None,
            }
        )

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": "binary_little_endian 1.0",
        "fileBytes": len(data),
        "payloadOffset": payload_offset,
        "payloadBytes": len(payload),
        "vertexCount": vertex_count,
        "floatPropertyCount": len(properties),
        "rowStrideBytes": row_stride,
        "properties": properties,
        "propertyStats": property_stats,
        "nonFiniteRowCount": len(invalid_rows),
        "nonFiniteValueCount": sum(non_finite_counts),
        "nonFiniteRows": invalid_rows,
        "selectedRows": selected_row_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--rows",
        default="",
        help="comma-separated zero-based vertex rows to include verbatim",
    )
    args = parser.parse_args()

    try:
        selected_rows = {int(value) for value in args.rows.split(",") if value}
    except ValueError as error:
        parser.error(f"invalid --rows value: {error}")

    try:
        reports = [audit(path, selected_rows) for path in args.paths]
    except (OSError, ValueError, struct.error) as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 1

    result: object = reports[0] if len(reports) == 1 else reports
    if args.compact:
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
