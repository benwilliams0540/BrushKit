#!/usr/bin/env python3

import argparse
import itertools
import json
import math
import pathlib
import statistics
import struct


GROUPS = {
    "position": range(0, 3),
    "log_scale": range(3, 6),
    "opacity": range(6, 7),
    "rotation": range(7, 11),
    "sh": range(11, 59),
}


def load_ply(path: pathlib.Path):
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("utf-8")
    lines = header.splitlines()
    vertex_count = int(next(line for line in lines if line.startswith("element vertex ")).split()[-1])
    properties = [line.split()[-1] for line in lines if line.startswith("property float ")]
    value_count = vertex_count * len(properties)
    values = struct.unpack(f"<{value_count}f", data[header_end:])
    return properties, vertex_count, values


def run_ply(run_dir: pathlib.Path):
    candidates = [
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".json"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one PLY attachment in {run_dir}, found {candidates}")
    return candidates[0]


def metric(left, right, property_count, selected_indices=None):
    if selected_indices is None:
        diffs = [abs(a - b) for a, b in zip(left, right)]
    else:
        selected = set(selected_indices)
        diffs = [
            abs(a - b)
            for index, (a, b) in enumerate(zip(left, right))
            if index % property_count in selected
        ]
    return {
        "mae": sum(diffs) / len(diffs),
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
        "p99Absolute": sorted(diffs)[int(0.99 * (len(diffs) - 1))],
        "maxAbsolute": max(diffs),
    }


def summarize(pairs):
    summary = {}
    for field in ("mae", "rmse", "p99Absolute", "maxAbsolute"):
        values = [pair["all"][field] for pair in pairs]
        summary[field] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    summary["groupMedianMAE"] = {
        group: statistics.median(pair["groups"][group]["mae"] for pair in pairs)
        for group in GROUPS
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attachments", type=pathlib.Path)
    args = parser.parse_args()

    runs = {}
    schema = None
    vertex_count = None
    for run_dir in sorted(path for path in args.attachments.iterdir() if path.is_dir()):
        properties, count, values = load_ply(run_ply(run_dir))
        if schema is None:
            schema = properties
            vertex_count = count
        if properties != schema or count != vertex_count:
            raise RuntimeError(f"schema mismatch in {run_dir.name}")
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite value in {run_dir.name}")
        runs[run_dir.name] = values

    property_count = len(schema)
    categories = {"baselineBaseline": [], "candidateCandidate": [], "baselineCandidate": []}
    for left_name, right_name in itertools.combinations(sorted(runs), 2):
        left_is_baseline = left_name.startswith("a")
        right_is_baseline = right_name.startswith("a")
        if left_is_baseline and right_is_baseline:
            category = "baselineBaseline"
        elif not left_is_baseline and not right_is_baseline:
            category = "candidateCandidate"
        else:
            category = "baselineCandidate"
        categories[category].append(
            {
                "pair": [left_name, right_name],
                "all": metric(runs[left_name], runs[right_name], property_count),
                "groups": {
                    group: metric(runs[left_name], runs[right_name], property_count, indices)
                    for group, indices in GROUPS.items()
                },
            }
        )

    result = {
        "schemaVersion": 1,
        "runCount": len(runs),
        "vertexCount": vertex_count,
        "floatPropertyCount": property_count,
        "scannedValueCountPerRun": vertex_count * property_count,
        "categorySummary": {name: summarize(pairs) for name, pairs in categories.items()},
        "pairs": categories,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
