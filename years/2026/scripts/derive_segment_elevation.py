#!/usr/bin/env python3
"""Precompute per-segment, per-direction DEM ascent/descent."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    elevation_gain_loss_for_line,
    load_dem_context,
    load_official_segments,
    round_miles,
)


DEFAULT_OUTPUT_JSON = YEAR_DIR / "derived" / "elevation" / "segment-elevation-2026-05-06.json"
DEFAULT_OUTPUT_CSV = YEAR_DIR / "derived" / "elevation" / "segment-elevation-2026-05-06.csv"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def elevation_rows(segments: list[dict[str, Any]], sampler) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        directions = ["forward"] if segment["direction"] == "ascent" else ["forward", "reverse"]
        for traversal in directions:
            coords = segment["coordinates"] if traversal == "forward" else list(reversed(segment["coordinates"]))
            ascent, descent, sampled = elevation_gain_loss_for_line(coords, sampler)
            rows.append(
                {
                    "seg_id": segment["seg_id"],
                    "seg_name": segment["seg_name"],
                    "trail_name": segment["trail_name"],
                    "official_miles": round_miles(segment["official_miles"]),
                    "official_direction_rule": segment["direction"],
                    "traversal": traversal,
                    "ascent_ft": round(ascent) if sampled else None,
                    "descent_ft": round(descent) if sampled else None,
                    "grade_adjusted_miles": round_miles(segment["official_miles"] + ascent / 1000)
                    if sampled
                    else None,
                    "sampled": sampled,
                    "allowed_for_credit": traversal == "forward" or segment["direction"] == "both",
                }
            )
    return rows


def build_dataset(official_geojson: Path, dem_tif: Path, dem_summary_json: Path) -> dict[str, Any]:
    segments, official_meta = load_official_segments(official_geojson)
    dem_context = load_dem_context(dem_tif, dem_summary_json)
    sampler = dem_context["sampler"]
    rows = elevation_rows(segments, sampler) if sampler else []
    return {
        "dataset": "segment-elevation-2026-05-06",
        "source_datasets": {
            "official_geojson": display_path(official_geojson),
            "dem_tif": display_path(dem_tif),
            "dem_summary_json": display_path(dem_summary_json),
        },
        "official_meta": official_meta,
        "dem_meta": dem_context["metadata"],
        "summary": {
            "official_segment_count": len(segments),
            "row_count": len(rows),
            "sampled_row_count": sum(1 for row in rows if row["sampled"]),
            "direction_rule_counts": dict(sorted(Counter(segment["direction"] for segment in segments).items())),
            "traversal_counts": dict(sorted(Counter(row["traversal"] for row in rows).items())),
        },
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seg_id",
        "seg_name",
        "trail_name",
        "official_miles",
        "official_direction_rule",
        "traversal",
        "ascent_ft",
        "descent_ft",
        "grade_adjusted_miles",
        "sampled",
        "allowed_for_credit",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = build_dataset(args.official_geojson, args.dem_tif, args.dem_summary_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, dataset["rows"])
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(json.dumps(dataset["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
