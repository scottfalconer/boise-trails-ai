#!/usr/bin/env python3
"""Summarize personal Strava segment-effort history against official segments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from derive_segment_crosswalk import name_similarity  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_STRAVA_DETAILS_DIR,
    METERS_PER_MILE,
    clean_trail_name,
    haversine_miles,
    load_official_segments,
    normalize_name,
    read_json,
    round_miles,
)


DEFAULT_PRIVATE_OUTPUT_JSON = YEAR_DIR / "inputs" / "personal" / "private" / "strava-segment-history-v1.json"
DEFAULT_PUBLIC_SUMMARY_JSON = YEAR_DIR / "derived" / "strava" / "strava-segment-history-summary-2026-05-06.json"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def latlng_to_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    lat, lon = value[:2]
    if lat is None or lon is None:
        return None
    return (float(lon), float(lat))


def effort_line(effort: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    segment = effort.get("segment") or {}
    start = latlng_to_point(segment.get("start_latlng"))
    end = latlng_to_point(segment.get("end_latlng"))
    if not start or not end:
        return None
    return start, end


def official_orientation_gap(
    effort_start: tuple[float, float],
    effort_end: tuple[float, float],
    official_segment: dict[str, Any],
) -> tuple[str, float]:
    forward = haversine_miles(effort_start, official_segment["start"]) + haversine_miles(
        effort_end, official_segment["end"]
    )
    reverse = haversine_miles(effort_start, official_segment["end"]) + haversine_miles(
        effort_end, official_segment["start"]
    )
    if forward <= reverse:
        return "forward", forward
    return "reverse", reverse


def best_official_match(effort: dict[str, Any], official_segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    line = effort_line(effort)
    if not line:
        return None
    effort_start, effort_end = line
    effort_name = normalize_name(str((effort.get("segment") or {}).get("name") or effort.get("name") or ""))
    best = None
    for segment in official_segments:
        orientation, endpoint_gap = official_orientation_gap(effort_start, effort_end, segment)
        similarity = max(
            name_similarity(effort_name, normalize_name(segment["seg_name"])),
            name_similarity(effort_name, normalize_name(segment["trail_name"])),
        )
        score = endpoint_gap + max(0.0, 0.15 - similarity * 0.15)
        if best is None or score < best["score"]:
            best = {
                "seg_id": segment["seg_id"],
                "seg_name": segment["seg_name"],
                "trail_name": segment["trail_name"],
                "official_direction_rule": segment["direction"],
                "orientation": orientation,
                "endpoint_gap_miles": endpoint_gap,
                "name_similarity": similarity,
                "score": score,
            }
    if not best:
        return None
    if best["endpoint_gap_miles"] <= 0.08 and best["name_similarity"] >= 0.45:
        confidence = "high"
    elif best["endpoint_gap_miles"] <= 0.15 or best["name_similarity"] >= 0.75:
        confidence = "medium"
    else:
        confidence = "low"
    best["confidence"] = confidence
    best["endpoint_gap_miles"] = round(best["endpoint_gap_miles"], 4)
    best["name_similarity"] = round(best["name_similarity"], 3)
    best.pop("score", None)
    return best


def iter_efforts(details_dir: Path) -> list[dict[str, Any]]:
    efforts = []
    for path in sorted(details_dir.glob("*.json")):
        activity = read_json(path)
        for effort in activity.get("segment_efforts") or []:
            copied = dict(effort)
            copied["_activity_id"] = activity.get("id")
            copied["_activity_name"] = activity.get("name")
            copied["_activity_date_local"] = activity.get("start_date_local")
            efforts.append(copied)
    return efforts


def pace_min_per_mile(effort: dict[str, Any]) -> float | None:
    distance_miles = float(effort.get("distance") or 0) / METERS_PER_MILE
    moving_minutes = float(effort.get("moving_time") or 0) / 60
    if distance_miles <= 0 or moving_minutes <= 0:
        return None
    return moving_minutes / distance_miles


def build_history(details_dir: Path, official_geojson: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    official_segments, official_meta = load_official_segments(official_geojson)
    raw_efforts = iter_efforts(details_dir)
    private_rows = []
    per_official: dict[int, list[dict[str, Any]]] = defaultdict(list)
    confidence_counts = Counter()
    for effort in raw_efforts:
        match = best_official_match(effort, official_segments)
        segment = effort.get("segment") or {}
        pace = pace_min_per_mile(effort)
        row = {
            "activity_id": effort.get("_activity_id"),
            "activity_name": effort.get("_activity_name"),
            "activity_date_local": effort.get("_activity_date_local"),
            "strava_effort_id": effort.get("id"),
            "strava_segment_id": segment.get("id"),
            "strava_segment_name": segment.get("name") or effort.get("name"),
            "distance_miles": round_miles(float(effort.get("distance") or 0) / METERS_PER_MILE),
            "moving_time_seconds": effort.get("moving_time"),
            "elapsed_time_seconds": effort.get("elapsed_time"),
            "pace_min_per_mile": round(pace, 2) if pace is not None else None,
            "official_match": match,
        }
        private_rows.append(row)
        confidence_counts[(match or {}).get("confidence") or "unmatched"] += 1
        if match and match["confidence"] in {"high", "medium"}:
            per_official[int(match["seg_id"])].append(row)

    official_rows = []
    for segment in official_segments:
        rows = per_official.get(segment["seg_id"], [])
        paces = [row["pace_min_per_mile"] for row in rows if row.get("pace_min_per_mile")]
        traversals = sorted({row["official_match"]["orientation"] for row in rows})
        dates = sorted({str(row["activity_date_local"])[:10] for row in rows if row.get("activity_date_local")})
        official_rows.append(
            {
                "seg_id": segment["seg_id"],
                "seg_name": segment["seg_name"],
                "trail_name": clean_trail_name(segment["seg_name"]),
                "official_direction_rule": segment["direction"],
                "matched_effort_count": len(rows),
                "matched_activity_count": len({row["activity_id"] for row in rows}),
                "observed_traversals": traversals,
                "best_pace_min_per_mile": round(min(paces), 2) if paces else None,
                "median_pace_min_per_mile": round(sorted(paces)[len(paces) // 2], 2) if paces else None,
                "first_observed_date": dates[0] if dates else None,
                "last_observed_date": dates[-1] if dates else None,
            }
        )

    private = {
        "dataset": "strava-segment-history-v1",
        "privacy": "private_activity_ids_and_segment_efforts",
        "source_datasets": {
            "details_dir": display_path(details_dir),
            "official_geojson": display_path(official_geojson),
        },
        "official_meta": official_meta,
        "summary": {
            "raw_effort_count": len(raw_efforts),
            "official_segment_count": len(official_segments),
            "matched_official_segment_count": sum(1 for row in official_rows if row["matched_effort_count"]),
            "match_confidence_counts": dict(sorted(confidence_counts.items())),
        },
        "official_segment_history": official_rows,
        "private_effort_rows": private_rows,
    }
    public_summary = {
        "dataset": "strava-segment-history-summary-2026-05-06",
        "private_history_path": display_path(DEFAULT_PRIVATE_OUTPUT_JSON),
        "privacy": "Exact activity ids and raw segment-effort rows stay in the private history file.",
        "summary": private["summary"],
        "matched_segments_by_effort_count": [
            {
                "seg_id": row["seg_id"],
                "seg_name": row["seg_name"],
                "matched_effort_count": row["matched_effort_count"],
                "observed_traversals": row["observed_traversals"],
            }
            for row in sorted(official_rows, key=lambda item: item["matched_effort_count"], reverse=True)
            if row["matched_effort_count"]
        ][:30],
    }
    return private, public_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details-dir", type=Path, default=DEFAULT_STRAVA_DETAILS_DIR)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--private-output-json", type=Path, default=DEFAULT_PRIVATE_OUTPUT_JSON)
    parser.add_argument("--public-summary-json", type=Path, default=DEFAULT_PUBLIC_SUMMARY_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    private, public_summary = build_history(args.details_dir, args.official_geojson)
    write_json(args.private_output_json, private)
    public_summary["private_history_path"] = display_path(args.private_output_json)
    write_json(args.public_summary_json, public_summary)
    print(f"Wrote {args.private_output_json}")
    print(f"Wrote {args.public_summary_json}")
    print(json.dumps(private["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
