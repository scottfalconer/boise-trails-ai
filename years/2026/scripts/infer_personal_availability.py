#!/usr/bin/env python3
"""Infer scheduler availability profiles from prior Strava activity days."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV = (
    YEAR_DIR
    / "inputs"
    / "strava"
    / "api-pulls"
    / "2026-05-03"
    / "activity_detail_summaries.csv"
)
DEFAULT_OUTPUT_JSON = YEAR_DIR / "derived" / "personal-availability-inference-2026-05-04.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "derived" / "personal-availability-inference-2026-05-04.md"


def parse_local_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1 - fraction) + ordered[upper_index] * fraction


def rounded_minutes(value: float | None) -> int:
    if value is None:
        return 0
    return int(round(value / 5) * 5)


def read_activity_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("sport_type") != "Run":
                continue
            start = parse_local_datetime(str(row["start_date_local"]))
            rows.append(
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "start_date_local": row.get("start_date_local"),
                    "activity_date_local": start.date().isoformat(),
                    "year": start.year,
                    "weekday": start.strftime("%A"),
                    "is_weekend": start.weekday() >= 5,
                    "elapsed_minutes": int(row.get("elapsed_time_s") or 0) / 60,
                    "moving_minutes": int(row.get("moving_time_s") or 0) / 60,
                    "distance_miles": float(row.get("distance_m") or 0) / 1609.344,
                }
            )
    return rows


def in_challenge_window(row: dict[str, Any], years: set[int]) -> bool:
    if int(row["year"]) not in years:
        return False
    month_day = str(row["activity_date_local"])[5:]
    return "06-18" <= month_day <= "07-18"


def daily_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["activity_date_local"])].append(row)

    days = []
    for day, activities in sorted(grouped.items()):
        days.append(
            {
                "date": day,
                "weekday": activities[0]["weekday"],
                "is_weekend": bool(activities[0]["is_weekend"]),
                "activity_count": len(activities),
                "elapsed_minutes": round(sum(float(item["elapsed_minutes"]) for item in activities), 1),
                "moving_minutes": round(sum(float(item["moving_minutes"]) for item in activities), 1),
                "distance_miles": round(sum(float(item["distance_miles"]) for item in activities), 2),
                "activity_names": [str(item["name"]) for item in activities],
            }
        )
    return days


def stats_for_days(days: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(day["elapsed_minutes"]) for day in days]
    moving = [float(day["moving_minutes"]) for day in days]
    distance = [float(day["distance_miles"]) for day in days]

    def series(values: list[float]) -> dict[str, Any]:
        return {
            "min": round(percentile(values, 0) or 0, 1),
            "median": round(percentile(values, 50) or 0, 1),
            "p75": round(percentile(values, 75) or 0, 1),
            "p90": round(percentile(values, 90) or 0, 1),
            "max": round(percentile(values, 100) or 0, 1),
        }

    return {
        "day_count": len(days),
        "multi_activity_days": len([day for day in days if int(day["activity_count"]) > 1]),
        "elapsed_minutes": series(elapsed),
        "moving_minutes": series(moving),
        "distance_miles": series(distance),
    }


def build_profile(
    name: str,
    weekday_elapsed_minutes: float | None,
    weekend_elapsed_minutes: float | None,
    logistics_buffer_minutes: int,
    rest_days_after_long: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "start_time": "07:00",
        "weekday_max_minutes": rounded_minutes((weekday_elapsed_minutes or 0) + logistics_buffer_minutes),
        "weekend_max_minutes": rounded_minutes((weekend_elapsed_minutes or 0) + logistics_buffer_minutes),
        "long_outing_minutes": 240,
        "rest_days_after_long": rest_days_after_long,
        "logistics_buffer_minutes_added_to_strava_elapsed": logistics_buffer_minutes,
    }


def infer_availability(
    rows: list[dict[str, Any]],
    years: set[int],
    logistics_buffer_minutes: int,
) -> dict[str, Any]:
    window_rows = [row for row in rows if in_challenge_window(row, years)]
    days = daily_totals(window_rows)
    weekday_days = [day for day in days if not day["is_weekend"]]
    weekend_days = [day for day in days if day["is_weekend"]]

    weekday_elapsed = [float(day["elapsed_minutes"]) for day in weekday_days]
    weekend_elapsed = [float(day["elapsed_minutes"]) for day in weekend_days]
    profiles = {
        "historical_p75_plus_logistics": build_profile(
            "historical_p75_plus_logistics",
            percentile(weekday_elapsed, 75),
            percentile(weekend_elapsed, 75),
            logistics_buffer_minutes,
            rest_days_after_long=0,
        ),
        "historical_p90_plus_logistics": build_profile(
            "historical_p90_plus_logistics",
            percentile(weekday_elapsed, 90),
            percentile(weekend_elapsed, 90),
            logistics_buffer_minutes,
            rest_days_after_long=0,
        ),
        "full_clear_sensitivity": {
            "name": "full_clear_sensitivity",
            "start_time": "07:00",
            "weekday_max_minutes": 240,
            "weekend_max_minutes": 480,
            "long_outing_minutes": 240,
            "rest_days_after_long": 0,
            "note": "Sensitivity profile that cleared all executable 2026 segments in the current scheduler; not inferred as historically typical.",
        },
    }

    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "activity_detail_summary_csv": str(DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV),
            "years": sorted(years),
            "challenge_window_month_day": "06-18 through 07-18",
            "activity_type": "Run",
            "logistics_buffer_minutes": logistics_buffer_minutes,
        },
        "activity_count": len(window_rows),
        "activity_day_count": len(days),
        "daily_totals": days,
        "stats": {
            "all_days": stats_for_days(days),
            "weekdays": stats_for_days(weekday_days),
            "weekends": stats_for_days(weekend_days),
        },
        "scheduler_profiles": profiles,
        "caveats": [
            "Strava elapsed time does not include driving to/from trailheads; profiles add a configurable logistics buffer.",
            "Past activity timing is evidence of what happened, not a confirmed 2026 calendar commitment.",
            "Weekend history is sparse and skewed shorter in the selected data, so full-clear planning still needs explicit user confirmation.",
        ],
    }


def render_markdown(inference: dict[str, Any]) -> str:
    lines = [
        "# Personal Availability Inference",
        "",
        f"- Source rows in challenge windows: {inference['activity_count']}",
        f"- Activity days: {inference['activity_day_count']}",
        f"- Years: {', '.join(str(year) for year in inference['source']['years'])}",
        f"- Logistics buffer added to Strava elapsed: {inference['source']['logistics_buffer_minutes']} min",
        "",
        "## Timing Stats",
        "",
        "| Group | Days | Elapsed median | Elapsed p75 | Elapsed p90 | Elapsed max | Distance p75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in inference["stats"].items():
        elapsed = stats["elapsed_minutes"]
        distance = stats["distance_miles"]
        lines.append(
            f"| {label} | {stats['day_count']} | {elapsed['median']} | {elapsed['p75']} | {elapsed['p90']} | {elapsed['max']} | {distance['p75']} |"
        )

    lines.extend(["", "## Scheduler Profiles", ""])
    for key, profile in inference["scheduler_profiles"].items():
        lines.append(
            f"- `{key}`: weekday {profile['weekday_max_minutes']} min, weekend {profile['weekend_max_minutes']} min, rest after long {profile['rest_days_after_long']}"
        )

    lines.extend(["", "## Caveats", ""])
    for caveat in inference["caveats"]:
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-detail-summary-csv", type=Path, default=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV)
    parser.add_argument("--years", default="2024,2025")
    parser.add_argument("--logistics-buffer-minutes", type=int, default=30)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = {int(value.strip()) for value in str(args.years).split(",") if value.strip()}
    rows = read_activity_rows(args.activity_detail_summary_csv)
    inference = infer_availability(rows, years, args.logistics_buffer_minutes)
    inference["source"]["activity_detail_summary_csv"] = str(args.activity_detail_summary_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(inference, indent=2) + "\n")
    args.output_md.write_text(render_markdown(inference))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(
        "Inferred profiles: "
        + ", ".join(
            f"{name}={profile['weekday_max_minutes']}/{profile['weekend_max_minutes']} min"
            for name, profile in inference["scheduler_profiles"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
