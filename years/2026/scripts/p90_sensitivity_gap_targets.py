#!/usr/bin/env python3
"""Summarize missing segments from availability sensitivity scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_joint_field_day_optimizer as joint_optimizer  # noqa: E402
from p90_completion_gap_analyzer import official_segments  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_SENSITIVITY_JSON = YEAR_DIR / "checkpoints" / "p90-availability-sensitivity-audit-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-sensitivity-gap-targets-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def missing_rows_for_scenario(
    scenario: dict[str, Any],
    segment_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing_ids = (scenario.get("max_coverage_solution") or {}).get("missing_segment_ids") or []
    rows = []
    for seg_id in missing_ids:
        segment = segment_index[int(seg_id)]
        rows.append(
            {
                "seg_id": int(seg_id),
                "seg_name": segment["seg_name"],
                "trail_name": segment["trail_name"],
                "official_miles": segment["official_miles"],
                "direction": segment["direction"],
            }
        )
    return rows


def group_missing_by_trail(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trail[row["trail_name"]].append(row)
    return [
        {
            "trail_name": trail_name,
            "segment_count": len(items),
            "official_miles": round(sum(float(item["official_miles"]) for item in items), 2),
            "segment_ids": [item["seg_id"] for item in sorted(items, key=lambda item: item["seg_id"])],
        }
        for trail_name, items in sorted(by_trail.items())
    ]


def field_day_options_for_missing(
    *,
    missing_ids: list[int],
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
) -> dict[int, list[dict[str, Any]]]:
    state = read_json(joint_optimizer.repaired.DEFAULT_STATE_JSON)
    target_ids, candidates, field_menu = joint_optimizer.all_repaired_candidates()
    bounded = [
        candidate
        for candidate in candidates
        if int(candidate["door_to_door_p90_minutes"]) <= max(weekday_bound, weekend_bound)
        and candidate.get("validation_passed") is True
        and candidate.get("manual_design_hold") is not True
    ]
    loops, _missing_coordinates = joint_optimizer.candidate_loop_records(bounded, field_menu, state)
    field_days = joint_optimizer.generate_direct_field_day_candidates(
        loops,
        state,
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        max_combo_size=max_combo_size,
        neighbor_limit=neighbor_limit,
    )
    options: dict[int, list[dict[str, Any]]] = {}
    for seg_id in missing_ids:
        containing = [day for day in field_days if seg_id in set(day["segment_ids"])]
        options[seg_id] = [
            {
                "day_type": day["day_type"],
                "p75_minutes": day["p75_minutes"],
                "p90_minutes": day["p90_minutes"],
                "loop_count": day["loop_count"],
                "segment_count": len(day["segment_ids"]),
                "loop_ids": day["loop_ids"],
            }
            for day in sorted(containing, key=lambda day: (day["p75_minutes"], day["p90_minutes"]))[:8]
        ]
    return options


def build_report(
    *,
    sensitivity: dict[str, Any],
    max_combo_size: int,
    neighbor_limit: int,
) -> dict[str, Any]:
    segment_index = official_segments(read_json(DEFAULT_OFFICIAL_GEOJSON))
    scenario_reports = []
    for scenario in sensitivity.get("scenarios") or []:
        rows = missing_rows_for_scenario(scenario, segment_index)
        scenario_reports.append(
            {
                "scenario": scenario["scenario"],
                "weekday_bound_minutes": scenario["weekday_bound_minutes"],
                "weekend_bound_minutes": scenario["weekend_bound_minutes"],
                "max_coverage_segments": (scenario.get("max_coverage_solution") or {}).get("covered_segment_count"),
                "missing_segment_count": len(rows),
                "missing_official_miles": round(sum(float(row["official_miles"]) for row in rows), 2),
                "missing_by_trail": group_missing_by_trail(rows),
                "missing_segments": rows,
            }
        )
    near_miss = next(
        (
            scenario
            for scenario in scenario_reports
            if scenario["scenario"] == "292_weekday_360_weekend"
        ),
        None,
    )
    near_miss_options = {}
    if near_miss:
        near_miss_options = field_day_options_for_missing(
            missing_ids=[row["seg_id"] for row in near_miss["missing_segments"]],
            weekday_bound=int(near_miss["weekday_bound_minutes"]),
            weekend_bound=int(near_miss["weekend_bound_minutes"]),
            max_combo_size=max_combo_size,
            neighbor_limit=neighbor_limit,
        )
    return {
        "objective": "identify route-redesign targets from p90 availability sensitivity near misses",
        "source_files": {
            "sensitivity_json": display_path(DEFAULT_SENSITIVITY_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
        },
        "config": {
            "max_combo_size": max_combo_size,
            "neighbor_limit": neighbor_limit,
        },
        "scenario_reports": scenario_reports,
        "near_miss_scenario": near_miss,
        "near_miss_field_day_options": near_miss_options,
        "caveats": [
            "Missing segments are from max-coverage schedules, not from a final accepted plan.",
            "A field-day option existing for a missing segment means the segment is schedulable alone; it does not mean it fits without displacing more valuable coverage.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P90 Sensitivity Gap Targets",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Scenario Missing Coverage",
        "",
        "| Scenario | Bounds | Max coverage | Missing segments | Missing mi | Missing trail groups |",
        "|---|---|---:|---:|---:|---|",
    ]
    for scenario in report["scenario_reports"]:
        groups = ", ".join(
            f"{group['trail_name']} ({group['segment_count']})"
            for group in scenario["missing_by_trail"][:12]
        )
        if len(scenario["missing_by_trail"]) > 12:
            groups += ", ..."
        lines.append(
            f"| {scenario['scenario']} | {scenario['weekday_bound_minutes']}/{scenario['weekend_bound_minutes']} | "
            f"{scenario['max_coverage_segments']}/251 | {scenario['missing_segment_count']} | "
            f"{scenario['missing_official_miles']} | {groups} |"
        )
    near_miss = report.get("near_miss_scenario")
    if near_miss:
        lines.extend(
            [
                "",
                "## Near-Miss Target: 292 Weekday / 360 Weekend",
                "",
                f"- Missing segments: {near_miss['missing_segment_count']}",
                f"- Missing official miles: {near_miss['missing_official_miles']}",
                "",
                "| Segment | Trail | Official mi | Direction | Best field-day options |",
                "|---:|---|---:|---|---|",
            ]
        )
        for row in near_miss["missing_segments"]:
            options = report["near_miss_field_day_options"].get(row["seg_id"], [])
            option_text = "; ".join(
                f"{option['day_type']} {option['p75_minutes']}/{option['p90_minutes']} min, {option['loop_count']} loops"
                for option in options[:3]
            )
            lines.append(
                f"| {row['seg_id']} | {row['trail_name']} | {row['official_miles']} | "
                f"{row['direction']} | {option_text or 'none generated'} |"
            )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-json", type=Path, default=DEFAULT_SENSITIVITY_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--neighbor-limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        sensitivity=read_json(args.sensitivity_json),
        max_combo_size=args.max_combo_size,
        neighbor_limit=args.neighbor_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    near_miss = report.get("near_miss_scenario") or {}
    print(json.dumps({
        "near_miss_scenario": near_miss.get("scenario"),
        "missing_segment_count": near_miss.get("missing_segment_count"),
        "missing_segments": [row["seg_id"] for row in near_miss.get("missing_segments", [])],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
