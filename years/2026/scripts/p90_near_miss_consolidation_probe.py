#!/usr/bin/env python3
"""Find route-consolidation targets for the 292/360 near-miss scenario."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_joint_field_day_optimizer as joint_optimizer  # noqa: E402
from personal_route_planner import read_json  # noqa: E402


DEFAULT_PRESSURE_JSON = YEAR_DIR / "checkpoints" / "p90-near-miss-pressure-audit-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-near-miss-consolidation-probe-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def short_day(day: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_day_id": day["field_day_id"],
        "day_type": day["day_type"],
        "p75_minutes": day["p75_minutes"],
        "p90_minutes": day["p90_minutes"],
        "loop_count": day["loop_count"],
        "segment_count": len(day["segment_ids"]),
        "loop_ids": day["loop_ids"],
    }


def close_weekend_only_days(
    selected: list[dict[str, Any]],
    *,
    weekday_bound: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for day in selected:
        if day["day_type"] != "weekend":
            continue
        excess = int(day["p90_minutes"]) - weekday_bound
        if excess <= 0:
            continue
        row = short_day(day)
        row["minutes_over_weekday_bound"] = excess
        rows.append(row)
    return sorted(rows, key=lambda row: (row["minutes_over_weekday_bound"], row["p90_minutes"]))[:limit]


def build_loop_context(
    *,
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int], set[tuple[str, frozenset[str]]]]:
    state = read_json(joint_optimizer.repaired.DEFAULT_STATE_JSON)
    _target_ids, candidates, field_menu = joint_optimizer.all_repaired_candidates()
    max_bound = max(weekday_bound, weekend_bound)
    bounded = [
        candidate
        for candidate in candidates
        if int(candidate["door_to_door_p90_minutes"]) <= max_bound
        and candidate.get("validation_passed") is True
        and candidate.get("manual_design_hold") is not True
    ]
    loops, _missing_coordinates = joint_optimizer.candidate_loop_records(bounded, field_menu, state)
    loop_index = {loop["loop_id"]: index for index, loop in enumerate(loops)}
    generated = joint_optimizer.generate_direct_field_day_candidates(
        loops,
        state,
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        max_combo_size=max_combo_size,
        neighbor_limit=neighbor_limit,
    )
    return loops, loop_index, {
        (str(day["day_type"]), frozenset(str(loop_id) for loop_id in day["loop_ids"]))
        for day in generated
    }


def selected_weekday_pair_consolidations(
    *,
    selected: list[dict[str, Any]],
    loops: list[dict[str, Any]],
    loop_index: dict[str, int],
    generated_field_day_signatures: set[tuple[str, frozenset[str]]],
    weekday_bound: int,
    max_combined_loops: int,
) -> list[dict[str, Any]]:
    state = read_json(joint_optimizer.repaired.DEFAULT_STATE_JSON)
    weekday_days = [day for day in selected if day["day_type"] == "weekday"]
    rows = []
    for left, right in itertools.combinations(weekday_days, 2):
        loop_ids = list(dict.fromkeys(left["loop_ids"] + right["loop_ids"]))
        if len(loop_ids) > max_combined_loops:
            continue
        if any(loop_id not in loop_index for loop_id in loop_ids):
            continue
        combo = tuple(loop_index[loop_id] for loop_id in loop_ids)
        combined = joint_optimizer.pack_audit.best_field_day_for_combo(loops, combo, state)
        if int(combined["p90_minutes"]) > weekday_bound:
            continue
        field_day_id = "weekday-" + "--".join(combined["order"])
        signature = ("weekday", frozenset(str(loop_id) for loop_id in combined["order"]))
        rows.append(
            {
                "combined_field_day_id": field_day_id,
                "already_generated": signature in generated_field_day_signatures,
                "p75_minutes": combined["p75_minutes"],
                "p90_minutes": combined["p90_minutes"],
                "p75_minutes_saved": int(left["p75_minutes"]) + int(right["p75_minutes"]) - int(combined["p75_minutes"]),
                "days_saved": 1,
                "combined_loop_count": len(loop_ids),
                "left": short_day(left),
                "right": short_day(right),
            }
        )
    return sorted(rows, key=lambda row: (row["p90_minutes"], row["p75_minutes"]))


def build_report(
    *,
    pressure: dict[str, Any],
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
    max_combined_loops: int,
    limit: int,
) -> dict[str, Any]:
    selected = (pressure["full_cover_pressure"]["min_weekdays_with_actual_weekends"] or {}).get("selected_field_days") or []
    loops, loop_index, generated_signatures = build_loop_context(
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        max_combo_size=max_combo_size,
        neighbor_limit=neighbor_limit,
    )
    pair_consolidations = selected_weekday_pair_consolidations(
        selected=selected,
        loops=loops,
        loop_index=loop_index,
        generated_field_day_signatures=generated_signatures,
        weekday_bound=weekday_bound,
        max_combined_loops=max_combined_loops,
    )
    close_weekends = close_weekend_only_days(selected, weekday_bound=weekday_bound, limit=limit)
    return {
        "objective": "find concrete consolidation targets for the 292/360 near-miss",
        "source_files": {
            "pressure_json": display_path(DEFAULT_PRESSURE_JSON),
            "joint_optimizer": display_path(YEAR_DIR / "scripts" / "p90_joint_field_day_optimizer.py"),
        },
        "config": {
            "weekday_bound_minutes": weekday_bound,
            "weekend_bound_minutes": weekend_bound,
            "max_combo_size": max_combo_size,
            "neighbor_limit": neighbor_limit,
            "max_combined_loops": max_combined_loops,
        },
        "starting_pressure": {
            "min_weekdays_with_9_weekends": pressure["full_cover_pressure"]["min_weekdays_with_actual_weekends"].get("weekday_field_day_count"),
            "available_weekdays": pressure["available_day_counts"]["weekday"],
            "weekdays_to_save": pressure["full_cover_pressure"].get("extra_weekdays_needed_if_weekends_fixed"),
        },
        "weekday_pair_consolidations_under_bound": pair_consolidations[:limit],
        "weekday_pair_consolidation_count": len(pair_consolidations),
        "close_weekend_only_redesign_targets": close_weekends,
        "interpretation": [
            "Simple selected-weekday pair consolidation can save at most one weekday here because every under-bound pair found shares the Shane's Trail singleton.",
            "The closest weekend-only redesign target is the Upper 8th / Corrals / Sidewinder block, which is only slightly over the 292-minute weekday bound in this model.",
            "The generated candidate universe likely needs a targeted new combo plus one manual time reduction or larger availability window to turn the near-miss into a full-cover plan.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P90 Near-Miss Consolidation Probe",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Starting Pressure",
        "",
        f"- Full cover with 9 weekends fixed needs {report['starting_pressure']['min_weekdays_with_9_weekends']} weekdays.",
        f"- Available weekdays: {report['starting_pressure']['available_weekdays']}",
        f"- Weekdays to save: {report['starting_pressure']['weekdays_to_save']}",
        "",
        "## Weekday Pair Consolidations Under 292 Min P90",
        "",
        "| Combined p90 | Combined p75 | Saved p75 | Already generated | Left | Right |",
        "|---:|---:|---:|---|---|---|",
    ]
    for row in report["weekday_pair_consolidations_under_bound"]:
        lines.append(
            f"| {row['p90_minutes']} | {row['p75_minutes']} | {row['p75_minutes_saved']} | "
            f"{row['already_generated']} | {row['left']['field_day_id']} | {row['right']['field_day_id']} |"
        )
    if not report["weekday_pair_consolidations_under_bound"]:
        lines.append("| | | | | none | |")
    lines.extend(
        [
            "",
            "## Closest Weekend-Only Redesign Targets",
            "",
            "| P90 | Minutes over weekday bound | P75 | Segments | Field day |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["close_weekend_only_redesign_targets"]:
        lines.append(
            f"| {row['p90_minutes']} | {row['minutes_over_weekday_bound']} | "
            f"{row['p75_minutes']} | {row['segment_count']} | {row['field_day_id']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-json", type=Path, default=DEFAULT_PRESSURE_JSON)
    parser.add_argument("--weekday-bound", type=int, default=292)
    parser.add_argument("--weekend-bound", type=int, default=360)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--neighbor-limit", type=int, default=20)
    parser.add_argument("--max-combined-loops", type=int, default=6)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        pressure=read_json(args.pressure_json),
        weekday_bound=args.weekday_bound,
        weekend_bound=args.weekend_bound,
        max_combo_size=args.max_combo_size,
        neighbor_limit=args.neighbor_limit,
        max_combined_loops=args.max_combined_loops,
        limit=args.limit,
    )
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        json.dumps(
            {
                "weekday_pair_consolidation_count": report["weekday_pair_consolidation_count"],
                "best_weekday_pair_p90": (
                    report["weekday_pair_consolidations_under_bound"][0]["p90_minutes"]
                    if report["weekday_pair_consolidations_under_bound"]
                    else None
                ),
                "closest_weekend_only_minutes_over": (
                    report["close_weekend_only_redesign_targets"][0]["minutes_over_weekday_bound"]
                    if report["close_weekend_only_redesign_targets"]
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
