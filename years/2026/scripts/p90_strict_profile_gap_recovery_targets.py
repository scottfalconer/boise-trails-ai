#!/usr/bin/env python3
"""Explain why strict-profile max coverage misses each remaining segment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_joint_field_day_optimizer as joint_optimizer  # noqa: E402
import p90_repaired_candidate_universe_audit as repaired  # noqa: E402
from p90_completion_gap_analyzer import official_segments  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_OPTIMIZER_JSON = YEAR_DIR / "checkpoints" / "p90-joint-field-day-optimizer-wide-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-strict-profile-gap-recovery-targets-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def select_strict_scenario(optimizer: dict[str, Any]) -> dict[str, Any]:
    for scenario in optimizer.get("scenarios") or []:
        if scenario.get("scenario") == "strict_current_p90_bounds" and scenario.get("current_rule_compliant") is True:
            return scenario
    raise ValueError("strict_current_p90_bounds scenario not found")


def official_segment_index() -> dict[int, dict[str, Any]]:
    segment_index = official_segments(read_json(DEFAULT_OFFICIAL_GEOJSON))
    rows = {}
    for seg_id, segment in segment_index.items():
        rows[int(seg_id)] = {
            "seg_id": int(seg_id),
            "seg_name": segment["seg_name"],
            "trail_name": segment["trail_name"],
            "official_miles": round(float(segment["official_miles"]), 2),
        }
    return rows


def strict_field_day_candidates(
    *,
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
) -> list[dict[str, Any]]:
    state = read_json(repaired.DEFAULT_STATE_JSON)
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
    return joint_optimizer.generate_direct_field_day_candidates(
        loops,
        state,
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        max_combo_size=max_combo_size,
        neighbor_limit=neighbor_limit,
    )


def field_days_covering_segment(field_days: list[dict[str, Any]], seg_id: int) -> list[dict[str, Any]]:
    return sorted(
        [day for day in field_days if int(seg_id) in {int(value) for value in day.get("segment_ids") or []}],
        key=lambda day: (
            int(day.get("p75_minutes") or 0),
            int(day.get("p90_minutes") or 0),
            int(day.get("loop_count") or 0),
            str(day.get("field_day_id") or ""),
        ),
    )


def option_summary(day: dict[str, Any], selected_segment_ids: set[int], missing_segment_ids: set[int]) -> dict[str, Any]:
    day_segment_ids = {int(value) for value in day.get("segment_ids") or []}
    return {
        "field_day_id": day.get("field_day_id"),
        "day_type": day.get("day_type"),
        "p75_minutes": day.get("p75_minutes"),
        "p90_minutes": day.get("p90_minutes"),
        "p90_bound_minutes": day.get("p90_bound_minutes"),
        "stress": day.get("stress"),
        "loop_count": day.get("loop_count"),
        "on_foot_miles": day.get("on_foot_miles"),
        "covered_segment_count": len(day_segment_ids),
        "new_missing_segments_recovered": len(day_segment_ids & missing_segment_ids),
        "already_selected_segments_repeated": len(day_segment_ids & selected_segment_ids),
        "loop_ids": day.get("loop_ids") or [],
    }


def classify_missing_segment(
    seg_id: int,
    *,
    official_row: dict[str, Any],
    field_days: list[dict[str, Any]],
    selected_segment_ids: set[int],
    missing_segment_ids: set[int] | None = None,
) -> dict[str, Any]:
    missing_segment_ids = missing_segment_ids or {int(seg_id)}
    options = field_days_covering_segment(field_days, seg_id)
    if not options:
        classification = "no_strict_field_day_candidate"
        best_option = None
        alternative_options = []
    else:
        classification = "strict_candidate_exists_but_not_selected"
        best_option = option_summary(options[0], selected_segment_ids, missing_segment_ids)
        alternative_options = [
            option_summary(day, selected_segment_ids, missing_segment_ids)
            for day in options[1:4]
        ]
    return {
        **official_row,
        "classification": classification,
        "candidate_option_count": len(options),
        "best_option": best_option,
        "alternative_options": alternative_options,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    return {
        "missing_segment_count": len(rows),
        "missing_official_miles": round(sum(float(row.get("official_miles") or 0) for row in rows), 2),
        "classification_counts": dict(sorted(counts.items())),
    }


def build_report(optimizer: dict[str, Any]) -> dict[str, Any]:
    scenario = select_strict_scenario(optimizer)
    max_solution = scenario["max_coverage_solution"]
    missing_ids = [int(seg_id) for seg_id in max_solution.get("missing_segment_ids") or []]
    selected_segment_ids = {
        int(seg_id)
        for day in max_solution.get("selected_field_days") or []
        for seg_id in day.get("segment_ids") or []
    }
    official_index = official_segment_index()
    field_days = strict_field_day_candidates(
        weekday_bound=int(scenario["weekday_bound_minutes"]),
        weekend_bound=int(scenario["weekend_bound_minutes"]),
        max_combo_size=int((optimizer.get("config") or {}).get("max_combo_size") or 3),
        neighbor_limit=int((optimizer.get("config") or {}).get("neighbor_limit") or 40),
    )
    rows = [
        classify_missing_segment(
            seg_id,
            official_row=official_index[seg_id],
            field_days=field_days,
            selected_segment_ids=selected_segment_ids,
            missing_segment_ids=set(missing_ids),
        )
        for seg_id in missing_ids
    ]
    summary = summarize_rows(rows)
    return {
        "objective": "classify strict-profile max-coverage missing segments by recovery path",
        "source_files": {
            "joint_optimizer_json": display_path(DEFAULT_OPTIMIZER_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
        },
        "profile": {
            "scenario": scenario["scenario"],
            "weekday_bound_minutes": scenario["weekday_bound_minutes"],
            "weekend_bound_minutes": scenario["weekend_bound_minutes"],
            "current_rule_compliant": scenario["current_rule_compliant"],
        },
        "summary": {
            **summary,
            "field_day_candidate_count": len(field_days),
            "selected_covered_segment_count": max_solution.get("covered_segment_count"),
            "selected_missing_segment_count": max_solution.get("missing_segment_count"),
        },
        "rows": rows,
        "interpretation": [
            "no_strict_field_day_candidate means route/access/time redesign is needed before that segment can appear under current bounds.",
            "strict_candidate_exists_but_not_selected means the segment is runnable under current bounds but loses a 31-day coverage tradeoff.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    profile = report["profile"]
    lines = [
        "# Strict Profile Gap Recovery Targets",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Scenario: `{profile['scenario']}`",
        f"- Bounds: {profile['weekday_bound_minutes']} weekday / {profile['weekend_bound_minutes']} weekend",
        f"- Missing segments: {summary['missing_segment_count']}",
        f"- Missing official miles: {summary['missing_official_miles']}",
        f"- Field-day candidates inspected: {summary['field_day_candidate_count']}",
        f"- Classification counts: `{summary['classification_counts']}`",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.extend(
        [
            "",
            "## Missing Segment Recovery Rows",
            "",
            "| Segment | Trail | Official mi | Classification | Options | Best option |",
            "|---:|---|---:|---|---:|---|",
        ]
    )
    for row in report["rows"]:
        best = row.get("best_option") or {}
        best_text = ""
        if best:
            best_text = (
                f"{best.get('day_type')} {best.get('p75_minutes')}/{best.get('p90_minutes')} min, "
                f"recovers {best.get('new_missing_segments_recovered')} missing segs"
            )
        lines.append(
            f"| {row['seg_id']} | {row['seg_name']} | {row['official_miles']} | "
            f"{row['classification']} | {row['candidate_option_count']} | {best_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-json", type=Path, default=DEFAULT_OPTIMIZER_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(read_json(args.optimizer_json))
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
