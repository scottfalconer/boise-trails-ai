#!/usr/bin/env python3
"""Export the relaxed-drive feasible sensitivity solution as a draft field-day plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_joint_field_day_optimizer as joint_optimizer  # noqa: E402
from p90_completion_gap_analyzer import official_segments  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_INPUT_JSON = YEAR_DIR / "checkpoints" / "p90-near-miss-pressure-audit-drive45-n40-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-relaxed-drive-draft-field-day-plan-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def candidate_loop_id(candidate: dict[str, Any]) -> str:
    return "::".join(joint_optimizer.pack_audit.candidate_key(candidate))


def build_loop_indexes(
    *,
    weekday_bound: int,
    weekend_bound: int,
    inter_trailhead_drive_minutes: int | None,
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    state = read_json(joint_optimizer.repaired.DEFAULT_STATE_JSON)
    if inter_trailhead_drive_minutes is not None:
        state.setdefault("availability_model", {})["acceptable_inter_trailhead_drive_minutes"] = int(inter_trailhead_drive_minutes)
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
    candidate_by_loop_id = {candidate_loop_id(candidate): candidate for candidate in bounded}
    loop_by_id = {}
    for loop in loops:
        candidate = candidate_by_loop_id.get(loop["loop_id"], {})
        loop_by_id[loop["loop_id"]] = {
            **loop,
            "validation_passed": candidate.get("validation_passed"),
            "manual_design_hold": candidate.get("manual_design_hold"),
            "route_status": candidate.get("route_status"),
            "trail_names": candidate.get("trail_names") or [],
        }
    official_index = official_segments(read_json(DEFAULT_OFFICIAL_GEOJSON))
    return loop_by_id, official_index


def summarize_segments(segment_ids: list[int], official_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(set(int(seg_id) for seg_id in segment_ids))
    return {
        "segment_count": len(ids),
        "official_miles": round(sum(float(official_index[seg_id]["official_miles"]) for seg_id in ids if seg_id in official_index), 2),
        "segment_ids": ids,
    }


def enrich_day(
    day_number: int,
    day: dict[str, Any],
    *,
    loop_by_id: dict[str, dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    loops = []
    for loop_id in day["loop_ids"]:
        loop = loop_by_id.get(loop_id)
        if not loop:
            loops.append({"loop_id": loop_id, "missing_loop_metadata": True})
            continue
        loops.append(
            {
                "loop_id": loop_id,
                "label": loop.get("label"),
                "source": loop.get("source"),
                "candidate_id": loop.get("candidate_id"),
                "trailhead": loop.get("trailhead"),
                "trail_names": loop.get("trail_names") or [],
                "segment_count": len(loop.get("segment_ids") or []),
                "official_miles": loop.get("official_miles"),
                "on_foot_miles": loop.get("on_foot_miles"),
                "grade_adjusted_miles": loop.get("grade_adjusted_miles"),
                "ascent_ft": loop.get("ascent_ft"),
                "p75_minutes": loop.get("door_to_door_p75_minutes"),
                "p90_minutes": loop.get("door_to_door_p90_minutes"),
                "validation_passed": loop.get("validation_passed"),
                "manual_design_hold": loop.get("manual_design_hold"),
                "parking_confidence": loop.get("parking_confidence"),
            }
        )
    segment_summary = summarize_segments(day["segment_ids"], official_index)
    return {
        "draft_day_number": day_number,
        "field_day_id": day["field_day_id"],
        "day_type": day["day_type"],
        "p75_minutes": day["p75_minutes"],
        "p90_minutes": day["p90_minutes"],
        "p90_bound_minutes": day["p90_bound_minutes"],
        "stress": day["stress"],
        "drive_minutes": day["drive_minutes"],
        "between_drive_minutes": day["between_drive_minutes"],
        "loop_count": day["loop_count"],
        "on_foot_miles": day["on_foot_miles"],
        "grade_adjusted_miles": day["grade_adjusted_miles"],
        "parking_risk": day["parking_risk"],
        "segment_summary": segment_summary,
        "loops": loops,
    }


def build_report(solution_json: dict[str, Any]) -> dict[str, Any]:
    config = solution_json.get("config") or {}
    weekday_bound = int(config.get("weekday_bound_minutes") or 292)
    weekend_bound = int(config.get("weekend_bound_minutes") or 360)
    inter_drive = config.get("inter_trailhead_drive_minutes")
    loop_by_id, official_index = build_loop_indexes(
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        inter_trailhead_drive_minutes=inter_drive,
    )
    solution = (
        solution_json.get("p75_min_full_cover")
        or solution_json.get("actual_day_count_full_cover")
        or {}
    )
    selected = solution.get("selected_field_days") or []
    days = [
        enrich_day(index + 1, day, loop_by_id=loop_by_id, official_index=official_index)
        for index, day in enumerate(sorted(selected, key=lambda item: (item["day_type"], item["p90_minutes"], item["field_day_id"])))
    ]
    covered_ids = sorted({seg_id for day in days for seg_id in day["segment_summary"]["segment_ids"]})
    target_ids = sorted(official_index)
    missing_ids = sorted(set(target_ids) - set(covered_ids))
    loops = [loop for day in days for loop in day["loops"]]
    return {
        "objective": "draft a reviewable field-day plan from the relaxed-drive 292/360 sensitivity solution",
        "source_files": {
            "solution_json": display_path(DEFAULT_INPUT_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
        },
        "config": config,
        "status": "draft_needs_human_review",
        "solution_source": "p75_min_full_cover" if solution_json.get("p75_min_full_cover") else "actual_day_count_full_cover",
        "coverage": {
            "official_segment_count": len(target_ids),
            "covered_segment_count": len(covered_ids),
            "missing_segment_count": len(missing_ids),
            "missing_segment_ids": missing_ids,
            "official_miles": round(sum(float(official_index[seg_id]["official_miles"]) for seg_id in target_ids), 2),
            "covered_official_miles": round(sum(float(official_index[seg_id]["official_miles"]) for seg_id in covered_ids), 2),
        },
        "time_and_logistics": {
            "field_day_count": len(days),
            "weekday_field_day_count": sum(1 for day in days if day["day_type"] == "weekday"),
            "weekend_field_day_count": sum(1 for day in days if day["day_type"] == "weekend"),
            "total_p75_minutes": int(sum(day["p75_minutes"] for day in days)),
            "max_p90_minutes": int(max(day["p90_minutes"] for day in days)) if days else 0,
            "days_over_p90_bound": [
                day["draft_day_number"]
                for day in days
                if int(day["p90_minutes"]) > int(day["p90_bound_minutes"])
            ],
            "multi_start_days": sum(1 for day in days if int(day["loop_count"]) > 1),
            "total_between_drive_minutes": int(sum(day["between_drive_minutes"] for day in days)),
            "max_between_drive_minutes": int(max(day["between_drive_minutes"] for day in days)) if days else 0,
        },
        "loop_validation_summary": {
            "loop_count": len(loops),
            "all_loop_metadata_found": all(not loop.get("missing_loop_metadata") for loop in loops),
            "all_loops_validation_passed": all(loop.get("validation_passed") is True for loop in loops if not loop.get("missing_loop_metadata")),
            "manual_design_hold_loop_count": sum(1 for loop in loops if loop.get("manual_design_hold") is True),
            "missing_loop_metadata_count": sum(1 for loop in loops if loop.get("missing_loop_metadata")),
        },
        "known_gaps": [
            "This is a set of draft field days, not a dated calendar assignment.",
            "Multi-loop field-day GPX files have not been exported and validated as day-level GPX.",
            "Lower Hulls odd/even date placement is not checked because dates are not assigned here.",
            "The plan uses 292 weekday / 360 weekend p90 bounds and a 45-minute inter-trailhead drive sensitivity, not the stricter current 260 / 180 committed defaults.",
            "The selected multi-start days need human review for outing quality before promotion to the phone field packet.",
        ],
        "field_days": days,
    }


def render_md(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    logistics = report["time_and_logistics"]
    validation = report["loop_validation_summary"]
    lines = [
        "# P90 Relaxed-Drive Draft Field-Day Plan",
        "",
        f"Objective: {report['objective']}",
        "",
        f"Status: `{report['status']}`",
        f"Solution source: `{report['solution_source']}`",
        "",
        "## Summary",
        "",
        f"- Coverage: {coverage['covered_segment_count']}/{coverage['official_segment_count']} official segments, {coverage['covered_official_miles']}/{coverage['official_miles']} official mi",
        f"- Field days: {logistics['field_day_count']} ({logistics['weekday_field_day_count']} weekday / {logistics['weekend_field_day_count']} weekend)",
        f"- Total p75: {logistics['total_p75_minutes']} min",
        f"- Max p90: {logistics['max_p90_minutes']} min",
        f"- Days over p90 bound: {len(logistics['days_over_p90_bound'])}",
        f"- Multi-start days: {logistics['multi_start_days']}",
        f"- Total between-start drive: {logistics['total_between_drive_minutes']} min",
        f"- Loop metadata found: {validation['all_loop_metadata_found']}",
        f"- Loop validation passed: {validation['all_loops_validation_passed']}",
        f"- Manual-design-hold loops: {validation['manual_design_hold_loop_count']}",
        "",
        "## Known Gaps",
        "",
    ]
    lines.extend(f"- {gap}" for gap in report["known_gaps"])
    lines.extend(
        [
            "",
            "## Draft Field Days",
            "",
            "| # | Type | P75 | P90 | Bound | Starts | Between drive | Official mi | On foot | Primary loops |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for day in report["field_days"]:
        loop_labels = "; ".join(
            str(loop.get("label") or loop.get("candidate_id") or loop.get("loop_id"))
            for loop in day["loops"]
        )
        lines.append(
            f"| {day['draft_day_number']} | {day['day_type']} | {day['p75_minutes']} | {day['p90_minutes']} | "
            f"{day['p90_bound_minutes']} | {day['loop_count']} | {day['between_drive_minutes']} | "
            f"{day['segment_summary']['official_miles']} | {day['on_foot_miles']} | {loop_labels} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(read_json(args.input_json))
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        json.dumps(
            {
                "covered_segments": report["coverage"]["covered_segment_count"],
                "field_days": report["time_and_logistics"]["field_day_count"],
                "days_over_p90_bound": len(report["time_and_logistics"]["days_over_p90_bound"]),
                "all_loops_validation_passed": report["loop_validation_summary"]["all_loops_validation_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
