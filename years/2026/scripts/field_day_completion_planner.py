#!/usr/bin/env python3
"""Validate and pack runnable loops into home-to-home field days."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, haversine_miles, read_json  # noqa: E402


DEFAULT_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_MANIFEST_JSON = REPO_ROOT / "docs" / "field-packet" / "manifest.json"
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "field-day-completion-plan-2026-05-06"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def official_target_ids(official_geojson: dict[str, Any]) -> list[int]:
    return sorted(
        int((feature.get("properties") or {})["segId"])
        for feature in official_geojson.get("features", [])
        if (feature.get("properties") or {}).get("segId") is not None
    )


def date_counts(start: str, end: str) -> dict[str, int]:
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    counts = {"weekday": 0, "weekend": 0, "total": 0}
    while current <= last:
        counts["weekend" if current.weekday() >= 5 else "weekday"] += 1
        counts["total"] += 1
        current += timedelta(days=1)
    return counts


def drive_minutes_between(
    left: tuple[float, float],
    right: tuple[float, float],
    drive_model: dict[str, Any],
    *,
    apply_minimum: bool = False,
) -> int:
    if left == right:
        return 0
    straight_line = haversine_miles(left, right)
    minutes = straight_line * float(drive_model.get("straight_line_factor") or 1.25) * float(
        drive_model.get("minutes_per_mile") or 2.2
    )
    if apply_minimum and straight_line > 0.05:
        minutes = max(minutes, float(drive_model.get("minimum_one_way_minutes") or 5))
    return int(math.ceil(minutes))


def manifest_by_candidate_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for route in manifest.get("routes") or []:
        for candidate_id in ((route.get("outing") or {}).get("candidate_ids") or []):
            lookup[str(candidate_id)] = route
    return lookup


def parking_risk_score(trailhead: dict[str, Any]) -> int:
    confidence = str(trailhead.get("parking_confidence") or "").lower()
    if trailhead.get("has_parking") is not True:
        return 3
    if "validated" in confidence or "strava_reused" in confidence:
        return 0
    if "strava_seen" in confidence or "osm_amenity" in confidence:
        return 1
    return 2


def loop_records(
    map_data: dict[str, Any],
    manifest: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    cues = map_data.get("route_cues") or {}
    manifest_lookup = manifest_by_candidate_id(manifest)
    drive_model = state.get("drive_model") or {}
    home = (float(drive_model["origin_lon"]), float(drive_model["origin_lat"]))
    loops = []
    for candidate_id, cue in sorted(cues.items()):
        manifest_route = manifest_lookup.get(candidate_id)
        trailhead = cue.get("trailhead") or {}
        parking = (float(trailhead["lon"]), float(trailhead["lat"]))
        time_estimates = cue.get("time_estimates_minutes") or {}
        p75 = int(round(float(time_estimates.get("door_to_door_p75") or cue.get("total_minutes") or 0)))
        p90 = int(round(float(time_estimates.get("door_to_door_p90") or max(p75, p75 * 1.12))))
        home_drive = drive_minutes_between(home, parking, drive_model, apply_minimum=True)
        parking_minutes = int(trailhead.get("parking_minutes") or state.get("parking_minutes") or 8)
        moving_p75 = time_estimates.get("moving_effort_p75")
        route_finding = int(round(float(time_estimates.get("route_finding_penalty") or 0)))
        if moving_p75 is not None:
            internal_p75 = parking_minutes + int(round(float(moving_p75))) + route_finding
        else:
            internal_p75 = max(0, p75 - 2 * home_drive)
        internal_p90 = internal_p75 + max(0, p90 - p75)
        validation = (manifest_route or {}).get("validation") or {}
        outing = (manifest_route or {}).get("outing") or {}
        segment_ids = [int(seg_id) for seg_id in (outing.get("segment_ids") or [seg.get("seg_id") for seg in cue.get("segments") or []]) if seg_id is not None]
        loops.append(
            {
                "candidate_id": candidate_id,
                "label": outing.get("label") or cue.get("label") or candidate_id,
                "title": cue.get("title"),
                "trailhead": trailhead.get("name"),
                "parking": {"lon": parking[0], "lat": parking[1]},
                "segment_ids": sorted(set(segment_ids)),
                "official_miles": float(cue.get("official_miles") or outing.get("official_miles") or 0),
                "on_foot_miles": float(cue.get("on_foot_miles") or outing.get("on_foot_miles") or 0),
                "ascent_ft": int(round(float(cue.get("ascent_ft") or 0))),
                "grade_adjusted_miles": float(cue.get("grade_adjusted_miles") or 0),
                "door_to_door_p75_minutes": p75,
                "door_to_door_p90_minutes": p90,
                "home_drive_minutes": home_drive,
                "internal_p75_minutes": internal_p75,
                "internal_p90_minutes": internal_p90,
                "p90_delta_minutes": max(0, p90 - p75),
                "parking_risk": parking_risk_score(trailhead),
                "route_status": cue.get("route_status"),
                "manifest_validation_passed": validation.get("passed") is True,
                "manual_design_hold": bool(outing.get("manual_design_hold")),
                "gpx_href": (manifest_route or {}).get("gpx_href"),
            }
        )
    return loops


def field_day_time(
    loops: list[dict[str, Any]],
    order: tuple[int, ...],
    drive_model: dict[str, Any],
    home: tuple[float, float],
) -> dict[str, Any]:
    ordered = [loops[index] for index in order]
    first_parking = (float(ordered[0]["parking"]["lon"]), float(ordered[0]["parking"]["lat"]))
    drive = drive_minutes_between(home, first_parking, drive_model, apply_minimum=True)
    between_drive = 0
    for left, right in zip(ordered, ordered[1:]):
        left_parking = (float(left["parking"]["lon"]), float(left["parking"]["lat"]))
        right_parking = (float(right["parking"]["lon"]), float(right["parking"]["lat"]))
        minutes = drive_minutes_between(
            left_parking,
            right_parking,
            drive_model,
            apply_minimum=False,
        )
        between_drive += minutes
        drive += minutes
    last_parking = (float(ordered[-1]["parking"]["lon"]), float(ordered[-1]["parking"]["lat"]))
    drive += drive_minutes_between(last_parking, home, drive_model, apply_minimum=True)
    internal_p75 = sum(int(loop["internal_p75_minutes"]) for loop in ordered)
    p75 = drive + internal_p75
    p90 = p75 + sum(int(loop["p90_delta_minutes"]) for loop in ordered)
    return {
        "order": [loop["candidate_id"] for loop in ordered],
        "drive_minutes": drive,
        "between_drive_minutes": between_drive,
        "p75_minutes": int(p75),
        "p90_minutes": int(p90),
        "on_foot_miles": round(sum(float(loop["on_foot_miles"]) for loop in ordered), 2),
        "grade_adjusted_miles": round(sum(float(loop["grade_adjusted_miles"] or 0) for loop in ordered), 2),
        "ascent_ft": int(sum(int(loop["ascent_ft"] or 0) for loop in ordered)),
        "parking_risk": int(sum(int(loop["parking_risk"]) for loop in ordered)),
    }


def best_field_day_for_combo(
    loops: list[dict[str, Any]],
    combo: tuple[int, ...],
    drive_model: dict[str, Any],
    home: tuple[float, float],
) -> dict[str, Any]:
    best = None
    for order in itertools.permutations(combo):
        result = field_day_time(loops, order, drive_model, home)
        if best is None or (
            result["p75_minutes"],
            result["p90_minutes"],
            result["between_drive_minutes"],
        ) < (
            best["p75_minutes"],
            best["p90_minutes"],
            best["between_drive_minutes"],
        ):
            best = result
    assert best is not None
    return best


def generate_field_day_candidates(
    loops: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    max_combo_size: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drive_model = state.get("drive_model") or {}
    home = (float(drive_model["origin_lon"]), float(drive_model["origin_lat"]))
    availability = state.get("availability_model") or {}
    bounds = {
        "weekday": int(availability.get("weekday_max_minutes") or 0),
        "weekend": int(availability.get("weekend_max_minutes") or 0),
    }
    blockers = []
    candidates = []
    max_bound = max(bounds.values())
    for loop in loops:
        if loop["door_to_door_p90_minutes"] > max_bound:
            blockers.append(
                {
                    "candidate_id": loop["candidate_id"],
                    "label": loop["label"],
                    "trailhead": loop["trailhead"],
                    "p90_minutes": loop["door_to_door_p90_minutes"],
                    "max_available_bound_minutes": max_bound,
                    "segment_count": len(loop["segment_ids"]),
                    "official_miles": loop["official_miles"],
                    "on_foot_miles": loop["on_foot_miles"],
                    "reason": "single_loop_exceeds_all_p90_daily_bounds",
                }
            )
    for size in range(1, max_combo_size + 1):
        for combo in itertools.combinations(range(len(loops)), size):
            day = best_field_day_for_combo(loops, combo, drive_model, home)
            for day_type, bound in bounds.items():
                if not bound or day["p90_minutes"] > bound:
                    continue
                stress = day["p90_minutes"] / bound
                candidates.append(
                    {
                        "field_day_id": f"{day_type}-" + "-".join(day["order"]),
                        "day_type": day_type,
                        "candidate_ids": day["order"],
                        "loop_count": len(day["order"]),
                        "p75_minutes": day["p75_minutes"],
                        "p90_minutes": day["p90_minutes"],
                        "p90_bound_minutes": bound,
                        "stress": round(stress, 3),
                        "drive_minutes": day["drive_minutes"],
                        "between_drive_minutes": day["between_drive_minutes"],
                        "on_foot_miles": day["on_foot_miles"],
                        "grade_adjusted_miles": day["grade_adjusted_miles"],
                        "ascent_ft": day["ascent_ft"],
                        "parking_risk": day["parking_risk"],
                    }
                )
    return candidates, blockers


def solve_field_day_partition(
    loops: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    date_type_counts: dict[str, int],
) -> dict[str, Any]:
    if not candidates:
        return {"success": False, "reason": "no_field_day_candidates"}
    loop_ids = [loop["candidate_id"] for loop in loops]
    loop_to_row = {candidate_id: index for index, candidate_id in enumerate(loop_ids)}
    matrix = lil_matrix((len(loop_ids) + 2, len(candidates)), dtype=float)
    lower = [1.0] * len(loop_ids) + [0.0, 0.0]
    upper = [1.0] * len(loop_ids) + [float(date_type_counts["weekday"]), float(date_type_counts["weekend"])]
    for col, candidate in enumerate(candidates):
        for candidate_id in candidate["candidate_ids"]:
            matrix[loop_to_row[candidate_id], col] = 1.0
        matrix[len(loop_ids), col] = 1.0 if candidate["day_type"] == "weekday" else 0.0
        matrix[len(loop_ids) + 1, col] = 1.0 if candidate["day_type"] == "weekend" else 0.0
    costs = np.array(
        [
            float(candidate["p75_minutes"])
            + float(candidate["stress"]) * 0.01
            + float(candidate["grade_adjusted_miles"]) * 0.001
            + float(candidate["on_foot_miles"]) * 0.0001
            + float(candidate["parking_risk"]) * 0.00001
            + 0.000001
            for candidate in candidates
        ],
        dtype=float,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(matrix.tocsr(), lb=np.array(lower), ub=np.array(upper)),
        options={"time_limit": 60},
    )
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}
    selected = [candidate for candidate, value in zip(candidates, result.x) if value >= 0.5]
    return {
        "success": True,
        "field_day_count": len(selected),
        "weekday_field_day_count": sum(1 for day in selected if day["day_type"] == "weekday"),
        "weekend_field_day_count": sum(1 for day in selected if day["day_type"] == "weekend"),
        "total_p75_minutes": int(sum(day["p75_minutes"] for day in selected)),
        "max_p90_stress": round(max(day["stress"] for day in selected), 3) if selected else 0,
        "total_on_foot_miles": round(sum(day["on_foot_miles"] for day in selected), 2),
        "total_grade_adjusted_miles": round(sum(day["grade_adjusted_miles"] for day in selected), 2),
        "total_parking_risk": int(sum(day["parking_risk"] for day in selected)),
        "field_days": sorted(selected, key=lambda day: (day["day_type"], day["p75_minutes"], day["field_day_id"])),
    }


def build_report(
    map_data: dict[str, Any],
    manifest: dict[str, Any],
    official_geojson: dict[str, Any],
    state: dict[str, Any],
    *,
    max_combo_size: int = 3,
) -> dict[str, Any]:
    target_ids = set(official_target_ids(official_geojson))
    loops = loop_records(map_data, manifest, state)
    covered_ids = set().union(*(set(loop["segment_ids"]) for loop in loops)) if loops else set()
    availability = state.get("availability_model") or {}
    date_window = availability.get("available_dates") or {}
    counts = date_counts(date_window["start"], date_window["end"])
    field_day_candidates, oversized_loop_blockers = generate_field_day_candidates(
        loops,
        state,
        max_combo_size=max_combo_size,
    )
    invalid_loops = [
        loop
        for loop in loops
        if not loop["manifest_validation_passed"] or loop["manual_design_hold"] or loop["route_status"] == "draft"
    ]
    if oversized_loop_blockers or invalid_loops or target_ids - covered_ids:
        solution = {
            "success": False,
            "reason": "field_day_feasibility_precheck_failed",
        }
    else:
        solution = solve_field_day_partition(loops, field_day_candidates, counts)
    feasible = (
        solution.get("success") is True
        and not oversized_loop_blockers
        and not invalid_loops
        and not (target_ids - covered_ids)
    )
    return {
        "objective": "pack runnable single-car loops into home-to-home field days under p90 personal daily bounds",
        "source_files": {
            "map_data_json": display_path(DEFAULT_MAP_DATA_JSON),
            "manifest_json": display_path(DEFAULT_MANIFEST_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "state_json": display_path(DEFAULT_STATE_JSON),
        },
        "constraints": {
            "weekday_max_minutes": availability.get("weekday_max_minutes"),
            "weekend_max_minutes": availability.get("weekend_max_minutes"),
            "date_window": date_window,
            "date_type_counts": counts,
            "max_combo_size": max_combo_size,
        },
        "summary": {
            "feasible": feasible,
            "target_segment_count": len(target_ids),
            "covered_segment_count": len(covered_ids & target_ids),
            "missing_segment_count": len(target_ids - covered_ids),
            "loop_count": len(loops),
            "invalid_loop_count": len(invalid_loops),
            "oversized_loop_count": len(oversized_loop_blockers),
            "field_day_candidate_count": len(field_day_candidates),
            "solver_success": solution.get("success") is True,
        },
        "solution": solution,
        "oversized_loop_blockers": oversized_loop_blockers,
        "invalid_loops": [
            {
                "candidate_id": loop["candidate_id"],
                "label": loop["label"],
                "manifest_validation_passed": loop["manifest_validation_passed"],
                "manual_design_hold": loop["manual_design_hold"],
                "route_status": loop["route_status"],
            }
            for loop in invalid_loops
        ],
        "missing_segment_ids": sorted(target_ids - covered_ids),
        "loops": loops,
        "caveats": [
            "This planner uses the current field-menu loops as atomic runnable loops; if a loop exceeds p90 bounds, it must be split/redesigned before a strict feasible schedule can exist.",
            "Between-trailhead drive is estimated from the private straight-line drive model, not OSRM.",
            "Date-specific rules such as Lower Hulls even-day are not yet assigned to exact dates when the precheck is infeasible.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    constraints = report["constraints"]
    solution = report["solution"]
    lines = [
        "# Field-Day Completion Plan",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Result",
        "",
        f"- Feasible: {summary['feasible']}",
        f"- Target segments: {summary['target_segment_count']}",
        f"- Covered segments: {summary['covered_segment_count']}",
        f"- Missing segments: {summary['missing_segment_count']}",
        f"- Runnable loops: {summary['loop_count']}",
        f"- Oversized loops: {summary['oversized_loop_count']}",
        f"- Invalid loops: {summary['invalid_loop_count']}",
        f"- Field-day candidates generated: {summary['field_day_candidate_count']}",
        f"- Solver success: {summary['solver_success']}",
        "",
        "## Bounds",
        "",
        f"- Weekday p90 max: {constraints['weekday_max_minutes']} min",
        f"- Weekend p90 max: {constraints['weekend_max_minutes']} min",
        f"- Date counts: {constraints['date_type_counts']}",
        "",
    ]
    if solution.get("success"):
        lines.extend(
            [
                "## Selected Field Days",
                "",
                f"- Field days: {solution['field_day_count']}",
                f"- Total p75 home-to-home time: {solution['total_p75_minutes']} min",
                f"- Max p90 stress: {solution['max_p90_stress']}",
                "",
                "| Day type | P75 | P90 | Stress | Loops |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for day in solution["field_days"]:
            lines.append(
                f"| {day['day_type']} | {day['p75_minutes']} | {day['p90_minutes']} | "
                f"{day['stress']} | {', '.join(day['candidate_ids'])} |"
            )
    else:
        lines.extend(
            [
                "## Feasibility Blockers",
                "",
                f"Reason: {solution.get('reason') or solution.get('message')}",
                "",
                "| Loop | Trailhead | P90 | Max bound | Official | On foot | Reason |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for blocker in report["oversized_loop_blockers"]:
            lines.append(
                f"| {blocker['label']} `{blocker['candidate_id']}` | {blocker['trailhead']} | "
                f"{blocker['p90_minutes']} | {blocker['max_available_bound_minutes']} | "
                f"{blocker['official_miles']} | {blocker['on_foot_miles']} | {blocker['reason']} |"
            )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--max-combo-size", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        read_json(args.map_data_json),
        read_json(args.manifest_json),
        read_json(args.official_geojson),
        read_json(args.state_json),
        max_combo_size=args.max_combo_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
