#!/usr/bin/env python3
"""Audit p90 coverage after adding split/forced-anchor probe candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from p90_completion_gap_analyzer import (  # noqa: E402
    DEFAULT_FIELD_MENU_JSON,
    DEFAULT_HYBRID_ROUTE_PASS_JSON,
    DEFAULT_PERSONAL_ROUTE_MENU_JSON,
    DEFAULT_STATE_JSON,
    as_float,
    as_int,
    display_path,
    load_candidates,
    official_segments,
    solve_maximum_bounded_coverage,
    usable_candidates,
)
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_SPLIT_PROBE_JSON = YEAR_DIR / "checkpoints" / "p90-segment-split-probe-2026-05-06.json"
DEFAULT_FORCED_PROBE_JSON = YEAR_DIR / "checkpoints" / "p90-forced-anchor-probe-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-repaired-candidate-universe-audit-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def probe_row_valid(row: dict[str, Any]) -> bool:
    return (
        row.get("route_status") == "graph_validated"
        and row.get("track_validation_passed") is True
        and row.get("validation_passed") is True
    )


def normalize_probe_row(
    row: dict[str, Any],
    *,
    source: str,
    require_field_ready: bool,
) -> dict[str, Any] | None:
    if not probe_row_valid(row):
        return None
    if require_field_ready and row.get("field_ready") is not True:
        return None
    seg_id = as_int(row.get("seg_id"))
    p75 = as_int(row.get("door_to_door_p75_minutes"))
    p90 = as_int(row.get("door_to_door_p90_minutes"))
    on_foot = as_float(row.get("on_foot_miles"))
    if seg_id is None or p75 is None or p90 is None or on_foot is None:
        return None
    candidate_id = str(row.get("candidate_id") or f"{source}-{seg_id}")
    if source == "forced_anchor_probe":
        candidate_id = f"{candidate_id}::{row.get('anchor_name') or row.get('trailhead')}"
    return {
        "source": source,
        "candidate_id": candidate_id,
        "label": row.get("seg_name") or candidate_id,
        "route_status": row.get("route_status"),
        "validation_passed": True,
        "track_validation_passed": True,
        "field_ready": row.get("field_ready", True),
        "trailhead": row.get("anchor_name") or row.get("trailhead"),
        "trail_names": [row.get("trail_name")] if row.get("trail_name") else [],
        "segment_ids": [seg_id],
        "official_miles": as_float(row.get("official_miles")) or 0.0,
        "on_foot_miles": on_foot,
        "door_to_door_p75_minutes": p75,
        "door_to_door_p90_minutes": p90,
        "grade_adjusted_miles": as_float(row.get("grade_adjusted_miles")),
        "ascent_ft": as_int(row.get("ascent_ft")),
        "manual_design_hold": False,
        "parking_confidence": row.get("parking_confidence"),
        "parking_risk": as_int(row.get("parking_risk")),
    }


def strict_probe_candidates(
    split_probe: dict[str, Any],
    forced_probe: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in split_probe.get("probe_rows") or []:
        normalized = normalize_probe_row(row, source="single_segment_split_probe", require_field_ready=False)
        if normalized:
            candidates.append(normalized)
    for row in forced_probe.get("probe_rows") or []:
        normalized = normalize_probe_row(row, source="forced_anchor_probe", require_field_ready=True)
        if normalized:
            candidates.append(normalized)
    return candidates


def best_over_bound_candidate_for_segment(
    candidates: list[dict[str, Any]],
    seg_id: int,
    bound: int,
) -> dict[str, Any] | None:
    matches = [
        candidate
        for candidate in candidates
        if seg_id in candidate["segment_ids"]
        and candidate["door_to_door_p90_minutes"] > bound
        and candidate.get("validation_passed") is True
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda candidate: (
            candidate["door_to_door_p90_minutes"],
            candidate["door_to_door_p75_minutes"],
            candidate["on_foot_miles"],
            candidate.get("parking_risk") or 99,
        ),
    )


def coverage_for(candidates: list[dict[str, Any]], target_ids: list[int]) -> dict[str, Any]:
    target_set = set(target_ids)
    covered = {
        seg_id
        for candidate in candidates
        for seg_id in candidate["segment_ids"]
        if seg_id in target_set
    }
    return {
        "covered_segment_count": len(covered),
        "missing_segment_count": len(target_set - covered),
        "missing_segment_ids": sorted(target_set - covered),
    }


def candidate_brief(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {
        key: candidate.get(key)
        for key in [
            "source",
            "candidate_id",
            "trailhead",
            "trail_names",
            "segment_ids",
            "official_miles",
            "on_foot_miles",
            "door_to_door_p75_minutes",
            "door_to_door_p90_minutes",
            "parking_confidence",
            "parking_risk",
        ]
    }


def solve_exact_set_cover(candidates: list[dict[str, Any]], target_ids: list[int]) -> dict[str, Any]:
    if not candidates:
        return {"success": False, "reason": "no_candidates"}
    target_index = {seg_id: row for row, seg_id in enumerate(target_ids)}
    matrix = lil_matrix((len(target_ids), len(candidates)), dtype=float)
    for col, candidate in enumerate(candidates):
        for seg_id in set(candidate["segment_ids"]):
            if seg_id in target_index:
                matrix[target_index[seg_id], col] = 1.0
    costs = np.array(
        [
            float(candidate["door_to_door_p75_minutes"])
            + float(candidate["door_to_door_p90_minutes"]) * 0.001
            + float(candidate["on_foot_miles"]) * 0.0001
            + float(candidate.get("parking_risk") or 1) * 0.00001
            + 0.000001
            for candidate in candidates
        ],
        dtype=float,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(
            matrix.tocsr(),
            lb=np.ones(len(target_ids)),
            ub=np.full(len(target_ids), np.inf),
        ),
        options={"time_limit": 60},
    )
    if not result.success:
        return {
            "success": False,
            "status": int(result.status),
            "message": result.message,
        }
    selected = [candidate for candidate, value in zip(candidates, result.x) if value >= 0.5]
    covered = sorted(
        {
            seg_id
            for candidate in selected
            for seg_id in candidate["segment_ids"]
            if seg_id in target_index
        }
    )
    return {
        "success": True,
        "selected_candidate_count": len(selected),
        "covered_segment_count": len(covered),
        "missing_segment_count": len(set(target_ids) - set(covered)),
        "total_p75_minutes": int(sum(candidate["door_to_door_p75_minutes"] for candidate in selected)),
        "total_p90_minutes": int(sum(candidate["door_to_door_p90_minutes"] for candidate in selected)),
        "total_on_foot_miles": round(sum(candidate["on_foot_miles"] for candidate in selected), 2),
        "selected_candidates": sorted(
            [candidate_brief(candidate) for candidate in selected],
            key=lambda candidate: (
                int(candidate.get("door_to_door_p90_minutes") or 0),
                str(candidate.get("candidate_id") or ""),
            ),
        ),
        "missing_segment_ids": sorted(set(target_ids) - set(covered)),
    }


def build_report(
    *,
    official_geojson: dict[str, Any],
    personal_route_menu: dict[str, Any],
    hybrid_route_pass: dict[str, Any],
    field_menu: dict[str, Any],
    state: dict[str, Any],
    split_probe: dict[str, Any],
    forced_probe: dict[str, Any],
) -> dict[str, Any]:
    segment_index = official_segments(official_geojson)
    target_ids = sorted(segment_index)
    availability = state.get("availability_model") or {}
    max_bound = max(
        int(availability.get("weekday_max_minutes") or 0),
        int(availability.get("weekend_max_minutes") or 0),
    )
    existing_all = load_candidates(personal_route_menu, hybrid_route_pass, field_menu, segment_index)
    existing_usable = usable_candidates(existing_all)
    probe_candidates = strict_probe_candidates(split_probe, forced_probe)
    repaired_all = existing_usable + probe_candidates
    strict_bounded = [
        candidate
        for candidate in repaired_all
        if candidate["door_to_door_p90_minutes"] <= max_bound
        and candidate.get("validation_passed") is True
        and candidate.get("manual_design_hold") is not True
    ]
    strict_coverage = coverage_for(strict_bounded, target_ids)
    shingle_exception = best_over_bound_candidate_for_segment(repaired_all, 1656, max_bound)
    exception_candidates = strict_bounded + ([shingle_exception] if shingle_exception else [])
    exception_coverage = coverage_for(exception_candidates, target_ids)
    strict_solution = solve_maximum_bounded_coverage(repaired_all, target_ids, max_bound)
    exception_solution = solve_maximum_bounded_coverage(exception_candidates, target_ids, 10_000)
    exact_strict_set_cover = solve_exact_set_cover(strict_bounded, target_ids)
    exact_exception_set_cover = solve_exact_set_cover(exception_candidates, target_ids)
    return {
        "objective": "audit p90-bounded official coverage after adding split and forced-anchor probe candidates",
        "source_files": {
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "personal_route_menu_json": display_path(DEFAULT_PERSONAL_ROUTE_MENU_JSON),
            "hybrid_route_pass_json": display_path(DEFAULT_HYBRID_ROUTE_PASS_JSON),
            "field_menu_json": display_path(DEFAULT_FIELD_MENU_JSON),
            "state_json": display_path(DEFAULT_STATE_JSON),
            "split_probe_json": display_path(DEFAULT_SPLIT_PROBE_JSON),
            "forced_probe_json": display_path(DEFAULT_FORCED_PROBE_JSON),
        },
        "bounds": {
            "max_daily_p90_bound_minutes": max_bound,
            "weekday_max_minutes": availability.get("weekday_max_minutes"),
            "weekend_max_minutes": availability.get("weekend_max_minutes"),
        },
        "summary": {
            "target_segment_count": len(target_ids),
            "existing_usable_candidate_count": len(existing_usable),
            "strict_probe_candidate_count": len(probe_candidates),
            "repaired_candidate_count": len(repaired_all),
            "strict_bounded_candidate_count": len(strict_bounded),
            "strict_bounded_covered_segment_count": strict_coverage["covered_segment_count"],
            "strict_bounded_missing_segment_count": strict_coverage["missing_segment_count"],
            "strict_bounded_missing_segment_ids": strict_coverage["missing_segment_ids"],
            "completion_possible_under_current_p90_bound": strict_coverage["missing_segment_count"] == 0,
            "completion_possible_if_shingle_exception_accepted": exception_coverage["missing_segment_count"] == 0,
            "exact_strict_set_cover_success": exact_strict_set_cover.get("success") is True,
            "exact_shingle_exception_set_cover_success": exact_exception_set_cover.get("success") is True,
            "exact_shingle_exception_selected_candidate_count": exact_exception_set_cover.get("selected_candidate_count"),
        },
        "strict_bounded_coverage": strict_coverage,
        "strict_bounded_solution": strict_solution,
        "exact_strict_bounded_set_cover": exact_strict_set_cover,
        "shingle_exception_candidate": candidate_brief(shingle_exception),
        "coverage_with_shingle_exception": exception_coverage,
        "solution_with_shingle_exception": exception_solution,
        "exact_set_cover_with_shingle_exception": exact_exception_set_cover,
        "caveats": [
            "This is a coverage/candidate-universe audit, not a final calendar schedule.",
            "Probe candidates are single-car, graph-validated, and GPX-continuous rows, but not all are promoted to the canonical phone menu yet.",
            "The exact set-cover solution counts selected loop candidates only; it does not prove those loops can be packed into dated field days under p90 bounds.",
            "The Shingle exception scenario is only a what-if; it does not satisfy the current strict p90 bound unless the user accepts an exception or changes the bound.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    exception = report.get("shingle_exception_candidate") or {}
    lines = [
        "# P90 Repaired Candidate Universe Audit",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Target segments: {summary['target_segment_count']}",
        f"- Existing usable candidates: {summary['existing_usable_candidate_count']}",
        f"- Strict probe candidates added: {summary['strict_probe_candidate_count']}",
        f"- Repaired candidates: {summary['repaired_candidate_count']}",
        f"- Strict bounded candidates: {summary['strict_bounded_candidate_count']}",
        f"- Strict bounded coverage: {summary['strict_bounded_covered_segment_count']} / {summary['target_segment_count']}",
        f"- Strict bounded missing: {summary['strict_bounded_missing_segment_ids']}",
        f"- Completion possible under current p90 bound: {summary['completion_possible_under_current_p90_bound']}",
        f"- Completion possible if Shingle exception accepted: {summary['completion_possible_if_shingle_exception_accepted']}",
        f"- Exact strict set cover success: {summary['exact_strict_set_cover_success']}",
        f"- Exact Shingle-exception set cover success: {summary['exact_shingle_exception_set_cover_success']}",
        f"- Exact Shingle-exception selected candidates: {summary['exact_shingle_exception_selected_candidate_count']}",
        "",
        "## Shingle Exception Candidate",
        "",
    ]
    if exception:
        lines.extend(
            [
                f"- Candidate: `{exception['candidate_id']}`",
                f"- Trailhead: {exception['trailhead']}",
                f"- P75 / P90: {exception['door_to_door_p75_minutes']} / {exception['door_to_door_p90_minutes']} min",
                f"- On foot: {exception['on_foot_miles']} mi",
                f"- Parking: {exception.get('parking_confidence')}",
            ]
        )
    else:
        lines.append("- No Shingle exception candidate found.")
    lines.extend(
        [
            "",
            "## Strict Bounded Greedy Coverage",
            "",
            f"- Selected candidates: {report['strict_bounded_solution'].get('selected_candidate_count')}",
            f"- Covered segments: {report['strict_bounded_solution'].get('covered_segment_count')}",
            f"- Missing segments: {report['strict_bounded_solution'].get('missing_segment_count')}",
            f"- Total p75 minutes: {report['strict_bounded_solution'].get('total_p75_minutes')}",
            f"- Total on-foot miles: {report['strict_bounded_solution'].get('total_on_foot_miles')}",
            "",
            "## Exact Set Cover",
            "",
            f"- Strict bounded set cover success: {report['exact_strict_bounded_set_cover'].get('success')}",
            f"- Shingle-exception set cover success: {report['exact_set_cover_with_shingle_exception'].get('success')}",
            f"- Shingle-exception selected loop candidates: {report['exact_set_cover_with_shingle_exception'].get('selected_candidate_count')}",
            f"- Shingle-exception total p75 minutes across selected loop candidates: {report['exact_set_cover_with_shingle_exception'].get('total_p75_minutes')}",
            f"- Shingle-exception total on-foot miles across selected loop candidates: {report['exact_set_cover_with_shingle_exception'].get('total_on_foot_miles')}",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--personal-route-menu-json", type=Path, default=DEFAULT_PERSONAL_ROUTE_MENU_JSON)
    parser.add_argument("--hybrid-route-pass-json", type=Path, default=DEFAULT_HYBRID_ROUTE_PASS_JSON)
    parser.add_argument("--field-menu-json", type=Path, default=DEFAULT_FIELD_MENU_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--split-probe-json", type=Path, default=DEFAULT_SPLIT_PROBE_JSON)
    parser.add_argument("--forced-probe-json", type=Path, default=DEFAULT_FORCED_PROBE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        official_geojson=read_json(args.official_geojson),
        personal_route_menu=read_json(args.personal_route_menu_json),
        hybrid_route_pass=read_json(args.hybrid_route_pass_json),
        field_menu=read_json(args.field_menu_json),
        state=read_json(args.state_json),
        split_probe=read_json(args.split_probe_json),
        forced_probe=read_json(args.forced_probe_json),
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
