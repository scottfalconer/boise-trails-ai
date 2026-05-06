#!/usr/bin/env python3
"""Challenge the full field menu against a global executable set-cover optimizer."""

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

import route_alternative_challenge as alternatives  # noqa: E402
from export_execution_gpx import candidate_track_coordinates, load_official_segment_index, validate_track_segments  # noqa: E402
from personal_route_planner import DEFAULT_CONNECTOR_GEOJSON, load_connector_graph, load_official_segments  # noqa: E402
from route_boundary_challenge import dominance_comparison  # noqa: E402


DEFAULT_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_OFFICIAL_SEGMENTS_GEOJSON = alternatives.DEFAULT_OFFICIAL_SEGMENTS_GEOJSON
DEFAULT_CONNECTOR_GEOJSON_PATH = DEFAULT_CONNECTOR_GEOJSON
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "route-global-optimizer-challenge-2026-05-06"

OBJECTIVES = {
    "on_foot_miles": {"route_penalty": 0.0},
    "door_to_door_p75_minutes": {"route_penalty": 0.0},
    "ascent_ft": {"route_penalty": 0.0},
    "balanced": {"route_penalty": 0.02},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def official_target_ids(official_segments_geojson: dict[str, Any]) -> list[int]:
    ids = []
    for feature in official_segments_geojson.get("features") or []:
        props = feature.get("properties") or {}
        seg_id = alternatives.as_int(props.get("segId") or props.get("seg_id"))
        if seg_id is not None:
            ids.append(seg_id)
    return sorted(set(ids))


def normalize_current_components(map_data: dict[str, Any], segment_miles: dict[int, float]) -> list[dict[str, Any]]:
    rows = alternatives.selected_components(map_data, segment_miles)
    for row in rows:
        row["source"] = "current_field_menu"
        row["is_current_component"] = True
    return rows


def is_executable_candidate(candidate: dict[str, Any]) -> bool:
    if candidate.get("route_status") == "draft":
        return False
    return (
        candidate.get("ascent_ft") is not None
        and candidate.get("door_to_door_p75_minutes") is not None
        and has_navigation_path_payload(candidate)
    )


def has_navigation_path_payload(candidate: dict[str, Any]) -> bool:
    """Reject generated candidates that cannot export a continuous nav GPX.

    Graph validation proves a route can connect in the routing graph, but the
    field packet also needs concrete path coordinates for access/return legs.
    Without those coordinates, a generated candidate can look faster while
    producing a large GPX gap.
    """

    if candidate.get("navigation_path_ready") is False:
        return False
    trailhead_access = candidate.get("trailhead_access") or {}
    if float(trailhead_access.get("mapped_access_miles") or 0.0) > 0.05 and not trailhead_access.get(
        "outbound_path_coordinates"
    ):
        return False
    return_to_car = candidate.get("return_to_car") or {}
    return_distance = sum(
        float(return_to_car.get(key) or 0.0)
        for key in ["official_repeat_miles", "connector_miles", "road_miles"]
    )
    if return_distance > 0.05 and not return_to_car.get("path_coordinates"):
        return False
    for link in ((candidate.get("between_trail_links") or {}).get("links") or []):
        if float(link.get("distance_miles") or 0.0) > 0.05 and not link.get("path_coordinates"):
            return False
    return True


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or "")


def raw_candidate_lookup(candidate_sources: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for path in candidate_sources:
        payload = read_json(path)
        source = path.stem
        for item in payload.get("routes") or []:
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id:
                lookup[(source, candidate_id)] = item
        for key, item in (payload.get("candidate_index") or {}).items():
            candidate_id = str(item.get("candidate_id") or key)
            if candidate_id:
                lookup[(source, candidate_id)] = item
    return lookup


def raw_candidate_exports_continuous_track(
    candidate: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None,
    max_gap_miles: float = 0.05,
) -> bool:
    if not candidate.get("segments"):
        return has_navigation_path_payload(candidate)
    coords = candidate_track_coordinates(
        candidate,
        official_index,
        connector_graph=connector_graph,
        densify_source_lines=True,
    )
    validation = validate_track_segments([coords], max_gap_miles=max_gap_miles)
    return validation["passed"] is True


def build_candidate_pool(
    map_data: dict[str, Any],
    official_segments_geojson: dict[str, Any],
    candidate_sources: list[Path],
    *,
    official_index: dict[int, dict[str, Any]] | None = None,
    connector_graph: dict[str, Any] | None = None,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    segment_miles = alternatives.official_segment_miles(official_segments_geojson)
    target_ids = official_target_ids(official_segments_geojson)
    current = normalize_current_components(map_data, segment_miles)
    raw_lookup = raw_candidate_lookup(candidate_sources)
    generated = []
    for candidate in alternatives.load_candidate_universe(candidate_sources, segment_miles):
        if not is_executable_candidate(candidate):
            continue
        raw = raw_lookup.get((str(candidate.get("source")), str(candidate.get("candidate_id"))))
        if raw and official_index and not raw_candidate_exports_continuous_track(raw, official_index, connector_graph):
            continue
        generated.append(candidate)
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in generated:
        key = candidate_key(candidate)
        existing = by_id.get(key)
        if existing is None or alternatives.candidate_richness(candidate) > alternatives.candidate_richness(existing):
            by_id[key] = candidate
    for candidate in current:
        by_id[candidate_key(candidate)] = candidate
    pool = [
        candidate
        for candidate in by_id.values()
        if set(candidate.get("segment_ids") or []) & set(target_ids)
    ]
    return target_ids, pool, current


def metric(candidate: dict[str, Any], key: str) -> float:
    if key == "door_to_door_p75_minutes":
        return float(candidate.get(key) or candidate.get("total_minutes") or 999999.0)
    return float(candidate.get(key) or 999999.0)


def aggregate_selection(candidates: list[dict[str, Any]], target_ids: list[int]) -> dict[str, Any]:
    target_set = set(target_ids)
    covered = set().union(*(set(candidate.get("segment_ids") or []) & target_set for candidate in candidates)) if candidates else set()
    extra = set().union(*(set(candidate.get("segment_ids") or []) - target_set for candidate in candidates)) if candidates else set()
    official = alternatives.sum_segment_miles(sorted(covered), {}) or 0.0
    # The official target is stable, but keep selected official for debug when no segment-mile map is passed.
    selected_official = sum(float(candidate.get("official_miles") or 0.0) for candidate in candidates)
    on_foot = sum(float(candidate.get("on_foot_miles") or 0.0) for candidate in candidates)
    p75 = sum(metric(candidate, "door_to_door_p75_minutes") for candidate in candidates)
    ascent = sum(float(candidate.get("ascent_ft") or 0.0) for candidate in candidates)
    grade = sum(float(candidate.get("grade_adjusted_miles") or 0.0) for candidate in candidates)
    return {
        "route_count": len(candidates),
        "covered_segment_count": len(covered),
        "missing_segment_count": len(target_set - covered),
        "extra_segment_count": len(extra),
        "selected_official_miles_sum": round(selected_official, 2),
        "on_foot_miles": round(on_foot, 2),
        "door_to_door_p75_minutes": int(round(p75)),
        "ascent_ft": int(round(ascent)),
        "grade_adjusted_miles": round(grade, 2),
        "candidate_ids": [candidate.get("candidate_id") for candidate in candidates],
        "current_component_count": sum(1 for candidate in candidates if candidate.get("is_current_component")),
        "generated_component_count": sum(1 for candidate in candidates if not candidate.get("is_current_component")),
    }


def current_solution(current: list[dict[str, Any]], target_ids: list[int]) -> dict[str, Any]:
    return aggregate_selection(current, target_ids)


def objective_costs(
    pool: list[dict[str, Any]],
    current: dict[str, Any],
    objective: str,
    *,
    route_penalty: float,
) -> np.ndarray:
    if objective != "balanced":
        return np.array([metric(candidate, objective) + route_penalty for candidate in pool], dtype=float)
    current_on_foot = float(current.get("on_foot_miles") or 1.0)
    current_p75 = float(current.get("door_to_door_p75_minutes") or 1.0)
    current_ascent = float(current.get("ascent_ft") or 1.0)
    current_grade = float(current.get("grade_adjusted_miles") or 1.0)
    costs = []
    for candidate in pool:
        costs.append(
            (float(candidate.get("on_foot_miles") or 0.0) / current_on_foot)
            + (metric(candidate, "door_to_door_p75_minutes") / current_p75)
            + (float(candidate.get("ascent_ft") or 0.0) / current_ascent)
            + (float(candidate.get("grade_adjusted_miles") or 0.0) / current_grade)
            + route_penalty
        )
    return np.array(costs, dtype=float)


def solve_set_cover(
    pool: list[dict[str, Any]],
    target_ids: list[int],
    current: dict[str, Any],
    objective: str,
    *,
    route_penalty: float,
) -> dict[str, Any]:
    id_to_row = {seg_id: index for index, seg_id in enumerate(target_ids)}
    matrix = lil_matrix((len(target_ids), len(pool)), dtype=float)
    for column, candidate in enumerate(pool):
        for seg_id in set(candidate.get("segment_ids") or []):
            row = id_to_row.get(int(seg_id))
            if row is not None:
                matrix[row, column] = 1.0
    constraints = LinearConstraint(matrix.tocsr(), lb=np.ones(len(target_ids)), ub=np.full(len(target_ids), np.inf))
    costs = objective_costs(pool, current, objective, route_penalty=route_penalty)
    result = milp(
        c=costs,
        integrality=np.ones(len(pool), dtype=int),
        bounds=Bounds(np.zeros(len(pool)), np.ones(len(pool))),
        constraints=constraints,
        options={"time_limit": 60},
    )
    if not result.success:
        return {
            "objective": objective,
            "success": False,
            "status": int(result.status),
            "message": result.message,
        }
    selected = [candidate for candidate, value in zip(pool, result.x) if value >= 0.5]
    summary = aggregate_selection(selected, target_ids)
    return {
        "objective": objective,
        "success": True,
        "solver_objective_value": round(float(result.fun), 6),
        "route_penalty": route_penalty,
        **summary,
        "routes": [compact_candidate(candidate) for candidate in sorted(selected, key=lambda row: str(row.get("candidate_id") or ""))],
    }


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": candidate.get("source"),
        "candidate_id": candidate.get("candidate_id"),
        "trailhead": candidate.get("trailhead"),
        "official_miles": candidate.get("official_miles"),
        "on_foot_miles": candidate.get("on_foot_miles"),
        "door_to_door_p75_minutes": candidate.get("door_to_door_p75_minutes") or candidate.get("total_minutes"),
        "ascent_ft": candidate.get("ascent_ft"),
        "grade_adjusted_miles": candidate.get("grade_adjusted_miles"),
        "segment_count": len(candidate.get("segment_ids") or []),
        "route_status": candidate.get("route_status"),
        "is_current_component": bool(candidate.get("is_current_component")),
    }


def build_report(
    map_data: dict[str, Any],
    official_segments_geojson: dict[str, Any],
    candidate_sources: list[Path],
    connector_geojson: Path = DEFAULT_CONNECTOR_GEOJSON_PATH,
) -> dict[str, Any]:
    official_index = load_official_segment_index(DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    official_segments, _official_meta = load_official_segments(DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    connector_graph = load_connector_graph(connector_geojson, official_segments=official_segments)
    target_ids, pool, current_components = build_candidate_pool(
        map_data,
        official_segments_geojson,
        candidate_sources,
        official_index=official_index,
        connector_graph=connector_graph,
    )
    current = current_solution(current_components, target_ids)
    solutions = [
        solve_set_cover(pool, target_ids, current, objective, route_penalty=config["route_penalty"])
        for objective, config in OBJECTIVES.items()
    ]
    successful = [solution for solution in solutions if solution.get("success")]
    dominance = [dominance_comparison(current, solution) for solution in successful]
    dominant = [row for row in dominance if row["dominates_current"]]
    return {
        "objective": "challenge the complete field menu against a global executable set-cover optimizer",
        "source_candidate_files": [str(path) for path in candidate_sources],
        "summary": {
            "target_segment_count": len(target_ids),
            "candidate_pool_count": len(pool),
            "current_route_count": current["route_count"],
            "successful_solution_count": len(successful),
            "dominant_solution_count": len(dominant),
            "global_optimizer_beats_current": bool(dominant),
        },
        "current": current,
        "solutions": solutions,
        "dominance": dominance,
        "best_dominant_solution": dominant[0] if dominant else None,
        "caveats": [
            "This optimizer only uses executable generated candidates with DEM ascent and p75 time, plus the current field-menu components.",
            "Draft generated routes are excluded. A draft route can seed manual GPX design but cannot beat the current executable menu.",
            "Set-cover results are proof candidates; any route selected outside the current menu still needs field-facing GPX and cue promotion before use.",
        ],
    }


def fmt_solution(solution: dict[str, Any] | None) -> str:
    if not solution:
        return "none"
    combo = solution.get("combo") if "combo" in solution else solution
    return (
        f"{combo.get('on_foot_miles')} mi, {combo.get('door_to_door_p75_minutes')} min p75, "
        f"{combo.get('ascent_ft')} ft ascent, {combo.get('route_count')} routes"
    )


def render_md(report: dict[str, Any]) -> str:
    current = report["current"]
    summary = report["summary"]
    lines = [
        "# Route Global Optimizer Challenge",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Target segments: {summary['target_segment_count']}",
        f"- Candidate pool: {summary['candidate_pool_count']}",
        f"- Current menu: {fmt_solution(current)}",
        f"- Successful optimizer solutions: {summary['successful_solution_count']}",
        f"- Dominant solutions: {summary['dominant_solution_count']}",
        f"- Global optimizer beats current: {summary['global_optimizer_beats_current']}",
        f"- Best dominant solution: {fmt_solution(report.get('best_dominant_solution'))}",
        "",
        "## Solutions",
        "",
        "| Objective | Success | On-foot | P75 min | Ascent | Grade-adjusted | Routes | Current routes | Generated routes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for solution in report["solutions"]:
        lines.append(
            f"| {solution['objective']} | {solution.get('success')} | {solution.get('on_foot_miles')} | "
            f"{solution.get('door_to_door_p75_minutes')} | {solution.get('ascent_ft')} | "
            f"{solution.get('grade_adjusted_miles')} | {solution.get('route_count')} | "
            f"{solution.get('current_component_count')} | {solution.get('generated_component_count')} |"
        )
    lines.extend(["", "## Dominance Checks", "", "| Candidate | Dominates | Better metrics | Worse metrics | Deltas |", "|---|---|---|---|---|"])
    for row in report["dominance"]:
        lines.append(
            f"| {', '.join(str(value) for value in row.get('candidate_ids') or [])[:120]} | "
            f"{row['dominates_current']} | {', '.join(row['materially_better_metrics'])} | "
            f"{', '.join(row['materially_worse_metrics'])} | {row['deltas']} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--official-segments-geojson", type=Path, default=DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON_PATH)
    parser.add_argument("--candidate-source", action="append", type=Path, dest="candidate_sources")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_sources = args.candidate_sources or alternatives.DEFAULT_CANDIDATE_SOURCES
    report = build_report(
        read_json(args.map_data_json),
        read_json(args.official_segments_geojson),
        candidate_sources,
        connector_geojson=args.connector_geojson,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
