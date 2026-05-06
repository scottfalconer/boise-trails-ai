#!/usr/bin/env python3
"""Challenge package boundaries against generated route combinations."""

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

import route_alternative_challenge as alternatives  # noqa: E402


DEFAULT_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_OFFICIAL_SEGMENTS_GEOJSON = alternatives.DEFAULT_OFFICIAL_SEGMENTS_GEOJSON
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "route-boundary-challenge-p02-p13-2026-05-06"
DEFAULT_PACKAGE_NUMBERS = [2, 13]

MEANINGFUL_IMPROVEMENT_MILES = 0.25
MEANINGFUL_IMPROVEMENT_MINUTES = 10
MEANINGFUL_IMPROVEMENT_ASCENT_FT = 250


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_title(package: dict[str, Any]) -> str:
    return str(package.get("block_name") or f"Package {package.get('package_number')}")


def segment_name_index(official_segments_geojson: dict[str, Any]) -> dict[int, str]:
    names = {}
    for feature in official_segments_geojson.get("features") or []:
        props = feature.get("properties") or {}
        seg_id = alternatives.as_int(props.get("segId") or props.get("seg_id"))
        if seg_id is not None:
            names[seg_id] = str(props.get("segName") or props.get("seg_name") or seg_id)
    return names


def selected_package_components(
    map_data: dict[str, Any],
    segment_miles: dict[int, float],
    package_numbers: list[int],
) -> list[dict[str, Any]]:
    wanted = set(package_numbers)
    return [
        component
        for component in alternatives.selected_components(map_data, segment_miles)
        if component.get("package_number") in wanted
    ]


def target_segment_ids(components: list[dict[str, Any]]) -> set[int]:
    target_ids: set[int] = set()
    for component in components:
        target_ids.update(int(seg_id) for seg_id in component.get("segment_ids") or [])
    return target_ids


def candidate_pool(
    current_components: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target_ids: set[int],
    *,
    min_overlap_count: int = 2,
    min_target_purity: float = 0.25,
    max_pool: int = 36,
    require_effort_metrics: bool = True,
    exclude_draft_candidates: bool = True,
) -> list[dict[str, Any]]:
    pool = []
    for component in current_components:
        pool.append({**component, "source": "current_field_menu"})
    for candidate in candidates:
        if exclude_draft_candidates and candidate.get("route_status") == "draft":
            continue
        if require_effort_metrics and (
            candidate.get("ascent_ft") is None or candidate.get("door_to_door_p75_minutes") is None
        ):
            continue
        candidate_ids = set(candidate.get("segment_ids") or [])
        overlap = candidate_ids & target_ids
        if len(overlap) < min_overlap_count:
            continue
        purity = len(overlap) / len(candidate_ids) if candidate_ids else 0.0
        if purity < min_target_purity:
            continue
        pool.append(candidate)

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in pool:
        key = str(candidate.get("candidate_id") or "")
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
        elif existing.get("source") == "current_field_menu":
            continue
        elif candidate.get("source") == "current_field_menu":
            deduped[key] = candidate
        elif alternatives.candidate_richness(candidate) > alternatives.candidate_richness(existing):
            deduped[key] = candidate

    return sorted(
        deduped.values(),
        key=lambda row: (
            -len(set(row.get("segment_ids") or []) & target_ids),
            float(row.get("on_foot_miles") or 9999.0),
            int(row.get("total_minutes") or 99999),
        ),
    )[:max_pool]


def sum_optional_int(candidates: tuple[dict[str, Any], ...], key: str) -> int | None:
    values = [candidate.get(key) for candidate in candidates]
    if any(value is None for value in values):
        return None
    return int(sum(int(value) for value in values))


def sum_optional_float(candidates: tuple[dict[str, Any], ...], key: str) -> float | None:
    values = [candidate.get(key) for candidate in candidates]
    if any(value is None for value in values):
        return None
    return round(sum(float(value) for value in values), 2)


def compact_candidate(candidate: dict[str, Any], target_ids: set[int]) -> dict[str, Any]:
    candidate_ids = set(candidate.get("segment_ids") or [])
    return {
        **alternatives.compact_candidate(candidate, target_ids),
        "target_segment_ids": sorted(candidate_ids & target_ids),
    }


def aggregate_combo(
    combo: tuple[dict[str, Any], ...],
    target_ids: set[int],
    segment_miles: dict[int, float],
    segment_names: dict[int, str],
) -> dict[str, Any]:
    covered = set().union(*(set(candidate.get("segment_ids") or []) & target_ids for candidate in combo))
    extra = set().union(*(set(candidate.get("segment_ids") or []) - target_ids for candidate in combo))
    missing = target_ids - covered
    on_foot = sum(float(candidate.get("on_foot_miles") or 0.0) for candidate in combo)
    official_selected = sum(float(candidate.get("official_miles") or 0.0) for candidate in combo)
    target_official = alternatives.sum_segment_miles(sorted(target_ids), segment_miles) or 0.0
    trailheads = sorted({str(candidate.get("trailhead") or "") for candidate in combo if candidate.get("trailhead")})
    return {
        "candidate_ids": [candidate.get("candidate_id") for candidate in combo],
        "sources": [candidate.get("source") for candidate in combo],
        "route_count": len(combo),
        "trailhead_count": len(trailheads),
        "trailheads": trailheads,
        "target_official_miles": round(target_official, 2),
        "selected_official_miles": round(official_selected, 2),
        "on_foot_miles": round(on_foot, 2),
        "ratio_vs_target_official": round(on_foot / target_official, 3) if target_official else None,
        "total_minutes": sum_optional_int(combo, "total_minutes"),
        "door_to_door_p75_minutes": sum_optional_int(combo, "door_to_door_p75_minutes")
        or sum_optional_int(combo, "total_minutes"),
        "moving_effort_p75_minutes": sum_optional_int(combo, "moving_effort_p75_minutes"),
        "route_finding_penalty_minutes": sum_optional_int(combo, "route_finding_penalty_minutes"),
        "ascent_ft": sum_optional_int(combo, "ascent_ft"),
        "descent_ft": sum_optional_int(combo, "descent_ft"),
        "grade_adjusted_miles": sum_optional_float(combo, "grade_adjusted_miles"),
        "covered_segment_count": len(covered),
        "missing_segment_count": len(missing),
        "extra_segment_count": len(extra),
        "missing_segments": [{"seg_id": seg_id, "name": segment_names.get(seg_id)} for seg_id in sorted(missing)],
        "extra_segments": [{"seg_id": seg_id, "name": segment_names.get(seg_id)} for seg_id in sorted(extra)],
        "components": [compact_candidate(candidate, target_ids) for candidate in combo],
    }


def combo_sort_key(metric: str):
    def sort(row: dict[str, Any]) -> tuple[Any, ...]:
        value = row.get(metric)
        if value is None:
            value = 999999
        return (
            value,
            row.get("route_count") or 999,
            row.get("extra_segment_count") or 999,
            row.get("on_foot_miles") or 999999,
            row.get("door_to_door_p75_minutes") or 999999,
        )

    return sort


def enumerate_covering_combos(
    pool: list[dict[str, Any]],
    target_ids: set[int],
    segment_miles: dict[int, float],
    segment_names: dict[int, str],
    *,
    max_routes: int,
    max_results: int,
) -> list[dict[str, Any]]:
    target_ids = set(target_ids)
    combos = []
    for route_count in range(1, max_routes + 1):
        for combo in itertools.combinations(pool, route_count):
            covered = set().union(*(set(candidate.get("segment_ids") or []) & target_ids for candidate in combo))
            if covered >= target_ids:
                combos.append(aggregate_combo(combo, target_ids, segment_miles, segment_names))
    return sorted(combos, key=combo_sort_key("on_foot_miles"))[:max_results]


def compare_metric(current: dict[str, Any], best: dict[str, Any] | None, metric: str, meaningful_delta: float) -> dict[str, Any]:
    if not best or current.get(metric) is None or best.get(metric) is None:
        return {
            "metric": metric,
            "status": "insufficient_data",
            "current": current.get(metric),
            "best": best.get(metric) if best else None,
            "delta": None,
        }
    delta = float(current[metric]) - float(best[metric])
    return {
        "metric": metric,
        "status": "better_generated_combo_found" if delta > meaningful_delta else "current_not_meaningfully_beaten",
        "current": current.get(metric),
        "best": best.get(metric),
        "delta": round(delta, 2),
        "best_candidate_ids": best.get("candidate_ids"),
    }


def dominance_comparison(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "on_foot_miles": MEANINGFUL_IMPROVEMENT_MILES,
        "door_to_door_p75_minutes": MEANINGFUL_IMPROVEMENT_MINUTES,
        "ascent_ft": MEANINGFUL_IMPROVEMENT_ASCENT_FT,
        "grade_adjusted_miles": MEANINGFUL_IMPROVEMENT_MILES,
    }
    deltas: dict[str, float | None] = {}
    materially_better = []
    materially_worse = []
    for metric, threshold in metrics.items():
        if current.get(metric) is None or candidate.get(metric) is None:
            deltas[metric] = None
            continue
        delta = float(current[metric]) - float(candidate[metric])
        deltas[metric] = round(delta, 2)
        if delta > threshold:
            materially_better.append(metric)
        elif delta < -threshold:
            materially_worse.append(metric)
    return {
        "candidate_ids": candidate.get("candidate_ids"),
        "dominates_current": bool(materially_better) and not materially_worse,
        "materially_better_metrics": materially_better,
        "materially_worse_metrics": materially_worse,
        "deltas": deltas,
        "combo": candidate,
    }


def build_report(
    map_data: dict[str, Any],
    official_segments: dict[str, Any],
    candidate_sources: list[Path],
    package_numbers: list[int],
    *,
    max_routes: int = 5,
    max_pool: int = 36,
    max_results: int = 20,
    require_effort_metrics: bool = True,
    exclude_draft_candidates: bool = True,
) -> dict[str, Any]:
    segment_miles = alternatives.official_segment_miles(official_segments)
    segment_names = segment_name_index(official_segments)
    current_components = selected_package_components(map_data, segment_miles, package_numbers)
    target_ids = target_segment_ids(current_components)
    packages = [
        package
        for package in map_data.get("packages") or []
        if package.get("package_number") in set(package_numbers)
    ]
    universe = alternatives.load_candidate_universe(candidate_sources, segment_miles)
    pool = candidate_pool(
        current_components,
        universe,
        target_ids,
        max_pool=max_pool,
        require_effort_metrics=require_effort_metrics,
        exclude_draft_candidates=exclude_draft_candidates,
    )
    current = aggregate_combo(tuple(current_components), target_ids, segment_miles, segment_names)
    covering = enumerate_covering_combos(
        pool,
        target_ids,
        segment_miles,
        segment_names,
        max_routes=max_routes,
        max_results=max_results,
    )
    best_by_on_foot = sorted(covering, key=combo_sort_key("on_foot_miles"))[0] if covering else None
    best_by_time = sorted(covering, key=combo_sort_key("door_to_door_p75_minutes"))[0] if covering else None
    best_by_ascent = sorted(covering, key=combo_sort_key("ascent_ft"))[0] if covering else None
    best_by_grade = sorted(covering, key=combo_sort_key("grade_adjusted_miles"))[0] if covering else None
    comparisons = [
        compare_metric(current, best_by_on_foot, "on_foot_miles", MEANINGFUL_IMPROVEMENT_MILES),
        compare_metric(current, best_by_time, "door_to_door_p75_minutes", MEANINGFUL_IMPROVEMENT_MINUTES),
        compare_metric(current, best_by_ascent, "ascent_ft", MEANINGFUL_IMPROVEMENT_ASCENT_FT),
        compare_metric(current, best_by_grade, "grade_adjusted_miles", MEANINGFUL_IMPROVEMENT_MILES),
    ]
    better = [row for row in comparisons if row["status"] == "better_generated_combo_found"]
    dominance = [dominance_comparison(current, row) for row in covering]
    dominant = [row for row in dominance if row["dominates_current"]]
    return {
        "objective": "challenge package boundary routing as a combined coverage problem with distance, elevation, and p75 time",
        "package_numbers": package_numbers,
        "package_titles": [package_title(package) for package in packages],
        "source_candidate_files": [str(path) for path in candidate_sources],
        "summary": {
            "target_segment_count": len(target_ids),
            "target_official_miles": current["target_official_miles"],
            "current_route_count": current["route_count"],
            "candidate_pool_count": len(pool),
            "covering_combo_count_returned": len(covering),
            "better_generated_metric_count": len(better),
            "dominant_generated_combo_count": len(dominant),
            "generated_combo_beats_current": bool(dominant),
            "all_covering_combos_include_elevation": all(row.get("ascent_ft") is not None for row in covering),
            "all_covering_combos_include_p75_time": all(row.get("door_to_door_p75_minutes") is not None for row in covering),
            "candidate_pool_requires_effort_metrics": require_effort_metrics,
            "candidate_pool_excludes_draft_routes": exclude_draft_candidates,
        },
        "current": current,
        "best": {
            "by_on_foot_miles": best_by_on_foot,
            "by_door_to_door_p75_minutes": best_by_time,
            "by_ascent_ft": best_by_ascent,
            "by_grade_adjusted_miles": best_by_grade,
        },
        "comparisons": comparisons,
        "dominance": dominance[:max_results],
        "best_dominant_combo": dominant[0] if dominant else None,
        "top_covering_combos_by_on_foot": covering,
        "candidate_pool": [compact_candidate(candidate, target_ids) for candidate in pool],
        "caveats": [
            "This is a generated-candidate boundary challenge. It can disprove obvious generated alternatives, but it does not replace manual GPX/local-map design.",
            "Door-to-door p75 times are summed per route, so splitting a boundary across more starts is penalized with each route's drive and prep time.",
            "Extra official segments are reported because a superset route may only be useful if it replaces work from another package.",
        ],
    }


def combo_label(row: dict[str, Any] | None) -> str:
    if not row:
        return "none"
    return (
        f"{row['on_foot_miles']} mi, {row.get('door_to_door_p75_minutes') or 'n/a'} min p75, "
        f"{row.get('ascent_ft') or 'n/a'} ft ascent, {row['route_count']} routes: "
        + ", ".join(str(value) for value in row.get("candidate_ids") or [])
    )


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    current = report["current"]
    lines = [
        "# Route Boundary Challenge",
        "",
        f"Objective: {report['objective']}",
        "",
        f"Packages: {', '.join(str(value) for value in report['package_numbers'])}",
        f"Areas: {', '.join(report['package_titles'])}",
        "",
        "## Summary",
        "",
        f"- Target segments: {summary['target_segment_count']}",
        f"- Target official miles: {summary['target_official_miles']}",
        f"- Current: {combo_label(current)}",
        f"- Candidate pool: {summary['candidate_pool_count']}",
        f"- Candidate pool requires DEM ascent + p75 time: {summary['candidate_pool_requires_effort_metrics']}",
        f"- Candidate pool excludes draft routes: {summary['candidate_pool_excludes_draft_routes']}",
        f"- Covering combos returned: {summary['covering_combo_count_returned']}",
        f"- Better generated metric count: {summary['better_generated_metric_count']}",
        f"- Dominant generated combo count: {summary['dominant_generated_combo_count']}",
        f"- Generated combo beats current: {summary['generated_combo_beats_current']}",
        f"- All returned combos include DEM ascent: {summary['all_covering_combos_include_elevation']}",
        f"- All returned combos include p75 time: {summary['all_covering_combos_include_p75_time']}",
        "",
        "## Best Generated Covers",
        "",
        f"- By on-foot miles: {combo_label(report['best']['by_on_foot_miles'])}",
        f"- By p75 door-to-door time: {combo_label(report['best']['by_door_to_door_p75_minutes'])}",
        f"- By ascent: {combo_label(report['best']['by_ascent_ft'])}",
        f"- By grade-adjusted miles: {combo_label(report['best']['by_grade_adjusted_miles'])}",
        f"- Best dominant combo: {combo_label((report.get('best_dominant_combo') or {}).get('combo'))}",
        "",
        "## Metric Comparisons",
        "",
        "| Metric | Status | Current | Best | Delta | Best candidate ids |",
        "|---|---|---:|---:|---:|---|",
    ]
    for comparison in report["comparisons"]:
        lines.append(
            f"| {comparison['metric']} | {comparison['status']} | {comparison['current']} | "
            f"{comparison['best']} | {comparison['delta']} | {', '.join(str(value) for value in comparison.get('best_candidate_ids') or [])} |"
        )
    lines.extend(
        [
            "",
            "## Top Covers By On-Foot Miles",
            "",
            "| On-foot | P75 min | Ascent | Grade-adjusted | Routes | Extra segs | Candidate ids |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["top_covering_combos_by_on_foot"][:10]:
        lines.append(
            f"| {row['on_foot_miles']} | {row.get('door_to_door_p75_minutes')} | {row.get('ascent_ft')} | "
            f"{row.get('grade_adjusted_miles')} | {row['route_count']} | {row['extra_segment_count']} | "
            f"{', '.join(str(value) for value in row.get('candidate_ids') or [])} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--official-segments-geojson", type=Path, default=DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    parser.add_argument("--candidate-source", action="append", type=Path, dest="candidate_sources")
    parser.add_argument("--package-number", action="append", type=int, dest="package_numbers")
    parser.add_argument("--max-routes", type=int, default=5)
    parser.add_argument("--max-pool", type=int, default=36)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--allow-missing-effort-metrics",
        action="store_true",
        help="Allow candidates without DEM ascent or p75 time into the comparison pool.",
    )
    parser.add_argument(
        "--allow-draft-candidates",
        action="store_true",
        help="Allow draft generated candidates into the comparison pool.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_sources = args.candidate_sources or alternatives.DEFAULT_CANDIDATE_SOURCES
    package_numbers = args.package_numbers or DEFAULT_PACKAGE_NUMBERS
    report = build_report(
        read_json(args.map_data_json),
        read_json(args.official_segments_geojson),
        candidate_sources,
        package_numbers,
        max_routes=args.max_routes,
        max_pool=args.max_pool,
        max_results=args.max_results,
        require_effort_metrics=not args.allow_missing_effort_metrics,
        exclude_draft_candidates=not args.allow_draft_candidates,
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
