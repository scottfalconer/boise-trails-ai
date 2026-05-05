#!/usr/bin/env python3
"""Build the user-facing human loop/block plan from the combo route pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_day_packager import (  # noqa: E402
    manual_design_area_for_candidate_ids,
    render_html as render_package_map_html,
    render_outing_menu_markdown,
)
from personal_route_planner import read_json  # noqa: E402


DEFAULT_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
DEFAULT_PACKAGE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1.json"
DEFAULT_PACKAGE_MAP_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1-map-data.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "human-loop-plan-v1"
DEFAULT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTING_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_MANUAL_DESIGN_JSON = YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json"
DEFAULT_MANUAL_DESIGN_REPORT_JSON = (
    YEAR_DIR / "outputs" / "private" / "route-blocks" / "package16-manual-route-design-v1.json"
)


NECESSARY_GRINDER_TERMS = {
    "cartwright",
    "cervidae",
    "dry creek",
    "harlow",
    "harris ridge",
    "mores",
    "oregon",
    "shingle",
    "spring creek",
    "stack rock",
    "sweet connie",
    "watchman",
}


def package_status(package: dict[str, Any], map_data: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    reasons = []
    if map_data and manual_design_area_for_candidate_ids(
        map_data,
        package.get("package_number"),
        [str(candidate_id) for candidate_id in package.get("component_candidate_ids") or []],
    ):
        return "manual_design_area", ["coverage_placeholder_needs_human_route_design"]
    ratio = float(package.get("ratio") or 0.0)
    name = str(package.get("block_name") or "").lower()
    if package.get("trailhead_count", 0) > 1:
        reasons.append("split_across_nearby_route_components")
    if package.get("component_routes_under_1_official_mile"):
        reasons.append("tiny_segments_absorbed")
    if package.get("boundary_review"):
        reasons.append("block_boundary_review_recorded")
    if ratio > 2.0 or any(term in name for term in NECESSARY_GRINDER_TERMS):
        reasons.append("necessary_grinder_or_geography_locked")
        return "necessary_grinder", reasons
    if package.get("trailhead_count", 0) > 1 or package.get("component_route_count", 0) > 1:
        return "accepted_split_block", reasons
    return "primary_loop_block", reasons


def build_human_plan(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    package_map: dict[str, Any],
    map_html_path: Path,
) -> dict[str, Any]:
    route_statuses = {str(route.get("route_status")) for route in route_pass.get("routes") or []}
    map_validation = package_map.get("map_validation") or {}
    packages = []
    for package in package_pass.get("packages") or []:
        status, reasons = package_status(package, package_map)
        packages.append(
            {
                **package,
                "human_plan_status": status,
                "human_plan_reasons": reasons,
            }
        )
    status_counts: dict[str, int] = {}
    for package in packages:
        status_counts[package["human_plan_status"]] = status_counts.get(package["human_plan_status"], 0) + 1
    unresolved = []
    if package_pass.get("summary", {}).get("covered_segment_count") != 251:
        unresolved.append("coverage_not_complete")
    if route_statuses != {"graph_validated"}:
        unresolved.append("non_graph_validated_route_components")
    if map_validation.get("rendered_passed") is not True:
        unresolved.append("map_render_validation_failed")
    return {
        "planning_status": "human_loop_plan",
        "summary": {
            **(package_pass.get("summary") or {}),
            "route_component_count": route_pass.get("summary", {}).get("selected_route_count"),
            "status_counts": status_counts,
            "unresolved_blocker_count": len(unresolved),
            "unresolved_blockers": unresolved,
            "all_route_components_graph_validated": route_statuses == {"graph_validated"},
            "map_rendered_passed": map_validation.get("rendered_passed") is True,
            "map_html": str(map_html_path),
            "manual_design_area_count": status_counts.get("manual_design_area", 0),
        },
        "packages": packages,
        "routes": route_pass.get("routes") or [],
        "caveats": [
            "This is a planning route book, not a day-of conditions clearance. Ridge to Rivers conditions/signage still need checking before each outing.",
            "Accepted split blocks intentionally keep short nearby components separate when merging them created excessive dead mileage.",
            "Necessary grinders are retained because the geography and single-car return-to-car constraint make them expensive but still required for 100 percent coverage.",
        ],
    }


def merge_manual_design_report(manual_design: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    if not manual_design or not report:
        return manual_design
    report_areas = {str(area.get("area_id")): area for area in report.get("areas") or []}
    merged = {**manual_design, "areas": []}
    for area in manual_design.get("areas") or []:
        report_area = report_areas.get(str(area.get("area_id"))) or {}
        enriched_area = {**area}
        for key in [
            "status",
            "recommendation",
            "default_split_probe",
            "current_demoted_on_foot_miles",
            "acceptance_target_on_foot_miles",
            "current_good_route",
        ]:
            if key in report_area:
                enriched_area[key] = report_area[key]
        if report.get("generated_route_artifacts"):
            enriched_area["generated_route_artifacts"] = report["generated_route_artifacts"]
        report_alternatives = {
            str(alternative.get("alternative_id")): alternative
            for alternative in report_area.get("alternatives") or []
        }
        enriched_area["alternatives"] = [
            {
                **alternative,
                **{
                    key: value
                    for key, value in (report_alternatives.get(str(alternative.get("alternative_id"))) or {}).items()
                    if key in {
                        "probe",
                        "target_on_foot_miles_text",
                        "route_design_status",
                        "generated_route_artifact",
                    }
                },
            }
            for alternative in area.get("alternatives") or []
        ]
        merged["areas"].append(enriched_area)
    return merged


def package_start_plan(package: dict[str, Any]) -> str:
    trailhead_count = int(package.get("trailhead_count") or 0)
    component_count = int(package.get("component_route_count") or 0)
    if trailhead_count > 1:
        return f"{trailhead_count} parked starts"
    if trailhead_count == 1 and component_count > 1:
        return f"1 parked start, {component_count} route components"
    if trailhead_count == 1:
        return "1 parked start"
    return f"{component_count} route components"


def render_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    status_counts = summary.get("status_counts") or {}
    lines = [
        "# 2026 Human Loop Plan v1",
        "",
        "Status: current user-facing route/block plan.",
        "",
        "## Summary",
        "",
        f"- Packages: {summary['package_count']}",
        f"- Route components: {summary['route_component_count']}",
        f"- Covered segments: {summary['covered_segment_count']} / 251",
        f"- Official miles: {summary['official_miles']}",
        f"- Total on-foot miles: {summary['total_on_foot_miles']}",
        f"- On-foot/official ratio: {summary['planwide_on_foot_to_official_ratio']}x",
        f"- Primary loop blocks: {status_counts.get('primary_loop_block', 0)}",
        f"- Accepted split blocks: {status_counts.get('accepted_split_block', 0)}",
        f"- Necessary grinders: {status_counts.get('necessary_grinder', 0)}",
        f"- Manual design areas: {status_counts.get('manual_design_area', 0)}",
        f"- Route components graph-validated: {summary['all_route_components_graph_validated']}",
        f"- Map rendered: {summary['map_rendered_passed']}",
        f"- Map: `{summary['map_html']}`",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in plan.get("caveats") or [])
    lines.append("- Package on-foot miles are totals if you do every listed parked start in that package; they are not additional mileage per start.")
    lines.extend(
        [
            "",
            "## Blocks",
            "",
            "| # | Block | Status | Plan | Trailhead(s) | Official mi | On-foot mi | Ratio | Why |",
            "|---:|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for package in plan.get("packages") or []:
        reasons = ", ".join(package.get("human_plan_reasons") or [])
        trailheads = ", ".join(package.get("trailheads") or [])
        lines.append(
            f"| {package['package_number']} | {package['block_name']} | {package['human_plan_status']} | "
            f"{package_start_plan(package)} | {trailheads} | {package['official_miles']} | "
            f"{package['on_foot_miles']} | {package['ratio']} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Route Components",
            "",
            "| # | Block | Route | Trailhead | Official mi | On-foot mi | Ratio |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for route in plan.get("routes") or []:
        lines.append(
            f"| {route['route_number']} | {route['block_name']} | {', '.join(route.get('trail_names') or [])} | "
            f"{route.get('trailhead') or ''} | {route['official_miles']} | {route['on_foot_miles']} | {route['ratio']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-pass-json", type=Path, default=DEFAULT_ROUTE_PASS_JSON)
    parser.add_argument("--package-pass-json", type=Path, default=DEFAULT_PACKAGE_PASS_JSON)
    parser.add_argument("--package-map-json", type=Path, default=DEFAULT_PACKAGE_MAP_JSON)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGN_JSON)
    parser.add_argument("--manual-design-report-json", type=Path, default=DEFAULT_MANUAL_DESIGN_REPORT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--map-html", type=Path, default=DEFAULT_MAP_HTML)
    parser.add_argument("--outing-menu-md", type=Path, default=DEFAULT_OUTING_MENU_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_pass = read_json(args.route_pass_json)
    package_pass = read_json(args.package_pass_json)
    package_map = read_json(args.package_map_json)
    manual_design = read_json(args.manual_design_json) if args.manual_design_json.exists() else {}
    manual_design_report = read_json(args.manual_design_report_json) if args.manual_design_report_json.exists() else {}
    if manual_design:
        manual_design = merge_manual_design_report(manual_design, manual_design_report)
        package_map["manual_design"] = manual_design
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    map_html_path = args.map_html
    outing_menu_md_path = args.outing_menu_md
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    map_html_path.parent.mkdir(parents=True, exist_ok=True)
    outing_menu_md_path.parent.mkdir(parents=True, exist_ok=True)
    map_html_path.write_text(render_package_map_html(package_map), encoding="utf-8")
    outing_menu_md_path.write_text(render_outing_menu_markdown(package_map, map_html_path), encoding="utf-8")
    plan = build_human_plan(route_pass, package_pass, package_map, map_html_path)
    write_json(json_path, plan)
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[
                path
                for path in [args.route_pass_json, args.package_pass_json, args.package_map_json, args.manual_design_json]
                + ([args.manual_design_report_json] if args.manual_design_report_json.exists() else [])
                if path.exists()
            ],
            outputs=[json_path, md_path, map_html_path, outing_menu_md_path],
            command="human_loop_plan.py",
            metadata={
                "unresolved_blocker_count": plan["summary"]["unresolved_blocker_count"],
                "covered_segment_count": plan["summary"]["covered_segment_count"],
                "route_component_count": plan["summary"]["route_component_count"],
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {map_html_path}")
    print(f"Wrote {outing_menu_md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
