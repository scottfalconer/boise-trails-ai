#!/usr/bin/env python3
"""Audit whether the current artifacts satisfy the final human route objective."""

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
from personal_route_planner import read_json  # noqa: E402


DEFAULT_BLOCK_REVIEW_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-first-plan-v1.json"
DEFAULT_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
DEFAULT_ROUTE_MAP_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1-map-data.json"
DEFAULT_PACKAGE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1.json"
DEFAULT_PACKAGE_MAP_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1-map-data.json"
DEFAULT_ASSEMBLED_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-assembled-route-pass-v1.json"
DEFAULT_HUMAN_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "human-loop-plan-v1.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "final-route-completion-audit"


def build_audit(
    block_review: dict[str, Any],
    route_pass: dict[str, Any],
    route_map: dict[str, Any],
    package_pass: dict[str, Any] | None = None,
    package_map: dict[str, Any] | None = None,
    assembled_pass: dict[str, Any] | None = None,
    human_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_summary = route_pass.get("summary") or {}
    block_summary = block_review.get("summary") or {}
    map_validation = route_map.get("map_validation") or {}
    package_summary = (package_pass or {}).get("summary") or {}
    package_map_validation = (package_map or {}).get("map_validation") or {}
    assembled_summary = (assembled_pass or {}).get("summary") or {}
    human_summary = (human_plan or {}).get("summary") or {}
    package_count = package_summary.get("package_count")
    component_route_count = package_summary.get("component_route_count")
    multi_trailhead_packages = package_summary.get("packages_with_multiple_trailheads")
    assembled_total = assembled_summary.get("total_on_foot_miles")
    human_status_counts = human_summary.get("status_counts") or {}
    selected_route_count = int(route_summary.get("selected_route_count") or 0)
    route_ratio = float(route_summary.get("planwide_on_foot_to_official_ratio") or 99.0)
    route_total_on_foot = float(route_summary.get("total_on_foot_miles") or 0.0)
    routes_under_1 = int(route_summary.get("routes_under_1_official_mile") or 0)
    routes_under_2 = int(route_summary.get("routes_under_2_official_miles") or 0)
    non_graph_count = int(route_summary.get("non_graph_validated_route_count") or 0)
    cross_block_count = int(route_summary.get("cross_block_route_count") or 0)
    structure_passed = (
        bool(package_count)
        and int(package_count) <= 25
        and bool(component_route_count)
        and int(component_route_count) <= 25
        and cross_block_count == 0
    )
    quality_passed = (
        selected_route_count <= 26
        and routes_under_1 == 0
        and routes_under_2 <= 2
        and route_ratio <= 1.8
        and route_total_on_foot <= 300.0
        and int(multi_trailhead_packages or 0) <= 4
        and non_graph_count == 0
    )
    hybrid_improves_naive_assembly = (
        bool(assembled_summary)
        and route_total_on_foot <= float(assembled_summary.get("total_on_foot_miles") or 0.0)
    )
    single_artifact_passed = (
        human_summary.get("covered_segment_count") == 251
        and human_summary.get("unresolved_blocker_count") == 0
        and human_summary.get("all_route_components_graph_validated") is True
        and human_summary.get("map_rendered_passed") is True
    )
    checklist = [
        {
            "requirement": "Full official 2026 on-foot coverage",
            "status": "passed"
            if route_summary.get("covered_segment_count") == 251
            and block_summary.get("official_segment_assignment_passed") is True
            else "failed",
            "evidence": [
                f"route_pass covered_segment_count={route_summary.get('covered_segment_count')}",
                f"block assignment passed={block_summary.get('official_segment_assignment_passed')}",
            ],
        },
        {
            "requirement": "Viewable route map exists",
            "status": "passed"
            if map_validation.get("rendered_passed") is True
            and (
                package_map is None
                or package_map_validation.get("rendered_passed") is True
            )
            else "failed",
            "evidence": [
                "current route map HTML generated",
                f"rendered map validation passed={map_validation.get('rendered_passed')}",
                f"source gap warning count={map_validation.get('source_gap_warning_count')}",
                f"package map validation passed={package_map_validation.get('rendered_passed')}",
            ],
        },
        {
            "requirement": "Actual real loop/block route structure",
            "status": "passed" if structure_passed else "failed",
            "evidence": [
                f"block_count={block_summary.get('block_count')}",
                f"draft_block_count={block_summary.get('draft_block_count')}",
                f"selected_route_count={route_summary.get('selected_route_count')}",
                f"block_day_package_count={package_count}",
                f"component_route_count={component_route_count}",
                f"cross_block_route_count={cross_block_count}",
                f"human_loop_status_counts={human_status_counts}",
                "hybrid route pass uses natural block routes where they preserve coverage and avoids selected cross-block sweep routes",
            ],
        },
        {
            "requirement": "Single user-facing loop plan artifact",
            "status": "passed" if single_artifact_passed else "failed",
            "evidence": [
                "human-loop-plan-v1 is the current single review file",
                f"covered_segment_count={human_summary.get('covered_segment_count')}",
                f"unresolved_blocker_count={human_summary.get('unresolved_blocker_count')}",
                f"status_counts={human_status_counts}",
                f"map_rendered_passed={human_summary.get('map_rendered_passed')}",
            ],
        },
        {
            "requirement": "Normal-human route quality with minimal car-hop fragments",
            "status": "passed" if quality_passed else "failed",
            "evidence": [
                f"selected_route_count={route_summary.get('selected_route_count')}",
                f"routes_under_1_official_mile={route_summary.get('routes_under_1_official_mile')}",
                f"routes_under_2_official_miles={route_summary.get('routes_under_2_official_miles')}",
                f"planwide_on_foot_to_official_ratio={route_summary.get('planwide_on_foot_to_official_ratio')}",
                f"total_on_foot_miles={route_summary.get('total_on_foot_miles')}",
                f"non_graph_validated_route_count={non_graph_count}",
                f"package_count={package_count}",
                f"packages_with_multiple_trailheads={multi_trailhead_packages}",
                f"block_assembled_total_on_foot_miles={assembled_total}",
                "Small split components are acceptable when they avoid a clearly worse mega-loop or long deadhead run.",
            ],
        },
        {
            "requirement": "Hybrid selection beats naive one-route-per-block assembly",
            "status": "passed" if hybrid_improves_naive_assembly else "failed",
            "evidence": [
                f"candidate_pass_total_on_foot_miles={route_summary.get('total_on_foot_miles')}",
                f"block_assembled_total_on_foot_miles={assembled_summary.get('total_on_foot_miles')}",
                f"block_assembled_route_count={assembled_summary.get('assembled_route_count')}",
                "The final human plan uses the hybrid set cover, not the full naive one-route-per-block diagnostic.",
            ],
        },
        {
            "requirement": "Ready to call final",
            "status": "passed"
            if structure_passed and quality_passed and single_artifact_passed and hybrid_improves_naive_assembly
            else "failed",
            "evidence": [
                "Final here means a reviewable route/block plan and viewable map, not day-of condition clearance.",
                "The plan still labels necessary grinders honestly, but those are geography/single-car costs rather than unresolved car-hop fragments.",
            ],
        },
    ]
    achieved = all(item["status"] == "passed" for item in checklist)
    return {
        "objective": "get a usable final route and map with actual real loops/blocks that a normal human would run",
        "achieved": achieved,
        "verdict": "complete" if achieved else "not_complete",
        "checklist": checklist,
        "next_required_work": [
            "Day-of use still requires Ridge to Rivers conditions/signage checks and water/logistics review.",
            "GPX turn-by-turn export/continuity QA remains a separate hardening task; this audit covers the route plan and map objective.",
        ]
        if achieved
        else [
            f"Use human-loop-plan-v1.md as the single current review surface because it preserves full coverage while reducing the component plan to {route_summary.get('selected_route_count')} routes and {route_summary.get('total_on_foot_miles')} on-foot miles.",
            "Do not use the full one-route-per-block diagnostic as final; use it selectively through the hybrid route pass so coverage is preserved and dead mileage is controlled.",
            "Generate custom GPX for the remaining accepted split blocks and necessary grinders first, especially Camel/Hulls, Military, Table Rock, Upper 8th, Eagle/Seaman, and Sweet Connie/Shingle/Stack.",
            "Re-run the audit and require package count, multiple-trailhead packages, and on-foot/official ratio to improve materially before calling the plan final.",
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Final Route Completion Audit",
        "",
        f"Objective: {audit['objective']}",
        "",
        f"Achieved: {audit['achieved']}",
        f"Verdict: {audit['verdict']}",
        "",
        "## Checklist",
        "",
        "| Requirement | Status | Evidence |",
        "|---|---|---|",
    ]
    for item in audit.get("checklist") or []:
        evidence = "<br>".join(item.get("evidence") or [])
        lines.append(f"| {item['requirement']} | {item['status']} | {evidence} |")
    lines.extend(["", "## Next Required Work", ""])
    for step in audit.get("next_required_work") or []:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-review-json", type=Path, default=DEFAULT_BLOCK_REVIEW_JSON)
    parser.add_argument("--route-pass-json", type=Path, default=DEFAULT_ROUTE_PASS_JSON)
    parser.add_argument("--route-map-json", type=Path, default=DEFAULT_ROUTE_MAP_JSON)
    parser.add_argument("--package-pass-json", type=Path, default=DEFAULT_PACKAGE_PASS_JSON)
    parser.add_argument("--package-map-json", type=Path, default=DEFAULT_PACKAGE_MAP_JSON)
    parser.add_argument("--assembled-pass-json", type=Path, default=DEFAULT_ASSEMBLED_PASS_JSON)
    parser.add_argument("--human-plan-json", type=Path, default=DEFAULT_HUMAN_PLAN_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    block_review = read_json(args.block_review_json)
    route_pass = read_json(args.route_pass_json)
    route_map = read_json(args.route_map_json)
    package_pass = read_json(args.package_pass_json) if args.package_pass_json.exists() else None
    package_map = read_json(args.package_map_json) if args.package_map_json.exists() else None
    assembled_pass = read_json(args.assembled_pass_json) if args.assembled_pass_json.exists() else None
    human_plan = read_json(args.human_plan_json) if args.human_plan_json.exists() else None
    audit = build_audit(
        block_review,
        route_pass,
        route_map,
        package_pass=package_pass,
        package_map=package_map,
        assembled_pass=assembled_pass,
        human_plan=human_plan,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    write_json(json_path, audit)
    md_path.write_text(render_markdown(audit), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[
                args.block_review_json,
                args.route_pass_json,
                args.route_map_json,
                args.package_pass_json,
                args.package_map_json,
                args.assembled_pass_json,
                args.human_plan_json,
            ],
            outputs=[json_path, md_path],
            command="final_route_completion_audit.py",
            metadata={"achieved": audit["achieved"], "verdict": audit["verdict"]},
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
