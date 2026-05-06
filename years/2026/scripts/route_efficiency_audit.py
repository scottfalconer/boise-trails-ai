#!/usr/bin/env python3
"""Audit whether the current route set is proven efficient enough to trust."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_FIELD_PACKET_JSON = REPO_ROOT / "docs" / "field-packet" / "manifest.json"
DEFAULT_HUMAN_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "human-loop-plan-v1.json"
DEFAULT_PACKAGE16_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "package16-manual-route-design-v1.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "route-efficiency-audit-2026-05-06"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def component_rows(map_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            official = float(component.get("official_miles") or 0.0)
            on_foot = float(component.get("on_foot_miles") or 0.0)
            rows.append(
                {
                    "label": str(component.get("label") or package.get("package_number") or ""),
                    "package_number": package.get("package_number"),
                    "block_name": package.get("block_name") or "",
                    "trailhead": component.get("trailhead") or "",
                    "trails": component.get("trail_names") or [],
                    "official_miles": round(official, 2),
                    "on_foot_miles": round(on_foot, 2),
                    "overhead_miles": round(on_foot - official, 2),
                    "ratio": round(on_foot / official, 2) if official else None,
                    "total_minutes": component.get("total_minutes") or component.get("door_to_door_minutes"),
                    "candidate_ids": component.get("candidate_ids") or [component.get("candidate_id")],
                }
            )
    return rows


def route_rows(field_packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for route in field_packet.get("routes") or []:
        outing = route.get("outing") or {}
        official = float(outing.get("official_miles") or 0.0)
        on_foot = float(outing.get("on_foot_miles") or 0.0)
        rows.append(
            {
                "label": str(outing.get("label") or ""),
                "trailhead": outing.get("trailhead") or "",
                "trails": outing.get("trails") or [],
                "official_miles": round(official, 2),
                "on_foot_miles": round(on_foot, 2),
                "overhead_miles": round(on_foot - official, 2),
                "ratio": round(on_foot / official, 2) if official else None,
                "total_minutes": outing.get("total_minutes"),
            }
        )
    return rows


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    official = sum(float(row.get("official_miles") or 0.0) for row in rows)
    on_foot = sum(float(row.get("on_foot_miles") or 0.0) for row in rows)
    return {
        "official_miles": round(official, 2),
        "on_foot_miles": round(on_foot, 2),
        "ratio": round(on_foot / official, 3) if official else None,
        "count": len(rows),
    }


def top_rows(rows: list[dict[str, Any]], key: str, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key) or 0.0), reverse=True)[:limit]


def package16_summary(package16: dict[str, Any]) -> dict[str, Any]:
    areas = package16.get("areas") or []
    if not areas:
        return {}
    area = areas[0]
    current = area.get("current_placeholder") or {}
    best = area.get("default_split_probe") or area.get("current_good_route") or {}
    return {
        "status": area.get("status"),
        "decision": area.get("decision"),
        "held_official_miles": current.get("official_miles") or area.get("current_demoted_official_miles"),
        "held_on_foot_miles": current.get("on_foot_miles") or area.get("current_demoted_on_foot_miles"),
        "accepted_split_official_miles": best.get("official_miles"),
        "accepted_split_on_foot_miles": best.get("on_foot_miles"),
        "improvement_miles": best.get("improvement_vs_current_on_foot_miles"),
        "remaining_blocker": (area.get("current_good_route") or {}).get("remaining_blocker"),
    }


def build_audit(
    map_data: dict[str, Any],
    field_packet: dict[str, Any],
    human_plan: dict[str, Any],
    package16: dict[str, Any],
) -> dict[str, Any]:
    components = component_rows(map_data)
    runnable = route_rows(field_packet)
    map_summary = map_data.get("summary") or {}
    field_summary = field_packet.get("summary") or {}
    human_summary = human_plan.get("summary") or {}
    manual_holds = field_packet.get("manual_holds") or []
    component_totals = totals(components)
    runnable_totals = totals(runnable)
    high_ratio_components = [row for row in components if float(row.get("ratio") or 0.0) > 2.0]
    high_overhead_components = [row for row in components if float(row.get("overhead_miles") or 0.0) >= 6.0]
    gates = [
        {
            "gate": "Full official coverage is represented",
            "status": "passed" if int(map_summary.get("covered_segment_count") or 0) == 251 else "failed",
            "evidence": f"map_data covered_segment_count={map_summary.get('covered_segment_count')}; official_miles={map_summary.get('official_miles')}",
        },
        {
            "gate": "Runnable field packet covers all official work",
            "status": "failed" if manual_holds else "passed",
            "evidence": f"runnable_official_miles={runnable_totals['official_miles']}; manual_holds={len(manual_holds)}",
        },
        {
            "gate": "Planwide on-foot/official ratio is within preferred 1.6x target",
            "status": "passed" if float(map_summary.get("planwide_on_foot_to_official_ratio") or 99.0) <= 1.6 else "failed",
            "evidence": f"current_all_component_ratio={map_summary.get('planwide_on_foot_to_official_ratio')}; runnable_ratio={runnable_totals['ratio']}",
        },
        {
            "gate": "No unresolved manual route-design area remains",
            "status": "failed" if int(human_summary.get("manual_design_area_count") or 0) else "passed",
            "evidence": f"manual_design_area_count={human_summary.get('manual_design_area_count')}; package16={package16_summary(package16).get('status')}",
        },
        {
            "gate": "No route exceeds 2.0x without a proven better-alternative comparison",
            "status": "failed" if high_ratio_components else "passed",
            "evidence": f"components_over_2x={len(high_ratio_components)}",
        },
        {
            "gate": "Largest overhead routes have been manually challenged",
            "status": "failed" if high_overhead_components else "passed",
            "evidence": f"components_with_6+_overhead_miles={len(high_overhead_components)}",
        },
    ]
    achieved = all(gate["status"] == "passed" for gate in gates)
    return {
        "objective": "prove the current Boise Trails Challenge route set is as efficient as practical under the user's constraints",
        "verdict": "proven" if achieved else "not_proven",
        "achieved": achieved,
        "summary": {
            "all_component_totals": component_totals,
            "runnable_field_packet_totals": runnable_totals,
            "manual_hold_count": len(manual_holds),
            "human_loop_plan_ratio": human_summary.get("planwide_on_foot_to_official_ratio"),
            "human_loop_plan_on_foot_miles": human_summary.get("total_on_foot_miles"),
        },
        "gates": gates,
        "package16": package16_summary(package16),
        "worst_ratio_components": top_rows(components, "ratio"),
        "worst_overhead_components": top_rows(components, "overhead_miles"),
        "longest_components": top_rows(components, "total_minutes"),
        "next_required_work": [
            "Integrate or explicitly reject the Package 16 accepted split probe in the runnable field packet.",
            "Manually challenge the highest-overhead routes first: Harlow/Spring north pod, Freestone/Three Bears/Shane/Curlew, Dry Creek lower, Cartwright/Peggy, and Bogus day 2.",
            "For each challenged route, record the best alternative found, its official miles, on-foot miles, parking/start, and why the current route wins or loses.",
            "Only then re-run this audit and consider tightening the preferred ratio gate below 1.7x.",
        ],
    }


def render_md(audit: dict[str, Any]) -> str:
    lines = [
        "# Route Efficiency Audit",
        "",
        f"Objective: {audit['objective']}",
        "",
        f"Verdict: {audit['verdict']}",
        f"Achieved: {audit['achieved']}",
        "",
        "## Summary",
        "",
    ]
    summary = audit["summary"]
    lines.extend(
        [
            f"- All-component plan: {summary['all_component_totals']['official_miles']} official mi / {summary['all_component_totals']['on_foot_miles']} on-foot mi / {summary['all_component_totals']['ratio']}x",
            f"- Runnable field packet: {summary['runnable_field_packet_totals']['official_miles']} official mi / {summary['runnable_field_packet_totals']['on_foot_miles']} on-foot mi / {summary['runnable_field_packet_totals']['ratio']}x",
            f"- Manual holds: {summary['manual_hold_count']}",
            f"- Human-loop plan: {summary['human_loop_plan_on_foot_miles']} on-foot mi / {summary['human_loop_plan_ratio']}x",
            "",
            "## Gates",
            "",
            "| Gate | Status | Evidence |",
            "|---|---|---|",
        ]
    )
    for gate in audit["gates"]:
        lines.append(f"| {gate['gate']} | {gate['status']} | {gate['evidence']} |")
    lines.extend(["", "## Package 16", ""])
    for key, value in audit.get("package16", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Worst Ratio Components", "", "| Label | Trailhead | Official | On-foot | Ratio | Trails |", "|---|---|---:|---:|---:|---|"])
    for row in audit["worst_ratio_components"]:
        lines.append(
            f"| {row['label']} | {row['trailhead']} | {row['official_miles']} | {row['on_foot_miles']} | {row['ratio']} | {', '.join(row['trails'])} |"
        )
    lines.extend(["", "## Worst Overhead Components", "", "| Label | Trailhead | Official | On-foot | Overhead | Ratio | Trails |", "|---|---|---:|---:|---:|---:|---|"])
    for row in audit["worst_overhead_components"]:
        lines.append(
            f"| {row['label']} | {row['trailhead']} | {row['official_miles']} | {row['on_foot_miles']} | {row['overhead_miles']} | {row['ratio']} | {', '.join(row['trails'])} |"
        )
    lines.extend(["", "## Next Required Work", ""])
    lines.extend(f"- {step}" for step in audit["next_required_work"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--field-packet-json", type=Path, default=DEFAULT_FIELD_PACKET_JSON)
    parser.add_argument("--human-plan-json", type=Path, default=DEFAULT_HUMAN_PLAN_JSON)
    parser.add_argument("--package16-json", type=Path, default=DEFAULT_PACKAGE16_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(
        read_json(args.map_data_json),
        read_json(args.field_packet_json),
        read_json(args.human_plan_json),
        read_json(args.package16_json),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.basename}.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"{args.basename}.md").write_text(render_md(audit), encoding="utf-8")
    print(f"Wrote {args.output_dir / f'{args.basename}.json'}")
    print(f"Wrote {args.output_dir / f'{args.basename}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
