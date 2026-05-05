#!/usr/bin/env python3
"""Build a block-first review artifact from the selected runbook."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_BLOCKS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-route-blocks-v1.json"
DEFAULT_RUNBOOK_JSON = YEAR_DIR / "outputs" / "private" / "2026-personal-ideal-plan.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "block-first-plan-v1"


def normalize_trail_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def trail_name_from_segment_name(value: str) -> str:
    return re.sub(r"\s+\d+$", "", value.strip())


def official_miles_from_feature(feature: dict[str, Any]) -> float:
    props = feature.get("properties") or {}
    if props.get("LengthFt") is None:
        return 0.0
    return float(props["LengthFt"]) / 5280.0


def load_official_trails(official_geojson: Path) -> dict[str, dict[str, Any]]:
    data = read_json(official_geojson)
    trails: dict[str, dict[str, Any]] = {}
    for feature in data.get("features") or []:
        props = feature.get("properties") or {}
        seg_name = str(props.get("segName") or "")
        trail_name = trail_name_from_segment_name(seg_name)
        key = normalize_trail_name(trail_name)
        trail = trails.setdefault(
            key,
            {
                "trail_name": trail_name,
                "segment_ids": [],
                "official_miles": 0.0,
                "direction_counts": Counter(),
            },
        )
        trail["segment_ids"].append(int(props["segId"]))
        trail["official_miles"] += official_miles_from_feature(feature)
        trail["direction_counts"][str(props.get("direction") or "both")] += 1
    for trail in trails.values():
        trail["segment_ids"] = sorted(trail["segment_ids"])
        trail["official_miles"] = round(trail["official_miles"], 2)
        trail["direction_counts"] = dict(trail["direction_counts"])
    return trails


def build_block_index(blocks_config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    trail_to_block: dict[str, dict[str, Any]] = {}
    duplicates = []
    for block in blocks_config.get("blocks") or []:
        for trail_name in block.get("trail_names") or []:
            key = normalize_trail_name(str(trail_name))
            if key in trail_to_block:
                duplicates.append(
                    {
                        "trail_name": trail_name,
                        "first_block_id": trail_to_block[key]["block_id"],
                        "second_block_id": block["block_id"],
                    }
                )
            trail_to_block[key] = block
    return trail_to_block, duplicates


def runbook_outings(runbook: dict[str, Any]) -> list[dict[str, Any]]:
    outings = []
    for day in runbook.get("days") or []:
        for outing in day.get("outings") or []:
            outings.append({**outing, "_date": day.get("date"), "_day": day})
    return outings


def block_ids_for_outing(
    outing: dict[str, Any],
    trail_to_block: dict[str, dict[str, Any]],
) -> list[str]:
    block_ids = []
    for trail_name in outing.get("trail_names") or []:
        block = trail_to_block.get(normalize_trail_name(str(trail_name)))
        if block:
            block_ids.append(str(block["block_id"]))
        else:
            block_ids.append("unassigned")
    return sorted(set(block_ids))


def classify_block_readiness(
    block: dict[str, Any],
    block_summary: dict[str, Any],
    acceptance: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = []
    if block.get("status") == "boundary_review":
        reasons.append("boundary_review_required")
    official_miles = float(block_summary.get("official_miles") or 0.0)
    min_official = float(acceptance.get("min_official_miles_unless_geography_locked") or 0.0)
    if official_miles < min_official and not block.get("geography_locked"):
        reasons.append("below_min_official_miles_without_geography_lock")
    if block_summary.get("current_standalone_sub1_count"):
        reasons.append("current_plan_has_sub1_fragment")
    if block_summary.get("current_standalone_sub2_count"):
        reasons.append("current_plan_has_sub2_fragment")
    if block_summary.get("current_max_trailheads_in_single_day", 0) > int(
        acceptance.get("max_normal_trailheads_per_day") or 1
    ):
        reasons.append("current_plan_has_trailhead_hopping")
    if block_summary.get("cross_block_current_outing_count"):
        reasons.append("current_plan_crosses_block_boundary")
    if block_summary.get("unassigned_trail_count"):
        reasons.append("unassigned_trails_present")

    if block.get("status") == "candidate_block" and not reasons:
        return "schedule_candidate_after_gpx", []
    if block.get("geography_locked") and reasons == ["below_min_official_miles_without_geography_lock"]:
        return "schedule_candidate_after_gpx", []
    return "draft_needs_route_design", reasons


def summarize_fragmentation(runbook: dict[str, Any]) -> dict[str, Any]:
    days = runbook.get("days") or []
    outings = runbook_outings(runbook)
    official_miles = [float(outing.get("new_official_miles") or 0.0) for outing in outings]
    trailheads_by_day = [set(day.get("trailheads") or []) for day in days]
    return {
        "scheduled_days": len(days),
        "executable_outings": len(outings),
        "outings_under_1_official_mile": sum(1 for miles in official_miles if miles < 1.0),
        "outings_under_2_official_miles": sum(1 for miles in official_miles if miles < 2.0),
        "days_with_3_plus_outings": sum(1 for day in days if len(day.get("outings") or []) >= 3),
        "days_with_3_plus_trailheads": sum(1 for trailheads in trailheads_by_day if len(trailheads) >= 3),
        "unique_trailheads": len({trailhead for trailheads in trailheads_by_day for trailhead in trailheads}),
        "route_label_counts": dict(Counter(outing.get("route_label") for outing in outings)),
    }


def build_block_first_plan(
    blocks_config: dict[str, Any],
    runbook: dict[str, Any],
    official_trails: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trail_to_block, duplicates = build_block_index(blocks_config)
    official_by_block: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "trail_names": [],
            "segment_ids": [],
            "official_miles": 0.0,
            "direction_counts": Counter(),
        }
    )
    unassigned_official_trails = []
    for key, trail in official_trails.items():
        block = trail_to_block.get(key)
        if not block:
            unassigned_official_trails.append(trail["trail_name"])
            block_id = "unassigned"
        else:
            block_id = str(block["block_id"])
        official_by_block[block_id]["trail_names"].append(trail["trail_name"])
        official_by_block[block_id]["segment_ids"].extend(trail["segment_ids"])
        official_by_block[block_id]["official_miles"] += float(trail["official_miles"])
        official_by_block[block_id]["direction_counts"].update(trail["direction_counts"])

    outings_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_block_outings = []
    for outing in runbook_outings(runbook):
        block_ids = block_ids_for_outing(outing, trail_to_block)
        if len(block_ids) > 1:
            cross_block_outings.append(
                {
                    "date": outing.get("_date"),
                    "outing_id": outing.get("outing_id"),
                    "trail_names": outing.get("trail_names") or [],
                    "block_ids": block_ids,
                    "official_miles": outing.get("new_official_miles"),
                    "on_foot_miles": outing.get("estimated_total_on_foot_miles"),
                }
            )
        for block_id in block_ids:
            outings_by_block[block_id].append(outing)

    block_summaries = []
    block_by_id = {str(block["block_id"]): block for block in blocks_config.get("blocks") or []}
    acceptance = blocks_config.get("acceptance_criteria") or {}
    for block in blocks_config.get("blocks") or []:
        block_id = str(block["block_id"])
        official = official_by_block.get(block_id) or {
            "trail_names": [],
            "segment_ids": [],
            "official_miles": 0.0,
            "direction_counts": Counter(),
        }
        outings = outings_by_block.get(block_id) or []
        days = defaultdict(list)
        for outing in outings:
            days[str(outing.get("_date"))].append(outing)
        current_standalone_sub1 = [
            outing
            for outing in outings
            if len(block_ids_for_outing(outing, trail_to_block)) == 1
            and float(outing.get("new_official_miles") or 0.0) < 1.0
        ]
        current_standalone_sub2 = [
            outing
            for outing in outings
            if len(block_ids_for_outing(outing, trail_to_block)) == 1
            and float(outing.get("new_official_miles") or 0.0) < 2.0
        ]
        trailheads_by_day = {
            day: {
                str(outing.get("trailhead"))
                for outing in day_outings
                if outing.get("trailhead")
            }
            for day, day_outings in days.items()
        }
        cross_count = sum(
            1
            for outing in outings
            if len(block_ids_for_outing(outing, trail_to_block)) > 1
        )
        summary = {
            "block_id": block_id,
            "name": block.get("name"),
            "configured_status": block.get("status"),
            "rationale": block.get("rationale"),
            "boundary_question": block.get("boundary_question"),
            "geography_locked": bool(block.get("geography_locked")),
            "official_trail_count": len(official["trail_names"]),
            "official_segment_count": len(set(official["segment_ids"])),
            "official_miles": round(float(official["official_miles"]), 2),
            "direction_counts": dict(official["direction_counts"]),
            "current_outing_count": len({str(outing.get("outing_id")) for outing in outings}),
            "current_day_count": len(days),
            "current_trailhead_count": len(
                {
                    str(outing.get("trailhead"))
                    for outing in outings
                    if outing.get("trailhead")
                }
            ),
            "current_max_trailheads_in_single_day": max(
                [len(trailheads) for trailheads in trailheads_by_day.values()] or [0]
            ),
            "current_standalone_sub1_count": len(current_standalone_sub1),
            "current_standalone_sub2_count": len(current_standalone_sub2),
            "cross_block_current_outing_count": cross_count,
            "official_trails": sorted(official["trail_names"]),
            "current_outings": [
                {
                    "date": outing.get("_date"),
                    "outing_id": outing.get("outing_id"),
                    "trail_names": outing.get("trail_names") or [],
                    "official_miles": outing.get("new_official_miles"),
                    "on_foot_miles": outing.get("estimated_total_on_foot_miles"),
                    "trailhead": outing.get("trailhead"),
                    "route_label": outing.get("route_label"),
                    "block_ids": block_ids_for_outing(outing, trail_to_block),
                }
                for outing in sorted(outings, key=lambda item: (str(item.get("_date")), str(item.get("outing_id"))))
            ],
            "absorption_candidates": [
                {
                    "date": outing.get("_date"),
                    "outing_id": outing.get("outing_id"),
                    "trail_names": outing.get("trail_names") or [],
                    "official_miles": outing.get("new_official_miles"),
                    "on_foot_miles": outing.get("estimated_total_on_foot_miles"),
                    "trailhead": outing.get("trailhead"),
                }
                for outing in current_standalone_sub2
            ],
        }
        readiness, reasons = classify_block_readiness(block, summary, acceptance)
        summary["block_readiness"] = readiness
        summary["readiness_reasons"] = reasons
        block_summaries.append(summary)

    unassigned_summary = None
    if "unassigned" in official_by_block or "unassigned" in outings_by_block:
        official = official_by_block.get("unassigned") or {
            "trail_names": [],
            "segment_ids": [],
            "official_miles": 0.0,
            "direction_counts": Counter(),
        }
        unassigned_summary = {
            "block_id": "unassigned",
            "name": "Unassigned official trails",
            "official_trail_count": len(official["trail_names"]),
            "official_segment_count": len(set(official["segment_ids"])),
            "official_miles": round(float(official["official_miles"]), 2),
            "official_trails": sorted(official["trail_names"]),
            "current_outings": [
                {
                    "date": outing.get("_date"),
                    "outing_id": outing.get("outing_id"),
                    "trail_names": outing.get("trail_names") or [],
                    "official_miles": outing.get("new_official_miles"),
                    "on_foot_miles": outing.get("estimated_total_on_foot_miles"),
                }
                for outing in outings_by_block.get("unassigned") or []
            ],
        }
        block_summaries.append(unassigned_summary)

    assigned_segment_ids = {
        segment_id
        for block_id, official in official_by_block.items()
        if block_id != "unassigned"
        for segment_id in official["segment_ids"]
    }
    total_official_segment_ids = {
        segment_id for trail in official_trails.values() for segment_id in trail["segment_ids"]
    }
    draft_blocks = [
        block for block in block_summaries if block.get("block_readiness") == "draft_needs_route_design"
    ]
    return {
        "version": blocks_config.get("version"),
        "planning_status": "block_first_review_not_schedule",
        "fallback_plan": {
            "profile_name": runbook.get("profile_name"),
            "run_id": runbook.get("run_id"),
            "summary": runbook.get("summary") or {},
            "audit": runbook.get("audit") or {},
        },
        "fragmentation_summary": summarize_fragmentation(runbook),
        "acceptance_criteria": blocks_config.get("acceptance_criteria") or {},
        "manual_review_priorities": blocks_config.get("manual_review_priorities") or [],
        "summary": {
            "block_count": len(blocks_config.get("blocks") or []),
            "official_segments_assigned_to_blocks": len(assigned_segment_ids),
            "official_segments_total": len(total_official_segment_ids),
            "official_segment_assignment_passed": assigned_segment_ids == total_official_segment_ids,
            "unassigned_official_trail_count": len(unassigned_official_trails),
            "duplicate_configured_trail_count": len(duplicates),
            "draft_block_count": len(draft_blocks),
            "schedule_candidate_block_count": len(block_summaries) - len(draft_blocks) - (1 if unassigned_summary else 0),
            "cross_block_current_outing_count": len(cross_block_outings),
        },
        "blocks": block_summaries,
        "cross_block_current_outings": cross_block_outings,
        "config_validation": {
            "duplicate_configured_trails": duplicates,
            "unassigned_official_trails": sorted(unassigned_official_trails),
        },
        "next_steps": [
            "Treat private-human-100-v2 as the validated coverage fallback, not the experience-optimized final plan.",
            "Generate continuous GPX candidates for draft_needs_route_design blocks before scheduling.",
            "Reject block routes that keep standalone sub-1 or sub-2 official-mile errands without written justification.",
            "After block GPX passes, run a calendar schedule from block outings instead of the 60-unit execution menu.",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    fallback_summary = (plan.get("fallback_plan") or {}).get("summary") or {}
    frag = plan.get("fragmentation_summary") or {}
    lines = [
        "# 2026 Block-First Route Review",
        "",
        "Status: draft route-block review, not a runnable schedule.",
        "",
        "## Why this exists",
        "",
        (
            f"The current fallback plan covers {fallback_summary.get('scheduled_segments')} segments "
            f"and {fallback_summary.get('scheduled_official_miles')} official miles, but it still "
            f"uses {frag.get('executable_outings')} executable outings and too many small car-hop fragments."
        ),
        "",
        "## Fragmentation Snapshot",
        "",
        f"- Outings: {frag.get('executable_outings')}",
        f"- Outings under 1 official mile: {frag.get('outings_under_1_official_mile')}",
        f"- Outings under 2 official miles: {frag.get('outings_under_2_official_miles')}",
        f"- Days with 3+ outings: {frag.get('days_with_3_plus_outings')}",
        f"- Days with 3+ trailheads: {frag.get('days_with_3_plus_trailheads')}",
        f"- Unique trailheads: {frag.get('unique_trailheads')}",
        "",
        "## Assignment QA",
        "",
        f"- Official segments assigned: {summary.get('official_segments_assigned_to_blocks')} / {summary.get('official_segments_total')}",
        f"- Assignment passed: {summary.get('official_segment_assignment_passed')}",
        f"- Unassigned official trails: {summary.get('unassigned_official_trail_count')}",
        f"- Duplicate configured trails: {summary.get('duplicate_configured_trail_count')}",
        f"- Current cross-block outings: {summary.get('cross_block_current_outing_count')}",
        "",
        "## Blocks",
        "",
        "| Block | Status | Official mi | Segments | Current outings | Trailheads | Sub-2 errands | Cross-block | Main issue |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for block in plan.get("blocks") or []:
        if block.get("block_id") == "unassigned":
            continue
        reasons = ", ".join(block.get("readiness_reasons") or []) or "ready for GPX candidate"
        lines.append(
            "| {name} | {status} | {miles} | {segments} | {outings} | {trailheads} | {sub2} | {cross} | {reason} |".format(
                name=block.get("name"),
                status=block.get("block_readiness"),
                miles=block.get("official_miles"),
                segments=block.get("official_segment_count"),
                outings=block.get("current_outing_count"),
                trailheads=block.get("current_trailhead_count"),
                sub2=block.get("current_standalone_sub2_count"),
                cross=block.get("cross_block_current_outing_count"),
                reason=reasons,
            )
        )

    lines.extend(
        [
            "",
            "## Manual GPX Review Priority",
            "",
        ]
    )
    for item in plan.get("manual_review_priorities") or []:
        lines.append(f"{item.get('priority')}. {item.get('area')}: {item.get('decision')}")

    lines.extend(
        [
            "",
            "## Absorption Candidates",
            "",
            "These are current standalone sub-2-official-mile outings that should be absorbed into block GPX where possible.",
            "",
        ]
    )
    for block in plan.get("blocks") or []:
        candidates = block.get("absorption_candidates") or []
        if not candidates:
            continue
        lines.append(f"### {block.get('name')}")
        for candidate in candidates:
            names = ", ".join(candidate.get("trail_names") or [])
            lines.append(
                f"- {candidate.get('date')} `{candidate.get('outing_id')}`: {names} "
                f"({candidate.get('official_miles')} official mi, {candidate.get('on_foot_miles')} on-foot mi)"
            )
        lines.append("")

    if plan.get("cross_block_current_outings"):
        lines.extend(
            [
                "## Current Cross-Block Outings",
                "",
                "These are not necessarily bad; they show where the current fallback already crosses proposed block boundaries.",
                "",
            ]
        )
        for outing in plan["cross_block_current_outings"]:
            names = ", ".join(outing.get("trail_names") or [])
            block_ids = ", ".join(outing.get("block_ids") or [])
            lines.append(
                f"- {outing.get('date')} `{outing.get('outing_id')}`: {names} -> {block_ids}"
            )
        lines.append("")

    if (plan.get("config_validation") or {}).get("unassigned_official_trails"):
        lines.extend(["## Unassigned Official Trails", ""])
        for trail in plan["config_validation"]["unassigned_official_trails"]:
            lines.append(f"- {trail}")
        lines.append("")

    lines.extend(
        [
            "## Next Steps",
            "",
        ]
    )
    for step in plan.get("next_steps") or []:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks-json", type=Path, default=DEFAULT_BLOCKS_JSON)
    parser.add_argument("--runbook-json", type=Path, default=DEFAULT_RUNBOOK_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blocks_config = read_json(args.blocks_json)
    runbook = read_json(args.runbook_json)
    official_trails = load_official_trails(args.official_geojson)
    plan = build_block_first_plan(blocks_config, runbook, official_trails)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    write_json(json_path, plan)
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=str(runbook.get("run_id") or blocks_config.get("version") or args.basename),
            inputs=[args.blocks_json, args.runbook_json, args.official_geojson],
            outputs=[json_path, md_path],
            command="route_block_planner.py",
            metadata={
                "planning_status": plan.get("planning_status"),
                "official_segment_assignment_passed": plan["summary"][
                    "official_segment_assignment_passed"
                ],
                "draft_block_count": plan["summary"]["draft_block_count"],
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
