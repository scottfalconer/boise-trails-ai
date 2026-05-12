#!/usr/bin/env python3
"""Reprice later route-card work unlocked by reconciled latent official credit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_LATENT_CREDIT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "field-latent-credit-audit-2026-05-11.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "latent-credit-delta-repricing-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "latent-credit-delta-repricing-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "latent-credit-delta-repricing-audit-2026-05-12-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def int_value(value: Any) -> int:
    return int(round(float_value(value)))


def rounded(value: Any, digits: int = 2) -> float:
    return round(float_value(value), digits)


def sort_id(value: str) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def normalized_ids(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("label") or route.get("block_name") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def build_route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for route in routes:
        keys = [
            route_key(route),
            str(route.get("outing_id") or ""),
            str(route.get("label") or ""),
            route_label(route),
        ]
        for candidate_id in route.get("candidate_ids") or []:
            keys.append(str(candidate_id))
        for key in keys:
            if key:
                index[key] = route
    return index


def route_segment_ids(route: dict[str, Any]) -> set[str]:
    return set(normalized_ids(route.get("segment_ids") or []))


def route_metrics(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "outing_id": route.get("outing_id"),
        "label": route_label(route),
        "candidate_ids": route.get("candidate_ids") or [],
        "trailhead": route.get("trailhead"),
        "segment_ids": normalized_ids(route.get("segment_ids") or []),
        "segment_count": len(normalized_ids(route.get("segment_ids") or [])),
        "official_miles": rounded(route.get("official_miles")),
        "on_foot_miles": rounded(route.get("on_foot_miles")),
        "door_to_door_minutes_p75": int_value(route.get("door_to_door_minutes_p75")),
        "door_to_door_minutes_p90": int_value(route.get("door_to_door_minutes_p90")),
    }


def add_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "on_foot_miles": rounded(float_value(left.get("on_foot_miles")) + float_value(right.get("on_foot_miles"))),
        "door_to_door_minutes_p75": int_value(left.get("door_to_door_minutes_p75"))
        + int_value(right.get("door_to_door_minutes_p75")),
        "door_to_door_minutes_p90": int_value(left.get("door_to_door_minutes_p90"))
        + int_value(right.get("door_to_door_minutes_p90")),
    }


def segment_miles_from_latent_audit(latent_audit: dict[str, Any]) -> dict[str, float]:
    miles: dict[str, float] = {}
    for route in latent_audit.get("route_reviews") or []:
        for segment in route.get("segments") or []:
            segment_id = str(segment.get("seg_id") or "")
            if segment_id and float_value(segment.get("official_miles")):
                miles[segment_id] = float_value(segment.get("official_miles"))
    return miles


def field_day_order(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    route_index = build_route_index(field_tool_data.get("routes") or [])
    order: dict[str, dict[str, Any]] = {}
    ordinal = 0
    field_day_layer = field_tool_data.get("field_day_layer") or {}
    for day_index, day in enumerate(field_day_layer.get("field_days") or []):
        for loop_index, loop in enumerate(day.get("loops") or []):
            ref = loop.get("route_card_ref") or {}
            route = route_index.get(str(ref.get("outing_id") or ""))
            if not route:
                route = route_index.get(str(loop.get("label") or ""))
            if not route:
                continue
            key = route_key(route)
            if key in order:
                continue
            order[key] = {
                "order_index": ordinal,
                "date": day.get("date"),
                "field_day_id": day.get("field_day_id"),
                "day_index": day_index,
                "loop_index": loop_index,
            }
            ordinal += 1
    for route in field_tool_data.get("routes") or []:
        key = route_key(route)
        if key in order:
            continue
        order[key] = {
            "order_index": ordinal,
            "date": None,
            "field_day_id": None,
            "day_index": None,
            "loop_index": None,
        }
        ordinal += 1
    return order


def compare_order(
    source_key: str,
    owner_key: str,
    order: dict[str, dict[str, Any]],
) -> str:
    source_order = order.get(source_key)
    owner_order = order.get(owner_key)
    if not source_order or not owner_order:
        return "order_unknown"
    if int(source_order["order_index"]) < int(owner_order["order_index"]):
        return "owner_is_future_route"
    if int(source_order["order_index"]) == int(owner_order["order_index"]):
        return "same_route"
    return "owner_scheduled_before_source"


def latent_relationships(
    latent_audit: dict[str, Any],
    route_index: dict[str, dict[str, Any]],
    order: dict[str, dict[str, Any]],
    segment_miles: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for source_review in latent_audit.get("reconciled_routes") or []:
        source_route = route_index.get(str(source_review.get("outing_id") or "")) or route_index.get(
            str(source_review.get("label") or "")
        )
        if not source_route:
            continue
        source_key = route_key(source_route)
        for segment in source_review.get("segments") or []:
            if segment.get("status") != "reconciled_owned_elsewhere":
                continue
            segment_id = str(segment.get("seg_id") or "")
            if not segment_id:
                continue
            if float_value(segment.get("official_miles")):
                segment_miles[segment_id] = float_value(segment.get("official_miles"))
            for owner in segment.get("claimed_by_other_routes") or []:
                owner_route = route_index.get(str(owner.get("outing_id") or "")) or route_index.get(
                    str(owner.get("label") or "")
                )
                if not owner_route:
                    continue
                owner_key = route_key(owner_route)
                if owner_key == source_key:
                    continue
                pair = (source_key, owner_key)
                row = grouped.setdefault(
                    pair,
                    {
                        "source_route": route_metrics(source_route),
                        "owner_route": route_metrics(owner_route),
                        "latent_segment_ids": set(),
                        "segments": [],
                    },
                )
                row["latent_segment_ids"].add(segment_id)
                row["segments"].append(
                    {
                        "seg_id": segment_id,
                        "trail_name": segment.get("trail_name"),
                        "seg_name": segment.get("seg_name"),
                        "official_miles": rounded(segment.get("official_miles")),
                    }
                )
    rows: list[dict[str, Any]] = []
    for (source_key, owner_key), row in grouped.items():
        owner_ids = set(row["owner_route"]["segment_ids"])
        latent_ids = set(row["latent_segment_ids"])
        remaining_ids = owner_ids - latent_ids
        removed = not remaining_ids
        source_metrics = row["source_route"]
        owner_metrics = row["owner_route"]
        baseline = add_metrics(source_metrics, owner_metrics)
        counterfactual_owner = (
            {
                "status": "removed",
                "remaining_segment_ids": [],
                "saved_on_foot_miles": owner_metrics["on_foot_miles"],
                "saved_p75_minutes": owner_metrics["door_to_door_minutes_p75"],
                "saved_p90_minutes": owner_metrics["door_to_door_minutes_p90"],
            }
            if removed
            else {
                "status": "shrink_needs_route_card_reprice",
                "remaining_segment_ids": normalized_ids(remaining_ids),
                "saved_on_foot_miles": 0.0,
                "saved_p75_minutes": 0,
                "saved_p90_minutes": 0,
            }
        )
        estimated_official_removed = sum(segment_miles.get(segment_id, 0.0) for segment_id in latent_ids)
        order_status = compare_order(source_key, owner_key, order)
        rows.append(
            {
                "source_route_key": source_key,
                "owner_route_key": owner_key,
                "source_route": source_metrics,
                "owner_route": owner_metrics,
                "schedule_order": {
                    "source": order.get(source_key),
                    "owner": order.get(owner_key),
                    "status": order_status,
                },
                "latent_segment_ids": normalized_ids(latent_ids),
                "remaining_owner_segment_ids": normalized_ids(remaining_ids),
                "latent_official_miles": rounded(estimated_official_removed),
                "baseline": baseline,
                "counterfactual": {
                    "source_route_unchanged": True,
                    "owner_route": counterfactual_owner,
                    "combined_on_foot_miles": rounded(
                        baseline["on_foot_miles"] - counterfactual_owner["saved_on_foot_miles"]
                    ),
                    "combined_p75_minutes": baseline["door_to_door_minutes_p75"]
                    - counterfactual_owner["saved_p75_minutes"],
                    "combined_p90_minutes": baseline["door_to_door_minutes_p90"]
                    - counterfactual_owner["saved_p90_minutes"],
                },
                "future_route_change": "removed" if removed else "shrink_unpriced",
                "proven_saved_on_foot_miles": rounded(counterfactual_owner["saved_on_foot_miles"]),
                "proven_saved_p75_minutes": counterfactual_owner["saved_p75_minutes"],
                "proven_saved_p90_minutes": counterfactual_owner["saved_p90_minutes"],
                "segments": sorted(row["segments"], key=lambda item: sort_id(str(item.get("seg_id") or ""))),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["future_route_change"] != "removed",
            -float_value(row["proven_saved_on_foot_miles"]),
            row["owner_route"]["label"],
            row["source_route"]["label"],
        ),
    )


def latent_segments_by_source(relationships: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_source: dict[str, set[str]] = {}
    for row in relationships:
        by_source.setdefault(row["source_route_key"], set()).update(row["latent_segment_ids"])
    return by_source


def latent_sources_by_segment(relationships: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for row in relationships:
        source = {
            "source_route_key": row["source_route_key"],
            "outing_id": row["source_route"]["outing_id"],
            "label": row["source_route"]["label"],
        }
        for segment_id in row["latent_segment_ids"]:
            by_segment.setdefault(segment_id, []).append(source)
    return by_segment


def simulate_calendar_repricing(
    field_tool_data: dict[str, Any],
    relationships: list[dict[str, Any]],
    order: dict[str, dict[str, Any]],
    segment_miles: dict[str, float],
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = {route_key(route): route for route in routes}
    ordered_keys = [
        key
        for key, _row in sorted(order.items(), key=lambda item: int(item[1]["order_index"]))
        if key in routes_by_key
    ]
    latent_by_source = latent_segments_by_source(relationships)
    sources_by_segment = latent_sources_by_segment(relationships)
    completed: set[str] = set()
    completed_by_latent: set[str] = set()
    removed_routes: list[dict[str, Any]] = []
    partial_routes: list[dict[str, Any]] = []
    executed_routes: list[str] = []
    for key in ordered_keys:
        route = routes_by_key[key]
        ids = route_segment_ids(route)
        prior_latent_ids = ids & completed_by_latent
        remaining_ids = ids - completed
        prior_segment_miles = rounded(sum(segment_miles.get(segment_id, 0.0) for segment_id in prior_latent_ids))
        if prior_latent_ids and not remaining_ids:
            removed_routes.append(
                {
                    "route_key": key,
                    "route": route_metrics(route),
                    "schedule": order.get(key),
                    "prior_latent_segment_ids": normalized_ids(prior_latent_ids),
                    "prior_latent_official_miles": prior_segment_miles,
                    "prior_sources": {
                        segment_id: sources_by_segment.get(segment_id, [])
                        for segment_id in normalized_ids(prior_latent_ids)
                    },
                    "saved_on_foot_miles": rounded(route.get("on_foot_miles")),
                    "saved_p75_minutes": int_value(route.get("door_to_door_minutes_p75")),
                    "saved_p90_minutes": int_value(route.get("door_to_door_minutes_p90")),
                    "future_route_change": "removed",
                }
            )
            continue
        if prior_latent_ids:
            partial_routes.append(
                {
                    "route_key": key,
                    "route": route_metrics(route),
                    "schedule": order.get(key),
                    "prior_latent_segment_ids": normalized_ids(prior_latent_ids),
                    "remaining_segment_ids": normalized_ids(remaining_ids),
                    "prior_latent_official_miles": prior_segment_miles,
                    "prior_sources": {
                        segment_id: sources_by_segment.get(segment_id, [])
                        for segment_id in normalized_ids(prior_latent_ids)
                    },
                    "future_route_change": "shrink_needs_route_card_reprice",
                    "proven_saved_on_foot_miles": 0.0,
                    "proven_saved_p75_minutes": 0,
                    "proven_saved_p90_minutes": 0,
                }
            )
        executed_routes.append(key)
        completed.update(ids)
        source_latent_ids = latent_by_source.get(key, set())
        completed.update(source_latent_ids)
        completed_by_latent.update(source_latent_ids)

    saved_on_foot = rounded(sum(float_value(row["saved_on_foot_miles"]) for row in removed_routes))
    saved_p75 = sum(int_value(row["saved_p75_minutes"]) for row in removed_routes)
    saved_p90 = sum(int_value(row["saved_p90_minutes"]) for row in removed_routes)
    return {
        "status": "proved_current_calendar_savings" if removed_routes else "no_current_calendar_route_removal",
        "removed_routes": removed_routes,
        "partial_reprice_routes": partial_routes,
        "executed_route_count": len(executed_routes),
        "summary": {
            "removed_route_count": len(removed_routes),
            "partial_reprice_route_count": len(partial_routes),
            "saved_on_foot_miles": saved_on_foot,
            "saved_p75_minutes": saved_p75,
            "saved_p90_minutes": saved_p90,
        },
    }


def build_latent_credit_delta_repricing_audit(
    field_tool_data: dict[str, Any],
    latent_audit: dict[str, Any],
    *,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    route_index = build_route_index(routes)
    order = field_day_order(field_tool_data)
    segment_miles = segment_miles_from_latent_audit(latent_audit)
    relationships = latent_relationships(latent_audit, route_index, order, segment_miles)
    full_removals = [row for row in relationships if row["future_route_change"] == "removed"]
    shrink_candidates = [row for row in relationships if row["future_route_change"] == "shrink_unpriced"]
    calendar = simulate_calendar_repricing(field_tool_data, relationships, order, segment_miles)
    unique_latent_segments = {
        segment_id
        for row in relationships
        for segment_id in row.get("latent_segment_ids") or []
    }
    status = (
        "proved_current_calendar_savings"
        if calendar["summary"]["removed_route_count"]
        else "pairwise_savings_only"
        if full_removals
        else "needs_route_card_repricing"
        if shrink_candidates
        else "no_delta_value_found"
    )
    return {
        "schema": "boise_trails_latent_credit_delta_repricing_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "scope": {
            "proves": [
                "which reconciled latent-credit relationships remove a future route card outright",
                "which current calendar route cards can be skipped if prior latent credit is accepted",
                "which remaining cases are only partial shrink candidates until a route-card replacement is generated",
            ],
            "does_not_prove": [
                "official BTC app credit before challenge-window activity validation",
                "field-ready replacement cards for partial shrink cases",
                "lower on-foot/p75/p90 for partial cases without a regenerated route card",
            ],
        },
        "summary": {
            "route_count": len(routes),
            "latent_relationship_count": len(relationships),
            "unique_latent_segment_count": len(unique_latent_segments),
            "pairwise_full_removal_relationship_count": len(full_removals),
            "pairwise_partial_shrink_relationship_count": len(shrink_candidates),
            "current_calendar_removed_route_count": calendar["summary"]["removed_route_count"],
            "current_calendar_partial_reprice_route_count": calendar["summary"]["partial_reprice_route_count"],
            "current_calendar_saved_on_foot_miles": calendar["summary"]["saved_on_foot_miles"],
            "current_calendar_saved_p75_minutes": calendar["summary"]["saved_p75_minutes"],
            "current_calendar_saved_p90_minutes": calendar["summary"]["saved_p90_minutes"],
        },
        "current_calendar_repricing": calendar,
        "pairwise_full_removals": full_removals,
        "pairwise_partial_shrink_candidates": shrink_candidates,
        "relationships": relationships,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Latent-Credit Delta Repricing Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Latent route relationships: {summary['latent_relationship_count']}",
        f"- Unique latent official segments: {summary['unique_latent_segment_count']}",
        f"- Pairwise full-removal relationships: {summary['pairwise_full_removal_relationship_count']}",
        f"- Pairwise partial-shrink relationships: {summary['pairwise_partial_shrink_relationship_count']}",
        f"- Current-calendar removed routes: {summary['current_calendar_removed_route_count']}",
        f"- Current-calendar partial reprices needed: {summary['current_calendar_partial_reprice_route_count']}",
        f"- Current-calendar proven savings: {summary['current_calendar_saved_on_foot_miles']:.2f} on-foot mi, {summary['current_calendar_saved_p75_minutes']} p75 min, {summary['current_calendar_saved_p90_minutes']} p90 min",
        "",
        "## Current Calendar Route Removals",
        "",
    ]
    removed = audit["current_calendar_repricing"]["removed_routes"]
    if removed:
        lines.extend(
            [
                "| Route | Date | Latent ids | Saved on-foot mi | Saved p75 | Saved p90 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in removed:
            route = row["route"]
            lines.append(
                f"| {route['label']} | {row.get('schedule', {}).get('date') or ''} | {', '.join(row['prior_latent_segment_ids'])} | {float_value(row['saved_on_foot_miles']):.2f} | {row['saved_p75_minutes']} | {row['saved_p90_minutes']} |"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Pairwise Full-Removal Opportunities", ""])
    full_removals = audit.get("pairwise_full_removals") or []
    if full_removals:
        lines.extend(
            [
                "| Source route | Future/owner route | Order status | Latent ids | Saved on-foot mi | Saved p75 | Saved p90 |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in full_removals:
            lines.append(
                f"| {row['source_route']['label']} | {row['owner_route']['label']} | {row['schedule_order']['status']} | {', '.join(row['latent_segment_ids'])} | {float_value(row['proven_saved_on_foot_miles']):.2f} | {row['proven_saved_p75_minutes']} | {row['proven_saved_p90_minutes']} |"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Partial Shrink Candidates", ""])
    partial = audit.get("current_calendar_repricing", {}).get("partial_reprice_routes") or []
    if partial:
        lines.extend(
            [
                "| Route | Date | Already credited ids | Remaining ids | Current on-foot mi | Current p75 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(partial, key=lambda item: -float_value(item["route"].get("on_foot_miles")))[:20]:
            route = row["route"]
            lines.append(
                f"| {route['label']} | {row.get('schedule', {}).get('date') or ''} | {', '.join(row['prior_latent_segment_ids'])} | {', '.join(row['remaining_segment_ids'])} | {float_value(route['on_foot_miles']):.2f} | {route['door_to_door_minutes_p75']} |"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- Full-removal rows are the only rows with proven on-foot and p75/p90 savings in this artifact.",
            "- Partial-shrink rows are intentionally priced at zero proven savings until a generated replacement route card exists.",
            "- This audit does not mark BTC progress; it only prices what the active route menu could stop asking the runner to do after validated latent credit is accepted.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--latent-credit-audit-json", type=Path, default=DEFAULT_LATENT_CREDIT_AUDIT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write artifacts but return success even when no delta value is found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_latent_credit_delta_repricing_audit(
        read_json(args.field_tool_data_json),
        read_json(args.latent_credit_audit_json),
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "latent_credit_audit_json": display_path(args.latent_credit_audit_json),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="latent-credit-delta-repricing-audit-2026-05-12",
        inputs=[args.field_tool_data_json, args.latent_credit_audit_json],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/latent_credit_delta_repricing_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] == "no_delta_value_found" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
