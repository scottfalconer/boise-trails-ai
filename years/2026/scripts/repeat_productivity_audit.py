#!/usr/bin/env python3
"""Classify official repeat mileage by productive, necessary, and dead-repeat pressure."""

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
from cluster_level_repricing_audit import (  # noqa: E402
    display_path,
    float_value,
    int_value,
    normalized_ids,
    official_segment_miles,
    route_index,
    route_key,
    route_label,
    route_metrics,
    rounded,
    sort_id,
    write_json,
)
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_ROUTE_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_OWNERSHIP_AUDIT_JSON = YEAR_DIR / "checkpoints" / "ownership-reassignment-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "repeat-productivity-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "repeat-productivity-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "repeat-productivity-audit-2026-05-12-manifest.json"

CONNECTOR_CUE_TYPES = {"connector_named_trail", "connector_road", "overlap_repeat"}
RETURN_CUE_TYPES = {"exit_access", "return_to_car"}
ACCESS_CUE_TYPES = {"start_access"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment.get("seg_id")): segment for segment in official_segments if segment.get("seg_id") is not None}


def segment_brief(
    segment_id: str,
    official_by_id: dict[str, dict[str, Any]],
    segment_miles: dict[str, float],
) -> dict[str, Any]:
    segment = official_by_id.get(str(segment_id), {})
    return {
        "seg_id": str(segment_id),
        "trail_name": segment.get("trail_name"),
        "seg_name": segment.get("seg_name"),
        "direction": segment.get("direction"),
        "official_miles": rounded(segment_miles.get(str(segment_id), segment.get("official_miles", 0.0))),
    }


def cue_text(cue: dict[str, Any]) -> str:
    return " ".join(
        str(cue.get(key) or "")
        for key in ("action", "display_detail", "note", "field_warning", "compact")
    ).lower()


def warning_codes(row: dict[str, Any] | None) -> set[str]:
    return {str(warning.get("code")) for warning in (row or {}).get("optimization_warnings") or []}


def avoidable_post_credit_repeat_ids(row: dict[str, Any] | None) -> set[str]:
    return {
        segment_id
        for instance in (row or {}).get("avoidable_post_credit_repeat_instances") or []
        for segment_id in normalized_ids(instance.get("repeated_segment_ids") or [])
    }


def route_repeat_index(
    route_repeat_audit: dict[str, Any],
    routes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup = route_index(routes)
    rows: dict[str, dict[str, Any]] = {}
    for row in route_repeat_audit.get("routes") or []:
        route = lookup.get(str(row.get("outing_id") or "")) or lookup.get(str(row.get("label") or ""))
        if route:
            rows[route_key(route)] = row
    return rows


def ownership_context(ownership_audit: dict[str, Any]) -> dict[str, Any]:
    productive_ids_by_route: dict[str, set[str]] = {}
    route_pressure: dict[str, set[str]] = {}
    route_component: dict[str, str] = {}
    productive_sources: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for component in ownership_audit.get("components") or []:
        component_id = str(component.get("component_id") or "")
        component_has_savings = float_value((component.get("order_free_savings") or {}).get("on_foot_miles")) > 0
        component_has_reassignment = int_value(component.get("reassigned_segment_count")) > 0
        for key in component.get("route_keys") or []:
            route_key_value = str(key)
            route_component[route_key_value] = component_id
            if component_has_savings:
                route_pressure.setdefault(route_key_value, set()).add("ownership_component_savings")
            elif component_has_reassignment:
                route_pressure.setdefault(route_key_value, set()).add("credit_only_reassignment_component")
        for impact in component.get("route_impacts") or []:
            key = str(impact.get("route_key") or "")
            if not key:
                continue
            status = str(impact.get("status") or "")
            if status:
                route_pressure.setdefault(key, set()).add(status)
            if status == "removed_proven":
                route_pressure.setdefault(key, set()).add("route_can_be_removed")
            if status == "shrunk_unpriced":
                route_pressure.setdefault(key, set()).add("route_can_be_shrunk")
            if str(impact.get("replacement_order_status") or "") == "requires_calendar_reorder":
                route_pressure.setdefault(key, set()).add("requires_calendar_reorder")
            for segment_id in normalized_ids(impact.get("gained_credit_segment_ids") or []):
                productive_ids_by_route.setdefault(key, set()).add(segment_id)
                productive_sources.setdefault((key, segment_id), []).append(
                    {
                        "component_id": component_id,
                        "route_impact_status": status,
                        "reason": "optimized_credit_owner",
                    }
                )
    return {
        "productive_ids_by_route": productive_ids_by_route,
        "route_pressure": route_pressure,
        "route_component": route_component,
        "productive_sources": productive_sources,
    }


def repeat_cue_instances(
    route: dict[str, Any],
    segment_miles: dict[str, float],
) -> list[dict[str, Any]]:
    instances = []
    for cue in route.get("wayfinding_cues") or []:
        segment_ids = normalized_ids(cue.get("official_repeat_segment_ids") or [])
        if not segment_ids:
            continue
        cue_miles = float_value(cue.get("official_repeat_miles"))
        cue_route_miles = float_value(cue.get("leg_miles"))
        weights = {segment_id: segment_miles.get(segment_id, 0.0) for segment_id in segment_ids}
        total_weight = sum(weights.values())
        if cue_miles <= 0:
            cue_miles = total_weight
        actual_route_miles = min(cue_miles, cue_route_miles) if cue_route_miles > 0 else cue_miles
        if total_weight <= 0:
            total_weight = float(len(segment_ids)) or 1.0
            weights = {segment_id: 1.0 for segment_id in segment_ids}
        for segment_id in segment_ids:
            allocated = cue_miles * (weights[segment_id] / total_weight)
            allocated_actual_route_miles = actual_route_miles * (weights[segment_id] / total_weight)
            instances.append(
                {
                    "source": "declared_official_repeat_cue",
                    "seq": cue.get("seq"),
                    "cue_type": cue.get("cue_type"),
                    "segment_id": segment_id,
                    "repeat_miles": allocated,
                    "actual_route_miles": allocated_actual_route_miles,
                    "cue_official_repeat_miles": cue_miles,
                    "cue_route_leg_miles": cue_route_miles,
                    "cue_segment_count": len(segment_ids),
                    "action": cue.get("action"),
                    "cue_text": cue_text(cue),
                }
            )
    return instances


def has_pressure(route_pressure: set[str], *codes: str) -> bool:
    return any(code in route_pressure for code in codes)


def classify_repeat_instance(
    instance: dict[str, Any],
    *,
    route_key_value: str,
    productive_ids: set[str],
    avoidable_ids: set[str],
    route_pressure: set[str],
    warnings: set[str],
) -> tuple[str, str, list[str]]:
    segment_id = str(instance["segment_id"])
    if segment_id in productive_ids:
        return (
            "productive_repeat",
            "repeat segment is assigned as optimized credit for this physical route",
            ["optimized_credit_owner"],
        )
    if segment_id in avoidable_ids:
        return (
            "dead_repeat_candidate",
            "post-credit repeat has a proven shorter legal connector",
            ["avoidable_post_credit_repeat"],
        )
    cue_type = str(instance.get("cue_type") or "")
    evidence = sorted(route_pressure | {f"warning:{code}" for code in warnings})
    route_removed = has_pressure(route_pressure, "route_can_be_removed", "removed_proven")
    route_shrunk = has_pressure(route_pressure, "route_can_be_shrunk", "shrunk_unpriced")
    same_trailhead = "same_trailhead_bundle_candidate" in warnings
    component_savings = "ownership_component_savings" in route_pressure
    if route_removed:
        return (
            "dead_repeat_candidate",
            "the containing route card can be removed by ownership reassignment",
            evidence,
        )
    if cue_type in ACCESS_CUE_TYPES and (same_trailhead or component_savings or route_shrunk):
        return (
            "dead_repeat_candidate",
            "start/access repeat has alternate ownership, ordering, or same-trailhead pressure",
            evidence,
        )
    if cue_type in CONNECTOR_CUE_TYPES and (same_trailhead or component_savings or route_shrunk):
        return (
            "dead_repeat_candidate",
            "connector/overlap repeat has plausible reorder/start/connector pressure",
            evidence,
        )
    if cue_type in RETURN_CUE_TYPES or "return leg" in str(instance.get("cue_text") or ""):
        return (
            "necessary_repeat",
            "repeat is currently represented as return-to-car access with no proven better connector",
            evidence,
        )
    if route_shrunk and (same_trailhead or component_savings):
        return (
            "dead_repeat_candidate",
            "route can be partially shrunk and this repeat has alternate-order pressure",
            evidence,
        )
    return (
        "necessary_repeat",
        "no known legal alternate start/order/connector evidence in current audits",
        evidence,
    )


def latent_productive_instances(
    route_key_value: str,
    route: dict[str, Any],
    declared_repeat_ids: set[str],
    productive_ids: set[str],
    segment_miles: dict[str, float],
) -> list[dict[str, Any]]:
    claimed_ids = set(normalized_ids(route.get("segment_ids") or []))
    latent_ids = productive_ids - declared_repeat_ids - claimed_ids
    return [
        {
            "source": "productive_latent_credit",
            "seq": None,
            "cue_type": "latent_credit",
            "segment_id": segment_id,
            "repeat_miles": segment_miles.get(segment_id, 0.0),
            "actual_route_miles": 0.0,
            "cue_official_repeat_miles": None,
            "cue_route_leg_miles": None,
            "cue_segment_count": None,
            "action": None,
            "cue_text": "",
        }
        for segment_id in normalized_ids(latent_ids)
    ]


def route_rows(
    field_tool_data: dict[str, Any],
    route_repeat_audit: dict[str, Any],
    ownership_audit: dict[str, Any],
    official_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    routes = field_tool_data.get("routes") or []
    repeat_rows_by_route = route_repeat_index(route_repeat_audit, routes)
    ownership = ownership_context(ownership_audit)
    productive_ids_by_route: dict[str, set[str]] = ownership["productive_ids_by_route"]
    pressure_by_route: dict[str, set[str]] = ownership["route_pressure"]
    productive_sources: dict[tuple[str, str], list[dict[str, Any]]] = ownership["productive_sources"]
    official_by_id = segment_index(official_segments)
    segment_miles = official_segment_miles(official_segments, field_tool_data)
    rows = []
    for route in routes:
        key = route_key(route)
        repeat_row = repeat_rows_by_route.get(key) or {}
        warnings = warning_codes(repeat_row)
        avoidable_ids = avoidable_post_credit_repeat_ids(repeat_row)
        route_pressure = set(pressure_by_route.get(key, set()))
        if "same_trailhead_bundle_candidate" in warnings:
            route_pressure.add("same_trailhead_bundle_candidate")
        productive_ids = set(productive_ids_by_route.get(key, set()))
        declared_instances = repeat_cue_instances(route, segment_miles)
        declared_repeat_ids = {str(instance["segment_id"]) for instance in declared_instances}
        instances = declared_instances + latent_productive_instances(
            key,
            route,
            declared_repeat_ids,
            productive_ids,
            segment_miles,
        )
        classified = []
        totals = {
            "productive_repeat_miles": 0.0,
            "productive_declared_repeat_miles": 0.0,
            "productive_latent_miles": 0.0,
            "necessary_repeat_miles": 0.0,
            "dead_repeat_candidate_miles": 0.0,
            "dead_repeat_actual_route_miles": 0.0,
        }
        segment_sets = {
            "productive_repeat_segment_ids": set(),
            "necessary_repeat_segment_ids": set(),
            "dead_repeat_candidate_segment_ids": set(),
        }
        for instance in instances:
            if instance["source"] == "productive_latent_credit":
                classification = "productive_repeat"
                reason = "latent physical coverage is assigned as optimized credit for this route"
                evidence = ["optimized_credit_owner", "productive_latent_credit"]
            else:
                classification, reason, evidence = classify_repeat_instance(
                    instance,
                    route_key_value=key,
                    productive_ids=productive_ids,
                    avoidable_ids=avoidable_ids,
                    route_pressure=route_pressure,
                    warnings=warnings,
                )
            miles = float_value(instance.get("repeat_miles"))
            actual_route_miles = float_value(instance.get("actual_route_miles"))
            totals[f"{classification}_miles"] += miles
            if classification == "dead_repeat_candidate":
                totals["dead_repeat_actual_route_miles"] += actual_route_miles
            if classification == "productive_repeat":
                if instance["source"] == "productive_latent_credit":
                    totals["productive_latent_miles"] += miles
                else:
                    totals["productive_declared_repeat_miles"] += miles
            segment_sets[f"{classification}_segment_ids"].add(str(instance["segment_id"]))
            classified.append(
                {
                    "classification": classification,
                    "classification_reason": reason,
                    "evidence": evidence,
                    "source": instance["source"],
                    "seq": instance.get("seq"),
                    "cue_type": instance.get("cue_type"),
                    "action": instance.get("action"),
                    "segment": segment_brief(str(instance["segment_id"]), official_by_id, segment_miles),
                    "repeat_miles": rounded(miles, 4),
                    "actual_route_miles": rounded(actual_route_miles, 4),
                    "cue_route_leg_miles": (
                        rounded(instance.get("cue_route_leg_miles"), 4)
                        if instance.get("cue_route_leg_miles") is not None
                        else None
                    ),
                    "productive_sources": productive_sources.get((key, str(instance["segment_id"])), []),
                }
            )
        row = {
            **route_metrics(route),
            "route_key": key,
            "route_repeat_audit": {
                "non_credit_miles": rounded(repeat_row.get("non_credit_miles")),
                "declared_repeat_miles": rounded(repeat_row.get("declared_repeat_miles")),
                "on_foot_to_official_ratio": repeat_row.get("on_foot_to_official_ratio"),
                "optimization_warning_codes": sorted(warnings),
            },
            "route_pressure": sorted(route_pressure),
            "ownership_component_id": ownership["route_component"].get(key),
            "productive_repeat_miles": rounded(totals["productive_repeat_miles"]),
            "productive_declared_repeat_miles": rounded(totals["productive_declared_repeat_miles"]),
            "productive_latent_miles": rounded(totals["productive_latent_miles"]),
            "necessary_repeat_miles": rounded(totals["necessary_repeat_miles"]),
            "dead_repeat_candidate_miles": rounded(totals["dead_repeat_candidate_miles"]),
            "dead_repeat_official_segment_miles": rounded(totals["dead_repeat_candidate_miles"]),
            "dead_repeat_actual_route_miles": rounded(totals["dead_repeat_actual_route_miles"]),
            "classified_repeat_miles": rounded(
                totals["productive_repeat_miles"]
                + totals["necessary_repeat_miles"]
                + totals["dead_repeat_candidate_miles"]
            ),
            "productive_repeat_segment_ids": normalized_ids(segment_sets["productive_repeat_segment_ids"]),
            "necessary_repeat_segment_ids": normalized_ids(segment_sets["necessary_repeat_segment_ids"]),
            "dead_repeat_candidate_segment_ids": normalized_ids(segment_sets["dead_repeat_candidate_segment_ids"]),
            "repeat_classification": classified,
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float_value(row["dead_repeat_actual_route_miles"]),
            -float_value(row["dead_repeat_candidate_miles"]),
            -float_value(row["productive_repeat_miles"]),
            -float_value(row["route_repeat_audit"]["non_credit_miles"]),
            row["label"],
        ),
    )


def build_repeat_productivity_audit(
    field_tool_data: dict[str, Any],
    route_repeat_audit: dict[str, Any],
    ownership_audit: dict[str, Any],
    official_segments: list[dict[str, Any]],
    *,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows = route_rows(field_tool_data, route_repeat_audit, ownership_audit, official_segments)
    dead_rows = [row for row in rows if float_value(row["dead_repeat_candidate_miles"]) > 0.005]
    productive_rows = [row for row in rows if float_value(row["productive_repeat_miles"]) > 0.005]
    high_non_credit_protected = sorted(
        [
            row for row in rows
            if float_value(row["route_repeat_audit"]["non_credit_miles"]) >= 2.0
            and float_value(row["dead_repeat_candidate_miles"]) <= 0.1
            and (
                float_value(row["productive_repeat_miles"]) > 0.0
                or float_value(row["necessary_repeat_miles"]) > 0.0
            )
        ],
        key=lambda row: -float_value(row["route_repeat_audit"]["non_credit_miles"]),
    )
    summary = {
        "route_count": len(rows),
        "routes_with_dead_repeat_candidate_count": len(dead_rows),
        "routes_with_productive_repeat_count": len(productive_rows),
        "total_productive_repeat_miles": rounded(sum(float_value(row["productive_repeat_miles"]) for row in rows)),
        "total_productive_declared_repeat_miles": rounded(
            sum(float_value(row["productive_declared_repeat_miles"]) for row in rows)
        ),
        "total_productive_latent_miles": rounded(sum(float_value(row["productive_latent_miles"]) for row in rows)),
        "total_necessary_repeat_miles": rounded(sum(float_value(row["necessary_repeat_miles"]) for row in rows)),
        "total_dead_repeat_candidate_miles": rounded(
            sum(float_value(row["dead_repeat_candidate_miles"]) for row in rows)
        ),
        "total_dead_repeat_actual_route_miles": rounded(
            sum(float_value(row["dead_repeat_actual_route_miles"]) for row in rows)
        ),
        "total_classified_repeat_miles": rounded(sum(float_value(row["classified_repeat_miles"]) for row in rows)),
        "top_dead_repeat_candidate_route": dead_rows[0]["label"] if dead_rows else None,
        "top_dead_repeat_candidate_miles": dead_rows[0]["dead_repeat_candidate_miles"] if dead_rows else 0.0,
        "top_dead_repeat_actual_route_miles": dead_rows[0]["dead_repeat_actual_route_miles"] if dead_rows else 0.0,
    }
    return {
        "schema": "boise_trails_repeat_productivity_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "dead_repeat_candidates_found" if dead_rows else "repeat_mostly_necessary_or_productive",
        "source_files": source_files or {},
        "parameters": {
            "repeat_source": "field-packet wayfinding_cues official_repeat_segment_ids and official_repeat_miles",
            "actual_route_mile_source": "per-cue leg_miles capped at official_repeat_miles and allocated across repeat segment ids",
            "productive_source": "ownership reassignment gained credit plus productive latent coverage",
            "dead_candidate_rule": "repeat with plausible alternate ownership/order/start/connector evidence, excluding return-to-car repeat unless the containing route can be removed",
            "ranking": "dead_repeat_actual_route_miles descending, then dead_repeat_official_segment_miles, not raw non_credit_miles",
        },
        "scope": {
            "proves": [
                "which official repeat/latent miles are productive because they reduce future route work in the ownership audit",
                "which repeat miles are currently necessary return/access repeat with no proven better connector in generated audits",
                "which repeat miles are candidate redesign targets because alternate order/start/connector pressure exists",
                "which candidate repeat has literal route-mile exposure versus broader official-segment ownership pressure",
            ],
            "does_not_prove": [
                "a legal replacement connector exists for every dead-repeat candidate",
                "partial-shrink p75/on-foot savings before regenerated route cards exist",
                "official BTC progress before challenge-window activity validation",
            ],
        },
        "summary": summary,
        "routes_ranked_by_dead_repeat_candidate": dead_rows,
        "routes_with_productive_repeat": sorted(
            productive_rows,
            key=lambda row: (-float_value(row["productive_repeat_miles"]), row["label"]),
        ),
        "high_non_credit_but_not_dead_repeat": high_non_credit_protected[:20],
        "routes": rows,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Repeat Productivity Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Routes with dead-repeat candidates: {summary['routes_with_dead_repeat_candidate_count']}",
        f"- Routes with productive repeat/latent credit: {summary['routes_with_productive_repeat_count']}",
        f"- Productive repeat/latent miles: {summary['total_productive_repeat_miles']:.2f} ({summary['total_productive_declared_repeat_miles']:.2f} declared repeat, {summary['total_productive_latent_miles']:.2f} latent)",
        f"- Necessary repeat miles: {summary['total_necessary_repeat_miles']:.2f}",
        f"- Dead-repeat official-segment pressure: {summary['total_dead_repeat_candidate_miles']:.2f} mi",
        f"- Dead-repeat actual route miles: {summary['total_dead_repeat_actual_route_miles']:.2f} mi",
        f"- Top dead-repeat candidate: {summary['top_dead_repeat_candidate_route'] or 'None'} ({summary['top_dead_repeat_actual_route_miles']:.2f} actual mi; {summary['top_dead_repeat_candidate_miles']:.2f} official-pressure mi)",
        "",
        "## Ranked Dead-Repeat Candidates",
        "",
    ]
    dead_rows = audit.get("routes_ranked_by_dead_repeat_candidate") or []
    if dead_rows:
        lines.extend(
            [
                "| Rank | Route | Dead actual route mi | Official pressure mi | Productive mi | Necessary mi | Raw non-credit mi | Evidence |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for index, row in enumerate(dead_rows[:20], start=1):
            evidence = ", ".join(row.get("route_pressure") or row["route_repeat_audit"].get("optimization_warning_codes") or [])
            lines.append(
                f"| {index} | {row['label']} | {float_value(row['dead_repeat_actual_route_miles']):.2f} | {float_value(row['dead_repeat_candidate_miles']):.2f} | {float_value(row['productive_repeat_miles']):.2f} | {float_value(row['necessary_repeat_miles']):.2f} | {float_value(row['route_repeat_audit']['non_credit_miles']):.2f} | {evidence} |"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## High Non-Credit But Not Dead-Repeat Ranked", ""])
    protected = audit.get("high_non_credit_but_not_dead_repeat") or []
    if protected:
        lines.extend(
            [
                "| Route | Raw non-credit mi | Productive mi | Necessary mi | Dead actual route mi | Official pressure mi | Why not top dead-repeat |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in protected[:12]:
            reason = (
                "productive future credit"
                if float_value(row["productive_repeat_miles"])
                else "necessary repeat in current audits; no proven alternate"
            )
            lines.append(
                f"| {row['label']} | {float_value(row['route_repeat_audit']['non_credit_miles']):.2f} | {float_value(row['productive_repeat_miles']):.2f} | {float_value(row['necessary_repeat_miles']):.2f} | {float_value(row['dead_repeat_actual_route_miles']):.2f} | {float_value(row['dead_repeat_candidate_miles']):.2f} | {reason} |"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- This audit ranks official repeat/latent mileage by usefulness, not total non-credit mileage.",
            "- `dead_repeat_actual_route_miles` is the field-effort queue metric; `dead_repeat_candidate_miles` is official-segment ownership pressure and can exceed raw non-credit miles when a short leg touches many official segment geometries.",
            "- `productive_repeat` means the ownership reassignment audit says this physical route should receive credit for a segment currently owned elsewhere.",
            "- `necessary_repeat` means the repeat is represented as access/return movement and current generated audits do not prove a better legal connector.",
            "- `dead_repeat_candidate` means redesign pressure exists; it is not itself proof that a replacement route is field-ready.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--route-repeat-audit-json", type=Path, default=DEFAULT_ROUTE_REPEAT_AUDIT_JSON)
    parser.add_argument("--ownership-audit-json", type=Path, default=DEFAULT_OWNERSHIP_AUDIT_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    audit = build_repeat_productivity_audit(
        read_json(args.field_tool_data_json),
        read_json(args.route_repeat_audit_json),
        read_json(args.ownership_audit_json),
        official_segments,
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "route_repeat_audit_json": display_path(args.route_repeat_audit_json),
            "ownership_audit_json": display_path(args.ownership_audit_json),
            "official_geojson": display_path(args.official_geojson),
            "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="repeat-productivity-audit-2026-05-12",
        inputs=[args.field_tool_data_json, args.route_repeat_audit_json, args.ownership_audit_json, args.official_geojson],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/repeat_productivity_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] == "repeat_mostly_necessary_or_productive" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
