#!/usr/bin/env python3
"""Audit accepted-anchor route-card preservation regressions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

import sys

sys.path.insert(0, str(SCRIPT_DIR))

from accepted_route_replacements import (  # noqa: E402
    ACTIVE_STATUS,
    INVESTIGATE_STATUS,
    WAIVED_STATUS,
    AcceptedRouteReplacementIndex,
    DEFAULT_ACCEPTED_REPLACEMENTS_JSON,
    anchor_refs_match,
    candidate_metrics,
    dominance_deltas,
    float_value,
    meets_material_savings,
    no_material_regression,
    route_metrics,
    segment_key,
)
from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402


DEFAULT_FIELD_TOOL_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_MULTI_START_AUDIT_JSON = YEAR_DIR / "checkpoints" / "multi-start-alternative-audit-2026-05-08.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "accepted-anchor-preservation-audit-2026-05-15.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "accepted-anchor-preservation-audit-2026-05-15.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "accepted-anchor-preservation-audit-2026-05-15-manifest.json"

DEFAULT_MIN_ON_FOOT_SAVINGS = 0.25
DEFAULT_MIN_P75_SAVINGS = 10


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


def routes_by_segment(field_tool_payload: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    rows = {}
    for route in field_tool_payload.get("routes") or []:
        key = segment_key(route.get("segment_ids"))
        if key:
            rows[key] = route
    return rows


def anchor_is_publicly_accepted(anchor: dict[str, Any], manifest_record: dict[str, Any] | None) -> bool:
    confidence = str(anchor.get("parking_confidence") or "").lower()
    source_type = str(anchor.get("source_type") or "").lower()
    if "user_review" in confidence:
        return True
    if manifest_record and source_type == "private_strava_anchor":
        return True
    return False


def candidate_row_from_component(
    *,
    alternative: dict[str, Any],
    component: dict[str, Any],
    current_route: dict[str, Any],
    manifest_record: dict[str, Any] | None,
) -> dict[str, Any]:
    anchor = component.get("start_anchor") or {}
    replacement = {
        "on_foot_miles": component.get("on_foot_miles"),
        "p75_minutes": component.get("p75_total_minutes_if_standalone"),
        "p90_minutes": component.get("p90_total_minutes_if_standalone"),
        "parking_confidence": component.get("parking_confidence") or anchor.get("parking_confidence"),
        "anchor_to_credit_endpoint_distance_miles": anchor.get("distance_to_component_miles"),
    }
    baseline = route_metrics(current_route)
    deltas = dominance_deltas(baseline, replacement)
    return {
        "alternative_id": alternative.get("alternative_id"),
        "status": alternative.get("status"),
        "current_route_label": current_route.get("label"),
        "current_route_trailhead": current_route.get("trailhead"),
        "target_segment_ids": list(segment_key(component.get("segment_ids"))),
        "trail_names": component.get("trail_names") or [],
        "accepted_anchor_ref": anchor.get("anchor_id"),
        "accepted_anchor_label": anchor.get("name"),
        "parking_confidence": replacement.get("parking_confidence"),
        "anchor_to_credit_endpoint_distance_miles": anchor.get("distance_to_component_miles"),
        "candidate_on_foot_miles": component.get("on_foot_miles"),
        "candidate_p75_minutes": component.get("p75_total_minutes_if_standalone"),
        "current_on_foot_miles": current_route.get("on_foot_miles"),
        "current_p75_minutes": current_route.get("door_to_door_minutes_p75"),
        "dominance_deltas": deltas,
        "manifest_replacement_id": (manifest_record or {}).get("replacement_id"),
    }


def material_discovery(row: dict[str, Any]) -> bool:
    deltas = row.get("dominance_deltas") or {}
    return (
        float_value(deltas.get("dominance_delta_on_foot_miles")) >= DEFAULT_MIN_ON_FOOT_SAVINGS
        or float_value(deltas.get("dominance_delta_p75_minutes")) >= DEFAULT_MIN_P75_SAVINGS
    )


def discovery_candidates(
    *,
    multi_start_audit: dict[str, Any],
    current_routes: dict[tuple[str, ...], dict[str, Any]],
    replacement_index: AcceptedRouteReplacementIndex,
) -> list[dict[str, Any]]:
    rows = []
    for alternative in multi_start_audit.get("alternatives") or []:
        for component in alternative.get("components") or []:
            key = segment_key(component.get("segment_ids"))
            current_route = current_routes.get(key)
            if not current_route:
                continue
            anchor = component.get("start_anchor") or {}
            manifest_record = replacement_index.match_for_segments(component.get("segment_ids"))
            if manifest_record and not anchor_refs_match(manifest_record.get("accepted_anchor_ref"), anchor.get("anchor_id")):
                manifest_record = None
            if not anchor_is_publicly_accepted(anchor, manifest_record):
                continue
            row = candidate_row_from_component(
                alternative=alternative,
                component=component,
                current_route=current_route,
                manifest_record=manifest_record,
            )
            if material_discovery(row):
                rows.append(row)
    return rows


def record_route_checks(record: dict[str, Any], current_route: dict[str, Any] | None) -> list[str]:
    failures = []
    status = record.get("status")
    if not current_route:
        return ["matching_route_missing_from_field_packet"]
    if status == ACTIVE_STATUS:
        if current_route.get("accepted_replacement_id") != record.get("replacement_id"):
            failures.append("active_replacement_not_applied")
        if current_route.get("route_card_status") != record.get("route_card_status"):
            failures.append("active_replacement_status_missing")
        if current_route.get("certified_route_card") is not False:
            failures.append("active_replacement_marked_certified")
        if record.get("public_anchor_label") and current_route.get("trailhead") != record.get("public_anchor_label"):
            failures.append("active_replacement_public_label_missing")
        baseline = route_metrics(record.get("baseline_card_ref") or {})
        replacement = route_metrics(current_route)
        deltas = dominance_deltas(baseline, replacement)
        if not meets_material_savings(record, deltas):
            failures.append("active_replacement_does_not_meet_required_savings")
        if not no_material_regression(record, deltas):
            failures.append("active_replacement_material_regression")
        distance = current_route.get("anchor_to_credit_endpoint_distance_miles")
        if distance is not None and float_value(distance) > float_value(record.get("max_anchor_to_credit_endpoint_distance_miles")):
            failures.append("active_replacement_anchor_endpoint_distance_too_large")
    elif status == INVESTIGATE_STATUS:
        if current_route.get("accepted_replacement_id") != record.get("replacement_id"):
            failures.append("investigate_replacement_not_marked_on_route")
        if current_route.get("certified_route_card") is not False:
            failures.append("investigate_replacement_silently_certified")
    elif status == WAIVED_STATUS and not record.get("waiver_reason"):
        failures.append("waived_replacement_missing_waiver_reason")
    return failures


def build_audit(
    *,
    field_tool_payload: dict[str, Any],
    multi_start_audit: dict[str, Any],
    replacements_payload: dict[str, Any],
) -> dict[str, Any]:
    replacement_index = AcceptedRouteReplacementIndex(list(replacements_payload.get("replacements") or []))
    current_routes = routes_by_segment(field_tool_payload)
    discovered = discovery_candidates(
        multi_start_audit=multi_start_audit,
        current_routes=current_routes,
        replacement_index=replacement_index,
    )
    missing_from_manifest = [
        row
        for row in discovered
        if not row.get("manifest_replacement_id")
    ]
    manifest_checks = []
    manifest_failures = []
    for record in replacement_index.records:
        current_route = current_routes.get(segment_key(record.get("target_segment_ids")))
        failures = record_route_checks(record, current_route)
        baseline = route_metrics(record.get("baseline_card_ref") or {})
        replacement = route_metrics(current_route or {})
        deltas = dominance_deltas(baseline, replacement) if current_route else None
        check = {
            "replacement_id": record.get("replacement_id"),
            "status": record.get("status"),
            "target_segment_ids": record.get("target_segment_ids"),
            "current_route_label": (current_route or {}).get("label"),
            "current_route_trailhead": (current_route or {}).get("trailhead"),
            "matched_route_label": (current_route or {}).get("label"),
            "matched_route_trailhead": (current_route or {}).get("trailhead"),
            "matched_route_on_foot_miles": (current_route or {}).get("on_foot_miles"),
            "matched_route_p75_minutes": (current_route or {}).get("door_to_door_minutes_p75"),
            "matched_route_p90_minutes": (current_route or {}).get("door_to_door_minutes_p90"),
            "route_card_status": (current_route or {}).get("route_card_status"),
            "packet_visibility": (current_route or {}).get("packet_visibility"),
            "certified_route_card": (current_route or {}).get("certified_route_card"),
            "requires_field_walkthrough": (current_route or {}).get("requires_field_walkthrough"),
            "cue_generation_mode": (current_route or {}).get("cue_generation_mode"),
            "anchor_to_credit_endpoint_distance_miles": (current_route or {}).get("anchor_to_credit_endpoint_distance_miles"),
            "credit_endpoint_used": (current_route or {}).get("credit_endpoint_used"),
            "dominance_deltas": deltas,
            "failures": failures,
        }
        manifest_checks.append(check)
        if failures:
            manifest_failures.append(check)
    failure_count = len(missing_from_manifest) + len(manifest_failures)
    return {
        "schema": "boise_trails_accepted_anchor_preservation_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "summary": {
            "accepted_anchor_dominance_candidate_count": len(discovered),
            "active_replacement_missing_from_manifest_count": len(missing_from_manifest),
            "manifest_record_count": len(replacement_index.records),
            "manifest_failure_count": len(manifest_failures),
            "failure_count": failure_count,
            "passed": failure_count == 0,
        },
        "accepted_anchor_dominance_candidates": discovered,
        "active_replacement_missing_from_manifest": missing_from_manifest,
        "manifest_checks": manifest_checks,
        "manifest_failures": manifest_failures,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Accepted Anchor Preservation Audit",
        "",
        f"Generated: {audit.get('generated_at')}",
        "",
        f"- Passed: `{summary.get('passed')}`",
        f"- Accepted-anchor dominance candidates: {summary.get('accepted_anchor_dominance_candidate_count')}",
        f"- Missing manifest records: {summary.get('active_replacement_missing_from_manifest_count')}",
        f"- Manifest failures: {summary.get('manifest_failure_count')}",
        "",
        "## Manifest Checks",
        "",
        "| Replacement | Status | Route | Trailhead | Failures |",
        "|---|---|---|---|---|",
    ]
    for check in audit.get("manifest_checks") or []:
        failures = ", ".join(check.get("failures") or []) or "none"
        lines.append(
            f"| `{check.get('replacement_id')}` | `{check.get('status')}` | {check.get('current_route_label') or ''} | {check.get('current_route_trailhead') or ''} | {failures} |"
        )
    if audit.get("active_replacement_missing_from_manifest"):
        lines.extend(["", "## Missing Manifest Records", ""])
        for row in audit["active_replacement_missing_from_manifest"]:
            deltas = row.get("dominance_deltas") or {}
            lines.append(
                "- `{route}` `{segments}` from `{anchor}` saves {miles:.2f} mi / {p75:.0f} p75 min and needs a manifest record.".format(
                    route=row.get("current_route_label"),
                    segments=",".join(row.get("target_segment_ids") or []),
                    anchor=row.get("accepted_anchor_label"),
                    miles=float_value(deltas.get("dominance_delta_on_foot_miles")),
                    p75=float_value(deltas.get("dominance_delta_p75_minutes")),
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-json", type=Path, default=DEFAULT_FIELD_TOOL_JSON)
    parser.add_argument("--multi-start-audit-json", type=Path, default=DEFAULT_MULTI_START_AUDIT_JSON)
    parser.add_argument("--accepted-replacements-json", type=Path, default=DEFAULT_ACCEPTED_REPLACEMENTS_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_payload = read_json(args.field_tool_json)
    multi_start_audit = read_json(args.multi_start_audit_json)
    replacements_payload = read_json(args.accepted_replacements_json) if args.accepted_replacements_json.exists() else {"replacements": []}
    audit = build_audit(
        field_tool_payload=field_tool_payload,
        multi_start_audit=multi_start_audit,
        replacements_payload=replacements_payload,
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="accepted-anchor-preservation-audit-2026-05-15",
        inputs=[
            path
            for path in [args.field_tool_json, args.multi_start_audit_json, args.accepted_replacements_json]
            if path.exists()
        ],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/accepted_anchor_preservation_audit.py",
        metadata={"schema": audit["schema"], "passed": audit["summary"]["passed"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps(audit["summary"], indent=2))
    return 0 if audit["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
