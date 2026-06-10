#!/usr/bin/env python3
"""Build 2026 route-review packs and deterministic exact-credit reviews."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from accepted_route_replacements import (
    ACTIVE_STATUS,
    BLOCKING_STATUSES,
    anchor_refs_match,
    dominance_deltas,
    float_value,
    int_value,
    route_metrics,
    segment_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_FIELD_TOOL_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_MULTI_START_AUDIT_JSON = YEAR_DIR / "checkpoints" / "multi-start-alternative-audit-2026-05-08.json"
DEFAULT_ACCEPTED_REPLACEMENTS_JSON = YEAR_DIR / "inputs" / "accepted-route-replacements-v1.json"
DEFAULT_PRIVATE_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-reviews"
DEFAULT_PUBLIC_OUTPUT_DIR = YEAR_DIR / "checkpoints"

MATERIAL_ON_FOOT_SAVINGS_MILES = 0.25
MATERIAL_P75_SAVINGS_MINUTES = 10

FAIL_DECISIONS = {
    "FAIL_DOMINATED",
    "FAIL_START_UNJUSTIFIED",
    "FAIL_PARKING_CONFIDENCE_REGRESSION",
    "FAIL_CREDIT_INTENT_CONFUSION",
}

PRIVATE_DROP_KEYS = {
    "lat",
    "lon",
    "latitude",
    "longitude",
    "raw_activity_ids",
    "activity_ids",
    "athlete_id",
    "uid",
    "parking_navigation_url",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def route_source_hash(route: dict[str, Any]) -> str:
    relevant = {
        key: route.get(key)
        for key in [
            "label",
            "trailhead",
            "candidate_ids",
            "segment_ids",
            "official_miles",
            "on_foot_miles",
            "door_to_door_minutes_p75",
            "door_to_door_minutes_p90",
            "accepted_replacement_id",
            "start_justification",
            "route_card_status",
            "packet_visibility",
            "certified_route_card",
            "requires_field_walkthrough",
        ]
    }
    parking = route.get("parking") or {}
    relevant["parking"] = {
        key: parking.get(key)
        for key in [
            "name",
            "parking_confidence",
            "source",
            "field_ready",
        ]
    }
    return stable_hash(relevant)


def public_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [public_safe(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        private_anchor = value.get("privacy") == "private_exact_coordinates" or value.get("source_type") == "private_strava_anchor"
        for key, item in value.items():
            if key in PRIVATE_DROP_KEYS:
                continue
            if key == "name" and private_anchor:
                result[key] = value.get("public_anchor_label") or "Private accepted parking anchor"
            elif key == "anchor_id" and private_anchor:
                result[key] = "private-anchor"
            else:
                result[key] = public_safe(item)
        return result
    return value


def normalized_label(value: Any) -> str:
    return str(value or "").strip()


def metric_summary(route: dict[str, Any]) -> dict[str, Any]:
    metrics = route_metrics(route)
    metrics["official_miles"] = float_value(route.get("official_miles"))
    metrics["on_foot_to_official_ratio"] = (
        round(float_value(metrics.get("on_foot_miles")) / float_value(metrics.get("official_miles")), 2)
        if float_value(metrics.get("official_miles"))
        else None
    )
    navigation = route.get("navigation_quality") or {}
    metrics["start_access_gap_miles"] = navigation.get("start_access_gap_miles")
    metrics["return_access_gap_miles"] = navigation.get("return_access_gap_miles")
    metrics["repeat_or_non_credit_burden_miles"] = round(
        max(0.0, float_value(route.get("on_foot_miles")) - float_value(route.get("official_miles"))),
        3,
    )
    return metrics


def manifest_records_by_segment(replacements_payload: dict[str, Any], segments: list[Any]) -> list[dict[str, Any]]:
    key = segment_key(segments)
    return [
        record
        for record in replacements_payload.get("replacements") or []
        if segment_key(record.get("target_segment_ids")) == key
    ]


def public_anchor_label_for_component(
    component: dict[str, Any],
    replacements_payload: dict[str, Any],
) -> tuple[str, str | None]:
    anchor = component.get("start_anchor") or {}
    anchor_ref = anchor.get("anchor_id") or anchor.get("anchor_ref") or anchor.get("name")
    for record in manifest_records_by_segment(replacements_payload, component.get("segment_ids") or []):
        if anchor_refs_match(record.get("accepted_anchor_ref"), anchor_ref):
            return (
                str(record.get("public_anchor_label") or anchor.get("name") or "accepted anchor"),
                str(record.get("replacement_id") or ""),
            )
    return str(anchor.get("name") or "candidate anchor"), None


def multi_start_same_credit_alternatives(
    route: dict[str, Any],
    multi_start_audit: dict[str, Any],
    replacements_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    route_segments = segment_key(route.get("segment_ids"))
    current_metrics = metric_summary(route)
    alternatives: list[dict[str, Any]] = []
    for alternative in multi_start_audit.get("alternatives") or []:
        for component in alternative.get("components") or []:
            if segment_key(component.get("segment_ids")) != route_segments:
                continue
            anchor = component.get("start_anchor") or {}
            label, replacement_id = public_anchor_label_for_component(component, replacements_payload)
            candidate_metrics = {
                "on_foot_miles": float_value(component.get("on_foot_miles")),
                "p75_minutes": int_value(component.get("p75_total_minutes_if_standalone")),
                "p90_minutes": int_value(component.get("p90_total_minutes_if_standalone")),
                "parking_confidence": component.get("parking_confidence") or anchor.get("parking_confidence"),
                "anchor_to_credit_endpoint_distance_miles": anchor.get("distance_to_component_miles"),
            }
            deltas = dominance_deltas(current_metrics, candidate_metrics)
            alternatives.append(
                {
                    "source": "multi_start_alternative_audit",
                    "alternative_id": alternative.get("alternative_id"),
                    "accepted_replacement_id": replacement_id,
                    "anchor_label": label,
                    "anchor_ref": anchor.get("anchor_id"),
                    "parking_confidence": candidate_metrics.get("parking_confidence"),
                    "parking_blockers": component.get("parking_blockers") or [],
                    "field_ready": anchor.get("field_ready"),
                    "same_credit": True,
                    "segment_ids": [str(value) for value in component.get("segment_ids") or []],
                    "official_miles": component.get("official_miles"),
                    "on_foot_miles": candidate_metrics.get("on_foot_miles"),
                    "p75_minutes": candidate_metrics.get("p75_minutes"),
                    "p90_minutes": candidate_metrics.get("p90_minutes"),
                    "material_savings": material_savings(deltas),
                    "deltas": deltas,
                }
            )
    return alternatives


def manifest_same_credit_alternatives(
    route: dict[str, Any],
    replacements_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    current_metrics = metric_summary(route)
    alternatives = []
    for record in manifest_records_by_segment(replacements_payload, route.get("segment_ids") or []):
        baseline = route_metrics(record.get("baseline_card_ref") or {})
        replacement = {
            "on_foot_miles": record.get("on_foot_miles"),
            "p75_minutes": record.get("p75_minutes"),
            "p90_minutes": record.get("p90_minutes"),
            "parking_confidence": record.get("parking_confidence"),
        }
        if replacement["on_foot_miles"] is None:
            replacement = current_metrics if route.get("accepted_replacement_id") == record.get("replacement_id") else baseline
        deltas = dominance_deltas(current_metrics, replacement)
        alternatives.append(
            {
                "source": "accepted_route_replacements_manifest",
                "accepted_replacement_id": record.get("replacement_id"),
                "status": record.get("status"),
                "anchor_label": record.get("public_anchor_label"),
                "anchor_ref": record.get("accepted_anchor_ref"),
                "same_credit": True,
                "segment_ids": [str(value) for value in record.get("target_segment_ids") or []],
                "applied_to_current_route": route.get("accepted_replacement_id") == record.get("replacement_id"),
                "baseline_card_ref": record.get("baseline_card_ref") or {},
                "start_justification": record.get("start_justification"),
                "material_savings": material_savings(deltas),
                "deltas": deltas,
            }
        )
    return alternatives


def material_savings(deltas: dict[str, Any]) -> bool:
    return (
        float_value(deltas.get("dominance_delta_on_foot_miles")) >= MATERIAL_ON_FOOT_SAVINGS_MILES
        or float_value(deltas.get("dominance_delta_p75_minutes")) >= MATERIAL_P75_SAVINGS_MINUTES
    )


def best_dominating_alternative(alternatives: list[dict[str, Any]]) -> dict[str, Any] | None:
    material = [
        alt
        for alt in alternatives
        if alt.get("material_savings")
        and not alt.get("applied_to_current_route")
        and not alt.get("parking_blockers")
    ]
    if not material:
        return None
    return max(
        material,
        key=lambda alt: (
            float_value((alt.get("deltas") or {}).get("dominance_delta_on_foot_miles")),
            float_value((alt.get("deltas") or {}).get("dominance_delta_p75_minutes")),
        ),
    )


def start_justification_for_route(route: dict[str, Any], replacements_payload: dict[str, Any]) -> str | None:
    if route.get("start_justification"):
        return str(route["start_justification"]).strip()
    replacement_id = route.get("accepted_replacement_id")
    if replacement_id:
        for record in replacements_payload.get("replacements") or []:
            if record.get("replacement_id") == replacement_id and record.get("start_justification"):
                return str(record["start_justification"]).strip()
    return None


# Generic location-type words that differ between a route's display anchor and an
# alternative's anchor label without meaning a different physical start (e.g.
# "West Climb" vs "West Climb Trailhead", "... road-parking" vs "... road-parking
# anchor"). Distinguishing location qualifiers like "N 36th St" are NOT here, so
# "Full Sail" and "Full Sail Trailhead, N 36th St Parking" stay DIFFERENT anchors.
_GENERIC_ANCHOR_TOKENS = {
    "anchor", "parking", "trailhead", "roadside", "road", "lot", "the", "at", "near",
}


def _anchor_token_set(value: Any) -> frozenset[str]:
    tokens = "".join(ch if ch.isalnum() else " " for ch in str(value or "").lower()).split()
    return frozenset(token for token in tokens if token not in _GENERIC_ANCHOR_TOKENS)


def _same_anchor(left: Any, right: Any) -> bool:
    a, b = _anchor_token_set(left), _anchor_token_set(right)
    return bool(a) and bool(b) and a == b


def route_pack(
    route: dict[str, Any],
    multi_start_audit: dict[str, Any],
    replacements_payload: dict[str, Any],
) -> dict[str, Any]:
    alternatives = manifest_same_credit_alternatives(route, replacements_payload)
    alternatives.extend(multi_start_same_credit_alternatives(route, multi_start_audit, replacements_payload))
    # A same-credit alternative can only DOMINATE if it offers a DIFFERENT,
    # better start anchor. When the route is already parked at the alternative's
    # anchor, it IS that alternative; its lower planning-estimate mileage is not
    # a re-anchoring opportunity (route-shape efficiency is handled by the
    # efficiency/repeat audits, not the start-dominance gate). This prevents the
    # gate from flagging routes as "dominated" by their own current anchor.
    current_anchor = route.get("trailhead_display") or route.get("trailhead")
    if not current_anchor:
        parking = route.get("parking") or {}
        current_anchor = parking.get("display_name") or parking.get("name")
    for alternative in alternatives:
        if not alternative.get("applied_to_current_route") and _same_anchor(
            current_anchor, alternative.get("anchor_label")
        ):
            alternative["applied_to_current_route"] = True
            alternative["same_as_current_anchor"] = True
    justification = start_justification_for_route(route, replacements_payload)
    return {
        "route_label": route.get("label"),
        "route_source_hash": route_source_hash(route),
        "current_start": route.get("trailhead"),
        "start_justification": justification,
        "official_segment_ids": [str(value) for value in route.get("segment_ids") or []],
        "trail_names": route.get("trails") or [],
        "metrics": metric_summary(route),
        "parking": route.get("parking") or {},
        "accepted_replacement_id": route.get("accepted_replacement_id"),
        "route_card_status": route.get("route_card_status"),
        "certified_route_card": route.get("certified_route_card"),
        "requires_field_walkthrough": route.get("requires_field_walkthrough"),
        "same_credit_alternatives": alternatives,
        "prior_audit_evidence": route.get("validation") or {},
    }


def review_for_pack(pack: dict[str, Any]) -> dict[str, Any]:
    route_label = str(pack.get("route_label") or "")
    segment_ids = [str(value) for value in pack.get("official_segment_ids") or []]
    start_justification = str(pack.get("start_justification") or "").strip()
    dominating = best_dominating_alternative(pack.get("same_credit_alternatives") or [])
    metrics = pack.get("metrics") or {}

    decision = "PASS_NON_DOMINATED"
    confidence = "high"
    required_action = "No route-review action required."
    waiver_allowed = False
    start_assessment = f"Current start is {pack.get('current_start') or 'unknown'}."
    if start_justification:
        start_assessment += f" Start justification: {start_justification}"
    else:
        decision = "FAIL_START_UNJUSTIFIED"
        required_action = "Add start_justification explaining why this start is correct for the exact official credit target."
        waiver_allowed = False

    if dominating:
        deltas = dominating.get("deltas") or {}
        decision = "FAIL_DOMINATED"
        waiver_allowed = True
        required_action = (
            f"Regenerate {route_label} from {dominating.get('anchor_label')} or add a route/source-hashed waiver "
            "explaining why that same-credit anchor is invalid."
        )

    if decision == "PASS_NON_DOMINATED" and pack.get("certified_route_card") is False:
        decision = "PASS_WITH_JUSTIFIED_BURDEN"
        required_action = "Route is justified for this gate but remains provisional until field walkthrough/certification is complete."

    evidence = [
        "route review pack",
        "field-tool route record",
        "accepted-route-replacements manifest",
        "multi-start alternative audit",
    ]
    if start_justification:
        evidence.append("start_justification")

    dominating_deltas = (dominating or {}).get("deltas") or {}
    dominance_reason = "No material accepted same-credit alternative found."
    if dominating:
        dominance_reason = (
            f"Same credit can be earned from {dominating.get('anchor_label')} with "
            f"{float_value(dominating_deltas.get('dominance_delta_on_foot_miles')):.2f} fewer miles and "
            f"{float_value(dominating_deltas.get('dominance_delta_p75_minutes')):.0f} fewer p75 minutes."
        )

    return {
        "schema": "boise_trails_route_review_v1",
        "route_id": route_label,
        "route_label": route_label,
        "route_source_hash": pack.get("route_source_hash"),
        "segment_ids": segment_ids,
        "decision": decision,
        "confidence": confidence,
        "official_credit_summary": f"Route buys exact official segment set {', '.join(segment_ids) or 'unknown'}.",
        "start_anchor_assessment": start_assessment,
        "human_footmile_assessment": (
            f"{metrics.get('on_foot_miles')} on-foot miles for {metrics.get('official_miles')} official miles; "
            f"non-credit/repeat burden is {metrics.get('repeat_or_non_credit_burden_miles')} miles."
        ),
        "dominance_assessment": {
            "is_dominated": bool(dominating),
            "dominating_anchor": (dominating or {}).get("anchor_label"),
            "same_credit": bool(dominating),
            "estimated_miles_saved": (
                round(float_value(dominating_deltas.get("dominance_delta_on_foot_miles")), 3)
                if dominating
                else None
            ),
            "estimated_minutes_saved": (
                round(float_value(dominating_deltas.get("dominance_delta_p75_minutes")), 3)
                if dominating
                else None
            ),
            "reason": dominance_reason,
        },
        "required_action": required_action,
        "waiver_allowed": waiver_allowed,
        "evidence": evidence,
    }


def build_report(
    field_tool_payload: dict[str, Any],
    multi_start_audit: dict[str, Any] | None = None,
    replacements_payload: dict[str, Any] | None = None,
    route_labels: set[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    multi_start_audit = multi_start_audit or {"alternatives": []}
    replacements_payload = replacements_payload or {"replacements": []}
    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    selected_routes = []
    requested = {label.upper() for label in route_labels or set()}
    for route in field_tool_payload.get("routes") or []:
        label = str(route.get("label") or "")
        if requested and label.upper() not in requested:
            continue
        selected_routes.append(route)

    packs = [route_pack(route, multi_start_audit, replacements_payload) for route in selected_routes]
    reviews = [review_for_pack(pack) for pack in packs]
    return {
        "schema": "boise_trails_route_review_pack_v1",
        "generated_at": generated_at,
        "thresholds": {
            "material_on_foot_savings_miles": MATERIAL_ON_FOOT_SAVINGS_MILES,
            "material_p75_savings_minutes": MATERIAL_P75_SAVINGS_MINUTES,
        },
        "summary": {
            "route_count": len(packs),
            "deterministic_failure_count": sum(1 for review in reviews if review.get("decision") in FAIL_DECISIONS),
            "single_segment_route_count": sum(1 for pack in packs if len(pack.get("official_segment_ids") or []) == 1),
        },
        "routes": packs,
        "reviews": reviews,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# 2026 Route Review Pack",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Summary",
        "",
        f"- Routes reviewed: {report.get('summary', {}).get('route_count')}",
        f"- Deterministic failures: {report.get('summary', {}).get('deterministic_failure_count')}",
        f"- Single-segment routes included: {report.get('summary', {}).get('single_segment_route_count')}",
        "",
    ]
    for pack, review in zip(report.get("routes") or [], report.get("reviews") or []):
        lines.extend(
            [
                f"## {pack.get('route_label')}",
                "",
                f"- Decision: `{review.get('decision')}`",
                f"- Start: {pack.get('current_start')}",
                f"- Start justification: {pack.get('start_justification') or 'MISSING'}",
                f"- Official segments: {', '.join(pack.get('official_segment_ids') or [])}",
                f"- Official miles: {pack.get('metrics', {}).get('official_miles')}",
                f"- On-foot miles: {pack.get('metrics', {}).get('on_foot_miles')}",
                f"- p75 / p90: {pack.get('metrics', {}).get('p75_minutes')} / {pack.get('metrics', {}).get('p90_minutes')} min",
                f"- Non-credit/repeat burden: {pack.get('metrics', {}).get('repeat_or_non_credit_burden_miles')} mi",
                "",
                "### Same-Credit Alternatives",
                "",
            ]
        )
        alternatives = pack.get("same_credit_alternatives") or []
        if not alternatives:
            lines.append("- None found.")
        for alt in alternatives:
            deltas = alt.get("deltas") or {}
            lines.append(
                "- "
                f"{alt.get('anchor_label')} ({alt.get('source')}): "
                f"{alt.get('on_foot_miles')} mi / {alt.get('p75_minutes')} min; "
                f"saves {float_value(deltas.get('dominance_delta_on_foot_miles')):.2f} mi / "
                f"{float_value(deltas.get('dominance_delta_p75_minutes')):.0f} min."
            )
        lines.extend(["", f"Required action: {review.get('required_action')}", ""])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-json", type=Path, default=DEFAULT_FIELD_TOOL_JSON)
    parser.add_argument("--multi-start-audit-json", type=Path, default=DEFAULT_MULTI_START_AUDIT_JSON)
    parser.add_argument("--accepted-replacements-json", type=Path, default=DEFAULT_ACCEPTED_REPLACEMENTS_JSON)
    parser.add_argument("--route-label", action="append", default=[])
    parser.add_argument("--all-field-packet-routes", action="store_true")
    parser.add_argument("--basename", default="route-review-2026")
    parser.add_argument("--private-output-dir", type=Path, default=DEFAULT_PRIVATE_OUTPUT_DIR)
    parser.add_argument("--public-output-dir", type=Path, default=DEFAULT_PUBLIC_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_payload = read_json(args.field_tool_json)
    multi_start_audit = read_json(args.multi_start_audit_json) if args.multi_start_audit_json.exists() else {"alternatives": []}
    replacements_payload = read_json(args.accepted_replacements_json) if args.accepted_replacements_json.exists() else {"replacements": []}
    labels = None if args.all_field_packet_routes or not args.route_label else set(args.route_label)
    report = build_report(
        field_tool_payload=field_tool_payload,
        multi_start_audit=multi_start_audit,
        replacements_payload=replacements_payload,
        route_labels=labels,
    )
    private_pack = args.private_output_dir / f"{args.basename}.pack.json"
    private_md = args.private_output_dir / f"{args.basename}.pack.md"
    private_review = args.private_output_dir / f"{args.basename}.review.json"
    public_pack = args.public_output_dir / f"{args.basename}.public.json"
    public_md = args.public_output_dir / f"{args.basename}.public.md"

    write_json(private_pack, report)
    private_md.parent.mkdir(parents=True, exist_ok=True)
    private_md.write_text(markdown_report(report), encoding="utf-8")
    review_payload: Any = report["reviews"][0] if len(report["reviews"]) == 1 else {"schema": "boise_trails_route_review_result_set_v1", "reviews": report["reviews"]}
    write_json(private_review, review_payload)

    safe_report = public_safe(copy.deepcopy(report))
    write_json(public_pack, safe_report)
    public_md.parent.mkdir(parents=True, exist_ok=True)
    public_md.write_text(markdown_report(safe_report), encoding="utf-8")

    failures = report.get("summary", {}).get("deterministic_failure_count", 0)
    print(f"Wrote {private_pack}")
    print(f"Wrote {private_review}")
    print(f"Wrote public-safe checkpoint {public_pack}")
    print(f"Reviewed {report['summary']['route_count']} route(s); deterministic failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
