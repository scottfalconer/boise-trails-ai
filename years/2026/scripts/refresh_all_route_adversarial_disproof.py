#!/usr/bin/env python3
"""Refresh the public-safe current-route proof registry from the field packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import gate_route_reviews  # noqa: E402

YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "all-route-adversarial-disproof-2026-05-16.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "all-route-adversarial-disproof-2026-05-16.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "all-route-adversarial-disproof-2026-05-16-manifest.json"
DEFAULT_PRIVATE_MAP_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_PUBLIC_MAP_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_REVIEW_JSON = YEAR_DIR / "outputs" / "private" / "route-reviews" / "route-review-all-dev.review.json"
DEFAULT_WAIVERS_JSON = gate_route_reviews.DEFAULT_WAIVERS_JSON
RUN_ID = "all-route-adversarial-disproof-2026-05-16"
# Decision per route is now derived from the deterministic review, not a constant.
ACCEPTED_DECISION = "HOLD_CURRENT_RECERTIFIED"
FAILED_DECISION = "NEEDS_REANCHOR_OR_WAIVER"


class MissingReviewError(RuntimeError):
    """Raised when a current route has no deterministic review (fail closed)."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def rounded(value: Any, digits: int = 2) -> float:
    return round(float(value or 0), digits)


def route_hash(route: dict[str, Any]) -> str:
    public_shape = {
        "route_code": route.get("route_code") or route.get("label"),
        "route_name": route.get("route_name"),
        "candidate_ids": route.get("candidate_ids") or [],
        "segment_ids": route.get("segment_ids") or [],
        "official_miles": rounded(route.get("official_miles")),
        "on_foot_miles": rounded(route.get("on_foot_miles")),
        "p75": route.get("door_to_door_minutes_p75"),
        "p90": route.get("door_to_door_minutes_p90"),
        "field_readiness_status": route.get("field_readiness_status"),
    }
    encoded = json.dumps(public_shape, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def route_area(route: dict[str, Any]) -> str:
    if route.get("route_name_area"):
        return str(route["route_name_area"])
    route_name = str(route.get("route_name") or "")
    if ":" in route_name:
        return route_name.split(":", 1)[0].strip()
    return str(route.get("block_name") or "Current field packet")


def repeat_miles(route: dict[str, Any]) -> float:
    total = 0.0
    for cue in route.get("wayfinding_cues") or []:
        total += float(cue.get("official_repeat_miles") or 0)
    return round(total, 2)


def metrics(route: dict[str, Any]) -> dict[str, Any]:
    official = rounded(route.get("official_miles"))
    on_foot = rounded(route.get("on_foot_miles"))
    return {
        "official_miles": official,
        "on_foot_miles": on_foot,
        "on_foot_to_official_ratio": round(on_foot / official, 2) if official else None,
        "p75_minutes": int(route.get("door_to_door_minutes_p75") or 0),
        "p90_minutes": int(route.get("door_to_door_minutes_p90") or 0),
        "non_credit_or_repeat_miles": round(max(0.0, on_foot - official), 2),
        "declared_repeat_miles": repeat_miles(route),
    }


def classify_route(
    route: dict[str, Any],
    review: dict[str, Any] | None,
    failed_labels: set[str],
    waived_labels: set[str],
) -> dict[str, Any]:
    """Fail-closed classification of one route against its deterministic review.

    A route is accepted only when a current review exists, describes the same
    official segment set, and the deterministic gate did not fail it. A missing
    or stale review, or an unwaived FAIL_* decision, is recorded as a failure so
    the registry can never rubber-stamp a route the dominance gate has not seen.
    """
    label = str(route.get("route_code") or route.get("label") or route.get("outing_id"))
    route_segments = [str(value) for value in route.get("segment_ids") or []]
    if review is None:
        return {"accepted": False, "state": "failed_missing_review", "review_decision": None}
    review_segments = [str(value) for value in review.get("segment_ids") or []]
    if sorted(review_segments) != sorted(route_segments):
        return {"accepted": False, "state": "failed_stale_review", "review_decision": review.get("decision")}
    decision = str(review.get("decision") or "")
    if label in waived_labels:
        return {"accepted": True, "state": "accepted_waiver", "review_decision": decision}
    if label in failed_labels or decision in gate_route_reviews.FAIL_DECISIONS:
        return {"accepted": False, "state": "failed_dominated", "review_decision": decision}
    return {"accepted": True, "state": "accepted_current", "review_decision": decision}


def route_record(route: dict[str, Any], review: dict[str, Any] | None, status: dict[str, Any]) -> dict[str, Any]:
    route_label = str(route.get("route_code") or route.get("label") or route.get("outing_id"))
    candidate_ids = [str(value) for value in route.get("candidate_ids") or [] if str(value)]
    accepted = status["accepted"]
    decision = ACCEPTED_DECISION if accepted else FAILED_DECISION
    review = review or {}
    required_action = review.get("required_action")
    dominance = review.get("dominance_assessment") or {}
    return {
        "route_label": route_label,
        "candidate_id": candidate_ids[0] if candidate_ids else "",
        "candidate_ids": candidate_ids,
        "route_source_hash": route_hash(route),
        "review_route_source_hash": review.get("route_source_hash"),
        "area": route_area(route),
        "decision": decision,
        "review_decision": status["review_decision"],
        "review_state": status["state"],
        "current_start": route.get("trailhead_display") or route.get("trailhead"),
        "official_segment_ids": [str(value) for value in route.get("segment_ids") or []],
        "trail_names": [str(value) for value in route.get("trails") or []],
        "metrics": metrics(route),
        "warning_codes_closed": [],
        "same_credit_anchor_attack": (
            dominance.get("summary")
            or ("No same-credit anchor beats this route in the current deterministic review."
                if accepted
                else "Deterministic review found a dominant same-credit anchor; route is NOT recertified.")
        ),
        "required_action": required_action,
        "disproof_result": (
            "Kept as the current recertified field-packet route."
            if accepted
            else f"Open dominance failure ({status['review_decision']}); re-anchor or add a route/source-hashed waiver."
        ),
        "what_would_disprove_later": "A certified replacement that covers the same required official segments from accepted access with lower runnable cost and no coverage, legality, timing, or future-day regression.",
        "requires_field_walkthrough": bool(route.get("requires_field_walkthrough") or False),
        "field_readiness_status": route.get("field_readiness_status"),
    }


def proof_record(route: dict[str, Any], review: dict[str, Any] | None, status: dict[str, Any]) -> dict[str, Any]:
    record = route_record(route, review, status)
    accepted = status["accepted"]
    checks = {
        "gpx_continuity_passed": (route.get("validation") or {}).get("passed") is True,
        "current_route_has_p75_time": bool(route.get("door_to_door_minutes_p75")),
        "current_route_has_dem_effort": bool(route.get("effort")),
        # Dominance checks are now sourced from the deterministic review, not hardcoded.
        "no_better_exact_generated_candidate": accepted,
        "no_dominant_boundary_recombination": accepted,
        "no_dominant_global_optimizer_replacement": accepted,
        "field_tool_completion_passed": route.get("field_ready") is True,
        "special_management_passed": (route.get("special_management") or {}).get("status") == "passed",
    }
    return {
        "candidate_id": record["candidate_id"],
        "candidate_ids": record["candidate_ids"],
        "labels": [record["route_label"]],
        "area": record["area"],
        "status": "accepted_current" if accepted else status["state"],
        "decision": record["decision"],
        "review_decision": status["review_decision"],
        "metrics": record["metrics"],
        "evidence": [
            "years/2026/outputs/private/route-reviews/route-review-all-dev.review.json: deterministic same-credit dominance review for this route",
            "years/2026/scripts/gate_route_reviews.py: pass/fail/waiver gate applied to that review",
            "docs/field-packet/field-tool-data.json: route is present in the regenerated phone packet",
        ],
        "checks": checks,
        "remaining_caveat": record["what_would_disprove_later"],
    }


def decision_counts(routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        decision = str(route.get("decision") or "unknown")
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def area_counts(routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        area = str(route.get("area") or "Current field packet")
        counts[area] = counts.get(area, 0) + 1
    return counts


def build_registry(
    field_tool_data: dict[str, Any],
    reviews: list[dict[str, Any]],
    waivers: list[dict[str, Any]],
    today: date,
    generated_at: str,
) -> dict[str, Any]:
    source_routes = list(field_tool_data.get("routes") or [])
    reviews_by_label = {gate_route_reviews.route_label(review): review for review in reviews}
    verdict = gate_route_reviews.evaluate_reviews(reviews, waivers, today)
    failed_labels = {row["route_label"] for row in verdict["failures"]}
    waived_labels = {row["route_label"] for row in verdict["waived"]}

    routes: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for route in source_routes:
        label = str(route.get("route_code") or route.get("label") or route.get("outing_id"))
        review = reviews_by_label.get(label)
        status = classify_route(route, review, failed_labels, waived_labels)
        routes.append(route_record(route, review, status))
        proofs.append(proof_record(route, review, status))

    failed_routes = [r for r in routes if r["review_state"] != "accepted_current" and r["review_state"] != "accepted_waiver"]
    failure_count = len(failed_routes)
    return {
        "schema": "boise_trails_all_route_adversarial_disproof_v1",
        "run_id": RUN_ID,
        "generated_at": generated_at,
        "artifact_contract": "Public-safe current-route proof registry keyed to the deterministic route-review dominance gate. A route is accepted only when a current review covers its exact segment set and did not fail (or carries a valid waiver).",
        "source_files": {
            "field_tool_data": "docs/field-packet/field-tool-data.json",
            "field_packet_manifest": "docs/field-packet/manifest.json",
            "route_review": "years/2026/outputs/private/route-reviews/route-review-all-dev.review.json",
            "private_canonical_map_data": "years/2026/outputs/private/2026-outing-menu-map-data.json",
            "public_sanitized_map_data": "outing-menu-map-data.json",
        },
        "summary": {
            "route_count": len(routes),
            "proof_count": len(proofs),
            "decision_counts": decision_counts(routes),
            "area_counts": area_counts(routes),
            "deterministic_same_credit_failure_count": failure_count,
            "failed_route_labels": [r["route_label"] for r in failed_routes],
            "route_efficiency_verdict": "current_routes_proofed" if failure_count == 0 else "open_dominance_failures",
            "route_efficiency_achieved": failure_count == 0,
            "accepted_active_candidate_target": len(routes) - failure_count,
            "review_today": today.isoformat(),
        },
        "routes": routes,
        "proofs": proofs,
    }


def render_markdown(registry: dict[str, Any]) -> str:
    failed = registry["summary"].get("failed_route_labels") or []
    lines = [
        "# All-Route Adversarial Disproof",
        "",
        "This public-safe registry is refreshed from the deterministic route-review dominance gate. A route is accepted only when a current review covers its exact official segment set and did not fail (or carries a valid waiver). It is not a global optimum proof; day-of condition, closure, signage, and access checks still apply.",
        "",
        "## Summary",
        "",
        f"- Routes: {registry['summary']['route_count']}",
        f"- Proofs: {registry['summary']['proof_count']}",
        f"- Accepted (recertified): {registry['summary']['accepted_active_candidate_target']}",
        f"- Deterministic same-credit failures: {registry['summary']['deterministic_same_credit_failure_count']}"
        + (f" ({', '.join(failed)})" if failed else ""),
        "",
        "## Routes",
        "",
    ]
    for route in registry.get("routes") or []:
        metrics_text = (
            f"{route['metrics']['official_miles']} official mi / "
            f"{route['metrics']['on_foot_miles']} on-foot mi / "
            f"p75 {route['metrics']['p75_minutes']} min"
        )
        lines.extend(
            [
                f"### {route['route_label']} - {route['area']}",
                "",
                f"- Candidate id: `{route['candidate_id']}`",
                f"- Decision: `{route['decision']}`",
                f"- Metrics: {metrics_text}",
                f"- Result: {route['disproof_result']}",
                f"- Later disproof: {route['what_would_disprove_later']}",
                "",
            ]
        )
    return "\n".join(lines)


def build_manifest(registry: dict[str, Any], output_json: Path, output_md: Path) -> dict[str, Any]:
    return {
        "run_id": registry["run_id"],
        "generated_at": registry["generated_at"],
        "source_files": registry["source_files"],
        "outputs": {
            "canonical_json": str(output_json.relative_to(REPO_ROOT)),
            "canonical_md": str(output_md.relative_to(REPO_ROOT)),
        },
        "summary": registry["summary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--waivers-json", type=Path, default=DEFAULT_WAIVERS_JSON)
    parser.add_argument("--today", default=None, help="Override current date as YYYY-MM-DD for tests/repro.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def load_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise MissingReviewError(
            f"route-review pack not found at {path}; run build_route_review_pack.py "
            "before refreshing the proof registry (fail closed: cannot prove dominance)."
        )
    return gate_route_reviews.normalize_reviews(read_json(path))


def load_waivers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return gate_route_reviews.normalize_waivers(read_json(path))


def main() -> int:
    args = parse_args()
    today = date.fromisoformat(args.today) if args.today else datetime.now(timezone.utc).date()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    reviews = load_reviews(args.review_json)
    waivers = load_waivers(args.waivers_json)
    registry = build_registry(read_json(args.field_tool_data_json), reviews, waivers, today, generated_at)
    write_json(args.output_json, registry)
    args.output_md.write_text(render_markdown(registry), encoding="utf-8")
    write_json(args.manifest_json, build_manifest(registry, args.output_json, args.output_md))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.manifest_json}")
    print(json.dumps(registry["summary"], indent=2))
    failure_count = registry["summary"]["deterministic_same_credit_failure_count"]
    if failure_count:
        print(
            f"FAIL-CLOSED: {failure_count} route(s) have open dominance failures or no current review: "
            f"{registry['summary']['failed_route_labels']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
