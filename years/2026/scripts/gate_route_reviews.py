#!/usr/bin/env python3
"""Gate 2026 route-review decisions with route/source-hashed waivers."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from accepted_route_replacements import segment_key


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent

DEFAULT_WAIVERS_JSON = YEAR_DIR / "inputs" / "personal" / "private" / "route-review-waivers-2026.private.json"

FAIL_DECISIONS = {
    "FAIL_DOMINATED",
    "FAIL_START_UNJUSTIFIED",
    "FAIL_PARKING_CONFIDENCE_REGRESSION",
    "FAIL_CREDIT_INTENT_CONFUSION",
}

WARN_DECISIONS = {
    "WARN_NEEDS_MAP_REVIEW",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_reviews(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("reviews"), list):
        return [item for item in payload["reviews"] if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def normalize_waivers(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [item for item in payload.get("waivers") or [] if isinstance(item, dict)]
    return []


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def route_label(review: dict[str, Any]) -> str:
    return str(review.get("route_label") or review.get("route_id") or "")


def valid_waiver_for_review(review: dict[str, Any], waiver: dict[str, Any], today: date) -> tuple[bool, str | None]:
    if str(waiver.get("route_label") or waiver.get("route_id") or "") != route_label(review):
        return False, "route_label_mismatch"
    if segment_key(waiver.get("segment_ids") or []) != segment_key(review.get("segment_ids") or []):
        return False, "segment_ids_mismatch"
    if str(waiver.get("route_source_hash") or "") != str(review.get("route_source_hash") or ""):
        return False, "route_source_hash_mismatch"
    if not str(waiver.get("reason") or "").strip():
        return False, "missing_reason"
    if not str(waiver.get("approver") or "").strip():
        return False, "missing_approver"
    if not parse_date(waiver.get("date")):
        return False, "invalid_date"
    expires = parse_date(waiver.get("expires"))
    if not expires:
        return False, "invalid_expiry"
    if expires < today:
        return False, "expired"
    return True, None


def matching_waiver(review: dict[str, Any], waivers: list[dict[str, Any]], today: date) -> dict[str, Any] | None:
    for waiver in waivers:
        valid, _reason = valid_waiver_for_review(review, waiver, today)
        if valid:
            return waiver
    return None


def waiver_rejection_reasons(review: dict[str, Any], waivers: list[dict[str, Any]], today: date) -> list[str]:
    reasons = []
    for waiver in waivers:
        label = str(waiver.get("route_label") or waiver.get("route_id") or "")
        if label != route_label(review):
            continue
        valid, reason = valid_waiver_for_review(review, waiver, today)
        if not valid and reason:
            reasons.append(reason)
    return sorted(set(reasons))


def evaluate_reviews(
    reviews: list[dict[str, Any]],
    waivers: list[dict[str, Any]] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or date.today()
    waivers = waivers or []
    failures = []
    warnings = []
    waived = []
    passed = []
    for review in reviews:
        decision = str(review.get("decision") or "")
        label = route_label(review)
        if decision in WARN_DECISIONS:
            warnings.append(
                {
                    "route_label": label,
                    "decision": decision,
                    "required_action": review.get("required_action"),
                }
            )
            passed.append(label)
            continue
        if decision in FAIL_DECISIONS:
            waiver = matching_waiver(review, waivers, today)
            if waiver:
                waived.append(
                    {
                        "route_label": label,
                        "decision": decision,
                        "waiver_expires": waiver.get("expires"),
                    }
                )
                passed.append(label)
                continue
            failures.append(
                {
                    "route_label": label,
                    "decision": decision,
                    "required_action": review.get("required_action"),
                    "waiver_rejection_reasons": waiver_rejection_reasons(review, waivers, today),
                }
            )
            continue
        passed.append(label)
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "waived_count": len(waived),
        "failures": failures,
        "warnings": warnings,
        "waived": waived,
        "passed_routes": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", type=Path, nargs="+")
    parser.add_argument("--waivers-json", type=Path, default=DEFAULT_WAIVERS_JSON)
    parser.add_argument("--today", default=None, help="Override current date as YYYY-MM-DD for tests/repro.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reviews: list[dict[str, Any]] = []
    for path in args.review_json:
        reviews.extend(normalize_reviews(read_json(path)))
    waivers = normalize_waivers(read_json(args.waivers_json)) if args.waivers_json.exists() else []
    today = parse_date(args.today) if args.today else date.today()
    if today is None:
        raise SystemExit("--today must be YYYY-MM-DD")
    result = evaluate_reviews(reviews, waivers=waivers, today=today)
    if result["warnings"]:
        print("AI route-review warnings:")
        for warning in result["warnings"]:
            print(f"- {warning['route_label']}: {warning['decision']}: {warning.get('required_action')}")
    if result["failures"]:
        print("AI route-review gate failed:")
        for failure in result["failures"]:
            reasons = ", ".join(failure.get("waiver_rejection_reasons") or [])
            suffix = f" waiver rejected: {reasons}" if reasons else ""
            print(f"- {failure['route_label']}: {failure['decision']}: {failure.get('required_action')}{suffix}")
        return 1
    print("AI route-review gate passed.")
    if result["waived"]:
        print("Waived failures:")
        for waived in result["waived"]:
            print(f"- {waived['route_label']}: {waived['decision']} until {waived.get('waiver_expires')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
