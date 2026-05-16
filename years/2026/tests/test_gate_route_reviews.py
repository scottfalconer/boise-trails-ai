from datetime import date
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import gate_route_reviews as gate  # noqa: E402


TODAY = date(2026, 5, 15)


def review(decision="FAIL_DOMINATED"):
    return {
        "route_id": "FD14D",
        "route_label": "FD14D",
        "route_source_hash": "route-hash-1",
        "segment_ids": ["1482"],
        "decision": decision,
        "required_action": "Regenerate from lower 36th.",
    }


def waiver(**overrides):
    payload = {
        "route_label": "FD14D",
        "segment_ids": ["1482"],
        "route_source_hash": "route-hash-1",
        "reason": "Closer anchor is invalid during a temporary closure.",
        "approver": "Scott",
        "date": "2026-05-15",
        "expires": "2026-12-31",
    }
    payload.update(overrides)
    return payload


def test_fd14d_deterministic_dominance_fails_without_waiver():
    result = gate.evaluate_reviews([review()], waivers=[], today=TODAY)

    assert result["passed"] is False
    assert result["failure_count"] == 1
    assert result["failures"][0]["route_label"] == "FD14D"


def test_valid_route_source_hashed_waiver_passes():
    result = gate.evaluate_reviews([review()], waivers=[waiver()], today=TODAY)

    assert result["passed"] is True
    assert result["failure_count"] == 0
    assert result["waived_count"] == 1


def test_expired_waiver_fails():
    result = gate.evaluate_reviews([review()], waivers=[waiver(expires="2026-05-14")], today=TODAY)

    assert result["passed"] is False
    assert "expired" in result["failures"][0]["waiver_rejection_reasons"]


def test_stale_hash_waiver_fails():
    result = gate.evaluate_reviews([review()], waivers=[waiver(route_source_hash="old-hash")], today=TODAY)

    assert result["passed"] is False
    assert "route_source_hash_mismatch" in result["failures"][0]["waiver_rejection_reasons"]


def test_segment_mismatched_waiver_fails():
    result = gate.evaluate_reviews([review()], waivers=[waiver(segment_ids=["1483"])], today=TODAY)

    assert result["passed"] is False
    assert "segment_ids_mismatch" in result["failures"][0]["waiver_rejection_reasons"]


def test_warn_needs_map_review_reports_without_blocking():
    result = gate.evaluate_reviews([review("WARN_NEEDS_MAP_REVIEW")], waivers=[], today=TODAY)

    assert result["passed"] is True
    assert result["warning_count"] == 1
    assert result["failure_count"] == 0
