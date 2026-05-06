from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_shingle_anchor_exhaustive_probe as probe  # noqa: E402


def row(anchor, p90, *, field_ready=True, route_status="graph_validated", track=True):
    return {
        "anchor_name": anchor,
        "route_status": route_status,
        "track_validation_passed": track,
        "field_ready": field_ready,
        "door_to_door_p90_minutes": p90,
        "door_to_door_p75_minutes": p90 - 20,
        "on_foot_miles": 10.0,
        "parking_risk": 1,
    }


def test_summarize_rows_requires_field_ready_graph_track_and_bound_for_success():
    rows = [
        row("fast-but-not-ready", 240, field_ready=False),
        row("ready-but-too-slow", 292),
        row("ready-invalid-track", 230, track=False),
    ]

    summary = probe.summarize_rows(rows, p90_bound_minutes=260)

    assert summary["strict_success"] is False
    assert summary["under_bound_graph_track_count"] == 1
    assert summary["best_any_anchor"] == "fast-but-not-ready"
    assert summary["best_field_ready_anchor"] == "ready-but-too-slow"
    assert summary["minutes_over_bound_best_field_ready"] == 32


def test_summarize_rows_reports_strict_success_when_ready_valid_under_bound():
    summary = probe.summarize_rows([row("ready", 250)], p90_bound_minutes=260)

    assert summary["strict_success"] is True
    assert summary["strict_success_count"] == 1
