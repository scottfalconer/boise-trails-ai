import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "net_effort_reduction_proof.py"


def load_module():
    spec = importlib.util.spec_from_file_location("net_effort_reduction_proof", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field_tool_data():
    return {
        "summary": {"segment_count_in_field_menu": 4},
        "routes": [
            {
                "outing_id": "15-1",
                "label": "15A-1",
                "segment_ids": ["1"],
                "official_miles": 1.0,
                "on_foot_miles": 10.0,
                "door_to_door_minutes_p75": 100,
                "door_to_door_minutes_p90": 120,
            },
            {
                "outing_id": "16-2",
                "label": "16A-2",
                "segment_ids": ["2", "3"],
                "official_miles": 3.0,
                "on_foot_miles": 20.0,
                "door_to_door_minutes_p75": 200,
                "door_to_door_minutes_p90": 240,
            },
            {
                "outing_id": "x",
                "label": "Other",
                "segment_ids": ["4"],
                "official_miles": 1.0,
                "on_foot_miles": 5.0,
                "door_to_door_minutes_p75": 50,
                "door_to_door_minutes_p90": 60,
            },
        ],
    }


def source_review(extra_ids=None, missed_ids=None, status="completed", direction_ok=True):
    return {
        "extra_completed_segment_ids": extra_ids if extra_ids is not None else ["2"],
        "missed_segment_ids": missed_ids or [],
        "segment_reviews": [
            {
                "seg_id": "2",
                "completion_status": status,
                "endpoints_ok": True,
                "direction_ok": direction_ok,
                "match_fraction": 1.0,
            }
        ],
    }


def replacement_probe(track_valid=True, include_effort=True):
    row = {
        "seg_id": "3",
        "trail_name": "Sheep Camp Trail",
        "candidate_id": "single-segment-3-sheep-camp",
        "official_miles": 1.0,
        "on_foot_miles": 4.0,
        "door_to_door_p75_minutes": 40,
        "door_to_door_p90_minutes": 55,
        "moving_effort_p75_minutes": 25,
        "ascent_ft": 300,
        "grade_adjusted_miles": 1.4,
        "validation_passed": True,
        "track_validation_passed": track_valid,
    }
    if not include_effort:
        row.pop("moving_effort_p75_minutes")
    return {"probe_rows": [row]}


def test_proves_full_menu_effort_reduction_and_coverage_preservation():
    module = load_module()
    proof = module.build_net_effort_reduction_proof(
        field_tool_data(),
        source_review(),
        replacement_probe(),
        latent_segment_id="2",
        retained_segment_id="3",
    )

    assert proof["status"] == "proved_planning_net_effort_reduction"
    assert proof["summary"]["reduction_on_foot_miles"] == 16.0
    assert proof["summary"]["reduction_p75_minutes"] == 160
    assert proof["summary"]["current_unique_segment_count"] == 4
    assert proof["summary"]["proposed_unique_segment_count"] == 4
    assert proof["coverage"]["missing_after_repair"] == []
    assert proof["coverage"]["proposed_duplicate_segment_ids"] == []
    assert all(gate["passed"] for gate in proof["gates"])


def test_missing_latent_credit_blocks_proof():
    module = load_module()
    proof = module.build_net_effort_reduction_proof(
        field_tool_data(),
        source_review(extra_ids=[]),
        replacement_probe(),
        latent_segment_id="2",
        retained_segment_id="3",
    )

    assert proof["status"] == "not_proven"
    gate = next(gate for gate in proof["gates"] if gate["name"] == "source_route_already_covers_latent_segment")
    assert gate["passed"] is False


def test_missing_replacement_effort_blocks_proof():
    module = load_module()
    proof = module.build_net_effort_reduction_proof(
        field_tool_data(),
        source_review(),
        replacement_probe(include_effort=False),
        latent_segment_id="2",
        retained_segment_id="3",
    )

    assert proof["status"] == "not_proven"
    gate = next(gate for gate in proof["gates"] if gate["name"] == "replacement_probe_has_timing_and_effort")
    assert gate["passed"] is False
