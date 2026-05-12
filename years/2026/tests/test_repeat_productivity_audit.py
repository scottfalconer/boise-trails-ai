import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repeat_productivity_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("repeat_productivity_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(outing_id, label, cues, on_foot=3.0, official=1.0):
    return {
        "outing_id": outing_id,
        "label": label,
        "candidate_ids": [label.lower()],
        "trailhead": f"{label} Trailhead",
        "segment_ids": [f"{outing_id}-claim"],
        "official_miles": official,
        "on_foot_miles": on_foot,
        "door_to_door_minutes_p75": 40,
        "door_to_door_minutes_p90": 50,
        "wayfinding_cues": cues,
    }


def repeat_cue(seq, cue_type, ids, miles, text=""):
    return {
        "seq": seq,
        "cue_type": cue_type,
        "official_repeat_segment_ids": ids,
        "official_repeat_miles": miles,
        "action": "FOLLOW",
        "display_detail": text,
        "note": text,
    }


def repeat_row(outing_id, warnings=None, repeat_miles=1.0, non_credit=1.0):
    return {
        "outing_id": outing_id,
        "label": outing_id,
        "declared_repeat_miles": repeat_miles,
        "non_credit_miles": non_credit,
        "optimization_warnings": [{"code": code} for code in (warnings or [])],
    }


def test_productive_repeat_uses_ownership_gained_credit():
    module = load_module()
    field_tool_data = {
        "routes": [
            route("a", "A", [repeat_cue(1, "connector_named_trail", ["2"], 1.0)]),
        ]
    }
    route_repeat_audit = {"routes": [repeat_row("a")]}
    ownership_audit = {
        "components": [
            {
                "component_id": "C01",
                "route_keys": ["a"],
                "reassigned_segment_count": 1,
                "order_free_savings": {"on_foot_miles": 1.0},
                "route_impacts": [
                    {
                        "route_key": "a",
                        "status": "expanded_owner_same_physical_route",
                        "gained_credit_segment_ids": ["2"],
                    }
                ],
            }
        ]
    }

    audit = module.build_repeat_productivity_audit(
        field_tool_data,
        route_repeat_audit,
        ownership_audit,
        [{"seg_id": "2", "official_miles": 1.0}],
    )

    row = audit["routes"][0]
    assert row["productive_repeat_miles"] == 1.0
    assert row["dead_repeat_candidate_miles"] == 0.0
    assert row["productive_repeat_segment_ids"] == ["2"]


def test_return_to_car_repeat_is_necessary_without_alternate_evidence():
    module = load_module()
    field_tool_data = {
        "routes": [
            route(
                "a",
                "A",
                [repeat_cue(1, "exit_access", ["2"], 1.0, "Return leg to parked car")],
            ),
        ]
    }
    audit = module.build_repeat_productivity_audit(
        field_tool_data,
        {"routes": [repeat_row("a")]},
        {"components": []},
        [{"seg_id": "2", "official_miles": 1.0}],
    )

    row = audit["routes"][0]
    assert row["necessary_repeat_miles"] == 1.0
    assert row["dead_repeat_candidate_miles"] == 0.0
    assert row["necessary_repeat_segment_ids"] == ["2"]


def test_connector_repeat_with_same_trailhead_pressure_is_dead_candidate():
    module = load_module()
    field_tool_data = {
        "routes": [
            route("a", "A", [repeat_cue(1, "connector_named_trail", ["2"], 1.0)]),
        ]
    }
    route_repeat_audit = {
        "routes": [
            repeat_row("a", warnings=["same_trailhead_bundle_candidate", "high_declared_repeat_miles"]),
        ]
    }
    audit = module.build_repeat_productivity_audit(
        field_tool_data,
        route_repeat_audit,
        {"components": []},
        [{"seg_id": "2", "official_miles": 1.0}],
    )

    row = audit["routes"][0]
    assert row["dead_repeat_candidate_miles"] == 1.0
    assert row["necessary_repeat_miles"] == 0.0
    assert row["dead_repeat_candidate_segment_ids"] == ["2"]
