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


def repeat_cue(seq, cue_type, ids, miles, text="", leg_miles=None, route_leg_miles=None):
    cue = {
        "seq": seq,
        "cue_type": cue_type,
        "official_repeat_segment_ids": ids,
        "official_repeat_miles": miles,
        "leg_miles": miles if leg_miles is None else leg_miles,
        "action": "FOLLOW",
        "display_detail": text,
        "note": text,
    }
    if route_leg_miles is not None:
        cue["route_leg_miles"] = route_leg_miles
    return cue


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


def test_avoidable_post_credit_repeat_is_dead_candidate_even_on_return_leg():
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
    route_repeat_audit = {
        "routes": [
            {
                **repeat_row("a"),
                "avoidable_post_credit_repeat_instances": [
                    {
                        "seq": 1,
                        "repeated_segment_ids": ["2"],
                        "estimated_savings_miles": 0.42,
                        "alternate_connector_names": ["Public shortcut"],
                    }
                ],
            }
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
    assert row["repeat_classification"][0]["classification_reason"] == "post-credit repeat has a proven shorter legal connector"


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


def test_dead_repeat_actual_route_miles_uses_connector_leg_miles_not_full_official_pressure():
    module = load_module()
    field_tool_data = {
        "routes": [
            route(
                "a",
                "A",
                [
                    repeat_cue(
                        1,
                        "connector_named_trail",
                        ["2", "3"],
                        2.0,
                        leg_miles=0.25,
                    )
                ],
                on_foot=4.0,
                official=2.0,
            ),
        ]
    }
    route_repeat_audit = {
        "routes": [
            repeat_row("a", warnings=["same_trailhead_bundle_candidate"], repeat_miles=2.0, non_credit=2.0),
        ]
    }
    audit = module.build_repeat_productivity_audit(
        field_tool_data,
        route_repeat_audit,
        {"components": []},
        [
            {"seg_id": "2", "official_miles": 1.0},
            {"seg_id": "3", "official_miles": 1.0},
        ],
    )

    row = audit["routes"][0]
    assert row["dead_repeat_candidate_miles"] == 2.0
    assert row["dead_repeat_actual_route_miles"] == 0.25
    assert audit["summary"]["total_dead_repeat_actual_route_miles"] == 0.25


def test_dead_repeat_actual_route_miles_ignores_gpx_route_leg_when_card_leg_is_shorter():
    module = load_module()
    field_tool_data = {
        "routes": [
            route(
                "a",
                "A",
                [
                    repeat_cue(
                        1,
                        "connector_named_trail",
                        ["2"],
                        1.5,
                        leg_miles=0.4,
                        route_leg_miles=7.5,
                    )
                ],
                on_foot=3.4,
                official=3.0,
            ),
        ]
    }
    route_repeat_audit = {
        "routes": [
            repeat_row("a", warnings=["same_trailhead_bundle_candidate"], repeat_miles=1.5, non_credit=0.4),
        ]
    }
    audit = module.build_repeat_productivity_audit(
        field_tool_data,
        route_repeat_audit,
        {"components": []},
        [{"seg_id": "2", "official_miles": 1.5}],
    )

    row = audit["routes"][0]
    assert row["dead_repeat_candidate_miles"] == 1.5
    assert row["dead_repeat_actual_route_miles"] == 0.4
