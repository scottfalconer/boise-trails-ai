import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ownership_reassignment_optimization_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ownership_reassignment_optimization_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(outing_id, label, segment_ids, actual_full_ids, on_foot, p75):
    return {
        "route": {
            "outing_id": outing_id,
            "label": label,
            "candidate_ids": [label.lower()],
            "trailhead": f"{label} Trailhead",
            "segment_ids": segment_ids,
            "official_miles": len(segment_ids),
            "on_foot_miles": on_foot,
            "door_to_door_minutes_p75": p75,
            "door_to_door_minutes_p90": p75 + 10,
        },
        "repeat_row": {
            "outing_id": outing_id,
            "label": f"{outing_id}: {label}",
            "actual_full_segment_ids": actual_full_ids,
        },
    }


def field_tool_data(routes):
    return {
        "routes": [row["route"] for row in routes],
        "field_day_layer": {
            "field_days": [
                {
                    "field_day_id": "day-1",
                    "date": "2026-06-18",
                    "loops": [
                        {"label": row["route"]["label"], "route_card_ref": {"outing_id": row["route"]["outing_id"]}}
                        for row in routes
                    ],
                }
            ]
        },
    }


def route_repeat_audit(routes):
    return {"routes": [row["repeat_row"] for row in routes]}


def official_segments(*ids):
    return [{"seg_id": segment_id, "official_miles": 1.0} for segment_id in ids]


def test_ownership_reassignment_removes_route_when_other_route_physically_covers_it():
    module = load_module()
    a = route("a", "A", ["1"], ["1", "2"], 5.0, 50)
    b = route("b", "B", ["2"], ["2"], 3.0, 30)

    audit = module.build_ownership_reassignment_optimization_audit(
        field_tool_data([a, b]),
        route_repeat_audit([a, b]),
        official_segments("1", "2"),
    )

    assert audit["status"] == "ownership_reassignment_reduces_existing_loop_work"
    assert audit["summary"]["order_free_removed_route_count"] == 1
    assert audit["summary"]["current_calendar_skip_ready_removed_route_count"] == 1
    assert audit["summary"]["order_free_saved_on_foot_miles"] == 3.0
    assert audit["summary"]["reassigned_segment_count"] == 1

    component = audit["components_with_order_free_savings"][0]
    assert [row["route"]["outing_id"] for row in component["removed_routes"]] == ["b"]
    segment_two = [row for row in component["segment_assignments"] if row["seg_id"] == "2"][0]
    assert segment_two["current_owner_route_keys"] == ["b"]
    assert segment_two["optimized_owner_route_key"] == "a"
    assert segment_two["assignment_reason"] == "earlier_selected_physical_coverer"


def test_ownership_reassignment_reports_credit_only_shrink_when_route_still_needed():
    module = load_module()
    a = route("a", "A", ["1"], ["1", "2"], 5.0, 50)
    b = route("b", "B", ["2", "3"], ["2", "3"], 5.0, 50)

    audit = module.build_ownership_reassignment_optimization_audit(
        field_tool_data([a, b]),
        route_repeat_audit([a, b]),
        official_segments("1", "2", "3"),
    )

    assert audit["status"] == "ownership_reassignment_changes_credit_only"
    assert audit["summary"]["order_free_removed_route_count"] == 0
    assert audit["summary"]["partial_shrink_route_count"] == 1
    assert audit["summary"]["reassigned_segment_count"] == 1
    assert audit["summary"]["order_free_saved_on_foot_miles"] == 0.0

    component = audit["credit_only_reassignment_components"][0]
    shrink = component["partial_shrink_routes"][0]
    assert shrink["route"]["outing_id"] == "b"
    assert shrink["lost_credit_segment_ids"] == ["2"]
    assert shrink["optimized_credit_segment_ids"] == ["3"]
    assert shrink["replacement_owner_route_keys"] == ["a"]
