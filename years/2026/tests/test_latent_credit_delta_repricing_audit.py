import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "latent_credit_delta_repricing_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("latent_credit_delta_repricing_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(outing_id, label, segment_ids, on_foot, p75, p90):
    return {
        "outing_id": outing_id,
        "label": label,
        "candidate_ids": [label.lower()],
        "trailhead": f"{label} Trailhead",
        "segment_ids": segment_ids,
        "official_miles": len(segment_ids),
        "on_foot_miles": on_foot,
        "door_to_door_minutes_p75": p75,
        "door_to_door_minutes_p90": p90,
    }


def field_tool_data():
    routes = [
        route("a", "A", ["1"], 2.0, 30, 35),
        route("b", "B", ["2"], 3.0, 40, 45),
        route("c", "C", ["3", "4"], 5.0, 70, 80),
    ]
    return {
        "routes": routes,
        "field_day_layer": {
            "field_days": [
                {
                    "date": "2026-06-18",
                    "field_day_id": "day-a",
                    "loops": [{"route_card_ref": {"outing_id": "a"}, "label": "A"}],
                },
                {
                    "date": "2026-06-19",
                    "field_day_id": "day-b",
                    "loops": [{"route_card_ref": {"outing_id": "b"}, "label": "B"}],
                },
                {
                    "date": "2026-06-20",
                    "field_day_id": "day-c",
                    "loops": [{"route_card_ref": {"outing_id": "c"}, "label": "C"}],
                },
            ]
        },
    }


def latent_audit():
    return {
        "reconciled_routes": [
            {
                "outing_id": "a",
                "label": "A",
                "segments": [
                    {
                        "seg_id": "2",
                        "trail_name": "Trail 2",
                        "official_miles": 1.0,
                        "status": "reconciled_owned_elsewhere",
                        "claimed_by_other_routes": [{"outing_id": "b", "label": "B"}],
                    },
                    {
                        "seg_id": "3",
                        "trail_name": "Trail 3",
                        "official_miles": 1.0,
                        "status": "reconciled_owned_elsewhere",
                        "claimed_by_other_routes": [{"outing_id": "c", "label": "C"}],
                    },
                ],
            }
        ]
    }


def test_current_calendar_repricing_removes_fully_covered_future_route():
    module = load_module()

    audit = module.build_latent_credit_delta_repricing_audit(field_tool_data(), latent_audit())

    assert audit["status"] == "proved_current_calendar_savings"
    assert audit["summary"]["pairwise_full_removal_relationship_count"] == 1
    assert audit["summary"]["pairwise_partial_shrink_relationship_count"] == 1
    assert audit["summary"]["current_calendar_removed_route_count"] == 1
    assert audit["summary"]["current_calendar_saved_on_foot_miles"] == 3.0
    assert audit["summary"]["current_calendar_saved_p75_minutes"] == 40
    removed = audit["current_calendar_repricing"]["removed_routes"][0]
    assert removed["route"]["outing_id"] == "b"
    assert removed["prior_latent_segment_ids"] == ["2"]


def test_owner_before_source_is_pairwise_only_not_current_calendar_savings():
    module = load_module()
    data = field_tool_data()
    data["field_day_layer"]["field_days"] = list(reversed(data["field_day_layer"]["field_days"]))

    audit = module.build_latent_credit_delta_repricing_audit(data, latent_audit())

    assert audit["status"] == "pairwise_savings_only"
    assert audit["summary"]["pairwise_full_removal_relationship_count"] == 1
    assert audit["summary"]["current_calendar_removed_route_count"] == 0
    full = audit["pairwise_full_removals"][0]
    assert full["future_route_change"] == "removed"
    assert full["schedule_order"]["status"] == "owner_scheduled_before_source"
