import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "simulated_progress_sweep_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("simulated_progress_sweep_audit", MODULE_PATH)
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


def field_tool_data(routes, field_days=None):
    return {
        "routes": [row["route"] for row in routes],
        "field_day_layer": {"field_days": field_days or []},
        "progress": {"completed_segment_ids_at_export": []},
    }


def route_repeat_audit(routes):
    return {"routes": [row["repeat_row"] for row in routes]}


def official_segments(*ids):
    return [{"seg_id": segment_id, "official_miles": 1.0} for segment_id in ids]


def test_route_sweep_prices_future_route_removed_by_latent_completion():
    module = load_module()
    a = route("a", "A", ["1"], ["1", "2"], 5.0, 50)
    b = route("b", "B", ["2"], ["2"], 3.0, 30)

    audit = module.build_simulated_progress_sweep_audit(
        field_tool_data([a, b]),
        route_repeat_audit([a, b]),
        official_segments("1", "2"),
    )

    a_sweep = [row for row in audit["route_sweeps_ranked"] if row["route_id"] == "a"][0]
    assert a_sweep["segments_newly_credited"] == ["1", "2"]
    assert a_sweep["latent_completed_segment_ids"] == ["2"]
    assert [row["route_key"] for row in a_sweep["future_routes_removed"]] == ["b"]
    assert a_sweep["future_collapse_savings"]["on_foot_miles"] == 3.0
    assert a_sweep["future_collapse_savings"]["door_to_door_minutes_p75"] == 30
    assert a_sweep["net_remaining_menu_saved"]["on_foot_miles"] == 8.0


def test_route_sweep_reports_partial_future_shrink_unpriced():
    module = load_module()
    a = route("a", "A", ["1"], ["1", "2"], 5.0, 50)
    b = route("b", "B", ["2", "3"], ["2", "3"], 6.0, 60)

    audit = module.build_simulated_progress_sweep_audit(
        field_tool_data([a, b]),
        route_repeat_audit([a, b]),
        official_segments("1", "2", "3"),
    )

    a_sweep = [row for row in audit["route_sweeps_ranked"] if row["route_id"] == "a"][0]
    assert a_sweep["future_routes_removed"] == []
    assert [row["route_key"] for row in a_sweep["future_routes_shrunk"]] == ["b"]
    assert a_sweep["future_shrink_unpriced"]["route_count"] == 1
    assert a_sweep["future_shrink_unpriced"]["completed_claimed_official_miles"] == 1.0
    assert a_sweep["future_collapse_savings"]["on_foot_miles"] == 0.0


def test_field_day_sweep_excludes_own_routes_from_future_collapse():
    module = load_module()
    a = route("a", "A", ["1"], ["1", "2"], 5.0, 50)
    b = route("b", "B", ["2"], ["2"], 3.0, 30)
    c = route("c", "C", ["3"], ["3"], 4.0, 40)
    field_days = [
        {
            "field_day_id": "day-1",
            "date": "2026-06-18",
            "on_foot_miles": 9.0,
            "field_day_schedule_p75_minutes": 90,
            "field_day_schedule_p90_minutes": 110,
            "loops": [
                {"label": "A", "route_card_ref": {"outing_id": "a"}},
                {"label": "C", "route_card_ref": {"outing_id": "c"}},
            ],
        }
    ]

    audit = module.build_simulated_progress_sweep_audit(
        field_tool_data([a, b, c], field_days),
        route_repeat_audit([a, b, c]),
        official_segments("1", "2", "3"),
    )

    day_sweep = audit["field_day_sweeps_ranked"][0]
    assert day_sweep["subject_route_keys"] == ["a", "c"]
    assert [row["route_key"] for row in day_sweep["future_routes_removed"]] == ["b"]
    assert day_sweep["future_collapse_savings"]["on_foot_miles"] == 3.0
    assert day_sweep["net_remaining_menu_saved"]["on_foot_miles"] == 12.0
