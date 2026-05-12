import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calendar_reorder_for_latent_credit_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("calendar_reorder_for_latent_credit_experiment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(outing_id, label, segment_ids, on_foot=1.0, p75=30, p90=40, repeat_ids=None):
    return {
        "outing_id": outing_id,
        "label": label,
        "candidate_ids": [label.lower()],
        "segment_ids": segment_ids,
        "on_foot_miles": on_foot,
        "door_to_door_minutes_p75": p75,
        "door_to_door_minutes_p90": p90,
        "validation": {"passed": True},
        "parking": {"has_parking": True},
        "gpx_href": f"gpx/{label}.gpx",
        "wayfinding_cues": [{"official_repeat_segment_ids": repeat_ids or []}],
        "segment_ownership_reconciliation": {
            "declared_owned_elsewhere_segment_ids": repeat_ids or [],
        },
    }


def loop(label, segment_ids, p75, p90, trails=None):
    return {
        "label": label,
        "route_card_ref": {"outing_id": label.lower(), "label": label},
        "segment_ids": segment_ids,
        "trail_names": trails or [label],
        "field_day_schedule_p75_minutes": p75,
        "field_day_schedule_p90_minutes": p90,
        "route_card_door_to_door_p75_minutes": p75,
        "route_card_door_to_door_p90_minutes": p90,
        "on_foot_miles": 1.0,
        "official_miles": 1.0,
    }


def day(date, day_type, loops, p75, p90, bound=300, constraints=None):
    return {
        "date": date,
        "day_type": day_type,
        "field_day_id": f"day-{date}",
        "field_day_schedule_p75_minutes": p75,
        "field_day_schedule_p90_minutes": p90,
        "p90_bound_minutes": bound,
        "constraints": constraints or [],
        "loops": loops,
    }


def pair(source, owner, latent_ids, saved=1.0, p75=30, p90=40):
    return {
        "source_route_key": source.lower(),
        "owner_route_key": owner.lower(),
        "source_route": {"outing_id": source.lower(), "label": source},
        "owner_route": {"outing_id": owner.lower(), "label": owner},
        "latent_segment_ids": latent_ids,
        "remaining_owner_segment_ids": [],
        "proven_saved_on_foot_miles": saved,
        "proven_saved_p75_minutes": p75,
        "proven_saved_p90_minutes": p90,
    }


def test_swap_source_day_before_owner_day_supports_removed_owner():
    module = load_module()
    field_tool_data = {
        "summary": {"segment_count_in_field_menu": 2},
        "routes": [
            route("source", "Source", ["1"], repeat_ids=["2"]),
            route("owner", "Owner", ["2"]),
        ],
        "field_day_layer": {
            "field_days": [
                day("2026-06-18", "weekday", [loop("Owner", ["2"], 60, 70)], 60, 70),
                day("2026-06-24", "weekday", [loop("Source", ["1"], 90, 100)], 90, 100),
            ]
        },
    }
    latent_delta = {"pairwise_full_removals": [pair("Source", "Owner", ["2"], saved=2.5, p75=60, p90=70)]}

    report = module.build_calendar_reorder_experiment(field_tool_data, latent_delta)

    scenario = report["pairwise_scenarios"][0]
    assert report["status"] == "supported_reorders_found"
    assert scenario["scenario_type"] == "swap_source_day_before_owner_day"
    assert scenario["source_target_date"] == "2026-06-18"
    assert scenario["field_day_count_after"] == 1
    assert scenario["route_count_before"] == 2
    assert scenario["route_count_delta"] == -1
    assert scenario["route_count_after"] == 1
    assert scenario["checks"]["coverage"]["status"] == "passed"
    assert scenario["source_latent_credit_support"]["status"] == "physically_cued_as_repeat_and_declared_owned_elsewhere"
    assert report["summary"]["pairwise_savings_are_additive"] is False
    assert report["recommended_non_overlapping_portfolio"]["saved_on_foot_miles"] == 2.5


def test_same_day_owner_removal_floors_schedule_at_remaining_loop_time():
    module = load_module()
    field_tool_data = {
        "summary": {"segment_count_in_field_menu": 3},
        "routes": [
            route("owner", "Owner", ["1"]),
            route("source", "Source", ["2"], repeat_ids=["1"]),
            route("other", "Other", ["3"]),
        ],
        "field_day_layer": {
            "field_days": [
                day(
                    "2026-07-04",
                    "weekend",
                    [loop("Owner", ["1"], 100, 120), loop("Source", ["2"], 180, 210), loop("Other", ["3"], 40, 50)],
                    220,
                    260,
                    bound=360,
                )
            ]
        },
    }
    latent_delta = {"pairwise_full_removals": [pair("Source", "Owner", ["1"], p75=100, p90=120)]}

    report = module.build_calendar_reorder_experiment(field_tool_data, latent_delta)

    scenario = report["pairwise_scenarios"][0]
    assert scenario["scenario_type"] == "same_day_owner_deletion"
    assert scenario["owner_day_after_removal"]["labels"] == ["Source", "Other"]
    assert scenario["source_day_after_reorder"]["labels"] == ["Source", "Other"]
    assert scenario["owner_day_after_removal"]["p75_minutes"] == 180
    assert scenario["owner_day_after_removal"]["p90_minutes"] == 210


def test_lower_hulls_odd_day_swap_blocks_candidate():
    module = load_module()
    field_tool_data = {
        "summary": {"segment_count_in_field_menu": 2},
        "routes": [
            route("source", "Source", ["1"], repeat_ids=["2"]),
            route("owner", "Owner", ["2"]),
        ],
        "field_day_layer": {
            "field_days": [
                day("2026-06-19", "weekday", [loop("Owner", ["2"], 60, 70, trails=["Owner Trail"])], 60, 70),
                day(
                    "2026-06-25",
                    "weekday",
                    [loop("Source", ["1"], 90, 100, trails=["Lower Hulls Gulch Trail"])],
                    90,
                    100,
                    constraints=["lower_hulls_even_day_on_foot"],
                ),
            ]
        },
    }
    latent_delta = {"pairwise_full_removals": [pair("Source", "Owner", ["2"])]}

    report = module.build_calendar_reorder_experiment(field_tool_data, latent_delta)

    scenario = report["pairwise_scenarios"][0]
    assert scenario["status"] == "blocked"
    assert scenario["checks"]["source_day_after_reorder"]["lower_hulls"]["status"] == "blocked_lower_hulls_odd_day"
    assert scenario["checks"]["owner_day_after_removal"]["removed_empty_day"]["status"] == "passed"
