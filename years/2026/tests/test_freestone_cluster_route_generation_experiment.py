import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "freestone_cluster_route_generation_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("freestone_cluster_route_generation_experiment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_template_segment_order_preserves_template_trail_sequence_and_natural_segment_order():
    module = load_module()
    template = {
        "trail_sequence": ["Mountain Cove", "Shane's Trail", "Central Ridge Trail"],
        "candidate_segment_ids": ["3", "10", "2", "99"],
    }
    official_by_id = {
        "2": {"seg_id": "2", "seg_name": "Mountain Cove 2", "trail_name": "Mountain Cove"},
        "10": {"seg_id": "10", "seg_name": "Mountain Cove 10", "trail_name": "Mountain Cove"},
        "3": {"seg_id": "3", "seg_name": "Shane's Trail 3", "trail_name": "Shane's Trail"},
        "99": {"seg_id": "99", "seg_name": "Unsequenced 99", "trail_name": "Unsequenced"},
    }

    assert module.template_segment_order(template, official_by_id) == ["2", "10", "3", "99"]


def test_generated_route_status_distinguishes_direct_gap_fallback_from_graph_validated_gpx():
    module = load_module()

    assert module.generated_route_status({"passed": False}, 0.0) == "generated_gpx_has_gaps"
    assert module.generated_route_status({"passed": True}, 0.01) == "generated_continuous_gpx_with_direct_gap_fallback"
    assert module.generated_route_status({"passed": True}, 0.0) == "generated_continuous_graph_gpx"


def test_segment_orientation_options_preserve_required_ascent_direction():
    module = load_module()

    assert module.segment_orientation_options({"direction": "ascent"}) == [False]
    assert module.segment_orientation_options({"direction": "both"}) == [False, True]
    assert module.segment_orientation_options({"direction": "ascent"}, preserve_ascent_direction=False) == [False, True]


def test_candidate_summary_keeps_replacement_blockers_and_scaled_time_visible():
    module = load_module()
    candidate = {
        "variant_id": "nearest-segment-greedy",
        "strategy": "nearest_segment_greedy",
        "status": "generated_continuous_gpx_with_direct_gap_fallback",
        "track_miles": 31.38,
        "official_miles": 12.46,
        "connector_miles": 10.29,
        "official_repeat_miles": 1.33,
        "direct_gap_fallback_miles": 0.01,
        "coverage_validation": {"status": "covers_template_segment_set"},
        "gpx_validation": {"passed": True},
        "self_repeat_segment_ids": ["1590"],
        "non_template_repeat_segment_ids": ["1629"],
        "ascent_direction_validation": {"status": "passed_no_ascent_segments", "ascent_segment_ids": []},
        "cue_complexity": {"official_cue_count": 7, "connector_or_return_cue_count": 9},
    }
    current_all = {"on_foot_miles": 39.54, "p75_minutes": 818, "p90_minutes": 919}
    contained_current = {"on_foot_miles": 17.86}
    touched_current = {"on_foot_miles": 21.68}

    summary = module.candidate_summary(
        candidate,
        current_all,
        contained_current,
        touched_current,
        ["1522", "1748"],
    )

    assert summary["p75_minutes_scaled"] == 649
    assert summary["p90_minutes_scaled"] == 729
    assert summary["comparison"]["direct_replace_contained_routes_delta_miles"] == 13.52
    assert summary["comparison"]["candidate_plus_unshrunk_touched_routes_delta_miles"] == 13.52
    assert summary["comparison"]["replacement_readiness"] == "not_direct_replacement_needs_additional_loops_or_shrunk_cards"
    assert summary["promotion_gates"]["continuous_gpx"] == "needs_direct_gap_review"
    assert summary["promotion_gates"]["repeat_credit"] == "needs_repeat_credit_review"
    assert summary["promotion_gates"]["recertification"] == "not_run_against_active_field_packet"
