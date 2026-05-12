import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "common_route_template_candidate_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("common_route_template_candidate_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_segment(seg_id, miles=1.0, direction="both"):
    return {
        "seg_id": str(seg_id),
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "official_miles": miles,
        "direction": direction,
    }


def route(outing_id, segment_ids, on_foot=5.0, p75=50):
    return {
        "outing_id": outing_id,
        "label": f"Route {outing_id}",
        "trailhead": "Trailhead",
        "segment_ids": segment_ids,
        "official_miles": len(segment_ids),
        "on_foot_miles": on_foot,
        "door_to_door_minutes_p75": p75,
        "door_to_door_minutes_p90": p75 + 10,
    }


def template(template_id="normal-loop", candidate_segments=None, sources=None):
    return {
        "template_id": template_id,
        "target_area": "Test",
        "normal_start": "Trailhead",
        "normal_direction": "Clockwise",
        "trail_sequence": ["Trail 1", "Trail 2"],
        "public_sources": sources
        if sources is not None
        else [{"source_type": "public_strava_route", "label": "Public route", "url": "https://example.com/route"}],
        "candidate_segments": candidate_segments if candidate_segments is not None else ["1", "2"],
        "connector_hints": ["Use the known loop."],
        "route_experiment_goal": "Generate a candidate.",
    }


def test_template_generates_candidate_and_route_pressure():
    module = load_module()
    audit = module.build_common_route_template_candidate_audit(
        {"templates": [template()]},
        {"routes": [route("a", ["1"], on_foot=4.0, p75=40), route("b", ["2", "3"], on_foot=6.0, p75=60)]},
        [official_segment("1", 0.5), official_segment("2", 0.7), official_segment("3", 1.2)],
        repeat_productivity_audit={
            "routes": [
                {"route_key": "a", "dead_repeat_candidate_miles": 1.5},
                {"route_key": "b", "dead_repeat_candidate_miles": 0.5},
            ]
        },
        simulated_progress_audit={
            "route_sweeps_ranked": [
                {
                    "route_id": "a",
                    "subject_route_keys": ["a"],
                    "priority_score": {
                        "future_collapse_on_foot_miles": 3.0,
                        "future_collapse_p75_minutes": 30,
                        "future_shrink_official_miles_unpriced": 0.4,
                    },
                }
            ]
        },
    )

    row = audit["templates_ranked"][0]
    assert audit["status"] == "template_candidates_generated"
    assert row["status"] == "ready_for_route_experiment"
    assert row["candidate_official_miles"] == 1.2
    assert row["current_route_pressure"]["matched_route_count"] == 2
    assert row["current_route_pressure"]["dead_repeat_candidate_miles"] == 2.0
    assert row["current_route_pressure"]["future_collapse_on_foot_miles"] == 3.0
    assert row["generator_candidate"]["official_segment_ids"] == ["1", "2"]


def test_invalid_candidate_segment_fails_audit():
    module = load_module()
    audit = module.build_common_route_template_candidate_audit(
        {"templates": [template(candidate_segments=["1", "missing"])]},
        {"routes": [route("a", ["1"])]},
        [official_segment("1")],
    )

    row = audit["templates_ranked"][0]
    assert audit["status"] == "failed_invalid_template_segments"
    assert row["status"] == "needs_segment_mapping"
    assert row["invalid_segment_ids"] == ["missing"]


def test_cluster_seed_without_public_source_is_explicitly_marked():
    module = load_module()
    audit = module.build_common_route_template_candidate_audit(
        {
            "templates": [
                template(
                    sources=[
                        {
                            "source_type": "current_field_packet_cluster",
                            "label": "Current field packet cluster",
                            "path": "docs/field-packet/field-tool-data.json",
                        }
                    ]
                )
            ]
        },
        {"routes": [route("a", ["1", "2"])]},
        [official_segment("1"), official_segment("2")],
    )

    row = audit["templates_ranked"][0]
    assert audit["status"] == "template_candidates_generated"
    assert row["status"] == "cluster_seed_needs_public_source"
    assert row["has_public_route_source"] is False
    assert "needs_public_route_source_capture" in row["warnings"]
