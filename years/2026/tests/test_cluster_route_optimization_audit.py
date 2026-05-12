import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cluster_route_optimization_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cluster_route_optimization_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(outing_id, segment_ids, trailhead="Normal Trailhead", on_foot=5.0, p75=50, cues=None):
    return {
        "outing_id": outing_id,
        "label": f"Route {outing_id}",
        "trailhead": trailhead,
        "segment_ids": segment_ids,
        "official_miles": len(segment_ids),
        "on_foot_miles": on_foot,
        "door_to_door_minutes_p75": p75,
        "door_to_door_minutes_p90": p75 + 10,
        "wayfinding_cues": cues or [],
    }


def template_candidate(matched_routes):
    return {
        "templates_ranked": [
            {
                "template_id": "normal-loop",
                "target_area": "Test Cluster",
                "normal_start": "Normal Trailhead",
                "normal_direction": "clockwise",
                "trail_sequence": ["Trail 1", "Trail 2"],
                "candidate_segment_ids": ["1", "2"],
                "candidate_official_miles": 2.0,
                "matched_current_routes": matched_routes,
                "current_route_pressure": {
                    "matched_route_count": len(matched_routes),
                    "matched_current_on_foot_miles": sum(row["on_foot_miles"] for row in matched_routes),
                    "dead_repeat_candidate_miles": 4.0,
                    "future_collapse_on_foot_miles": 1.0,
                    "future_shrink_official_miles_unpriced": 0.5,
                },
                "warnings": [],
            }
        ]
    }


def matched(route_obj, match_type="contained_route_card"):
    return {
        "route_key": route_obj["outing_id"],
        "outing_id": route_obj["outing_id"],
        "label": f"{route_obj['outing_id']}: {route_obj['label']}",
        "trailhead": route_obj["trailhead"],
        "segment_ids": route_obj["segment_ids"],
        "official_miles": route_obj["official_miles"],
        "on_foot_miles": route_obj["on_foot_miles"],
        "door_to_door_minutes_p75": route_obj["door_to_door_minutes_p75"],
        "door_to_door_minutes_p90": route_obj["door_to_door_minutes_p90"],
        "overlap_segment_ids": route_obj["segment_ids"],
        "match_type": match_type,
    }


def test_archetype_mismatch_scores_fragments_unusual_start_and_burden():
    module = load_module()
    a = route("a", ["1"], on_foot=4.0)
    b = route("b", ["2"], trailhead="Odd Roadside", on_foot=5.0)

    audit = module.build_cluster_route_optimization_audit(
        {"routes": [a, b]},
        template_candidate([matched(a), matched(b)]),
        {"routes": []},
        {"routes": []},
        {"route_sweeps_ranked": []},
    )

    row = audit["archetype_mismatch_ranked"][0]
    assert row["template_id"] == "normal-loop"
    assert row["score_components"]["unusual_start_penalty"] == 1.0
    assert row["score_components"]["route_fragments_common_loop_penalty"] == 1.5
    assert row["score_components"]["high_repeat_without_future_savings_penalty"] == 3.0
    assert row["score_components"]["noncredit_burden_vs_public_route_penalty"] == 1.75
    assert row["mismatch_score"] == 7.25


def test_access_corridor_groups_repeated_paid_access_and_shrink_relation():
    module = load_module()
    cue = {
        "seq": 1,
        "cue_type": "start_access",
        "signed_as": ["Access Trail"],
        "target": "Trail 1",
        "route_leg_miles": 0.5,
    }
    a = route("a", ["1"], cues=[cue])
    b = route("b", ["1", "2"], cues=[cue])
    audit = module.build_cluster_route_optimization_audit(
        {"routes": [a, b]},
        template_candidate([matched(a), matched(b)]),
        {"routes": []},
        {"routes": []},
        {
            "route_sweeps_ranked": [
                {
                    "route_id": "a",
                    "subject_route_keys": ["a"],
                    "future_routes_shrunk": [{"route_key": "b", "route": {"label": "b"}}],
                }
            ]
        },
    )

    group = audit["already_paid_access_corridors"][0]
    assert group["route_count"] == 2
    assert group["same_day_bundle_possible"] is True
    assert group["can_one_route_shrink_after_other"] is True
    assert group["total_corridor_leg_miles"] == 1.0


def test_bundle_rows_and_dominance_checks_keep_lower_bound_unpromoted():
    module = load_module()
    a = route("a", ["1", "2"], on_foot=4.0, p75=40)
    b = route("b", ["2"], on_foot=6.0, p75=60)
    audit = module.build_cluster_route_optimization_audit(
        {"routes": [a, b]},
        template_candidate([matched(a), matched(b), matched(route("c", ["3"]), match_type="partial_route_card_overlap")]),
        {"routes": [{"outing_id": "a", "actual_full_segment_ids": ["1", "2"]}]},
        {"routes": []},
        {
            "route_sweeps_ranked": [
                {
                    "route_id": "a",
                    "subject_route_keys": ["a"],
                    "future_routes_removed": [{"route_key": "b", "route": {"label": "b"}}],
                }
            ]
        },
    )

    bundle = audit["cluster_bundle_replacements"][0]
    assert bundle["replacement_type"] == "cluster_bundle"
    assert bundle["replacement_status"] == "needs_additional_loops"
    assert bundle["total_new_on_foot"] is None
    assert bundle["total_new_on_foot_lower_bound"] == 2.0
    assert bundle["uncovered_current_segment_ids"] == ["3"]
    assert any(row["dominance_type"] == "current_route_dominates_current_route" for row in audit["dominance_checks"])
    assert any(row["dominance_type"] == "post_progress_route_removal" for row in audit["dominance_checks"])
    assert any(row["dominance_type"] == "cluster_bundle_lower_bound_candidate" for row in audit["dominance_checks"])
