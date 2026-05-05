import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_block_planner.py"


def load_planner():
    spec = importlib.util.spec_from_file_location("route_block_planner", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_block_first_plan_assigns_segments_and_flags_fragments():
    planner = load_planner()
    config = {
        "version": "test-blocks",
        "acceptance_criteria": {
            "min_official_miles_unless_geography_locked": 1.0,
            "max_normal_trailheads_per_day": 1,
        },
        "manual_review_priorities": [{"priority": 1, "area": "Boundary", "decision": "Review"}],
        "blocks": [
            {
                "block_id": "alpha",
                "name": "Alpha",
                "status": "candidate_block",
                "trail_names": ["Alpha Trail"],
            },
            {
                "block_id": "beta",
                "name": "Beta",
                "status": "boundary_review",
                "trail_names": ["Beta Trail"],
            },
        ],
    }
    official_trails = {
        "alpha trail": {
            "trail_name": "Alpha Trail",
            "segment_ids": [1, 2],
            "official_miles": 1.5,
            "direction_counts": {"both": 2},
        },
        "beta trail": {
            "trail_name": "Beta Trail",
            "segment_ids": [3],
            "official_miles": 0.8,
            "direction_counts": {"ascent": 1},
        },
    }
    runbook = {
        "profile_name": "fallback",
        "summary": {"scheduled_segments": 3, "scheduled_official_miles": 2.3},
        "audit": {"execution_validation_passed": True},
        "days": [
            {
                "date": "2026-06-18",
                "trailheads": ["A", "B"],
                "outings": [
                    {
                        "outing_id": "alpha-small",
                        "trail_names": ["Alpha Trail"],
                        "new_official_miles": 0.7,
                        "estimated_total_on_foot_miles": 1.4,
                        "trailhead": "A",
                        "route_label": "mop_up",
                    },
                    {
                        "outing_id": "cross",
                        "trail_names": ["Alpha Trail", "Beta Trail"],
                        "new_official_miles": 1.6,
                        "estimated_total_on_foot_miles": 2.2,
                        "trailhead": "B",
                        "route_label": "B-route",
                    },
                ],
            }
        ],
    }

    plan = planner.build_block_first_plan(config, runbook, official_trails)

    assert plan["summary"]["official_segment_assignment_passed"] is True
    assert plan["summary"]["cross_block_current_outing_count"] == 1
    alpha = next(block for block in plan["blocks"] if block["block_id"] == "alpha")
    beta = next(block for block in plan["blocks"] if block["block_id"] == "beta")
    assert alpha["current_standalone_sub1_count"] == 1
    assert alpha["block_readiness"] == "draft_needs_route_design"
    assert "current_plan_has_sub1_fragment" in alpha["readiness_reasons"]
    assert beta["block_readiness"] == "draft_needs_route_design"
    assert "boundary_review_required" in beta["readiness_reasons"]


def test_load_official_trails_derives_trail_names_from_segment_names(tmp_path):
    planner = load_planner()
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"properties": {"segId": 1, "segName": "Example Trail 1", "LengthFt": 5280, "direction": "both"}},
            {"properties": {"segId": 2, "segName": "Example Trail 2", "LengthFt": 2640, "direction": "ascent"}},
        ],
    }
    path = tmp_path / "official.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    trails = planner.load_official_trails(path)

    assert list(trails) == ["example trail"]
    assert trails["example trail"]["trail_name"] == "Example Trail"
    assert trails["example trail"]["segment_ids"] == [1, 2]
    assert trails["example trail"]["official_miles"] == 1.5
    assert trails["example trail"]["direction_counts"] == {"both": 1, "ascent": 1}


def test_render_markdown_includes_assignment_and_absorption_sections():
    planner = load_planner()
    plan = {
        "summary": {
            "official_segments_assigned_to_blocks": 2,
            "official_segments_total": 2,
            "official_segment_assignment_passed": True,
            "unassigned_official_trail_count": 0,
            "duplicate_configured_trail_count": 0,
            "cross_block_current_outing_count": 0,
        },
        "fallback_plan": {"summary": {"scheduled_segments": 2, "scheduled_official_miles": 3.0}},
        "fragmentation_summary": {
            "executable_outings": 2,
            "outings_under_1_official_mile": 1,
            "outings_under_2_official_miles": 1,
            "days_with_3_plus_outings": 0,
            "days_with_3_plus_trailheads": 0,
            "unique_trailheads": 1,
        },
        "manual_review_priorities": [{"priority": 1, "area": "A", "decision": "B"}],
        "blocks": [
            {
                "block_id": "alpha",
                "name": "Alpha",
                "block_readiness": "draft_needs_route_design",
                "official_miles": 3.0,
                "official_segment_count": 2,
                "current_outing_count": 2,
                "current_trailhead_count": 1,
                "current_standalone_sub2_count": 1,
                "cross_block_current_outing_count": 0,
                "readiness_reasons": ["current_plan_has_sub2_fragment"],
                "absorption_candidates": [
                    {
                        "date": "2026-06-18",
                        "outing_id": "small",
                        "trail_names": ["Small"],
                        "official_miles": 0.5,
                        "on_foot_miles": 1.0,
                    }
                ],
            }
        ],
        "cross_block_current_outings": [],
        "config_validation": {"unassigned_official_trails": []},
        "next_steps": ["Do the thing"],
    }

    rendered = planner.render_markdown(plan)

    assert "Assignment QA" in rendered
    assert "Absorption Candidates" in rendered
    assert "`small`" in rendered
    assert "Do the thing" in rendered
