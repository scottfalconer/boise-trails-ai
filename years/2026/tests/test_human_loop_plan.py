import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "human_loop_plan.py"


def load_human_loop_plan():
    spec = importlib.util.spec_from_file_location("human_loop_plan", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_status_accepts_split_block_for_nearby_components():
    module = load_human_loop_plan()

    status, reasons = module.package_status(
        {
            "block_name": "Military core",
            "ratio": 1.45,
            "trailhead_count": 2,
            "component_route_count": 2,
            "component_routes_under_1_official_mile": 1,
        }
    )

    assert status == "accepted_split_block"
    assert "split_across_nearby_route_components" in reasons
    assert "tiny_segments_absorbed" in reasons


def test_package_status_marks_high_ratio_or_geography_locked_blocks_as_grinders():
    module = load_human_loop_plan()

    status, reasons = module.package_status(
        {
            "block_name": "Sweet Connie / Shingle / Stack Rock",
            "ratio": 1.7,
            "trailhead_count": 1,
            "component_route_count": 1,
        }
    )

    assert status == "necessary_grinder"
    assert "necessary_grinder_or_geography_locked" in reasons


def test_package_status_promotes_manual_design_hold():
    module = load_human_loop_plan()

    status, reasons = module.package_status(
        {
            "package_number": 16,
            "component_candidate_ids": ["sweet-connie-trail", "stack-rock-connector"],
            "block_name": "Sweet Connie / Shingle / Stack Rock",
            "ratio": 2.7,
            "trailhead_count": 2,
            "component_route_count": 2,
        },
        {
            "manual_design": {
                "areas": [
                    {
                        "package_number": 16,
                        "demote_candidate_ids": ["sweet-connie-trail"],
                    }
                ]
            }
        },
    )

    assert status == "manual_design_area"
    assert "coverage_placeholder_needs_human_route_design" in reasons


def test_package_status_does_not_match_harris_term_inside_harrison():
    module = load_human_loop_plan()

    status, reasons = module.package_status(
        {
            "block_name": "Hillside / Harrison / West Climb frontside",
            "ratio": 1.58,
            "trailhead_count": 2,
            "component_route_count": 2,
        }
    )

    assert status == "accepted_split_block"
    assert "necessary_grinder_or_geography_locked" not in reasons


def test_build_human_plan_reports_no_blockers_when_inputs_are_complete(tmp_path):
    module = load_human_loop_plan()
    route_pass = {
        "summary": {"selected_route_count": 1},
        "routes": [
            {
                "route_number": 1,
                "route_status": "graph_validated",
                "block_name": "Polecat core",
                "trail_names": ["Polecat Loop"],
                "trailhead": "Polecat Trailhead",
                "official_miles": 5.0,
                "on_foot_miles": 7.0,
                "ratio": 1.4,
            }
        ],
    }
    package_pass = {
        "summary": {
            "package_count": 1,
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "total_on_foot_miles": 308.55,
            "planwide_on_foot_to_official_ratio": 1.88,
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Polecat core",
                "ratio": 1.4,
                "trailhead_count": 1,
                "component_route_count": 1,
                "trailheads": ["Polecat Trailhead"],
                "official_miles": 5.0,
                "on_foot_miles": 7.0,
            }
        ],
    }
    package_map = {"map_validation": {"rendered_passed": True}}

    plan = module.build_human_plan(route_pass, package_pass, package_map, tmp_path / "plan-map.html")

    assert plan["summary"]["unresolved_blocker_count"] == 0
    assert plan["summary"]["all_route_components_graph_validated"] is True
    assert plan["summary"]["map_rendered_passed"] is True
    assert plan["summary"]["status_counts"] == {"primary_loop_block": 1}


def test_render_markdown_includes_status_counts_and_block_table(tmp_path):
    module = load_human_loop_plan()
    plan = {
        "summary": {
            "package_count": 1,
            "route_component_count": 1,
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "total_on_foot_miles": 308.55,
            "planwide_on_foot_to_official_ratio": 1.88,
            "status_counts": {"primary_loop_block": 1},
            "all_route_components_graph_validated": True,
            "map_rendered_passed": True,
            "map_html": str(tmp_path / "plan-map.html"),
        },
        "caveats": ["Check current conditions."],
        "packages": [
            {
                "package_number": 1,
                "block_name": "Polecat core",
                "human_plan_status": "primary_loop_block",
                "human_plan_reasons": [],
                "component_route_count": 1,
                "trailhead_count": 1,
                "trailheads": ["Polecat Trailhead"],
                "official_miles": 5.0,
                "on_foot_miles": 7.0,
                "ratio": 1.4,
            }
        ],
        "routes": [
            {
                "route_number": 1,
                "block_name": "Polecat core",
                "trail_names": ["Polecat Loop"],
                "trailhead": "Polecat Trailhead",
                "official_miles": 5.0,
                "on_foot_miles": 7.0,
                "ratio": 1.4,
            }
        ],
    }

    rendered = module.render_markdown(plan)

    assert "2026 Human Loop Plan v1" in rendered
    assert "Primary loop blocks: 1" in rendered
    assert "1 parked start" in rendered
    assert "Polecat core" in rendered


def test_render_markdown_labels_split_blocks_as_parked_starts(tmp_path):
    module = load_human_loop_plan()
    plan = {
        "summary": {
            "package_count": 1,
            "route_component_count": 2,
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "total_on_foot_miles": 308.55,
            "planwide_on_foot_to_official_ratio": 1.88,
            "status_counts": {"accepted_split_block": 1},
            "all_route_components_graph_validated": True,
            "map_rendered_passed": True,
            "map_html": str(tmp_path / "plan-map.html"),
        },
        "caveats": ["Check current conditions."],
        "packages": [
            {
                "package_number": 1,
                "block_name": "Hillside / Harrison / West Climb frontside",
                "human_plan_status": "accepted_split_block",
                "human_plan_reasons": ["split_across_nearby_route_components"],
                "component_route_count": 2,
                "trailhead_count": 2,
                "trailheads": ["West Climb Trailhead", "Harrison Hollow Trailhead"],
                "official_miles": 8.58,
                "on_foot_miles": 13.08,
                "ratio": 1.52,
            }
        ],
        "routes": [],
    }

    rendered = module.render_markdown(plan)

    assert "2 parked starts" in rendered
    assert "Package on-foot miles are totals if you do every listed parked start" in rendered
