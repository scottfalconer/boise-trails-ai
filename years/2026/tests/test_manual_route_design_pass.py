import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "manual_route_design_pass.py"


def load_module():
    spec = importlib.util.spec_from_file_location("manual_route_design_pass", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_probe_summary_preserves_time_and_effort_fields():
    module = load_module()
    candidate = {
        "route_status": "graph_validated",
        "official_new_miles": 2.45,
        "estimated_total_on_foot_miles": 4.45,
        "raw_total_minutes": 111,
        "total_minutes": 134,
        "time_breakdown_minutes": {"drive_to_trailhead": 20, "moving_time": 63},
        "time_estimates_minutes": {"door_to_door_p75": 134, "moving_effort_p75": 74},
        "effort": {"ascent_ft": 286, "grade_adjusted_miles": 2.74, "elevation_source": "dem"},
        "validation": {
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
            "trailhead_snap_confidence": "medium",
        },
        "direction_validation": {"planned_traversal_direction": {}},
        "trailhead": {"name": "Dry Creek Parking Area/Trailhead"},
    }

    summary = module.candidate_probe_summary(candidate)

    assert summary["raw_total_minutes"] == 111
    assert summary["time_breakdown_minutes"] == {"drive_to_trailhead": 20, "moving_time": 63}
    assert summary["time_estimates_minutes"] == {"door_to_door_p75": 134, "moving_effort_p75": 74}
    assert summary["effort"] == {"ascent_ft": 286, "grade_adjusted_miles": 2.74, "elevation_source": "dem"}


def test_manual_probe_uses_dominant_exact_generated_candidate():
    module = load_module()
    manual_design = {
        "areas": [
            {
                "area_id": "package10",
                "anchors": [
                    {
                        "anchor_id": "dry",
                        "name": "Dry Creek Parking Area/Trailhead",
                    }
                ],
                "alternatives": [
                    {
                        "alternative_id": "10B",
                        "start_anchor_id": "dry",
                        "required_segment_ids": [1497, 1538, 1537, 1536],
                    }
                ],
            }
        ]
    }
    manual_candidate = {
        "candidate_id": "manual-10b",
        "route_status": "graph_validated",
        "segment_ids": [1497, 1538, 1537, 1536],
        "estimated_total_on_foot_miles": 5.43,
        "total_minutes": 152,
        "trailhead": {"name": "Dry Creek Parking Area/Trailhead"},
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
        },
    }
    generated_candidate = {
        "candidate_id": "combo-currant-creek-bitterbrush-trail",
        "route_status": "graph_validated",
        "segment_ids": [1497, 1538, 1537, 1536],
        "estimated_total_on_foot_miles": 4.45,
        "total_minutes": 134,
        "trailhead": {"name": "Dry Creek Parking Area/Trailhead"},
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
        },
    }
    probe_candidate_objects = {"10B": manual_candidate}

    replacements = module.replace_manual_probes_with_dominant_generated_candidates(
        manual_design,
        probe_candidate_objects,
        {"candidate_index": {"combo-currant-creek-bitterbrush-trail": generated_candidate}},
    )

    assert probe_candidate_objects["10B"]["candidate_id"] == "combo-currant-creek-bitterbrush-trail"
    assert replacements["10B"]["reason"] == "generated_exact_candidate_dominates_manual_probe"
    assert replacements["10B"]["manual_on_foot_miles"] == 5.43
    assert replacements["10B"]["generated_total_minutes"] == 134


def test_manual_probe_does_not_use_generated_candidate_with_worse_time():
    module = load_module()
    area = {
        "anchors": [{"anchor_id": "dry", "name": "Dry Creek Parking Area/Trailhead"}],
    }
    alternative = {
        "alternative_id": "10B",
        "start_anchor_id": "dry",
        "required_segment_ids": [1, 2],
    }
    manual_candidate = {
        "candidate_id": "manual",
        "estimated_total_on_foot_miles": 5.0,
        "total_minutes": 100,
    }
    generated_candidate = {
        "candidate_id": "generated",
        "route_status": "graph_validated",
        "segment_ids": [1, 2],
        "estimated_total_on_foot_miles": 4.5,
        "total_minutes": 105,
        "trailhead": {"name": "Dry Creek Parking Area/Trailhead"},
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
        },
    }

    replacement = module.find_dominant_exact_generated_candidate(
        area,
        alternative,
        manual_candidate,
        {"candidate_index": {"generated": generated_candidate}},
    )

    assert replacement is None


def test_manual_probe_does_not_use_generated_candidate_without_return_path_geometry():
    module = load_module()
    area = {
        "anchors": [{"anchor_id": "dry", "name": "Dry Creek Parking Area/Trailhead"}],
    }
    alternative = {
        "alternative_id": "10B",
        "start_anchor_id": "dry",
        "required_segment_ids": [1, 2],
    }
    manual_candidate = {
        "candidate_id": "manual",
        "estimated_total_on_foot_miles": 5.0,
        "total_minutes": 100,
    }
    generated_candidate = {
        "candidate_id": "generated-gap",
        "route_status": "graph_validated",
        "segment_ids": [1, 2],
        "estimated_total_on_foot_miles": 4.5,
        "total_minutes": 90,
        "trailhead": {"name": "Dry Creek Parking Area/Trailhead"},
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
        },
        "return_to_car": {"strategy": "mapped_connector_loop", "connector_miles": 0.5},
    }

    replacement = module.find_dominant_exact_generated_candidate(
        area,
        alternative,
        manual_candidate,
        {"candidate_index": {"generated-gap": generated_candidate}},
    )

    assert replacement is None


def test_manual_route_design_report_splits_held_and_kept_components():
    module = load_module()
    package_pass = {
        "packages": [
            {
                "package_number": 16,
                "block_id": "sweet",
                "components": [
                    {
                        "candidate_id": "bad",
                        "trailhead": "Hawkins",
                        "official_miles": 2.0,
                        "on_foot_miles": 10.0,
                        "less_optimal_flags": ["long_access"],
                    },
                    {
                        "candidate_id": "good",
                        "trailhead": "Stack",
                        "official_miles": 3.0,
                        "on_foot_miles": 4.0,
                        "less_optimal_flags": [],
                    },
                ],
            }
        ]
    }
    manual_design = {
        "areas": [
            {
                "area_id": "area",
                "package_number": 16,
                "block_id": "sweet",
                "title": "Area",
                "status": "manual_design_area",
                "decision": "Hold bad.",
                "demote_candidate_ids": ["bad"],
                "keep_candidate_ids": ["good"],
                "alternatives": [
                    {
                        "alternative_id": "A",
                        "title": "Better",
                        "status": "manual_gpx_required",
                        "target_official_miles": 2.0,
                        "target_on_foot_miles_range": [1.5, 2.0],
                        "required_segment_ids": [1],
                    }
                ],
                "acceptance_gates": ["No fake gap."],
            }
        ]
    }

    report = module.build_design_report(package_pass, manual_design)
    rendered = module.render_markdown(report)

    area = report["areas"][0]
    assert area["current_demoted_on_foot_miles"] == 10.0
    assert area["acceptance_target_on_foot_miles"] == 2.0
    assert area["demoted_components"][0]["candidate_id"] == "bad"
    assert area["kept_components"][0]["candidate_id"] == "good"
    assert area["alternatives"][0]["beats_current_placeholder_by_min_improvement_if"] is True
    assert "Held Placeholder Components" in rendered
    assert "Runnable/Kept Components" in rendered
