import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "human_loop_plan.py"


def load_human_loop_plan():
    spec = importlib.util.spec_from_file_location("human_loop_plan", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_field_menu_overrides_prefers_generated_multi_start_replacements(tmp_path, monkeypatch):
    module = load_human_loop_plan()
    public_overrides = tmp_path / "public-overrides.json"
    generated_replacements = tmp_path / "generated-replacements.private.json"
    public_overrides.write_text("{}", encoding="utf-8")
    generated_replacements.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "DEFAULT_FIELD_MENU_OVERRIDES_JSON", public_overrides)
    monkeypatch.setattr(module, "DEFAULT_GENERATED_MULTI_START_FIELD_MENU_REPLACEMENTS_JSON", generated_replacements)

    assert module.default_field_menu_overrides_json() == generated_replacements

    generated_replacements.unlink()
    assert module.default_field_menu_overrides_json() == public_overrides


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


def test_promote_package16_manual_routes_replaces_hawkins_placeholders(tmp_path):
    module = load_human_loop_plan()
    accepted_geojson = tmp_path / "accepted.geojson"
    accepted_geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"kind": "route", "alternative_id": "16A-1", "title": "Sweet Connie"},
                        "geometry": {"type": "LineString", "coordinates": [[-116.18, 43.69], [-116.17, 43.7]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "parking", "alternative_id": "16A-1", "name": "Dry/Sweet"},
                        "geometry": {"type": "Point", "coordinates": [-116.18, 43.69]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "route", "alternative_id": "16A-2", "title": "Shingle Sheep"},
                        "geometry": {"type": "LineString", "coordinates": [[-116.18, 43.69], [-116.16, 43.71]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "parking", "alternative_id": "16A-2", "name": "Dry/Sweet"},
                        "geometry": {"type": "Point", "coordinates": [-116.18, 43.69]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    route_pass = {
        "summary": {"total_on_foot_miles": 40.87},
        "routes": [
            {
                "candidate_id": "sweet-connie-trail",
                "segment_ids": [1665, 1666, 1667],
                "trailhead": "Hawkins Range Reserve Trailhead",
            },
            {
                "candidate_id": "shingle-creek-trail-sheep-camp-trail",
                "segment_ids": [1656, 1653],
                "trailhead": "Hawkins Range Reserve Trailhead",
            },
        ],
    }
    package_pass = {
        "summary": {"total_on_foot_miles": 40.87, "official_miles": 15.12, "manual_route_design_package_count": 1},
        "packages": [
            {
                "package_number": 16,
                "block_id": "sweet_connie_shingle_sheep_stack",
                "block_name": "Sweet Connie / Shingle / Sheep Camp / Stack Rock",
                "components": [
                    {
                        "candidate_id": "sweet-connie-trail",
                        "trail_names": ["Sweet Connie Trail"],
                        "official_miles": 6.09,
                        "on_foot_miles": 16.86,
                        "total_minutes": 348,
                        "trailhead": "Hawkins Range Reserve Trailhead",
                        "segment_ids": [1665, 1666, 1667],
                    },
                    {
                        "candidate_id": "shingle-creek-trail-sheep-camp-trail",
                        "trail_names": ["Shingle Creek Trail", "Sheep Camp Trail"],
                        "official_miles": 5.53,
                        "on_foot_miles": 19.62,
                        "total_minutes": 401,
                        "trailhead": "Hawkins Range Reserve Trailhead",
                        "segment_ids": [1656, 1653],
                    },
                    {
                        "candidate_id": "stack-rock-connector",
                        "trail_names": ["Stack Rock Connector"],
                        "official_miles": 3.5,
                        "on_foot_miles": 4.39,
                        "total_minutes": 131,
                        "trailhead": "Freddy's Stack Rock Trailhead",
                        "segment_ids": [1664, 1663],
                    },
                ],
            }
        ],
    }
    package_map = {
        "summary": dict(package_pass["summary"]),
        "packages": json.loads(json.dumps(package_pass["packages"])),
        "route_cues": {
            "sweet-connie-trail": {
                "candidate_id": "sweet-connie-trail",
                "segments": [
                    {"seg_id": 1665, "trail_name": "Sweet Connie Trail"},
                    {"seg_id": 1666, "trail_name": "Sweet Connie Trail"},
                    {"seg_id": 1667, "trail_name": "Sweet Connie Trail"},
                    {"seg_id": 9999, "trail_name": "Unrelated Old Cue Trail"},
                ],
                "between_links": [
                    {"from_trail": "Sweet Connie Trail", "to_trail": "Sweet Connie Trail"},
                    {"from_trail": "Unrelated Old Cue Trail", "to_trail": "Sweet Connie Trail"},
                ],
                "return_to_car": {"connector_names": ["old connector"], "connector_classes": ["old_class"]},
            },
            "shingle-creek-trail-sheep-camp-trail": {
                "candidate_id": "shingle-creek-trail-sheep-camp-trail",
                "segments": [
                    {"seg_id": 1656, "trail_name": "Shingle Creek Trail"},
                    {"seg_id": 1653, "trail_name": "Sheep Camp Trail"},
                ],
                "return_to_car": {},
            },
        },
        "feature_collections": {
            "routes": {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"candidate_id": "sweet-connie-trail"}, "geometry": {}},
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "shingle-creek-trail-sheep-camp-trail"},
                        "geometry": {},
                    },
                ],
            },
            "parking": {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"candidate_id": "sweet-connie-trail"}, "geometry": {}},
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "shingle-creek-trail-sheep-camp-trail"},
                        "geometry": {},
                    },
                ],
            },
        },
        "map_validation": {
            "source_gap_warning_count": 2,
            "route_validations": [
                {"candidate_id": "sweet-connie-trail", "source_gap_warning": True},
                {"candidate_id": "shingle-creek-trail-sheep-camp-trail", "source_gap_warning": True},
            ],
        },
    }
    manual_design = {
        "areas": [
            {
                "area_id": "package16",
                "package_number": 16,
                "demote_candidate_ids": ["sweet-connie-trail", "shingle-creek-trail-sheep-camp-trail"],
                "keep_candidate_ids": ["stack-rock-connector"],
                "anchors": [
                    {
                        "anchor_id": "lower",
                        "name": "Dry Creek / Sweet Connie roadside parking",
                        "lat": 43.69,
                        "lon": -116.18,
                        "has_parking": True,
                        "parking_confidence": "source_verified_roadside",
                        "field_ready": False,
                    }
                ],
            }
        ]
    }
    manual_report = {
        "areas": [
            {
                "area_id": "package16",
                "current_demoted_on_foot_miles": 36.48,
                "current_good_route": {"alternative_ids": ["16A-1", "16A-2"], "on_foot_miles": 27.16},
                "alternatives": [
                    {
                        "alternative_id": "16A-1",
                        "route_design_status": "gpx_generated_parking_manual",
                        "start_anchor_id": "lower",
                        "required_segment_ids": [1665, 1666, 1667],
                        "target_official_miles": 6.09,
                        "probe": {
                            "route_status": "graph_validated",
                            "official_miles": 6.09,
                            "on_foot_miles": 12.2,
                            "raw_total_minutes": 194,
                            "total_minutes": 249,
                            "time_breakdown_minutes": {"drive_to_trailhead": 13, "moving_time": 160},
                            "time_estimates_minutes": {
                                "door_to_door_p75": 249,
                                "moving_effort_p75": 215,
                                "route_finding_penalty": 0,
                            },
                            "effort": {
                                "ascent_ft": 3191,
                                "descent_ft": 1540,
                                "grade_adjusted_miles": 9.28,
                                "elevation_source": "dem",
                            },
                            "trailhead": "Dry Creek / Sweet Connie roadside parking",
                            "trailhead_snap_confidence": "high",
                        },
                        "generated_route_artifact": {"track_validation": {"passed": True}},
                    },
                    {
                        "alternative_id": "16A-2",
                        "route_design_status": "gpx_generated_parking_manual",
                        "start_anchor_id": "lower",
                        "required_segment_ids": [1656, 1653],
                        "target_official_miles": 5.53,
                        "probe": {
                            "route_status": "graph_validated",
                            "official_miles": 5.53,
                            "on_foot_miles": 14.96,
                            "raw_total_minutes": 242,
                            "total_minutes": 310,
                            "time_breakdown_minutes": {"drive_to_trailhead": 13, "moving_time": 208},
                            "time_estimates_minutes": {
                                "door_to_door_p75": 310,
                                "moving_effort_p75": 264,
                                "route_finding_penalty": 12,
                            },
                            "effort": {
                                "ascent_ft": 2712,
                                "descent_ft": 1307,
                                "grade_adjusted_miles": 8.24,
                                "elevation_source": "dem",
                            },
                            "trailhead": "Dry Creek / Sweet Connie roadside parking",
                            "trailhead_snap_confidence": "medium",
                        },
                        "generated_route_artifact": {"track_validation": {"passed": True}},
                    },
                ],
            }
        ],
        "generated_route_artifacts": {"geojson_path": str(accepted_geojson)},
    }

    module.promote_package16_manual_routes(route_pass, package_pass, package_map, manual_design, manual_report)

    package = package_map["packages"][0]
    assert package["planning_status"] == "accepted_manual_split_parking_manual"
    assert package["component_candidate_ids"] == ["manual-16a-1", "manual-16a-2", "stack-rock-connector"]
    assert package["on_foot_miles"] == 31.55
    assert package_map["summary"]["manual_route_design_package_count"] == 0
    assert "sweet-connie-trail" not in package_map["route_cues"]
    assert package_map["route_cues"]["manual-16a-1"]["trailhead"]["name"] == "Dry Creek / Sweet Connie roadside parking"
    assert package_map["route_cues"]["manual-16a-1"]["raw_total_minutes"] == 194
    assert package_map["route_cues"]["manual-16a-1"]["time_estimates_minutes"]["door_to_door_p75"] == 249
    assert package_map["route_cues"]["manual-16a-1"]["time_estimates_minutes"]["moving_effort_p75"] == 215
    assert package_map["route_cues"]["manual-16a-1"]["effort"]["ascent_ft"] == 3191
    assert package["components"][0]["time_estimates_minutes"]["door_to_door_p75"] == 249
    assert package["components"][0]["effort"]["grade_adjusted_miles"] == 9.28
    assert [segment["seg_id"] for segment in package_map["route_cues"]["manual-16a-1"]["segments"]] == [
        1665,
        1666,
        1667,
    ]
    assert package_map["route_cues"]["manual-16a-1"]["between_links"] == [
        {"from_trail": "Sweet Connie Trail", "to_trail": "Sweet Connie Trail"}
    ]
    assert package_map["route_cues"]["manual-16a-1"]["return_to_car"]["connector_names"] == []
    assert {
        feature["properties"]["candidate_id"]
        for feature in package_map["feature_collections"]["routes"]["features"]
    } == {"manual-16a-1", "manual-16a-2"}
    assert {
        validation["candidate_id"]
        for validation in package_map["map_validation"]["route_validations"]
    } == {"manual-16a-1", "manual-16a-2"}
    assert route_pass["routes"][0]["candidate_id"] == "manual-16a-1"


def test_promote_manual_routes_skips_partial_replacement_that_would_drop_segments(tmp_path):
    module = load_human_loop_plan()
    accepted_geojson = tmp_path / "accepted.geojson"
    accepted_geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"kind": "route", "alternative_id": "A"},
                        "geometry": {"type": "LineString", "coordinates": [[-116.18, 43.69], [-116.17, 43.7]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "parking", "alternative_id": "A"},
                        "geometry": {"type": "Point", "coordinates": [-116.18, 43.69]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    route_pass = {"routes": [{"candidate_id": "original", "segment_ids": [1, 2]}]}
    package_pass = {
        "packages": [
            {
                "package_number": 10,
                "components": [
                    {
                        "candidate_id": "original",
                        "trail_names": ["Original"],
                        "official_miles": 2.0,
                        "on_foot_miles": 4.0,
                        "segment_ids": [1, 2],
                    }
                ],
            }
        ]
    }
    package_map = {
        "packages": json.loads(json.dumps(package_pass["packages"])),
        "route_cues": {"original": {"candidate_id": "original", "segments": [{"seg_id": 1}, {"seg_id": 2}]}},
        "feature_collections": {"routes": {"features": []}, "parking": {"features": []}},
        "map_validation": {"route_validations": []},
    }
    manual_design = {
        "areas": [
            {
                "area_id": "partial",
                "package_number": 10,
                "demote_candidate_ids": ["original"],
                "anchors": [{"anchor_id": "start", "name": "Start"}],
            }
        ]
    }
    manual_report = {
        "areas": [
            {
                "area_id": "partial",
                "current_good_route": {"alternative_ids": ["A", "B"], "on_foot_miles": 3.0},
                "alternatives": [
                    {
                        "alternative_id": "A",
                        "route_design_status": "gpx_generated_parking_manual",
                        "start_anchor_id": "start",
                        "required_segment_ids": [1],
                        "target_official_miles": 1.0,
                        "probe": {
                            "route_status": "graph_validated",
                            "official_miles": 1.0,
                            "on_foot_miles": 1.5,
                            "total_minutes": 30,
                        },
                        "generated_route_artifact": {"track_validation": {"passed": True}},
                    },
                    {
                        "alternative_id": "B",
                        "route_design_status": "gpx_generated_parking_manual",
                        "start_anchor_id": "start",
                        "required_segment_ids": [2],
                        "target_official_miles": 1.0,
                        "probe": {"route_status": "manual_review"},
                        "generated_route_artifact": {"track_validation": {"passed": False}},
                    },
                ],
            }
        ],
        "generated_route_artifacts": {"geojson_path": str(accepted_geojson)},
    }

    module.promote_package16_manual_routes(route_pass, package_pass, package_map, manual_design, manual_report)

    package = package_map["packages"][0]
    assert package["components"][0]["candidate_id"] == "original"
    assert package["components"][0]["segment_ids"] == [1, 2]
    assert "original" in package_map["route_cues"]
    assert package_map["manual_promotion_skips"][0]["missing_segment_ids"] == ["2"]


def test_promote_manual_routes_allows_missing_segment_when_covered_elsewhere(tmp_path):
    module = load_human_loop_plan()
    accepted_geojson = tmp_path / "accepted.geojson"
    accepted_geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"kind": "route", "alternative_id": "A"},
                        "geometry": {"type": "LineString", "coordinates": [[-116.18, 43.69], [-116.17, 43.7]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "parking", "alternative_id": "A"},
                        "geometry": {"type": "Point", "coordinates": [-116.18, 43.69]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    route_pass = {"routes": [{"candidate_id": "original", "segment_ids": [1, 2]}]}
    package_pass = {
        "packages": [
            {
                "package_number": 10,
                "components": [
                    {
                        "candidate_id": "original",
                        "trail_names": ["Original"],
                        "official_miles": 2.0,
                        "on_foot_miles": 4.0,
                        "segment_ids": [1, 2],
                    }
                ],
            }
        ]
    }
    package_map = {
        "summary": {"total_on_foot_miles": 4.0, "official_miles": 2.0, "manual_route_design_package_count": 1},
        "packages": json.loads(json.dumps(package_pass["packages"])),
        "route_cues": {"original": {"candidate_id": "original", "segments": [{"seg_id": 1}, {"seg_id": 2}]}},
        "feature_collections": {"routes": {"features": []}, "parking": {"features": []}},
        "map_validation": {"route_validations": []},
    }
    manual_design = {
        "areas": [
            {
                "area_id": "partial",
                "package_number": 10,
                "demote_candidate_ids": ["original"],
                "covered_elsewhere_segment_ids": [2],
                "anchors": [{"anchor_id": "start", "name": "Start"}],
            }
        ]
    }
    manual_report = {
        "areas": [
            {
                "area_id": "partial",
                "current_good_route": {"alternative_ids": ["A"], "on_foot_miles": 1.5},
                "alternatives": [
                    {
                        "alternative_id": "A",
                        "route_design_status": "gpx_generated_parking_manual",
                        "start_anchor_id": "start",
                        "required_segment_ids": [1],
                        "target_official_miles": 1.0,
                        "probe": {
                            "route_status": "graph_validated",
                            "official_miles": 1.0,
                            "on_foot_miles": 1.5,
                            "total_minutes": 30,
                        },
                        "generated_route_artifact": {"track_validation": {"passed": True}},
                    }
                ],
            }
        ],
        "generated_route_artifacts": {"geojson_path": str(accepted_geojson)},
    }

    module.promote_package16_manual_routes(route_pass, package_pass, package_map, manual_design, manual_report)

    package = package_map["packages"][0]
    assert package["component_candidate_ids"] == ["manual-a"]
    assert package["components"][0]["segment_ids"] == [1]
    assert "original" not in package_map["route_cues"]
    assert package_map.get("manual_promotion_skips", []) == []


def test_apply_manual_design_reports_allows_later_complete_report_to_promote(tmp_path):
    module = load_human_loop_plan()
    accepted_geojson = tmp_path / "accepted.geojson"
    accepted_geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"kind": "route", "alternative_id": "A"},
                        "geometry": {"type": "LineString", "coordinates": [[-116.18, 43.69], [-116.17, 43.7]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "parking", "alternative_id": "A"},
                        "geometry": {"type": "Point", "coordinates": [-116.18, 43.69]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "route", "alternative_id": "B"},
                        "geometry": {"type": "LineString", "coordinates": [[-116.18, 43.69], [-116.16, 43.71]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"kind": "parking", "alternative_id": "B"},
                        "geometry": {"type": "Point", "coordinates": [-116.18, 43.69]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    route_pass = {"routes": [{"candidate_id": "original", "segment_ids": [1, 2]}]}
    package_pass = {
        "packages": [
            {
                "package_number": 10,
                "components": [
                    {
                        "candidate_id": "original",
                        "trail_names": ["Original"],
                        "official_miles": 2.0,
                        "on_foot_miles": 4.0,
                        "segment_ids": [1, 2],
                    }
                ],
            }
        ]
    }
    package_map = {
        "packages": json.loads(json.dumps(package_pass["packages"])),
        "route_cues": {"original": {"candidate_id": "original", "segments": [{"seg_id": 1}, {"seg_id": 2}]}},
        "feature_collections": {"routes": {"features": []}, "parking": {"features": []}},
        "map_validation": {"route_validations": []},
    }
    manual_design = {
        "areas": [
            {
                "area_id": "area",
                "package_number": 10,
                "demote_candidate_ids": ["original"],
                "anchors": [{"anchor_id": "start", "name": "Start"}],
                "alternatives": [
                    {"alternative_id": "A", "required_segment_ids": [1], "start_anchor_id": "start"},
                    {"alternative_id": "B", "required_segment_ids": [2], "start_anchor_id": "start"},
                ],
            }
        ]
    }
    stale_report = {
        "areas": [
            {
                "area_id": "area",
                "current_good_route": {"alternative_ids": ["A", "B"], "on_foot_miles": 3.0},
                "alternatives": [
                    {
                        "alternative_id": "A",
                        "route_design_status": "gpx_generated_parking_manual",
                        "required_segment_ids": [1],
                        "probe": {"route_status": "graph_validated", "official_miles": 1.0, "on_foot_miles": 1.5},
                        "generated_route_artifact": {"track_validation": {"passed": True}},
                    },
                    {
                        "alternative_id": "B",
                        "route_design_status": "gpx_generated_parking_manual",
                        "required_segment_ids": [2],
                        "probe": {"route_status": "manual_review"},
                        "generated_route_artifact": {"track_validation": {"passed": False}},
                    },
                ],
            }
        ],
        "generated_route_artifacts": {"geojson_path": str(accepted_geojson)},
    }
    fresh_report = json.loads(json.dumps(stale_report))
    fresh_report["areas"][0]["alternatives"][1]["probe"] = {
        "route_status": "graph_validated",
        "official_miles": 1.0,
        "on_foot_miles": 1.5,
    }
    fresh_report["areas"][0]["alternatives"][1]["generated_route_artifact"] = {"track_validation": {"passed": True}}

    module.apply_manual_design_reports(
        route_pass,
        package_pass,
        package_map,
        manual_design,
        [stale_report, fresh_report],
    )

    package = package_map["packages"][0]
    assert package["component_candidate_ids"] == ["manual-a", "manual-b"]
    assert {component["candidate_id"] for component in package["components"]} == {"manual-a", "manual-b"}
    assert "original" not in package_map["route_cues"]
    assert package_map["manual_promotion_skips"] == []


def test_apply_field_menu_overrides_replaces_collapsed_package():
    module = load_human_loop_plan()
    collapsed_package = {
        "package_number": 1,
        "block_name": "Hillside / Harrison / West Climb frontside",
        "component_candidate_ids": ["block-hillside_harrison_frontside"],
        "official_miles": 8.59,
        "on_foot_miles": 13.66,
        "components": [
            {
                "candidate_id": "block-hillside_harrison_frontside",
                "trail_names": ["Everything"],
                "official_miles": 8.59,
                "on_foot_miles": 13.66,
                "total_minutes": 299,
                "trailhead": "Harrison Hollow Trailhead",
                "segment_ids": [1, 2],
            }
        ],
    }
    replacement_package = {
        "package_number": 1,
        "block_name": "Hillside / Harrison / West Climb frontside",
        "component_candidate_ids": ["west", "harrison"],
        "official_miles": 8.58,
        "on_foot_miles": 13.08,
        "components": [
            {
                "candidate_id": "west",
                "trail_names": ["West Climb"],
                "official_miles": 3.86,
                "on_foot_miles": 7.39,
                "ratio": 1.91,
                "total_minutes": 128,
                "trailhead": "West Climb Trailhead",
                "segment_ids": [1],
            },
            {
                "candidate_id": "harrison",
                "trail_names": ["Harrison Hollow"],
                "official_miles": 4.72,
                "on_foot_miles": 5.69,
                "ratio": 1.21,
                "total_minutes": 96,
                "trailhead": "Harrison Hollow Trailhead",
                "segment_ids": [2],
            },
        ],
    }
    route_pass = {
        "summary": {"total_on_foot_miles": 13.66, "planwide_on_foot_to_official_ratio": 1.59},
        "routes": [{"candidate_id": "block-hillside_harrison_frontside", "on_foot_miles": 13.66}],
    }
    package_pass = {"summary": {"total_on_foot_miles": 13.66, "official_miles": 8.59}, "packages": [collapsed_package]}
    package_map = {
        "summary": {"total_on_foot_miles": 13.66, "official_miles": 8.59},
        "packages": [json.loads(json.dumps(collapsed_package))],
        "route_cues": {"block-hillside_harrison_frontside": {}, "other": {}},
        "feature_collections": {
            "routes": {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"candidate_id": "block-hillside_harrison_frontside"}},
                    {"type": "Feature", "properties": {"candidate_id": "other"}},
                ],
            }
        },
        "map_validation": {
            "source_gap_warning_count": 1,
            "route_validations": [
                {"candidate_id": "block-hillside_harrison_frontside", "source_gap_warning": True},
                {"candidate_id": "other", "source_gap_warning": False},
            ],
        },
    }
    overrides = {
        "overrides": [
            {
                "package_number": 1,
                "remove_candidate_ids": ["block-hillside_harrison_frontside"],
                "replace_package": replacement_package,
                "route_cues": {"west": {"candidate_id": "west"}, "harrison": {"candidate_id": "harrison"}},
                "route_validations": [
                    {"candidate_id": "west", "source_gap_warning": False},
                    {"candidate_id": "harrison", "source_gap_warning": False},
                ],
                "feature_collections": {
                    "routes": {
                        "type": "FeatureCollection",
                        "features": [
                            {"type": "Feature", "properties": {"candidate_id": "west"}},
                            {"type": "Feature", "properties": {"candidate_id": "harrison"}},
                        ],
                    }
                },
            }
        ]
    }

    module.apply_field_menu_overrides(route_pass, package_pass, package_map, overrides)

    assert package_map["packages"][0]["component_candidate_ids"] == ["west", "harrison"]
    assert package_map["summary"]["total_on_foot_miles"] == 13.08
    assert "block-hillside_harrison_frontside" not in package_map["route_cues"]
    assert {"west", "harrison"} <= set(package_map["route_cues"])
    assert {
        feature["properties"]["candidate_id"]
        for feature in package_map["feature_collections"]["routes"]["features"]
    } == {"other", "west", "harrison"}
    assert {
        validation["candidate_id"]
        for validation in package_map["map_validation"]["route_validations"]
    } == {"other", "west", "harrison"}
    assert [route["candidate_id"] for route in route_pass["routes"]] == ["west", "harrison"]


def test_apply_field_menu_overrides_skips_absent_package_without_orphan_features():
    module = load_human_loop_plan()
    route_pass = {"summary": {}, "routes": [{"candidate_id": "active"}]}
    package_pass = {"summary": {}, "packages": [{"package_number": 1, "components": []}]}
    package_map = {
        "summary": {},
        "packages": [{"package_number": 1, "components": []}],
        "route_cues": {"active": {"candidate_id": "active"}},
        "feature_collections": {
            "routes": {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {"candidate_id": "active"}}],
            }
        },
        "map_validation": {"route_validations": [{"candidate_id": "active"}]},
    }
    overrides = {
        "overrides": [
            {
                "package_number": 112,
                "remove_candidate_ids": ["stale"],
                "replace_package": {
                    "package_number": 112,
                    "component_candidate_ids": ["stale-replacement"],
                    "components": [{"candidate_id": "stale-replacement"}],
                },
                "route_cues": {"stale-replacement": {"candidate_id": "stale-replacement"}},
                "route_validations": [{"candidate_id": "stale-replacement"}],
                "feature_collections": {
                    "routes": {
                        "type": "FeatureCollection",
                        "features": [
                            {"type": "Feature", "properties": {"candidate_id": "stale-replacement"}}
                        ],
                    }
                },
            }
        ]
    }

    module.apply_field_menu_overrides(route_pass, package_pass, package_map, overrides)

    assert package_map["packages"] == [{"package_number": 1, "components": []}]
    assert package_pass["packages"] == [{"package_number": 1, "components": []}]
    assert package_map["route_cues"] == {"active": {"candidate_id": "active"}}
    assert package_map["feature_collections"]["routes"]["features"] == [
        {"type": "Feature", "properties": {"candidate_id": "active"}}
    ]
    assert package_map["map_validation"]["route_validations"] == [{"candidate_id": "active"}]
    assert route_pass["routes"] == [{"candidate_id": "active"}]


def test_sync_progress_from_state_replaces_stale_map_progress():
    module = load_human_loop_plan()
    package_map = {"progress": {"completed_segment_ids": [999], "blocked_segment_ids": [998]}}
    state = {
        "completed_segment_ids": ["2", 1, "2"],
        "blocked_segment_ids": ["5"],
        "blocked_trail_names": ["Closed Trail"],
    }

    module.sync_progress_from_state(package_map, state)

    assert package_map["progress"] == {
        "completed_segment_ids": [1, 2],
        "blocked_segment_ids": [5],
        "blocked_trail_names": ["Closed Trail"],
    }


def test_recompute_package_summary_uses_actual_components_after_replacements():
    module = load_human_loop_plan()
    package_map = {
        "summary": {
            "covered_segment_count": 1,
            "official_miles": 99.0,
            "total_on_foot_miles": 99.0,
            "manual_route_design_package_count": 3,
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Frontside",
                "components": [
                    {
                        "candidate_id": "a",
                        "trail_names": ["A"],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                        "trailhead": "Trailhead A",
                        "segment_ids": [101, 102],
                    },
                    {
                        "candidate_id": "b",
                        "trail_names": ["B"],
                        "official_miles": 0.5,
                        "on_foot_miles": 1.0,
                        "trailhead": "Trailhead B",
                        "segment_ids": [103],
                    },
                ],
            }
        ],
    }

    summary = module.recompute_package_summary(package_map, package_map)

    assert summary["package_count"] == 1
    assert summary["component_route_count"] == 2
    assert summary["covered_segment_count"] == 3
    assert summary["official_miles"] == 1.5
    assert summary["total_on_foot_miles"] == 2.5
    assert summary["planwide_on_foot_to_official_ratio"] == 1.67
    assert summary["packages_with_multiple_trailheads"] == 1
    assert summary["component_routes_under_1_official_mile"] == 1
    assert summary["component_routes_under_2_official_miles"] == 2
    assert summary["manual_route_design_package_count"] == 0


def test_sync_official_segment_features_rebuilds_layer_from_active_components():
    module = load_human_loop_plan()
    package_map = {
        "packages": [
            {
                "package_number": 1,
                "block_name": "Frontside",
                "components": [
                    {
                        "candidate_id": "active",
                        "trail_names": ["Active Trail"],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                        "trailhead": "Trailhead",
                        "segment_ids": [101, 102],
                    }
                ],
            }
        ],
        "feature_collections": {
            "official_segments": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                        "properties": {"seg_id": 101, "candidate_id": "stale", "color": "#111111"},
                    }
                ],
            }
        },
    }
    official_segments = [
        {
            "seg_id": 101,
            "seg_name": "Active Trail 1",
            "trail_name": "Active Trail",
            "official_miles": 0.4,
            "direction": "both",
            "coordinates": [[0, 0], [1, 1]],
        },
        {
            "seg_id": 102,
            "seg_name": "Active Trail 2",
            "trail_name": "Active Trail",
            "official_miles": 0.6,
            "direction": "ascent",
            "coordinates": [[1, 1], [2, 2]],
        },
    ]

    module.sync_official_segment_features(package_map, official_segments)

    features = package_map["feature_collections"]["official_segments"]["features"]
    assert [feature["properties"]["seg_id"] for feature in features] == [101, 102]
    assert features[0]["properties"]["candidate_id"] == "active"
    assert features[0]["properties"]["color"] == "#111111"
    assert features[0]["properties"]["official_miles"] == 0.4
    assert features[0]["properties"]["route_official_miles"] == 1.0
    assert features[1]["geometry"]["coordinates"] == [[1, 1], [2, 2]]
    assert features[1]["properties"]["direction_cue"] == "Ascent-only official segment; follow the planned uphill direction."


def test_build_human_plan_reports_package_component_count_when_route_pass_is_stale(tmp_path):
    module = load_human_loop_plan()
    route_pass = {
        "summary": {"selected_route_count": 99},
        "routes": [{"route_status": "graph_validated"}],
    }
    package_pass = {
        "summary": {
            "package_count": 1,
            "component_route_count": 2,
            "covered_segment_count": 251,
            "official_miles": 1.5,
            "total_on_foot_miles": 2.5,
            "planwide_on_foot_to_official_ratio": 1.67,
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Frontside",
                "ratio": 1.67,
                "trailhead_count": 2,
                "component_route_count": 2,
            }
        ],
    }
    package_map = {"map_validation": {"rendered_passed": True}}

    plan = module.build_human_plan(route_pass, package_pass, package_map, tmp_path / "plan-map.html")

    assert plan["summary"]["route_component_count"] == 2


def test_apply_time_calibrations_updates_components_routes_and_cues():
    module = load_human_loop_plan()
    route_pass = {
        "routes": [
            {"candidate_id": "harrison", "total_minutes": 96},
            {"candidate_id": "other", "total_minutes": 30},
        ]
    }
    package_pass = {
        "packages": [
            {
                "package_number": 1,
                "total_minutes_components": 96,
                "components": [
                    {"candidate_id": "harrison", "total_minutes": 96},
                ],
            }
        ]
    }
    package_map = {
        "packages": [
            {
                "package_number": 1,
                "total_minutes_components": 96,
                "components": [
                    {"candidate_id": "harrison", "total_minutes": 96},
                ],
            }
        ],
        "route_cues": {
            "harrison": {
                "candidate_id": "harrison",
                "total_minutes": 96,
                "time_estimates_minutes": {"door_to_door_p75": 96},
            }
        },
    }
    calibrations = {
        "calibrations": [
            {
                "candidate_id": "harrison",
                "total_minutes": 141,
                "raw_total_minutes": 96,
                "time_estimates_minutes": {
                    "door_to_door_p75": 141,
                    "moving_effort_p75": 105,
                    "route_finding_penalty": 18,
                },
                "effort": {
                    "ascent_ft": 1572,
                    "grade_adjusted_miles": 6.29,
                    "elevation_source": "dem",
                },
            }
        ]
    }

    module.apply_time_calibrations(route_pass, package_pass, package_map, calibrations)

    assert route_pass["routes"][0]["total_minutes"] == 141
    assert package_pass["packages"][0]["components"][0]["total_minutes"] == 141
    assert package_pass["packages"][0]["total_minutes_components"] == 141
    assert package_map["packages"][0]["components"][0]["time_estimates_minutes"]["door_to_door_p75"] == 141
    assert package_map["packages"][0]["components"][0]["effort"]["ascent_ft"] == 1572
    assert package_map["route_cues"]["harrison"]["total_minutes"] == 141
    assert package_map["route_cues"]["harrison"]["time_estimates_minutes"]["moving_effort_p75"] == 105
    assert package_map["route_cues"]["harrison"]["effort"]["grade_adjusted_miles"] == 6.29
