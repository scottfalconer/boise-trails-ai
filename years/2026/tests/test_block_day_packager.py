import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "block_day_packager.py"


def load_packager():
    spec = importlib.util.spec_from_file_location("block_day_packager", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_connector_cue_preserves_source_path_endpoints():
    packager = load_packager()

    cue = packager.connector_cue(
        {
            "from_trail": "Kemper",
            "to_trail": "Kemper",
            "distance_miles": 0.25,
            "path_coordinates": [[-116.1, 43.1], [-116.101, 43.101], [-116.102, 43.102]],
        }
    )

    assert cue["path_start"] == [-116.1, 43.1]
    assert cue["path_end"] == [-116.102, 43.102]
    assert cue["path_coordinates"] == [[-116.1, 43.1], [-116.101, 43.101], [-116.102, 43.102]]


def test_build_packages_absorbs_tiny_routes_into_block():
    packager = load_packager()
    route_pass = {
        "summary": {"covered_segment_count": 3},
        "routes": [
            {
                "route_number": 1,
                "candidate_id": "big",
                "block_id": "alpha",
                "trail_names": ["Big"],
                "official_miles": 5.0,
                "on_foot_miles": 7.0,
                "ratio": 1.4,
                "total_minutes": 100,
                "trailhead": "A",
                "segment_ids": [10, 11],
            },
            {
                "route_number": 2,
                "candidate_id": "tiny",
                "block_id": "alpha",
                "trail_names": ["Tiny"],
                "official_miles": 0.4,
                "on_foot_miles": 0.8,
                "ratio": 2.0,
                "total_minutes": 20,
                "trailhead": "A",
                "segment_ids": [12],
            },
        ],
    }
    blocks = {
        "acceptance_criteria": {
            "preferred_max_on_foot_to_official_ratio": 1.6,
            "max_normal_trailheads_per_day": 1,
        },
        "blocks": [{"block_id": "alpha", "name": "Alpha", "status": "candidate_block"}],
    }

    packaged = packager.build_packages(route_pass, blocks)

    assert packaged["summary"]["package_count"] == 1
    assert packaged["summary"]["component_route_count"] == 2
    assert packaged["summary"]["component_routes_under_1_official_mile"] == 1
    assert packaged["packages"][0]["planning_status"] == "same_trailhead_package_after_gpx"
    assert "contains_absorbed_sub1_components" in packaged["packages"][0]["planning_reasons"]
    assert packaged["packages"][0]["components"][0]["segment_ids"] == [10, 11]


def test_build_packages_flags_multiple_trailheads():
    packager = load_packager()
    route_pass = {
        "summary": {"covered_segment_count": 2},
        "routes": [
            {
                "route_number": 1,
                "candidate_id": "one",
                "block_id": "alpha",
                "trail_names": ["One"],
                "official_miles": 3.0,
                "on_foot_miles": 4.0,
                "trailhead": "A",
            },
            {
                "route_number": 2,
                "candidate_id": "two",
                "block_id": "alpha",
                "trail_names": ["Two"],
                "official_miles": 3.0,
                "on_foot_miles": 4.0,
                "trailhead": "B",
            },
        ],
    }
    blocks = {
        "acceptance_criteria": {"max_normal_trailheads_per_day": 1},
        "blocks": [{"block_id": "alpha", "name": "Alpha", "status": "boundary_review"}],
    }

    packaged = packager.build_packages(route_pass, blocks)

    package = packaged["packages"][0]
    assert package["trailhead_count"] == 2
    assert package["planning_status"] == "needs_manual_route_design"
    assert "multiple_trailheads_need_route_design" in package["planning_reasons"]


def test_build_map_data_uses_route_pass_candidate_index_for_combos():
    packager = load_packager()
    package_pass = {
        "summary": {"package_count": 1},
        "packages": [
            {
                "package_number": 1,
                "block_id": "alpha",
                "block_name": "Alpha",
                "component_candidate_ids": ["combo-a-b"],
            }
        ],
    }
    source_route_pass = {
        "routes": [
            {
                "candidate_id": "combo-a-b",
                "official_miles": 2.0,
                "on_foot_miles": 3.0,
                "trailhead": "A",
                "segment_ids": [1],
            }
        ],
        "candidate_index": {
            "combo-a-b": {
                "candidate_id": "combo-a-b",
                "segments": [
                    {
                        "seg_id": 1,
                        "trail_name": "A",
                        "ascent_ft": 120,
                        "descent_ft": 20,
                        "grade_adjusted_miles": 1.12,
                        "estimated_moving_minutes_p75": 18,
                    }
                ],
                "trail_names": ["A"],
                "route_orientation": {"direction": "forward"},
                "direction_validation": {"planned_traversal_direction": {}},
                "return_to_car": {},
                "trailhead": {
                    "name": "A Trailhead",
                    "lat": 0.0,
                    "lon": 0.0,
                    "has_parking": True,
                    "parking_minutes": 8,
                },
            }
        },
    }
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "A",
            "direction": "both",
            "coordinates": [(0.0, 0.0), (0.0001, 0.0001)],
        }
    }

    map_data = packager.build_map_data(
        package_pass,
        source_route_pass,
        {"state_inputs": {"completed_segment_ids": [1]}},
        official_index,
        None,
    )

    assert map_data["feature_collections"]["routes"]["features"]
    assert map_data["feature_collections"]["parking"]["features"]
    assert map_data["progress"]["completed_segment_ids"] == [1]
    assert map_data["route_cues"]["combo-a-b"]["trailhead"]["name"] == "A Trailhead"
    assert map_data["route_cues"]["combo-a-b"]["segments"][0]["direction_cue"] == (
        "Either direction allowed; follow map arrows."
    )
    assert map_data["route_cues"]["combo-a-b"]["segments"][0]["ascent_ft"] == 120
    assert map_data["route_cues"]["combo-a-b"]["segments"][0]["estimated_moving_minutes_p75"] == 18


def test_human_route_name_uses_r2r_starting_trail_system_and_common_name():
    packager = load_packager()

    hulls = packager.human_route_name(
        ["Kestral Trail", "Lower Hull's Gulch Trail", "Red Cliffs"],
        "Hulls Gulch Trailhead",
    )
    hillside = packager.human_route_name(
        ["Who Now Loop Trail", "Harrison Ridge", "Full Sail Trail"],
        "West Climb",
    )
    connector = packager.human_route_name(["Connector"], "MillerGulch Parking Area/Trailhead")
    chukar = packager.human_route_name(["Chukar Butte Trail"], "Chukar Butte private Strava parking anchor")
    veterans = packager.human_route_name(["Veterans"], "Veterans Trailhead")
    polecat = packager.human_route_name(["Polecat Loop"], "Cartwright")

    assert hulls["route_name"] == "Camels Back / Hulls Gulch: Kestrel"
    assert hulls["route_name_source"] == "r2r_starting_trail_subsystem"
    assert hillside["route_name"] == "Hillside to Hollow: Who Now Loop"
    assert connector["route_name"] == "Dry Creek: Connector"
    assert connector["route_name_source"] == "trailhead_fallback"
    assert chukar["route_name"] == "Dry Creek: Chukar Butte"
    assert "Strava" not in chukar["route_name"]
    assert veterans["route_name"] == "Western Foothills: Veterans"
    assert polecat["route_name"] == "Polecat Gulch: Polecat Loop"


def test_build_map_data_adds_car_pass_and_known_water_logistics():
    packager = load_packager()
    west_leg = [(-116.0 - index * 0.001, 43.0) for index in range(13)]
    north_leg = [(-116.0, 43.0 + index * 0.001) for index in range(13)]
    package_pass = {
        "summary": {"package_count": 1},
        "packages": [
            {
                "package_number": 1,
                "block_id": "alpha",
                "block_name": "Alpha",
                "component_candidate_ids": ["combo-car-water"],
            }
        ],
    }
    source_route_pass = {
        "routes": [
            {
                "candidate_id": "combo-car-water",
                "official_miles": 2.0,
                "on_foot_miles": 4.0,
                "trailhead": "A",
                "segment_ids": [1, 2],
            }
        ],
        "candidate_index": {
            "combo-car-water": {
                "candidate_id": "combo-car-water",
                "segments": [
                    {"seg_id": 1, "trail_name": "Outbound"},
                    {"seg_id": 2, "trail_name": "Second Loop"},
                ],
                "trail_names": ["Outbound", "Second Loop"],
                "route_orientation": {"direction": "forward"},
                "direction_validation": {"planned_traversal_direction": {}},
                "between_trail_links": {
                    "links": [
                            {
                                "from_trail": "Outbound",
                                "to_trail": "Second Loop",
                                "distance_miles": 0.5,
                                "path_coordinates": list(reversed(west_leg)),
                                "connector_names": ["Back by car"],
                            }
                        ]
                    },
                    "return_to_car": {
                        "path_coordinates": list(reversed(north_leg)),
                    },
                "trailhead": {
                    "name": "A Trailhead",
                    "lat": 43.0,
                    "lon": -116.0,
                    "has_parking": True,
                    "has_water": True,
                    "water_confidence": "user_verified",
                    "has_restroom": True,
                    "parking_minutes": 8,
                },
            }
        },
    }
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Outbound",
            "direction": "both",
            "coordinates": west_leg,
        },
        2: {
            "seg_id": 2,
            "trail_name": "Second Loop",
            "direction": "both",
            "coordinates": north_leg,
        },
    }

    map_data = packager.build_map_data(package_pass, source_route_pass, {}, official_index, None)

    logistics = map_data["route_cues"]["combo-car-water"]["logistics"]
    assert logistics["car_passes"][0]["name"] == "Pass by car again"
    assert logistics["car_passes"][0]["mile_from_start"] > 0.5
    assert logistics["known_water"][0]["name"] == "A Trailhead"
    assert logistics["known_water"][0]["confidence"] == "user_verified"

    marker_kinds = [
        feature["properties"]["kind"]
        for feature in map_data["feature_collections"]["logistics"]["features"]
    ]
    assert marker_kinds == ["car_pass", "water"]


def test_build_map_data_adds_signpost_labels_to_route_cues():
    packager = load_packager()
    package_pass = {
        "summary": {"package_count": 1},
        "packages": [
            {
                "package_number": 1,
                "block_id": "harrison",
                "block_name": "Harrison",
                "component_candidate_ids": ["combo-who-now-hippie"],
            }
        ],
    }
    source_route_pass = {
        "routes": [
            {
                "candidate_id": "combo-who-now-hippie",
                "official_miles": 1.0,
                "on_foot_miles": 2.0,
                "trailhead": "Harrison Hollow",
                "segment_ids": [1697, 1578],
            }
        ],
        "candidate_index": {
            "combo-who-now-hippie": {
                "candidate_id": "combo-who-now-hippie",
                "segments": [
                    {"seg_id": 1697, "seg_name": "Who Now Loop Trail 1", "trail_name": "Who Now Loop Trail"},
                    {"seg_id": 1578, "seg_name": "Hippie Shake Trail 1", "trail_name": "Hippie Shake Trail"},
                ],
                "trail_names": ["Who Now Loop Trail", "Hippie Shake Trail"],
                "route_orientation": {"direction": "forward"},
                "direction_validation": {"planned_traversal_direction": {}},
                "between_trail_links": {
                    "links": [
                        {
                            "from_trail": "Who Now Loop Trail",
                            "to_trail": "Hippie Shake Trail",
                            "distance_miles": 0.1,
                            "official_repeat_miles": 0.05,
                            "official_repeat_segment_ids": [1697],
                            "connector_names": ["Kemper's Ridge #52", "Who Now Loop #51"],
                        }
                    ]
                },
                "return_to_car": {
                    "official_repeat_miles": 0.2,
                    "official_repeat_segment_ids": [1578],
                },
                "trailhead": {"name": "Harrison Hollow", "lat": 0.0, "lon": 0.0},
            }
        },
    }
    official_index = {
        1697: {
            "seg_id": 1697,
            "seg_name": "Who Now Loop Trail 1",
            "trail_name": "Who Now Loop Trail",
            "direction": "both",
            "coordinates": [(0.0, 0.0), (0.0001, 0.0001)],
        },
        1578: {
            "seg_id": 1578,
            "seg_name": "Hippie Shake Trail 1",
            "trail_name": "Hippie Shake Trail",
            "direction": "both",
            "coordinates": [(0.0001, 0.0001), (0.0002, 0.0002)],
        },
    }

    map_data = packager.build_map_data(package_pass, source_route_pass, {}, official_index, None)
    cue = map_data["route_cues"]["combo-who-now-hippie"]

    assert cue["segments"][0]["signpost_label"] == "#51 Who Now Loop Trail"
    assert cue["segments"][1]["signpost_label"] == "#50 Hippie Shake Trail"
    assert cue["between_links"][0]["signpost_labels"] == ["#52 Kemper's Ridge", "#51 Who Now Loop"]
    assert cue["between_links"][0]["official_repeat_segment_ids"] == [1697]
    assert cue["return_to_car"]["official_repeat_segment_ids"] == [1578]


def test_route_cue_reorients_next_segment_before_stale_inter_link(monkeypatch):
    packager = load_packager()
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Out",
            "direction": "both",
            "coordinates": [(-116.100, 43.100), (-116.101, 43.100)],
        },
        2: {
            "seg_id": 2,
            "trail_name": "Fork",
            "direction": "both",
            "coordinates": [(-116.103, 43.100), (-116.101, 43.100)],
        },
    }
    monkeypatch.setattr(packager, "route_cue_official_index", lambda: official_index)
    candidate = {
        "candidate_id": "stale-inter-link-route",
        "segments": [
            {"seg_id": 1, "trail_name": "Out"},
            {"seg_id": 2, "trail_name": "Fork"},
        ],
        "trail_names": ["Out", "Fork"],
        "inter_segment_links": {
            "links": [
                {
                    "from_segment_id": 1,
                    "to_segment_id": 2,
                    "distance_miles": 0.5,
                    "path_coordinates": [
                        [-116.101, 43.100],
                        [-116.102, 43.101],
                        [-116.103, 43.100],
                    ],
                }
            ]
        },
        "route_orientation": {"direction": "forward"},
        "direction_validation": {"planned_traversal_direction": {}},
        "return_to_car": {},
        "trailhead": {"name": "Trailhead", "lat": 43.100, "lon": -116.100},
    }

    cue = packager.route_cue(
        candidate,
        {"candidate_id": "stale-inter-link-route"},
        connector_graph={"graph": {}, "nodes": [(-116.101, 43.100), (-116.103, 43.100)]},
    )

    assert cue["segments"][1]["start"] == [-116.101, 43.1]
    assert cue["segments"][1]["end"] == [-116.103, 43.1]
    assert "pre_connector_link" not in cue["segments"][1]
    assert cue["between_links"] == []


def test_render_html_includes_direction_arrow_controls():
    packager = load_packager()
    html = packager.render_html(
        {
            "summary": {
                "package_count": 0,
                "covered_segment_count": 0,
                "total_on_foot_miles": 0,
                "planwide_on_foot_to_official_ratio": 0,
            },
            "packages": [],
            "feature_collections": {
                "routes": {"type": "FeatureCollection", "features": []},
                "official_segments": {"type": "FeatureCollection", "features": []},
                "logistics": {"type": "FeatureCollection", "features": []},
            },
        }
    )

    assert "drawDirectionArrows" in html
    assert "drawRouteCues" in html
    assert "Outing Menu Map" in html
    assert "Executable parked-start outings" in html
    assert "path-marker" in html
    assert "one clear cased line" in html
    assert "dir-arrow" in html
    assert "double-backs" in html
    assert "parking-marker" in html
    assert "logistics-marker" in html
    assert "CAR markers" in html
    assert "Known water" in html
    assert "drawLogisticsMarkers" in html
    assert "where to park" in html
    assert "selectedPanel" in html
    assert "mapSummary" in html
    assert "parking-label" in html
    assert "Door to door" in html
    assert "Return to car" in html
    assert "direction_cue" in html
    assert "direction_rule" in html
    assert "routeCuesForOuting" in html
    assert "Related package" in html
    assert "formatMinutes" in html
    assert "parkedStarts" in html
    assert "buildOutings" in html
    assert "outingListHtml" in html
    assert "time-filter" in html
    assert "≤4h" in html
    assert "outingMatchesFilter" in html
    assert "completed_segment_ids" in html
    assert "completed" in html
    assert "Remaining segs" in html
    assert "remainingSegmentCount" in html
    assert "package-meta" in html
    assert "start-row" in html
    assert "manualDesignAreas" in html
    assert "manual_design_hold" in html
    assert "Manual holds" in html


def test_render_outing_menu_markdown_matches_map_cards_and_progress():
    packager = load_packager()
    map_data = {
        "summary": {
            "official_miles": 5.0,
            "total_on_foot_miles": 8.0,
            "planwide_on_foot_to_official_ratio": 1.6,
        },
        "progress": {"completed_segment_ids": [3]},
        "packages": [
            {
                "package_number": 1,
                "block_name": "Alpha Trails",
                "components": [
                    {
                        "candidate_id": "a",
                        "trail_names": ["Alpha"],
                        "official_miles": 2.0,
                        "on_foot_miles": 3.0,
                        "total_minutes": 95,
                        "trailhead": "Alpha Trailhead",
                        "segment_ids": [1, 2],
                    },
                    {
                        "candidate_id": "b",
                        "trail_names": ["Beta"],
                        "official_miles": 1.0,
                        "on_foot_miles": 2.0,
                        "total_minutes": 80,
                        "trailhead": "Beta Trailhead",
                        "segment_ids": [3],
                    },
                ],
            },
            {
                "package_number": 2,
                "block_name": "Gamma Trails",
                "components": [
                    {
                        "candidate_id": "c",
                        "trail_names": ["Gamma"],
                        "official_miles": 2.0,
                        "on_foot_miles": 3.0,
                        "total_minutes": 185,
                        "trailhead": "Gamma Parking/Trailhead",
                        "segment_ids": [4],
                    }
                ],
            },
        ],
    }

    outings = packager.build_outing_menu(map_data)
    markdown = packager.render_outing_menu_markdown(map_data, Path("/tmp/2026-outing-menu-map.html"))

    assert [outing["label"] for outing in outings] == ["1A", "2"]
    assert outings[0]["trailhead"] == "Alpha"
    assert outings[0]["time_bucket"] == "2 hours or less"
    assert outings[1]["time_bucket"] == "3-4 hours"
    assert "Beta" not in markdown
    assert "# 2026 Outing Menu" in markdown
    assert "one executable parked-start outing" in markdown
    assert "Door-to-door" in markdown
    assert "1h 35m" in markdown
    assert "3h 5m" in markdown
    assert "1A" in markdown
    assert "Remaining official segments represented: 3" in markdown
    assert "`/tmp/2026-outing-menu-map.html`" in markdown


def test_render_outing_menu_markdown_computes_missing_planwide_ratio():
    packager = load_packager()
    map_data = {
        "summary": {
            "official_miles": 5.0,
            "total_on_foot_miles": 9.0,
        },
        "progress": {"completed_segment_ids": []},
        "packages": [],
    }

    markdown = packager.render_outing_menu_markdown(map_data)

    assert "Full-plan on-foot/official ratio: 1.8x" in markdown


def test_manual_design_hold_drops_placeholder_from_runnable_menu():
    packager = load_packager()
    map_data = {
        "summary": {
            "official_miles": 5.0,
            "total_on_foot_miles": 12.0,
            "planwide_on_foot_to_official_ratio": 2.4,
        },
        "progress": {"completed_segment_ids": []},
        "manual_design": {
            "areas": [
                {
                    "area_id": "area-16",
                    "package_number": 16,
                    "title": "16A Manual Area",
                    "status": "manual_design_area",
                    "decision": "Hold the placeholder until a human route is designed.",
                    "demote_candidate_ids": ["bad-placeholder"],
                    "current_placeholder": {
                        "label": "16A",
                        "trailhead": "Hawkins",
                        "official_miles": 2.0,
                        "on_foot_miles": 10.0,
                        "door_to_door_minutes": 300,
                        "reason": "Too much access mileage.",
                    },
                    "alternatives": [
                        {
                            "alternative_id": "16A-1",
                            "title": "Lower access design",
                            "status": "manual_gpx_required",
                            "target_official_miles": 2.0,
                            "target_on_foot_miles_range": [4.0, 6.0],
                            "required_segment_ids": [1],
                            "design_notes": ["Build GPX first."],
                        }
                    ],
                    "acceptance_gates": ["No fake connector."],
                }
            ]
        },
        "packages": [
            {
                "package_number": 16,
                "block_name": "Problem Area",
                "components": [
                    {
                        "candidate_id": "bad-placeholder",
                        "trail_names": ["Bad"],
                        "official_miles": 2.0,
                        "on_foot_miles": 10.0,
                        "total_minutes": 300,
                        "trailhead": "Hawkins Trailhead",
                        "segment_ids": [1],
                    },
                    {
                        "candidate_id": "good-stack",
                        "trail_names": ["Stack"],
                        "official_miles": 3.0,
                        "on_foot_miles": 4.0,
                        "total_minutes": 100,
                        "trailhead": "Stack Trailhead",
                        "segment_ids": [2],
                    },
                ],
            }
        ],
    }

    outings = packager.build_outing_menu(map_data)
    markdown = packager.render_outing_menu_markdown(map_data, Path("/tmp/map.html"))

    held = [outing for outing in outings if outing["manual_design_hold"]]
    runnable = [outing for outing in outings if not outing["manual_design_hold"]]
    assert [outing["trailhead"] for outing in held] == ["Hawkins"]
    assert [outing["trailhead"] for outing in runnable] == ["Stack"]
    assert "Manual Design Areas" in markdown
    assert "16A Manual Area" in markdown
    assert "manual_gpx_required" in markdown
    assert "Open runnable outings: 1" in markdown
    assert "Manual design holds: 1" in markdown


def test_build_outing_menu_can_keep_same_parking_components_separate():
    packager = load_packager()
    map_data = {
        "summary": {},
        "progress": {"completed_segment_ids": []},
        "packages": [
            {
                "package_number": 16,
                "block_name": "Manual split",
                "components": [
                    {
                        "candidate_id": "manual-16a-1",
                        "field_menu_group_id": "manual-16a-1",
                        "field_menu_label": "16A-1",
                        "trail_names": ["Sweet Connie Trail"],
                        "official_miles": 6.09,
                        "on_foot_miles": 12.2,
                        "total_minutes": 249,
                        "trailhead": "Dry Creek / Sweet Connie roadside parking",
                        "segment_ids": [1665, 1666, 1667],
                    },
                    {
                        "candidate_id": "manual-16a-2",
                        "field_menu_group_id": "manual-16a-2",
                        "field_menu_label": "16A-2",
                        "trail_names": ["Shingle Creek Trail", "Sheep Camp Trail"],
                        "official_miles": 5.53,
                        "on_foot_miles": 14.96,
                        "total_minutes": 310,
                        "trailhead": "Dry Creek / Sweet Connie roadside parking",
                        "segment_ids": [1656, 1653],
                    },
                ],
            }
        ],
    }

    outings = packager.build_outing_menu(map_data)

    assert [outing["label"] for outing in outings] == ["16A-1", "16A-2"]
    assert [outing["trailhead"] for outing in outings] == [
        "Dry Creek / Sweet Connie roadside parking",
        "Dry Creek / Sweet Connie roadside parking",
    ]
    assert [outing["candidate_ids"] for outing in outings] == [["manual-16a-1"], ["manual-16a-2"]]
