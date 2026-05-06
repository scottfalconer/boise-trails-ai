import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "personal_route_planner.py"


def load_planner():
    spec = importlib.util.spec_from_file_location("personal_route_planner", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_geojson(path, features):
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "lastUpdatedUTC": "2026-05-04T00:00:00Z",
                "features": features,
            },
            indent=2,
        )
    )


def feature(seg_id, name, length_ft, lon, lat, direction="both"):
    return {
        "type": "Feature",
        "properties": {
            "segId": seg_id,
            "segName": name,
            "LengthFt": length_ft,
            "direction": direction,
            "specInst": "",
            "activity_type": "both",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat], [lon + 0.001, lat + 0.001]],
        },
    }


def base_state(**overrides):
    state = {
        "completed_segment_ids": [],
        "blocked_segment_ids": [],
        "blocked_trail_names": [],
        "pace_min_per_mile": 15,
        "parking_minutes": 8,
        "drive_model": {
            "origin_label": "North End planning origin",
            "origin_lat": 43.626,
            "origin_lon": -116.205,
            "straight_line_factor": 1.2,
            "minutes_per_mile": 2.0,
            "minimum_one_way_minutes": 4,
        },
        "outing_model": {
            "connector_miles_per_official_mile": 0.1,
            "repeat_miles_per_official_mile": 0.05,
            "minimum_connector_miles": 0.0,
            "road_miles_per_outing": 0.0,
            "connector_return_factor": 1.25,
            "prefer_connector_if_shorter_than_repeat": True,
        },
        "trailheads": [
            {
                "name": "Test Trailhead",
                "lat": 43.626,
                "lon": -116.205,
                "parking_minutes": 8,
            }
        ],
    }
    state.update(overrides)
    return state


def test_completed_and_blocked_segments_are_removed_from_remaining_plan(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            feature(1, "Alpha Trail 1", 5280, -116.205, 43.626),
            feature(2, "Alpha Trail 2", 5280, -116.204, 43.627),
            feature(3, "Bravo Trail 1", 2640, -116.203, 43.628, direction="ascent"),
            feature(4, "Closed Trail 1", 5280, -116.202, 43.629),
        ],
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(completed_segment_ids=[2], blocked_segment_ids=[4]),
        generated_at="2026-05-04T12:00:00Z",
    )

    assert plan["summary"]["official_segments"] == 4
    assert plan["summary"]["completed_segments"] == 1
    assert plan["summary"]["blocked_segments"] == 1
    assert plan["summary"]["remaining_available_segments"] == 2
    assert plan["summary"]["remaining_available_official_miles"] == 1.5
    assert plan["coverage_validation"]["valid"] is True

    remaining_ids = {
        seg_id
        for trail in plan["remaining_trails"]
        for seg_id in trail["remaining_segment_ids"]
    }
    assert remaining_ids == {1, 3}
    assert all(4 not in candidate["segment_ids"] for candidate in plan["route_menu"]["all_candidates"])


def test_candidates_include_logistics_time_and_requested_buckets(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            feature(10, "Quick Trail 1", 5280, -116.205, 43.626),
            feature(20, "Medium Trail 1", 21120, -116.205, 43.626),
        ],
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
    )

    quick = next(candidate for candidate in plan["route_menu"]["all_candidates"] if candidate["trail_names"] == ["Quick Trail"])
    assert quick["time_breakdown_minutes"] == {
        "drive_to_trailhead": 4,
        "parking_and_prep": 8,
        "trailhead_access": 0,
        "moving_time": 30,
        "return_drive": 4,
    }
    assert quick["raw_total_minutes"] == 46
    assert quick["time_estimates_minutes"]["door_to_door_p75"] == 50
    assert quick["total_minutes"] == 50
    assert quick["time_bucket"] == "under_1_hour"
    assert quick["official_new_miles"] == 1.0
    assert quick["estimated_total_on_foot_miles"] == 2.0
    assert quick["return_to_car"]["strategy"] == "out_and_back"
    assert quick["return_to_car"]["official_repeat_miles"] == 1.0
    assert quick["return_to_car"]["needs_map_validation"] is False

    buckets = plan["route_menu"]["buckets"]
    assert "under_1_hour" in buckets
    assert "two_to_three_hours" in buckets
    assert "four_plus_hours" in buckets


def test_candidate_preserves_selected_trailhead_parking_metadata(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(official, [feature(10, "Parking Metadata Trail 1", 5280, -116.205, 43.626)])

    state = base_state(
        trailheads=[
            {
                "name": "Configured Trailhead",
                "lat": 43.626,
                "lon": -116.205,
                "parking_minutes": 9,
                "has_parking": True,
                "facility_status": "Open",
                "parking_confidence": "user_configured_trailhead",
                "source": "personal_planner_state",
            }
        ]
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=state,
        generated_at="2026-05-04T12:00:00Z",
    )

    trailhead = plan["route_menu"]["all_candidates"][0]["trailhead"]
    assert trailhead["name"] == "Configured Trailhead"
    assert trailhead["has_parking"] is True
    assert trailhead["facility_status"] == "Open"
    assert trailhead["parking_confidence"] == "user_configured_trailhead"
    assert trailhead["source"] == "personal_planner_state"


def test_replacing_official_map_changes_remaining_plan_without_state_change(tmp_path):
    planner = load_planner()
    first = tmp_path / "official-first.geojson"
    second = tmp_path / "official-second.geojson"
    write_geojson(first, [feature(1, "Alpha Trail 1", 5280, -116.205, 43.626)])
    write_geojson(
        second,
        [
            feature(1, "Alpha Trail 1", 5280, -116.205, 43.626),
            feature(2, "Map Add Trail 1", 10560, -116.205, 43.626),
        ],
    )

    state = base_state(completed_segment_ids=[1])
    first_plan = planner.build_plan(first, state, generated_at="2026-05-04T12:00:00Z")
    second_plan = planner.build_plan(second, state, generated_at="2026-05-04T12:00:00Z")

    assert first_plan["summary"]["remaining_available_segments"] == 0
    assert first_plan["route_menu"]["all_candidates"] == []
    assert second_plan["summary"]["remaining_available_segments"] == 1
    assert second_plan["remaining_trails"][0]["trail_name"] == "Map Add Trail"


def test_unknown_completed_segment_is_reported_in_coverage_validation(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(official, [feature(1, "Alpha Trail 1", 5280, -116.205, 43.626)])

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(completed_segment_ids=[999]),
        generated_at="2026-05-04T12:00:00Z",
    )

    assert plan["coverage_validation"]["valid"] is False
    assert plan["coverage_validation"]["unknown_completed_segment_ids"] == [999]


def test_open_routes_can_fall_back_to_out_and_back_when_return_gap_is_too_large(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            {
                "type": "Feature",
                "properties": {
                    "segId": 1,
                    "segName": "Long Ridge 1",
                    "LengthFt": 5280,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.205, 43.626], [-116.155, 43.676]],
                },
            }
        ],
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    assert candidate["return_to_car"]["strategy"] == "out_and_back"
    assert candidate["return_to_car"]["official_repeat_miles"] == 1.0
    assert candidate["return_to_car"]["connector_miles"] == 0
    assert candidate["estimated_total_on_foot_miles"] == 2.0


def test_unmapped_short_connector_estimate_uses_out_and_back_for_field_validity(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            {
                "type": "Feature",
                "properties": {
                    "segId": 1,
                    "segName": "No Guess Trail 1",
                    "LengthFt": 5280,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.205, 43.626], [-116.202, 43.626]],
                },
            }
        ],
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    assert candidate["return_to_car"]["strategy"] == "out_and_back"
    assert candidate["return_to_car"]["graph_validated"] is True
    assert candidate["return_to_car"]["needs_map_validation"] is False
    assert candidate["return_to_car"]["path_coordinates"]
    assert "return_connector_needs_map_validation" not in candidate["less_optimal_flags"]
    assert "requires_official_repeat_to_get_back_to_car" in candidate["less_optimal_flags"]


def test_strava_segment_efforts_override_generic_pace_when_names_overlap(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    strava_dir = tmp_path / "activity_details"
    strava_dir.mkdir()
    write_geojson(official, [feature(42, "Gold Finch 1", 5280, -116.205, 43.626)])
    (strava_dir / "activity.json").write_text(
        json.dumps(
            {
                "id": 123,
                "name": "Prior Gold Finch run",
                "start_date_local": "2025-07-01T07:00:00Z",
                "segment_efforts": [
                    {
                        "name": "Gold Finch",
                        "distance": 1609.344,
                        "moving_time": 600,
                        "elapsed_time": 650,
                        "segment": {"id": 999, "name": "gold finch", "distance": 1609.344},
                    }
                ],
            }
        )
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(pace_min_per_mile=16),
        generated_at="2026-05-04T12:00:00Z",
        strava_activity_details_dir=strava_dir,
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    segment = candidate["segments"][0]
    assert segment["estimated_moving_minutes"] == 10
    assert segment["time_source"]["source_type"] == "matched_strava_segment_effort"
    assert segment["time_source"]["matched_name"] == "gold finch"
    assert plan["performance_profile"]["matched_strava_segment_count"] == 1


def test_time_buckets_put_most_efficient_options_first(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            feature(1, "Efficient Loop 1", 2640, -116.205, 43.626),
            feature(2, "Efficient Loop 2", 2640, -116.204, 43.627),
            {
                "type": "Feature",
                "properties": {
                    "segId": 3,
                    "segName": "Fallback Outback 1",
                        "LengthFt": 528,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.205, 43.626], [-116.155, 43.676]],
                },
            },
        ],
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
    )

    bucket = plan["route_menu"]["buckets"]["under_1_hour"]
    assert bucket[0]["trail_names"] == ["Efficient Loop"]
    assert bucket[0]["optimality_rank_in_bucket"] == 1
    assert bucket[0]["efficiency_score"] > bucket[1]["efficiency_score"]
    assert bucket[1]["trail_names"] == ["Fallback Outback"]


def test_long_time_buckets_include_nearby_trailhead_bundles(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]
    write_geojson(
        official,
        [
            feature(index, f"Bundle {name} 1", 10560, -116.205 + index * 0.0001, 43.626)
            for index, name in enumerate(names, start=1)
        ],
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
    )

    four_plus = plan["route_menu"]["buckets"]["four_plus_hours"]
    assert four_plus
    assert four_plus[0]["candidate_type"] == "trailhead_bundle"
    assert len(four_plus[0]["trail_names"]) > 1
    assert four_plus[0]["official_new_miles"] >= 8


def test_activity_geometry_matching_uses_track_shape_not_segment_names(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            feature(1, "Renamed Official Segment 1", 5280, -116.205, 43.626),
            feature(2, "Other Official Segment 1", 5280, -116.25, 43.626),
        ],
    )
    segments, _ = planner.load_official_segments(official)
    activity_coords = [
        (-116.205, 43.626),
        (-116.2045, 43.6265),
        (-116.204, 43.627),
    ]

    matches = planner.match_activity_geometry_to_segments(
        activity_coords,
        segments,
        threshold_miles=0.03,
        min_fraction=0.6,
    )

    assert [match["seg_id"] for match in matches] == [1]
    assert matches[0]["match_source"] == "activity_geometry"


def test_bundle_connector_uses_mapped_graph_path_between_trails(tmp_path):
    planner = load_planner()
    first = feature(1, "First Trail 1", 10560, -116.205, 43.626)
    second = feature(2, "Second Trail 1", 10560, -116.155, 43.626)
    official = tmp_path / "official.geojson"
    write_geojson(official, [first, second])
    connector = tmp_path / "connector.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"TrailName": "Mapped Connector"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.204, 43.627],
                                [-116.19, 43.635],
                                [-116.155, 43.626],
                            ],
                        },
                    }
                ],
            }
        )
    )

    segments, _ = planner.load_official_segments(official)
    trails = planner.group_remaining_by_trail(segments)
    graph = planner.load_connector_graph(connector, official_segments=segments)
    links = planner.build_between_trail_links(
        trails,
        planner.get_outing_model(base_state()),
        graph,
    )

    assert links["links"][0]["source"] == "mapped_graph"
    assert links["links"][0]["path_coordinates"][0] == [-116.204, 43.627]
    assert links["links"][0]["path_coordinates"][-1] == [-116.155, 43.626]
    assert links["connector_miles"] > 0
    assert links["official_repeat_miles"] == 0


def test_multiline_connector_parts_do_not_create_artificial_graph_edge(tmp_path):
    planner = load_planner()
    connector = tmp_path / "multipart.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"TrailName": "Multipart Connector"},
                        "geometry": {
                            "type": "MultiLineString",
                            "coordinates": [
                                [[-116.205, 43.626], [-116.204, 43.626]],
                                [[-116.155, 43.626], [-116.154, 43.626]],
                            ],
                        },
                    }
                ],
            }
        )
    )
    graph = planner.load_connector_graph(connector)

    path = planner.shortest_connector_path(
        (-116.205, 43.626),
        (-116.154, 43.626),
        graph,
        0.01,
    )

    assert path is None


def test_shortest_connector_path_reports_connector_edge_classes(tmp_path):
    planner = load_planner()
    connector = tmp_path / "road.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "TrailName": "Public Road",
                            "source": "openstreetmap",
                            "highway": "primary",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-116.205, 43.626], [-116.204, 43.626]],
                        },
                    }
                ],
            }
        )
    )
    graph = planner.load_connector_graph(connector)

    path = planner.shortest_connector_path(
        (-116.205, 43.626),
        (-116.204, 43.626),
        graph,
        0.01,
    )

    assert path["connector_classes"] == ["osm_public_road"]
    assert path["connector_edges"][0]["connector_class"] == "osm_public_road"


def test_shortest_connector_path_caches_nearest_node_snaps(tmp_path):
    planner = load_planner()
    connector = tmp_path / "connector.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"TrailName": "Cache Connector"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.205, 43.626],
                                [-116.204, 43.626],
                                [-116.203, 43.626],
                            ],
                        },
                    }
                ],
            }
        )
    )
    graph = planner.load_connector_graph(connector)

    first = planner.shortest_connector_path(
        (-116.205, 43.626),
        (-116.203, 43.626),
        graph,
        0.1,
    )
    second = planner.shortest_connector_path(
        (-116.205, 43.626),
        (-116.203, 43.626),
        graph,
        0.1,
    )

    assert first == second
    assert graph["_nearest_node_cache"]


def test_elevation_effort_does_not_count_jump_between_disconnected_segments():
    planner = load_planner()
    segments = [
        {"coordinates": [(-116.205, 43.626), (-116.204, 43.626)]},
        {"coordinates": [(-116.105, 43.626), (-116.104, 43.626)]},
    ]

    effort = planner.build_elevation_effort(
        segments,
        elevation_sampler=lambda point: 100.0 if point[0] < -116.15 else 500.0,
        official_miles=2.0,
        estimated_moving_minutes=30,
    )

    assert effort["ascent_ft"] == 0
    assert effort["descent_ft"] == 0


def test_segment_estimates_include_per_segment_elevation_gain():
    planner = load_planner()
    segment = {
        "seg_id": 1,
        "seg_name": "Climber Trail 1",
        "trail_name": "Climber Trail",
        "official_miles": 1.0,
        "estimated_moving_minutes": 20,
        "coordinates": [(-116.205, 43.626), (-116.204, 43.627)],
    }

    enriched = planner.enrich_segment_estimates_with_elevation(
        [dict(segment)],
        [segment],
        elevation_sampler=lambda point: 100.0 if point[0] < -116.2045 else 260.0,
    )

    assert enriched[0]["ascent_ft"] == 160
    assert enriched[0]["descent_ft"] == 0
    assert enriched[0]["grade_adjusted_miles"] == 1.16
    assert enriched[0]["estimated_moving_minutes_p75"] == 25


def test_conservative_time_estimates_include_elevation_and_wayfinding_penalty():
    planner = load_planner()

    estimates = planner.build_time_estimates_minutes(
        drive_to=5,
        parking_minutes=8,
        raw_moving_minutes=78,
        effort={
            "effort_score": 94,
            "estimated_moving_minutes_p75": 105,
            "elevation_source": "dem",
        },
        route_finding_penalty_minutes=18,
    )

    assert estimates["door_to_door_raw"] == 96
    assert estimates["door_to_door_p50"] == 112
    assert estimates["door_to_door_p75"] == 141
    assert estimates["recommended_door_to_door"] == 141


def test_build_plan_counts_mapped_official_repeat_return_edges(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(official, [feature(1, "Repeat Return 1", 5280, -116.205, 43.626)])
    connector = tmp_path / "empty-connectors.geojson"
    connector.write_text(json.dumps({"type": "FeatureCollection", "features": []}))

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
        connector_geojson=connector,
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    assert candidate["return_to_car"]["strategy"] == "mapped_official_repeat_return"
    assert candidate["return_to_car"]["official_repeat_miles"] == 1.0
    assert candidate["return_to_car"]["official_repeat_segment_ids"] == [1]
    assert candidate["return_to_car"]["path_coordinates"][0] == [-116.204, 43.627]
    assert candidate["return_to_car"]["path_coordinates"][-1] == [-116.205, 43.626]
    assert candidate["official_repeat_miles"] == 1.0
    assert "return_connector_needs_map_validation" not in candidate["less_optimal_flags"]


def test_bidirectional_candidate_orients_to_practical_trailhead_endpoint(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            {
                "type": "Feature",
                "properties": {
                    "segId": 1,
                    "segName": "Practical Start 1",
                    "LengthFt": 5280,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.25, 43.66], [-116.205, 43.626]],
                },
            }
        ],
    )
    connector = tmp_path / "empty-connectors.geojson"
    connector.write_text(json.dumps({"type": "FeatureCollection", "features": []}))

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
        connector_geojson=connector,
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    assert candidate["route_orientation"]["direction"] == "reversed"
    assert candidate["validation"]["trailhead_snap_confidence"] == "high"
    assert candidate["validation"]["trailhead_snap"]["direct_gap_miles"] == 0
    assert candidate["route_status"] == "graph_validated"


def test_candidate_counts_trailhead_access_out_and_back(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            {
                "type": "Feature",
                "properties": {
                    "segId": 1,
                    "segName": "Access Trail 1",
                    "LengthFt": 5280,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.202, 43.626], [-116.201, 43.626]],
                },
            }
        ],
    )
    connector = tmp_path / "connectors.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"TrailName": "Parking Access"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.205, 43.626],
                                [-116.202, 43.626],
                            ],
                        },
                    }
                ],
            }
        )
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(),
        generated_at="2026-05-04T12:00:00Z",
        connector_geojson=connector,
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    assert candidate["trailhead_access"]["source"] == "mapped_graph"
    assert candidate["trailhead_access"]["round_trip_connector_miles"] > 0
    assert candidate["time_breakdown_minutes"]["trailhead_access"] > 0
    assert candidate["connector_miles"] >= candidate["trailhead_access"]["round_trip_connector_miles"]
    assert candidate["estimated_total_on_foot_miles"] == planner.round_miles(
        candidate["official_new_miles"]
        + candidate["official_repeat_miles"]
        + candidate["connector_miles"]
        + candidate["road_miles"]
    )


def test_mapped_access_within_configured_limit_is_graph_validated(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            {
                "type": "Feature",
                "properties": {
                    "segId": 1,
                    "segName": "Mapped Access 1",
                    "LengthFt": 5280,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.197, 43.626], [-116.196, 43.626]],
                },
            }
        ],
    )
    connector = tmp_path / "connectors.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"TrailName": "Access Path"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.205, 43.626],
                                [-116.197, 43.626],
                            ],
                        },
                    }
                ],
            }
        )
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(
            outing_model={
                **base_state()["outing_model"],
                "mapped_trailhead_access_max_miles": 0.75,
            }
        ),
        generated_at="2026-05-04T12:00:00Z",
        connector_geojson=connector,
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    assert candidate["trailhead_access"]["one_way_miles"] > 0.25
    assert candidate["validation"]["trailhead_snap_confidence"] == "medium"
    assert candidate["route_status"] == "graph_validated"


def test_long_mapped_access_is_executable_but_flagged_less_optimal(tmp_path):
    planner = load_planner()
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            {
                "type": "Feature",
                "properties": {
                    "segId": 1,
                    "segName": "Long Access 1",
                    "LengthFt": 5280,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.17, 43.626], [-116.169, 43.626]],
                },
            }
        ],
    )
    connector = tmp_path / "connectors.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"TrailName": "Long Mapped Access"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.205, 43.626],
                                [-116.17, 43.626],
                            ],
                        },
                    }
                ],
            }
        )
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(
            outing_model={
                **base_state()["outing_model"],
                "mapped_trailhead_access_max_miles": 0.75,
            }
        ),
        generated_at="2026-05-04T12:00:00Z",
        connector_geojson=connector,
    )

    candidate = plan["route_menu"]["all_candidates"][0]
    assert candidate["trailhead_access"]["one_way_miles"] > 1.0
    assert candidate["validation"]["trailhead_snap"]["graph_validated"] is True
    assert candidate["route_status"] == "graph_validated"
    assert "long_mapped_trailhead_access" in candidate["less_optimal_flags"]


def test_nearly_closed_loop_return_is_graph_validated_without_connector_guess(tmp_path):
    planner = load_planner()
    trail = {
        "official_miles": 5.0,
        "start": (-116.205, 43.626),
        "end": (-116.20499, 43.62601),
        "remaining_segment_ids": [1],
    }

    return_to_car = planner.build_return_to_car(
        trail,
        {**planner.default_outing_model(), "closed_loop_gap_tolerance_miles": 0.05},
        connector_graph=None,
    )

    assert return_to_car["strategy"] == "closed_loop"
    assert return_to_car["graph_validated"] is True
    assert return_to_car["needs_map_validation"] is False
    assert return_to_car["connector_miles"] == 0


def test_reversed_trail_orientation_reverses_segment_coordinates():
    planner = load_planner()
    trail = {
        "trail_name": "Geometry Trail",
        "segments": [
            {
                "seg_id": 1,
                "start": (-116.25, 43.66),
                "end": (-116.205, 43.626),
                "coordinates": [(-116.25, 43.66), (-116.23, 43.64), (-116.205, 43.626)],
            }
        ],
        "remaining_segment_ids": [1],
        "start": (-116.25, 43.66),
        "end": (-116.205, 43.626),
    }

    reversed_trail = planner.reverse_trail_orientation(trail)

    segment = reversed_trail["segments"][0]
    assert segment["start"] == (-116.205, 43.626)
    assert segment["end"] == (-116.25, 43.66)
    assert segment["coordinates"] == [(-116.205, 43.626), (-116.23, 43.64), (-116.25, 43.66)]


def test_primary_bucket_prefers_graph_validated_candidate_over_more_efficient_draft():
    planner = load_planner()
    draft = {
        "candidate_id": "fast-draft",
        "time_bucket": "under_1_hour",
        "route_status": "draft",
        "route_quality_score": 0,
        "efficiency_score": 0.2,
        "official_new_miles": 4.0,
        "estimated_total_on_foot_miles": 4.5,
        "total_minutes": 40,
        "trail_names": ["Fast Draft"],
    }
    graph_validated = {
        "candidate_id": "validated-loop",
        "time_bucket": "under_1_hour",
        "route_status": "graph_validated",
        "route_quality_score": 2,
        "efficiency_score": 0.1,
        "official_new_miles": 2.0,
        "estimated_total_on_foot_miles": 2.1,
        "total_minutes": 45,
        "trail_names": ["Validated Loop"],
    }

    menu = planner.build_route_menu([draft, graph_validated])

    assert menu["primary_candidates_by_bucket"]["under_1_hour"]["candidate_id"] == "validated-loop"
    assert menu["primary_validated_candidates_by_bucket"]["under_1_hour"]["candidate_id"] == "validated-loop"
    assert menu["primary_draft_candidates_by_bucket"]["under_1_hour"]["candidate_id"] == "fast-draft"
    assert menu["buckets"]["under_1_hour"][0]["optimality_rank_in_bucket"] == 1


def test_direction_validation_requires_and_records_ascent_delta():
    planner = load_planner()
    segment = {
        "seg_id": 9,
        "seg_name": "Ascent Trail 1",
        "trail_name": "Ascent Trail",
        "direction": "ascent",
        "start": (-116.205, 43.626),
        "end": (-116.204, 43.627),
    }

    missing = planner.build_direction_validation([segment])
    checked = planner.build_direction_validation(
        [segment],
        elevation_sampler=lambda point: 100.0 if point == segment["start"] else 160.0,
    )

    assert missing["passed"] is False
    assert missing["reason"] == "ascent_segments_need_elevation_validation"
    assert checked["passed"] is True
    assert checked["ascent_segment_checks"][0]["elevation_delta_ft"] == 60.0


def test_direction_validation_plans_reverse_when_geometry_runs_downhill():
    planner = load_planner()
    segment = {
        "seg_id": 10,
        "seg_name": "Reverse Ascent Trail 1",
        "trail_name": "Reverse Ascent Trail",
        "direction": "ascent",
        "start": (-116.205, 43.626),
        "end": (-116.204, 43.627),
    }

    checked = planner.build_direction_validation(
        [segment],
        elevation_sampler=lambda point: 160.0 if point == segment["start"] else 100.0,
    )

    assert checked["passed"] is True
    assert checked["planned_traversal_direction"]["10"] == "official_geometry_end_to_start"
    assert checked["ascent_segment_checks"][0]["planned_elevation_gain_ft"] == 60.0


def test_long_access_scraps_are_bundled_without_requiring_largest_remote_trail(tmp_path):
    planner = load_planner()
    parking = (-116.205, 43.626)
    remote = (-116.17, 43.626)
    official = tmp_path / "official.geojson"
    write_geojson(
        official,
        [
            {
                "type": "Feature",
                "properties": {
                    "segId": 1,
                    "segName": "Big Remote 1",
                    "LengthFt": 105600,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[remote[0], remote[1]], [-116.168, 43.626]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "segId": 2,
                    "segName": "Tiny A 1",
                    "LengthFt": 1584,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[remote[0], remote[1]], [-116.169, 43.626]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "segId": 3,
                    "segName": "Tiny B 1",
                    "LengthFt": 1584,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.169, 43.626], [-116.168, 43.626]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "segId": 4,
                    "segName": "Tiny C 1",
                    "LengthFt": 1584,
                    "direction": "both",
                    "specInst": "",
                    "activity_type": "both",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.168, 43.626], [-116.167, 43.626]],
                },
            },
        ],
    )
    connector = tmp_path / "connector.geojson"
    connector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"TrailName": "Remote Access"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [parking[0], parking[1]],
                                [remote[0], remote[1]],
                            ],
                        },
                    }
                ],
            }
        )
    )

    plan = planner.build_plan(
        official_geojson=official,
        state=base_state(
            outing_model={
                **base_state()["outing_model"],
                "mapped_trailhead_access_max_miles": 0.25,
            }
        ),
        generated_at="2026-05-04T12:00:00Z",
        connector_geojson=connector,
    )

    scrap_bundle = next(
        candidate
        for candidate in plan["route_menu"]["all_candidates"]
        if candidate["candidate_type"] == "long_access_bundle"
        and candidate["trail_names"] == ["Tiny A", "Tiny B", "Tiny C"]
    )

    assert scrap_bundle["route_status"] == "graph_validated"
    assert scrap_bundle["official_new_miles"] == 0.9
    assert "long_mapped_trailhead_access" in scrap_bundle["less_optimal_flags"]
