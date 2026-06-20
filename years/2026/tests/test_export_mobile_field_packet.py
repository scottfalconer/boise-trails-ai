import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_mobile_field_packet.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_mobile_field_packet", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dense_line(start, end, steps):
    """Linear interpolation between two coordinates returning steps+1 points.

    The exporter now drops degenerate synthetic routes whose track segments are
    too short (<2 points) or whose consecutive points jump farther than the gap
    tolerance (~0.05 mi). Densifying each leg keeps the synthetic geometry valid
    under that stricter validation while preserving the leg's shape.
    """

    return [
        [
            start[0] + (end[0] - start[0]) * index / steps,
            start[1] + (end[1] - start[1]) * index / steps,
        ]
        for index in range(steps + 1)
    ]


def _joined_path(*parts):
    """Concatenate densified legs, dropping the duplicated joint point."""

    path = []
    for part in parts:
        path.extend(part if not path else part[1:])
    return path


def test_trail_groups_split_same_trail_when_inter_segment_connector_is_declared():
    exporter = load_exporter()
    link = {"distance_miles": 0.25, "connector_names": ["Split Connector"]}
    groups = exporter.trail_groups(
        [
            {"seg_id": 1, "trail_name": "Split Trail"},
            {"seg_id": 2, "trail_name": "Split Trail", "pre_connector_link": link},
        ]
    )

    assert len(groups) == 2
    assert groups[1]["incoming_link"] == link


def test_track_selection_prefers_special_management_legal_cue_track():
    exporter = load_exporter()
    official_index = {
        "1": {
            "parts": [[(-116.205, 43.626), (-116.202, 43.626)]],
            "part_bboxes": [(-116.205, 43.626, -116.202, 43.626)],
        }
    }
    rules = [
        {
            "rule_type": "directional_segment_traversal",
            "segment_direction_overrides": {"1": ["forward"]},
        }
    ]

    selected, source = exporter.select_track_segments_for_outing(
        cue_track_segments=[[(-116.205, 43.626), (-116.202, 43.626)]],
        feature_track_segments=[[(-116.202, 43.626), (-116.205, 43.626)]],
        parking={"lon": -116.205, "lat": 43.626},
        max_parking_gap_miles=0.2,
        special_management_index=official_index,
        special_management_rules=rules,
    )

    assert source == "route_cues"
    assert selected == [[(-116.205, 43.626), (-116.202, 43.626)]]


def test_track_selection_prefers_foot_legal_cue_track_over_blocked_feature_track():
    exporter = load_exporter()
    from field_route_walkthrough_audit import TrailEdge, TrailGraph

    graph = TrailGraph(
        [
            TrailEdge(
                edge_id="unsafe",
                name="Bike Only Connector",
                normalized_name="bike only connector",
                signposts=set(),
                source_class="private_or_blocked",
                coords=[(-116.205, 43.626), (-116.202, 43.626)],
            ),
            TrailEdge(
                edge_id="safe",
                name="Foot Legal Connector",
                normalized_name="foot legal connector",
                signposts=set(),
                source_class="r2r_trail",
                coords=[(-116.205, 43.636), (-116.202, 43.636)],
            ),
        ]
    )

    selected, source = exporter.select_track_segments_for_outing(
        cue_track_segments=[[(-116.205, 43.636), (-116.202, 43.636)]],
        feature_track_segments=[[(-116.205, 43.626), (-116.202, 43.626)]],
        parking={"lon": -116.205, "lat": 43.626},
        max_parking_gap_miles=1.0,
        walkthrough_graph=graph,
    )

    assert source == "route_cues"
    assert selected == [[(-116.205, 43.636), (-116.202, 43.636)]]


def test_track_selection_prefers_graph_matched_cue_track_before_raw_gap_count():
    exporter = load_exporter()
    from field_route_walkthrough_audit import TrailEdge, TrailGraph

    graph = TrailGraph(
        [
            TrailEdge(
                edge_id="signed",
                name="Signed Trail",
                normalized_name="signed trail",
                signposts=set(),
                source_class="r2r_trail",
                coords=[(-116.000, 43.000), (-116.003, 43.000)],
            ),
        ]
    )

    selected, source = exporter.select_track_segments_for_outing(
        cue_track_segments=[[(-116.000, 43.000), (-116.003, 43.000)]],
        feature_track_segments=[
            [
                (-116.000, 43.010),
                (-116.001, 43.010),
                (-116.002, 43.010),
                (-116.003, 43.010),
                (-116.000, 43.010),
            ]
        ],
        parking={"lon": -116.000, "lat": 43.000},
        max_parking_gap_miles=1.0,
        max_track_gap_miles=0.05,
        walkthrough_graph=graph,
    )

    assert source == "route_cues"
    assert selected == [[(-116.000, 43.000), (-116.003, 43.000)]]


def test_track_selection_uses_cue_source_when_graph_quality_ties():
    exporter = load_exporter()
    from field_route_walkthrough_audit import TrailEdge, TrailGraph

    graph = TrailGraph(
        [
            TrailEdge(
                edge_id="cue",
                name="Cue Trail",
                normalized_name="cue trail",
                signposts=set(),
                source_class="r2r_trail",
                coords=[(-116.000, 43.000), (-116.001, 43.000), (-116.000, 43.000)],
            ),
            TrailEdge(
                edge_id="feature",
                name="Feature Trail",
                normalized_name="feature trail",
                signposts=set(),
                source_class="r2r_trail",
                coords=[(-116.000, 43.001), (-116.001, 43.001), (-116.000, 43.001)],
            ),
        ]
    )
    cue_track = [[(-116.000, 43.000), (-116.001, 43.000), (-116.000, 43.000)]]
    feature_track = [[(-116.000, 43.001), (-116.001, 43.001), (-116.000, 43.001)]]

    selected, source = exporter.select_track_segments_for_outing(
        cue_track,
        feature_track,
        parking={"lon": -116.000, "lat": 43.000},
        max_parking_gap_miles=0.1,
        walkthrough_graph=graph,
    )

    assert source == "route_cues"
    assert selected == cue_track


def test_shortest_connector_repair_replaces_unsafe_path_even_when_legal_path_is_longer(monkeypatch):
    exporter = load_exporter()
    from field_route_walkthrough_audit import TrailEdge, TrailGraph

    graph = TrailGraph(
        [
            TrailEdge(
                edge_id="unsafe",
                name="Bike Only Connector",
                normalized_name="bike only connector",
                signposts=set(),
                source_class="private_or_blocked",
                coords=[(-116.205, 43.626), (-116.204, 43.626)],
            )
        ]
    )

    def fake_shortest_connector_path(*args, **kwargs):
        return {
            "distance_miles": 0.1,
            "connector_miles": 0.1,
            "official_repeat_miles": 0,
            "official_repeat_segment_ids": [],
            "connector_names": ["Foot Legal Connector"],
            "connector_classes": ["r2r_trail"],
            "connector_edges": [{"name": "Foot Legal Connector"}],
            "path_coordinates": [[-116.205, 43.627], [-116.204, 43.627]],
        }

    monkeypatch.setattr(exporter, "shortest_connector_path", fake_shortest_connector_path)
    link = {
        "distance_miles": 0.08,
        "path_coordinates": [[-116.205, 43.626], [-116.204, 43.626]],
    }

    repaired = exporter.apply_shortest_connector_path_to_link(
        link,
        {"graph": {}, "nodes": []},
        snap_tolerance_miles=0.045,
        improvement_tolerance_miles=0.005,
        walkthrough_graph=graph,
    )

    assert repaired is True
    assert link["unsafe_connector_repaired"] is True
    assert link["connector_names"] == ["Foot Legal Connector"]
    assert link["distance_miles"] == 0.1


def test_unsafe_connector_labels_are_removed_without_path_repair(monkeypatch):
    exporter = load_exporter()
    from field_route_walkthrough_audit import TrailEdge, TrailGraph

    graph = TrailGraph(
        [
            TrailEdge(
                edge_id="bucktail",
                name="#20A Bucktail",
                normalized_name="bucktail",
                signposts={"20A"},
                source_class="private_or_blocked",
                coords=[(-116.3, 43.7), (-116.301, 43.7)],
                raw_properties={"TrailName": "Bucktail", "Name": "#20A Bucktail"},
            )
        ]
    )

    monkeypatch.setattr(exporter, "shortest_connector_path", lambda *args, **kwargs: None)
    link = {
        "distance_miles": 0.2,
        "connector_names": ["Two Point", "Bucktail"],
        "signpost_labels": ["#20A Bucktail", "#26A Shane's"],
        "path_coordinates": [[-116.205, 43.626], [-116.204, 43.626]],
    }

    repaired = exporter.apply_shortest_connector_path_to_link(
        link,
        {"graph": {}, "nodes": []},
        snap_tolerance_miles=0.045,
        improvement_tolerance_miles=0.005,
        walkthrough_graph=graph,
        unsafe_connector_label_keys=exporter.unsafe_connector_label_keys(graph),
    )

    assert repaired is False
    assert link["connector_names"] == ["Two Point"]
    assert link["signpost_labels"] == ["#26A Shane's"]
    assert "Bucktail" in link["unsafe_connector_labels_removed"]


def test_stitch_inter_segment_track_gaps_splits_unstitched_internal_gap():
    exporter = load_exporter()

    stitched = exporter.stitch_inter_segment_track_gaps(
        [[(-116.205, 43.626), (-116.155, 43.626), (-116.1549, 43.626)]],
        {"graph": {}, "nodes": []},
        max_gap_miles=0.05,
    )

    assert stitched == [
        [(-116.205, 43.626)],
        [(-116.155, 43.626), (-116.1549, 43.626)],
    ]


def test_link_for_group_transition_matches_trail_names_before_position():
    exporter = load_exporter()
    links = [
        {"from_trail": "Sunshine XC", "to_trail": "Deer Point Trail"},
        {"from_trail": "Around the Mountain Trail", "to_trail": "The Face Trail"},
    ]

    assert (
        exporter.link_for_group_transition(
            links,
            1,
            from_trail="Deer Point Trail",
            to_trail="Around the Mountain Trail",
        )
        == {}
    )
    assert exporter.link_for_group_transition(
        links,
        1,
        from_trail="Around the Mountain Trail",
        to_trail="The Face Trail",
    ) == links[1]


def test_track_segments_for_route_cues_prefers_repaired_between_link_over_stale_prelink():
    exporter = load_exporter()
    stale = {
        "from_trail": "Trail A",
        "to_trail": "Trail B",
        "path_coordinates": [(-116.1, 43.1), (-116.2, 43.2)],
    }
    repaired = {
        "from_trail": "Trail A",
        "to_trail": "Trail B",
        "path_coordinates": [(-116.1, 43.1), (-116.15, 43.15)],
    }

    tracks = exporter.track_segments_for_route_cues(
        [
            {
                "segments": [
                    {"trail_name": "Trail A", "coordinates": [(-116.0, 43.0), (-116.1, 43.1)]},
                    {
                        "trail_name": "Trail B",
                        "pre_connector_link": stale,
                        "coordinates": [(-116.15, 43.15), (-116.2, 43.2)],
                    },
                ],
                "between_links": [repaired],
            }
        ]
    )

    assert tracks == [[(-116.0, 43.0), (-116.1, 43.1), (-116.15, 43.15), (-116.2, 43.2)]]


def test_track_segments_for_route_cues_uses_proven_prelink_over_estimated_between_gap():
    exporter = load_exporter()
    estimated = {
        "from_trail": "Trail A",
        "to_trail": "Trail B",
        "source": "estimated_gap",
        "path_coordinates": [(-116.1, 43.1), (-116.3, 43.3)],
    }
    proven = {
        "from_trail": "Trail A",
        "to_trail": "Trail B",
        "source": "mapped_graph",
        "connector_edges": [{"name": "Connector"}],
        "path_coordinates": [(-116.1, 43.1), (-116.15, 43.15)],
    }

    tracks = exporter.track_segments_for_route_cues(
        [
            {
                "segments": [
                    {"trail_name": "Trail A", "coordinates": [(-116.0, 43.0), (-116.1, 43.1)]},
                    {
                        "trail_name": "Trail B",
                        "pre_connector_link": proven,
                        "coordinates": [(-116.15, 43.15), (-116.2, 43.2)],
                    },
                ],
                "between_links": [estimated],
            }
        ]
    )

    assert tracks == [[(-116.0, 43.0), (-116.1, 43.1), (-116.15, 43.15), (-116.2, 43.2)]]


def test_track_segments_for_route_cues_orients_segment_to_current_cursor():
    exporter = load_exporter()

    tracks = exporter.track_segments_for_route_cues(
        [
            {
                "segments": [
                    {"trail_name": "Trail A", "coordinates": [(-116.0, 43.0), (-116.1, 43.1)]},
                    {
                        "trail_name": "Trail B",
                        "coordinates": [(-116.2, 43.2), (-116.1, 43.1)],
                    },
                ],
            }
        ]
    )

    assert tracks == [[(-116.0, 43.0), (-116.1, 43.1), (-116.2, 43.2)]]


def sample_map_data():
    p0 = [-116.1, 43.1]
    p1 = [-116.1144, 43.1144]
    p2 = [-116.1158, 43.1158]
    p3 = [-116.12165, 43.12165]

    def dense_line(start, end, steps):
        return [
            [
                start[0] + (end[0] - start[0]) * index / steps,
                start[1] + (end[1] - start[1]) * index / steps,
            ]
            for index in range(steps + 1)
        ]

    def joined_path(*parts):
        path = []
        for part in parts:
            path.extend(part if not path else part[1:])
        return path

    route_path = joined_path(
        dense_line(p0, p1, 34),
        dense_line(p1, p2, 4),
        dense_line(p2, p3, 14),
        dense_line(p3, p2, 14),
        dense_line(p2, p1, 4),
        dense_line(p1, p0, 34),
    )
    data = {
        "summary": {
            "package_count": 2,
            "covered_segment_count": 2,
            "official_miles": 2.23,
            "total_on_foot_miles": 3.84,
        },
        "progress": {"completed_segment_ids": [], "blocked_segment_ids": []},
        "manual_design": {
            "areas": [
                {
                    "area_id": "manual-area",
                    "package_number": 2,
                    "title": "Manual Area",
                    "decision": "Needs route design.",
                    "demote_candidate_ids": ["hold-route"],
                    "current_placeholder": {
                        "label": "2",
                        "trailhead": "Hold Trailhead",
                        "door_to_door_minutes": 60,
                        "official_miles": 1.0,
                        "on_foot_miles": 3.0,
                    },
                }
            ]
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Runnable Block",
                "components": [
                    {
                        "candidate_id": "test-route",
                        "trail_names": ["Test Trail"],
                        "official_miles": 1.23,
                        "on_foot_miles": 2.34,
                        "total_minutes": 45,
                        "trailhead": "Test Trailhead",
                        "segment_ids": [101],
                    }
                ],
            },
            {
                "package_number": 2,
                "block_name": "Held Block",
                "components": [
                    {
                        "candidate_id": "hold-route",
                        "trail_names": ["Hold Trail"],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                        "total_minutes": 30,
                        "trailhead": "Hold Trailhead",
                        "segment_ids": [102],
                    }
                ],
            },
        ],
        "feature_collections": {
            "routes": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "route",
                            "candidate_id": "test-route",
                            "title": "Test Trail",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": route_path,
                        },
                    }
                ],
            },
            "official_segments": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "seg_id": 101,
                            "segment_name": "Test Trail 1",
                            "trail_name": "Test Trail",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [p0, p1],
                        },
                    }
                ],
            },
            "parking": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "parking",
                            "candidate_id": "test-route",
                            "name": "Test Trailhead",
                            "has_parking": True,
                        },
                        "geometry": {"type": "Point", "coordinates": [-116.1, 43.1]},
                    }
                ],
            },
        },
        "route_cues": {
            "test-route": {
                "candidate_id": "test-route",
                "title": "Test Trail",
                "route_status": "graph_validated",
                "official_miles": 1.23,
                "on_foot_miles": 2.34,
                "total_minutes": 45,
                "time_estimates_minutes": {
                    "door_to_door_p75": 45,
                    "door_to_door_p90": 59,
                },
                "trailhead": {
                    "name": "Test Trailhead",
                    "lat": 43.1,
                    "lon": -116.1,
                    "has_parking": True,
                    "has_restroom": None,
                    "has_water": True,
                    "water_confidence": "user_verified",
                },
                "logistics": {
                    "car_passes": [
                        {
                            "name": "Pass by car again",
                            "mile_from_start": 1.2,
                            "distance_to_car_miles": 0.01,
                            "lon": -116.1,
                            "lat": 43.1,
                        }
                    ],
                    "known_water": [
                        {
                            "name": "Test Trailhead",
                            "location": "parking/start",
                            "confidence": "user_verified",
                            "lon": -116.1,
                            "lat": 43.1,
                        }
                    ],
                },
                "start_access": {
                    "confidence": "medium",
                    "direct_gap_miles": 0.07,
                    "mapped_access_miles": 0.05,
                    "access_class": "direct",
                    "graph_validated": True,
                },
                "segments": [
                    {
                        "order": 1,
                        "seg_id": 101,
                        "segment_name": "Test Trail 1",
                        "trail_name": "Test Trail",
                        "official_miles": 1.23,
                        "direction_rule": "ascent",
                        "direction_cue": "Climb this segment.",
                        "ascent_ft": 220,
                        "estimated_moving_minutes": 18,
                        "estimated_moving_minutes_p75": 24,
                        "grade_adjusted_miles": 1.0,
                    }
                ],
                "return_to_car": {
                    "description": "Double back to parking.",
                    "official_repeat_miles": 1.23,
                    "official_repeat_segment_ids": [101],
                    "connector_miles": 0,
                    "road_miles": 0,
                },
            }
        },
    }
    data["packages"][0]["components"][0]["trail_names"] = ["Test Trail", "Second Trail"]
    data["packages"][0]["components"][0]["segment_ids"] = [101, 103]
    data["feature_collections"]["official_segments"]["features"].append(
        {
            "type": "Feature",
            "properties": {
                "seg_id": 103,
                "segment_name": "Second Trail 1",
                "trail_name": "Second Trail",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [p2, p3],
            },
        }
    )
    data["route_cues"]["test-route"]["segments"].append(
        {
            "order": 2,
            "seg_id": 103,
            "segment_name": "Second Trail 1",
            "trail_name": "Second Trail",
            "official_miles": 0.5,
            "direction_rule": "both",
            "direction_cue": "Either direction allowed.",
        }
    )
    data["route_cues"]["test-route"]["between_links"] = [
        {
            "from_trail": "Test Trail",
            "to_trail": "Second Trail",
            "distance_miles": 0.12,
            "connector_miles": 0.12,
            "official_repeat_miles": 0,
            "connector_names": ["Road Connector"],
            "connector_classes": ["osm_public_road"],
            "path_start": p1,
            "path_end": p2,
            "path_coordinates": [p1, p2],
        }
    ]
    return data


def sample_field_day_layer():
    return {
        "schema": "boise_trails_human_executable_field_day_layer_v1",
        "generated_at": "2026-05-10T04:50:44Z",
        "publication_status": "needs_route_card_promotion",
        "source_files": {
            "calendar_assignment": "years/2026/checkpoints/test-calendar-assignment.json",
            "field_tool_data": "docs/field-packet/field-tool-data.json",
        },
        "summary": {
            "field_day_count": 1,
            "loop_count": 2,
            "multi_start_day_count": 1,
            "total_p75_minutes": 120,
            "max_p90_minutes": 80,
            "total_between_drive_minutes": 10,
            "schedule_authority": "calendar_assignment",
            "day_gpx_validation_passed": True,
            "schedule_p90_violation_day_count": 0,
            "schedule_p90_violation_days": [],
            "certified_route_card_loop_count": 1,
            "needs_route_card_promotion_loop_count": 1,
            "official_segment_count": 2,
            "covered_segment_count": 2,
            "missing_segment_count": 0,
        },
        "field_days": [
            {
                "date": "2026-06-18",
                "weekday_name": "Thursday",
                "day_type": "weekday",
                "constraints": ["workday-window"],
                "field_day_id": "sample-day",
                "p75_minutes": 120,
                "p90_minutes": 136,
                "field_day_schedule_p75_minutes": 120,
                "field_day_schedule_p90_minutes": 136,
                "route_card_door_to_door_p75_sum": 45,
                "route_card_door_to_door_p90_sum": 59,
                "legacy_recomputed_p75_minutes": 55,
                "legacy_recomputed_p90_minutes": 69,
                "timing_authority": "calendar_assignment",
                "route_card_timing_double_count_risk": True,
                "p90_bound_minutes": 180,
                "stress": 0.756,
                "drive_minutes": 20,
                "between_drive_minutes": 10,
                "loop_count": 2,
                "transfer_count": 1,
                "official_miles": 2.23,
                "on_foot_miles": 3.84,
                "segment_count": 2,
                "segment_ids": [101, 103],
                "schedule_integrity": "passed",
                "execution_status": "needs_route_card_promotion",
                "loops": [
                    {
                        "loop_id": "canonical_field_menu::test-route::Test Trailhead",
                        "source": "canonical_field_menu",
                        "candidate_id": "test-route",
                        "label": "Test Trail",
                        "trailhead": "Test Trailhead",
                        "trail_names": ["Test Trail", "Second Trail"],
                        "segment_count": 2,
                        "official_miles": 1.73,
                        "on_foot_miles": 2.34,
                        "p75_minutes": 45,
                        "p90_minutes": 59,
                        "field_day_schedule_p75_minutes": 45,
                        "field_day_schedule_p90_minutes": 59,
                        "route_card_door_to_door_p75_minutes": 45,
                        "route_card_door_to_door_p90_minutes": 59,
                        "timing_source": "route_card_door_to_door",
                        "validation_passed": True,
                        "manual_design_hold": False,
                        "certification_status": "certified_route_card",
                        "route_card_ref": {
                            "outing_id": "1-1",
                            "label": "1",
                            "candidate_ids": ["test-route"],
                            "gpx_href": "gpx/official/test-trail.gpx",
                            "validation_passed": True,
                        },
                    },
                    {
                        "loop_id": "personal_route_menu::needs-card::Other Trailhead",
                        "source": "personal_route_menu",
                        "candidate_id": "needs-card",
                        "label": "Needs Card",
                        "trailhead": "Other Trailhead",
                        "trail_names": ["Needs Card"],
                        "segment_count": 1,
                        "official_miles": 0.5,
                        "on_foot_miles": 1.5,
                        "p75_minutes": 75,
                        "p90_minutes": 84,
                        "validation_passed": True,
                        "manual_design_hold": False,
                        "certification_status": "needs_route_card_promotion",
                        "route_card_ref": None,
                    },
                ],
            }
        ],
    }


def test_export_field_packet_writes_gpx_for_runnable_outings_and_skips_manual_holds(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)

    assert manifest["summary"]["runnable_outing_count"] == 1
    assert manifest["summary"]["manual_hold_count"] == 1
    assert len(manifest["routes"]) == 1
    assert not list((tmp_path / "gpx").rglob("*hold-route*.gpx"))

    route = manifest["routes"][0]
    repeat_cues = [
        cue for cue in route["wayfinding_cues"] if cue.get("official_repeat_segment_ids")
    ]
    assert repeat_cues[0]["official_repeat_segment_ids"] == ["101"]
    assert repeat_cues[0]["official_repeat_miles"] == 1.23
    nav_gpx = Path(route["gpx_path"]).read_text(encoding="utf-8")
    cue_gpx = Path(route["cue_gpx_path"]).read_text(encoding="utf-8")
    audit_gpx = Path(route["audit_gpx_path"]).read_text(encoding="utf-8")

    assert route["gpx_href"].startswith("gpx/official/")
    assert route["cue_gpx_href"].startswith("gpx/cues/")
    assert route["audit_gpx_href"].startswith("gpx/audit/")
    assert "<name>PARK/START Test Trailhead</name>" in nav_gpx
    assert "<name>CAR PASS 1</name>" in nav_gpx
    assert "<name>WATER Test Trailhead</name>" in nav_gpx
    assert "<name>RETURN TO CAR</name>" in nav_gpx
    assert "<name>CUE 01 Test Trail</name>" in nav_gpx
    assert "<name>ASCENT 1 Test Trail 1</name>" not in nav_gpx
    assert "<trk>" in nav_gpx
    assert '<trkpt lat="43.100000" lon="-116.100000" />' in nav_gpx

    assert "<name>CUE 01 Test Trail</name>" in cue_gpx
    assert "<trk>" not in cue_gpx

    assert "<name>ASCENT 1 Test Trail 1</name>" in audit_gpx
    assert "<name>TURN</name>" in audit_gpx
    assert "Official 1.23 mi; On-foot 3.7 mi; Door-to-door p75 45 min" in audit_gpx


def test_repeat_note_mentions_zero_rounded_repeat_ids():
    module = load_exporter()

    note = module.non_credit_repeat_note(
        "This access leg is not official challenge credit.",
        0,
        [1596],
    )

    assert "repeat official" in note
    assert "no new credit" in note


def test_make_wayfinding_cue_prices_zero_rounded_repeat_ids():
    module = load_exporter()

    cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=0.1,
        cue_type="start_access",
        action="FOLLOW",
        official_repeat_segment_ids=[1596],
        official_repeat_miles=0,
    )

    assert cue["official_repeat_segment_ids"] == ["1596"]
    assert cue["official_repeat_miles"] == 0.01


def test_car_pass_split_prevents_connector_anchor_mismatch():
    module = load_exporter()
    cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=1.0,
        cue_type="connector_named_trail",
        action="FOLLOW",
        signed_as=["Connector"],
        target="Next Trail",
        until="signed junction with Next Trail",
    )
    cue["route_miles"] = 2.0
    cue["route_leg_miles"] = 3.0
    route = {
        "parking": {"name": "Test Trailhead"},
        "logistics": {
            "car_passes": [
                {"mile_from_start": 3.5, "distance_to_car_miles": 0.01}
            ]
        },
        "wayfinding_cues": [cue],
    }

    module.split_wayfinding_cues_at_car_passes(route)

    assert len(route["wayfinding_cues"]) == 2
    assert route["wayfinding_cues"][0]["cue_type"] == "car_pass_connector"
    assert route["wayfinding_cues"][0]["target"] == "Test Trailhead"
    assert route["wayfinding_cues"][0]["route_miles"] == 2.0
    assert route["wayfinding_cues"][0]["route_leg_miles"] == 1.5
    assert route["wayfinding_cues"][1]["target"] == "Next Trail"
    assert module.route_navigation_source_failures(route) == []


def test_non_credit_claimed_repeat_declarations_add_hidden_self_repeat():
    module = load_exporter()
    official_feature = {
        "type": "Feature",
        "properties": {
            "seg_id": 101,
            "segment_name": "Repeated Segment",
            "trail_name": "Repeat Trail",
            "official_miles": 0.5,
            "direction_rule": "both",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[-116.0, 43.0], [-116.01, 43.0]],
        },
    }
    credit_cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=0.5,
        cue_type="follow_official_segment",
        action="FOLLOW",
        official_segment_ids=["101"],
    )
    segment_miles = module.track_distance_miles([[(-116.0, 43.0), (-116.01, 43.0)]])
    credit_cue["route_miles"] = 0
    credit_cue["route_leg_miles"] = segment_miles
    cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=0.5,
        cue_type="exit_access",
        action="FOLLOW",
        note="Return leg does not count as new official challenge credit.",
    )
    cue["route_miles"] = segment_miles
    cue["route_leg_miles"] = segment_miles
    route = {
        "segment_ids": ["101"],
        "wayfinding_cues": [credit_cue, cue],
        "_track_segments": [[(-116.0, 43.0), (-116.01, 43.0), (-116.0, 43.0)]],
        "_official_segment_index": {"101": official_feature},
    }

    module.apply_non_credit_claimed_repeat_declarations(route)

    assert cue["official_repeat_segment_ids"] == ["101"]
    assert cue["official_repeat_miles"] == 0.5
    assert "no new credit" in cue["note"]
    assert "repeat official" in cue["display_detail"]


def test_repeat_miles_never_exceed_leg_miles():
    module = load_exporter()
    # A long official segment (4.5 mi) is credited once, then a short exit/access cue
    # (~0.2 mi) re-touches it. The repeat accounting must never attribute the whole
    # segment's mileage to the short leg the user actually walks.
    official_feature = {
        "type": "Feature",
        "properties": {
            "seg_id": 101,
            "segment_name": "Long Repeated Segment",
            "trail_name": "Long Repeat Trail",
            "official_miles": 4.5,
            "direction_rule": "both",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[-116.0, 43.0], [-116.06, 43.0]],
        },
    }
    short_exit = (-116.002, 43.0)
    credit_cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=4.5,
        cue_type="follow_official_segment",
        action="FOLLOW",
        official_segment_ids=["101"],
    )
    credit_cue["route_miles"] = 0
    credit_cue["route_leg_miles"] = 4.5
    # The short connector/exit cue carries a repeat declaration that, due to upstream
    # whole-segment attribution, exceeds the ~0.2-mi leg it covers.
    exit_cue = module.make_wayfinding_cue(
        seq=2,
        cum_miles=4.5,
        leg_miles=0.21,
        cue_type="exit_access",
        action="FOLLOW",
        official_repeat_segment_ids=["101"],
        official_repeat_miles=4.5,
        source_path_coordinates=[(-116.0, 43.0), short_exit],
    )
    exit_cue["route_miles"] = 4.5
    exit_cue["route_leg_miles"] = 0.21
    # Model the buggy note that attributes the whole 4.5-mi segment to this short leg.
    exit_cue["note"] = module.non_credit_repeat_note(
        module.non_credit_repeat_prefix_for_cue(exit_cue),
        4.5,
        ["101"],
    )
    module.refresh_wayfinding_text(exit_cue)
    assert "4.5 mi repeat official" in exit_cue["display_detail"]
    route = {
        "segment_ids": ["101"],
        "wayfinding_cues": [credit_cue, exit_cue],
        "_track_segments": [[(-116.0, 43.0), (-116.06, 43.0), (-116.0, 43.0)]],
        "_official_segment_index": {"101": official_feature},
    }

    module.cap_route_cue_repeat_miles(route)

    for cue in route["wayfinding_cues"]:
        leg_cap = max(
            float(cue.get("leg_miles") or 0),
            module.cue_field_leg_miles(cue),
            module.path_length_miles(
                cue.get("source_path_coordinates") or cue.get("source_path") or []
            ),
        )
        assert float(cue.get("official_repeat_miles") or 0) <= leg_cap + 1e-6

    # The short exit cue keeps a real (non-zero) repeat note, just capped to its leg.
    assert exit_cue["official_repeat_miles"] > 0
    assert exit_cue["official_repeat_miles"] <= float(exit_cue["leg_miles"]) + 1e-6
    assert "repeat official" in exit_cue["display_detail"]


def test_refresh_wayfinding_measurements_caps_repeat_to_leg():
    module = load_exporter()
    cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=0.2,
        cue_type="exit_access",
        action="FOLLOW",
        note="Return leg does not count as new official challenge credit.",
        official_repeat_segment_ids=["101"],
        official_repeat_miles=4.5,
    )
    route = {"outing": {}, "wayfinding_cues": [cue]}
    module.refresh_wayfinding_measurements(route, declare_non_credit_repeats=False)
    assert cue["official_repeat_miles"] <= float(cue["leg_miles"]) + 1e-6
    assert cue["official_repeat_miles"] > 0


def test_non_credit_repeat_declaration_uses_source_path_geometry():
    module = load_exporter()
    a = (-116.0, 43.0)
    b = (-116.01, 43.0)
    c = (-116.02, 43.0)
    official_feature = {
        "type": "Feature",
        "properties": {
            "seg_id": 101,
            "segment_name": "Repeated Segment",
            "trail_name": "Repeat Trail",
            "official_miles": 0.5,
            "direction_rule": "both",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [a, b],
        },
    }
    credit_cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=0.5,
        cue_type="follow_official_segment",
        action="FOLLOW",
        official_segment_ids=["101"],
    )
    segment_miles = module.track_distance_miles([[a, b]])
    credit_cue["route_miles"] = 0
    credit_cue["route_leg_miles"] = segment_miles
    cue = module.make_wayfinding_cue(
        seq=2,
        cum_miles=0.5,
        leg_miles=0.5,
        cue_type="exit_access",
        action="FOLLOW",
        note="Return leg does not count as new official challenge credit.",
        source_path_coordinates=[b, a],
    )
    route = {
        "segment_ids": ["101"],
        "wayfinding_cues": [credit_cue, cue],
        "_track_segments": [[a, b, c]],
        "_official_segment_index": {"101": official_feature},
    }

    module.apply_non_credit_claimed_repeat_declarations(route)

    assert cue["official_repeat_segment_ids"] == ["101"]
    assert cue["official_repeat_miles"] == 0.5


def test_non_credit_cue_that_earns_required_segment_is_promoted():
    module = load_exporter()
    a = (-116.0, 43.0)
    b = (-116.01, 43.0)
    c = (-116.02, 43.0)
    official_index = {
        "101": {
            "type": "Feature",
            "properties": {
                "seg_id": 101,
                "seg_name": "First Segment",
                "trail_name": "First Trail",
                "official_miles": 0.5,
                "direction_rule": "both",
            },
            "geometry": {"type": "LineString", "coordinates": [a, b]},
        },
        "102": {
            "type": "Feature",
            "properties": {
                "seg_id": 102,
                "seg_name": "Second Segment",
                "trail_name": "Second Trail",
                "official_miles": 0.5,
                "direction_rule": "both",
            },
            "geometry": {"type": "LineString", "coordinates": [b, c]},
        },
    }
    first_miles = module.track_distance_miles([[a, b]])
    second_miles = module.track_distance_miles([[b, c]])
    credit_cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=first_miles,
        cue_type="follow_official_segment",
        action="FOLLOW",
        official_segment_ids=["101"],
    )
    credit_cue["route_miles"] = 0
    credit_cue["route_leg_miles"] = first_miles
    connector_cue = module.make_wayfinding_cue(
        seq=2,
        cum_miles=first_miles,
        leg_miles=second_miles,
        cue_type="connector_named_trail",
        action="FOLLOW",
        target="Second Trail",
        source_path_coordinates=[b, c],
    )
    later_cue = module.make_wayfinding_cue(
        seq=3,
        cum_miles=first_miles + second_miles,
        leg_miles=second_miles,
        cue_type="junction_turn",
        action="TAKE",
        official_segment_ids=["102"],
    )
    route = {
        "segment_ids": ["101", "102"],
        "wayfinding_cues": [credit_cue, connector_cue, later_cue],
        "_track_segments": [[a, b, c]],
        "_official_segment_index": official_index,
    }

    module.promote_non_credit_required_segment_cues(route)

    assert connector_cue["cue_type"] == "junction_turn"
    assert connector_cue["official_segment_ids"] == ["102"]
    assert "This earns" in connector_cue["note"]
    assert later_cue.get("official_segment_ids") is None
    assert later_cue["official_repeat_segment_ids"] == ["102"]
    assert "no new credit" in later_cue["note"]


def test_rejected_avoidable_repeat_repair_does_not_mutate_track_segments(tmp_path):
    module = load_exporter()
    start = (-116.0, 43.0)
    end = (-115.99, 43.0)
    long_detour = (-115.995, 43.02)
    leg_miles = module.track_distance_miles([[start, end]])

    def official_feature(segment_id, coords, official_miles):
        return {
            "type": "Feature",
            "properties": {
                "seg_id": segment_id,
                "segment_name": f"Segment {segment_id}",
                "trail_name": f"Trail {segment_id}",
                "official_miles": official_miles,
                "direction_rule": "both",
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        }

    connector_path = tmp_path / "connectors.geojson"
    connector_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    connector_graph = module.load_connector_graph(
        connector_path,
        official_segments=[
            {
                "seg_id": 101,
                "trail_name": "Repeated Trail",
                "direction": "both",
                "official_miles": leg_miles,
                "coordinates": [start, end],
            },
            {
                "seg_id": 202,
                "trail_name": "Underpriced Detour",
                "direction": "both",
                "official_miles": 0.01,
                "coordinates": [end, long_detour, start],
            },
        ],
    )
    route = {
        "segment_ids": ["101"],
        "outing": {"segment_ids": ["101"]},
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "route_miles": 0.0,
                "route_leg_miles": leg_miles,
                "official_segment_ids": ["101"],
            },
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": leg_miles,
                "route_leg_miles": leg_miles,
                "official_repeat_segment_ids": ["101"],
            },
        ],
        "_track_segments": [[start, end, start]],
        "_official_segment_index": {
            "101": official_feature(101, [start, end], leg_miles),
            "202": official_feature(202, [end, long_detour, start], 0.01),
        },
    }
    before_segments = [[tuple(point) for point in segment] for segment in route["_track_segments"]]
    before_miles = module.track_distance_miles(route["_track_segments"])

    repairs = module.repair_avoidable_post_credit_repeats(
        route,
        connector_graph=connector_graph,
        completed_at_export_ids=set(),
        snap_tolerance_miles=0.02,
        rebuild_first=False,
    )

    assert repairs == []
    assert route["_track_segments"] == before_segments
    assert module.track_distance_miles(route["_track_segments"]) == before_miles


def test_route_truth_lollipop_skips_avoidable_repeat_repair():
    module = load_exporter()
    route = {
        "route_cues": [{"cue_generation_mode": "route_truth_repair_explicit_lollipop"}],
        "wayfinding_cues": [
            {"seq": 1, "official_segment_ids": ["101"]},
            {"seq": 2, "official_repeat_segment_ids": ["101"]},
        ],
        "_track_segments": [[(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)]],
        "_official_segment_index": {},
    }

    repairs = module.repair_avoidable_post_credit_repeats(
        route,
        connector_graph={"nodes": [{"id": "placeholder"}]},
        completed_at_export_ids={"101"},
        snap_tolerance_miles=0.02,
        rebuild_first=False,
    )

    assert repairs == []
    assert route["intentional_post_credit_repeat_policy"] == "route_truth_lollipop"


def test_load_map_data_prefers_canonical_json_over_html_snapshot(tmp_path):
    module = load_exporter()
    canonical = sample_map_data()
    html_snapshot = sample_map_data()
    html_snapshot["packages"][0]["components"][0]["candidate_id"] = "html-snapshot-route"
    html = "<script>\nconst DATA = " + json.dumps(html_snapshot) + ";\nconst map = {};\n</script>"
    json_path = tmp_path / "canonical-map-data.json"
    html_path = tmp_path / "snapshot.html"
    json_path.write_text(json.dumps(canonical), encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    loaded, source_path = module.load_map_data(map_html=html_path, map_data_json=json_path)

    assert source_path == json_path
    assert loaded["packages"][0]["components"][0]["candidate_id"] == "test-route"


def test_load_map_data_allows_explicit_html_when_json_is_missing(tmp_path):
    module = load_exporter()
    html_snapshot = sample_map_data()
    html_snapshot["packages"][0]["components"][0]["candidate_id"] = "explicit-html-route"
    html = "<script>\nconst DATA = " + json.dumps(html_snapshot) + ";\nconst map = {};\n</script>"
    html_path = tmp_path / "explicit.html"
    html_path.write_text(html, encoding="utf-8")

    loaded, source_path = module.load_map_data(map_html=html_path, map_data_json=tmp_path / "missing.json")

    assert source_path == html_path
    assert loaded["packages"][0]["components"][0]["candidate_id"] == "explicit-html-route"


def test_export_field_packet_allows_long_single_card_without_accepted_replacement(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    collapsed = data["packages"][0]["components"][0]
    collapsed["candidate_id"] = "block-generic_merged_field_card"
    collapsed["trailhead"] = "Any Trailhead"
    collapsed["trail_names"] = ["Long Trail", "Second Trail", "Third Trail"]
    collapsed["official_miles"] = 8.59
    collapsed["on_foot_miles"] = 13.66
    collapsed["total_minutes"] = 299

    manifest = module.export_field_packet(data, tmp_path)

    assert manifest["summary"]["runnable_outing_count"] == 1


def test_export_field_packet_rejects_missing_accepted_replacement_candidate(tmp_path, monkeypatch):
    module = load_exporter()
    data = sample_map_data()
    replacements_path = tmp_path / "field-menu-replacements.json"
    replacements_path.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "package_number": 1,
                        "replace_package": {
                            "components": [
                                {
                                    "candidate_id": "replacement-route-a",
                                    "source": "multi_start_field_menu_replacement",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_FIELD_MENU_REPLACEMENTS_JSON", replacements_path)

    with pytest.raises(ValueError, match="Accepted field-menu replacement package 1 is missing candidates"):
        module.export_field_packet(data, tmp_path)


def test_export_field_packet_uses_explicit_replacements_json(tmp_path, monkeypatch):
    module = load_exporter()
    data = sample_map_data()
    stale_replacements = tmp_path / "stale-field-menu-replacements.json"
    stale_replacements.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "package_number": 1,
                        "replace_package": {
                            "components": [
                                {
                                    "candidate_id": "stale-replacement-route",
                                    "source": "multi_start_field_menu_replacement",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    active_replacements = tmp_path / "active-field-menu-replacements.json"
    active_replacements.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "package_number": 1,
                        "replace_package": {
                            "components": [
                                {
                                    "candidate_id": "test-route",
                                    "source": "multi_start_field_menu_replacement",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_FIELD_MENU_REPLACEMENTS_JSON", stale_replacements)

    manifest = module.export_field_packet(
        data,
        tmp_path,
        field_menu_replacements_json=active_replacements,
    )

    assert manifest["summary"]["runnable_outing_count"] == 1


def test_field_packet_html_is_phone_first_and_links_to_gpx_and_parking(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "Phone Field Packet" in html
    assert "Open Field GPX" in html
    assert "Open Live Map" in html
    assert "Open parking in Google Maps" in html
    assert manifest["routes"][0]["gpx_href"] in html
    assert f"live-map.html?outing={manifest['routes'][0]['outing_id']}" in html
    assert "<h2>Test Trail</h2>" in html
    assert "1 · Test" in html
    assert "<b>Climb</b><strong>220 ft</strong>" in html
    assert "<b>Door to door p90</b><strong>59 min</strong>" in html
    assert "Cue GPX" not in html
    assert "Audit GPX" not in html
    assert manifest["routes"][0]["cue_gpx_href"] not in html
    assert manifest["routes"][0]["audit_gpx_href"] not in html
    assert "https://www.google.com/maps/dir/?api=1&amp;destination=43.100000,-116.100000" in html
    assert "PARK/START" in html
    assert "What to do next" in html
    assert "Tap the cue you are working on" in html
    assert "decision-cards" in html
    assert "current-step" in html
    assert "Turn-by-turn from car" not in html
    assert "Park/start at Test Trailhead" in html
    assert "Pass by car again" in html
    assert "Known water" in html
    assert "Test Trailhead · parking/start · user_verified" in html
    assert "Leave car toward Test Trail" not in html
    assert "OFFICIAL START" in html
    assert "Follow Test Trail toward Second Trail" in html
    assert "This earns: Test Trail segment 1" in html
    assert "Includes 1.23 mi repeat official; no new credit." in html
    assert "220 ft climb" in html
    assert "~24 min moving" in html
    assert "ROAD" in html
    assert "Follow Road Connector toward Second Trail" in html
    assert "JCT" in html
    assert "Second Trail toward return to car" in html
    assert "EXIT" in html
    assert "Return leg does not count as new official challenge credit." in html
    assert "Pin active" in html
    assert "Clear active" in html
    assert "fieldPacketActiveOuting" in html
    assert "Planner snap" not in html
    assert "Official segment order" not in html
    assert "Before leaving" not in html
    assert "Phone run card" not in html


def test_field_packet_embeds_field_day_layer_in_json_and_html(tmp_path):
    module = load_exporter()

    module.export_field_packet(
        sample_map_data(),
        tmp_path,
        field_day_layer_data=sample_field_day_layer(),
    )
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert field_data["execution_model"]["primary_execution_artifact"] == "route_cards"
    assert field_data["execution_model"]["default_phone_view"] == "routes"
    assert field_data["execution_model"]["route_cards_are_proof_units"] is True
    assert field_data["execution_model"]["field_days_publication_status"] == "field_day_certified"

    field_day_layer = field_data["field_day_layer"]
    assert field_day_layer["publication_status"] == "field_day_certified"
    assert field_day_layer["execution_model"]["primary_execution_artifact"] == "route_cards"
    assert field_day_layer["execution_model"]["proof_unit"] == "certified_route_card"
    assert field_day_layer["execution_model"]["default_phone_view"] == "routes"
    assert field_day_layer["summary"]["field_day_count"] == 1
    assert field_day_layer["summary"]["covered_segment_count"] == 2
    assert field_day_layer["summary"]["schedule_authority"] == "calendar_assignment"
    assert field_day_layer["summary"]["schedule_p90_violation_day_count"] == 0
    assert field_day_layer["field_days"][0]["segment_ids"] == ["101", "103"]
    assert field_day_layer["field_days"][0]["field_day_schedule_p75_minutes"] == 120
    assert field_day_layer["field_days"][0]["route_card_door_to_door_p75_sum"] == 45
    assert field_day_layer["field_days"][0]["schedule_integrity"] == "passed"
    assert (
        field_day_layer["field_days"][0]["loops"][0]["route_card_door_to_door_p75_minutes"]
        == 45
    )
    assert field_day_layer["field_days"][0]["loops"][0]["route_card_ref"]["outing_id"] == "1-1"
    assert field_day_layer["field_days"][0]["loops"][0]["route_name"] == "Test Trail"
    assert field_day_layer["field_days"][0]["loops"][0]["route_code"] == "1"
    assert field_day_layer["field_days"][0]["loops"][0]["route_card_ref"]["route_name"] == "Test Trail"

    assert '<body class="view-routes">' in html
    assert 'const DEFAULT_VIEW = "routes";' in html
    assert '<button type="button" class="active" data-view="routes">Route Cards</button>' in html
    assert '<button type="button" data-view="field-days">Field Days</button>' in html
    assert html.index('data-view="routes"') < html.index('data-view="field-days"')
    assert 'data-view="field-days"' in html
    assert 'new URLSearchParams(window.location.search).get("view")' in html
    assert 'requestedView === "field-days"' in html
    assert 'requestedView === "routes"' in html
    assert 'location.hash === "#field-days"' in html
    assert 'id="field-day-view"' in html
    assert "Open route cards for GPX, parking, cues, and return-to-car detail" in html
    assert "Route Cards" in html
    assert "Field Days" in html
    assert "Thursday, 2026-06-18" in html
    assert "1 certified loop" in html
    assert "0 needs route-card promotion" not in html
    assert 'href="#1-1"' in html
    assert f'href="{field_day_layer["field_days"][0]["loops"][0]["route_card_ref"]["gpx_href"]}"' in html
    assert "Needs Card" not in html


def test_field_day_view_labels_reserve_days_explicitly():
    module = load_exporter()
    field_day_layer = sample_field_day_layer()
    field_day_layer["summary"].update(
        {
            "calendar_day_count": 2,
            "active_execution_day_count": 1,
            "reserve_day_count": 1,
        }
    )
    field_day_layer["field_days"].append(
        {
            "date": "2026-07-18",
            "weekday_name": "Saturday",
            "field_day_id": "reserve-day",
            "execution_status": "reusable_empty_field_day",
            "p75_minutes": 0,
            "p90_minutes": 0,
            "on_foot_miles": 0,
            "segment_count": 0,
            "loop_count": 0,
            "between_drive_minutes": 0,
            "loops": [],
        }
    )

    html = module.render_field_day_view(field_day_layer)

    assert "2 calendar days" in html
    assert "1 active execution day" in html
    assert "1 reserve day" in html
    assert "Reserve / buffer day - no route planned." in html


def test_field_packet_omits_stale_field_day_layer_route_refs(tmp_path):
    module = load_exporter()
    field_day_layer = sample_field_day_layer()
    stale_ref = field_day_layer["field_days"][0]["loops"][0]["route_card_ref"]
    stale_ref["outing_id"] = "112-1"
    stale_ref["label"] = "FD12A"

    manifest = module.export_field_packet(
        sample_map_data(),
        tmp_path,
        field_day_layer_data=field_day_layer,
    )
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert manifest["field_day_layer_consistency"]["status"] == "omitted_stale_route_refs"
    assert manifest["field_day_layer_consistency"]["route_ref_failure_count"] == 1
    assert manifest["summary"]["field_day_layer_included"] is False
    assert manifest["summary"]["field_day_layer_route_ref_failure_count"] == 1
    assert field_data["execution_model"]["primary_execution_artifact"] == "route_cards"
    assert field_data["execution_model"]["default_phone_view"] == "routes"
    assert "field_day_layer" not in field_data
    assert "field_day_layer" not in public_manifest
    assert 'href="#112-1"' not in html
    assert "FD12A" not in html
    assert public_manifest["field_day_layer_consistency"] == {
        "status": "omitted_stale_route_refs",
        "route_ref_failure_count": 1,
    }
    assert "112-1" not in json.dumps(public_manifest)
    assert "FD12A" not in json.dumps(public_manifest)


def test_field_packet_omits_field_day_layer_with_stale_segment_refs(tmp_path):
    module = load_exporter()
    field_day_layer = sample_field_day_layer()
    loop = field_day_layer["field_days"][0]["loops"][0]
    loop["segment_ids"] = [101]

    manifest = module.export_field_packet(
        sample_map_data(),
        tmp_path,
        field_day_layer_data=field_day_layer,
    )
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))

    assert manifest["field_day_layer_consistency"]["status"] == "omitted_stale_route_refs"
    assert manifest["field_day_layer_consistency"]["route_ref_failures"] == [
        {
            "code": "field_day_route_ref_segment_mismatch",
            "field_day_index": 0,
            "loop_index": 0,
            "field_day_id": "sample-day",
            "loop_id": "canonical_field_menu::test-route::Test Trailhead",
            "loop_label": "Test Trail",
            "outing_id": "1-1",
            "route_ref_segment_ids": ["101"],
            "current_route_segment_ids": ["101", "103"],
        }
    ]
    assert "field_day_layer" not in field_data


def test_field_packet_writes_live_gps_map_and_precaches_it(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)

    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")
    service_worker = (tmp_path / "service-worker.js").read_text(encoding="utf-8")

    assert "Live GPS Route Map" in live_map_html
    assert "field-tool-data.json" in live_map_html
    assert "navigator.geolocation.watchPosition" in live_map_html
    assert "DOMParser" in live_map_html
    assert "Route display" in live_map_html
    assert "Distance to route" not in live_map_html
    assert "GPS accuracy" not in live_map_html
    assert "route-progress" not in live_map_html
    assert "map-field-packet-link" not in live_map_html
    assert '<a class="overview-link" href="index.html">Return to overview</a>' in live_map_html
    assert "Back to field packet" not in live_map_html
    header_html = live_map_html[: live_map_html.index("</header>")]
    footer_html = live_map_html[live_map_html.index("<footer>") : live_map_html.index("</footer>")]
    assert '<div class="button-row"' not in header_html
    assert '<div class="status"' not in header_html
    assert '<div class="button-row"' in footer_html
    assert '<div class="status"' not in footer_html
    assert "data-style=\"ribbon\"" in live_map_html
    assert "data-style=\"cue-legs\"" in live_map_html
    assert "data-style=\"napkin\"" not in live_map_html
    assert 'id="show-all-route"' in live_map_html
    assert 'aria-pressed="false">Show all</button>' in live_map_html
    assert "leaflet" not in live_map_html.lower()
    assert '"live-map.html"' in service_worker


def test_live_gps_map_can_render_optional_basemap_tiles_without_leaflet(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert 'id="tile-layer"' in live_map_html
    assert 'id="basemap-button"' in live_map_html
    assert 'id="tile-attribution"' in live_map_html
    assert "const TILE_BASEMAPS" in live_map_html
    assert "https://tile.openstreetmap.org" in live_map_html
    assert "FoothillsMosaic2025" in live_map_html
    assert "OpenStreetMap contributors" in live_map_html
    assert "R2R / Ada County imagery" in live_map_html
    assert "function tileXYForLatLon" in live_map_html
    assert "function latLonForTileXY" in live_map_html
    assert "function drawTiles" in live_map_html
    assert "drawTiles();" in live_map_html
    assert "function cycleBasemap" in live_map_html
    assert 'state.basemap: "osm"' not in live_map_html
    assert 'basemap: "osm"' in live_map_html
    assert "leaflet" not in live_map_html.lower()


def test_live_gps_map_uses_active_outing_and_gpx_href_from_field_data(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "fieldPacketActiveOuting" in live_map_html
    assert "URLSearchParams" in live_map_html
    assert "route.gpx_href" in live_map_html
    assert "parseGpx" in live_map_html
    assert "projectPointToRoute" in live_map_html
    assert "nearest cue" in live_map_html.lower()


def test_live_gps_map_preserves_track_segments_and_does_not_draw_hidden_gaps(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "trackSegments" in live_map_html
    assert 'gpxNodes(xml, "trkseg")' in live_map_html
    assert "projectedSegments" in live_map_html
    assert "pathForSegments" in live_map_html
    assert "gap-warning" in live_map_html
    assert 'gpxNodes(xml, "trkpt").map' not in live_map_html
    assert "buildSegmentCumulative" in live_map_html


def test_live_gps_map_uses_wayfinding_cues_as_primary_markers(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "positionForRouteM" in live_map_html
    assert "function routeCues()" in live_map_html
    assert "state.route?.live_map_cues || state.route?.wayfinding_cues || []" in live_map_html
    assert "String(cue.seq || index + 1)" in live_map_html
    assert "state.waypoints\n        .filter" not in live_map_html
    assert "MAX_OVERVIEW_CHEVRONS" in live_map_html
    assert "simplifyPolyline" in live_map_html
    assert "function refreshDisplaySegments() {\n      state.displayedSegments = state.projectedSegments.map" in live_map_html
    assert "function refreshDisplaySegments() {\n      refreshDisplaySegments();" not in live_map_html
    assert 'class="route-slice route-line' not in live_map_html
    assert "function smoothPathFor(points)" in live_map_html
    assert "SCHEMATIC_COLOR_STEP_M = 80" in live_map_html
    assert "SCHEMATIC_COLOR_OVERLAP_M = 8" in live_map_html
    assert "SCHEMATIC_CONTEXT_TOLERANCE_M = 10" in live_map_html
    assert "pathForSegments(chunkSegments, { smooth: true })" in live_map_html
    assert 'stroke="${routeColorAt(mid / total)}"' in live_map_html
    assert "ROUTE_GRADIENT_STOPS" in live_map_html
    assert "{ at: 0, color: [220, 38, 38] }" in live_map_html
    assert "{ at: 0.33, color: [234, 179, 8] }" in live_map_html
    assert "{ at: 0.66, color: [22, 163, 74] }" in live_map_html
    assert "hue = 215" not in live_map_html
    assert "segment[pointIndex].routeM = total" in live_map_html
    assert "SCHEMATIC_LANE_SPACING_PX = 10" in live_map_html
    assert "SCHEMATIC_MIN_LANE_SPACING_M = 1" in live_map_html
    assert "SCHEMATIC_MAX_LANE_SPACING_M = 22" in live_map_html
    assert ".transit-trunk-halo" in live_map_html
    assert ".transit-trunk-band" in live_map_html
    assert ".transit-trunk-separator" in live_map_html
    assert "stroke-width:30" in live_map_html
    assert "stroke-width:22" in live_map_html
    assert "stroke-width:4" in live_map_html
    assert "SCHEMATIC_LANE_JUMP_MIN_M" not in live_map_html
    assert "function currentSchematicLaneSpacingM()" in live_map_html
    assert "function refreshContextSegments(force = false)" in live_map_html
    assert "function selfOverlapRangesForRouteRange(startM, endM)" in live_map_html
    assert "function activeSelfOverlapRanges()" in live_map_html
    assert "function routeSampleAt(routeM)" in live_map_html
    assert "angleDifference(left.angle, right.angle) < 2.2" in live_map_html
    assert "cue.official_repeat_segment_ids || []" not in live_map_html
    assert "for (let nextIndex = index + 1; nextIndex < cues.length; nextIndex += 1)" not in live_map_html
    assert "function subtractActiveSelfOverlapRanges(startM, endM)" in live_map_html
    assert "for (const repeat of activeSelfOverlapRanges())" in live_map_html
    assert "function nonActiveSelfOverlapSegmentsForRouteRange(startM, endM, options = {})" in live_map_html
    assert "function activeSelfOverlapTrunkSegmentsForRouteRange(startM, endM)" in live_map_html
    assert "segmentsFromDisplaySourceForRouteRange(state.displayedSegments" in live_map_html
    assert "const trunkSegments = activeSelfOverlapTrunkSegmentsForRouteRange(leg.startM, leg.endM)" in live_map_html
    assert "const visibleSegments = nonActiveSelfOverlapSegmentsForRouteRange(visibleStartM, state.totalRouteM, { context: true })" in live_map_html
    assert "const activeSegments = smoothSegmentsForDisplay(" in live_map_html
    assert "const arrowSegments = sourceActiveSegments.length" in live_map_html
    assert "activeLegArrows(leg.startM, leg.endM, { segments: arrowSegments })" in live_map_html
    assert "transit-trunk-halo" in live_map_html
    assert "transit-trunk-band" in live_map_html
    assert "transit-trunk-separator" in live_map_html
    assert "function isSchematicLaneJump(previous, current)" not in live_map_html
    assert "function splitTransitMapDisplaySegments(segments)" not in live_map_html
    assert "function isBacktrackingTurn(previous, current, next)" not in live_map_html
    assert "function splitBacktrackingDisplaySegments(segments)" not in live_map_html
    assert "function mergeContinuousDisplaySegments(segments, maxGapM)" in live_map_html
    assert "state.contextSegments = mergeContinuousDisplaySegments(state.projectedSegments.map" in live_map_html
    assert "offsetRepeatedCorridors" not in live_map_html
    assert "segmentsForRouteRange(visibleStartM, state.totalRouteM, { context: true })" not in live_map_html
    assert "state.totalRouteM = metrics.totalRouteM;\n      refreshDisplaySegments();" in live_map_html
    assert "state.totalRouteM = metrics.totalRouteM;\n      state.displayedSegments =" not in live_map_html
    assert "const cueStops = routeCues()\n          .map(cue => cueRouteM(cue))" in live_map_html
    assert "const cueM = cueRouteM(cue);" in live_map_html
    assert "const cueM = cardMilesToRouteM(cue.cum_miles)" not in live_map_html


def test_live_gps_map_offsets_repeated_corridors_like_transit_lanes(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "SCHEMATIC_LANE_SPACING_PX = 10" in live_map_html
    assert "SCHEMATIC_MAX_LANE_SPACING_M = 22" in live_map_html
    assert "SCHEMATIC_CONTEXT_TOLERANCE_M = 10" in live_map_html
    assert "SCHEMATIC_LANE_JUMP_MIN_M" not in live_map_html
    assert "function isSchematicLaneJump(previous, current)" not in live_map_html
    assert "function splitTransitMapDisplaySegments(segments)" not in live_map_html
    assert "function selfOverlapRangesForRouteRange(startM, endM)" in live_map_html
    assert "function activeSelfOverlapRanges()" in live_map_html
    assert "function routeSampleAt(routeM)" in live_map_html
    assert "angleDifference(left.angle, right.angle) < 2.2" in live_map_html
    assert "cue.official_repeat_segment_ids || []" not in live_map_html
    assert "for (let nextIndex = index + 1; nextIndex < cues.length; nextIndex += 1)" not in live_map_html
    assert "function subtractActiveSelfOverlapRanges(startM, endM)" in live_map_html
    assert "for (const repeat of activeSelfOverlapRanges())" in live_map_html
    assert "function nonActiveSelfOverlapSegmentsForRouteRange(startM, endM, options = {})" in live_map_html
    assert "function activeSelfOverlapTrunkSegmentsForRouteRange(startM, endM)" in live_map_html
    assert "segmentsFromDisplaySourceForRouteRange(state.displayedSegments" in live_map_html
    assert "const trunkSegments = activeSelfOverlapTrunkSegmentsForRouteRange(leg.startM, leg.endM)" in live_map_html
    assert "const visibleSegments = nonActiveSelfOverlapSegmentsForRouteRange(visibleStartM, state.totalRouteM, { context: true })" in live_map_html
    assert "const activeSegments = smoothSegmentsForDisplay(" in live_map_html
    assert "const arrowSegments = sourceActiveSegments.length" in live_map_html
    assert "activeLegArrows(leg.startM, leg.endM, { segments: arrowSegments })" in live_map_html
    assert "offsetRepeatedCorridors" not in live_map_html
    assert "function mergeContinuousDisplaySegments(segments, maxGapM)" in live_map_html
    assert "state.contextSegments = mergeContinuousDisplaySegments(state.projectedSegments.map" in live_map_html
    assert 'class="transit-trunk-halo"' in live_map_html
    assert 'class="transit-trunk-band"' in live_map_html
    assert 'class="transit-trunk-separator"' in live_map_html
    assert "return output;" in live_map_html


def test_live_gps_map_is_active_cue_leg_navigation_artifact(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "Active leg" in live_map_html
    assert "Field cue-leg map" in live_map_html
    assert "The blue ribbon is the active cue-to-cue leg" in live_map_html
    assert "state.activeCueIndex" in live_map_html
    assert "function activeLegRange" in live_map_html
    assert "function visibleRouteStartM()" in live_map_html
    assert "function activeContextStartM" in live_map_html
    assert "return state.showAllRoute ? 0 : activeContextStartM()" in live_map_html
    assert "function activeLegMilesText(leg)" in live_map_html
    assert 'return `+${cueMiles.toFixed(2)} mi cue · map +${routeMiles.toFixed(2)} mi`' in live_map_html
    assert "function setActiveCueIndex" in live_map_html
    assert "function cueIndexForRouteM" in live_map_html
    assert "function requestedCueIndex()" in live_map_html
    assert "function nextDistinctCueIndex" in live_map_html
    assert "function previousDistinctCueIndex" in live_map_html
    assert "previousCue.addEventListener(\"click\", () => setActiveCueIndex(previousDistinctCueIndex(), { fit: true }));" in live_map_html
    assert "nextCue.addEventListener(\"click\", () => setActiveCueIndex(nextDistinctCueIndex(), { fit: true }));" in live_map_html
    assert "function fitActiveLeg" in live_map_html
    assert "const contextStartM = activeContextStartM(leg.index)" in live_map_html
    assert "segmentsForRouteRange(contextStartM, leg.endM, { context: true })" in live_map_html
    assert "setActiveCueIndex(requestedCueIndex() ?? cueIndexForRouteM(0), { render: false });" in live_map_html
    assert 'class="route-context"' in live_map_html
    assert 'class="active-line"' in live_map_html
    assert "segmentsForRouteRange(leg.startM, leg.endM, { context: true })" in live_map_html
    assert "function smoothPolylinePoints(points, steps = 6)" in live_map_html
    assert "function smoothSegmentsForDisplay(segments)" in live_map_html
    assert "const activeSegments = smoothSegmentsForDisplay(" in live_map_html
    assert "const activePath = pathForSegments(activeSegments);" in live_map_html
    assert "const activePath = pathForSegments(activeSegments, { smooth: true })" not in live_map_html
    assert "activeLegArrows(leg.startM, leg.endM, { segments: arrowSegments })" in live_map_html
    assert 'id="previous-cue"' in live_map_html
    assert 'id="next-cue"' in live_map_html
    assert 'id="fit-leg"' in live_map_html

def test_live_gps_map_shows_home_eta_from_active_cue_start(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert 'id="cue-home-eta"' in live_map_html
    assert 'id="cue-home-eta-text"' in live_map_html
    assert "Home ETA from cue start" in live_map_html
    assert "function remainingMinutesFromActiveCueStart" in live_map_html
    assert "function homeEtaText" in live_map_html
    assert "Date.now()" in live_map_html
    assert "state.route?.door_to_door_minutes_p75" in live_map_html
    assert "state.route?.door_to_door_minutes_p90" in live_map_html
    assert "updateCueHomeEta();" in live_map_html
    assert 'showAllRouteButton.addEventListener("click"' in live_map_html
    assert live_map_html.index('id="cue-home-eta"') < live_map_html.index('class="overview-link"')


def test_live_gps_map_keeps_previous_cue_context_by_default_with_show_all_override(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "showAllRoute: false" in live_map_html
    assert '<button type="button" id="show-all-route" aria-pressed="false">Show all</button>' in live_map_html
    assert "function visibleRouteStartM()" in live_map_html
    assert "function activeContextStartM" in live_map_html
    assert "const previousIndex = previousDistinctCueIndex(clamped)" in live_map_html
    assert "return state.showAllRoute ? 0 : activeContextStartM()" in live_map_html
    assert "const legMiles = activeLegMilesText(leg);" in live_map_html
    assert "const visibleStartM = visibleRouteStartM();" in live_map_html
    assert "const visibleSegments = nonActiveSelfOverlapSegmentsForRouteRange(visibleStartM, state.totalRouteM, { context: true })" in live_map_html
    assert "drawProgressRibbon(visibleStartM)" in live_map_html
    assert "value >= visibleStartM - 8" in live_map_html
    assert "cueM < visibleStartM - 8" in live_map_html
    assert 'showAllRouteButton.addEventListener("click"' in live_map_html
    assert 'showAllRouteButton.setAttribute("aria-pressed", state.showAllRoute ? "true" : "false")' in live_map_html


def test_live_gps_map_halves_active_line_width_for_split_overlap_cues(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert ".active-line.split-lane { fill:none; stroke:#2563eb; stroke-width:5;" in live_map_html
    assert ".active-halo.split-lane { fill:none; stroke:#fff; stroke-width:12;" in live_map_html
    assert "function activeLegUsesSplitLane(leg)" in live_map_html
    assert 'cueType === "overlap_repeat"' in live_map_html
    assert "Boolean(cue.overlap_match)" in live_map_html
    assert "const activeLaneClass = activeLegUsesSplitLane(leg) ? \" split-lane\" : \"\";" in live_map_html
    assert 'class="active-line split-lane"' in live_map_html
    assert 'class="active-line" d="${activePath}"' in live_map_html


def test_live_gps_map_does_not_hide_start_when_start_and_finish_overlap(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "sameStartFinish" in live_map_html
    assert "START/FINISH" in live_map_html
    assert "const endpointMarkers" in live_map_html
    assert "function endpointCallout" in live_map_html
    assert ".endpoint-anchor" in live_map_html
    assert ".endpoint-callout-line" in live_map_html
    assert "const endpointAnchorRadius = 5 * unit" in live_map_html
    assert "const endpointMarkerRadius = 11 * unit" in live_map_html
    assert 'r="17"' not in live_map_html
    assert 'r="15"' not in live_map_html
    assert "...endpointMarkers,\n        ...cueMarkers" in live_map_html


def test_live_gps_map_default_viewport_is_single_screen_follow_surface(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "height:100dvh" in live_map_html
    assert "overflow:hidden" in live_map_html
    assert ".map-shell { position:relative; min-height:0;" in live_map_html
    assert 'id="map-leg-banner"' in live_map_html
    assert "updateMapLegBanner" in live_map_html
    assert "FROM" in live_map_html
    assert "NEXT" in live_map_html
    assert "function mapUnitsPerPixel" in live_map_html
    assert "const unit = mapUnitsPerPixel()" in live_map_html
    assert "context-marker" in live_map_html
    assert "context-label" in live_map_html
    assert "const isActive = index === state.activeCueIndex" in live_map_html
    assert "const isNext = index === leg.nextIndex" in live_map_html


def test_live_gps_map_top_cue_banner_can_be_hidden(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert 'id="map-leg-banner-close"' in live_map_html
    assert 'aria-label="Hide cue banner"' in live_map_html
    assert 'id="map-leg-banner-content"' in live_map_html
    assert ".map-leg-banner-close" in live_map_html
    assert 'const mapLegBannerContent = document.getElementById("map-leg-banner-content")' in live_map_html
    assert 'const mapLegBannerClose = document.getElementById("map-leg-banner-close")' in live_map_html
    assert "dismissedMapLegBannerKey" in live_map_html
    assert "function mapLegBannerKey" in live_map_html
    assert "if (routeUnavailable(state.route))" in live_map_html
    assert "mapLegBannerContent.textContent = \"\";" in live_map_html
    assert 'mapLegBannerClose.addEventListener("click"' in live_map_html
    assert "state.dismissedMapLegBannerKey = mapLegBannerKey(leg)" in live_map_html
    assert "mapLegBannerContent.innerHTML" in live_map_html
    assert "mapLegBanner.innerHTML =" not in live_map_html


def test_live_gps_map_offsets_active_cue_markers_from_exact_junction(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "const isCallout = isActive || isNext" in live_map_html
    assert "const calloutDistance = isCallout ? 44 * unit" in live_map_html
    assert "const calloutAngle = (point.angle || 0) + (isActive ? -Math.PI / 2 : Math.PI / 2)" in live_map_html
    assert ".cue-anchor {" in live_map_html
    assert 'class="cue-anchor${anchorClass}"' in live_map_html
    assert ".cue-callout-line" in live_map_html
    assert "const radiusForCue = (isActive || isNext ? 16 : 6) * unit" in live_map_html
    assert "const radius = nearby ? 24 * unit : 0" not in live_map_html


def test_live_gps_map_uses_consistent_active_leg_direction_arrows(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "function displayedRoutePositionForM" in live_map_html
    assert "function activeLegArrows" in live_map_html
    assert "function displaySegmentLength(segment)" in live_map_html
    assert "function displayPointAtDistance(segment, targetDistance)" in live_map_html
    assert "function directionArrowPath(center, angle, unit)" in live_map_html
    assert 'class="direction-arrow"' in live_map_html
    assert "const unit = mapUnitsPerPixel()" in live_map_html
    assert "const displaySegments = options.segments || []" in live_map_html
    assert "for (const segment of displaySegments)" in live_map_html
    assert "Math.max(1, Math.min(8" in live_map_html
    assert "arrowSpacing" in live_map_html
    assert "angle - Math.PI" in live_map_html
    assert "const sample = displayedRoutePositionForM(target, options)" in live_map_html
    assert "const displaySource = options.segments || (options.context ? state.contextSegments : state.displayedSegments)" in live_map_html
    assert "const center = sample" in live_map_html
    assert "const angle = sample.angle" in live_map_html
    assert "function sourcePathSegmentsForCue(cue)" in live_map_html
    assert "const sourceActiveSegments = sourcePathSegmentsForCue(leg.cue)" in live_map_html
    assert "activeLegArrows(leg.startM, leg.endM, { segments: arrowSegments })" in live_map_html
    assert "routeLayer.innerHTML = routeHtml + activeLegArrows" in live_map_html
    assert "chevrons(state.style === \"napkin\" ? 8 : 5, leg.startM, leg.endM)" not in live_map_html


def test_live_gps_map_keeps_transit_lanes_continuous(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "function isBacktrackingTurn(previous, current, next)" not in live_map_html
    assert "function splitBacktrackingDisplaySegments(segments)" not in live_map_html
    assert "cosine < -0.78" not in live_map_html
    assert "return output;" in live_map_html
    assert "segmentsForRouteRange(leg.startM, leg.endM, { context: true })" in live_map_html


def test_live_gps_map_is_gesture_map_with_passive_gps_dot(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert 'id="follow-button"' not in live_map_html
    assert "state.follow" not in live_map_html
    assert "followButton" not in live_map_html
    assert "GPS-driven active-cue" not in live_map_html
    assert "const activePointers = new Map()" in live_map_html
    assert "function svgPointFromClient" in live_map_html
    assert "function panViewBox" in live_map_html
    assert "function zoomAt" in live_map_html
    assert 'svg.addEventListener("pointerdown"' in live_map_html
    assert 'svg.addEventListener("pointermove"' in live_map_html
    assert 'svg.addEventListener("wheel"' in live_map_html
    assert 'svg.addEventListener("pointercancel"' in live_map_html
    assert "setActiveCueIndex(cueIndexForRouteM(nearest.routeM), { render: false });" not in live_map_html
    assert "fitActiveLeg(true)" not in live_map_html
    assert "render();" in live_map_html


def test_live_gps_map_surfaces_offscreen_gps_without_autofollow(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "function pointInViewBox" in live_map_html
    assert "function offscreenUserIndicator" in live_map_html
    assert "GPS off map" in live_map_html
    assert ".user-offscreen" in live_map_html
    assert ".user-offscreen-label" in live_map_html
    assert 'class="user-dot"' in live_map_html
    assert 'class="user-dot" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${(10 * unit).toFixed(1)}"' in live_map_html
    assert "fitButton.textContent = state.user ? \"Fit GPS\" : \"Fit\"" in live_map_html
    assert "nearestCue.textContent = \"GPS acquired; tap Fit GPS to include your dot.\"" in live_map_html
    assert "function fitGpsToNextCue" in live_map_html
    assert "function nextCueIndexAfterRouteM" in live_map_html
    assert "function cuePointForIndex" in live_map_html
    assert "state.user ? fitGpsToNextCue() : fitRoute(false); render();" in live_map_html
    assert "fitPoints([userPoint, nextCuePoint || finishPoint], 90, 0.42, 85);" in live_map_html
    assert "fitRoute(Boolean(state.user)); render();" not in live_map_html
    assert "fitRoute(true); render();" not in live_map_html
    assert "fitActiveLeg(true)" not in live_map_html


def test_live_gps_map_uses_nav_gpx_without_source_mismatch_masking(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    live_map_html = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert "Route GPX length" not in live_map_html
    assert "differs from route card" not in live_map_html
    assert "Route review needed" not in live_map_html
    assert "Math.min(plannedMeters, state.totalRouteM)" not in live_map_html
    assert "displayRouteEndM" not in live_map_html
    assert "progressTotalM" not in live_map_html
    assert "function cardRouteTotalM" in live_map_html
    assert "function routeMToCardM" in live_map_html
    assert "function cardMilesToRouteM" in live_map_html
    assert "state.projected = state.routePositions" in live_map_html
    assert "const cueColor = routeColorAt" not in live_map_html
    assert 'fill="${cueColor}"' not in live_map_html


def test_validate_outing_export_does_not_treat_named_connector_as_hidden_track_gap():
    module = load_exporter()
    outing = {"candidate_ids": ["gap-route"]}
    track_segments = [
        [(-116.0, 43.0), (-116.0, 43.01)],
        [(-116.2, 43.2), (-116.2, 43.21)],
    ]
    route_cues = {
        "gap-route": {
            "between_links": [
                {
                    "from_trail": "Trail A",
                    "to_trail": "Trail B",
                    "connector_names": ["Named Connector"],
                    "connector_classes": ["r2r_trail"],
                    "connector_miles": 0.4,
                }
            ]
        }
    }

    validation = module.validate_outing_export(
        outing,
        track_segments,
        parking={"lon": -116.0, "lat": 43.0},
        route_cues=route_cues,
        max_gap_miles=0.05,
        max_parking_gap_miles=100,
    )

    assert validation["passed"] is False
    assert any(failure["code"] == "unexplained_inter_segment_gap" for failure in validation["failures"])


def test_field_packet_prices_route_card_from_exported_gpx_when_source_mileage_drifts(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["packages"][0]["components"][0]["on_foot_miles"] = 1.0
    data["route_cues"]["test-route"]["on_foot_miles"] = 1.0
    data["feature_collections"]["routes"]["features"][0]["properties"]["on_foot_miles"] = 1.0

    manifest = module.export_field_packet(data, tmp_path)

    route = manifest["routes"][0]
    assert manifest["summary"]["gpx_validation_passed"] is True
    assert route["validation"]["passed"] is True
    assert route["outing"]["on_foot_miles"] > 1.0
    assert route["outing"]["source_card_on_foot_miles"] == 1.0
    assert "field_mileage_reconciled_from_gpx" not in route["outing"]
    assert "route_gpx_mileage_mismatch" not in [
        failure["code"] for failure in route["validation"]["failures"]
    ]


def test_wayfinding_cue_mileage_reconciles_to_route_card(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["packages"][0]["components"][0]["on_foot_miles"] = 1.0
    data["route_cues"]["test-route"]["on_foot_miles"] = 1.0
    data["route_cues"]["test-route"]["segments"][0]["official_miles"] = 1.5
    # The card on_foot_miles is recomputed from the (densified) track geometry
    # (~3.7 mi), so the route card always wins regardless of the fixture's
    # on_foot_miles. To make the SOURCE cue total diverge from the card by more
    # than route_card_mileage_tolerance (~0.30 mi) -- and therefore force the
    # scaling path -- shrink the second credit segment's declared official
    # mileage so the un-scaled cue total drops well below the card.
    data["route_cues"]["test-route"]["segments"][1]["official_miles"] = 0.1
    data["route_cues"]["test-route"]["logistics"]["car_passes"] = []

    manifest = module.export_field_packet(data, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    route = manifest["routes"][0]
    field_route = field_data["routes"][0]
    cue_total = max(
        float(cue.get("cum_miles") or 0) + float(cue.get("leg_miles") or 0)
        for cue in field_route["wayfinding_cues"]
    )
    route_anchor_total = max(
        float(cue.get("route_miles") or 0) + float(cue.get("route_leg_miles") or 0)
        for cue in field_route["wayfinding_cues"]
    )

    assert route["outing"]["on_foot_miles"] > 1.0
    # After scaling, the cue total is pinned to the route-card on_foot_miles.
    assert cue_total == route["outing"]["on_foot_miles"]
    assert route_anchor_total == pytest.approx(route["outing"]["on_foot_miles"], abs=0.01)
    # Scaling provably fired: every cue now carries a source_leg_miles snapshot
    # of its pre-scale mileage, and the scaled leg_miles differ from it.
    cues = field_route["wayfinding_cues"]
    assert all(cue.get("source_leg_miles") is not None for cue in cues)
    scaled_cue = next(
        cue
        for cue in cues
        if cue.get("source_leg_miles")
        and cue["leg_miles"] != cue["source_leg_miles"]
    )
    assert scaled_cue["leg_miles"] != scaled_cue["source_leg_miles"]
    # The first credit segment's declared mileage survives as its source leg.
    assert max(cue.get("source_leg_miles", 0) for cue in cues) == 1.5
    # NOTE: the persisted reconciliation status reads "already_consistent" rather
    # than "scaled_to_route_card". reconcile_wayfinding_miles_to_route_card runs
    # again after the scaling pass (export re-runs refresh_wayfinding_measurements
    # at the end); by then the scaled cue total already matches the card within
    # tolerance, so the final status is recorded as consistent even though the
    # source_leg_miles snapshots above prove scaling occurred.
    assert field_route["wayfinding_mileage_reconciliation"]["status"] == "already_consistent"


def test_certifiable_export_allows_schematic_gpx_with_authoritative_route_card(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["packages"][0]["components"][0]["on_foot_miles"] = 1.0
    data["route_cues"]["test-route"]["logistics"]["car_passes"] = []

    manifest = module.export_field_packet(data, tmp_path, require_certifiable=True)

    assert manifest["summary"]["gpx_validation_passed"] is True
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "live-map.html").exists()
    assert (tmp_path / "field-tool-data.json").exists()
    assert (tmp_path / "manifest.webmanifest").exists()
    assert (tmp_path / "service-worker.js").exists()


def test_diagnostic_export_does_not_render_invalid_route_as_field_card_warning(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["packages"][0]["components"][0]["on_foot_miles"] = 1.0

    module.export_field_packet(data, tmp_path)

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "GPX validation failed" not in html
    assert "Do not use this route in the field until reviewed" not in html


def test_field_packet_names_non_official_access_trail_before_first_credit(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(
        sample_map_data(),
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    route = manifest["routes"][0]
    access_steps = [step for step in route["turn_by_turn_steps"] if step["kind"] == "access"]

    assert "START/ACCESS" in html
    assert "Follow #99 Access Trail toward Test Trail" in html
    assert "UNTIL signed junction with Test Trail" in html
    assert "This access leg is not official challenge credit." in html
    assert access_steps[0]["title"] == "Start on #99 Access Trail"


def test_wayfinding_enrichment_names_start_access_edge_matched_from_route_line():
    module = load_exporter()
    from field_route_walkthrough_audit import TrailEdge

    route = {
        "outing_id": "synthetic",
        "segment_ids": ["official-1"],
        "turn_by_turn_steps": [
            {
                "kind": "access",
                "title": "Leave car toward #51 Who Now Loop",
                "detail": "From the car, head toward #51 Who Now Loop.",
            }
        ],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cum_miles": 0.0,
                "leg_miles": 0.3,
                "cue_type": "start_access",
                "action": "FOLLOW",
                "signed_as": ["#51 Who Now Loop"],
                "target": "#51 Who Now Loop",
                "until": "signed junction with #51 Who Now Loop",
            },
            {
                "seq": 2,
                "cum_miles": 0.3,
                "leg_miles": 0.4,
                "cue_type": "follow_official_segment",
                "action": "FOLLOW",
                "signed_as": ["#51 Who Now Loop"],
                "target": "return to car",
                "until": "end of #51 Who Now Loop for this route",
                "official_segment_ids": ["official-1"],
            },
        ],
    }
    track_segments = [[(-116.0, 43.0), (-116.001, 43.001), (-116.002, 43.002)]]
    graph_edges = [
        TrailEdge(
            edge_id="connector-57",
            name="#57 Harrison Hollow",
            normalized_name="harrison hollow",
            signposts={"57"},
            source_class="r2r_trail",
            coords=[(-116.0, 43.0), (-116.001, 43.001)],
        ),
        TrailEdge(
            edge_id="official-51",
            name="#51 Who Now Loop",
            normalized_name="who now",
            signposts={"51"},
            source_class="official_segment",
            segment_id="official-1",
            coords=[(-116.001, 43.001), (-116.002, 43.002)],
        ),
    ]

    module.enrich_route_with_walkthrough_edge_names(route, track_segments, graph_edges)

    cue_text = json.dumps(route["wayfinding_cues"], ensure_ascii=False)
    step_text = json.dumps(route["turn_by_turn_steps"], ensure_ascii=False)
    assert "#57 Harrison Hollow" in cue_text
    assert "#57 Harrison Hollow" in step_text
    assert "01 0.00 mi" in route["wayfinding_cues"][0]["compact"]


def test_wayfinding_enrichment_names_short_start_access_edge():
    module = load_exporter()
    from field_route_walkthrough_audit import TrailEdge

    route = {
        "outing_id": "synthetic",
        "segment_ids": ["official-1"],
        "turn_by_turn_steps": [
            {
                "kind": "access",
                "title": "Leave car toward #19A Shoshone-Bannock Tribes Trail",
                "detail": "From the car, head toward #19A Shoshone-Bannock Tribes Trail.",
            },
            {
                "kind": "return",
                "title": "Return on North Klotz Lane",
                "detail": "After the official route, return toward the car on North Klotz Lane.",
            }
        ],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cum_miles": 0.0,
                "leg_miles": 0.3,
                "cue_type": "start_access",
                "action": "FOLLOW",
                "signed_as": ["#19A Shoshone-Bannock Tribes Trail"],
                "target": "Shoshone-Paiute",
                "until": "signed junction with Shoshone-Paiute",
            },
            {
                "seq": 2,
                "cum_miles": 0.3,
                "leg_miles": 0.4,
                "cue_type": "follow_official_segment",
                "action": "FOLLOW",
                "signed_as": ["Shoshone-Paiute"],
                "target": "Quarry Trail - Castle Rock",
                "until": "signed junction with Quarry Trail - Castle Rock",
                "official_segment_ids": ["official-1"],
            },
        ],
    }
    track_segments = [[(-116.1680, 43.6033), (-116.1677, 43.6035), (-116.1684, 43.6044)]]
    graph_edges = [
        TrailEdge(
            edge_id="generic-overlap",
            name="OSM footway connector 72484",
            normalized_name="osm footway connector 72484",
            signposts=set(),
            source_class="osm_path_footway",
            coords=[(-116.1680, 43.6033), (-116.1677, 43.6035)],
        ),
        TrailEdge(
            edge_id="connector-short-road",
            name="North Klotz Lane",
            normalized_name="north klotz lane",
            signposts=set(),
            source_class="osm_public_road",
            coords=[(-116.1680, 43.6033), (-116.1677, 43.6035)],
        ),
        TrailEdge(
            edge_id="official-shoshone",
            name="Shoshone-Paiute",
            normalized_name="shoshone paiute",
            signposts=set(),
            source_class="official_segment",
            segment_id="official-1",
            coords=[(-116.1677, 43.6035), (-116.1684, 43.6044)],
        ),
    ]

    module.enrich_route_with_walkthrough_edge_names(route, track_segments, graph_edges)

    cue_text = json.dumps(route["wayfinding_cues"], ensure_ascii=False)
    step_text = json.dumps(route["turn_by_turn_steps"], ensure_ascii=False)
    assert "North Klotz Lane" in cue_text
    assert "North Klotz Lane" in step_text


def test_wayfinding_enrichment_names_between_connector_edge_matched_from_route_line():
    module = load_exporter()
    from field_route_walkthrough_audit import TrailEdge

    route = {
        "outing_id": "synthetic",
        "segment_ids": ["official-1", "official-2"],
        "turn_by_turn_steps": [
            {"kind": "navigate", "title": "Take First Trail", "detail": "Follow First Trail."},
            {"kind": "navigate", "title": "Take Second Trail", "detail": "Turn onto Second Trail."},
        ],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cum_miles": 0.0,
                "leg_miles": 0.4,
                "cue_type": "follow_official_segment",
                "action": "FOLLOW",
                "signed_as": ["First Trail"],
                "target": "Second Trail",
                "until": "signed junction with Second Trail",
                "official_segment_ids": ["official-1"],
            },
            {
                "seq": 2,
                "cum_miles": 0.4,
                "leg_miles": 0.2,
                "cue_type": "connector_named_trail",
                "action": "FOLLOW",
                "signed_as": ["connector/access"],
                "target": "Second Trail",
                "until": "signed junction with Second Trail",
            },
            {
                "seq": 3,
                "cum_miles": 0.6,
                "leg_miles": 0.4,
                "cue_type": "junction_turn",
                "action": "TAKE",
                "signed_as": ["Second Trail"],
                "target": "return to car",
                "until": "end of Second Trail for this route",
                "official_segment_ids": ["official-2"],
            },
        ],
    }
    track_segments = [[(-116.0, 43.0), (-116.001, 43.0), (-116.002, 43.0), (-116.003, 43.0)]]
    graph_edges = [
        TrailEdge(
            edge_id="official-1",
            name="First Trail",
            normalized_name="first",
            signposts=set(),
            source_class="official_segment",
            segment_id="official-1",
            coords=[(-116.0, 43.0), (-116.001, 43.0)],
        ),
        TrailEdge(
            edge_id="connector-road",
            name="Connector Road",
            normalized_name="connector road",
            signposts=set(),
            source_class="osm_public_road",
            coords=[(-116.001, 43.0), (-116.002, 43.0)],
        ),
        TrailEdge(
            edge_id="official-2",
            name="Second Trail",
            normalized_name="second",
            signposts=set(),
            source_class="official_segment",
            segment_id="official-2",
            coords=[(-116.002, 43.0), (-116.003, 43.0)],
        ),
    ]

    module.enrich_route_with_walkthrough_edge_names(route, track_segments, graph_edges)

    cue_text = json.dumps(route["wayfinding_cues"], ensure_ascii=False)
    step_text = json.dumps(route["turn_by_turn_steps"], ensure_ascii=False)
    assert "Connector Road" in cue_text
    assert "Connector Road" in step_text


def test_turn_step_sync_mentions_primary_return_access_name_from_wayfinding():
    module = load_exporter()
    route = {
        "turn_by_turn_steps": [
            {"kind": "park", "title": "Park/start at Dry Creek"},
            {"kind": "navigate", "title": "Take Currant Creek"},
            {
                "kind": "return",
                "title": "Return via #71 Red Tail",
                "detail": "Follow the signed connector/access back toward #71 Red Tail.",
            },
        ],
        "wayfinding_cues": [
            {
                "cue_type": "follow_official_segment",
                "signed_as": ["Currant Creek"],
            },
            {
                "cue_type": "exit_access",
                "signed_as": ["#70 Landslide Loop", "#71 Red Tail", "OSM path connector 111703"],
            },
        ],
    }

    module.synchronize_turn_steps_with_wayfinding_cues(route)

    return_step = route["turn_by_turn_steps"][-1]
    assert "#70 Landslide Loop" in return_step["detail"]


def test_field_packet_exports_wayfinding_cue_sheet_notation(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(
        sample_map_data(),
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    field_tool_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    route = manifest["routes"][0]
    public_route = public_manifest["routes"][0]
    field_route = field_tool_data["routes"][0]

    cues = route["wayfinding_cues"]
    assert cues[0]["seq"] == 1
    assert cues[0]["cue_type"] == "start_access"
    assert cues[0]["action"] == "FOLLOW"
    assert cues[0]["cum_miles"] == 0.0
    assert cues[0]["leg_miles"] > 0
    assert cues[0]["signed_as"] == ["#99 Access Trail"]
    assert cues[0]["target"] == "Test Trail"
    assert cues[0]["until"] == "signed junction with Test Trail"
    assert cues[0]["compact"].startswith("01 0.00")
    assert "START/ACCESS" in cues[0]["compact"]
    # manifest.json / field-tool-data.json round-trip through JSON, so in-memory
    # tuples (e.g. source_path_coordinates) become lists. Compare against the
    # JSON-normalized in-memory cues to preserve the real invariant:
    # serialized == in-memory up to tuple/list coercion.
    serialized_cues = json.loads(json.dumps(cues))
    assert public_route["wayfinding_cues"] == serialized_cues
    assert field_route["wayfinding_cues"] == serialized_cues
    assert "Field Cue Sheet" in html
    assert '<details class="cue-sheet">' in html
    assert '<details class="cue-sheet" open>' not in html
    assert '<section><h3>Field Cue Sheet</h3>' not in html
    assert f'<span class="summary-meta">{module.pluralize(len(cues), "cue")}</span>' in html
    assert "Field Cue Sheet</span> <span" in html
    assert "01 0.00 mi" in html
    assert "START/ACCESS" in html
    assert "UNTIL signed junction with Test Trail" in html
    assert "VERIFY: watch for signs: #99 Access Trail" in html


def test_overlap_exit_warning_reaches_next_cue_and_live_map():
    module = load_exporter()
    route = {
        "label": "synthetic-overlap",
        "outing": {"label": "synthetic-overlap", "trailhead": "Example Trailhead"},
        "wayfinding_cues": [
            module.make_wayfinding_cue(
                seq=7,
                cum_miles=3.8,
                leg_miles=0.77,
                cue_type="connector_named_trail",
                action="FOLLOW",
                signed_as=["#51 Who Now Loop"],
                target="#58 Harrison Ridge",
                until="signed junction with #58 Harrison Ridge",
            ),
            module.make_wayfinding_cue(
                seq=8,
                cum_miles=4.57,
                leg_miles=1.26,
                cue_type="junction_turn",
                action="BEAR LEFT",
                signed_as=["#58 Harrison Ridge"],
                target="return to car",
                until="end of #58 Harrison Ridge for this route",
            ),
        ],
    }

    route["wayfinding_cues"][0]["cue_type"] = "overlap_repeat"
    route["wayfinding_cues"][0]["action"] = "DOUBLE BACK"
    route["wayfinding_cues"][0]["field_warning"] = "Double-back overlap: this leg reuses GPS line from cue 6."
    route["wayfinding_cues"][0]["overlap_match"] = {
        "matched_cue_seq": 6,
        "direction": "opposite",
        "matched_fraction": 1.0,
        "matched_miles": 0.77,
    }
    module.add_cue_avoid(route["wayfinding_cues"][0], "do not read the overlapping full-route line as a separate trail")
    module.refresh_wayfinding_text(route["wayfinding_cues"][0])

    module.apply_overlap_exit_wayfinding_cautions(route)
    cue_7, cue_8 = route["wayfinding_cues"]
    live_map_html = module.render_live_map_html()

    assert cue_7["cue_type"] == "overlap_repeat"
    assert cue_7["action"] == "DOUBLE BACK"
    assert "Double-back overlap" in cue_7["field_warning"]
    assert "OVERLAP" in cue_7["compact"]
    assert "overlapping full-route line" in " ".join(cue_7["avoid"])
    assert "Exit the overlap" in cue_8["field_warning"]
    assert "function cueWarning" in live_map_html
    assert "field_warning" in live_map_html
    assert "leg-warning" in live_map_html


def test_geometry_overlap_detector_marks_future_same_trail_double_backs():
    module = load_exporter()
    track_segments = [
        [
            (-116.0000, 43.0000),
            (-115.9900, 43.0000),
            (-116.0000, 43.0000),
            (-116.0000, 43.0050),
        ]
    ]
    first_leg_miles = module.haversine_miles(track_segments[0][0], track_segments[0][1])
    second_leg_miles = module.haversine_miles(track_segments[0][1], track_segments[0][2])
    total_miles = module.track_distance_miles(track_segments)
    route = {
        "outing": {"on_foot_miles": total_miles},
        "_track_segments": track_segments,
        "wayfinding_cues": [
            module.make_wayfinding_cue(
                seq=1,
                cum_miles=0,
                leg_miles=first_leg_miles,
                cue_type="follow_official_segment",
                action="FOLLOW",
                signed_as=["#99 Test Trail"],
                target="#100 Return Trail",
                until="turnaround",
                official_segment_ids=["1"],
            ),
            module.make_wayfinding_cue(
                seq=2,
                cum_miles=first_leg_miles,
                leg_miles=second_leg_miles,
                cue_type="connector_named_trail",
                action="FOLLOW",
                signed_as=["#99 Test Trail"],
                target="#100 Return Trail",
                until="signed junction with #100 Return Trail",
            ),
            module.make_wayfinding_cue(
                seq=3,
                cum_miles=first_leg_miles + second_leg_miles,
                leg_miles=total_miles - first_leg_miles - second_leg_miles,
                cue_type="junction_turn",
                action="TURN LEFT",
                signed_as=["#100 Return Trail"],
                target="finish",
                until="finish",
            ),
        ],
    }

    module.apply_geometry_overlap_wayfinding_cautions(route)
    cue_2 = route["wayfinding_cues"][1]

    assert cue_2["cue_type"] == "overlap_repeat"
    assert cue_2["action"] == "DOUBLE BACK"
    assert "Double-back overlap" in cue_2["field_warning"]
    assert cue_2["overlap_match"]["matched_cue_seq"] == 1
    assert cue_2["overlap_match"]["direction"] == "opposite"
    assert "OVERLAP" in cue_2["compact"]


def test_missing_segment_effort_is_enriched_without_reverse_direction_warning():
    module = load_exporter()
    cue = {
        "segments": [
            {
                "seg_id": 1579,
                "segment_name": "Kemper's Ridge Trail 1",
                "trail_name": "Kemper's Ridge Trail",
                "official_miles": 0.2,
                "direction_cue": "Either direction allowed; follow map arrows.",
                "estimated_moving_minutes": 4,
            },
            {
                "seg_id": 1581,
                "segment_name": "Kemper's Ridge Trail 3",
                "trail_name": "Kemper's Ridge Trail",
                "official_miles": 0.48,
                "direction_cue": "Either direction allowed; follow map arrows.",
                "estimated_moving_minutes": 8,
            },
            {
                "seg_id": 1582,
                "segment_name": "Kemper's Ridge Trail 4",
                "trail_name": "Kemper's Ridge Trail",
                "official_miles": 0.12,
                "direction_cue": "Either direction allowed; follow map arrows.",
                "estimated_moving_minutes": 2,
            },
        ]
    }
    elevation_index = module.load_segment_elevation_index()

    module.enrich_route_cues_with_segment_elevation([cue], elevation_index)
    effort = module.group_effort_sentence(cue["segments"])
    field_cue = module.official_group_wayfinding_cue(
        seq=1,
        cum_miles=0.0,
        group={"trail_name": "Kemper's Ridge Trail", "segments": cue["segments"]},
        next_group=None,
        first_group=True,
    )
    field_text = " ".join(
        str(field_cue.get(key) or "")
        for key in ("note", "field_warning", "display_detail", "compact")
    )

    assert "170 ft climb" in effort
    assert "482 ft descent" in effort
    assert "Reverse direction would be steep" not in field_text
    assert "482 ft climb over" not in field_text


def test_field_packet_computes_non_official_start_access_gap_from_geometry(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["feature_collections"]["routes"]["features"][0]["geometry"]["coordinates"] = _joined_path(
        _dense_line([-116.1, 43.1], [-116.105, 43.105], 12),
        _dense_line([-116.105, 43.105], [-116.11, 43.11], 12),
        _dense_line([-116.11, 43.11], [-116.12, 43.12], 24),
        _dense_line([-116.12, 43.12], [-116.1, 43.1], 60),
    )
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = _dense_line(
        [-116.105, 43.105], [-116.11, 43.11], 12
    )
    data["route_cues"]["test-route"]["start_access"] = {
        "confidence": "medium",
        "direct_gap_miles": 0,
        "mapped_access_miles": 0,
        "access_class": "direct",
        "graph_validated": True,
    }

    manifest = module.export_field_packet(
        data,
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    route = manifest["routes"][0]
    access_steps = [step for step in route["turn_by_turn_steps"] if step["kind"] == "access"]

    assert route["navigation_quality"]["start_access_gap_miles"] > 0.05
    assert access_steps[0]["title"] == "Start on #99 Access Trail"
    assert "Follow the GPX access line for about" in access_steps[0]["detail"]
    assert "before official credit starts" not in access_steps[0]["detail"]


def test_field_nav_gpx_rejects_unexplained_inter_trkseg_gaps(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["between_links"] = []
    data["route_cues"]["test-route"]["return_to_car"] = {}
    data["feature_collections"]["routes"]["features"][0]["geometry"] = {
        "type": "MultiLineString",
        "coordinates": [
            [[-116.1, 43.1], [-116.101, 43.101]],
            [[-116.2, 43.2], [-116.1, 43.1]],
        ],
    }
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.101, 43.101],
    ]
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = [
        [-116.2, 43.2],
        [-116.1, 43.1],
    ]

    manifest = module.export_field_packet(data, tmp_path)

    assert manifest["summary"]["gpx_validation_passed"] is False
    failures = manifest["routes"][0]["validation"]["failures"]
    assert any(failure["code"] == "unexplained_inter_segment_gap" for failure in failures)


def test_return_to_car_metadata_does_not_explain_unrelated_inter_trkseg_gap(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["between_links"] = []
    data["feature_collections"]["routes"]["features"][0]["geometry"] = {
        "type": "MultiLineString",
        "coordinates": [
            [[-116.1, 43.1], [-116.101, 43.101]],
            [[-116.2, 43.2], [-116.1, 43.1]],
        ],
    }
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.101, 43.101],
    ]
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = [
        [-116.2, 43.2],
        [-116.1, 43.1],
    ]

    manifest = module.export_field_packet(data, tmp_path)

    assert manifest["summary"]["gpx_validation_passed"] is False
    failures = manifest["routes"][0]["validation"]["failures"]
    assert any(failure["code"] == "unexplained_inter_segment_gap" for failure in failures)


def test_field_nav_gpx_rejects_inter_trkseg_gap_even_when_named_connector_is_declared(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["between_links"] = [
        {
            "from_trail": "Test Trail",
            "to_trail": "Second Trail",
            "distance_miles": 0.5,
            "connector_miles": 0.5,
            "connector_names": ["Road Connector"],
            "connector_classes": ["osm_public_road"],
        }
    ]
    data["feature_collections"]["routes"]["features"][0]["geometry"] = {
        "type": "MultiLineString",
        "coordinates": [
            [[-116.1, 43.1], [-116.101, 43.101]],
            [[-116.2, 43.2], [-116.1, 43.1]],
        ],
    }
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.101, 43.101],
    ]
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = [
        [-116.2, 43.2],
        [-116.1, 43.1],
    ]

    manifest = module.export_field_packet(data, tmp_path)

    assert manifest["summary"]["gpx_validation_passed"] is False
    failures = manifest["routes"][0]["validation"]["failures"]
    assert any(failure["code"] == "unexplained_inter_segment_gap" for failure in failures)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert field_data["routes"] == []
    assert "Follow Road Connector toward Second Trail" not in html


def test_field_packet_names_non_official_return_trail_after_last_credit(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    component = data["packages"][0]["components"][0]
    component["trailhead"] = "Test Trailhead"
    component["trail_names"] = ["Test Trail", "Second Trail"]
    data["feature_collections"]["routes"]["features"][0]["geometry"]["coordinates"] = _joined_path(
        _dense_line([-116.1, 43.1], [-116.11, 43.11], 24),
        _dense_line([-116.11, 43.11], [-116.12, 43.12], 24),
        _dense_line([-116.12, 43.12], [-116.13, 43.13], 24),
        _dense_line([-116.13, 43.13], [-116.1, 43.1], 72),
    )
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = _dense_line(
        [-116.11, 43.11], [-116.12, 43.12], 24
    )
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = _dense_line(
        [-116.12, 43.12], [-116.13, 43.13], 24
    )
    route_cue = data["route_cues"]["test-route"]
    route_cue["return_to_car"] = {
        "description": "Route endpoint is already at the start trailhead within geometry tolerance.",
        "official_repeat_miles": 0,
        "connector_miles": 0,
        "road_miles": 0,
    }

    manifest = module.export_field_packet(
        data,
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    route = manifest["routes"][0]
    return_steps = [step for step in route["turn_by_turn_steps"] if step["kind"] == "return"]

    assert route["navigation_quality"]["return_access_gap_miles"] > 0.05
    assert "EXIT" in html
    assert "Follow #99 Access Trail toward Test Trailhead" in html
    assert "UNTIL parked car / trailhead" in html
    assert "Return leg does not count as new official challenge credit." in html
    assert return_steps[0]["title"] == "Return via #99 Access Trail"


def test_field_packet_omits_unknown_water_from_phone_card(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    logistics = data["route_cues"]["test-route"]["logistics"]
    logistics["known_water"] = []
    logistics["car_passes"] = []
    data["route_cues"]["test-route"]["trailhead"]["has_water"] = False
    data["route_cues"]["test-route"]["trailhead"]["water_confidence"] = None

    module.export_field_packet(data, tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert "Known water" not in html
    assert "No verified water in planner data" not in html
    # Honest water disclosure is always rendered now: no fabricated source, but a
    # clear "carry all water" advisory instead of silently omitting the section.
    assert "Field logistics" in html
    assert "No verified on-trail or trailhead water" in html


def test_field_packet_surfaces_r2r_signpost_cues(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    route_cue = data["route_cues"]["test-route"]
    route_cue["segments"][0]["segment_name"] = "Who Now Loop Trail 1"
    route_cue["segments"][0]["trail_name"] = "Who Now Loop Trail"
    route_cue["segments"][1]["segment_name"] = "Hippie Shake Trail 1"
    route_cue["segments"][1]["trail_name"] = "Hippie Shake Trail"
    route_cue["between_links"][0]["from_trail"] = "Who Now Loop Trail"
    route_cue["between_links"][0]["to_trail"] = "Hippie Shake Trail"
    route_cue["between_links"][0]["connector_names"] = [
        "Kemper's Ridge #52",
        "Who Now Loop #51",
    ]

    manifest = module.export_field_packet(data, tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    gpx = Path(manifest["routes"][0]["gpx_path"]).read_text(encoding="utf-8")

    assert "VERIFY: watch for signs: #51 Who Now Loop Trail" in html
    assert "#50 Hippie Shake Trail" in html
    assert "VERIFY: watch for signs: #52 Kemper's Ridge; #51 Who Now Loop" in html
    assert "<h3>Signpost cues</h3>" not in html
    step_details = [step["detail"] for step in public_manifest["routes"][0]["turn_by_turn_steps"]]
    assert any("#51 Who Now Loop Trail" in detail for detail in step_details)
    assert any("Look for signs: #52 Kemper's Ridge; #51 Who Now Loop" in detail for detail in step_details)
    assert "Signpost: #51 Who Now Loop Trail" in gpx


def test_wayfinding_prefers_signposts_over_generic_osm_connector_ids(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    route_cue = data["route_cues"]["test-route"]
    route_cue["between_links"][0]["signpost_labels"] = ["#52 Kemper's Ridge", "#51 Who Now Loop"]
    route_cue["between_links"][0]["connector_names"] = [
        "#52 Kemper's Ridge",
        "OSM path connector 113689",
        "OSM path connector 12787",
        "Who Now Loop #51",
    ]

    public_manifest = module.export_field_packet(data, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    route = public_manifest["routes"][0]
    connector_cues = [
        cue
        for cue in field_data["routes"][0]["wayfinding_cues"]
        if cue.get("cue_type") == "connector_road"
        or cue.get("cue_type") == "connector_named_trail"
    ]

    assert connector_cues
    assert connector_cues[0]["signed_as"] == ["#52 Kemper's Ridge", "#51 Who Now Loop"]
    assert "OSM path connector 113689" not in connector_cues[0]["display_detail"]
    assert "OSM path connector 12787" not in json.dumps(route["wayfinding_cues"])


def test_turn_by_turn_is_trail_navigation_not_segment_credit_order(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["segments"].insert(
        1,
        {
            "order": 2,
            "seg_id": 104,
            "segment_name": "Test Trail 2",
            "trail_name": "Test Trail",
            "official_miles": 0.4,
            "direction_rule": "both",
            "direction_cue": "Either direction allowed.",
        },
    )
    data["route_cues"]["test-route"]["segments"][2]["order"] = 3
    data["packages"][0]["components"][0]["segment_ids"] = [101, 104, 103]

    module.export_field_packet(data, tmp_path)
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    route_steps = public_manifest["routes"][0]["turn_by_turn_steps"]
    step_titles = [step["title"] for step in route_steps]
    step_details = [step["detail"] for step in route_steps]

    assert "Take Test Trail" in step_titles
    assert "Turn onto Second Trail" in step_titles
    assert "Complete Test Trail 1" not in step_titles
    assert "Complete Test Trail 2" not in step_titles
    assert any("This earns: both Test Trail official segments." in detail for detail in step_details)
    assert any("At the signed junction with Second Trail" in detail for detail in step_details)


def test_turn_by_turn_includes_left_right_when_geometry_is_clear(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["logistics"]["car_passes"] = []
    data["feature_collections"]["routes"]["features"][0]["geometry"]["coordinates"] = _joined_path(
        _dense_line([-116.1, 43.1], [-116.1, 43.11], 24),
        _dense_line([-116.1, 43.11], [-116.11, 43.11], 24),
        _dense_line([-116.11, 43.11], [-116.1, 43.1], 36),
    )
    for feature in data["feature_collections"]["official_segments"]["features"]:
        props = feature["properties"]
        if props["seg_id"] == 101:
            feature["geometry"]["coordinates"] = _dense_line([-116.1, 43.1], [-116.1, 43.11], 24)
        if props["seg_id"] == 103:
            feature["geometry"]["coordinates"] = _dense_line([-116.1, 43.11], [-116.11, 43.11], 24)

    module.export_field_packet(data, tmp_path)
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    step_titles = [step["title"] for step in public_manifest["routes"][0]["turn_by_turn_steps"]]
    step_details = [step["detail"] for step in public_manifest["routes"][0]["turn_by_turn_steps"]]

    assert "Turn left onto Second Trail" in step_titles
    assert any("At the signed junction with Second Trail, turn left onto Second Trail." in detail for detail in step_details)


def test_export_field_packet_writes_installable_pwa_artifacts(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    pwa_manifest = json.loads((tmp_path / "manifest.webmanifest").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    service_worker = (tmp_path / "service-worker.js").read_text(encoding="utf-8")

    assert pwa_manifest["name"] == "Boise Trails Field Packet"
    assert pwa_manifest["short_name"] == "Trails Packet"
    assert pwa_manifest["start_url"] == "./"
    assert pwa_manifest["scope"] == "./"
    assert pwa_manifest["display"] == "standalone"
    assert pwa_manifest["theme_color"] == "#111827"
    assert {icon["sizes"] for icon in pwa_manifest["icons"]} >= {"192x192", "512x512"}
    assert (tmp_path / "icons" / "icon-192.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "icons" / "icon-512.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert '<link rel="manifest" href="manifest.webmanifest">' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in html
    assert 'navigator.serviceWorker.register("service-worker.js")' in html
    assert "Add to Home Screen" in html
    assert "Offline-ready" in html

    assert "self.addEventListener('install'" in service_worker
    assert "manifest.webmanifest" in service_worker
    assert manifest["routes"][0]["gpx_href"] in service_worker
    assert manifest["routes"][0]["cue_gpx_href"] in service_worker
    assert manifest["routes"][0]["audit_gpx_href"] in service_worker
    assert "NETWORK_FIRST_URLS" in service_worker
    assert "live-map.html" in service_worker
    assert "field-tool-data.json" in service_worker
    assert "requestUrl.search = ''" in service_worker
    assert "return fetch(event.request).then(response =>" in service_worker
    assert "caches.match(cacheKey)" in service_worker


def test_field_packet_supports_local_progress_filters_and_screenshot_cards(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert '<body class="view-routes">' in html
    assert 'const DEFAULT_VIEW = "routes";' in html
    assert 'data-outing-id="1-1"' in html
    assert 'data-completion-safe="true"' in html
    assert 'data-segment-ids="101 103"' in html
    assert "Mark done" in html
    assert "Undo done" in html
    assert "Pin active" in html
    assert "Clear active" in html
    assert "Hide completed" in html
    assert "Show completed" in html
    assert "Export progress" in html
    assert "boise-trails-progress.json" in html
    assert "fieldPacketCompletedOutings" in html
    assert "completedSegmentSet" in html
    assert "fieldPacketActiveOuting" in html
    assert "active-outing" in html
    assert "localStorage" in html
    assert "Screenshot mode" in html
    assert "Today&apos;s best options" in html
    assert '<button type="button" class="active" data-filter="all">All</button>' in html
    assert '<button type="button" class="active" data-filter="120">' not in html
    assert 'data-filter="60"' in html
    assert 'data-filter="360"' in html
    assert 'id="remaining-segment-count"' in html
    assert "Best today" in html
    assert "completion-safe in the current menu" in html


def test_export_field_packet_writes_public_field_tool_data_for_daily_decisions(tmp_path):
    module = load_exporter()
    map_data = sample_map_data()
    certificate = {
        "certificate_status": "passed",
        "profile": {
            "profile_id": "test-profile",
            "bounds": {
                "weekday_p90_minutes": 120,
                "weekend_p90_minutes": 180,
                "max_on_foot_miles_per_field_day": 18,
            },
        },
        "segment_set": {
            "official_segment_count": 2,
            "selected_calendar_segment_count": 2,
            "missing_segment_count": 0,
        },
        "field_days": {
            "field_day_count": 2,
            "total_p75_minutes": 90,
            "total_on_foot_miles": 3.84,
            "max_on_foot_miles": 2.34,
            "max_p90_minutes": 120,
        },
        "gpx_validation": {
            "day_track_validation_passed": True,
            "actual_max_day_trackpoint_gap_miles": 0.01,
        },
    }

    source_path = tmp_path / "canonical-map-data.json"
    source_path.write_text(json.dumps(map_data), encoding="utf-8")
    module.export_field_packet(
        map_data,
        tmp_path,
        certificate_data=certificate,
        source_metadata=module.source_metadata_for_map_data(map_data, source_path),
    )
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))

    assert field_data["execution_model"]["primary_execution_artifact"] == "route_cards"
    assert field_data["execution_model"]["default_phone_view"] == "routes"
    assert field_data["execution_model"]["route_cards_are_proof_units"] is True
    assert field_data["execution_model"]["field_days_publication_status"] is None
    assert field_data["source"]["canonical_data_role"] == "2026-outing-menu-map-data"
    assert field_data["source"]["source_label"] == "canonical-map-data.json"
    assert field_data["source"]["map_data_sha256"] == module.stable_json_sha256(map_data)
    assert field_data["source"]["source_file_sha256"] == module.file_sha256(source_path)
    assert field_data["time_filters_minutes"] == [60, 90, 120, 180, 240, 360]
    assert field_data["certified_baseline"]["status"] == "passed"
    assert field_data["certified_baseline"]["profile_id"] == "test-profile"
    assert field_data["certified_baseline"]["official_segment_count"] == 2
    assert field_data["certified_baseline"]["covered_segment_count"] == 2
    assert field_data["certified_baseline"]["missing_segment_count"] == 0
    assert field_data["progress"]["remaining_segment_count_at_start"] == 2
    assert field_data["routes"][0]["outing_id"] == "1-1"
    assert field_data["routes"][0]["label"] == "1"
    assert field_data["routes"][0]["route_code"] == "1"
    assert field_data["routes"][0]["route_name"] == "Test Trail"
    assert field_data["routes"][0]["segment_ids"] == ["101", "103"]
    assert field_data["routes"][0]["door_to_door_minutes_p90"] == 59
    assert field_data["routes"][0]["effort"]["ascent_ft"] == 220
    assert field_data["routes"][0]["effort"]["grade_adjusted_miles"] == 1.0
    assert field_data["routes"][0]["effort"]["elevation_source"] == "dem"
    assert field_data["routes"][0]["parking"]["name"] == "Test Trailhead"
    assert field_data["routes"][0]["gpx_href"].startswith("gpx/official/")
    assert field_data["routes"][0]["validation"]["passed"] is True
    assert (
        field_data["routes"][0]["completion_safety"][
            "normal_completion_preserves_remaining_menu_coverage"
        ]
        is True
    )


def special_management_failed_route():
    return {
        "outing_id": "118-1",
        "label": "FD18A",
        "route_code": "FD18A",
        "route_name": "Cartwright / Polecat",
        "outing": {
            "outing_id": "118-1",
            "label": "FD18A",
            "route_code": "FD18A",
            "route_name": "Cartwright / Polecat",
            "trailhead": "Cartwright Trailhead",
            "trails": ["Polecat Loop"],
            "segment_ids": ["1602"],
            "remaining_segment_ids": ["1602"],
            "remaining_segment_count": 1,
            "official_miles": 1.2,
            "on_foot_miles": 2.4,
            "total_minutes": 42,
            "route_card_status": "certified_route_card",
            "packet_visibility": "published",
            "certified_route_card": True,
            "route_card_audit_blockers": [],
        },
        "parking": {"name": "Cartwright Trailhead"},
        "logistics": {"car_passes": [], "known_water": []},
        "validation": {"passed": True, "failures": []},
        "gpx_href": "gpx/official/fd18a.gpx",
        "parking_navigation_url": "https://maps.example.test/cartwright",
        "route_cues": [],
        "wayfinding_cues": [],
        "completion_safety": {
            "normal_completion_preserves_remaining_menu_coverage": True,
            "missing_remaining_segment_ids_after_completion": [],
        },
    }


def special_management_failed_audit(outing_id="118-1", label="FD18A"):
    return {
        "status": "failed",
        "routes": [
            {
                "outing_id": outing_id,
                "label": label,
                "passed": False,
                "checked_segments": [],
                "date_context_required": [],
                "failures": [
                    {
                        "code": "special_management_direction_violated",
                        "rule_id": "r2r-polecat-81-clockwise-through-2026",
                        "segment_id": "1602",
                        "message": "FD18A traverses Polecat Loop 1 counter to the published rule.",
                    }
                ],
            }
        ],
    }


def test_special_management_failures_hold_route_card_and_field_tool_record():
    module = load_exporter()
    route = special_management_failed_route()

    module.apply_special_management_audit_to_routes([route], special_management_failed_audit())

    outing = route["outing"]
    assert route["field_readiness_status"] == "blocked_special_management"
    assert route["validation"]["passed"] is True
    assert outing["route_card_status"] == "blocked_special_management"
    assert outing["packet_visibility"] == "blocked_not_field_ready"
    assert outing["certified_route_card"] is False
    assert "special_management_direction_violated" in outing["route_card_audit_blockers"][0]
    assert "r2r-polecat-81-clockwise-through-2026" in outing["route_card_audit_blockers"][0]

    record = module.route_field_tool_record(route)
    assert record["field_readiness_status"] == "blocked_special_management"
    assert record["field_ready"] is False
    assert record["special_management"]["status"] == "failed"
    assert record["route_card_audit_blockers"] == outing["route_card_audit_blockers"]


def test_navigation_source_anchor_mismatch_holds_route_card_and_field_tool_record():
    module = load_exporter()
    route = special_management_failed_route()
    route["outing_id"] = "112-1"
    route["label"] = "FD12A"
    route["route_code"] = "FD12A"
    route["route_name"] = "Hillside to Hollow: Who Now Loop"
    route["outing"]["outing_id"] = "112-1"
    route["outing"]["label"] = "FD12A"
    route["outing"]["route_code"] = "FD12A"
    route["outing"]["route_name"] = "Hillside to Hollow: Who Now Loop"
    route["logistics"] = {
        "car_passes": [
            {
                "name": "Pass by car again",
                "mile_from_start": 6.32,
                "distance_to_car_miles": 0.0,
            }
        ],
        "known_water": [],
    }
    route["wayfinding_cues"] = [
        {
            "seq": 9,
            "cue_type": "connector_named_trail",
            "leg_miles": 1.16,
            "route_miles": 6.158,
            "route_leg_miles": 4.495,
            "signed_as": ["#53 Buena Vista", "#52 Kemper's Ridge", "#51 Who Now Loop"],
            "target": "Full Sail Trail",
        }
    ]

    module.apply_navigation_source_audit_to_routes([route])

    outing = route["outing"]
    assert route["field_readiness_status"] == "blocked_navigation_source"
    assert route["navigation_source_audit"]["status"] == "failed"
    assert route["navigation_source_audit"]["failures"][0]["cue_seq"] == 9
    assert route["navigation_source_audit"]["failures"][0]["car_pass_miles"] == [6.32]
    assert outing["route_card_status"] == "blocked_navigation_source"
    assert outing["packet_visibility"] == "blocked_not_field_ready"
    assert outing["certified_route_card"] is False
    assert "mid_route_car_pass_anchor_mismatch" in outing["route_card_audit_blockers"][0]

    record = module.route_field_tool_record(route)
    assert record["field_readiness_status"] == "blocked_navigation_source"
    assert record["field_ready"] is False
    assert record["navigation_source_audit"]["status"] == "failed"
    assert record["route_card_audit_blockers"] == outing["route_card_audit_blockers"]

    with pytest.raises(module.FieldPacketCertificationError, match="Cannot render non-field-ready route card 112-1"):
        module.render_card(route)


def test_navigation_source_cue_map_mileage_mismatch_holds_route_card():
    module = load_exporter()
    route = special_management_failed_route()
    route["outing_id"] = "16-4"
    route["label"] = "16C-2"
    route["route_code"] = "16C-2"
    route["route_name"] = "Dry Creek: Shingle Creek"
    route["outing"]["outing_id"] = "16-4"
    route["outing"]["label"] = "16C-2"
    route["outing"]["route_code"] = "16C-2"
    route["outing"]["route_name"] = "Dry Creek: Shingle Creek"
    route["logistics"] = {"car_passes": [], "known_water": []}
    route["wayfinding_cues"] = [
        {
            "seq": 2,
            "cue_type": "follow_official_segment",
            "leg_miles": 7.13,
            "route_miles": 2.406,
            "route_leg_miles": 9.493,
            "signed_as": ["Shingle Creek Trail"],
            "target": "return to car",
        }
    ]

    module.apply_navigation_source_audit_to_routes([route])

    assert route["field_readiness_status"] == "blocked_navigation_source"
    failure = route["navigation_source_audit"]["failures"][0]
    assert failure["code"] == "cue_map_mileage_mismatch"
    assert failure["cue_seq"] == 2
    assert failure["cue_miles"] == 7.13
    assert failure["map_miles"] == 9.493
    assert "cue_map_mileage_mismatch" in route["outing"]["route_card_audit_blockers"][0]


def test_public_route_surfaces_sanitize_private_strava_anchor_display_text():
    module = load_exporter()
    route = special_management_failed_route()
    route["route_name"] = "Dry Creek: Chukar Butte"
    route["outing"]["route_name"] = "Dry Creek: Chukar Butte"
    route["outing"]["trailhead"] = "Chukar Butte private Strava parking anchor"
    route["outing"]["start_justification"] = (
        "Chosen because the private Strava-derived parking anchor is accepted for exact Chukar Butte segments."
    )
    route["parking"]["name"] = "Chukar Butte private Strava parking anchor"

    record = module.route_field_tool_record(route)
    html = module.render_card(route)

    assert record["trailhead"] == "Chukar Butte prior parking anchor"
    assert record["parking"]["name"] == "Chukar Butte prior parking anchor"
    assert "prior parking anchor" in record["start_justification"]
    assert "Strava" not in record["start_justification"]
    assert "<h2>Dry Creek: Chukar Butte</h2>" in html
    assert "Chukar Butte prior parking anchor" in html
    assert "Strava" not in html


def test_render_card_marks_special_management_failure_not_runnable():
    module = load_exporter()
    route = special_management_failed_route()
    module.apply_special_management_audit_to_routes([route], special_management_failed_audit())

    with pytest.raises(module.FieldPacketCertificationError, match="Cannot render non-field-ready route card"):
        module.render_card(route)


def test_render_card_long_route_water_and_heat_annotations():
    module = load_exporter()
    route = special_management_failed_route()
    route["outing"]["on_foot_miles"] = 14.5
    route["outing"]["total_minutes"] = 245
    route["outing"]["official_miles"] = 12.0
    route["logistics"] = {"car_passes": [], "known_water": []}

    html = module.render_card(route)

    # Honest water line is always present, never fabricated.
    assert "No verified on-trail or trailhead water" in html
    assert "Carry all water" in html
    # Derived heat/exposure note for a long exposed outing.
    assert "Long exposed outing" in html
    assert "R2R 6" in html
    # Honest bailout line that claims no confirmed mid-route node.
    assert "no verified mid-route bailout node" in html
    assert "parked car" in html


def test_render_card_short_route_omits_heat_note_but_keeps_bailout():
    module = load_exporter()
    route = special_management_failed_route()
    route["outing"]["on_foot_miles"] = 2.4
    route["outing"]["total_minutes"] = 42
    route["logistics"] = {"car_passes": [], "known_water": []}

    html = module.render_card(route)

    assert "Long exposed outing" not in html
    # Bailout line is honest and always present.
    assert "no verified mid-route bailout node" in html
    # Water disclosure still present.
    assert "No verified on-trail or trailhead water" in html


def test_start_justification_fallback_is_flagged_placeholder():
    module = load_exporter()
    route = special_management_failed_route()
    route["outing"].pop("start_justification", None)

    record = module.route_field_tool_record(route)

    assert record["start_justification_is_placeholder"] is True
    assert record["start_justification"].startswith("PLACEHOLDER_START_JUSTIFICATION_REQUIRED: ")
    # Human-readable text is preserved after the prefix so the card still shows something.
    assert "parking/start anchor" in record["start_justification"]


def test_start_justification_present_is_not_flagged():
    module = load_exporter()
    route = special_management_failed_route()
    route["outing"]["start_justification"] = "Chosen because the field reviewer verified this anchor."

    record = module.route_field_tool_record(route)

    assert record["start_justification_is_placeholder"] is False
    assert "PLACEHOLDER_START_JUSTIFICATION_REQUIRED" not in record["start_justification"]


def test_field_visible_link_names_prefers_real_name_over_osm_id():
    module = load_exporter()
    # Only a generic OSM id in signposts, but a real road name in connectors.
    link = {
        "signpost_labels": ["OSM footway connector 72484"],
        "connector_names": ["8th Street", "OSM footway 72484"],
    }
    assert module.field_visible_link_names(link) == ["8th Street"]

    # Genuinely only an OSM id is usable -> fall back to it rather than nothing.
    osm_only = {"signpost_labels": [], "connector_names": ["OSM footway connector 72484"]}
    assert module.field_visible_link_names(osm_only) == ["OSM footway connector 72484"]


def test_primary_field_visible_names_skips_osm_connector_pattern():
    module = load_exporter()
    assert module.primary_field_visible_names(
        ["OSM path connector 113689", "Camel's Back Trail"]
    ) == ["Camel's Back Trail"]
    assert module.primary_field_visible_names(["OSM path connector 113689"]) == [
        "OSM path connector 113689"
    ]


def test_special_management_failure_propagates_to_field_day_layer_summary():
    module = load_exporter()
    route = special_management_failed_route()
    route["outing"]["outing_id"] = "1-1"
    route["outing_id"] = "1-1"
    route["outing"]["label"] = "1"
    route["label"] = "1"
    route["outing"]["candidate_ids"] = ["test-route"]
    module.apply_special_management_audit_to_routes([route], special_management_failed_audit("1-1", "1"))
    field_day_layer = module.public_field_day_layer_record(sample_field_day_layer())

    module.apply_route_names_to_field_day_layer(field_day_layer, [route])

    loop = field_day_layer["field_days"][0]["loops"][0]
    assert loop["certification_status"] == "blocked_special_management"
    assert loop["route_card_status"] == "blocked_special_management"
    assert loop["certified_route_card"] is False
    assert loop["route_card_ref"]["field_readiness_status"] == "blocked_special_management"
    assert field_day_layer["summary"]["certified_route_card_loop_count"] == 0
    assert field_day_layer["summary"]["needs_route_card_audit_fix_loop_count"] == 1
    assert field_day_layer["publication_status"] == "blocked_by_special_management"


def test_live_map_keeps_blocked_routes_unavailable_without_source_mismatch_copy():
    module = load_exporter()

    live_map_html = module.render_live_map_html()

    assert 'id="route-blocked-warning"' in live_map_html
    assert "function routeUnavailable(route)" in live_map_html
    assert 'startsWith("blocked_")' in live_map_html
    assert "source route/cue geometry mismatch" not in live_map_html
    assert "blocked_navigation_source" not in live_map_html
    assert "special-management rule violation" in live_map_html
    assert "locateButton.disabled = routeUnavailable(state.route)" in live_map_html
    assert "Route unavailable" in live_map_html
    assert "state.routes.map(route => {)" not in live_map_html
    assert "state.routes.map(route => {{" not in live_map_html
    assert "${{" not in live_map_html


def test_segment_ownership_reconciliation_declares_cross_route_ownership():
    module = load_exporter()
    data = sample_map_data()
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = _dense_line(
        [-116.1, 43.1], [-116.11, 43.11], 24
    )
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = _dense_line(
        [-116.11, 43.11], [-116.12, 43.12], 24
    )
    segments_by_id = module.official_segment_index(data)
    routes = [
        {
            "outing_id": "1-1",
            "label": "1",
            "outing": {
                "outing_id": "1-1",
                "label": "1",
                "candidate_ids": ["test-route"],
                "remaining_segment_ids": [101],
            },
            "segment_ids": ["101"],
            "_track_segments": [
                _joined_path(
                    _dense_line([-116.1, 43.1], [-116.11, 43.11], 24),
                    _dense_line([-116.11, 43.11], [-116.12, 43.12], 24),
                    _dense_line([-116.12, 43.12], [-116.1, 43.1], 48),
                )
            ],
        },
        {
            "outing_id": "2-1",
            "label": "2",
            "outing": {
                "outing_id": "2-1",
                "label": "2",
                "candidate_ids": ["second-route"],
                "remaining_segment_ids": [103],
            },
            "segment_ids": ["103"],
            "_track_segments": [[(-116.11, 43.11), (-116.12, 43.12)]],
        },
    ]

    module.apply_segment_ownership_reconciliation(routes, segments_by_id)
    first_route = routes[0]
    reconciliation = first_route["segment_ownership_reconciliation"]

    assert first_route["segment_ids"] == ["101"]
    assert reconciliation["status"] == "reconciled"
    assert reconciliation["declared_owned_elsewhere_segment_ids"] == ["103"]
    assert reconciliation["segments_owned_elsewhere"][0]["owned_by_routes"][0]["outing_id"] == "2-1"


def test_route_claim_index_includes_manual_holds_for_ownership_only():
    module = load_exporter()

    claims = module.route_claim_index(
        [
            {
                "outing": {
                    "outing_id": "1-1",
                    "label": "Runnable",
                    "segment_ids": [101],
                    "candidate_ids": ["runnable-candidate"],
                }
            }
        ],
        ownership_cards=[
            {
                "outing_id": "held-1",
                "label": "Held",
                "manual_design_hold": True,
                "remaining_segment_ids": [102],
                "candidate_ids": ["held-candidate"],
            }
        ],
    )

    assert claims["101"][0]["ownership_status"] == "active_route"
    assert claims["102"][0]["outing_id"] == "held-1"
    assert claims["102"][0]["candidate_ids"] == ["held-candidate"]
    assert claims["102"][0]["ownership_status"] == "manual_hold"


def test_field_packet_uses_route_level_dem_effort_when_segment_effort_is_missing(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    cue = data["route_cues"]["test-route"]
    cue["effort"] = {
        "ascent_ft": 640,
        "descent_ft": 420,
        "grade_adjusted_miles": 3.88,
        "elevation_source": "dem",
    }
    cue["time_estimates_minutes"] = {
        "door_to_door_p75": 45,
        "door_to_door_p90": 59,
        "moving_effort_p50": 34,
        "moving_effort_p75": 42,
    }
    for segment in cue["segments"]:
        segment.pop("ascent_ft", None)
        segment.pop("descent_ft", None)
        segment.pop("grade_adjusted_miles", None)
        segment.pop("estimated_moving_minutes", None)
        segment.pop("estimated_moving_minutes_p75", None)

    module.export_field_packet(data, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    effort = field_data["routes"][0]["effort"]

    assert effort["ascent_ft"] == 640
    assert effort["descent_ft"] == 420
    assert effort["grade_adjusted_miles"] == 3.88
    assert effort["estimated_moving_minutes_p50"] == 34
    assert effort["estimated_moving_minutes_p75"] == 42
    assert effort["elevation_source"] == "dem"
    assert "<b>Climb</b><strong>640 ft</strong>" in html


def test_field_packet_falls_back_to_p50_moving_effort_when_p75_is_missing(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    cue = data["route_cues"]["test-route"]
    for segment in cue["segments"]:
        segment["estimated_moving_minutes"] = 31
        segment.pop("estimated_moving_minutes_p75", None)

    module.export_field_packet(data, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    effort = field_data["routes"][0]["effort"]

    assert effort["estimated_moving_minutes_p50"] == 62
    assert effort["estimated_moving_minutes_p75"] == 62


def test_wayfinding_cues_use_gpx_route_miles_and_warn_on_off_label_connectors(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["packages"][0]["components"][0]["candidate_id"] = "off-label-route"
    data["packages"][0]["components"][0]["trail_names"] = ["Main Trail", "Next Trail"]
    data["packages"][0]["components"][0]["segment_ids"] = [201, 203, 204]
    data["feature_collections"]["routes"]["features"][0]["properties"]["candidate_id"] = "off-label-route"
    data["feature_collections"]["routes"]["features"][0]["geometry"]["coordinates"] = _joined_path(
        _dense_line([-116.0, 43.0], [-116.001, 43.0], 4),
        _dense_line([-116.001, 43.0], [-116.002, 43.0], 4),
        _dense_line([-116.002, 43.0], [-116.001, 43.0], 4),
        _dense_line([-116.001, 43.0], [-116.0, 43.0], 4),
        _dense_line([-116.0, 43.0], [-116.0, 42.997], 12),
        _dense_line([-116.0, 42.997], [-116.0, 42.995], 8),
    )
    data["feature_collections"]["parking"]["features"][0]["properties"]["candidate_id"] = "off-label-route"
    data["feature_collections"]["parking"]["features"][0]["geometry"]["coordinates"] = [-116.0, 43.0]
    data["feature_collections"]["official_segments"]["features"] = [
        {
            "type": "Feature",
            "properties": {
                "seg_id": 201,
                "segment_name": "Main Trail 1",
                "seg_name": "Main Trail 1",
                "trail_name": "Main Trail",
                "LengthFt": 300,
            },
            "geometry": {"type": "LineString", "coordinates": _dense_line([-116.001, 43.0], [-116.002, 43.0], 4)},
        },
        {
            "type": "Feature",
            "properties": {
                "seg_id": 202,
                "segment_name": "Connector Trail 1",
                "seg_name": "Connector Trail 1",
                "trail_name": "Connector Trail",
                "LengthFt": 300,
            },
            "geometry": {"type": "LineString", "coordinates": _dense_line([-116.0, 43.0], [-116.001, 43.0], 4)},
        },
        {
            "type": "Feature",
            "properties": {
                "seg_id": 203,
                "segment_name": "Main Trail 2",
                "seg_name": "Main Trail 2",
                "trail_name": "Main Trail",
                "LengthFt": 1100,
            },
            "geometry": {"type": "LineString", "coordinates": _dense_line([-116.0, 43.0], [-116.0, 42.997], 12)},
        },
        {
            "type": "Feature",
            "properties": {
                "seg_id": 204,
                "segment_name": "Next Trail 1",
                "seg_name": "Next Trail 1",
                "trail_name": "Next Trail",
                "LengthFt": 700,
            },
            "geometry": {"type": "LineString", "coordinates": _dense_line([-116.0, 42.997], [-116.0, 42.995], 8)},
        },
    ]
    data["route_cues"] = {
        "off-label-route": {
            "candidate_id": "off-label-route",
            "title": "Off-label connector route",
            "official_miles": 0.4,
            "on_foot_miles": 0.6,
            "total_minutes": 30,
            "time_estimates_minutes": {"door_to_door_p75": 30, "door_to_door_p90": 40},
            "trailhead": {"name": "Test Trailhead", "lat": 43.0, "lon": -116.0, "has_parking": True},
            "start_access": {"mapped_access_miles": 0},
            "segments": [
                {
                    "order": 1,
                    "seg_id": 201,
                    "segment_name": "Main Trail 1",
                    "trail_name": "Main Trail",
                    "official_miles": 0.06,
                    "direction_rule": "both",
                },
                {
                    "order": 2,
                    "seg_id": 203,
                    "segment_name": "Main Trail 2",
                    "trail_name": "Main Trail",
                    "official_miles": 0.21,
                    "direction_rule": "both",
                },
                {
                    "order": 3,
                    "seg_id": 204,
                    "segment_name": "Next Trail 1",
                    "trail_name": "Next Trail",
                    "official_miles": 0.13,
                    "direction_rule": "both",
                },
            ],
        }
    }

    module.export_field_packet(data, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    main_cue = next(cue for cue in field_data["routes"][0]["wayfinding_cues"] if cue.get("official_segment_ids") == ["201", "203"])
    live_map = (tmp_path / "live-map.html").read_text(encoding="utf-8")

    assert main_cue["route_miles"] <= 0.03
    assert main_cue["route_leg_miles"] > main_cue.get("source_leg_miles", main_cue["leg_miles"])
    assert "Connector Trail 1" in main_cue["field_warning"]
    assert "cue?.route_miles" in live_map
    assert "cue?.route_leg_miles" in live_map


def test_non_credit_cue_route_interval_uses_source_path_endpoints():
    module = load_exporter()
    a = (-116.000, 43.000)
    b = (-116.001, 43.000)
    c = (-116.002, 43.000)
    d = (-116.003, 43.000)
    e = (-116.004, 43.000)
    official_one_miles = module.haversine_miles(a, b)
    connector_miles = module.haversine_miles(b, c)
    skipped_miles = module.haversine_miles(c, d)
    official_two_miles = module.haversine_miles(d, e)
    route = {
        "_track_segments": [[a, b, c, d, e]],
        "_official_segment_index": {
            "1": {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [a, b]}},
            "2": {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [d, e]}},
        },
        "wayfinding_cues": [
            module.make_wayfinding_cue(
                seq=1,
                cum_miles=0,
                leg_miles=official_one_miles,
                cue_type="follow_official_segment",
                action="FOLLOW",
                signed_as=["Official One"],
                official_segment_ids=[1],
            ),
            module.make_wayfinding_cue(
                seq=2,
                cum_miles=official_one_miles,
                leg_miles=connector_miles,
                cue_type="connector_named_trail",
                action="FOLLOW",
                signed_as=["Connector"],
                target="Official Two",
                source_path_start=b,
                source_path_end=c,
            ),
            module.make_wayfinding_cue(
                seq=3,
                cum_miles=official_one_miles + connector_miles + skipped_miles,
                leg_miles=official_two_miles,
                cue_type="junction_turn",
                action="TAKE",
                signed_as=["Official Two"],
                official_segment_ids=[2],
            ),
        ],
    }

    module.assign_wayfinding_route_miles(route)

    connector = route["wayfinding_cues"][1]
    assert connector["route_leg_miles"] == pytest.approx(connector_miles, abs=0.005)
    assert connector["route_leg_miles"] < connector_miles + skipped_miles - 0.005


def test_final_non_credit_cue_extends_to_actual_route_end_even_with_short_source_path():
    module = load_exporter()
    a = (-116.000, 43.000)
    b = (-116.001, 43.000)
    c = (-116.002, 43.000)
    d = (-116.003, 43.000)
    official_miles = module.haversine_miles(a, b)
    short_return_miles = module.haversine_miles(b, c)
    route_total = module.track_distance_miles([[a, b, c, d]])
    route = {
        "_track_segments": [[a, b, c, d]],
        "_official_segment_index": {
            "1": {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [a, b]}},
        },
        "wayfinding_cues": [
            module.make_wayfinding_cue(
                seq=1,
                cum_miles=0,
                leg_miles=official_miles,
                cue_type="follow_official_segment",
                action="FOLLOW",
                signed_as=["Official One"],
                official_segment_ids=[1],
            ),
            module.make_wayfinding_cue(
                seq=2,
                cum_miles=official_miles,
                leg_miles=short_return_miles,
                cue_type="return_to_car",
                action="FOLLOW",
                signed_as=["Connector"],
                target="Trailhead",
                source_path_start=b,
                source_path_end=c,
            ),
        ],
    }

    module.assign_wayfinding_route_miles(route)

    final_cue = route["wayfinding_cues"][-1]
    assert final_cue["route_miles"] + final_cue["route_leg_miles"] == pytest.approx(route_total, abs=0.005)


def test_track_segments_for_route_cues_follow_cue_source_order():
    module = load_exporter()
    first = [(-116.000, 43.000), (-116.001, 43.000)]
    connector = [(-116.001, 43.000), (-116.002, 43.000)]
    second = [(-116.002, 43.000), (-116.003, 43.000)]
    return_path = [(-116.003, 43.000), (-116.004, 43.000)]

    track_segments = module.track_segments_for_route_cues(
        [
            {
                "segments": [
                    {"seg_id": 1, "trail_name": "First", "coordinates": first},
                    {"seg_id": 2, "trail_name": "Second", "coordinates": second},
                ],
                "between_links": [{"path_coordinates": connector}],
                "return_to_car": {"path_coordinates": return_path},
            }
        ]
    )

    assert track_segments == [[first[0], first[1], connector[1], second[1], return_path[1]]]


def test_select_track_segments_falls_back_when_cue_track_is_not_car_to_car():
    module = load_exporter()
    parking = {"lon": -116.000, "lat": 43.000}
    cue_track = [[(-116.010, 43.000), (-116.011, 43.000)]]
    feature_track = [[(-116.000, 43.000), (-116.001, 43.000), (-116.000, 43.000)]]

    selected, source = module.select_track_segments_for_outing(
        cue_track,
        feature_track,
        parking,
        max_parking_gap_miles=0.05,
    )

    assert selected == feature_track
    assert source == "route_feature"


def test_select_track_segments_prefers_car_to_car_route_feature_over_cue_accounting_track():
    module = load_exporter()
    parking = {"lon": -116.000, "lat": 43.000}
    cue_track = [[(-116.000, 43.000), (-116.001, 43.000), (-116.000, 43.000)]]
    feature_track = [[(-116.000, 43.000), (-116.002, 43.000), (-116.000, 43.000)]]

    selected, source = module.select_track_segments_for_outing(
        cue_track,
        feature_track,
        parking,
        max_parking_gap_miles=0.05,
    )

    assert selected == feature_track
    assert source == "route_feature"


def test_make_wayfinding_cue_preserves_source_path_coordinates():
    module = load_exporter()

    cue = module.make_wayfinding_cue(
        seq=1,
        cum_miles=0,
        leg_miles=0.1,
        cue_type="connector_named_trail",
        action="FOLLOW",
        signed_as=["Connector"],
        official_miles=0.1,
        source_path_coordinates=[[-116.0, 43.0], [-116.001, 43.0]],
    )

    assert cue["source_path_coordinates"] == [(-116.0, 43.0), (-116.001, 43.0)]
    assert cue["official_miles"] == 0.1


def test_apply_shortest_connector_repairs_rewrites_long_link():
    module = load_exporter()
    start = (-116.0, 43.0)
    end = (-115.99, 43.0)
    graph = {
        "nodes": [start, end],
        "graph": {
            start: [
                {
                    "to": end,
                    "distance": 0.25,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
            end: [
                {
                    "to": start,
                    "distance": 0.25,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
        },
    }
    cue_list = [
        {
            "between_links": [
                {
                    "source": "mapped_graph",
                    "distance_miles": 0.5,
                    "connector_miles": 0.5,
                    "connector_edges": [{"name": "Long Connector"}],
                    "path_start": list(start),
                    "path_end": list(end),
                    "path_coordinates": [list(start), [-115.995, 43.005], list(end)],
                    "avoided_unearned_segment_ids": [101],
                }
            ]
        }
    ]

    repair_count = module.apply_shortest_connector_repairs(cue_list, graph, snap_tolerance_miles=0.01)

    link = cue_list[0]["between_links"][0]
    assert repair_count == 1
    assert link["source"] == "shortest_connector_repair"
    assert link["distance_miles"] == 0.25
    assert link["shortest_repair_savings_miles"] == 0.25


def test_apply_shortest_connector_repairs_rehydrates_zero_mile_link_with_path_geometry():
    module = load_exporter()
    start = (-116.0, 43.0)
    end = (-115.99, 43.0)
    graph = {
        "nodes": [start, end],
        "graph": {
            start: [
                {
                    "to": end,
                    "distance": 0.25,
                    "name": "Mapped Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
            end: [],
        },
    }
    cue_list = [
        {
            "between_links": [
                {
                    "source": "mapped_graph",
                    "distance_miles": 0,
                    "connector_miles": 0,
                    "path_start": list(start),
                    "path_end": list(end),
                    "path_coordinates": [list(start), [-115.995, 43.0], list(end)],
                }
            ]
        }
    ]

    repair_count = module.apply_shortest_connector_repairs(cue_list, graph, snap_tolerance_miles=0.01)

    link = cue_list[0]["between_links"][0]
    assert repair_count == 1
    assert link["source"] == "shortest_connector_repair"
    assert link["connector_proof_rehydrated"] is True
    assert link["connector_edges"]
    assert link["connector_names"] == ["Mapped Connector"]


def test_apply_shortest_connector_repairs_uses_unproved_path_geometry_over_stale_miles():
    module = load_exporter()
    start = (-116.0, 43.0)
    end = (-115.99, 43.0)
    graph = {
        "nodes": [start, end],
        "graph": {
            start: [
                {
                    "to": end,
                    "distance": 0.25,
                    "name": "Mapped Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
            end: [],
        },
    }
    cue_list = [
        {
            "return_to_car": {
                "leg_miles": 0.1,
                "path_start": list(start),
                "path_end": list(end),
                "path_coordinates": [list(start), [-115.995, 43.01], list(end)],
            }
        }
    ]

    repair_count = module.apply_shortest_connector_repairs(cue_list, graph, snap_tolerance_miles=0.01)

    link = cue_list[0]["return_to_car"]
    assert repair_count == 1
    assert link["source"] == "shortest_connector_repair"
    assert link["distance_miles"] == 0.25
    assert link["connector_proof_rehydrated"] is True
    assert link["connector_edges"]
    assert link["shortest_repair_savings_miles"] > 0.1


def test_return_wayfinding_cue_uses_total_repaired_distance():
    module = load_exporter()

    cue = module.return_wayfinding_cue(
        seq=1,
        cum_miles=2.0,
        parking={"name": "Parking", "nearest_open_trail_name": "Connector"},
        last_trail="Trail",
        return_to_car={
            "distance_miles": 1.26,
            "connector_miles": 1.07,
            "official_repeat_miles": 0.19,
            "official_repeat_segment_ids": [10],
            "connector_names": ["Connector"],
        },
        return_access_gap_miles=0.0,
    )

    assert cue["leg_miles"] == pytest.approx(1.26)
    assert cue["official_repeat_miles"] == pytest.approx(0.19)


def test_apply_shortest_repairs_to_wayfinding_cues_uses_route_interval_endpoints():
    module = load_exporter()
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = module.haversine_miles(start, mid)
    connector_leg = module.haversine_miles(mid, end)
    graph = {
        "nodes": [mid, end],
        "graph": {
            mid: [
                {
                    "to": end,
                    "distance": 0.25,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
            end: [
                {
                    "to": mid,
                    "distance": 0.25,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
        },
    }
    route = {
        "segment_ids": ["101", "102"],
        "_track_segments": [[start, mid, end]],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": ["101"],
                "route_miles": 0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "exit_access",
                "leg_miles": 0.5,
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
                "signed_as": ["Long Return"],
            },
        ],
    }

    repair_count = module.apply_shortest_repairs_to_wayfinding_cues(route, graph, snap_tolerance_miles=0.01)

    cue = route["wayfinding_cues"][1]
    assert repair_count == 1
    assert cue["leg_miles"] == 0.25
    assert cue["source_path_coordinates"][0] == [mid[0], mid[1]]
    assert "shortest_repair_savings_miles" not in cue
    assert cue["applied_shortest_repair_savings_miles"] == pytest.approx(
        module.haversine_miles(mid, end) - 0.25,
        abs=0.005,
    )


def test_apply_shortest_repairs_to_wayfinding_cues_falls_back_when_route_leg_collapses():
    module = load_exporter()
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = module.haversine_miles(start, mid)
    graph = {
        "nodes": [mid, end],
        "graph": {
            mid: [
                {
                    "to": end,
                    "distance": 0.25,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
            end: [
                {
                    "to": mid,
                    "distance": 0.25,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                }
            ],
        },
    }
    route = {
        "segment_ids": ["101", "102"],
        "_track_segments": [[start, mid, end]],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": ["101"],
                "route_miles": 0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "exit_access",
                "leg_miles": 0.5,
                "route_miles": first_leg,
                "route_leg_miles": 0,
                "signed_as": ["Long Return"],
            },
        ],
    }

    repair_count = module.apply_shortest_repairs_to_wayfinding_cues(route, graph, snap_tolerance_miles=0.01)

    assert repair_count == 1
    assert route["wayfinding_cues"][1]["leg_miles"] == pytest.approx(0.26, abs=0.005)


def test_apply_shortest_repairs_to_wayfinding_cues_prices_stale_source_mileage_from_geometry():
    module = load_exporter()
    start = (-116.0, 43.0)
    detour = (-116.0, 43.01)
    end = (-115.99, 43.0)
    graph = {
        "nodes": [start, end],
        "graph": {
            start: [
                {
                    "to": end,
                    "distance": 0.1,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                    "coordinates": [start, end],
                }
            ],
            end: [
                {
                    "to": start,
                    "distance": 0.1,
                    "name": "Short Connector",
                    "edge_type": "connector",
                    "connector_class": "test_connector",
                    "source": "test",
                    "coordinates": [end, start],
                }
            ],
        },
    }
    route = {
        "segment_ids": ["101", "102"],
        "_track_segments": [[(-116.01, 43.0), start, detour, end]],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": ["101"],
                "route_miles": 0,
                "route_leg_miles": 0.1,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "leg_miles": 0.05,
                "source_leg_miles": 0.05,
                "source_path_coordinates": [start, detour, end],
                "signed_as": ["Long Connector"],
            },
        ],
    }

    repair_count = module.apply_shortest_repairs_to_wayfinding_cues(route, graph, snap_tolerance_miles=0.01)

    cue = route["wayfinding_cues"][1]
    assert repair_count == 1
    assert cue["leg_miles"] == 0.1
    assert "shortest_repair_savings_miles" not in cue
    assert cue["applied_shortest_repair_savings_miles"] > 0.05


def test_live_map_wayfinding_cues_keeps_phone_cues_and_starts_on_first_movement_cue():
    module = load_exporter()
    route = {
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "start_access",
                "action": "FOLLOW",
                "signed_as": ["Access Trail"],
                "target": "First Trail",
                "route_miles": 0,
                "route_leg_miles": 0,
                "leg_miles": 0,
            },
            {
                "seq": 2,
                "cue_type": "follow_official_segment",
                "action": "FOLLOW",
                "signed_as": ["First Trail"],
                "target": "Next Trail",
                "route_miles": 0,
                "route_leg_miles": 1.2,
            },
        ],
    }

    live_cues = module.live_map_wayfinding_cues(route)

    assert [cue["seq"] for cue in route["wayfinding_cues"]] == [1, 2]
    assert route["wayfinding_cues"][0]["cue_type"] == "start_access"
    assert [cue["seq"] for cue in live_cues] == [1]
    first = live_cues[0]
    assert first["cue_type"] == "follow_official_segment"
    assert first["route_miles"] == 0
    assert first["merged_start_marker"]["signed_as"] == ["Access Trail"]


def test_return_wayfinding_cue_ignores_gap_for_closed_loop_return():
    module = load_exporter()

    cue = module.return_wayfinding_cue(
        seq=1,
        cum_miles=0,
        parking={"name": "Trailhead"},
        last_trail="Loop Trail",
        return_to_car={
            "strategy": "closed_loop",
            "connector_miles": 0,
            "road_miles": 0,
            "official_repeat_miles": 0,
        },
        return_access_gap_miles=0.69,
    )

    assert "leg_miles" not in cue
    assert cue["cue_type"] == "return_to_car"


def test_export_field_packet_does_not_promote_phone_outing_taps_to_segment_progress(tmp_path):
    module = load_exporter()
    data = sample_map_data()

    updated = module.apply_progress_to_map_data(
        data,
        {"completed_outing_ids": ["1-1"], "missed_segment_ids": ["103"]},
    )

    assert updated["progress"]["completed_segment_ids"] == []
    assert updated["progress"]["provisional_completed_outing_ids"] == ["1-1"]
    assert updated["progress"]["missed_segment_ids"] == [103]
    assert data["progress"]["completed_segment_ids"] == []

    manifest = module.export_field_packet(updated, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))

    assert manifest["summary"]["runnable_outing_count"] == 1
    assert field_data["progress"]["completed_segment_ids_at_export"] == []
    assert field_data["progress"]["remaining_segment_count_at_start"] == 2


def test_export_field_packet_applies_validated_segment_progress_before_building_remaining_menu(tmp_path):
    module = load_exporter()
    data = sample_map_data()

    updated = module.apply_progress_to_map_data(
        data,
        {"completed_segment_ids": ["101"], "extra_completed_segment_ids": ["103"]},
    )

    assert updated["progress"]["completed_segment_ids"] == [101, 103]
    assert data["progress"]["completed_segment_ids"] == []


def test_field_packet_treats_completed_official_geometry_as_repeat_not_new_credit(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    # The mid-route car_pass anchor no longer matches the (stricter-validated)
    # synthetic geometry, which would otherwise drop the route via
    # blocked_navigation_source. Clear it to match the sibling routes so the
    # route survives and the repeat-vs-credit assertions below can run.
    data["route_cues"]["test-route"]["logistics"]["car_passes"] = []
    data["progress"]["completed_segment_ids"] = [101]

    module.export_field_packet(data, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    route = field_data["routes"][0]
    cue_segment_ids = {
        segment_id
        for cue in route["wayfinding_cues"]
        for segment_id in cue.get("official_segment_ids") or []
    }

    assert route["segment_ids"] == ["103"]
    assert cue_segment_ids == {"103"}
    assert set(route["segment_direction_evidence"]) == {"103"}
    assert "This earns: Test Trail segment 1" not in html
    assert "Official-repeat mileage: Test Trail segment 1; do not count as new credit." in html
    assert "This earns: Second Trail segment 1" in html


def test_segment_direction_evidence_prefers_special_management_override(monkeypatch):
    module = load_exporter()
    monkeypatch.setattr(
        module,
        "special_management_segment_direction_overrides",
        lambda: {"1490": "forward"},
    )
    route = {
        "outing": {"remaining_segment_ids": ["1490"]},
        "route_cues": [
            {
                "segments": [
                    {
                        "seg_id": 1490,
                        "direction_rule": "ascent",
                        "direction_cue": "ASCENT REQUIRED: follow map arrows opposite official geometry.",
                    }
                ]
            }
        ],
    }

    evidence = module.segment_direction_evidence_for_route(route)

    assert evidence["1490"]["allowed_geometry_direction"] == "forward"


def test_export_field_packet_writes_downloadable_gpx_zip_and_precaches_it(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    service_worker = (tmp_path / "service-worker.js").read_text(encoding="utf-8")
    zip_path = tmp_path / "gpx" / "all-field-packet-gpx.zip"

    assert zip_path.exists()
    assert "Download all GPX" not in html
    assert manifest["summary"]["gpx_zip_href"] == "gpx/all-field-packet-gpx.zip"
    assert "gpx/all-field-packet-gpx.zip" in service_worker
    with zipfile.ZipFile(zip_path) as archive:
        gpx_names = [name for name in archive.namelist() if name.endswith(".gpx")]
    assert sorted(gpx_names) == sorted(
        [
            manifest["routes"][0]["gpx_href"].removeprefix("gpx/"),
            manifest["routes"][0]["cue_gpx_href"].removeprefix("gpx/"),
            manifest["routes"][0]["audit_gpx_href"].removeprefix("gpx/"),
        ]
    )


def test_field_packet_public_outputs_do_not_leak_private_origin_or_paths(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in tmp_path.rglob("*") if path.is_file()
    )

    assert "/Users/scott" not in combined
    assert "outputs/private" not in combined
    assert str(0x38F) not in combined
    assert f"{int('10010', 2)}th" not in combined


def test_densify_track_segments_reduces_point_gaps_below_limit():
    module = load_exporter()
    sparse = [[(-116.1, 43.1), (-116.2, 43.2)]]

    dense = module.densify_track_segments(sparse, max_gap_miles=0.05)
    validation = module.validate_track_segments(dense, max_gap_miles=0.05)

    assert len(dense[0]) > 2
    assert validation["passed"] is True
