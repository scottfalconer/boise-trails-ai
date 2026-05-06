import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_execution_gpx.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_execution_gpx", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_track_coordinates_honor_direction_and_return_path():
    exporter = load_exporter()
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "First",
            "direction": "both",
            "coordinates": [(-116.10, 43.10), (-116.11, 43.11)],
        },
        2: {
            "seg_id": 2,
            "trail_name": "Second",
            "direction": "ascent",
            "coordinates": [(-116.20, 43.20), (-116.21, 43.21)],
        },
    }
    candidate = {
        "candidate_id": "test-route",
        "segments": [
            {"seg_id": 1, "trail_name": "First"},
            {"seg_id": 2, "trail_name": "Second"},
        ],
        "route_orientation": {"direction": "forward"},
        "direction_validation": {
            "planned_traversal_direction": {"2": "official_geometry_end_to_start"}
        },
        "between_trail_links": {
            "links": [
                {
                    "path_coordinates": [
                        [-116.11, 43.11],
                        [-116.15, 43.15],
                        [-116.21, 43.21],
                    ]
                }
            ]
        },
        "return_to_car": {
            "path_coordinates": [[-116.20, 43.20], [-116.10, 43.10]],
        },
    }

    coords = exporter.candidate_track_coordinates(candidate, official_index)

    assert coords == [
        (-116.10, 43.10),
        (-116.11, 43.11),
        (-116.15, 43.15),
        (-116.21, 43.21),
        (-116.20, 43.20),
        (-116.10, 43.10),
    ]


def test_candidate_track_coordinates_include_trailhead_access_out_and_back():
    exporter = load_exporter()
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Access",
            "direction": "both",
            "coordinates": [(-116.10, 43.10), (-116.11, 43.11)],
        }
    }
    candidate = {
        "candidate_id": "access-route",
        "segments": [{"seg_id": 1, "trail_name": "Access"}],
        "route_orientation": {"direction": "forward"},
        "direction_validation": {"planned_traversal_direction": {}},
        "trailhead_access": {
            "outbound_path_coordinates": [
                [-116.08, 43.08],
                [-116.10, 43.10],
            ],
            "return_path_coordinates": [
                [-116.10, 43.10],
                [-116.08, 43.08],
            ],
        },
        "return_to_car": {
            "path_coordinates": [[-116.11, 43.11], [-116.10, 43.10]],
        },
    }

    coords = exporter.candidate_track_coordinates(candidate, official_index)

    assert coords == [
        (-116.08, 43.08),
        (-116.10, 43.10),
        (-116.11, 43.11),
        (-116.10, 43.10),
        (-116.08, 43.08),
    ]


def test_candidate_track_coordinates_skips_stale_return_path_after_reversed_ascent_segment():
    exporter = load_exporter()
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Ascent",
            "direction": "ascent",
            "coordinates": [(-116.10, 43.10), (-116.11, 43.11)],
        }
    }
    candidate = {
        "candidate_id": "reversed-ascent-access-route",
        "segments": [{"seg_id": 1, "trail_name": "Ascent"}],
        "route_orientation": {"direction": "forward"},
        "direction_validation": {
            "planned_traversal_direction": {"1": "official_geometry_end_to_start"}
        },
        "trailhead_access": {
            "outbound_path_coordinates": [
                [-116.08, 43.08],
                [-116.11, 43.11],
            ],
            "return_path_coordinates": [
                [-116.10, 43.10],
                [-116.08, 43.08],
            ],
        },
        # This path was built before the ascent-only reversal. Once the actual
        # traversal ends at -116.10/43.10, the route is already at the access
        # return point and should not jump back to -116.11/43.11.
        "return_to_car": {
            "path_coordinates": [[-116.11, 43.11], [-116.10, 43.10]],
        },
    }

    coords = exporter.candidate_track_coordinates(candidate, official_index)

    assert coords == [
        (-116.08, 43.08),
        (-116.11, 43.11),
        (-116.10, 43.10),
        (-116.08, 43.08),
    ]


def test_candidate_track_coordinates_reverse_bidirectional_candidate():
    exporter = load_exporter()
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "First",
            "direction": "both",
            "coordinates": [(-116.10, 43.10), (-116.11, 43.11)],
        }
    }
    candidate = {
        "candidate_id": "reversed-route",
        "segments": [{"seg_id": 1, "trail_name": "First"}],
        "route_orientation": {"direction": "reversed"},
        "direction_validation": {"planned_traversal_direction": {}},
        "return_to_car": {},
    }

    coords = exporter.candidate_track_coordinates(candidate, official_index)

    assert coords == [(-116.11, 43.11), (-116.10, 43.10)]


def test_candidate_track_coordinates_insert_graph_connector_for_segment_gap():
    exporter = load_exporter()
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Gap",
            "direction": "both",
            "coordinates": [(-116.10, 43.10), (-116.11, 43.11)],
        },
        2: {
            "seg_id": 2,
            "trail_name": "Gap",
            "direction": "both",
            "coordinates": [(-116.20, 43.20), (-116.21, 43.21)],
        },
    }
    candidate = {
        "candidate_id": "gap-route",
        "segments": [
            {"seg_id": 1, "trail_name": "Gap"},
            {"seg_id": 2, "trail_name": "Gap"},
        ],
        "route_orientation": {"direction": "forward"},
        "direction_validation": {"planned_traversal_direction": {}},
        "return_to_car": {},
    }
    connector_graph = {
        "graph": {
            (-116.11, 43.11): [
                {
                    "to": (-116.15, 43.15),
                    "distance": 0.01,
                    "name": "Connector",
                    "edge_type": "connector",
                    "connector_class": "r2r_trail",
                }
            ],
            (-116.15, 43.15): [
                {
                    "to": (-116.20, 43.20),
                    "distance": 0.01,
                    "name": "Connector",
                    "edge_type": "connector",
                    "connector_class": "r2r_trail",
                }
            ],
        },
        "nodes": [(-116.11, 43.11), (-116.15, 43.15), (-116.20, 43.20)],
    }

    coords = exporter.candidate_track_coordinates(
        candidate,
        official_index,
        connector_graph=connector_graph,
        stitch_gap_threshold_miles=0.01,
        stitch_snap_tolerance_miles=0.01,
    )

    assert coords == [
        (-116.10, 43.10),
        (-116.11, 43.11),
        (-116.15, 43.15),
        (-116.20, 43.20),
        (-116.21, 43.21),
    ]


def test_render_gpx_outputs_track_points():
    exporter = load_exporter()

    gpx = exporter.render_gpx("Route & Check", [(-116.1, 43.1), (-116.2, 43.2)])

    assert "<name>Route &amp; Check</name>" in gpx
    assert '<trkpt lat="43.100000" lon="-116.100000" />' in gpx
    assert '<trkpt lat="43.200000" lon="-116.200000" />' in gpx


def test_render_gpx_segments_keeps_multi_stop_tracks_separate():
    exporter = load_exporter()

    gpx = exporter.render_gpx_segments(
        "Multi Stop",
        [
            [(-116.1, 43.1), (-116.2, 43.2)],
            [(-116.3, 43.3), (-116.4, 43.4)],
        ],
    )

    assert gpx.count("<trkseg>") == 2
    assert '<trkpt lat="43.200000" lon="-116.200000" />' in gpx
    assert '<trkpt lat="43.300000" lon="-116.300000" />' in gpx


def test_validate_track_segments_flags_large_gpx_gaps():
    exporter = load_exporter()

    validation = exporter.validate_track_segments(
        [[(-116.10, 43.10), (-116.50, 43.50)]],
        max_gap_miles=0.1,
    )

    assert validation["passed"] is False
    assert validation["failures"][0]["code"] == "max_trackpoint_gap_exceeded"


def test_densify_coordinates_adds_points_inside_source_linework():
    exporter = load_exporter()

    dense = exporter.densify_coordinates(
        [(-116.10, 43.10), (-116.11, 43.11)],
        max_gap_miles=0.05,
    )

    assert dense[0] == (-116.10, 43.10)
    assert dense[-1] == (-116.11, 43.11)
    assert len(dense) > 2
    assert exporter.validate_track_segments([dense], max_gap_miles=0.05)["passed"] is True


def test_export_manifest_records_gpx_validation(tmp_path):
    exporter = load_exporter()
    plan = {
        "route_menu": {
            "all_candidates": [
                {
                    "candidate_id": "ready-route",
                    "segments": [{"seg_id": 1, "trail_name": "Ready"}],
                    "route_orientation": {"direction": "forward"},
                    "direction_validation": {"planned_traversal_direction": {}},
                    "return_to_car": {},
                }
            ]
        }
    }
    execution = {
        "single_car_menu": {
            "recommended_by_bucket": {
                "under_1_hour": {
                    "id": "ready-route",
                    "recommendation_type": "single_outing",
                    "trail_names": ["Ready"],
                    "official_new_miles": 1.0,
                    "simulated_total_minutes": 40,
                }
            }
        }
    }
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Ready",
            "direction": "both",
            "coordinates": [(-116.10, 43.10), (-116.1005, 43.1005)],
        }
    }

    manifest = exporter.export_recommendation_gpx(
        plan,
        execution,
        official_index,
        tmp_path,
        "single-car",
    )

    assert manifest["summary"]["route_count"] == 1
    assert manifest["summary"]["gpx_validation_passed"] is True
    assert manifest["routes"][0]["gpx_validation"]["passed"] is True
