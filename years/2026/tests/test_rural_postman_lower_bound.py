import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rural_postman_lower_bound.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rural_postman_lower_bound", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature(seg_id, coords, length_ft=5280, direction="both"):
    return {
        "type": "Feature",
        "properties": {
            "segId": seg_id,
            "segName": f"Segment {seg_id}",
            "LengthFt": length_ft,
            "direction": direction,
        },
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def connector_graph(nodes, edges):
    graph = {node: [] for node in nodes}
    for start, end, distance in edges:
        graph.setdefault(start, []).append(
            {
                "to": end,
                "distance": distance,
                "name": "Public connector",
                "edge_type": "connector",
                "connector_class": "osm_public_road",
                "source": "test",
                "highway": "residential",
            }
        )
    return {
        "path": "/tmp/test-connectors.geojson",
        "graph": graph,
        "nodes": sorted(set(nodes)),
        "feature_count": len(edges),
        "official_segment_count": 0,
        "connector_class_counts": {"osm_public_road": len(edges)},
    }


def test_lower_bound_for_simple_path_adds_straight_line_odd_pair():
    module = load_module()
    official = {
        "type": "FeatureCollection",
        "features": [
            feature(1, [[0.0, 0.0], [0.01, 0.0]], length_ft=5280),
            feature(2, [[0.01, 0.0], [0.02, 0.0]], length_ft=5280),
        ],
    }

    report = module.build_report(official, snap_tolerance_feet=20)

    assert report["summary"]["required_segment_count"] == 2
    assert report["summary"]["odd_node_count"] == 2
    assert report["matching"]["pair_count"] == 1
    assert report["summary"]["rural_postman_lower_bound_miles"] > report["summary"]["official_miles"]
    assert report["quality_checks"]["matching_pair_count_expected"] is True


def test_endpoint_snapping_can_make_a_tiny_gap_free_for_lower_bound():
    module = load_module()
    official = {
        "type": "FeatureCollection",
        "features": [
            feature(1, [[0.0, 0.0], [0.01, 0.0]], length_ft=5280),
            feature(2, [[0.010001, 0.0], [0.02, 0.0]], length_ft=5280),
        ],
    }

    loose = module.build_report(official, snap_tolerance_feet=50)
    strict = module.build_report(official, snap_tolerance_feet=0)

    assert loose["summary"]["required_graph_node_count"] < strict["summary"]["required_graph_node_count"]
    assert loose["summary"]["odd_node_count"] < strict["summary"]["odd_node_count"]


def test_multilinestring_parts_are_preserved_as_separate_required_edges():
    module = load_module()
    official = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "segId": 10,
                    "segName": "Multipart",
                    "LengthFt": 10560,
                    "direction": "ascent",
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[0.0, 0.0], [0.01, 0.0]],
                        [[0.02, 0.0], [0.03, 0.0]],
                    ],
                },
            }
        ],
    }

    report = module.build_report(official, snap_tolerance_feet=20)

    assert report["summary"]["required_segment_count"] == 1
    assert report["summary"]["required_edge_part_count"] == 2
    assert report["summary"]["direction_counts"] == {"ascent": 2}


def test_current_plan_comparison_is_reported_when_audit_is_supplied():
    module = load_module()
    official = {"type": "FeatureCollection", "features": [feature(1, [[0.0, 0.0], [0.01, 0.0]])]}
    audit = {"summary": {"runnable_field_packet_totals": {"on_foot_miles": 3.0}}}

    report = module.build_report(official, snap_tolerance_feet=20, current_plan_audit=audit)

    comparison = report["summary"]["current_plan_comparison"]
    assert comparison["on_foot_miles"] == 3.0
    assert comparison["gap_to_lower_bound_miles"] >= 0


def test_connector_graph_lower_bound_uses_real_connector_distance():
    module = load_module()
    start = (0.0, 0.0)
    end = (0.01, 0.0)
    official = {"type": "FeatureCollection", "features": [feature(1, [start, end])]}
    graph = connector_graph(
        [start, end],
        [
            (start, end, 2.0),
            (end, start, 2.0),
        ],
    )

    report = module.build_report(
        official,
        snap_tolerance_feet=20,
        connector_graph=graph,
        connector_snap_tolerance_feet=20,
    )

    assert report["connector_graph_matching"]["available"] is True
    assert report["summary"]["connector_graph_parity_add_on_miles"] == 2.0
    assert report["summary"]["connector_graph_lower_bound_miles"] == 3.0
    assert report["quality_checks"]["connector_graph_lower_bound_available"] is True
    assert report["quality_checks"]["connector_graph_lower_bound_at_least_straight_line_lower_bound"] is True


def test_connector_graph_lower_bound_reports_unsnapped_nodes():
    module = load_module()
    official = {"type": "FeatureCollection", "features": [feature(1, [[0.0, 0.0], [0.01, 0.0]])]}
    far_graph = connector_graph(
        [(10.0, 10.0), (10.01, 10.0)],
        [((10.0, 10.0), (10.01, 10.0), 1.0)],
    )

    report = module.build_report(
        official,
        snap_tolerance_feet=20,
        connector_graph=far_graph,
        connector_snap_tolerance_feet=20,
    )

    assert report["connector_graph_matching"]["available"] is False
    assert len(report["connector_graph_matching"]["unsnapped_odd_nodes"]) == 2
    assert report["quality_checks"]["connector_graph_lower_bound_available"] is False
