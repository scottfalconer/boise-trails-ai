import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_connector_shortest_path_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("derive_connector_shortest_path_matrix", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_matrix_records_directed_shortest_paths_with_snap_distance():
    module = load_module()
    a = (0.0, 0.0)
    b = (0.01, 0.0)
    graph = {
        "graph": {
            a: [{"to": b, "distance": 1.0}],
            b: [{"to": a, "distance": 1.2}],
        },
        "nodes": [a, b],
    }
    nodes = [
        {"node_id": "a", "node_type": "test", "lon": 0.0, "lat": 0.0},
        {"node_id": "b", "node_type": "test", "lon": 0.01, "lat": 0.0},
    ]

    matrix = module.build_matrix(nodes, graph, snap_tolerance_feet=20)

    assert matrix["summary"]["snapped_node_count"] == 2
    assert matrix["summary"]["matrix_row_count"] == 2
    rows = {(row["source_node_id"], row["target_node_id"]): row for row in matrix["rows"]}
    assert rows[("a", "b")]["distance_miles"] == 1.0
    assert rows[("b", "a")]["distance_miles"] == 1.2


def test_build_matrix_reports_unsnapped_nodes():
    module = load_module()
    graph = {"graph": {}, "nodes": [(0.0, 0.0)]}
    nodes = [{"node_id": "far", "node_type": "test", "lon": 10.0, "lat": 10.0}]

    matrix = module.build_matrix(nodes, graph, snap_tolerance_feet=20)

    assert matrix["summary"]["snapped_node_count"] == 0
    assert matrix["summary"]["unsnapped_node_count"] == 1
