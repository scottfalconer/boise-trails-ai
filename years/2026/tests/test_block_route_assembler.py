import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "block_route_assembler.py"


def load_assembler():
    spec = importlib.util.spec_from_file_location("block_route_assembler", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_order_trails_by_connector_cost_reverses_bidirectional_trail():
    assembler = load_assembler()
    trail = {
        "trail_name": "Example",
        "segments": [
            {
                "direction": "both",
                "start": (10.0, 10.0),
                "end": (0.1, 0.1),
                "coordinates": [(10.0, 10.0), (0.1, 0.1)],
            }
        ],
        "start": (10.0, 10.0),
        "end": (0.1, 0.1),
        "remaining_segment_ids": [1],
    }

    ordered = assembler.order_trails_by_connector_cost(
        [trail],
        start_point=(0.0, 0.0),
        connector_graph=None,
        snap_tolerance_miles=0.1,
    )

    assert ordered[0]["start"] == (0.1, 0.1)
    assert ordered[0]["end"] == (10.0, 10.0)


def test_block_acceptance_flags_boundary_and_high_ratio():
    assembler = load_assembler()
    candidate = {
        "official_new_miles": 5.0,
        "estimated_total_on_foot_miles": 11.0,
        "route_status": "graph_validated",
        "validation": {"ascent_direction_passed": True},
    }
    status, reasons = assembler.block_acceptance_status(
        {"block_id": "dry", "status": "boundary_review"},
        candidate,
        {"preferred_max_on_foot_to_official_ratio": 1.6, "min_official_miles_unless_geography_locked": 5.0},
    )

    assert status == "needs_manual_gpx_review"
    assert "boundary_review_block" in reasons
    assert "ratio_above_preferred_limit" in reasons


def test_build_map_data_uses_assembled_summary_shape():
    assembler = load_assembler()
    route_pass = {
        "summary": {
            "assembled_route_count": 1,
            "covered_segment_count": 1,
            "total_on_foot_miles": 2.0,
            "planwide_on_foot_to_official_ratio": 1.2,
        },
        "routes": [
            {
                "route_number": 1,
                "candidate_id": "block-alpha",
                "block_name": "Alpha",
                "trail_names": ["Alpha Trail"],
                "official_miles": 1.0,
                "on_foot_miles": 2.0,
                "trailhead": "A",
            }
        ],
        "candidate_index": {
            "block-alpha": {
                "candidate_id": "block-alpha",
                "segments": [{"seg_id": 1, "trail_name": "Alpha Trail"}],
                "route_orientation": {"direction": "forward"},
                "direction_validation": {"planned_traversal_direction": {}},
                "return_to_car": {"path_coordinates": [[1.0, 1.0], [0.0, 0.0]]},
            }
        },
    }
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Alpha Trail",
            "direction": "both",
            "coordinates": [(0.0, 0.0), (1.0, 1.0)],
        }
    }

    map_data = assembler.build_map_data(route_pass, official_index, connector_graph=None)

    assert map_data["summary"]["selected_route_count"] == 1
    assert map_data["summary"]["covered_segment_count"] == 1
    assert map_data["feature_collections"]["official_segments"]["features"][0]["properties"]["seg_id"] == 1
