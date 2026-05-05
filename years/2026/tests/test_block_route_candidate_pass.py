import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "block_route_candidate_pass.py"


def load_pass():
    spec = importlib.util.spec_from_file_location("block_route_candidate_pass", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_block_for_candidate_uses_segment_mileage_not_trail_count_tie():
    module = load_pass()
    trail_to_block = {
        "connector": {"block_id": "westside"},
        "dry creek trail": {"block_id": "dry_creek"},
        "highlands trail": {"block_id": "hillside"},
    }
    candidate = {
        "trail_names": ["Connector", "Highlands Trail", "Dry Creek Trail"],
        "segments": [
            {"trail_name": "Connector", "official_miles": 0.6},
            {"trail_name": "Highlands Trail", "official_miles": 1.5},
            {"trail_name": "Dry Creek Trail", "official_miles": 7.0},
        ],
    }

    assert module.block_for_candidate(candidate, trail_to_block) == "dry_creek"


def test_summarize_selection_reports_fragment_counts():
    module = load_pass()
    blocks = {
        "blocks": [
            {"block_id": "alpha", "name": "Alpha", "trail_names": ["Alpha Trail"]},
            {"block_id": "beta", "name": "Beta", "trail_names": ["Beta Trail"]},
        ]
    }
    selected = [
        {
            "candidate_id": "alpha",
            "trail_names": ["Alpha Trail"],
            "segments": [{"seg_id": 1, "trail_name": "Alpha Trail", "official_miles": 0.8}],
            "segment_ids": [1],
            "official_new_miles": 0.8,
            "estimated_total_on_foot_miles": 1.2,
            "total_minutes": 30,
            "trailhead": {"name": "A"},
            "route_status": "graph_validated",
            "less_optimal_flags": [],
        },
        {
            "candidate_id": "beta",
            "trail_names": ["Beta Trail"],
            "segments": [{"seg_id": 2, "trail_name": "Beta Trail", "official_miles": 2.0}],
            "segment_ids": [2],
            "official_new_miles": 2.0,
            "estimated_total_on_foot_miles": 3.0,
            "total_minutes": 60,
            "trailhead": {"name": "B"},
            "route_status": "graph_validated",
            "less_optimal_flags": [],
        },
    ]

    route_pass = module.summarize_selection(selected, blocks)

    assert route_pass["summary"]["selected_route_count"] == 2
    assert route_pass["summary"]["covered_segment_count"] == 2
    assert route_pass["summary"]["routes_under_1_official_mile"] == 1
    assert route_pass["summary"]["routes_under_2_official_miles"] == 1
    assert route_pass["routes"][0]["block_id"] == "beta"


def test_render_html_includes_direction_arrow_controls():
    module = load_pass()
    html = module.render_html(
        {
            "summary": {
                "selected_route_count": 0,
                "covered_segment_count": 0,
                "total_on_foot_miles": 0,
                "planwide_on_foot_to_official_ratio": 0,
            },
            "routes": [],
            "feature_collections": {
                "routes": {"type": "FeatureCollection", "features": []},
                "official_segments": {"type": "FeatureCollection", "features": []},
            },
        }
    )

    assert "drawDirectionArrows" in html
    assert "drawRouteCues" in html
    assert "path-marker" in html
    assert "one clear cased line" in html
    assert "dir-arrow" in html
    assert "double-backs" in html
    assert "parking-marker" in html
    assert "where to park" in html
    assert "selectedPanel" in html
    assert "mapSummary" in html
    assert "parking-label" in html
    assert "Selected route" in html
