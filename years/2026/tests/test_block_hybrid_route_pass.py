import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "block_hybrid_route_pass.py"


def load_hybrid():
    spec = importlib.util.spec_from_file_location("block_hybrid_route_pass", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_should_use_assembled_when_it_reduces_mileage():
    module = load_hybrid()

    use_assembled, reason = module.should_use_assembled(
        {"on_foot_miles": 10.0, "component_route_count": 1},
        {"on_foot_miles": 8.0, "route_status": "graph_validated"},
        max_extra_miles_per_saved_route=2.0,
    )

    assert use_assembled is True
    assert reason == "assembled_reduces_mileage"


def test_should_use_assembled_when_it_saves_start_with_small_extra_mileage():
    module = load_hybrid()

    use_assembled, reason = module.should_use_assembled(
        {"on_foot_miles": 10.0, "component_route_count": 2},
        {"on_foot_miles": 11.5, "route_status": "graph_validated"},
        max_extra_miles_per_saved_route=2.0,
    )

    assert use_assembled is True
    assert reason == "assembled_reduces_route_starts_with_acceptable_extra_miles"


def test_should_not_use_draft_assembled_route():
    module = load_hybrid()

    use_assembled, reason = module.should_use_assembled(
        {"on_foot_miles": 10.0, "component_route_count": 2},
        {"on_foot_miles": 8.0, "route_status": "draft"},
        max_extra_miles_per_saved_route=2.0,
    )

    assert use_assembled is False
    assert reason == "assembled_not_graph_validated"


def test_should_not_use_assembled_route_that_drops_package_segments():
    module = load_hybrid()

    use_assembled, reason = module.should_use_assembled(
        {"on_foot_miles": 10.0, "component_route_count": 2},
        {"on_foot_miles": 8.0, "route_status": "graph_validated", "segment_ids": [1]},
        max_extra_miles_per_saved_route=2.0,
        required_segment_ids={1, 2},
    )

    assert use_assembled is False
    assert reason == "assembled_does_not_cover_component_segments"


def test_build_hybrid_route_pass_mixes_assembled_and_components():
    module = load_hybrid()
    combo_route_pass = {
        "candidate_index": {"component-a": {"candidate_id": "component-a"}},
        "routes": [
            {
                "candidate_id": "component-a",
                "trail_names": ["A"],
                "official_miles": 1.0,
                "on_foot_miles": 2.0,
                "trailhead": "A TH",
                "route_status": "graph_validated",
                "segment_ids": [1],
            },
            {
                "candidate_id": "component-b",
                "trail_names": ["B"],
                "official_miles": 1.0,
                "on_foot_miles": 2.0,
                "trailhead": "B TH",
                "route_status": "graph_validated",
                "segment_ids": [2],
            },
        ],
    }
    combo_package_pass = {
        "packages": [
            {
                "block_id": "alpha",
                "block_name": "Alpha",
                "component_route_count": 2,
                "on_foot_miles": 4.0,
                "components": [
                    {"candidate_id": "component-a"},
                    {"candidate_id": "component-b"},
                ],
            },
            {
                "block_id": "beta",
                "block_name": "Beta",
                "component_route_count": 1,
                "on_foot_miles": 3.0,
                "components": [{"candidate_id": "component-c", "segment_ids": [3]}],
            },
        ]
    }
    assembled_pass = {
        "candidate_index": {"assembled-alpha": {"candidate_id": "assembled-alpha"}},
        "routes": [
            {
                "candidate_id": "assembled-alpha",
                "block_id": "alpha",
                "official_miles": 2.0,
                "on_foot_miles": 3.5,
                "trail_names": ["A", "B"],
                "route_status": "graph_validated",
                "segment_ids": [1, 2],
            },
            {
                "candidate_id": "assembled-beta",
                "block_id": "beta",
                "official_miles": 1.0,
                "on_foot_miles": 8.0,
                "trail_names": ["C"],
                "route_status": "graph_validated",
                "segment_ids": [3],
            },
        ],
    }

    hybrid = module.build_hybrid_route_pass(combo_route_pass, combo_package_pass, assembled_pass)

    assert hybrid["summary"]["selected_route_count"] == 2
    assert hybrid["summary"]["assembled_block_route_count"] == 1
    assert hybrid["summary"]["covered_segment_count"] == 3
    assert hybrid["routes"][0]["candidate_id"] == "assembled-alpha"
    assert hybrid["routes"][1]["route_source"] == "combo_package_component"


def test_global_hybrid_penalizes_cross_block_sweeps():
    module = load_hybrid()
    official_segments = [
        {"seg_id": 1, "trail_name": "A", "official_miles": 1.0},
        {"seg_id": 2, "trail_name": "B", "official_miles": 1.0},
    ]
    blocks_config = {
        "blocks": [
            {"block_id": "alpha", "name": "Alpha", "trail_names": ["A"]},
            {"block_id": "beta", "name": "Beta", "trail_names": ["B"]},
        ]
    }
    combo_route_pass = {
        "candidate_index": {},
        "routes": [
            {
                "candidate_id": "sweep",
                "block_id": "alpha",
                "block_name": "Alpha",
                "official_miles": 2.0,
                "on_foot_miles": 2.0,
                "trail_names": ["A", "B"],
                "route_status": "graph_validated",
                "segment_ids": [1, 2],
            }
        ],
    }
    assembled_pass = {
        "candidate_index": {},
        "routes": [
            {
                "candidate_id": "alpha",
                "block_id": "alpha",
                "block_name": "Alpha",
                "official_miles": 1.0,
                "on_foot_miles": 1.5,
                "trail_names": ["A"],
                "route_status": "graph_validated",
                "segment_ids": [1],
            },
            {
                "candidate_id": "beta",
                "block_id": "beta",
                "block_name": "Beta",
                "official_miles": 1.0,
                "on_foot_miles": 1.5,
                "trail_names": ["B"],
                "route_status": "graph_validated",
                "segment_ids": [2],
            },
        ],
    }

    hybrid = module.build_global_hybrid_route_pass(
        combo_route_pass,
        assembled_pass,
        official_segments,
        blocks_config,
        route_count_weight=1.0,
        cross_block_mile_penalty=20.0,
    )

    assert [route["candidate_id"] for route in hybrid["routes"]] == ["alpha", "beta"]
    assert hybrid["summary"]["cross_block_official_miles"] == 0.0


def test_global_hybrid_rejects_assembled_route_that_adds_running_against_package():
    module = load_hybrid()
    official_segments = [
        {"seg_id": 1, "trail_name": "A", "official_miles": 1.0},
        {"seg_id": 2, "trail_name": "B", "official_miles": 1.0},
    ]
    blocks_config = {
        "blocks": [{"block_id": "alpha", "name": "Alpha", "trail_names": ["A", "B"]}]
    }
    combo_route_pass = {
        "candidate_index": {},
        "routes": [
            {
                "candidate_id": "component-a",
                "block_id": "alpha",
                "block_name": "Alpha",
                "official_miles": 1.0,
                "on_foot_miles": 2.0,
                "trail_names": ["A"],
                "route_status": "graph_validated",
                "segment_ids": [1],
            },
            {
                "candidate_id": "component-b",
                "block_id": "alpha",
                "block_name": "Alpha",
                "official_miles": 1.0,
                "on_foot_miles": 2.0,
                "trail_names": ["B"],
                "route_status": "graph_validated",
                "segment_ids": [2],
            },
        ],
    }
    combo_package_pass = {
        "packages": [
            {
                "block_id": "alpha",
                "on_foot_miles": 4.0,
                "components": [
                    {"candidate_id": "component-a"},
                    {"candidate_id": "component-b"},
                ],
            }
        ]
    }
    assembled_pass = {
        "candidate_index": {},
        "routes": [
            {
                "candidate_id": "assembled-alpha",
                "block_id": "alpha",
                "block_name": "Alpha",
                "official_miles": 2.0,
                "on_foot_miles": 5.0,
                "trail_names": ["A", "B"],
                "route_status": "graph_validated",
                "segment_ids": [1, 2],
            }
        ],
    }

    hybrid = module.build_global_hybrid_route_pass(
        combo_route_pass,
        assembled_pass,
        official_segments,
        blocks_config,
        combo_package_pass=combo_package_pass,
        route_count_weight=20.0,
    )

    assert [route["candidate_id"] for route in hybrid["routes"]] == ["component-a", "component-b"]


def test_global_hybrid_covers_remaining_required_segments_after_progress():
    module = load_hybrid()
    official_segments = [
        {"seg_id": 1, "trail_name": "Done", "official_miles": 1.0},
        {"seg_id": 2, "trail_name": "A", "official_miles": 1.0},
        {"seg_id": 3, "trail_name": "B", "official_miles": 1.0},
    ]
    blocks_config = {
        "blocks": [
            {"block_id": "alpha", "name": "Alpha", "trail_names": ["A"]},
            {"block_id": "beta", "name": "Beta", "trail_names": ["B"]},
        ]
    }
    combo_route_pass = {
        "candidate_index": {},
        "routes": [
            {
                "candidate_id": "a",
                "block_id": "alpha",
                "block_name": "Alpha",
                "official_miles": 1.0,
                "on_foot_miles": 1.5,
                "trail_names": ["A"],
                "route_status": "graph_validated",
                "segment_ids": [2],
            },
            {
                "candidate_id": "b",
                "block_id": "beta",
                "block_name": "Beta",
                "official_miles": 1.0,
                "on_foot_miles": 1.5,
                "trail_names": ["B"],
                "route_status": "graph_validated",
                "segment_ids": [3],
            },
        ],
    }
    assembled_pass = {"candidate_index": {}, "routes": []}

    hybrid = module.build_global_hybrid_route_pass(
        combo_route_pass,
        assembled_pass,
        official_segments,
        blocks_config,
        required_segment_ids={2, 3},
    )

    assert [route["candidate_id"] for route in hybrid["routes"]] == ["a", "b"]
    assert hybrid["summary"]["target_segment_count"] == 2
    assert hybrid["summary"]["covered_segment_count"] == 2


def test_required_segment_ids_from_plan_uses_remaining_trails():
    module = load_hybrid()

    required = module.required_segment_ids_from_plan(
        {"remaining_trails": [{"remaining_segment_ids": [2, "3"]}]},
        [
            {"seg_id": 1, "trail_name": "Done"},
            {"seg_id": 2, "trail_name": "A"},
            {"seg_id": 3, "trail_name": "B"},
        ],
    )

    assert required == {2, 3}
