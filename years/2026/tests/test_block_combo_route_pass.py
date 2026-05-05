import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "block_combo_route_pass.py"


def load_combo():
    spec = importlib.util.spec_from_file_location("block_combo_route_pass", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combo_is_acceptable_requires_graph_validated_and_mileage_cap():
    module = load_combo()
    components = [{"on_foot_miles": 3.0}, {"on_foot_miles": 4.0}]

    assert module.combo_is_acceptable(
        {"route_status": "graph_validated", "estimated_total_on_foot_miles": 8.5},
        components,
        max_extra_miles_per_saved_route=2.0,
    )
    assert not module.combo_is_acceptable(
        {"route_status": "draft", "estimated_total_on_foot_miles": 5.0},
        components,
        max_extra_miles_per_saved_route=2.0,
    )
    assert not module.combo_is_acceptable(
        {"route_status": "graph_validated", "estimated_total_on_foot_miles": 9.5},
        components,
        max_extra_miles_per_saved_route=2.0,
    )


def test_select_package_candidates_prefers_combo_when_it_saves_route_start():
    module = load_combo()
    original_index = {
        "a": {
            "candidate_id": "a",
            "segment_ids": [1],
            "official_new_miles": 1.0,
            "estimated_total_on_foot_miles": 2.0,
        },
        "b": {
            "candidate_id": "b",
            "segment_ids": [2],
            "official_new_miles": 1.0,
            "estimated_total_on_foot_miles": 2.0,
        },
    }
    combo = {
        "candidate_id": "combo-a-b",
        "segment_ids": [1, 2],
        "official_new_miles": 2.0,
        "estimated_total_on_foot_miles": 4.5,
        "source_component_candidate_ids": ["a", "b"],
    }

    selected = module.select_package_candidates(
        [{"candidate_id": "a"}, {"candidate_id": "b"}],
        [combo],
        original_index,
        route_count_weight=1.0,
    )

    assert [item["candidate_id"] for item in selected] == ["combo-a-b"]


def test_route_row_marks_combo_source_components():
    module = load_combo()
    row = module.route_row(
        1,
        {
            "candidate_id": "combo-a-b",
            "source_component_candidate_ids": ["a", "b"],
            "trail_names": ["A", "B"],
            "official_new_miles": 2.0,
            "estimated_total_on_foot_miles": 3.0,
            "trailhead": {"name": "TH"},
            "segment_ids": [1, 2],
        },
        {"block_id": "alpha", "block_name": "Alpha"},
    )

    assert row["is_combo"] is True
    assert row["source_component_candidate_ids"] == ["a", "b"]
    assert row["ratio"] == 1.5
