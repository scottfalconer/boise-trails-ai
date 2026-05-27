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
    assert module.combo_is_acceptable(
        {
            "route_status": "graph_validated",
            "estimated_total_on_foot_miles": 8.5,
            "field_track_miles": 9.5,
        },
        components,
        max_extra_miles_per_saved_route=2.0,
    )


def combo_trail(name, seg_id):
    return {
        "trail_name": name,
        "segments": [
            {
                "direction": "ascent",
                "start": (float(seg_id), 0.0),
                "end": (float(seg_id) + 0.5, 0.0),
                "coordinates": [(float(seg_id), 0.0), (float(seg_id) + 0.5, 0.0)],
            }
        ],
        "start": (float(seg_id), 0.0),
        "end": (float(seg_id) + 0.5, 0.0),
        "remaining_segment_ids": [seg_id],
    }


def test_build_combo_candidate_orders_by_legal_connector_cost(monkeypatch):
    module = load_combo()
    alpha = combo_trail("Alpha Trail", 1)
    bravo = combo_trail("Bravo Trail", 2)
    calls = {}

    def fake_order(trails, start_point, connector_graph, snap_tolerance_miles, avoid_official_segment_ids=None):
        calls["trail_names"] = [trail["trail_name"] for trail in trails]
        calls["start_point"] = start_point
        calls["connector_graph"] = connector_graph
        calls["snap_tolerance_miles"] = snap_tolerance_miles
        calls["avoid_official_segment_ids"] = set(avoid_official_segment_ids or set())
        return [bravo, alpha]

    def fake_candidate_from_trail_group(ordered, state, performance_profile, connector_graph, **kwargs):
        _ = (state, performance_profile, connector_graph, kwargs)
        return {
            "candidate_id": "raw",
            "trail_names": [trail["trail_name"] for trail in ordered],
            "segment_ids": [
                seg_id
                for trail in ordered
                for seg_id in trail.get("remaining_segment_ids") or []
            ],
            "route_status": "graph_validated",
        }

    monkeypatch.setattr(module, "order_trails_by_legal_connector_cost", fake_order)
    monkeypatch.setattr(module, "candidate_from_trail_group", fake_candidate_from_trail_group)

    candidate = module.build_combo_candidate(
        [{"candidate_id": "a"}, {"candidate_id": "b"}],
        {
            "a": {"candidate_id": "a", "trail_names": ["Alpha Trail"]},
            "b": {"candidate_id": "b", "trail_names": ["Bravo Trail"]},
        },
        {
            "alpha trail": alpha,
            "bravo trail": bravo,
        },
        {
            "state": {"outing_model": {"mapped_connector_snap_tolerance_miles": 0.07}},
            "performance_profile": {},
            "connector_graph": {"graph": {}},
            "elevation_sampler": None,
        },
    )

    assert calls["trail_names"] == ["Alpha Trail", "Bravo Trail"]
    assert calls["start_point"] == alpha["start"]
    assert calls["connector_graph"] == {"graph": {}}
    assert calls["snap_tolerance_miles"] == 0.07
    assert calls["avoid_official_segment_ids"] == {1, 2}
    assert candidate["candidate_id"] == "combo-a-b"
    assert candidate["trail_names"] == ["Bravo Trail", "Alpha Trail"]


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
