import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "template_route_candidate_builder.py"


def load_module():
    spec = importlib.util.spec_from_file_location("template_route_candidate_builder", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ordered_unique_ids_preserves_template_route_order():
    module = load_module()

    assert module.ordered_unique_ids(["1585", "1532", "1585", 1616]) == ["1585", "1532", "1616"]


def test_promotion_gates_block_ungenerated_harlow_west_probe():
    module = load_module()
    gates = module.promotion_gates(
        definition={
            "extra_hard_failures": [
                "no_verified_harlow_west_anchor",
                "hidden_springs_pattern_uses_stale_or_conflicted_broken_horn_access",
            ]
        },
        loops=[],
        current_scope={"on_foot_miles": 34.0, "p75_minutes": 991, "p90_minutes": 1117},
        candidate_scope=None,
        latent_rows=[],
    )

    assert gates["status"] == "blocked_by_promotion_gates"
    assert "no_verified_harlow_west_anchor" in gates["hard_failures"]
    assert "route_geometry_missing" in gates["hard_failures"]
    assert gates["continuous_gpx"] == "not_generated"


def test_recommendation_requires_negative_delta_and_respects_material_threshold():
    module = load_module()

    assert (
        module.recommendation_for_bundle(
            {
                "generated_loops": [{}],
                "delta_vs_current_total_scope": {"on_foot_miles": -6.0},
                "material_savings_threshold_miles": 5.0,
            }
        )
        == "promising_candidate_needs_hard_gate_repair"
    )
    assert (
        module.recommendation_for_bundle(
            {
                "generated_loops": [{}],
                "delta_vs_current_total_scope": {"on_foot_miles": -3.0},
                "material_savings_threshold_miles": 5.0,
            }
        )
        == "minor_candidate_keep_as_backlog"
    )
    assert (
        module.recommendation_for_bundle(
            {
                "generated_loops": [{}],
                "delta_vs_current_total_scope": {"on_foot_miles": 0.5},
                "material_savings_threshold_miles": 0.0,
            }
        )
        == "do_not_promote_current_cards_are_cheaper"
    )


def test_route_metrics_for_labels_ignores_missing_labels_after_active_packet_change():
    module = load_module()
    routes_by_label = {
        "H1": {
            "label": "H1",
            "outing_id": "127-1",
            "segment_ids": ["1626"],
            "on_foot_miles": 9.64,
            "official_miles": 7.3,
            "door_to_door_minutes_p75": 289,
            "door_to_door_minutes_p90": 324,
        }
    }

    metrics = module.route_metrics_for_labels(routes_by_label, ["FD27A", "H1"])

    assert metrics["route_count"] == 1
    assert metrics["labels"] == ["H1"]
    assert module.missing_route_labels(routes_by_label, ["FD27A", "H1"]) == ["FD27A"]


def test_safe_route_impact_rows_marks_missing_replaced_routes_as_absent():
    module = load_module()
    rows = module.safe_route_impact_rows({}, ["FD27A"], set())

    assert rows == [
        {
            "label": "FD27A",
            "outing_id": None,
            "status": "route_absent_from_active_packet",
            "covered_segment_ids": [],
            "remaining_segment_ids": [],
        }
    ]
