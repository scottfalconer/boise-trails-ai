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
