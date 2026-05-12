import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "freestone_military_candidate_bundle_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("freestone_military_candidate_bundle_experiment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promotion_gates_block_hidden_repeat_latent_credit_and_worse_timing():
    module = load_module()
    loops = [
        {
            "official_segment_ids": ["1", "2"],
            "self_repeat_segment_ids": ["2"],
            "non_template_repeat_segment_ids": ["9"],
            "direct_gap_fallback_miles": 0.0,
            "ascent_direction_validation": {"direction_failed_segment_ids": []},
        }
    ]
    gates = module.promotion_gates(
        loops=loops,
        current_scope={"on_foot_miles": 10.0, "p75_minutes": 100, "p90_minutes": 120},
        candidate_scope={"on_foot_miles": 11.0, "p75_minutes": 110, "p90_minutes": 130},
        latent_rows=[{"segment_id": "9", "status": "owned_by_other_active_route_needs_declared_elsewhere"}],
        protected_segment_ids=[],
    )

    assert gates["status"] == "blocked_by_promotion_gates"
    assert "hidden_self_repeat" in gates["hard_failures"]
    assert "latent_credit_needs_ownership_decision" in gates["hard_failures"]
    assert "not_better_on_on_foot_p75_p90" in gates["hard_failures"]


def test_promotion_gates_require_protected_curlew_fat_tire_ids():
    module = load_module()
    loops = [
        {
            "official_segment_ids": ["1555", "1711"],
            "self_repeat_segment_ids": [],
            "non_template_repeat_segment_ids": [],
            "direct_gap_fallback_miles": 0.0,
            "ascent_direction_validation": {"direction_failed_segment_ids": []},
        }
    ]
    gates = module.promotion_gates(
        loops=loops,
        current_scope={"on_foot_miles": 10.0, "p75_minutes": 100, "p90_minutes": 120},
        candidate_scope={"on_foot_miles": 9.0, "p75_minutes": 90, "p90_minutes": 110},
        latent_rows=[],
        protected_segment_ids=["1555", "1711", "1710"],
    )

    assert gates["ascent_direction"] == "failed"
    assert gates["missing_protected_segment_ids"] == ["1710"]
    assert "ascent_or_protected_segment_failure" in gates["hard_failures"]


def test_route_impact_rows_show_shrink_remaining_ids():
    module = load_module()
    routes = {
        "FD20A": {
            "label": "FD20A",
            "outing_id": "120-1",
            "segment_ids": ["1681", "1682", "1563"],
        }
    }

    rows = module.route_impact_rows(routes, ["FD20A"], {"1681", "1682"})

    assert rows == [
        {
            "label": "FD20A",
            "outing_id": "120-1",
            "status": "shrunk_if_candidate_promoted",
            "covered_segment_ids": ["1681", "1682"],
            "remaining_segment_ids": ["1563"],
        }
    ]
