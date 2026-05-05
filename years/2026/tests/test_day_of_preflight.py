import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "day_of_preflight.py"


def load_preflight():
    spec = importlib.util.spec_from_file_location("day_of_preflight", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(*trail_names):
    return {
        "id": "test-route",
        "recommendation_type": "single_outing",
        "trail_names": list(trail_names),
        "trailheads": ["Test Trailhead"],
    }


def test_lower_hulls_is_blocked_for_foot_on_odd_numbered_days():
    preflight = load_preflight()

    result = preflight.evaluate_route_preflight(
        route("Lower Hull's Gulch Trail"),
        run_date="2026-06-19",
        start_time="07:00",
    )

    assert result["field_status"] == "blocked"
    assert "lower_hulls_odd_day_closed_to_foot" in result["blocking_reasons"]


def test_lower_hulls_passes_static_rule_on_even_numbered_days_but_still_needs_conditions():
    preflight = load_preflight()

    result = preflight.evaluate_route_preflight(
        route("Lower Hull's Gulch Trail"),
        run_date="2026-06-20",
        start_time="07:00",
    )

    assert result["field_status"] == "needs_day_of_check"
    assert result["blocking_reasons"] == []
    assert "current_r2r_conditions" in result["manual_checks_required"]


def test_polecat_always_requires_current_directional_signage_check():
    preflight = load_preflight()

    result = preflight.evaluate_route_preflight(
        route("Polecat Loop"),
        run_date="2026-06-20",
        start_time="07:00",
    )

    assert result["field_status"] == "needs_day_of_check"
    assert "polecat_current_directional_signage" in result["manual_checks_required"]


def test_late_low_foothills_start_gets_heat_warning():
    preflight = load_preflight()

    result = preflight.evaluate_route_preflight(
        route("Polecat Loop"),
        run_date="2026-07-05",
        start_time="10:30",
    )

    assert "late_start_low_foothills_heat_risk" in result["warnings"]
