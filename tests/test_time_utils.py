import pytest
from trail_route_ai import planner_utils

def test_parse_time_budget_hours_minutes():
    assert planner_utils.parse_time_budget('1h30') == pytest.approx(90.0)
    assert planner_utils.parse_time_budget('2h15m') == pytest.approx(135.0)

def test_parse_time_budget_formats():
    assert planner_utils.parse_time_budget('1:45') == pytest.approx(105.0)
    assert planner_utils.parse_time_budget('90') == pytest.approx(90.0)
    assert planner_utils.parse_time_budget('1.5h') == pytest.approx(90.0)
