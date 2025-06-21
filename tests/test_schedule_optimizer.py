import datetime
from trail_route_ai.challenge_planner import ScheduleOptimizer

def make_plan(day, has_activity=True):
    return {
        "date": day,
        "activities": [1] if has_activity else [],
        "total_activity_time": 10.0 if has_activity else 0.0,
        "total_drive_time": 0.0,
        "notes": "",
    }

def test_optimize_schedule_spreads_plan():
    start = datetime.date(2024, 7, 1)
    end = datetime.date(2024, 7, 4)
    plans = [make_plan(start), make_plan(start + datetime.timedelta(days=1))]
    budgets = {start + datetime.timedelta(days=i): 60.0 for i in range(4)}
    optimized = ScheduleOptimizer.optimize_schedule(plans, start, end, budgets)
    assert len(optimized) == 4
    assert optimized[0]["date"] == start
    assert optimized[-1]["date"] == end
    assert optimized[0]["activities"]
    assert optimized[-1]["activities"]
    assert not optimized[1]["activities"] and not optimized[2]["activities"]

def test_optimize_schedule_early_completion():
    start = datetime.date(2024, 7, 1)
    end = datetime.date(2024, 7, 4)
    plans = [make_plan(start), make_plan(start + datetime.timedelta(days=1))]
    budgets = {start + datetime.timedelta(days=i): 60.0 for i in range(4)}
    optimized = ScheduleOptimizer.optimize_schedule(plans, start, end, budgets, allow_early_completion=True)
    assert len(optimized) == 4
    assert optimized[0]["date"] == start
    assert optimized[1]["date"] == start + datetime.timedelta(days=1)
    assert optimized[-1]["date"] == end
    assert optimized[-1]["activities"] == []
