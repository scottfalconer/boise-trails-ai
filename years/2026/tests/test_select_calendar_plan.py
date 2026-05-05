import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "select_calendar_plan.py"


def load_selector():
    spec = importlib.util.spec_from_file_location("select_calendar_plan", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_runbook(path, profile_name, official_miles, segments, validation=True):
    path.write_text(
        json.dumps(
            {
                "profile_name": profile_name,
                "run_id": f"{profile_name}-run",
                "summary": {
                    "scheduled_official_miles": official_miles,
                    "scheduled_segments": segments,
                    "scheduled_total_on_foot_miles": official_miles + 10,
                },
                "audit": {
                    "scheduled_days": 3,
                    "scheduled_executable_units": 4,
                    "execution_validation_passed": validation,
                },
            }
        ),
        encoding="utf-8",
    )


def test_select_plan_prefers_requested_profile(tmp_path):
    selector = load_selector()
    conservative = tmp_path / "default.json"
    full = tmp_path / "full.json"
    write_runbook(conservative, "default-120s", 121.4, 213)
    write_runbook(full, "full-clear-sensitivity", 164.4, 251)

    selected = selector.build_selected_plan(
        [conservative, full],
        selected_profile="default-120s",
    )

    assert selected["selected_profile"] == "default-120s"
    assert selected["selected_runbook_path"] == str(conservative)
    assert selected["alternatives"][0]["profile_name"] == "full-clear-sensitivity"


def test_select_plan_can_choose_highest_valid_coverage(tmp_path):
    selector = load_selector()
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    write_runbook(low, "low", 80, 100)
    write_runbook(high, "high", 164.4, 251)

    selected = selector.build_selected_plan([low, high])

    assert selected["selected_profile"] == "high"
    assert selected["coverage_basis"]["scheduled_segments"] == 251
