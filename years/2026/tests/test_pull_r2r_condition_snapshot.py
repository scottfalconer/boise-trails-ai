import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pull_r2r_condition_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pull_r2r_condition_snapshot", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_statusfy_extracts_status_detail_and_update_age():
    module = load_module()
    raw = """
    <h1>Ridge to Rivers Trail Condition Reports</h1>
    <div>Updated 23 days ago</div>
    <h2>Status Detail</h2>
    Wet Weather = Trails susceptible to damage. Do Not Use.
    <div>Last Updated On By David Gordon</div>
    <span>1776087266</span>
    """

    parsed = module.parse_statusfy(raw, "https://example.test")

    assert parsed["status_detail"] == "Wet Weather = Trails susceptible to damage. Do Not Use."
    assert parsed["updated_age_text"] == "23 days ago"
    assert parsed["updated_epoch"] == 1776087266
