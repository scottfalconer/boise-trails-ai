import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "years" / "2026" / "scripts" / "post_h1_field_day_cleanup.py"


def load_module():
    spec = importlib.util.spec_from_file_location("post_h1_field_day_cleanup", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assignment(date, candidate_id, p75=0, p90=0, draft_day_number=1):
    return {
        "date": date,
        "weekday_name": "Sunday",
        "day_type": "weekend",
        "field_day": {
            "draft_day_number": draft_day_number,
            "field_day_id": f"day-{candidate_id}",
            "day_type": "weekend",
            "p75_minutes": p75,
            "p90_minutes": p90,
            "p90_bound_minutes": 360,
            "loops": [{"candidate_id": candidate_id}],
        },
    }


def test_move_bogus_to_reserve_slot_swaps_field_days_and_preserves_date_metadata():
    module = load_module()
    payload = {
        "assignments": [
            assignment("2026-07-12", "skipped-harlow", p75=315, p90=353, draft_day_number=30),
            assignment("2026-07-18", "block-bogus_mores_lodge_tempest", p75=320, p90=359, draft_day_number=31),
        ],
        "known_gaps": [],
    }

    cleaned, report = module.move_bogus_to_reserve_slot(payload)
    by_date = {row["date"]: row for row in cleaned["assignments"]}

    assert report["status"] == "passed"
    assert by_date["2026-07-12"]["field_day"]["loops"][0]["candidate_id"] == "block-bogus_mores_lodge_tempest"
    assert by_date["2026-07-12"]["field_day"]["draft_day_number"] == 30
    assert by_date["2026-07-18"]["field_day"]["loops"][0]["candidate_id"] == "skipped-harlow"
    assert by_date["2026-07-18"]["field_day"]["draft_day_number"] == 31
    assert report["accepted_changes"][0]["final_buffer_date"] == "2026-07-18"


def test_move_bogus_to_reserve_slot_means_july_12_is_not_expected_empty():
    module = load_module()
    payload = {
        "assignments": [
            assignment("2026-07-12", "skipped-harlow", p75=315, p90=353, draft_day_number=30),
            assignment("2026-07-18", "block-bogus_mores_lodge_tempest", p75=320, p90=359, draft_day_number=31),
        ],
        "known_gaps": [],
    }

    cleaned, _report = module.move_bogus_to_reserve_slot(payload)
    by_date = {row["date"]: row for row in cleaned["assignments"]}

    assert by_date["2026-07-12"]["field_day"]["loops"]
    assert by_date["2026-07-18"]["field_day"]["loops"][0]["candidate_id"] == "skipped-harlow"


def test_move_bogus_to_reserve_slot_requires_bogus_source_date():
    module = load_module()
    payload = {
        "assignments": [
            assignment("2026-07-12", "skipped-harlow"),
            assignment("2026-07-18", "not-bogus"),
        ]
    }

    try:
        module.move_bogus_to_reserve_slot(payload)
    except ValueError as exc:
        assert "source_date_does_not_contain_bogus_18" in str(exc)
    else:
        raise AssertionError("expected missing Bogus source validation")
