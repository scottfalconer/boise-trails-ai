from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_relaxed_drive_draft_plan as plan  # noqa: E402


def test_summarize_segments_counts_unique_ids_and_miles():
    official = {
        1: {"official_miles": 1.0},
        2: {"official_miles": 0.25},
        3: {"official_miles": 0.5},
    }

    summary = plan.summarize_segments([2, 1, 2], official)

    assert summary == {
        "segment_count": 2,
        "official_miles": 1.25,
        "segment_ids": [1, 2],
    }


def test_enrich_day_preserves_loop_validation_and_time():
    day = {
        "field_day_id": "weekday-a",
        "day_type": "weekday",
        "loop_ids": ["loop-a"],
        "segment_ids": [1],
        "loop_count": 1,
        "p75_minutes": 100,
        "p90_minutes": 120,
        "p90_bound_minutes": 292,
        "stress": 0.411,
        "drive_minutes": 20,
        "between_drive_minutes": 0,
        "on_foot_miles": 3.5,
        "grade_adjusted_miles": 4.0,
        "parking_risk": 1,
    }
    loop_by_id = {
        "loop-a": {
            "loop_id": "loop-a",
            "label": "Loop A",
            "source": "fixture",
            "candidate_id": "a",
            "trailhead": "Trailhead",
            "trail_names": ["Trail"],
            "segment_ids": [1],
            "official_miles": 1.0,
            "on_foot_miles": 3.5,
            "grade_adjusted_miles": 4.0,
            "ascent_ft": 100,
            "door_to_door_p75_minutes": 100,
            "door_to_door_p90_minutes": 120,
            "validation_passed": True,
            "manual_design_hold": False,
            "parking_confidence": "source_verified",
        }
    }

    enriched = plan.enrich_day(1, day, loop_by_id=loop_by_id, official_index={1: {"official_miles": 1.0}})

    assert enriched["draft_day_number"] == 1
    assert enriched["segment_summary"]["official_miles"] == 1.0
    assert enriched["loops"][0]["validation_passed"] is True
    assert enriched["loops"][0]["parking_confidence"] == "source_verified"
