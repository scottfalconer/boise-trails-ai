import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_activity_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_activity_review", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment(seg_id, coords, direction="both"):
    return {
        "seg_id": seg_id,
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "official_miles": 0.5,
        "direction": direction,
        "coordinates": coords,
        "start": coords[0],
        "end": coords[-1],
    }


def test_activity_review_records_missed_partial_and_extra_completed_segments():
    module = load_module()
    planned = segment(1, [(-116.0, 43.0), (-115.99, 43.0)])
    extra = segment(2, [(-116.0, 43.02), (-115.99, 43.02)])
    activity = [
        (-116.0, 43.0),
        (-115.996, 43.0),
        (-116.0, 43.02),
        (-115.99, 43.02),
    ]

    review = module.review_activity_against_segments(
        activity,
        [planned, extra],
        planned_segment_ids=[1],
        planned_outing_id="route-a",
        threshold_miles=0.015,
        min_fraction=0.8,
        partial_min_fraction=0.2,
    )

    assert review["completed_segment_ids"] == ["2"]
    assert review["extra_completed_segment_ids"] == ["2"]
    assert review["missed_segment_ids"] == ["1"]
    assert review["partial_segment_ids"] == ["1"]
    assert review["planned_outing_id"] == "route-a"


def test_activity_review_rejects_ascent_segment_in_wrong_direction():
    module = load_module()
    ascent = segment(3, [(-116.0, 43.0), (-115.99, 43.0)], direction="ascent")
    reversed_activity = [(-115.99, 43.0), (-116.0, 43.0)]

    review = module.review_activity_against_segments(
        reversed_activity,
        [ascent],
        planned_segment_ids=[3],
        threshold_miles=0.015,
        min_fraction=0.8,
        elevation_sampler=lambda point: (point[0] + 116.0) * 10000,
    )

    assert review["completed_segment_ids"] == []
    assert review["missed_segment_ids"] == ["3"]
    assert review["segment_reviews"][0]["direction_ok"] is False
    assert review["segment_reviews"][0]["completion_status"] == "wrong_ascent_direction"
