import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_strava_segment_history.py"


def load_module():
    spec = importlib.util.spec_from_file_location("derive_strava_segment_history", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_orientation_gap_detects_forward_and_reverse():
    module = load_module()
    official = {"start": (0.0, 0.0), "end": (0.01, 0.0)}

    assert module.official_orientation_gap((0.0, 0.0), (0.01, 0.0), official)[0] == "forward"
    assert module.official_orientation_gap((0.01, 0.0), (0.0, 0.0), official)[0] == "reverse"


def test_best_official_match_uses_name_and_endpoint_geometry():
    module = load_module()
    effort = {
        "name": "Test Segment",
        "segment": {
            "name": "Test Segment",
            "start_latlng": [0.0, 0.0],
            "end_latlng": [0.0, 0.01],
        },
    }
    official = [
        {
            "seg_id": 1,
            "seg_name": "Other 1",
            "trail_name": "Other",
            "direction": "both",
            "start": (1.0, 1.0),
            "end": (1.01, 1.0),
        },
        {
            "seg_id": 2,
            "seg_name": "Test Segment 1",
            "trail_name": "Test Segment",
            "direction": "both",
            "start": (0.0, 0.0),
            "end": (0.01, 0.0),
        },
    ]

    match = module.best_official_match(effort, official)

    assert match["seg_id"] == 2
    assert match["confidence"] == "high"


def test_pace_min_per_mile_converts_meters_to_miles():
    module = load_module()
    effort = {
        "distance": module.METERS_PER_MILE * 2,
        "moving_time": 30 * 60,
    }

    assert module.pace_min_per_mile(effort) == 15
