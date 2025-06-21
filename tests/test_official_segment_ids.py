import json
from trail_route_ai import planner_utils


def test_trail_json_contains_all_official_segments():
    """Ensure trail.json retains all official segment IDs."""
    official_path = "data/traildata/GETChallengeTrailData_v2.json"
    with open(official_path) as f:
        data = json.load(f)
    official_ids = {
        str(seg.get("segId") or seg.get("id"))
        for seg in data.get("trailSegments", [])
        if seg.get("segId") or seg.get("id")
    }

    loaded = planner_utils.load_segments("data/traildata/trail.json")
    loaded_ids = {str(e.seg_id) for e in loaded if e.seg_id is not None}

    missing = official_ids - loaded_ids
    assert not missing, f"Missing IDs from trail.json: {sorted(missing)}"
