import json
from trail_route_ai import planner_utils


def test_official_data_loads_correctly():
    """Ensure the official dataset loads without losing any segment IDs."""
    official_path = "data/traildata/GETChallengeTrailData_v2.json"
    with open(official_path) as f:
        data = json.load(f)
    official_ids = {
        str(seg.get("segId") or seg.get("id"))
        for seg in data.get("trailSegments", [])
        if seg.get("segId") or seg.get("id")
    }

    loaded = planner_utils.load_segments(official_path)
    loaded_ids = {str(e.seg_id) for e in loaded if e.seg_id is not None}

    assert official_ids == loaded_ids
