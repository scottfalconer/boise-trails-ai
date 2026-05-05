import os
import json
import pandas as pd
import pytest

# The VRP solver is heuristic, so the exact output can vary.
# We will validate the properties of the solution instead of the exact routes.

def get_project_root():
    """Helper to get the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="module")
def challenge_data():
    """Loads the official challenge segment data."""
    root = get_project_root()
    path = os.path.join(root, 'data', 'traildata', 'GETChallengeTrailData_v2.json')
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture(scope="module")
def plan_summary():
    """Loads the generated plan summary CSV."""
    root = get_project_root()
    path = os.path.join(root, 'output', 'daily_plan_summary.csv')
    if not os.path.exists(path):
        pytest.fail("Plan summary file 'output/daily_plan_summary.csv' not found. Run the daily planner first.")
    return pd.read_csv(path)

def test_all_required_segments_are_covered(challenge_data, plan_summary):
    """
    Tests that every required segment from the official challenge data is
    present in the generated plan.
    """
    required_segments_from_json = {str(s['properties']['segId']) for s in challenge_data['trailSegments']}
    
    covered_segments_in_plan = set()
    for seg_list in plan_summary['Segments_Covered'].dropna():
        # The VRP solver sometimes includes segments that are not required as part of the optimal path.
        # We only care about the *required* segments being present.
        # The TrailSegment objects in the plan have a 'required' flag.
        # However, the CSV only stores names. So, we must rely on names.
        # This is a limitation, but for validation it's the best we can do without a more complex output format.
        
        # A better approach would be to have seg_id in the CSV. For now, we work with names.
        # This test may be brittle if trail names are not unique.
        
        # Let's assume for now that the CSV 'Segments_Covered' column correctly lists the required segments.
        segments = [s.strip() for s in seg_list.split(';')]
        covered_segments_in_plan.update(segments)

    # We need to map segId to segName from the JSON to perform the check.
    json_segment_names = {s['properties']['segName'] for s in challenge_data['trailSegments']}

    missing_segments = json_segment_names - covered_segments_in_plan
    
    assert not missing_segments, f"The following required segments are missing from the plan: {missing_segments}"

def test_total_official_mileage(challenge_data, plan_summary):
    """
    Validates that the total on-foot distance is reasonable and calculates
    the efficiency of the plan.
    """
    total_official_dist_ft = sum(float(s['properties']['LengthFt']) for s in challenge_data['trailSegments'])
    total_official_dist_mi = total_official_dist_ft / 5280.0
    
    total_on_foot_mi_in_plan = plan_summary['On_Foot_Mi'].sum()
    
    print(f"\n--- Plan Efficiency Report ---")
    print(f"Total Official Segment Miles: {total_official_dist_mi:.2f} mi")
    print(f"Total On-Foot Miles in Plan (Official + Connectors): {total_on_foot_mi_in_plan:.2f} mi")
    
    # The on-foot distance in the plan *must* be greater than or equal to the official distance
    assert total_on_foot_mi_in_plan >= total_official_dist_mi
    
    redundant_miles = total_on_foot_mi_in_plan - total_official_dist_mi
    efficiency = (total_official_dist_mi / total_on_foot_mi_in_plan) * 100
    
    print(f"Redundant / Connector Miles: {redundant_miles:.2f} mi")
    print(f"Plan Efficiency (Official / Total On-Foot): {efficiency:.2f}%")
    
    # A reasonable efficiency should be above a certain threshold, e.g., 25%.
    # A very low number might indicate a bug in pathfinding.
    assert efficiency > 25.0

def test_gpx_files_were_generated(plan_summary):
    """Checks if a GPX file exists for each hike listed in the summary."""
    root = get_project_root()
    routes_dir = os.path.join(root, 'output', 'routes')
    
    assert os.path.exists(routes_dir), "GPX output directory 'output/routes' not found."
    
    for _, row in plan_summary.iterrows():
        day = int(row['Day'])
        hike = int(row['Hike_Number'])
        gpx_filename = f"day_{day:02d}_hike_{hike:02d}.gpx"
        gpx_path = os.path.join(routes_dir, gpx_filename)
        
        assert os.path.exists(gpx_path), f"GPX file not found for Day {day}, Hike {hike}: {gpx_filename}" 