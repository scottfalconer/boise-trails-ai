#!/usr/bin/env python3
"""
Test full plan generation but with limited time to see results.
"""

import os
import sys
from pathlib import Path
import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_full_plan():
    # Create config with just major trailheads
    config = PlannerConfig(
        solver_time_limit_seconds=30,
        trailhead_depots=[
            {"name": "Camel's Back Park", "lat": 43.628, "lon": -116.202, "capacity": 100},
            {"name": "Military Reserve", "lat": 43.621, "lon": -116.165, "capacity": 100},
            {"name": "Stack Rock", "lat": 43.675, "lon": -116.130, "capacity": 50},
            {"name": "Bogus Basin Lower", "lat": 43.764, "lon": -116.155, "capacity": 100},
            {"name": "Highland Valley", "lat": 43.657, "lon": -116.235, "capacity": 50},
        ],
        daily_capacities={"weekday": {"short": 6, "medium": 15, "long": 25}},
        cost_model={"elevation_beta": 10.0},
        drive_threshold_miles=20.0,
        short_day_limit=6.0,
        medium_day_limit=15.0,
        long_day_limit=25.0
    )
    
    # Create router
    router = TrailheadRouter(config)
    
    # Load data
    print("Loading trail data...")
    router.load_data(
        required_segments_path="data/traildata/GETChallengeTrailData_v2.json",
        all_trails_path="data/traildata/Boise_Parks_Trails_Open_Data.geojson",
        osm_pbf_path="data/osm/idaho-latest.osm.pbf"
    )
    
    # Quick check of trail families
    all_segments = list(router.required_segments.values())
    families = router._group_segments_by_trail_system(all_segments)
    print(f"\nTrail families found: {len(families)}")
    combined_families = router._combine_related_trail_families(families)
    print(f"After combining: {len(combined_families)} trail systems")
    
    # Show expected hike count
    total_expected_hikes = len(combined_families)  # Assuming one hike per family
    print(f"\nExpected hikes (if 1 per trail system): {total_expected_hikes}")
    
    # Try to generate just the loops from major trailheads
    print("\nGenerating loops from major trailheads only...")
    all_loops = []
    
    # Set a timeout for loop generation
    signal.signal(signal.SIGALRM, timeout_handler)
    
    for i, th in enumerate(router.trailheads[:5]):  # Just first 5 trailheads
        print(f"\nProcessing {th.name}...")
        signal.alarm(10)  # 10 second timeout per trailhead
        try:
            loops = router._generate_trailhead_loops(th)
            all_loops.extend(loops)
            print(f"  Generated {len(loops)} loops")
        except TimeoutError:
            print(f"  Timed out - skipping")
        finally:
            signal.alarm(0)
    
    print(f"\nTotal loops from major trailheads: {len(all_loops)}")
    
    # Try consolidation
    print("\nConsolidating loops...")
    consolidated = router._consolidate_nearby_loops(all_loops)
    print(f"After consolidation: {len(consolidated)} hikes")
    
    # Show coverage
    covered = set()
    for loop in consolidated:
        covered.update(loop.required_coverage)
    print(f"\nSegments covered: {len(covered)}/{len(router.required_segments)}")

if __name__ == "__main__":
    test_full_plan()