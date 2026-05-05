#!/usr/bin/env python3
"""
Test the final plan generation to see actual hike count.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig

def test_final():
    # Create config
    config = PlannerConfig(
        solver_time_limit_seconds=30,
        trailhead_depots=[
            {"name": "Camel's Back Park", "lat": 43.628, "lon": -116.202},
            {"name": "Military Reserve", "lat": 43.621, "lon": -116.165},
            {"name": "Stack Rock", "lat": 43.675, "lon": -116.130},
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
    
    # Generate full plan
    print("\nGenerating full plan...")
    plan = router.generate_plan()
    
    # Summary
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total days: {len(plan.days)}")
    
    total_hikes = sum(len(day.hikes) for day in plan.days)
    print(f"Total hikes: {total_hikes}")
    
    # Check coverage
    covered = set()
    for day in plan.days:
        for hike in day.hikes:
            for seg in hike.segments:
                if seg.required:
                    covered.add(seg.seg_id)
    
    print(f"Segments covered: {len(covered)}/247")
    
    # Show hike distribution
    hike_sizes = {}
    for day in plan.days:
        for hike in day.hikes:
            req_count = sum(1 for s in hike.segments if s.required)
            if req_count not in hike_sizes:
                hike_sizes[req_count] = 0
            hike_sizes[req_count] += 1
    
    print("\nHike size distribution:")
    for size, count in sorted(hike_sizes.items()):
        print(f"  {size} segments: {count} hikes")

if __name__ == "__main__":
    test_final()