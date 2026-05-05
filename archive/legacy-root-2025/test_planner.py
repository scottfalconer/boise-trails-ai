#!/usr/bin/env python3
"""
Quick test of the trailhead router to see how many hikes it generates.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig

def test_router():
    # Create minimal config
    config = PlannerConfig(
        solver_time_limit_seconds=300,
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
    
    # Generate plan
    print("\nGenerating plan...")
    plan = router.generate_plan()
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total days: {len(plan.days)}")
    
    total_hikes = sum(len(day.hikes) for day in plan.days)
    print(f"Total hikes: {total_hikes}")
    
    covered_segments = set()
    for day in plan.days:
        for hike in day.hikes:
            for segment in hike.segments:
                if segment.required:
                    covered_segments.add(segment.seg_id)
    
    print(f"Segments covered: {len(covered_segments)}/{len(router.required_segments)}")
    
    # Show first few days
    print("\n=== FIRST 3 DAYS ===")
    for i, day in enumerate(plan.days[:3]):
        print(f"\nDay {i+1}:")
        for j, hike in enumerate(day.hikes):
            req_segments = [s for s in hike.segments if s.required]
            print(f"  Hike {j+1}: {hike.trailhead} - {len(req_segments)} required segments, {hike.total_distance:.1f} miles")

if __name__ == "__main__":
    test_router()