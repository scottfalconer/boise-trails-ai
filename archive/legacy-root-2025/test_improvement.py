#!/usr/bin/env python3
"""
Show the improvement in hike count.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig

def show_improvement():
    # Create config
    config = PlannerConfig(
        solver_time_limit_seconds=30,
        trailhead_depots=[
            {"name": "Camel's Back Park", "lat": 43.628, "lon": -116.202},
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
    
    print(f"\nTotal required segments: 247")
    
    # Show trail family grouping
    all_segments = list(router.required_segments.values())
    original_families = router._group_segments_by_trail_system(all_segments)
    print(f"Original trail families: {len(original_families)}")
    
    combined_families = router._combine_related_trail_families(original_families)
    print(f"After combining related systems: {len(combined_families)}")
    
    print("\nExpected improvement:")
    print(f"  Old system: 247 hikes (one per segment)")
    print(f"  New system: ~{len(combined_families)} hikes (one per trail system)")
    print(f"  Reduction: {247 - len(combined_families)} fewer hikes ({(1 - len(combined_families)/247)*100:.0f}% reduction)")
    
    # Show some example combinations
    print("\nExample trail system combinations:")
    for name, segments in list(combined_families.items())[:5]:
        print(f"  {name}: {len(segments)} segments combined into 1 hike")
        seg_names = [s.name for s in segments[:3]]
        print(f"    Including: {', '.join(seg_names)}{'...' if len(segments) > 3 else ''}")

if __name__ == "__main__":
    show_improvement()