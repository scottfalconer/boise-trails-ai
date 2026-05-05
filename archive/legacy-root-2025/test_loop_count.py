#!/usr/bin/env python3
"""
Quick test to count loops generated from a single trailhead.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig, Trailhead

def test_single_trailhead():
    # Create minimal config
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
    
    print(f"\nTotal required segments: {len(router.required_segments)}")
    print(f"Total trailheads found: {len(router.trailheads)}")
    
    # Test loop generation from first trailhead
    if router.trailheads:
        th = router.trailheads[0]
        print(f"\nTesting trailhead: {th.name}")
        print(f"Accessible segments: {len(th.accessible_segments)}")
        
        # Generate loops
        loops = router._generate_trailhead_loops(th)
        print(f"Loops generated: {len(loops)}")
        
        # Show loop details
        for i, loop in enumerate(loops[:5]):  # First 5 loops
            req_segs = [s for s in loop.segments if s.required]
            print(f"\nLoop {i+1}:")
            print(f"  Required segments: {len(req_segs)}")
            print(f"  Total segments: {len(loop.segments)}")
            print(f"  Distance: {loop.total_distance:.1f} miles")
            print(f"  Coverage: {len(loop.required_coverage)} unique segments")
            
            # Show segment names
            if req_segs:
                names = [router._extract_base_trail_name(s.name) for s in req_segs[:3]]
                print(f"  Trail systems: {', '.join(set(names))}")

if __name__ == "__main__":
    test_single_trailhead()