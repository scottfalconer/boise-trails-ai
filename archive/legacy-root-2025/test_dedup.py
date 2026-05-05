#!/usr/bin/env python3
"""
Test deduplication of loops.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig

def test_dedup():
    # Create config
    config = PlannerConfig(
        solver_time_limit_seconds=30,
        trailhead_depots=[
            {"name": "Camel's Back Park", "lat": 43.628, "lon": -116.202},
            {"name": "Military Reserve", "lat": 43.621, "lon": -116.165},
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
    
    # Generate loops from both trailheads
    all_loops = []
    for th in router.trailheads[:2]:
        print(f"\nGenerating loops from {th.name}...")
        loops = router._generate_trailhead_loops(th)
        all_loops.extend(loops)
        print(f"  Generated {len(loops)} loops")
    
    print(f"\nTotal loops before dedup: {len(all_loops)}")
    
    # Test the selection algorithm
    selected = router._select_optimal_loops(all_loops)
    print(f"Selected loops: {len(selected)}")
    
    # Check coverage
    covered = set()
    for loop in selected:
        covered.update(loop.required_coverage)
    print(f"Segments covered: {len(covered)}/{len(router.required_segments)}")

if __name__ == "__main__":
    test_dedup()