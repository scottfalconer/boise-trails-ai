#!/usr/bin/env python3
"""
Test to diagnose why the planner gets stuck at loop 64/65.
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig, Loop, TrailSegment

def create_test_loops():
    """Create test loops to reproduce the issue"""
    loops = []
    
    # Create 64 normal loops
    for i in range(64):
        segments = [
            TrailSegment(
                seg_id=f"test_{i}",
                name=f"Test Trail {i}",
                coordinates=[(-116.0, 43.0), (-116.01, 43.01)],
                length_ft=5280 * 2,  # 2 miles
                direction="both",
                required=True
            )
        ]
        
        loop = Loop(
            trailhead="Test Trailhead",
            segments=segments,
            total_distance=2.0,
            required_coverage={f"test_{i}"},
            connector_ratio=0.0
        )
        loops.append(loop)
    
    # Create one problematic loop (too big)
    big_segments = []
    for j in range(50):  # Many segments
        big_segments.append(
            TrailSegment(
                seg_id=f"big_{j}",
                name=f"Big Trail {j}",
                coordinates=[(-116.0, 43.0), (-116.01, 43.01)],
                length_ft=5280,  # 1 mile each
                direction="both",
                required=True
            )
        )
    
    big_loop = Loop(
        trailhead="Test Trailhead",
        segments=big_segments,
        total_distance=50.0,  # Very large!
        required_coverage={f"big_{j}" for j in range(50)},
        connector_ratio=0.0
    )
    loops.append(big_loop)
    
    return loops

def main():
    print("🔍 Testing stuck issue with 65 loops...")
    
    # Load config
    with open('config/daily_planner_config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    
    config = PlannerConfig(**config_data)
    
    print(f"\nDay capacities:")
    print(f"  Short: {config.short_day_limit} miles")
    print(f"  Medium: {config.medium_day_limit} miles")
    print(f"  Long: {config.long_day_limit} miles")
    
    # Create router
    router = TrailheadRouter(config)
    
    # Create test loops
    loops = create_test_loops()
    print(f"\nCreated {len(loops)} test loops")
    print(f"Loop 65 distance: {loops[-1].total_distance} miles")
    
    # Test organize_into_days directly
    print("\nTesting _organize_into_days...")
    daily_plans = router._organize_into_days(loops)
    
    print(f"\nResult: {len(daily_plans)} days created")
    
    # Check if all loops were assigned
    total_hikes = sum(len(day.hikes) for day in daily_plans)
    print(f"Total hikes assigned: {total_hikes}/{len(loops)}")
    
    # Show any loops that are too big
    for i, loop in enumerate(loops):
        if loop.total_distance > config.long_day_limit:
            print(f"\n⚠️  Loop {i+1} is too big: {loop.total_distance} miles > {config.long_day_limit} miles capacity")

if __name__ == "__main__":
    main()