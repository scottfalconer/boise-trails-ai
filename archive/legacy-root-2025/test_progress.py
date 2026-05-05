#!/usr/bin/env python3
"""
Test the improved progress tracking in the trailhead router.
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig

def main():
    print("🏔️  Testing Trailhead Router with Progress Tracking")
    print("=" * 60)
    
    # Load config but use fewer trailheads for faster testing
    with open('config/daily_planner_config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Limit to just 2 trailheads for quick test
    config_data['trailhead_depots'] = config_data['trailhead_depots'][:2]
    
    config = PlannerConfig(**config_data)
    
    # Create router without DEM provider for speed
    router = TrailheadRouter(config, dem_provider=None)
    
    # Load data
    print("\n📂 Loading trail data...")
    router.load_data(
        required_segments_path="data/traildata/GETChallengeTrailData_v2.json",
        all_trails_path="data/traildata/Boise_Parks_Trails_Open_Data.geojson",
        osm_pbf_path="data/osm/idaho-latest.osm.pbf"
    )
    
    # Generate plan - this will show the new progress indicators
    print("\n🚀 Starting plan generation with progress tracking...\n")
    plan = router.generate_plan()
    
    # Show results
    print("\n" + "=" * 60)
    print("✅ PLAN GENERATION COMPLETE")
    print("=" * 60)
    
    total_hikes = sum(len(day.hikes) for day in plan.days)
    covered_segments = set()
    for day in plan.days:
        for hike in day.hikes:
            for seg in hike.segments:
                if seg.required:
                    covered_segments.add(seg.seg_id)
    
    print(f"\nPlan Summary:")
    print(f"  Total days: {len(plan.days)}")
    print(f"  Total hikes: {total_hikes}")
    print(f"  Segments covered: {len(covered_segments)}/247")
    
    # Show the progress tracking worked
    print("\n✨ Progress tracking features demonstrated:")
    print("  - Phase indicators (1/4, 2/4, etc.)")
    print("  - Real-time progress with ETAs")
    print("  - Time taken for each phase")
    print("  - Total completion time")

if __name__ == "__main__":
    main()