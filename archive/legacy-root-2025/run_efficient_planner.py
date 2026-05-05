#!/usr/bin/env python3
"""
Run the efficient trailhead-based planner and generate a summary CSV.
"""

import sys
from pathlib import Path
import csv
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trail_route_ai.trailhead_router import TrailheadRouter
from trail_route_ai.core.models import PlannerConfig

def main():
    # Load config
    with open('config/daily_planner_config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    
    config = PlannerConfig(**config_data)
    
    # Create router
    print("🏔️  Running Efficient Trailhead-Based Planner")
    print("=" * 60)
    
    router = TrailheadRouter(config)
    
    # Load data
    print("Loading trail data...")
    router.load_data(
        required_segments_path="data/traildata/GETChallengeTrailData_v2.json",
        all_trails_path="data/traildata/Boise_Parks_Trails_Open_Data.geojson",
        osm_pbf_path="data/osm/idaho-latest.osm.pbf"
    )
    
    # Show expected efficiency
    total_required_miles = sum(seg.length_mi for seg in router.required_segments.values())
    print(f"\nTotal required trail miles: {total_required_miles:.1f}")
    print(f"Target total on-foot miles: {total_required_miles * 1.1:.1f} - {total_required_miles * 1.2:.1f} (10-20% redundancy)")
    
    # Generate plan
    print("\nGenerating efficient plan...")
    plan = router.generate_plan()
    
    # Calculate totals
    total_hikes = sum(len(day.hikes) for day in plan.days)
    total_on_foot = sum(hike.total_distance for day in plan.days for hike in day.hikes)
    total_driving = sum(hike.drive_to_distance for day in plan.days for hike in day.hikes if hasattr(hike, 'drive_to_distance'))
    
    # Calculate coverage
    covered_segments = set()
    for day in plan.days:
        for hike in day.hikes:
            for segment in hike.segments:
                if segment.required:
                    covered_segments.add(segment.seg_id)
    
    redundancy_pct = ((total_on_foot / total_required_miles) - 1) * 100
    
    print(f"\n=== RESULTS ===")
    print(f"Total days: {len(plan.days)}")
    print(f"Total hikes: {total_hikes}")
    print(f"Total on-foot miles: {total_on_foot:.1f}")
    print(f"Total driving miles: {total_driving:.1f}")
    print(f"Segments covered: {len(covered_segments)}/247")
    print(f"Redundancy: {redundancy_pct:.1f}% (target: 10-20%)")
    
    # Write summary CSV
    output_path = Path("output/efficient_plan_summary.csv")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Day', 'Hike_Number', 'Trailhead', 'On_Foot_Mi', 'Drive_To_Next_Hike_Mi', 'Segments_Covered'])
        
        for day in plan.days:
            for hike in day.hikes:
                segments_str = '; '.join([seg.name for seg in hike.segments if seg.required])
                drive_miles = getattr(hike, 'drive_to_distance', 0.0)
                writer.writerow([
                    day.day_number,
                    hike.hike_number,
                    hike.trailhead,
                    f"{hike.total_distance:.2f}",
                    f"{drive_miles:.2f}",
                    segments_str
                ])
    
    print(f"\nPlan saved to: {output_path}")
    
    # Show some example hikes
    print("\nSample efficient hikes:")
    for i, day in enumerate(plan.days[:3]):
        for j, hike in enumerate(day.hikes[:2]):
            req_count = sum(1 for s in hike.segments if s.required)
            print(f"  Day {day.day_number}, Hike {hike.hike_number}: {req_count} segments in {hike.total_distance:.1f} miles")

if __name__ == "__main__":
    main()