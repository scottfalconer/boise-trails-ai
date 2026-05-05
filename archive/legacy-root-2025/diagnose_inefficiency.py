#!/usr/bin/env python3
"""
Diagnose why the planner is producing inefficient routes.
"""

import json
from pathlib import Path

def analyze_plan():
    # Load the plan
    plan_path = Path("output/efficient_plan/trailhead_plan_summary.json")
    with open(plan_path, 'r') as f:
        plan = json.load(f)
    
    print("=== EFFICIENCY ANALYSIS ===\n")
    
    # 1. Show the key metrics
    stats = plan['statistics']
    print("Key Metrics:")
    print(f"  Total hikes: {plan['total_hikes']} (target: ~30)")
    print(f"  Redundancy: {stats['redundancy_percent']:.1f}% (target: <20%)")
    print(f"  Connector miles: {stats['connector_miles']} (should be >0)")
    print(f"  Road miles: {stats['road_miles']:.1f}")
    print(f"  Unique trailheads: {stats['unique_trailheads']} across {plan['total_days']} days")
    
    # 2. Analyze hike distribution
    print("\nHikes per day:")
    hike_counts = {}
    for day in plan['days_overview']:
        count = day['hike_count']
        hike_counts[count] = hike_counts.get(count, 0) + 1
        if count > 3:
            print(f"  Day {day['day_number']}: {count} hikes (TOO MANY!)")
    
    print("\nHike count distribution:")
    for count, days in sorted(hike_counts.items()):
        print(f"  {count} hikes/day: {days} days")
    
    # 3. Check for actual detailed plan
    detailed_path = Path("output/efficient_plan/trailhead_plan_detailed.json")
    if detailed_path.exists():
        with open(detailed_path, 'r') as f:
            detailed = json.load(f)
        
        # Sample some hikes to see segment counts
        print("\nSample hikes (first 10):")
        hike_num = 0
        for day in detailed['days'][:3]:
            for hike in day['hikes'][:4]:
                hike_num += 1
                if hike_num > 10:
                    break
                    
                segments = hike.get('segments', [])
                required = [s for s in segments if s.get('required', False)]
                connectors = [s for s in segments if not s.get('required', False)]
                
                print(f"  Hike {hike_num}: {len(required)} required, {len(connectors)} connectors, "
                      f"{hike['total_distance']:.1f}mi total")
                
                if len(connectors) == 0 and len(required) > 1:
                    print(f"    ⚠️  No connectors used - likely out-and-back!")
    
    print("\n=== DIAGNOSIS ===")
    print("Problems identified:")
    print("1. Zero connector miles - connector trails aren't being used")
    print("2. 44% redundancy - massive out-and-backs instead of loops")
    print("3. 64 hikes instead of ~30 - not combining segments effectively")
    print("4. Multiple trailheads per day - poor consolidation")
    
    print("\nRoot cause: The trail family grouping works, but:")
    print("- Connector trails aren't incorporated into routes")
    print("- Each trail family becomes a separate hike")
    print("- No loop formation, just out-and-backs")

if __name__ == "__main__":
    analyze_plan()