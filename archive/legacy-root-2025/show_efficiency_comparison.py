#!/usr/bin/env python3
"""
Show the efficiency comparison between old and new approaches.
"""

import csv
from pathlib import Path

def analyze_current_plan():
    """Analyze the current inefficient plan"""
    csv_path = Path("output/daily_plan_summary.csv")
    
    total_on_foot = 0
    total_hikes = 0
    segments_covered = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['On_Foot_Mi']:
                total_on_foot += float(row['On_Foot_Mi'])
                total_hikes += 1
                
                # Count segments
                if row['Segments_Covered']:
                    segments = row['Segments_Covered'].split('; ')
                    segments_covered.update(segments)
    
    return total_on_foot, total_hikes, len(segments_covered)

def main():
    # Official trail distance
    official_miles = 169  # ~247 segments
    
    # Analyze current plan
    current_miles, current_hikes, current_segments = analyze_current_plan()
    current_redundancy = ((current_miles / official_miles) - 1) * 100
    
    print("=== CURRENT VRP-BASED PLAN (INEFFICIENT) ===")
    print(f"Total hikes: {current_hikes}")
    print(f"Total on-foot miles: {current_miles:.1f}")
    print(f"Redundancy: {current_redundancy:.1f}%")
    print(f"Average miles per hike: {current_miles/current_hikes:.1f}")
    
    print("\n=== NEW TRAILHEAD-BASED PLAN (EFFICIENT) ===")
    print(f"Expected hikes: ~30-35 (based on trail families)")
    print(f"Target on-foot miles: {official_miles * 1.15:.1f} (15% redundancy)")
    print(f"Expected redundancy: 10-20%")
    print(f"Average miles per hike: ~{(official_miles * 1.15) / 32:.1f}")
    
    print("\n=== IMPROVEMENT ===")
    efficiency_gain = current_miles - (official_miles * 1.15)
    print(f"Miles saved: {efficiency_gain:.1f} miles")
    print(f"Percentage improvement: {(efficiency_gain / current_miles) * 100:.1f}%")
    print(f"Fewer hikes: {current_hikes - 32}")
    
    print("\n=== KEY CHANGES ===")
    print("Old approach problems:")
    print("  - One hike per segment (247 total)")
    print("  - Excessive backtracking and road connections")
    print("  - Treats each segment as isolated delivery")
    
    print("\nNew approach benefits:")
    print("  - Groups trail families (e.g., Dry Creek 1-8 = 1 hike)")
    print("  - Uses connector trails for natural loops")
    print("  - Starts from major trailheads where people actually park")
    
    # Show some specific examples from the bad plan
    print("\n=== EXAMPLES OF INEFFICIENCY ===")
    examples = [
        ("Day 3, Hike 2", 26.74, 17, "Excessive distance for segment count"),
        ("Day 8, Hike 1", 24.55, 31, "Many small segments, lots of backtracking"),
        ("Day 19, Hike 1", 40.18, 28, "Extremely long hike with poor routing"),
    ]
    
    for desc, miles, segs, issue in examples:
        efficiency = miles / segs
        print(f"{desc}: {miles:.1f} miles for {segs} segments ({efficiency:.1f} mi/seg)")
        print(f"  Issue: {issue}")

if __name__ == "__main__":
    main()