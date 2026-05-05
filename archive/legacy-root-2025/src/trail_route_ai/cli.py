#!/usr/bin/env python3
"""
Command line interface for the Boise Trails Challenge route planner.
"""

import argparse
import os
import sys
import yaml
import json
from pathlib import Path

from .trailhead_router import TrailheadRouter
from .core.models import PlannerConfig


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Boise Trails Challenge Route Planner')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Trailhead-based planning command
    plan_parser = subparsers.add_parser('plan', help='Generate trailhead-based hiking plan')
    plan_parser.add_argument('--config', '-c', 
                            default='config/daily_planner_config.yaml',
                            help='Configuration file path')
    plan_parser.add_argument('--required-segments', '-r',
                            default='data/traildata/GETChallengeTrailData_v2.json',
                            help='Required segments JSON file')
    plan_parser.add_argument('--all-trails', '-t',
                            default='data/traildata/Boise_Parks_Trails_Open_Data.geojson',
                            help='All trails GeoJSON file')
    plan_parser.add_argument('--osm-data', '-o',
                            default='data/osm/idaho-latest.osm.pbf',
                            help='OSM road network file')
    plan_parser.add_argument('--output-dir', 
                            default='output',
                            help='Output directory for results')
    
    # Legacy VRP planning command  
    vrp_parser = subparsers.add_parser('vrp', help='Generate VRP-based plan (legacy)')
    vrp_parser.add_argument('--config', '-c',
                           default='config/daily_planner_config.yaml',
                           help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.command == 'plan':
        generate_trailhead_plan(args)
    elif args.command == 'vrp':
        print("VRP mode not implemented in CLI yet. Use the daily_planner module directly.")
    else:
        parser.print_help()


def generate_trailhead_plan(args):
    """Generate a trailhead-based hiking plan"""
    print("🏔️  Boise Trails Challenge - Trailhead-Based Route Planner")
    print("=" * 60)
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    config = PlannerConfig(**config_data)
    print(f"✅ Loaded configuration from {args.config}")
    
    # Initialize router
    router = TrailheadRouter(config)
    
    # Load data
    try:
        router.load_data(args.required_segments, args.all_trails, args.osm_data)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Generate plan
    print("\n🔄 Generating optimized hiking plan...")
    try:
        plan = router.generate_plan()
    except Exception as e:
        print(f"❌ Error generating plan: {e}")
        sys.exit(1)
    
    # Output results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save summary
    save_plan_summary(plan, output_dir / "trailhead_plan_summary.json")
    
    # Save detailed plan
    save_detailed_plan(plan, output_dir / "trailhead_plan_detailed.json")
    
    # Generate GPX files
    generate_gpx_files(plan, output_dir / "routes")
    
    # Print summary
    print_plan_summary(plan)
    
    print(f"\n✅ Plan saved to {output_dir}")
    print(f"📂 Check {output_dir}/routes/ for individual GPX files")


def save_plan_summary(plan, output_path):
    """Save plan summary statistics"""
    summary = {
        'plan_type': 'trailhead_based',
        'total_days': len(plan.days),
        'total_hikes': len(plan.all_hikes),
        'statistics': plan.summary_stats,
        'days_overview': []
    }
    
    for day in plan.days:
        day_info = {
            'day_number': day.number,
            'day_type': day.type,
            'total_distance': round(day.total_distance, 1),
            'total_elevation': round(day.total_elevation, 0),
            'hike_count': len(day.hikes),
            'trailheads': list(set(hike.trailhead for hike in day.hikes))
        }
        summary['days_overview'].append(day_info)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def save_detailed_plan(plan, output_path):
    """Save detailed plan with full navigation"""
    detailed = {
        'plan_type': 'trailhead_based',
        'generated_by': 'TrailheadRouter',
        'summary_stats': plan.summary_stats,
        'days': []
    }
    
    for day in plan.days:
        day_data = {
            'day_number': day.number,
            'day_type': day.type,
            'total_distance': day.total_distance,
            'total_elevation': day.total_elevation,
            'hikes': []
        }
        
        for hike in day.hikes:
            hike_data = {
                'hike_number': hike.hike_number,
                'trailhead': hike.trailhead,
                'total_distance': hike.total_distance,
                'elevation_gain': hike.elevation_gain,
                'difficulty': hike.difficulty,
                'estimated_minutes': hike.estimated_minutes,
                'trail_conditions': hike.trail_conditions,
                'parking': {
                    'coords': hike.parking_coords,
                    'type': hike.parking_type,
                    'fee': hike.parking_fee,
                    'notes': hike.parking_notes
                },
                'segments': [
                    {
                        'id': seg.seg_id,
                        'name': seg.name,
                        'length_mi': seg.length_mi,
                        'required': seg.required
                    }
                    for seg in hike.segments
                ],
                'navigation': [
                    {
                        'distance': nav.distance_from_start,
                        'instruction': nav.instruction,
                        'landmark': nav.landmark,
                        'gps_coords': nav.gps_coords
                    }
                    for nav in hike.navigation_points
                ],
                'escape_routes': [
                    {
                        'at_mile': escape.at_mile,
                        'description': escape.description,
                        'saves_miles': escape.saves_miles
                    }
                    for escape in hike.escape_routes
                ]
            }
            day_data['hikes'].append(hike_data)
        
        detailed['days'].append(day_data)
    
    with open(output_path, 'w') as f:
        json.dump(detailed, f, indent=2)


def generate_gpx_files(plan, output_dir):
    """Generate GPX files for each hike"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for day in plan.days:
        for hike in day.hikes:
            filename = f"day_{day.number:02d}_hike_{hike.hike_number:02d}.gpx"
            filepath = output_dir / filename
            
            create_gpx_file(hike, filepath)


def create_gpx_file(hike, filepath):
    """Create a GPX file for a single hike"""
    import gpxpy
    import gpxpy.gpx
    
    # Create GPX object
    gpx = gpxpy.gpx.GPX()
    gpx.name = f"Day {hike.hike_number} - {hike.trailhead}"
    gpx.description = f"{hike.difficulty.title()} hike from {hike.trailhead}"
    
    # Create track
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx_track.name = f"{hike.trailhead} Loop"
    gpx.tracks.append(gpx_track)
    
    # Create track segment
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    
    # Add track points from segments
    for segment in hike.segments:
        for coord in segment.coordinates:
            lon, lat = coord
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))
    
    # Add waypoints for navigation
    for nav_point in hike.navigation_points:
        if nav_point.gps_coords:
            lat, lon = nav_point.gps_coords
            waypoint = gpxpy.gpx.GPXWaypoint(lat, lon, name=nav_point.landmark)
            waypoint.description = nav_point.instruction
            gpx.waypoints.append(waypoint)
    
    # Save GPX file
    with open(filepath, 'w') as f:
        f.write(gpx.to_xml())


def print_plan_summary(plan):
    """Print a summary of the generated plan"""
    stats = plan.summary_stats
    
    print("\n📊 Plan Summary")
    print("-" * 40)
    print(f"Total Days:           {stats['total_days']}")
    print(f"Total Hikes:          {len(plan.all_hikes)}")
    print(f"Total Distance:       {stats['total_miles']} miles")
    print(f"Required Miles:       {stats['required_miles']} miles")
    print(f"Connector Miles:      {stats['connector_miles']} miles")
    print(f"Road Miles:           {stats['road_miles']} miles")
    print(f"Redundancy:           {stats['redundancy_percent']}%")
    print(f"Efficiency Score:     {stats['efficiency_score']}")
    print(f"Unique Trailheads:    {stats['unique_trailheads']}")
    print(f"Avg Hikes/Day:        {stats['average_hikes_per_day']}")
    
    print("\n📅 Daily Breakdown")
    print("-" * 40)
    for day in plan.days[:5]:  # Show first 5 days
        trailheads = list(set(hike.trailhead for hike in day.hikes))
        print(f"Day {day.number:2d} ({day.type:6s}): {day.total_distance:5.1f} mi, "
              f"{len(day.hikes)} hike{'s' if len(day.hikes) != 1 else ''}, "
              f"from {', '.join(trailheads)}")
    
    if len(plan.days) > 5:
        print(f"... and {len(plan.days) - 5} more days")


if __name__ == '__main__':
    main() 