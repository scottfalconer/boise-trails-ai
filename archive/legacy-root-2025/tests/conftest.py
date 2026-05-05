import os
import sys
import pytest
import json
from src.trail_route_ai.trailhead_router import TrailheadRouter
from src.trail_route_ai.core.models import PlannerConfig, GeneratedPlan, Day, Hike, TrailSegment, NavigationPoint
from pathlib import Path

# Ensure the src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

@pytest.fixture
def config():
    """Test configuration"""
    return PlannerConfig(
        solver_time_limit_seconds=300,
        trailhead_depots=[
            {'name': "Camel's Back Park", 'lat': 43.635278, 'lon': -116.205, 'capacity': 50},
            {'name': "Military Reserve", 'lat': 43.62, 'lon': -116.18, 'capacity': 40},
            {'name': "Stack Rock Trailhead", 'lat': 43.75, 'lon': -116.10, 'capacity': 30},
            {'name': "Bogus Basin", 'lat': 43.76, 'lon': -116.10, 'capacity': 25}
        ],
        daily_capacities={
            'short_day': {'capacity': 6, 'vehicles': 10},
            'medium_day': {'capacity': 15, 'vehicles': 15}, 
            'long_day': {'capacity': 25, 'vehicles': 5}
        },
        cost_model={'elevation_beta': 10.0},
        drive_threshold_miles=2.0,
        base_pace_min_per_mile=16.0,
        short_day_limit=6.0,
        medium_day_limit=15.0,
        long_day_limit=25.0
    )

@pytest.fixture
def generated_plan(config):
    """Generate a test plan using the trailhead router"""
    router = TrailheadRouter(config)
    
    # Use test data if available
    required_path = "data/traildata/GETChallengeTrailData_v2.json"
    all_trails_path = "data/traildata/Boise_Parks_Trails_Open_Data.geojson"
    osm_path = "data/osm/idaho-latest.osm.pbf"
    
    if os.path.exists(required_path) and os.path.exists(all_trails_path):
        router.load_data(required_path, all_trails_path, osm_path)
        return router.generate_plan()
    else:
        # Return mock plan for testing
        # Create mock segments
        mock_segments = [
            TrailSegment(
                seg_id="test_1",
                name="Test Trail 1",
                coordinates=[(-116.2, 43.6), (-116.19, 43.61)],
                length_ft=1000,
                direction="both",
                required=True
            ),
            TrailSegment(
                seg_id="test_2", 
                name="Test Trail 2",
                coordinates=[(-116.19, 43.61), (-116.18, 43.62)],
                length_ft=1500,
                direction="both", 
                required=True
            )
        ]
        
        # Create mock hikes
        mock_hike = Hike(
            hike_number=1,
            trailhead="Camel's Back Park",
            segments=mock_segments,
            total_distance=0.5,
            elevation_gain=200,
            difficulty='moderate',
            trail_conditions="Good trail conditions",
            estimated_minutes=30,
            parking_coords=(43.635278, -116.205),
            parking_type='paved_lot',
            parking_fee=0.0,
            parking_notes="Free parking available"
        )
        
        # Add navigation points
        mock_hike.navigation_points = [
            NavigationPoint(
                distance_from_start=0,
                instruction="Start from Camel's Back Park parking",
                landmark="Trailhead",
                gps_coords=(43.635278, -116.205)
            ),
            NavigationPoint(
                distance_from_start=0.25,
                instruction="Continue on Test Trail 1",
                landmark="Trail junction",
                gps_coords=(43.61, -116.19)
            )
        ]
        
        mock_day = Day(
            number=1,
            hikes=[mock_hike],
            total_distance=0.5,
            total_elevation=200,
            type='short'
        )
        
        summary_stats = {
            'total_miles': 0.5,
            'required_miles': 0.5,
            'connector_miles': 0.0,
            'road_miles': 0.0,
            'redundancy_percent': 0.0,
            'efficiency_score': 100.0,
            'total_driving_miles': 0.0,
            'unique_trailheads': 1,
            'average_hikes_per_day': 1.0,
            'total_days': 1
        }
        
        return GeneratedPlan(days=[mock_day], summary_stats=summary_stats)
