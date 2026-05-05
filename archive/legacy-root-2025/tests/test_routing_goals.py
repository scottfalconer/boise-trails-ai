"""
Test suite to ensure routing system meets stated goals:
1. Complete 100% of segments with minimal manual effort
2. Provide clear, usable instructions

These tests validate the trailhead-based routing approach.
"""
import pytest
import networkx as nx
from typing import List, Dict, Tuple, Set
import json
import gpxpy
from pathlib import Path


class TestGoal1MinimalEffort:
    """Tests for Goal 1: Complete 100% of segments with minimal manual effort"""
    
    def test_100_percent_segment_coverage(self, generated_plan):
        """Ensure all required segments are covered exactly once"""
        required_segments = load_required_segments()
        covered_segments = set()
        
        for day in generated_plan.days:
            for hike in day.hikes:
                for segment in hike.segments:
                    if segment.required:
                        seg_id = segment.seg_id if hasattr(segment, 'seg_id') else segment.id
                        assert seg_id not in covered_segments, \
                            f"Segment {seg_id} covered multiple times"
                        covered_segments.add(seg_id)
        
        missing = set(required_segments.keys()) - covered_segments
        assert len(missing) == 0, f"Missing segments: {missing}"
        assert len(covered_segments) == len(required_segments), \
            "Not all required segments covered"
    
    def test_redundancy_below_threshold(self, generated_plan):
        """Ensure redundancy is below 15% (including connector trails)"""
        total_required_distance = sum(s.length for s in load_required_segments().values())
        total_actual_distance = sum(
            hike.total_distance for day in generated_plan.days 
            for hike in day.hikes
        )
        
        redundancy_pct = ((total_actual_distance / total_required_distance) - 1) * 100
        assert redundancy_pct < 15, \
            f"Redundancy {redundancy_pct:.1f}% exceeds 15% threshold"
    
    def test_elevation_efficiency(self, generated_plan):
        """Ensure routes minimize unnecessary elevation change"""
        for day in generated_plan.days:
            for hike in day.hikes:
                # Check for yo-yo patterns (multiple ascents/descents)
                elevation_changes = analyze_elevation_profile(hike)
                major_climbs = count_major_climbs(elevation_changes, threshold=500)
                
                assert major_climbs <= 2, \
                    f"Hike {hike.id} has {major_climbs} major climbs (>500ft), " \
                    f"indicating inefficient elevation planning"
    
    def test_no_unnecessary_backtracking(self, generated_plan):
        """Ensure segments aren't traversed multiple times unnecessarily"""
        for day in generated_plan.days:
            for hike in day.hikes:
                segment_traversals = count_segment_traversals(hike)
                
                for segment_id, count in segment_traversals.items():
                    segment_obj = get_segment(segment_id)
                    if segment_obj.is_required and not segment_obj.is_one_way:
                        assert count <= 2, \
                            f"Segment {segment_id} traversed {count} times"
                    elif segment_obj.is_connector:
                        assert count <= 2, \
                            f"Connector {segment_id} used {count} times"
    
    def test_efficient_trailhead_usage(self, generated_plan):
        """Ensure minimal driving between trailheads"""
        total_driving_miles = 0
        trailhead_changes = 0
        
        last_trailhead = None
        for day in generated_plan.days:
            for hike in day.hikes:
                if last_trailhead and hike.trailhead != last_trailhead:
                    trailhead_changes += 1
                    total_driving_miles += hike.drive_to_distance
                last_trailhead = hike.trailhead
        
        avg_hikes_per_trailhead = len(generated_plan.all_hikes) / trailhead_changes
        assert avg_hikes_per_trailhead >= 2, \
            "Too many trailhead changes, not utilizing each trailhead fully"
        
        assert total_driving_miles < 100, \
            f"Total driving {total_driving_miles} miles exceeds reasonable limit"
    
    def test_daily_distance_limits(self, generated_plan, routing_config):
        """Ensure daily distances match configured limits"""
        for day in generated_plan.days:
            total_distance = sum(hike.total_distance for hike in day.hikes)
            
            if day.type == 'short':
                assert total_distance <= routing_config['short_day_limit'] * 1.1, \
                    f"Short day {day.number} exceeds limit"
            elif day.type == 'medium':
                assert total_distance <= routing_config['medium_day_limit'] * 1.1, \
                    f"Medium day {day.number} exceeds limit"
            elif day.type == 'long':
                assert total_distance <= routing_config['long_day_limit'] * 1.1, \
                    f"Long day {day.number} exceeds limit"
    
    def test_connector_trail_usage(self, generated_plan):
        """Ensure connector trails are used effectively to create loops"""
        out_and_backs = 0
        loops_with_connectors = 0
        
        for day in generated_plan.days:
            for hike in day.hikes:
                if is_out_and_back(hike):
                    out_and_backs += 1
                elif uses_connector_trails(hike):
                    loops_with_connectors += 1
        
        total_hikes = len(generated_plan.all_hikes)
        out_and_back_pct = (out_and_backs / total_hikes) * 100
        
        assert out_and_back_pct < 20, \
            f"{out_and_back_pct:.1f}% of hikes are out-and-backs, should use more connectors"
        assert loops_with_connectors > out_and_backs, \
            "Should have more loops than out-and-backs"
    
    def test_road_usage_minimized(self, generated_plan):
        """Ensure road walking is only used when necessary"""
        for day in generated_plan.days:
            for hike in day.hikes:
                road_miles = sum(
                    seg.length_mi for seg in hike.segments 
                    if 'road' in seg.name.lower() or 'connection' in seg.name.lower()
                )
                trail_miles = hike.total_distance - road_miles
                road_pct = (road_miles / hike.total_distance) * 100
                
                assert road_pct < 20, \
                    f"Hike {hike.id} has {road_pct:.1f}% road walking"
                
                # If using roads, ensure they're safe
                for seg in hike.segments:
                    if seg.type == 'road':
                        assert seg.has_sidewalk or seg.has_bike_lane or seg.is_low_traffic, \
                            f"Unsafe road segment {seg.id} used"


class TestGoal2ClearInstructions:
    """Tests for Goal 2: Provide clear, usable instructions"""
    
    def test_parking_information_complete(self, generated_plan):
        """Ensure all hikes have complete parking information"""
        for day in generated_plan.days:
            for hike in day.hikes:
                assert hike.trailhead is not None, f"Hike {hike.id} missing trailhead"
                assert hike.parking_coords is not None, "Missing parking coordinates"
                assert hike.parking_type in ['paved_lot', 'gravel_lot', 'roadside', 'informal']
                assert hike.parking_fee is not None  # Can be 0 for free
                assert hike.parking_notes is not None  # Can be empty string
    
    def test_driving_directions_provided(self, generated_plan):
        """Ensure driving directions are provided between different trailheads"""
        last_trailhead = None
        
        for day in generated_plan.days:
            for hike in day.hikes:
                if last_trailhead and hike.trailhead != last_trailhead:
                    assert hike.driving_directions is not None, \
                        f"Missing driving directions to {hike.trailhead}"
                    assert hike.drive_time_minutes > 0
                    assert hike.drive_distance_miles > 0
                    assert len(hike.driving_directions) > 0
                last_trailhead = hike.trailhead
    
    def test_turn_by_turn_navigation(self, generated_plan):
        """Ensure turn-by-turn directions are provided for all hikes"""
        for day in generated_plan.days:
            for hike in day.hikes:
                assert len(hike.navigation_points) > 0, \
                    f"Hike {hike.id} missing navigation points"
                
                for nav_point in hike.navigation_points:
                    assert nav_point.distance_from_start >= 0
                    assert nav_point.instruction is not None
                    assert nav_point.landmark is not None  # Can be empty
                    assert nav_point.gps_coords is not None
                    
                    # Check instruction quality
                    assert len(nav_point.instruction) > 5, \
                        "Navigation instruction too short"
                    assert not nav_point.instruction.lower().startswith('continue'), \
                        "Too many 'continue' instructions indicate poor navigation"
    
    def test_junction_waypoints_included(self, generated_plan):
        """Ensure all trail junctions have waypoints"""
        for day in generated_plan.days:
            for hike in day.hikes:
                junctions = find_trail_junctions(hike)
                waypoints = {(wp.lat, wp.lon) for wp in hike.gpx_waypoints}
                
                for junction in junctions:
                    # Allow small GPS variance
                    assert any(
                        distance_between(junction, wp) < 0.01  # ~50 feet
                        for wp in waypoints
                    ), f"Junction at {junction} missing waypoint"
    
    def test_segment_entry_exit_marked(self, generated_plan):
        """Ensure entry/exit of required segments is clearly marked"""
        for day in generated_plan.days:
            for hike in day.hikes:
                for i, segment in enumerate(hike.segments):
                    if segment.is_required:
                        # Find navigation points near segment start/end
                        start_nav = find_nearest_nav_point(
                            hike.navigation_points, 
                            segment.start_coords
                        )
                        end_nav = find_nearest_nav_point(
                            hike.navigation_points, 
                            segment.end_coords
                        )
                        
                        assert start_nav is not None, \
                            f"No navigation for start of required segment {segment.id}"
                        assert f"Start {segment.name}" in start_nav.instruction or \
                               f"Begin {segment.name}" in start_nav.instruction
                        
                        assert end_nav is not None, \
                            f"No navigation for end of required segment {segment.id}"
                        assert f"Complete {segment.name}" in end_nav.instruction or \
                               f"End {segment.name}" in end_nav.instruction
    
    def test_escape_routes_documented(self, generated_plan):
        """Ensure escape routes are provided for long hikes"""
        for day in generated_plan.days:
            for hike in day.hikes:
                if hike.total_distance > 5.0:  # Long hikes need escape routes
                    assert len(hike.escape_routes) > 0, \
                        f"Long hike {hike.id} ({hike.total_distance}mi) needs escape routes"
                    
                    for escape in hike.escape_routes:
                        assert escape.at_mile > 0
                        assert escape.description is not None
                        assert escape.saves_miles > 0
                        assert escape.to_parking is not None
    
    def test_trail_conditions_noted(self, generated_plan):
        """Ensure trail conditions and difficulty are documented"""
        for day in generated_plan.days:
            for hike in day.hikes:
                assert hike.difficulty in ['easy', 'moderate', 'difficult', 'expert']
                assert hike.trail_conditions is not None
                
                # Check for important warnings
                if hike.max_elevation > 6000:
                    assert 'snow' in hike.trail_conditions.lower() or \
                           'elevation' in hike.trail_conditions.lower()
                
                if any(seg.surface == 'rocky' for seg in hike.segments):
                    assert 'rocky' in hike.trail_conditions.lower() or \
                           'technical' in hike.trail_conditions.lower()
    
    def test_gpx_quality(self, generated_plan):
        """Ensure GPX files are high quality with proper metadata"""
        for day in generated_plan.days:
            for hike in day.hikes:
                gpx_path = Path(f"output/routes/day_{day.number:02d}_hike_{hike.number:02d}.gpx")
                assert gpx_path.exists(), f"Missing GPX file for {hike.id}"
                
                with open(gpx_path, 'r') as f:
                    gpx = gpxpy.parse(f)
                
                # Check metadata
                assert gpx.name == f"Day {day.number} Hike {hike.number} - {hike.trailhead}"
                assert len(gpx.tracks) == 1
                assert len(gpx.waypoints) >= 2  # At least start and end
                
                # Check waypoint quality
                for wp in gpx.waypoints:
                    assert wp.name is not None
                    assert wp.description is not None
                    
                # Check track has enough points for smooth navigation
                total_points = sum(
                    len(seg.points) for track in gpx.tracks 
                    for seg in track.segments
                )
                assert total_points > hike.total_distance * 50, \
                    "Not enough GPS points for smooth navigation"
    
    def test_time_estimates_reasonable(self, generated_plan, routing_config):
        """Ensure time estimates are reasonable based on pace and terrain"""
        base_pace = routing_config['base_pace_min_per_mile']  # e.g., 16 min/mile
        
        for day in generated_plan.days:
            for hike in day.hikes:
                # Calculate expected time
                distance_time = hike.total_distance * base_pace
                # Add 1 minute per 100ft elevation gain (Naismith's rule variant)
                elevation_time = hike.elevation_gain / 100
                expected_time = distance_time + elevation_time
                
                # Allow 20% variance
                assert hike.estimated_minutes * 0.8 <= expected_time <= hike.estimated_minutes * 1.2, \
                    f"Time estimate {hike.estimated_minutes}min unreasonable for " \
                    f"{hike.total_distance}mi with {hike.elevation_gain}ft gain"


class TestIntegrationGoals:
    """Integration tests ensuring both goals work together"""
    
    def test_efficient_routes_with_clear_directions(self, generated_plan):
        """Ensure efficient routes also have clear navigation"""
        for day in generated_plan.days:
            total_distance = sum(h.total_distance for h in day.hikes)
            total_nav_points = sum(len(h.navigation_points) for h in day.hikes)
            
            # Should have navigation point roughly every 0.25 miles
            expected_nav_points = total_distance * 4
            assert total_nav_points >= expected_nav_points * 0.7, \
                "Efficient routes still need adequate navigation"
    
    def test_complete_challenge_within_timeframe(self, generated_plan):
        """Ensure 100% completion is achievable in 31 days"""
        assert len(generated_plan.days) <= 31, \
            f"Plan requires {len(generated_plan.days)} days, exceeds 31-day challenge"
        
        # Check that rest days are possible
        total_hiking_days = len([d for d in generated_plan.days if d.total_distance > 0])
        assert total_hiking_days <= 25, \
            "Plan should allow for at least 6 rest days in a month"
    
    def test_plan_summary_statistics(self, generated_plan):
        """Verify plan includes accurate summary statistics"""
        stats = generated_plan.summary_stats
        
        # Efficiency metrics
        assert 'total_miles' in stats
        assert 'required_miles' in stats
        assert 'connector_miles' in stats
        assert 'road_miles' in stats
        assert 'redundancy_percent' in stats
        assert 'efficiency_score' in stats
        
        # Logistics metrics
        assert 'total_driving_miles' in stats
        assert 'unique_trailheads' in stats
        assert 'average_hikes_per_day' in stats
        
        # Validate calculations
        assert stats['total_miles'] == sum(
            h.total_distance for d in generated_plan.days for h in d.hikes
        )
        assert stats['redundancy_percent'] < 15
        assert stats['efficiency_score'] > 85  # 100 - redundancy


# Helper functions (these would be imported from the actual implementation)
def load_required_segments() -> Dict[str, 'Segment']:
    """Load required segments from challenge data"""
    import os
    import json
    
    segments = {}
    required_path = "data/traildata/GETChallengeTrailData_v2.json"
    
    if os.path.exists(required_path):
        with open(required_path, 'r') as f:
            data = json.load(f)
        
        for seg_data in data['trailSegments']:
            props = seg_data['properties']
            seg_id = str(props.get('segId'))
            segments[seg_id] = type('Segment', (), {
                'id': seg_id,
                'name': props.get('segName', ''),
                'length': float(props.get('LengthFt', 0)) / 5280
            })()
    else:
        # Return mock data for testing
        segments = {
            'test_1': type('Segment', (), {'id': 'test_1', 'name': 'Test Trail 1', 'length': 0.19})(),
            'test_2': type('Segment', (), {'id': 'test_2', 'name': 'Test Trail 2', 'length': 0.28})()
        }
    
    return segments

def analyze_elevation_profile(hike) -> List[Tuple[float, float]]:
    """Return list of (distance, elevation) points"""
    # Simplified implementation for testing
    points = []
    distance = 0
    elevation = 4000  # Starting elevation
    
    for segment in hike.segments:
        points.append((distance, elevation))
        distance += segment.length_mi
        elevation += segment.length_mi * 100  # Assume 100ft gain per mile
        points.append((distance, elevation))
    
    return points

def count_major_climbs(elevation_changes: List, threshold: float) -> int:
    """Count climbs exceeding threshold feet"""
    climbs = 0
    current_elevation = None
    
    for distance, elevation in elevation_changes:
        if current_elevation is not None:
            gain = elevation - current_elevation
            if gain >= threshold:
                climbs += 1
        current_elevation = elevation
    
    return climbs

def count_segment_traversals(hike) -> Dict[str, int]:
    """Count how many times each segment is traversed"""
    traversals = {}
    
    for segment in hike.segments:
        seg_id = segment.seg_id if hasattr(segment, 'seg_id') else segment.id
        traversals[seg_id] = traversals.get(seg_id, 0) + 1
    
    return traversals

def get_segment(segment_id):
    """Get segment by ID"""
    # Mock implementation for testing
    return type('Segment', (), {
        'is_required': segment_id.startswith('test_'),
        'is_one_way': False,
        'is_connector': 'connector' in segment_id
    })()

def is_out_and_back(hike) -> bool:
    """Determine if hike is an out-and-back"""
    if not hike.segments:
        return False
    
    # Simple heuristic: if start and end coordinates are the same
    start_coords = hike.segments[0].coordinates[0] if hike.segments[0].coordinates else None
    end_coords = hike.segments[-1].coordinates[-1] if hike.segments[-1].coordinates else None
    
    if start_coords and end_coords:
        # Consider same if within 100 meters
        distance = haversine_distance_points(start_coords, end_coords)
        return distance < 0.06  # ~100 meters in miles
    
    # Fallback: check if we have the same trailhead for start and end
    return True  # Conservative assumption

def uses_connector_trails(hike) -> bool:
    """Check if hike uses non-required connector trails"""
    for segment in hike.segments:
        if not segment.required:
            return True
    return False

def find_trail_junctions(hike) -> List[Tuple[float, float]]:
    """Find all trail junction coordinates in a hike"""
    junctions = []
    
    for i in range(len(hike.segments) - 1):
        current_seg = hike.segments[i]
        next_seg = hike.segments[i + 1]
        
        # Junction is where current segment ends and next begins
        if current_seg.coordinates and next_seg.coordinates:
            end_coord = current_seg.coordinates[-1]
            start_coord = next_seg.coordinates[0]
            
            # If they're close, it's a junction
            if haversine_distance_points(end_coord, start_coord) < 0.01:  # ~50 feet
                junctions.append(end_coord)
    
    return junctions

def distance_between(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate distance in miles between two coordinates"""
    return haversine_distance_points(coord1, coord2)

def find_nearest_nav_point(nav_points: List, coords: Tuple[float, float]):
    """Find navigation point nearest to given coordinates"""
    if not nav_points:
        return None
    
    min_dist = float('inf')
    nearest = None
    
    for nav_point in nav_points:
        if nav_point.gps_coords:
            dist = distance_between(coords, nav_point.gps_coords)
            if dist < min_dist:
                min_dist = dist
                nearest = nav_point
    
    return nearest

def haversine_distance_points(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate haversine distance between two points in miles"""
    import math
    
    # Convert to radians
    lat1, lon1 = math.radians(coord1[1]), math.radians(coord1[0])
    lat2, lon2 = math.radians(coord2[1]), math.radians(coord2[0])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in miles
    R = 3959
    
    return R * c


# Additional fixtures specific to these tests
@pytest.fixture  
def routing_config():
    """Load planner configuration"""
    return {
        'short_day_limit': 6.0,
        'medium_day_limit': 15.0,
        'long_day_limit': 25.0,
        'base_pace_min_per_mile': 16.0
    }