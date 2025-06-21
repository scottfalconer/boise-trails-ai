"""
Tests for efficiency metrics as defined in AGENTS.md.

These tests validate the key evaluation metrics:
- Progress (Distance/Elevation) %
- % Over Target (Distance/Elevation)  
- Efficiency Score (Distance/Elevation)
- Redundancy minimization
"""
import pytest
from typing import List, Dict, Tuple
from trail_route_ai.planner_utils import Edge


class TestEfficiencyMetrics:
    """Test the efficiency metrics defined in AGENTS.md."""
    
    def test_progress_percentage_calculation(self):
        """
        Test Progress % = (Total New Official Trail Distance / Challenge Target Distance) * 100
        """
        # Sample route with official and non-official segments
        route = [
            Edge(seg_id="off1", name="Official Trail 1", start=(0,0), end=(1,0), 
                 length_mi=2.0, elev_gain_ft=200, coords=[(0,0), (1,0)], kind="trail"),
            Edge(seg_id="off2", name="Official Trail 2", start=(1,0), end=(2,0), 
                 length_mi=3.0, elev_gain_ft=300, coords=[(1,0), (2,0)], kind="trail"),
            Edge(seg_id=None, name="Connector", start=(2,0), end=(3,0), 
                 length_mi=0.5, elev_gain_ft=50, coords=[(2,0), (3,0)], kind="connector"),
        ]
        
        # Calculate metrics
        official_distance = sum(e.length_mi for e in route if e.seg_id and e.kind == "trail")
        official_elevation = sum(e.elev_gain_ft for e in route if e.seg_id and e.kind == "trail")
        
        target_distance = 169.35
        target_elevation = 36000.0
        
        progress_distance_pct = (official_distance / target_distance) * 100
        progress_elevation_pct = (official_elevation / target_elevation) * 100
        
        # Should calculate correctly
        assert abs(progress_distance_pct - (5.0 / 169.35) * 100) < 0.01
        assert abs(progress_elevation_pct - (500.0 / 36000.0) * 100) < 0.01
    
    def test_over_target_percentage_calculation(self):
        """
        Test % Over Target = ((Total On-Foot Distance / Challenge Target Distance) - 1) * 100
        """
        route = [
            Edge(seg_id="off1", name="Official Trail 1", start=(0,0), end=(1,0), 
                 length_mi=2.0, elev_gain_ft=200, coords=[(0,0), (1,0)], kind="trail"),
            Edge(seg_id=None, name="Connector", start=(1,0), end=(2,0), 
                 length_mi=1.0, elev_gain_ft=100, coords=[(1,0), (2,0)], kind="connector"),
            Edge(seg_id=None, name="Road", start=(2,0), end=(3,0), 
                 length_mi=0.5, elev_gain_ft=0, coords=[(2,0), (3,0)], kind="road"),
        ]
        
        total_distance = sum(e.length_mi for e in route)
        total_elevation = sum(e.elev_gain_ft for e in route)
        
        target_distance = 2.0  # Only the official segment
        target_elevation = 200.0
        
        over_target_distance_pct = ((total_distance / target_distance) - 1) * 100
        over_target_elevation_pct = ((total_elevation / target_elevation) - 1) * 100
        
        # Should show 75% over target distance (3.5 total vs 2.0 target)
        expected_over_distance = ((3.5 / 2.0) - 1) * 100  # 75%
        expected_over_elevation = ((300.0 / 200.0) - 1) * 100  # 50%
        
        assert abs(over_target_distance_pct - expected_over_distance) < 0.01
        assert abs(over_target_elevation_pct - expected_over_elevation) < 0.01
    
    def test_efficiency_score_calculation(self):
        """
        Test Efficiency Score = (Challenge Target Distance / Total On-Foot Distance) * 100
        """
        route = [
            Edge(seg_id="off1", name="Official Trail 1", start=(0,0), end=(1,0), 
                 length_mi=10.0, elev_gain_ft=1000, coords=[(0,0), (1,0)], kind="trail"),
            Edge(seg_id=None, name="Extra", start=(1,0), end=(2,0), 
                 length_mi=2.0, elev_gain_ft=200, coords=[(1,0), (2,0)], kind="connector"),
        ]
        
        total_distance = sum(e.length_mi for e in route)  # 12.0
        total_elevation = sum(e.elev_gain_ft for e in route)  # 1200
        
        target_distance = 10.0
        target_elevation = 1000.0
        
        efficiency_distance = (target_distance / total_distance) * 100
        efficiency_elevation = (target_elevation / total_elevation) * 100
        
        # Should be 83.33% efficient for distance, 83.33% for elevation
        assert abs(efficiency_distance - 83.33) < 0.01
        assert abs(efficiency_elevation - 83.33) < 0.01
    
    def test_redundancy_minimization_goal(self):
        """
        Test that redundancy (non-official mileage) is tracked and minimized.
        """
        # Route with some redundancy
        route = [
            Edge(seg_id="off1", name="Official 1", start=(0,0), end=(1,0), 
                 length_mi=5.0, elev_gain_ft=500, coords=[(0,0), (1,0)], kind="trail"),
            Edge(seg_id="off2", name="Official 2", start=(1,0), end=(2,0), 
                 length_mi=3.0, elev_gain_ft=300, coords=[(1,0), (2,0)], kind="trail"),
            Edge(seg_id=None, name="Backtrack", start=(2,0), end=(1,0), 
                 length_mi=1.0, elev_gain_ft=0, coords=[(2,0), (1,0)], kind="connector"),  # Redundant
            Edge(seg_id=None, name="Road Return", start=(1,0), end=(0,0), 
                 length_mi=0.8, elev_gain_ft=0, coords=[(1,0), (0,0)], kind="road"),  # Redundant
        ]
        
        official_distance = sum(e.length_mi for e in route if e.seg_id and e.kind == "trail")
        total_distance = sum(e.length_mi for e in route)
        redundant_distance = total_distance - official_distance
        
        redundancy_ratio = redundant_distance / official_distance
        
        # Should track redundancy correctly
        assert official_distance == 8.0
        assert total_distance == 9.8
        assert abs(redundant_distance - 1.8) < 0.001  # Handle floating point precision
        assert abs(redundancy_ratio - (1.8 / 8.0)) < 0.01  # 22.5% redundancy


class TestRouteQualityMetrics:
    """Test route quality assessment based on AGENTS.md goals."""
    
    def test_loop_closure_validation(self):
        """
        Test that routes properly return to start (loop closure requirement).
        """
        # Good loop route
        good_loop = [
            Edge(seg_id="1", name="Trail 1", start=(0,0), end=(1,0), 
                 length_mi=1.0, elev_gain_ft=100, coords=[(0,0), (1,0)]),
            Edge(seg_id="2", name="Trail 2", start=(1,0), end=(1,1), 
                 length_mi=1.0, elev_gain_ft=100, coords=[(1,0), (1,1)]),
            Edge(seg_id=None, name="Return", start=(1,1), end=(0,0), 
                 length_mi=1.4, elev_gain_ft=0, coords=[(1,1), (0,0)]),
        ]
        
        # Bad non-loop route
        bad_route = [
            Edge(seg_id="1", name="Trail 1", start=(0,0), end=(1,0), 
                 length_mi=1.0, elev_gain_ft=100, coords=[(0,0), (1,0)]),
            Edge(seg_id="2", name="Trail 2", start=(1,0), end=(5,5), 
                 length_mi=1.0, elev_gain_ft=100, coords=[(1,0), (5,5)]),
        ]
        
        def is_loop_closed(route: List[Edge], tolerance: float = 0.001) -> bool:
            if not route:
                return True
            start = route[0].start_actual
            end = route[-1].end_actual
            distance = ((start[0] - end[0])**2 + (start[1] - end[1])**2)**0.5
            return distance <= tolerance
        
        assert is_loop_closed(good_loop), "Good loop should be detected as closed"
        assert not is_loop_closed(bad_route), "Bad route should be detected as non-loop"
    
    def test_daily_feasibility_assessment(self):
        """
        Test that daily routes are feasible within reasonable time limits.
        """
        # Reasonable daily route
        reasonable_route = [
            Edge(seg_id="1", name="Trail 1", start=(0,0), end=(1,0), 
                 length_mi=8.0, elev_gain_ft=2000, coords=[(0,0), (1,0)]),
        ]
        
        # Excessive daily route
        excessive_route = [
            Edge(seg_id="1", name="Monster Trail", start=(0,0), end=(1,0), 
                 length_mi=40.0, elev_gain_ft=15000, coords=[(0,0), (1,0)]),
        ]
        
        def estimate_route_time_hours(route: List[Edge], pace_min_per_mile: float = 16.0) -> float:
            """Estimate route time using AGENTS.md pace assumptions."""
            total_distance = sum(e.length_mi for e in route)
            total_elevation = sum(e.elev_gain_ft for e in route)
            
            # 16 min/mile + 1 min per 100ft elevation
            moving_time_min = total_distance * pace_min_per_mile + total_elevation * 0.01
            return moving_time_min / 60.0
        
        reasonable_time = estimate_route_time_hours(reasonable_route)
        excessive_time = estimate_route_time_hours(excessive_route)
        
        # Reasonable route should be under 8 hours
        assert reasonable_time < 8.0, f"Reasonable route takes {reasonable_time:.1f} hours"
        
        # Excessive route should be flagged as too long
        assert excessive_time > 12.0, f"Excessive route only takes {excessive_time:.1f} hours"
    
    def test_connector_usage_efficiency(self):
        """
        Test that connector trails are used efficiently (not excessively).
        """
        # Efficient use of connectors
        efficient_route = [
            Edge(seg_id="off1", name="Official 1", start=(0,0), end=(1,0), 
                 length_mi=4.0, elev_gain_ft=400, coords=[(0,0), (1,0)], kind="trail"),
            Edge(seg_id="off2", name="Official 2", start=(2,0), end=(3,0), 
                 length_mi=4.0, elev_gain_ft=400, coords=[(2,0), (3,0)], kind="trail"),
            Edge(seg_id=None, name="Short Connector", start=(1,0), end=(2,0), 
                 length_mi=0.5, elev_gain_ft=0, coords=[(1,0), (2,0)], kind="connector"),
        ]
        
        # Inefficient overuse of connectors
        inefficient_route = [
            Edge(seg_id="off1", name="Official 1", start=(0,0), end=(1,0), 
                 length_mi=2.0, elev_gain_ft=200, coords=[(0,0), (1,0)], kind="trail"),
            Edge(seg_id=None, name="Long Connector 1", start=(1,0), end=(5,0), 
                 length_mi=4.0, elev_gain_ft=0, coords=[(1,0), (5,0)], kind="connector"),
            Edge(seg_id=None, name="Long Connector 2", start=(5,0), end=(10,0), 
                 length_mi=5.0, elev_gain_ft=0, coords=[(5,0), (10,0)], kind="connector"),
        ]
        
        def calculate_connector_ratio(route: List[Edge]) -> float:
            """Calculate ratio of connector distance to official distance."""
            official_dist = sum(e.length_mi for e in route if e.seg_id and e.kind == "trail")
            connector_dist = sum(e.length_mi for e in route if not e.seg_id or e.kind != "trail")
            return connector_dist / official_dist if official_dist > 0 else 0
        
        efficient_ratio = calculate_connector_ratio(efficient_route)
        inefficient_ratio = calculate_connector_ratio(inefficient_route)
        
        # Efficient route should have low connector ratio
        assert efficient_ratio < 0.2, f"Efficient route has {efficient_ratio:.2%} connector ratio"
        
        # Inefficient route should have high connector ratio
        assert inefficient_ratio > 1.0, f"Inefficient route has {inefficient_ratio:.2%} connector ratio"


@pytest.fixture
def sample_daily_plan():
    """Sample daily plan for testing metrics."""
    return {
        'date': '2025-06-19',
        'routes': [
            [
                Edge(seg_id="dc1", name="Dry Creek 1", start=(0,0), end=(1,0), 
                     length_mi=2.0, elev_gain_ft=200, coords=[(0,0), (1,0)], kind="trail"),
                Edge(seg_id="dc2", name="Dry Creek 2", start=(1,0), end=(2,0), 
                     length_mi=2.0, elev_gain_ft=200, coords=[(1,0), (2,0)], kind="trail"),
                Edge(seg_id=None, name="Return Connector", start=(2,0), end=(0,0), 
                     length_mi=2.8, elev_gain_ft=0, coords=[(2,0), (0,0)], kind="connector"),
            ]
        ],
        'total_distance_mi': 6.8,
        'official_distance_mi': 4.0,
        'total_elevation_ft': 400,
        'estimated_time_hours': 2.1
    } 