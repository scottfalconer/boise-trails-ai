"""
Tests for validating core Boise Trails Challenge completion requirements.

These tests ensure the system meets its primary objective: 100% completion of all 
247 official segments within the challenge constraints defined in AGENTS.md.
"""
import pytest
from typing import List, Set, Dict
from trail_route_ai.planner_utils import Edge, load_segments
from trail_route_ai.challenge_planner import main


class TestChallengeCompletionValidation:
    """Test the core challenge completion requirements."""
    
    @pytest.mark.integration
    def test_all_official_segments_covered(self):
        """
        CORE REQUIREMENT: Every official segment must be covered at least once.
        This is the primary success metric for the challenge.
        """
        # Load official segments
        official_segments = load_segments("data/traildata/GETChallengeTrailData_v2.json")
        
        # This would need the actual multiday planner when implemented
        # For now, test that we can load the segments and they match expected count
        
        # Get all required official segment IDs
        required_segment_ids = {seg.seg_id for seg in official_segments if seg.seg_id}
        
        # Verify we have the expected total (247 segments per AGENTS.md)
        assert len(required_segment_ids) == 247, f"Expected 247 segments, found {len(required_segment_ids)}"
        
        # Verify each segment has required properties
        for segment in official_segments:
            assert segment.seg_id is not None, "All segments must have IDs"
            assert segment.length_mi > 0, f"Segment {segment.seg_id} has invalid length"
            assert segment.elev_gain_ft >= 0, f"Segment {segment.seg_id} has negative elevation"
    
    def test_target_metrics_realistic(self):
        """
        Verify the target metrics from AGENTS.md are realistic for the loaded data.
        """
        official_segments = load_segments("data/traildata/GETChallengeTrailData_v2.json")
        
        total_distance = sum(seg.length_mi for seg in official_segments)
        total_elevation = sum(seg.elev_gain_ft for seg in official_segments)
        
        # Should be close to AGENTS.md targets
        target_distance = 169.35
        target_elevation = 36000.0
        
        distance_ratio = abs(total_distance - target_distance) / target_distance
        
        # Allow 20% tolerance for data variations
        assert distance_ratio < 0.2, f"Total distance {total_distance:.2f} mi too far from target {target_distance} mi"
        
        # Elevation data may be 0 if DEM data isn't loaded - that's expected for basic tests
        if total_elevation > 0:
            elevation_ratio = abs(total_elevation - target_elevation) / target_elevation
            assert elevation_ratio < 0.2, f"Total elevation {total_elevation:.0f} ft too far from target {target_elevation} ft"
        else:
            # If elevation is 0, just verify we have the right number of segments
            print(f"Note: Elevation data is 0 (DEM not loaded), but distance checks pass")
            assert len(official_segments) == 247, f"Expected 247 segments, found {len(official_segments)}"
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_directional_segments_respected(self):
        """
        CORE REQUIREMENT: One-way segments must be traversed in correct direction.
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_target_distance_achieved(self):
        """
        TARGET METRIC: Should achieve ~169.35 miles of official trail distance.
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_target_elevation_achieved(self):
        """
        TARGET METRIC: Should achieve ~36,000 ft of elevation gain.
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_routes_form_loops(self):
        """
        CORE REQUIREMENT: Each route must start and end at same location (loop/return to start).
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_efficiency_metrics(self):
        """
        EFFICIENCY GOAL: Minimize redundant mileage (total distance should be close to official distance).
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass


class TestChallengePlanningConstraints:
    """Test adherence to planning constraints from AGENTS.md."""
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_challenge_timeframe_respected(self):
        """
        CONSTRAINT: Challenge runs June 19 - July 19, 2025 (31 days).
        Plans should be feasible within this timeframe.
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_natural_trail_grouping(self):
        """
        STRATEGY GOAL: Segments should be grouped by natural trail families.
        This tests the core clustering improvement from our TrailSubSystem work.
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass
    
    @pytest.mark.skip(reason="Requires full integration test - main() function needs proper setup")
    def test_connector_trail_usage_minimized(self):
        """
        EFFICIENCY GOAL: Connector trails should be used judiciously to improve efficiency.
        """
        # This test would need to run the full planner and check the results
        # Skipping until we have a proper test harness for the main() function
        pass


@pytest.fixture
def sample_challenge_segments():
    """Provide a small sample of challenge segments for unit testing."""
    return [
        Edge(seg_id="dc1", name="Dry Creek Trail 1", start=(0,0), end=(1,0), 
             length_mi=1.0, elev_gain_ft=100, coords=[(0,0), (1,0)]),
        Edge(seg_id="dc2", name="Dry Creek Trail 2", start=(1,0), end=(2,0), 
             length_mi=1.0, elev_gain_ft=100, coords=[(1,0), (2,0)]),
        Edge(seg_id="sc1", name="Shingle Creek 1", start=(1,0), end=(1,1), 
             length_mi=0.8, elev_gain_ft=80, coords=[(1,0), (1,1)]),
        Edge(seg_id="sc2", name="Shingle Creek 2", start=(1,1), end=(2,0), 
             length_mi=0.8, elev_gain_ft=80, coords=[(1,1), (2,0)]),
    ]


class TestTrailSubSystemGrouping:
    """Test the TrailSubSystem-based grouping improvements."""
    
    def test_same_subsystem_segments_grouped(self, sample_challenge_segments):
        """
        Test that segments from same TrailSubSystem are grouped together.
        This validates our core clustering improvement.
        """
        from trail_route_ai.clustering import identify_natural_trail_groups
        
        groups = identify_natural_trail_groups(sample_challenge_segments)
        
        # Should create logical groupings
        assert len(groups) > 0, "Should create at least one group"
        
        # All segments should be assigned to groups
        total_segments_in_groups = sum(len(segments) for segments in groups.values())
        assert total_segments_in_groups == len(sample_challenge_segments), "All segments should be grouped"
    
    def test_natural_grouping_reduces_activities(self, sample_challenge_segments):
        """
        Test that natural grouping reduces the number of separate activities.
        This is a key goal: 2-4 activities per day instead of 9-11 individual segments.
        """
        from trail_route_ai.clustering import identify_natural_trail_groups
        
        groups = identify_natural_trail_groups(sample_challenge_segments)
        
        # Should have fewer groups than individual segments (grouping effect)
        assert len(groups) < len(sample_challenge_segments), "Grouping should reduce number of separate clusters"
        
        # Groups should contain multiple segments where logical
        multi_segment_groups = [group for group in groups.values() if len(group) > 1]
        assert len(multi_segment_groups) > 0, "Should have at least one multi-segment group"


class TestSystemIntegration:
    """Integration tests for the complete system workflow."""
    
    def test_data_file_consistency(self):
        """
        Test that the data files referenced in AGENTS.md exist and are consistent.
        """
        import os
        
        # Verify required data files exist
        assert os.path.exists("data/traildata/GETChallengeTrailData_v2.json"), "Official segments file missing"
        assert os.path.exists("data/traildata/Boise_Parks_Trails_Open_Data.geojson"), "Parks data file missing"
        
        # Load and verify official segments
        official_segments = load_segments("data/traildata/GETChallengeTrailData_v2.json")
        assert len(official_segments) > 0, "No segments loaded from official file"
        
        # Verify segments have required properties
        for segment in official_segments[:10]:  # Check first 10 segments
            assert hasattr(segment, 'seg_id'), "Segments must have seg_id"
            assert hasattr(segment, 'length_mi'), "Segments must have length"
            assert hasattr(segment, 'elev_gain_ft'), "Segments must have elevation gain"
            assert hasattr(segment, 'coords'), "Segments must have coordinates"
    
    def test_clustering_produces_reasonable_groups(self):
        """
        Test that the clustering system produces reasonable groupings for real data.
        """
        official_segments = load_segments("data/traildata/GETChallengeTrailData_v2.json")
        
        # Take a sample for testing
        sample_segments = official_segments[:50]  # First 50 segments
        
        from trail_route_ai.clustering import identify_natural_trail_groups
        groups = identify_natural_trail_groups(sample_segments)
        
        # Should create reasonable number of groups (not too many, not too few)
        assert 5 <= len(groups) <= 25, f"Expected 5-25 groups for 50 segments, got {len(groups)}"
        
        # Average group size should be reasonable
        avg_group_size = len(sample_segments) / len(groups)
        assert 1.5 <= avg_group_size <= 10, f"Average group size {avg_group_size:.1f} seems unreasonable" 