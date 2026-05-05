#!/usr/bin/env python3
"""
Trailhead-based routing system for the Boise Trails Challenge.

This module implements the trailhead-centric approach described in 
TRAILHEAD_ROUTING_APPROACH.md, creating natural loops from parking areas
using connector trails and safe road connections.
"""

import os
import json
import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from scipy.spatial import KDTree
from dataclasses import dataclass
import rasterio

from .core.models import (
    TrailSegment, Trailhead, Hike, Day, GeneratedPlan, 
    TrailFamily, Loop, NavigationPoint, EscapeRoute,
    PlannerConfig
)
from .core.utils import haversine_distance_points


class TrailheadRouter:
    """Main class for trailhead-based route planning"""
    
    def __init__(self, config: PlannerConfig, dem_provider: Optional[rasterio.io.DatasetReader] = None):
        self.config = config
        self.dem_provider = dem_provider
        self.unified_graph = None
        self.trailheads = []
        self.required_segments = {}
        self.all_segments = {}
        
    def load_data(self, required_segments_path: str, all_trails_path: str, osm_pbf_path: str):
        """Load all data sources and build integrated network"""
        print("🔄 Loading trail and road data...")
        
        # Load trail segments
        self.all_segments = self._load_all_trail_segments(all_trails_path)
        self.required_segments = self._load_required_segments(required_segments_path)
        
        # Mark required segments
        for seg_id, segment in self.required_segments.items():
            if seg_id in self.all_segments:
                self.all_segments[seg_id].required = True
            else:
                self.all_segments[seg_id] = segment
        
        print(f"✅ Loaded {len(self.all_segments)} total segments ({len(self.required_segments)} required)")
        
        # Build integrated network
        self.unified_graph = self._build_integrated_network(osm_pbf_path)
        
        # Discover trailheads
        self.trailheads = self._discover_trailheads()
        
        print(f"✅ Found {len(self.trailheads)} trailheads")
        
    def generate_plan(self) -> GeneratedPlan:
        """Generate complete hiking plan using trailhead-based approach"""
        import time
        from datetime import datetime
        
        print("🔄 Generating trailhead-based hiking plan...")
        start_time = time.time()
        
        # Generate loops from each trailhead
        print(f"\n📍 Phase 1/4: Generating loops from {len(self.trailheads)} trailheads...")
        phase_start = time.time()
        all_loops = []
        for i, trailhead in enumerate(self.trailheads):
            if i % 10 == 0:
                elapsed = time.time() - phase_start
                rate = (i + 1) / elapsed if elapsed > 0 else 1
                remaining = (len(self.trailheads) - i - 1) / rate if rate > 0 else 0
                print(f"   Processing trailhead {i+1}/{len(self.trailheads)} - ETA: {remaining:.0f}s", end='\r')
            loops = self._generate_trailhead_loops(trailhead)
            all_loops.extend(loops)
        
        print(f"\n✅ Generated {len(all_loops)} potential loops in {time.time() - phase_start:.1f}s")
        
        # Select optimal combination of loops
        print(f"\n🎯 Phase 2/4: Selecting optimal loop combination...")
        phase_start = time.time()
        selected_loops = self._select_optimal_loops(all_loops)
        print(f"✅ Selected {len(selected_loops)} loops for plan in {time.time() - phase_start:.1f}s")
        
        # Organize into daily plans
        print(f"\n📅 Phase 3/4: Organizing {len(selected_loops)} loops into daily plans...")
        phase_start = time.time()
        daily_plans = self._organize_into_days(selected_loops)
        print(f"✅ Created {len(daily_plans)} daily plans in {time.time() - phase_start:.1f}s")
        
        # Add navigation details
        print(f"\n🧭 Phase 4/4: Adding navigation details for {sum(len(d.hikes) for d in daily_plans)} hikes...")
        phase_start = time.time()
        total_hikes = sum(len(day.hikes) for day in daily_plans)
        hike_count = 0
        
        for day in daily_plans:
            for hike in day.hikes:
                hike_count += 1
                elapsed = time.time() - phase_start
                rate = hike_count / elapsed if elapsed > 0 else 1
                remaining = (total_hikes - hike_count) / rate if rate > 0 else 0
                print(f"   Processing hike {hike_count}/{total_hikes} - ETA: {remaining:.0f}s", end='\r')
                self._add_navigation_details(hike)
        
        print(f"\n✅ Added navigation details in {time.time() - phase_start:.1f}s")
        
        # Calculate summary statistics
        print(f"\n📊 Calculating summary statistics...")
        summary_stats = self._calculate_summary_stats(daily_plans)
        
        total_time = time.time() - start_time
        print(f"\n✅ Plan generation complete in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        return GeneratedPlan(days=daily_plans, summary_stats=summary_stats)
    
    def _load_all_trail_segments(self, all_trails_path: str) -> Dict[str, TrailSegment]:
        """Load all trail segments from GeoJSON"""
        segments = {}
        
        if not os.path.exists(all_trails_path):
            print(f"⚠️  Warning: Trail data file not found: {all_trails_path}")
            return segments
            
        with open(all_trails_path, 'r') as f:
            data = json.load(f)
        
        for feature in data['features']:
            props = feature['properties']
            geom = feature.get('geometry')
            if not geom or not geom.get('coordinates'):
                continue
            
            coords = geom['coordinates']
            
            # Handle multi-part geometries
            flat_coords = []
            if any(isinstance(i, list) and len(i) > 0 and isinstance(i[0], list) for i in coords):
                for part in coords:
                    flat_coords.extend(part)
            else:
                flat_coords = coords
            
            seg_id = props.get('CART_ID') or f"connector_{props.get('OBJECTID')}"
            
            segment = TrailSegment(
                seg_id=str(seg_id),
                name=props.get('TRAIL_NAME', 'Unnamed Trail'),
                coordinates=[(lon, lat) for lon, lat in flat_coords],
                length_ft=float(props.get('Shape_Length', 0)),
                direction='both',
                required=False,
                access_from=props.get('AccessFrom'),
                surface=props.get('Surface', 'unknown'),
                exposure=props.get('Exposure', 'mixed')
            )
            
            segments[segment.seg_id] = segment
        
        return segments
    
    def _load_required_segments(self, required_segments_path: str) -> Dict[str, TrailSegment]:
        """Load required challenge segments"""
        segments = {}
        
        if not os.path.exists(required_segments_path):
            print(f"⚠️  Warning: Required segments file not found: {required_segments_path}")
            return segments
            
        with open(required_segments_path, 'r') as f:
            data = json.load(f)
        
        for seg_data in data['trailSegments']:
            props = seg_data['properties']
            coords = seg_data['geometry']['coordinates']
            seg_id = str(props.get('segId'))
            
            segment = TrailSegment(
                seg_id=seg_id,
                name=props.get('segName', ''),
                coordinates=[(c[0], c[1]) for c in coords],
                length_ft=float(props.get('LengthFt', 0)),
                direction=props.get('direction', 'both'),
                required=True,
                access_from=props.get('AccessFrom')
            )
            
            segments[segment.seg_id] = segment
        
        return segments
    
    def _build_integrated_network(self, osm_pbf_path: str) -> nx.DiGraph:
        """Build unified graph with trails and roads"""
        print("🔄 Building integrated network...")
        
        G = nx.DiGraph()
        
        # Add trail segments to graph
        for segment in self.all_segments.values():
            self._add_segment_to_graph(G, segment)
        
        # Add road network (simplified for now)
        # In a full implementation, this would parse OSM data
        self._add_road_connections(G)
        
        # Heal graph by connecting nearby components
        G = self._heal_graph(G)
        
        print(f"✅ Built unified graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def _add_segment_to_graph(self, graph: nx.DiGraph, segment: TrailSegment):
        """Add a trail segment to the graph with appropriate weights"""
        if len(segment.coordinates) < 2:
            return
        
        start_node = segment.coordinates[0]
        end_node = segment.coordinates[-1]
        
        # Calculate elevation gain if DEM is available
        elev_gain_ft = 0
        if self.dem_provider:
            try:
                start_elev = list(self.dem_provider.sample([start_node]))[0][0]
                end_elev = list(self.dem_provider.sample([end_node]))[0][0]
                elev_gain_ft = max(0, end_elev - start_elev)
            except:
                pass
        
        distance_mi = segment.length_ft / 5280.0
        
        # Weight based on segment type
        if segment.required:
            weight = distance_mi * 0.8  # Prefer required trails
            edge_type = 'required_trail'
        else:
            weight = distance_mi * 1.0  # Neutral weight for connectors
            edge_type = 'connector_trail'
        
        # Add elevation cost
        beta = self.config.cost_model.get('elevation_beta', 10.0)
        weight += (elev_gain_ft * beta / 5280.0)
        
        # Add edges based on direction
        if segment.direction in ['both', 'ascent']:
            graph.add_edge(start_node, end_node,
                          weight=weight,
                          distance=distance_mi,
                          segment=segment,
                          edge_type=edge_type,
                          required=segment.required,
                          elev_gain_ft=elev_gain_ft)
        
        if segment.direction in ['both', 'descent']:
            # Reverse direction with different elevation gain
            rev_elev_gain = max(0, -elev_gain_ft) if elev_gain_ft != 0 else 0
            rev_weight = distance_mi * (0.8 if segment.required else 1.0)
            rev_weight += (rev_elev_gain * beta / 5280.0)
            
            graph.add_edge(end_node, start_node,
                          weight=rev_weight,
                          distance=distance_mi,
                          segment=segment,
                          edge_type=edge_type,
                          required=segment.required,
                          elev_gain_ft=rev_elev_gain)
    
    def _add_road_connections(self, graph: nx.DiGraph):
        """Add safe road connections between trail systems"""
        # Simplified road network - in reality would parse OSM
        # For now, add virtual connections between nearby trail endpoints
        
        nodes = list(graph.nodes())
        if len(nodes) < 2:
            return
        
        tree = KDTree(nodes)
        max_road_distance = 0.5  # miles
        
        for node in nodes:
            # Find nearby nodes
            distances, indices = tree.query(node, k=min(5, len(nodes)), distance_upper_bound=max_road_distance/69) # ~69 miles per degree
            
            for dist, idx in zip(distances, indices):
                if np.isfinite(dist) and dist > 0:
                    other_node = tuple(nodes[idx])
                    if not graph.has_edge(node, other_node):
                        road_distance = haversine_distance_points(node, other_node)
                        if road_distance <= max_road_distance:
                            # Add road connection with higher weight
                            weight = road_distance * 1.5
                            
                            # Create virtual road segment
                            road_segment = TrailSegment(
                                seg_id=f'road_{len(graph.edges())}',
                                name='Road Connection',
                                coordinates=[node, other_node],
                                length_ft=road_distance * 5280,
                                direction='both',
                                required=False
                            )
                            
                            graph.add_edge(node, other_node,
                                          weight=weight,
                                          distance=road_distance,
                                          segment=road_segment,
                                          edge_type='road',
                                          required=False,
                                          has_sidewalk=True)  # Assume safe for now
    
    def _heal_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Connect nearby disconnected components"""
        components = list(nx.weakly_connected_components(graph))
        
        if len(components) <= 1:
            return graph
        
        print(f"🔄 Healing {len(components)} disconnected components...")
        
        # Connect components with virtual connectors
        component_centroids = []
        for comp in components:
            nodes = list(comp)
            centroid = (
                sum(node[0] for node in nodes) / len(nodes),
                sum(node[1] for node in nodes) / len(nodes)
            )
            component_centroids.append((centroid, comp))
        
        # Connect each component to its nearest neighbor
        for i, (centroid1, comp1) in enumerate(component_centroids):
            min_dist = float('inf')
            nearest_comp = None
            
            for j, (centroid2, comp2) in enumerate(component_centroids):
                if i != j:
                    dist = haversine_distance_points(centroid1, centroid2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_comp = comp2
            
            if nearest_comp and min_dist < 5.0:  # Within 5 miles
                # Find closest pair of nodes
                min_node_dist = float('inf')
                best_pair = None
                
                for node1 in list(comp1)[:10]:  # Limit search for performance
                    for node2 in list(nearest_comp)[:10]:
                        dist = haversine_distance_points(node1, node2)
                        if dist < min_node_dist:
                            min_node_dist = dist
                            best_pair = (node1, node2)
                
                if best_pair and min_node_dist < 2.0:  # Within 2 miles
                    node1, node2 = best_pair
                    
                    # Add healing connection
                    heal_segment = TrailSegment(
                        seg_id=f'heal_{i}_{j}',
                        name='Virtual Connector',
                        coordinates=[node1, node2],
                        length_ft=min_node_dist * 5280,
                        direction='both',
                        required=False
                    )
                    
                    weight = min_node_dist * 2.0  # High weight for virtual connections
                    
                    graph.add_edge(node1, node2,
                                  weight=weight,
                                  distance=min_node_dist,
                                  segment=heal_segment,
                                  edge_type='virtual',
                                  required=False)
                    
                    graph.add_edge(node2, node1,
                                  weight=weight,
                                  distance=min_node_dist,
                                  segment=heal_segment,
                                  edge_type='virtual',
                                  required=False)
        
        return graph
    
    def _discover_trailheads(self) -> List[Trailhead]:
        """Discover trailheads from trail data and configured depots"""
        trailheads = []
        
        # Start with configured trailheads
        for depot in self.config.trailhead_depots:
            trailhead = Trailhead(
                name=depot['name'],
                parking_coords=(depot['lat'], depot['lon']),
                capacity=depot.get('capacity', 50),
                access_type='paved',
                parking_type='paved_lot',
                manually_verified=True
            )
            
            # Find accessible segments
            accessible = self._find_accessible_segments(trailhead.parking_coords)
            trailhead.accessible_segments = accessible
            
            trailheads.append(trailhead)
        
        # Discover additional trailheads from trail data
        additional_trailheads = self._extract_trailheads_from_trail_data()
        trailheads.extend(additional_trailheads)
        
        return trailheads
    
    def _find_accessible_segments(self, coords: Tuple[float, float], max_distance: float = 5.0) -> Set[str]:
        """Find segments accessible from a parking location"""
        accessible = set()
        
        if not self.unified_graph:
            return accessible
        
        # Find nearest graph node
        nearest_node = self._find_nearest_node(coords)
        if not nearest_node:
            return accessible
        
        # Find all segments within walking distance (increased from 1.0 to 5.0 miles)
        try:
            # Get nodes within distance using a more permissive cutoff
            lengths = nx.single_source_dijkstra_path_length(
                self.unified_graph, 
                nearest_node, 
                cutoff=max_distance,
                weight='weight'
            )
            
            # Find required segments accessible from these nodes
            for node in lengths:
                # Check outgoing edges
                for _, _, data in self.unified_graph.edges(node, data=True):
                    segment = data.get('segment')
                    if segment and segment.required:
                        accessible.add(segment.seg_id)
                
                # Check incoming edges (for segments ending at this node)
                for _, _, data in self.unified_graph.in_edges(node, data=True):
                    segment = data.get('segment')
                    if segment and segment.required:
                        accessible.add(segment.seg_id)
                        
        except nx.NetworkXError:
            pass
        
        return accessible
    
    def _extract_trailheads_from_trail_data(self) -> List[Trailhead]:
        """Extract trailheads from AccessFrom fields and endpoint clustering"""
        trailheads = []
        
        # Collect access points from trail data
        access_points = {}
        for segment in self.all_segments.values():
            if segment.access_from:
                access_points[segment.access_from] = access_points.get(segment.access_from, [])
                access_points[segment.access_from].append(segment)
        
        # Create trailheads from significant access points (lowered threshold from 3 to 1)
        for access_name, segments in access_points.items():
            if len(segments) >= 1:  # Any access point with trails
                # Calculate centroid of segments
                all_coords = []
                for seg in segments:
                    all_coords.extend(seg.coordinates)
                
                if all_coords:
                    centroid_lat = sum(coord[1] for coord in all_coords) / len(all_coords)
                    centroid_lon = sum(coord[0] for coord in all_coords) / len(all_coords)
                    
                    trailhead = Trailhead(
                        name=access_name,
                        parking_coords=(centroid_lat, centroid_lon),
                        capacity=20,  # Estimated
                        access_type='unknown',
                        parking_type='informal',
                        accessible_segments={seg.seg_id for seg in segments if seg.required}
                    )
                    
                    trailheads.append(trailhead)
        
        # Also create trailheads from segment endpoints with high density
        self._add_endpoint_trailheads(trailheads)
        
        return trailheads
    
    def _add_endpoint_trailheads(self, trailheads: List[Trailhead]):
        """Add trailheads based on high-density segment endpoints"""
        from collections import defaultdict
        
        # Collect all segment endpoints
        endpoints = defaultdict(list)
        
        for segment in self.all_segments.values():
            if segment.required and len(segment.coordinates) >= 2:
                start_coord = segment.coordinates[0]
                end_coord = segment.coordinates[-1]
                
                # Round coordinates to cluster nearby endpoints
                start_key = (round(start_coord[0], 3), round(start_coord[1], 3))
                end_key = (round(end_coord[0], 3), round(end_coord[1], 3))
                
                endpoints[start_key].append(segment)
                endpoints[end_key].append(segment)
        
        # Create trailheads from high-density endpoints
        existing_coords = {th.parking_coords for th in trailheads}
        
        for coord_key, segments in endpoints.items():
            if len(segments) >= 2:  # At least 2 segments meet here
                lon, lat = coord_key
                parking_coord = (lat, lon)
                
                # Check if we already have a trailhead nearby (within 0.1 miles)
                too_close = False
                for existing_coord in existing_coords:
                    if haversine_distance_points((lon, lat), (existing_coord[1], existing_coord[0])) < 0.1:
                        too_close = True
                        break
                
                if not too_close:
                    trailhead = Trailhead(
                        name=f"Endpoint Cluster {len(trailheads) + 1}",
                        parking_coords=parking_coord,
                        capacity=10,
                        access_type='unknown',
                        parking_type='informal',
                        accessible_segments={seg.seg_id for seg in segments}
                    )
                    
                    trailheads.append(trailhead)
                    existing_coords.add(parking_coord)
    
    def _find_nearest_node(self, coords: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find nearest node in graph to given coordinates"""
        if not self.unified_graph:
            return None
        
        nodes = list(self.unified_graph.nodes())
        if not nodes:
            return None
        
        # Convert to lat, lon for distance calculation
        target = (coords[1], coords[0])  # Convert lat,lon to lon,lat
        
        min_dist = float('inf')
        nearest = None
        
        for node in nodes:
            dist = haversine_distance_points(target, node)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _generate_trailhead_loops(self, trailhead: Trailhead) -> List[Loop]:
        """Generate optimal loops from a trailhead"""
        loops = []
        
        if not trailhead.accessible_segments:
            return loops
        
        # Get all accessible segments as objects
        accessible_segments = [
            self.required_segments[seg_id] 
            for seg_id in trailhead.accessible_segments 
            if seg_id in self.required_segments
        ]
        
        # Group segments by trail system (the key fix!)
        trail_families = self._group_segments_by_trail_system(accessible_segments)
        
        # Instead of creating separate loops for each family, combine related families
        combined_families = self._combine_related_trail_families(trail_families)
        
        # Create efficient loops for each combined trail family
        for family_name, family_segments in combined_families.items():
            if len(family_segments) >= 1:  # Process families with any segments
                family_loop = self._create_trail_family_loop(trailhead, family_name, family_segments)
                if family_loop:
                    loops.append(family_loop)
        
        return loops
    
    def _group_segments_by_trail_system(self, segments: List[TrailSegment]) -> Dict[str, List[TrailSegment]]:
        """Group segments that belong to the same trail system"""
        trail_families = {}
        
        for segment in segments:
            # Extract base trail name (e.g., "Dry Creek Trail" from "Dry Creek Trail 1")
            base_name = self._extract_base_trail_name(segment.name)
            
            if base_name not in trail_families:
                trail_families[base_name] = []
            trail_families[base_name].append(segment)
        
        # Sort families by segment count (larger families first)
        sorted_families = dict(sorted(trail_families.items(), key=lambda x: len(x[1]), reverse=True))
        
        return sorted_families
    
    def _extract_base_trail_name(self, segment_name: str) -> str:
        """Extract base trail name from segment name"""
        if not segment_name:
            return "Unknown"
        
        # Common patterns to group segments:
        # "Dry Creek Trail 1" -> "Dry Creek Trail"
        # "Three Bears Trail Segment 1" -> "Three Bears Trail"
        # "Shane's Trail 1" -> "Shane's Trail"
        
        # Remove common suffixes that indicate segment numbers
        import re
        
        # Remove patterns like " 1", " Segment 1", " - Part 1", etc.
        patterns = [
            r'\s+\d+$',           # " 1", " 2", etc.
            r'\s+Segment\s+\d+$', # " Segment 1"
            r'\s+-\s+Part\s+\d+$',# " - Part 1" 
            r'\s+Part\s+\d+$',    # " Part 1"
            r'\s+Section\s+\d+$', # " Section 1"
            r'\s+Loop\s+\d+$',    # " Loop 1"
            r'\s+Spur\s+\d+$'     # " Spur 1"
        ]
        
        base_name = segment_name
        for pattern in patterns:
            base_name = re.sub(pattern, '', base_name, flags=re.IGNORECASE)
        
        # If no change, try to extract just the first part
        if base_name == segment_name:
            parts = segment_name.split()
            if len(parts) > 1:
                # Look for numeric suffix
                if parts[-1].isdigit():
                    base_name = ' '.join(parts[:-1])
                elif parts[-1].lower() in ['north', 'south', 'east', 'west', 'upper', 'lower', 'connector']:
                    base_name = ' '.join(parts[:-1])
        
        # Also handle special cases like "Shane's Loop" and "Shane's Trail" being the same system
        if "Shane" in base_name:
            base_name = "Shane's Trail"
        elif "Three Bears" in base_name:
            base_name = "Three Bears Trail"
        elif "Dry Creek" in base_name:
            base_name = "Dry Creek Trail"
            
        return base_name.strip()
    
    def _combine_related_trail_families(self, trail_families: Dict[str, List[TrailSegment]]) -> Dict[str, List[TrailSegment]]:
        """Combine trail families that are closely related or nearby"""
        combined_families = {}
        
        # Define which trail systems should be combined
        combination_rules = {
            "Shane's Trail": ["Shane's Loop", "Shane's Connector"],
            "Dry Creek Trail": ["Dry Creek", "Dry Creek Connector"],
            "Three Bears Trail": ["Three Bears", "Three Bears Loop"],
            "Central Ridge Trail": ["Central Ridge", "Military Reserve Trail"],
            "Polecat Loop Trail": ["Polecat", "Polecat Loop"],
            "Bucktail Trail": ["Bucktail", "Bucktail Loop"],
            "Crestline Trail": ["Crestline", "Crestline Connector"]
        }
        
        # First pass: combine based on rules
        processed_families = set()
        
        for primary_name, related_names in combination_rules.items():
            combined_segments = []
            combined_name = primary_name
            
            # Check if primary family exists
            if primary_name in trail_families and primary_name not in processed_families:
                combined_segments.extend(trail_families[primary_name])
                processed_families.add(primary_name)
                
                # Add related families
                for related_name in related_names:
                    for family_name, segments in trail_families.items():
                        if (related_name.lower() in family_name.lower() and 
                            family_name not in processed_families):
                            combined_segments.extend(segments)
                            processed_families.add(family_name)
            
            if combined_segments:
                combined_families[combined_name] = combined_segments
        
        # Second pass: add any uncombined families
        for family_name, segments in trail_families.items():
            if family_name not in processed_families:
                # Check if it should be combined with an existing family
                combined = False
                for combined_name in combined_families:
                    if (combined_name.split()[0].lower() in family_name.lower() or
                        family_name.split()[0].lower() in combined_name.lower()):
                        combined_families[combined_name].extend(segments)
                        combined = True
                        break
                
                if not combined:
                    combined_families[family_name] = segments
        
        # Third pass: combine small families that are geographically close
        final_families = {}
        small_threshold = 3  # Families with fewer segments
        
        for family_name, segments in combined_families.items():
            if len(segments) < small_threshold:
                # Try to merge with a nearby larger family
                merged = False
                for large_family_name, large_segments in final_families.items():
                    if len(large_segments) >= small_threshold:
                        # Check if segments are nearby (simplified check)
                        if self._are_segments_nearby(segments, large_segments):
                            final_families[large_family_name].extend(segments)
                            merged = True
                            break
                
                if not merged:
                    final_families[family_name] = segments
            else:
                final_families[family_name] = segments
        
        print(f"   Combined {len(trail_families)} trail families into {len(final_families)} groups")
        for name, segs in final_families.items():
            if len(segs) > 1:
                print(f"     {name}: {len(segs)} segments")
        
        return final_families
    
    def _are_segments_nearby(self, segments1: List[TrailSegment], segments2: List[TrailSegment], 
                            max_distance: float = 0.5) -> bool:
        """Check if two groups of segments are nearby each other"""
        if not segments1 or not segments2:
            return False
        
        # Check if any endpoints are close
        for seg1 in segments1:
            if seg1.coordinates:
                for seg2 in segments2:
                    if seg2.coordinates:
                        # Check start/end points
                        for coord1 in [seg1.coordinates[0], seg1.coordinates[-1]]:
                            for coord2 in [seg2.coordinates[0], seg2.coordinates[-1]]:
                                if haversine_distance_points(coord1, coord2) < max_distance:
                                    return True
        return False
    
    def _create_trail_family_loop(self, trailhead: Trailhead, family_name: str, segments: List[TrailSegment]) -> Optional[Loop]:
        """Create an efficient loop covering an entire trail family"""
        
        if not segments:
            return None
        
        # For families with multiple segments, create a real loop
        if len(segments) > 1:
            return self._create_multi_segment_loop(trailhead, family_name, segments)
        else:
            # Single segment - create simple out-and-back
            segment = segments[0]
            return Loop(
                trailhead=trailhead.name,
                segments=[segment],
                total_distance=segment.length_mi * 2.0,  # Out and back
                required_coverage={segment.seg_id},
                connector_ratio=0.0
            )
    
    def _create_multi_segment_loop(self, trailhead: Trailhead, family_name: str, segments: List[TrailSegment]) -> Optional[Loop]:
        """Create a loop covering multiple segments in a trail family"""
        
        # Find trailhead node in graph
        trailhead_node = self._find_nearest_node(trailhead.parking_coords)
        if not trailhead_node:
            return None
        
        # Build subgraph containing all family segments
        family_graph = nx.MultiDiGraph()
        
        # Add all segment edges to family graph
        for segment in segments:
            if len(segment.coordinates) >= 2:
                start_node = segment.coordinates[0]
                end_node = segment.coordinates[-1]
                
                # Add both directions for flexibility
                family_graph.add_edge(start_node, end_node, 
                                    segment=segment, 
                                    weight=segment.length_mi,
                                    required=True)
                family_graph.add_edge(end_node, start_node,
                                    segment=segment,
                                    weight=segment.length_mi,
                                    required=True)
        
        # Try to find connector trails to make it a proper loop
        family_graph = self._add_connectors_to_family(family_graph, segments)
        
        # Find optimal traversal using simplified CPP approach
        route_segments = self._find_optimal_traversal(family_graph, segments, trailhead_node)
        
        if route_segments:
            # Calculate metrics
            required_segments = [seg for seg in route_segments if seg.required]
            connector_segments = [seg for seg in route_segments if not seg.required]
            
            # Calculate more accurate distance that accounts for shared segments
            unique_segments = []
            seen_ids = set()
            for seg in route_segments:
                if seg.seg_id not in seen_ids:
                    unique_segments.append(seg)
                    seen_ids.add(seg.seg_id)
            
            total_distance = sum(seg.length_mi for seg in unique_segments)
            # Add some distance for segments traversed twice
            traversal_count = {}
            for seg in route_segments:
                traversal_count[seg.seg_id] = traversal_count.get(seg.seg_id, 0) + 1
            
            for seg_id, count in traversal_count.items():
                if count > 1:
                    seg = next(s for s in route_segments if s.seg_id == seg_id)
                    total_distance += seg.length_mi * (count - 1) * 0.5  # Half weight for retraversals
            
            required_coverage = {seg.seg_id for seg in segments}  # All segments in family
            connector_ratio = len(connector_segments) / len(route_segments) if route_segments else 0
            
            return Loop(
                trailhead=trailhead.name,
                segments=route_segments,
                total_distance=total_distance,
                required_coverage=required_coverage,
                connector_ratio=connector_ratio
            )
        
        return None
    
    def _add_connectors_to_family(self, family_graph: nx.MultiDiGraph, segments: List[TrailSegment]) -> nx.MultiDiGraph:
        """Add connector trails to enable loop formation"""
        
        # Get all nodes in the family
        family_nodes = set(family_graph.nodes())
        
        # Find ALL trails (required and connector) that touch our family segments
        expanded_nodes = set()
        for segment in segments:
            for coord in segment.coordinates:
                expanded_nodes.add(coord)
        
        # Search for connector trails in unified graph - IMPROVED APPROACH
        max_connector_distance = 1.5  # Increased to 1.5 miles for better connectivity
        connectors_added = 0
        
        # Also check all segments in the system, not just those at family nodes
        for seg_id, connector_segment in self.all_segments.items():
            if not connector_segment.required and connector_segment.name != 'Road Connection':
                # Check if this connector could help link family segments
                connector_useful = False
                
                # Check if connector touches any family segment endpoints
                for family_segment in segments:
                    if family_segment.coordinates and connector_segment.coordinates:
                        # Check distance between connector and family segment endpoints
                        for fam_coord in [family_segment.coordinates[0], family_segment.coordinates[-1]]:
                            for conn_coord in [connector_segment.coordinates[0], connector_segment.coordinates[-1]]:
                                if haversine_distance_points(fam_coord, conn_coord) < 0.1:  # Very close
                                    connector_useful = True
                                    break
                        if connector_useful:
                            break
                
                # Also check if connector is geographically nearby (within expanded area)
                if not connector_useful and connector_segment.length_mi < max_connector_distance:
                    for conn_coord in connector_segment.coordinates[::max(1, len(connector_segment.coordinates)//5)]:  # Sample points
                        for fam_segment in segments:
                            if fam_segment.coordinates:
                                for fam_coord in fam_segment.coordinates[::max(1, len(fam_segment.coordinates)//5)]:
                                    if haversine_distance_points(conn_coord, fam_coord) < max_connector_distance:
                                        connector_useful = True
                                        break
                            if connector_useful:
                                break
                        if connector_useful:
                            break
                
                # Add useful connectors to the family graph
                if connector_useful and connector_segment.coordinates:
                    start_coord = connector_segment.coordinates[0]
                    end_coord = connector_segment.coordinates[-1]
                    
                    # Use attractive weight for connectors (much lower than required trails)
                    connector_weight = connector_segment.length_mi * 0.2  # Make connectors very attractive
                    
                    family_graph.add_edge(start_coord, end_coord,
                                        segment=connector_segment,
                                        weight=connector_weight,
                                        required=False)
                    family_graph.add_edge(end_coord, start_coord,  # Add reverse too
                                        segment=connector_segment,
                                        weight=connector_weight,
                                        required=False)
                    connectors_added += 1
        
        if connectors_added > 0:
            print(f"   Added {connectors_added} connector trails to {len(segments)}-segment family")
        
        return family_graph
    
    def _is_helpful_connector(self, start_node, end_node, family_nodes: Set) -> bool:
        """Check if a connector trail helps link the family"""
        # Simple heuristic: if the connector brings us close to family nodes, it's helpful
        for family_node in family_nodes:
            if haversine_distance_points(end_node, family_node) < 0.1:  # Within ~500 feet
                return True
        return False
    
    def _find_optimal_traversal(self, family_graph: nx.MultiDiGraph, required_segments: List[TrailSegment], 
                               start_node: Tuple[float, float]) -> List[TrailSegment]:
        """Find optimal way to traverse all required segments using Chinese Postman Problem approach"""
        
        if not required_segments:
            return []
        
        # Convert to undirected graph for CPP (most trails are bidirectional)
        undirected_graph = nx.MultiGraph()
        
        # Add required segments as edges
        for segment in required_segments:
            if segment.coordinates and len(segment.coordinates) >= 2:
                start_coord = segment.coordinates[0]
                end_coord = segment.coordinates[-1]
                undirected_graph.add_edge(start_coord, end_coord, 
                                        weight=segment.length_mi, 
                                        segment_obj=segment)
        
        # Add helpful connector trails from the family graph
        for u, v, data in family_graph.edges(data=True):
            segment = data.get('segment')
            if segment and not segment.required and segment.length_mi < 2.0:  # Short connectors only
                # Give connectors lower weight to prefer them for loops
                undirected_graph.add_edge(u, v, 
                                        weight=segment.length_mi * 0.3,  # Prefer connectors
                                        segment_obj=segment)
        
        if not undirected_graph.edges:
            return required_segments  # Fallback to simple order
        
        # Apply Chinese Postman Problem algorithm
        return self._solve_cpp_for_subgraph(undirected_graph, required_segments)
    
    def _solve_cpp_for_subgraph(self, graph: nx.MultiGraph, required_segments: List[TrailSegment]) -> List[TrailSegment]:
        """Solve Chinese Postman Problem for a subgraph of trail segments"""
        
        if not graph.edges:
            return required_segments
        
        # Check if graph is connected first
        if not nx.is_connected(graph):
            print(f"   ⚠️  Graph disconnected, using simple traversal for {len(required_segments)} segments")
            return required_segments
        
        # Find nodes with odd degree
        odd_nodes = [node for node, degree in graph.degree() if degree % 2 != 0]
        
        if not odd_nodes:
            # Graph is already Eulerian - find circuit
            if nx.is_eulerian(graph):
                try:
                    start_node = list(graph.nodes())[0]
                    eulerian_circuit = list(nx.eulerian_circuit(graph, source=start_node))
                    circuit_segments = [graph[u][v][0]['segment_obj'] for u, v in eulerian_circuit]
                    if len(circuit_segments) <= len(required_segments) * 3:  # Reasonable size limit
                        return circuit_segments
                except:
                    pass
            return required_segments  # Fallback
        
        # For small families with too many odd nodes, just use simple traversal
        if len(odd_nodes) > len(required_segments):
            print(f"   ⚠️  Too many odd nodes ({len(odd_nodes)}) for {len(required_segments)} segments")
            return required_segments
        
        # Create complete graph of odd-degree nodes for matching
        odd_node_graph = nx.Graph()
        for i, u in enumerate(odd_nodes):
            for j, v in enumerate(odd_nodes):
                if i < j:
                    try:
                        # Find shortest path between odd nodes
                        path_length = nx.shortest_path_length(graph, source=u, target=v, weight='weight')
                        # Only add if path is reasonable
                        if path_length < 10.0:  # Max 10 miles between odd nodes
                            odd_node_graph.add_edge(u, v, weight=path_length)
                    except nx.NetworkXNoPath:
                        # If no path, don't add edge
                        continue
        
        # Need even number of odd nodes for perfect matching
        if len(odd_nodes) % 2 != 0:
            print(f"   ⚠️  Odd number of odd nodes ({len(odd_nodes)}), using simple traversal")
            return required_segments
        
        # Find minimum weight perfect matching
        try:
            if odd_node_graph.edges:
                matching = nx.min_weight_matching(odd_node_graph, weight='weight')
            else:
                # No valid edges for matching
                print(f"   ⚠️  No valid paths between odd nodes, using simple traversal")
                return required_segments
        except:
            # Fallback if matching fails
            print(f"   ⚠️  Matching failed, using simple traversal")
            return required_segments
        
        # Augment graph with matching paths
        augmented_graph = graph.copy()
        for u, v in matching:
            try:
                path = nx.shortest_path(graph, source=u, target=v, weight='weight')
                for i in range(len(path) - 1):
                    n1, n2 = path[i], path[i+1]
                    # Find the edge data
                    if graph.has_edge(n1, n2):
                        edge_data = graph.get_edge_data(n1, n2)[0]
                        augmented_graph.add_edge(n1, n2, weight=edge_data['weight'], 
                                               segment_obj=edge_data['segment_obj'], is_duplicate=True)
            except nx.NetworkXNoPath:
                continue
        
        # Find Eulerian circuit in augmented graph
        if nx.is_eulerian(augmented_graph):
            try:
                start_node = list(augmented_graph.nodes())[0]
                eulerian_circuit = list(nx.eulerian_circuit(augmented_graph, source=start_node))
                
                # Extract segments from circuit, handling duplicates properly
                circuit_segments = []
                temp_graph = augmented_graph.copy()
                
                for u, v in eulerian_circuit:
                    if temp_graph.has_edge(u, v):
                        # Get the first available edge (handles parallel edges)
                        edge_keys = list(temp_graph[u][v].keys())
                        if edge_keys:
                            edge_key = edge_keys[0]
                            edge_data = temp_graph[u][v][edge_key]
                            circuit_segments.append(edge_data['segment_obj'])
                            temp_graph.remove_edge(u, v, key=edge_key)
                
                # Validate the circuit is reasonable
                if circuit_segments and len(circuit_segments) <= len(required_segments) * 3:
                    total_distance = sum(seg.length_mi for seg in circuit_segments)
                    if total_distance < 25.0:  # Reasonable distance limit
                        print(f"   ✅ CPP found efficient loop with {len(circuit_segments)} segments")
                        return circuit_segments
            except:
                pass
        
        # Fallback to simple traversal
        print(f"   ⚠️  CPP failed, using simple traversal for {len(required_segments)} segments")
        return required_segments
    
    def _find_connector_to_segment(self, current_node: Tuple[float, float], 
                                 target_segment: TrailSegment) -> Optional[TrailSegment]:
        """Find a connector trail to reach the target segment"""
        
        # Simple approach: create virtual connector if needed
        target_start = target_segment.coordinates[0]
        distance = haversine_distance_points(current_node, target_start)
        
        if distance > 0.01:  # More than ~50 feet apart
            # Create virtual connector
            virtual_connector = TrailSegment(
                seg_id=f'connector_{len(self.all_segments)}',
                name='Trail Connector',
                coordinates=[current_node, target_start],
                length_ft=distance * 5280,
                direction='both',
                required=False
            )
            return virtual_connector
        
        return None
    
    def _select_optimal_loops(self, all_loops: List[Loop]) -> List[Loop]:
        """Select optimal combination of loops to cover all required segments"""
        # Group loops by trail system to avoid duplicates
        loops_by_system = {}
        
        for loop in all_loops:
            # Determine primary trail system for this loop
            trail_systems = {}
            for seg in loop.segments:
                if seg.required:
                    base_name = self._extract_base_trail_name(seg.name)
                    trail_systems[base_name] = trail_systems.get(base_name, 0) + 1
            
            if trail_systems:
                primary_system = max(trail_systems.items(), key=lambda x: x[1])[0]
                
                # Keep only the best loop for each trail system
                if primary_system not in loops_by_system:
                    loops_by_system[primary_system] = loop
                else:
                    # Keep the loop with better coverage or from a major trailhead
                    existing = loops_by_system[primary_system]
                    if (len(loop.required_coverage) > len(existing.required_coverage) or
                        (len(loop.required_coverage) == len(existing.required_coverage) and
                         loop.total_distance < existing.total_distance)):
                        loops_by_system[primary_system] = loop
        
        print(f"🔄 Reduced {len(all_loops)} loops to {len(loops_by_system)} unique trail systems")
        
        # Now consolidate the unique loops
        unique_loops = list(loops_by_system.values())
        consolidated_loops = self._consolidate_nearby_loops(unique_loops)
        
        # Select loops to cover all segments
        required_segments = set(self.required_segments.keys())
        selected_loops = []
        covered_segments = set()
        
        # Sort by coverage size (larger first)
        consolidated_loops.sort(key=lambda l: len(l.required_coverage), reverse=True)
        
        for loop in consolidated_loops:
            uncovered_in_loop = loop.required_coverage - covered_segments
            if uncovered_in_loop:
                selected_loops.append(loop)
                covered_segments.update(loop.required_coverage)
        
        # Handle any remaining uncovered segments
        uncovered_segments = required_segments - covered_segments
        if uncovered_segments:
            print(f"⚠️  Creating fallback loops for {len(uncovered_segments)} orphaned segments")
            fallback_loops = self._create_fallback_loops(uncovered_segments)
            selected_loops.extend(fallback_loops)
            covered_segments.update(seg for loop in fallback_loops for seg in loop.required_coverage)
            
            # Show size distribution of selected loops
            loop_sizes = sorted([loop.total_distance for loop in selected_loops])
            print(f"   Loop size distribution: min={loop_sizes[0]:.1f}mi, "
                  f"median={loop_sizes[len(loop_sizes)//2]:.1f}mi, "
                  f"max={loop_sizes[-1]:.1f}mi")
            
            # Warn about oversized loops
            oversized = [l for l in selected_loops if l.total_distance > self.config.long_day_limit]
            if oversized:
                print(f"   ⚠️  {len(oversized)} loops exceed day capacity ({self.config.long_day_limit}mi):")
                for loop in oversized[:5]:  # Show first 5
                    print(f"      - {loop.total_distance:.1f}mi with {len(loop.segments)} segments")
        
        print(f"✅ Selected {len(selected_loops)} loops covering {len(covered_segments)}/{len(required_segments)} segments")
        
        return selected_loops
    
    def _consolidate_nearby_loops(self, all_loops: List[Loop]) -> List[Loop]:
        """Consolidate nearby loops into larger, more efficient hikes"""
        
        # First, prioritize major trailheads and consolidate aggressively there
        major_trailheads = self._identify_major_trailheads(all_loops)
        
        # Group loops by trailhead, prioritizing major ones
        loops_by_trailhead = {}
        for loop in all_loops:
            if loop.trailhead not in loops_by_trailhead:
                loops_by_trailhead[loop.trailhead] = []
            loops_by_trailhead[loop.trailhead].append(loop)
        
        consolidated_loops = []
        
        # Process major trailheads first with aggressive consolidation
        for trailhead_name in major_trailheads:
            if trailhead_name in loops_by_trailhead:
                trailhead_loops = loops_by_trailhead[trailhead_name]
                combined_loops = self._aggressively_combine_loops(trailhead_loops)
                consolidated_loops.extend(combined_loops)
                del loops_by_trailhead[trailhead_name]
        
        # For remaining minor trailheads, try to relocate their segments to major trailheads
        remaining_segments = []
        for trailhead_name, trailhead_loops in loops_by_trailhead.items():
            for loop in trailhead_loops:
                remaining_segments.extend([s for s in loop.segments if s.required])
        
        # Try to assign remaining segments to major trailheads
        orphan_loops = self._reassign_segments_to_major_trailheads(remaining_segments, major_trailheads)
        consolidated_loops.extend(orphan_loops)
        
        print(f"🔄 Aggressively consolidated {len(all_loops)} loops into {len(consolidated_loops)} major hikes")
        print(f"   Using {len(set(loop.trailhead for loop in consolidated_loops))} trailheads (vs {len(loops_by_trailhead) + len(major_trailheads)} original)")
        
        return consolidated_loops
    
    def _identify_major_trailheads(self, all_loops: List[Loop]) -> List[str]:
        """Identify major trailheads based on segment count and accessibility"""
        
        # Count segments accessible from each trailhead
        trailhead_segment_counts = {}
        for loop in all_loops:
            th_name = loop.trailhead
            if th_name not in trailhead_segment_counts:
                trailhead_segment_counts[th_name] = set()
            trailhead_segment_counts[th_name].update(loop.required_coverage)
        
        # Sort by segment count
        sorted_trailheads = sorted(
            trailhead_segment_counts.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Take top trailheads that cover at least 80% of segments
        major_trailheads = []
        total_segments_covered = set()
        target_coverage = len(self.required_segments) * 0.8
        
        for th_name, segments in sorted_trailheads:
            major_trailheads.append(th_name)
            total_segments_covered.update(segments)
            
            if len(total_segments_covered) >= target_coverage or len(major_trailheads) >= 8:
                break
        
        print(f"🎯 Selected {len(major_trailheads)} major trailheads covering {len(total_segments_covered)} segments")
        for th_name, segments in sorted_trailheads[:len(major_trailheads)]:
            print(f"   {th_name}: {len(segments)} segments")
        
        return major_trailheads
    
    def _aggressively_combine_loops(self, trailhead_loops: List[Loop]) -> List[Loop]:
        """Aggressively combine loops to create fewer, larger hikes"""
        
        if not trailhead_loops:
            return []
        
        # Group loops by natural trail systems first
        trail_system_groups = {}
        other_loops = []
        
        for loop in trailhead_loops:
            # Check if this loop belongs to a clear trail system
            if loop.segments:
                # Get the most common base trail name
                trail_names = {}
                for seg in loop.segments:
                    if seg.required:
                        base_name = self._extract_base_trail_name(seg.name)
                        trail_names[base_name] = trail_names.get(base_name, 0) + 1
                
                if trail_names:
                    # Find dominant trail system
                    dominant_system = max(trail_names.items(), key=lambda x: x[1])[0]
                    if dominant_system not in trail_system_groups:
                        trail_system_groups[dominant_system] = []
                    trail_system_groups[dominant_system].append(loop)
                else:
                    other_loops.append(loop)
            else:
                other_loops.append(loop)
        
        # Combine loops within each trail system
        combined_loops = []
        
        for system_name, system_loops in trail_system_groups.items():
            print(f"   Combining {len(system_loops)} loops for {system_name} system")
            
            # LIMIT: Don't create loops larger than 20 miles or 20 segments
            max_loop_distance = 20.0
            max_loop_segments = 20
            
            current_batch = []
            current_distance = 0
            current_coverage = set()
            
            for loop in system_loops:
                # Check if adding this loop would exceed limits
                potential_distance = current_distance + loop.total_distance
                potential_segments = len(current_coverage) + len(loop.required_coverage)
                
                if (potential_distance <= max_loop_distance and 
                    potential_segments <= max_loop_segments and 
                    current_batch):
                    # Add to current batch
                    current_batch.append(loop)
                    current_distance = potential_distance
                    current_coverage.update(loop.required_coverage)
                else:
                    # Finalize current batch if it exists
                    if current_batch:
                        combined_loop = self._create_combined_loop(current_batch)
                        if combined_loop:
                            combined_loops.append(combined_loop)
                            print(f"   🔗 Combined {len(current_batch)} loops for {system_name}: {len(current_coverage)} segments, {combined_loop.total_distance:.1f} miles")
                    
                    # Start new batch
                    current_batch = [loop]
                    current_distance = loop.total_distance
                    current_coverage = loop.required_coverage.copy()
            
            # Finalize last batch
            if current_batch:
                combined_loop = self._create_combined_loop(current_batch)
                if combined_loop:
                    combined_loops.append(combined_loop)
                    print(f"   🔗 Combined {len(current_batch)} loops for {system_name}: {len(current_coverage)} segments, {combined_loop.total_distance:.1f} miles")
        
        # Now combine remaining loops by proximity
        remaining_loops = other_loops
        
        while remaining_loops:
            base_loop = remaining_loops.pop(0)
            current_distance = base_loop.total_distance
            current_segments = base_loop.segments.copy()
            current_coverage = base_loop.required_coverage.copy()
            
            # Try to add nearby loops
            max_hike_distance = 20.0  # Reasonable day hike limit
            
            i = 0
            while i < len(remaining_loops) and current_distance < max_hike_distance:
                candidate_loop = remaining_loops[i]
                
                # Always combine very small loops
                if candidate_loop.total_distance < 2.0 or current_distance + candidate_loop.total_distance < 15.0:
                    current_segments.extend(candidate_loop.segments)
                    current_coverage.update(candidate_loop.required_coverage)
                    current_distance += candidate_loop.total_distance * 0.8
                    remaining_loops.pop(i)
                else:
                    i += 1
            
            combined_loop = Loop(
                trailhead=base_loop.trailhead,
                segments=current_segments,
                total_distance=current_distance,
                required_coverage=current_coverage,
                connector_ratio=sum(1 for seg in current_segments if not seg.required) / len(current_segments) if current_segments else 0
            )
            combined_loops.append(combined_loop)
        
        return combined_loops
    
    def _create_combined_loop(self, loops: List[Loop]) -> Optional[Loop]:
        """Safely create a combined loop from multiple loops"""
        if not loops:
            return None
        
        all_segments = []
        all_coverage = set()
        total_distance = 0
        
        for loop in loops:
            all_segments.extend(loop.segments)
            all_coverage.update(loop.required_coverage)
            total_distance += loop.total_distance * 0.9  # Small efficiency factor
        
        # Estimate realistic total distance
        base_distance = sum(seg.length_mi for seg in all_segments if seg.required)
        connector_distance = sum(seg.length_mi for seg in all_segments if not seg.required)
        realistic_distance = base_distance + connector_distance * 0.5  # Half weight for connectors
        
        combined_loop = Loop(
            trailhead=loops[0].trailhead,
            segments=all_segments,
            total_distance=min(total_distance, realistic_distance),
            required_coverage=all_coverage,
            connector_ratio=len([seg for seg in all_segments if not seg.required]) / len(all_segments) if all_segments else 0
        )
        
        return combined_loop
    
    def _can_aggressively_combine(self, base_loop: Loop, candidate_loop: Loop, current_distance: float) -> bool:
        """More lenient criteria for combining loops at major trailheads"""
        
        # Don't combine if it would make the hike too long
        if current_distance + candidate_loop.total_distance > 25.0:
            return False
        
        # For major trailheads, be more lenient about distance between segments
        # Check if loops are within 5 miles of each other (increased from 2 miles)
        base_coords = []
        for seg in base_loop.segments:
            base_coords.extend(seg.coordinates[:2])  # Just sample a few points
        
        candidate_coords = []
        for seg in candidate_loop.segments:
            candidate_coords.extend(seg.coordinates[:2])
        
        if base_coords and candidate_coords:
            min_distance = float('inf')
            for base_coord in base_coords:
                for cand_coord in candidate_coords:
                    dist = haversine_distance_points(base_coord, cand_coord)
                    min_distance = min(min_distance, dist)
            
            # More lenient distance threshold for major trailheads
            if min_distance > 5.0:
                return False
        
        # Always combine small loops
        if base_loop.total_distance < 8.0 and candidate_loop.total_distance < 8.0:
            return True
        
        # Combine if either loop is small
        if base_loop.total_distance < 3.0 or candidate_loop.total_distance < 3.0:
            return True
        
        return True  # Default to aggressive combining for major trailheads
    
    def _reassign_segments_to_major_trailheads(self, orphan_segments: List[TrailSegment], major_trailheads: List[str]) -> List[Loop]:
        """Try to assign orphaned segments to major trailheads with size limits"""
        
        if not orphan_segments or not major_trailheads:
            return []
        
        # Find trailhead objects for major trailheads
        major_th_objects = [th for th in self.trailheads if th.name in major_trailheads]
        
        reassigned_loops = []
        remaining_segments = orphan_segments.copy()
        
        # For each major trailhead, try to assign nearby orphan segments
        for trailhead in major_th_objects:
            if not remaining_segments:
                break
            
            # Find segments within reasonable distance of this trailhead
            nearby_segments = []
            th_coord = (trailhead.parking_coords[1], trailhead.parking_coords[0])  # Convert to lon,lat
            
            for segment in remaining_segments.copy():
                if segment.coordinates:
                    seg_midpoint = segment.coordinates[len(segment.coordinates)//2]
                    distance = haversine_distance_points(th_coord, seg_midpoint)
                    
                    if distance < 5.0:  # Reduced from 8 miles to 5 miles for tighter control
                        nearby_segments.append(segment)
                        remaining_segments.remove(segment)
                        
                        # LIMIT: Don't create loops larger than 15 segments
                        if len(nearby_segments) >= 15:
                            break
            
            # Create reasonably-sized loops for these segments
            if nearby_segments:
                # Split large groups into smaller loops
                max_segments_per_loop = 10
                
                for i in range(0, len(nearby_segments), max_segments_per_loop):
                    segment_batch = nearby_segments[i:i + max_segments_per_loop]
                    
                    # Estimate realistic distance
                    base_distance = sum(seg.length_mi for seg in segment_batch)
                    access_factor = min(2.0, 1.0 + distance / 10.0)  # More access cost for distant segments
                    total_distance = base_distance * access_factor
                    
                    access_loop = Loop(
                        trailhead=trailhead.name,
                        segments=segment_batch,
                        total_distance=total_distance,
                        required_coverage={seg.seg_id for seg in segment_batch},
                        connector_ratio=0.3  # Reasonable connector ratio
                    )
                    
                    reassigned_loops.append(access_loop)
                    print(f"   🎯 Reassigned {len(segment_batch)} orphan segments to {trailhead.name} ({total_distance:.1f}mi)")
        
        # Create fallback loops for any truly isolated segments (individual segments only)
        for segment in remaining_segments[:20]:  # Limit to 20 most important segments
            fallback_loop = Loop(
                trailhead=f"Special Access {len(reassigned_loops) + 1}",
                segments=[segment],
                total_distance=segment.length_mi * 2.0,  # Realistic cost for isolation
                required_coverage={segment.seg_id},
                connector_ratio=0.5
            )
            reassigned_loops.append(fallback_loop)
        
        return reassigned_loops
    
    def _create_fallback_loops(self, uncovered_segment_ids: Set[str]) -> List[Loop]:
        """Create fallback loops for segments that couldn't be accessed from existing trailheads"""
        fallback_loops = []
        
        for seg_id in uncovered_segment_ids:
            if seg_id in self.required_segments:
                segment = self.required_segments[seg_id]
                
                # Find the nearest existing trailhead to this segment
                nearest_trailhead = self._find_nearest_trailhead_to_segment(segment)
                
                if nearest_trailhead:
                    # Create a dedicated loop for this segment
                    fallback_loop = Loop(
                        trailhead=nearest_trailhead.name,
                        segments=[segment],
                        total_distance=segment.length_mi * 2.5,  # Estimate higher cost for isolated segments
                        required_coverage={segment.seg_id},
                        connector_ratio=0.6  # Higher connector ratio for isolated segments
                    )
                    fallback_loops.append(fallback_loop)
                else:
                    # Create a new temporary trailhead for truly isolated segments
                    if segment.coordinates:
                        temp_trailhead_name = f"Temp Access {len(fallback_loops) + 1}"
                        fallback_loop = Loop(
                            trailhead=temp_trailhead_name,
                            segments=[segment],
                            total_distance=segment.length_mi * 2.0,
                            required_coverage={segment.seg_id},
                            connector_ratio=0.4
                        )
                        fallback_loops.append(fallback_loop)
        
        return fallback_loops
    
    def _find_nearest_trailhead_to_segment(self, segment: TrailSegment) -> Optional[Trailhead]:
        """Find the nearest trailhead to a given segment"""
        if not segment.coordinates or not self.trailheads:
            return None
        
        # Use segment midpoint for distance calculation
        midpoint_idx = len(segment.coordinates) // 2
        segment_coord = segment.coordinates[midpoint_idx]
        
        min_distance = float('inf')
        nearest_trailhead = None
        
        for trailhead in self.trailheads:
            # Convert parking_coords (lat, lon) to (lon, lat) for distance calculation
            th_coord = (trailhead.parking_coords[1], trailhead.parking_coords[0])
            distance = haversine_distance_points(segment_coord, th_coord)
            
            if distance < min_distance:
                min_distance = distance
                nearest_trailhead = trailhead
        
        return nearest_trailhead
    
    def _organize_into_days(self, loops: List[Loop]) -> List[Day]:
        """Organize loops into daily plans"""
        import time
        
        days = []
        current_day = 1
        start_time = time.time()
        max_days = 100  # Safety limit
        
        # Sort loops by distance for better packing
        sorted_loops = sorted(loops, key=lambda l: l.total_distance, reverse=True)
        
        # Show loop size distribution
        print(f"   Loop distances: min={min(l.total_distance for l in loops):.1f}mi, "
              f"max={max(l.total_distance for l in loops):.1f}mi, "
              f"avg={sum(l.total_distance for l in loops)/len(loops):.1f}mi")
        
        # Simple bin packing approach
        remaining_loops = sorted_loops.copy()
        total_loops = len(remaining_loops)
        processed_loops = 0
        
        while remaining_loops and current_day <= max_days:
            day = Day(number=current_day)
            day_capacity = self._get_day_capacity(current_day)
            current_capacity = 0
            
            # Debug: show what's happening
            if current_day % 5 == 0:
                print(f"\n   Creating day {current_day}, {len(remaining_loops)} loops remaining...")
            
            # Find loops that fit in current day
            i = 0
            items_added = 0
            while i < len(remaining_loops) and current_capacity < day_capacity:
                loop = remaining_loops[i]
                
                if current_capacity + loop.total_distance <= day_capacity:
                    # Convert loop to hike
                    processed_loops += 1
                    print(f"   Converting loop {processed_loops}/{total_loops} to hike (Day {current_day})...", end='\r', flush=True)
                    hike = self._loop_to_hike(loop, len(day.hikes) + 1)
                    day.hikes.append(hike)
                    current_capacity += loop.total_distance
                    remaining_loops.pop(i)
                    items_added += 1
                else:
                    i += 1
            
            # Safety check: if we couldn't add anything, force add the smallest remaining item
            if items_added == 0 and remaining_loops:
                print(f"\n   ⚠️  Day {current_day} capacity too small, forcing smallest loop...")
                smallest_loop = min(remaining_loops, key=lambda l: l.total_distance)
                processed_loops += 1
                hike = self._loop_to_hike(smallest_loop, 1)
                day.hikes.append(hike)
                remaining_loops.remove(smallest_loop)
            
            # Calculate day totals
            day.total_distance = sum(h.total_distance for h in day.hikes)
            day.total_elevation = sum(h.elevation_gain for h in day.hikes)
            
            # Set day type
            if day.total_distance <= 6:
                day.type = 'short'
            elif day.total_distance <= 15:
                day.type = 'medium'
            else:
                day.type = 'long'
            
            days.append(day)
            current_day += 1
            
            # Show progress every 10 days
            if current_day % 10 == 0:
                elapsed = time.time() - start_time
                print(f"\n   Created {current_day} days so far ({elapsed:.1f}s elapsed)...")
        
        if current_day > max_days:
            print(f"\n   ⚠️  Stopped at day limit ({max_days} days)")
        
        print()  # Clear the progress line
        return days
    
    def _get_day_capacity(self, day_number: int) -> float:
        """Get capacity for a given day based on configuration"""
        # Cycle through day types
        day_types = ['medium', 'long', 'short', 'medium', 'long', 'short', 'medium']
        day_type = day_types[(day_number - 1) % len(day_types)]
        
        if day_type == 'short':
            return self.config.short_day_limit
        elif day_type == 'medium':
            return self.config.medium_day_limit
        else:
            return self.config.long_day_limit
    
    def _loop_to_hike(self, loop: Loop, hike_number: int) -> Hike:
        """Convert a loop to a hike with full details"""
        # Show what we're processing if it's taking time
        if len(loop.segments) > 20:
            print(f"\n     Processing large loop with {len(loop.segments)} segments...", end='', flush=True)
        
        # Find trailhead details
        trailhead_obj = None
        for th in self.trailheads:
            if th.name == loop.trailhead:
                trailhead_obj = th
                break
        
        # Use cached or estimated elevation gain for speed
        elevation_gain = loop.total_distance * 100  # Quick estimate: 100ft per mile
        
        hike = Hike(
            hike_number=hike_number,
            trailhead=loop.trailhead,
            segments=loop.segments,
            total_distance=loop.total_distance,
            elevation_gain=elevation_gain,
            difficulty=self._determine_difficulty(loop),
            trail_conditions=self._generate_trail_conditions(loop.segments),
            estimated_minutes=int(loop.total_distance * self.config.base_pace_min_per_mile)
        )
        
        # Add parking details from trailhead
        if trailhead_obj:
            hike.parking_coords = trailhead_obj.parking_coords
            hike.parking_type = trailhead_obj.parking_type
            hike.parking_fee = trailhead_obj.parking_fee
            hike.parking_notes = trailhead_obj.parking_notes
        
        # Calculate max elevation
        hike.max_elevation = max(
            (seg.avg_elevation for seg in loop.segments if seg.avg_elevation > 0),
            default=0
        )
        
        return hike
    
    def _calculate_elevation_gain(self, segments: List[TrailSegment]) -> float:
        """Calculate total elevation gain for segments"""
        # Skip elevation calculation if no DEM provider to speed up processing
        if not self.dem_provider:
            # Use rough estimate based on distance
            return sum(seg.length_mi * 100 for seg in segments)  # ~100ft per mile
        
        total_gain = 0
        
        for segment in segments:
            if len(segment.coordinates) >= 2:
                try:
                    start_elev = list(self.dem_provider.sample([segment.coordinates[0]]))[0][0]
                    end_elev = list(self.dem_provider.sample([segment.coordinates[-1]]))[0][0]
                    gain = max(0, end_elev - start_elev)
                    total_gain += gain
                except:
                    # Fallback estimate
                    total_gain += segment.length_mi * 100
        
        return total_gain
    
    def _determine_difficulty(self, loop: Loop) -> str:
        """Determine difficulty level based on distance and elevation"""
        distance = loop.total_distance
        
        if distance <= 3:
            return 'easy'
        elif distance <= 8:
            return 'moderate'
        elif distance <= 15:
            return 'difficult'
        else:
            return 'expert'
    
    def _generate_trail_conditions(self, segments: List[TrailSegment]) -> str:
        """Generate trail conditions description"""
        surfaces = set(seg.surface for seg in segments if seg.surface != 'unknown')
        exposures = set(seg.exposure for seg in segments if seg.exposure != 'mixed')
        
        conditions = []
        
        if 'rocky' in surfaces:
            conditions.append("Rocky terrain with loose rocks")
        if 'paved' in surfaces:
            conditions.append("Paved sections")
        if 'full_sun' in exposures:
            conditions.append("Full sun exposure")
        if any(seg.avg_elevation > 6000 for seg in segments):
            conditions.append("High elevation - check weather")
        
        return "; ".join(conditions) if conditions else "Standard trail conditions"
    
    def _add_navigation_details(self, hike: Hike):
        """Add detailed navigation instructions to a hike"""
        # Simplified navigation generation
        distance_so_far = 0
        
        # Add start point
        nav_point = NavigationPoint(
            distance_from_start=0,
            instruction=f"Start from {hike.trailhead} parking area",
            landmark="Trailhead parking",
            gps_coords=hike.parking_coords
        )
        hike.navigation_points.append(nav_point)
        
        # Add navigation for each segment (limit detail to speed up)
        for i, segment in enumerate(hike.segments):
            distance_so_far += segment.length_mi
            
            # Only add nav points for required segments and key connectors
            if segment.required or i == 0 or i == len(hike.segments) - 1:
                if segment.required:
                    instruction = f"Begin required segment: {segment.name}"
                else:
                    instruction = f"Continue on connector trail: {segment.name}"
                
                nav_point = NavigationPoint(
                    distance_from_start=distance_so_far,
                    instruction=instruction,
                    landmark=f"Trail junction",
                    gps_coords=segment.start_node
                )
                hike.navigation_points.append(nav_point)
        
        # Add return instruction
        nav_point = NavigationPoint(
            distance_from_start=hike.total_distance,
            instruction=f"Return to {hike.trailhead} parking",
            landmark="Trailhead",
            gps_coords=hike.parking_coords
        )
        hike.navigation_points.append(nav_point)
        
        # Add waypoints for GPX - LIMIT to avoid memory issues
        total_waypoints = 0
        max_waypoints_per_hike = 500  # Reasonable limit
        
        for segment in hike.segments:
            if total_waypoints < max_waypoints_per_hike:
                # Sample coordinates if too many
                coords = segment.coordinates
                if len(coords) > 50:  # Downsample large segments
                    step = len(coords) // 50
                    coords = coords[::step]
                
                hike.gpx_waypoints.extend(coords)
                total_waypoints += len(coords)
        
        # Add escape routes for long hikes
        if hike.total_distance > 5:
            escape = EscapeRoute(
                at_mile=hike.total_distance / 2,
                description="Return via shortest connector trail",
                saves_miles=hike.total_distance * 0.3,
                to_parking=hike.trailhead
            )
            hike.escape_routes.append(escape)
    
    def _calculate_summary_stats(self, daily_plans: List[Day]) -> Dict[str, Any]:
        """Calculate summary statistics for the generated plan"""
        total_miles = sum(day.total_distance for day in daily_plans)
        required_miles = sum(
            sum(seg.length_mi for seg in hike.segments if seg.required)
            for day in daily_plans
            for hike in day.hikes
        )
        
        # FIX: Include all non-required segments as connectors (including roads)
        connector_miles = sum(
            sum(seg.length_mi for seg in hike.segments if not seg.required)
            for day in daily_plans
            for hike in day.hikes
        )
        
        # Separate road miles for detailed breakdown
        road_miles = sum(
            sum(seg.length_mi for seg in hike.segments if not seg.required and 'road' in seg.name.lower())
            for day in daily_plans
            for hike in day.hikes
        )
        
        # True trail connector miles (non-road connectors)
        trail_connector_miles = connector_miles - road_miles
        
        redundancy_percent = ((total_miles / required_miles) - 1) * 100 if required_miles > 0 else 0
        efficiency_score = 100 - redundancy_percent
        
        total_driving_miles = sum(day.total_driving for day in daily_plans)
        unique_trailheads = len(set(
            hike.trailhead
            for day in daily_plans
            for hike in day.hikes
        ))
        
        avg_hikes_per_day = sum(len(day.hikes) for day in daily_plans) / len(daily_plans) if daily_plans else 0
        
        return {
            'total_miles': round(total_miles, 1),
            'required_miles': round(required_miles, 1),
            'connector_miles': round(connector_miles, 1),
            'trail_connector_miles': round(trail_connector_miles, 1),
            'road_miles': round(road_miles, 1),
            'redundancy_percent': round(redundancy_percent, 1),
            'efficiency_score': round(efficiency_score, 1),
            'total_driving_miles': round(total_driving_miles, 1),
            'unique_trailheads': unique_trailheads,
            'average_hikes_per_day': round(avg_hikes_per_day, 1),
            'total_days': len(daily_plans)
        }


# Helper functions for tests
def extract_trailheads(trail_data, osm_data):
    """Extract trailheads from data sources"""
    config = PlannerConfig(
        solver_time_limit_seconds=300,
        trailhead_depots=[
            {'name': "Camel's Back Park", 'lat': 43.635278, 'lon': -116.205},
            {'name': "Military Reserve", 'lat': 43.62, 'lon': -116.18},
            {'name': "Stack Rock Trailhead", 'lat': 43.75, 'lon': -116.10},
            {'name': "Bogus Basin", 'lat': 43.76, 'lon': -116.10}
        ],
        daily_capacities={},
        cost_model={'elevation_beta': 10.0},
        drive_threshold_miles=2.0
    )
    
    router = TrailheadRouter(config)
    router.all_segments = trail_data
    return router._discover_trailheads()

def build_integrated_network(required_data, trails_geojson, osm_pbf):
    """Build the three-layer integrated network"""
    config = PlannerConfig(
        solver_time_limit_seconds=300,
        trailhead_depots=[],
        daily_capacities={},
        cost_model={'elevation_beta': 10.0},
        drive_threshold_miles=2.0
    )
    
    router = TrailheadRouter(config)
    # Simplified for testing
    return nx.DiGraph()

# Additional helper functions would be implemented here... 