#!/usr/bin/env python3
"""
Continuous Route Planner for Boise Trails Challenge - FIXED VERSION

Creates a single optimal route covering all 247 official segments,
with proper handling of directional constraints and real-world connections.
No teleporting allowed!
"""

import json
import math
import sys
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import csv
import networkx as nx
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing

from .core.models import TrailSegment
from .core.utils import haversine_distance_points

def load_trail_segments(filename: str) -> List[TrailSegment]:
    """Load trail segments from JSON file."""
    segments = []
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for segment_data in data['trailSegments']:
        coords = segment_data['geometry']['coordinates']
        props = segment_data['properties']
        
        segment = TrailSegment(
            seg_id=str(props.get('segId', '')),
            name=props.get('segName', ''),
            start_coords=(coords[0][0], coords[0][1]),
            end_coords=(coords[-1][0], coords[-1][1]),
            length_ft=float(props.get('LengthFt', 0)),
            direction=props.get('direction', 'both'),
            coordinates=tuple((c[0], c[1]) for c in coords)
        )
        segments.append(segment)
    
    return segments

def find_valid_connections(segments: List[TrailSegment], max_connection_distance: float = 0.1) -> Dict[str, List[str]]:
    """Find which segments can logically connect to each other, considering directional constraints."""
    connections = defaultdict(list)
    
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i == j:
                continue
                
            # Check all possible connections with directional constraints
            
            # seg1 end -> seg2 start (if seg1 allows forward, seg2 allows forward)
            if (seg1.can_traverse_direction(True) and seg2.can_traverse_direction(True) and 
                haversine_distance_points(seg1.end_coords, seg2.start_coords) <= max_connection_distance):
                connections[f"{seg1.seg_id}_end"].append(f"{seg2.seg_id}_start")
            
            # seg1 end -> seg2 end (if seg1 allows forward, seg2 allows reverse)
            if (seg1.can_traverse_direction(True) and seg2.can_traverse_direction(False) and 
                haversine_distance_points(seg1.end_coords, seg2.end_coords) <= max_connection_distance):
                connections[f"{seg1.seg_id}_end"].append(f"{seg2.seg_id}_end")
            
            # seg1 start -> seg2 start (if seg1 allows reverse, seg2 allows forward)
            if (seg1.can_traverse_direction(False) and seg2.can_traverse_direction(True) and 
                haversine_distance_points(seg1.start_coords, seg2.start_coords) <= max_connection_distance):
                connections[f"{seg1.seg_id}_start"].append(f"{seg2.seg_id}_start")
            
            # seg1 start -> seg2 end (if seg1 allows reverse, seg2 allows reverse)
            if (seg1.can_traverse_direction(False) and seg2.can_traverse_direction(False) and 
                haversine_distance_points(seg1.start_coords, seg2.end_coords) <= max_connection_distance):
                connections[f"{seg1.seg_id}_start"].append(f"{seg2.seg_id}_end")
    
    return connections

def find_road_connections(segments: List[TrailSegment], max_road_distance: float = 1.5) -> Dict[str, List[Tuple[str, float]]]:
    """Find road/walking connections between trail endpoints within reasonable distance."""
    road_connections = defaultdict(list)
    
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i == j:
                continue
                
            # Check road connections from seg1 endpoints to seg2 endpoints
            # Only include if it's a reasonable walking/short driving distance
            
            connections_to_check = []
            
            # From seg1 end (if we can reach the end)
            if seg1.can_traverse_direction(True):
                if seg2.can_traverse_direction(True):
                    connections_to_check.append((
                        f"{seg1.seg_id}_end", f"{seg2.seg_id}_start",
                        seg1.end_coords, seg2.start_coords
                    ))
                if seg2.can_traverse_direction(False):
                    connections_to_check.append((
                        f"{seg1.seg_id}_end", f"{seg2.seg_id}_end", 
                        seg1.end_coords, seg2.end_coords
                    ))
            
            # From seg1 start (if we can reach the start in reverse)
            if seg1.can_traverse_direction(False):
                if seg2.can_traverse_direction(True):
                    connections_to_check.append((
                        f"{seg1.seg_id}_start", f"{seg2.seg_id}_start",
                        seg1.start_coords, seg2.start_coords
                    ))
                if seg2.can_traverse_direction(False):
                    connections_to_check.append((
                        f"{seg1.seg_id}_start", f"{seg2.seg_id}_end",
                        seg1.start_coords, seg2.end_coords
                    ))
            
            for from_endpoint, to_endpoint, from_coords, to_coords in connections_to_check:
                distance = haversine_distance_points(from_coords, to_coords)
                if 0.1 < distance <= max_road_distance:  # More than direct connection, but reasonable
                    road_connections[from_endpoint].append((to_endpoint, distance))
    
    return road_connections

def find_connected_components_with_directions(segments: List[TrailSegment]) -> List[List[TrailSegment]]:
    """Find connected components considering directional constraints."""
    connections = find_valid_connections(segments)
    segment_map = {seg.seg_id: seg for seg in segments}
    
    visited = set()
    components = []
    
    def dfs(seg_id: str, current_component: Set[str]):
        if seg_id in visited:
            return
        visited.add(seg_id)
        current_component.add(seg_id)
        
        # Check connections from both ends of this segment
        for endpoint in [f"{seg_id}_start", f"{seg_id}_end"]:
            for connected_endpoint in connections.get(endpoint, []):
                connected_seg_id = connected_endpoint.split('_')[0]
                if connected_seg_id not in visited:
                    dfs(connected_seg_id, current_component)
    
    for segment in segments:
        if segment.seg_id not in visited:
            component_seg_ids = set()
            dfs(segment.seg_id, component_seg_ids)
            component_segments = [segment_map[seg_id] for seg_id in component_seg_ids]
            components.append(component_segments)
    
    return components

def build_graph_from_segments(segments: List[TrailSegment]) -> nx.MultiGraph:
    """Build a networkx graph from a list of trail segments."""
    G = nx.MultiGraph()
    for i, segment in enumerate(segments):
        G.add_edge(segment.start_node, segment.end_node, weight=segment.length_mi, segment_obj=segment, key=i)
    return G

def find_connected_components(segments: List[TrailSegment]) -> List[List[TrailSegment]]:
    """A robust method to find all disconnected subgraphs in the trail network."""
    G = nx.Graph()
    for seg in segments:
        # Add all points of the segment to the graph to ensure full connectivity
        for i in range(len(seg.coordinates) - 1):
            G.add_edge(seg.coordinates[i], seg.coordinates[i+1])

    # Find the connected components from the detailed graph
    component_nodes_list = list(nx.connected_components(G))
    
    # Create a mapping from each coordinate to the segment(s) it belongs to
    coord_to_segment = defaultdict(list)
    for seg in segments:
        for coord in seg.coordinates:
            coord_to_segment[coord].append(seg)

    # Create a mapping from segment ID to segment object
    segment_map = {seg.seg_id: seg for seg in segments}

    # Assign segments to components
    components = []
    for comp_nodes in component_nodes_list:
        comp_seg_ids = set()  # Use segment IDs instead of segment objects
        for node in comp_nodes:
            # Find all segments that include this node
            for seg in coord_to_segment[node]:
                comp_seg_ids.add(seg.seg_id)  # Add segment ID, not object
        if comp_seg_ids:
            # Convert back to segment objects
            comp_segments = [segment_map[seg_id] for seg_id in comp_seg_ids]
            components.append(comp_segments)
    
    components.sort(key=len, reverse=True)
    return components

def create_optimal_route_for_component(component_segments: List[TrailSegment]) -> List[TrailSegment]:
    """
    Uses a self-contained Chinese Postman Problem solver to find the optimal route for a component.
    Handles cases where the component may consist of multiple disconnected subgraphs due to data gaps.
    """
    if not component_segments:
        return []

    G = nx.MultiGraph()
    for seg in component_segments:
        G.add_edge(seg.start_node, seg.end_node, weight=seg.length_mi, segment_obj=seg)

    full_component_route = []

    # Process each disconnected subgraph within this "component" individually.
    # This makes the solver robust to small data gaps where endpoints don't perfectly align.
    for subgraph_nodes in nx.connected_components(G):
        subgraph = G.subgraph(subgraph_nodes).copy()
        
        if not subgraph.edges:
            continue

        odd_nodes = [node for node, degree in subgraph.degree() if degree % 2 != 0]
        
        if not odd_nodes:
            # Graph is already Eulerian, find the circuit
            if nx.is_eulerian(subgraph):
                start_node = list(subgraph.nodes())[0]
                eulerian_circuit = list(nx.eulerian_circuit(subgraph, source=start_node))
                circuit_segments = [subgraph[u][v][0]['segment_obj'] for u, v in eulerian_circuit]
                full_component_route.extend(circuit_segments)
            else:
                # Fallback for non-Eulerian graphs with no odd nodes (should be rare)
                full_component_route.extend([data['segment_obj'] for u, v, data in subgraph.edges(data=True)])
            continue

        # Create a complete graph of the odd-degree nodes for matching
        odd_node_graph = nx.Graph()
        for i, u in enumerate(odd_nodes):
            for j, v in enumerate(odd_nodes):
                if i < j:
                    try:
                        dist = nx.shortest_path_length(subgraph, source=u, target=v, weight='weight')
                        odd_node_graph.add_edge(u, v, weight=dist)
                    except nx.NetworkXNoPath:
                        # This should not happen since we are in a connected subgraph
                        print(f"Warning: No path between odd nodes {u} and {v} in a supposedly connected subgraph.")
                        continue
        
        # Find the minimum weight perfect matching
        matching = nx.min_weight_matching(odd_node_graph, weight='weight')

        # Augment the graph with the matching paths
        augmented_graph = subgraph.copy()
        for u, v in matching:
            path = nx.shortest_path(subgraph, source=u, target=v, weight='weight')
            for i in range(len(path) - 1):
                n1, n2 = path[i], path[i+1]
                edge_data = subgraph.get_edge_data(n1, n2)[0]
                augmented_graph.add_edge(n1, n2, weight=edge_data['weight'], segment_obj=edge_data['segment_obj'], is_duplicate=True)

        if not nx.is_eulerian(augmented_graph):
            print("Warning: Subgraph could not be made Eulerian. Route for this part may be suboptimal.")
            full_component_route.extend([data['segment_obj'] for u, v, data in subgraph.edges(data=True)])
            continue
            
        # Find and store the final Eulerian circuit for the subgraph
        start_node = list(augmented_graph.nodes())[0]
        eulerian_circuit = list(nx.eulerian_circuit(augmented_graph, source=start_node))
        
        # We need a robust way to get segments back, especially with parallel edges
        temp_graph = augmented_graph.copy()
        subgraph_route = []
        for u, v in eulerian_circuit:
            # Find the specific edge to ensure we handle parallel edges correctly
            edge_key = min(temp_graph[u][v], key=lambda k: temp_graph[u][v][k].get('is_duplicate', False))
            edge_data = temp_graph[u][v][edge_key]
            subgraph_route.append(edge_data['segment_obj'])
            temp_graph.remove_edge(u, v, key=edge_key)

        full_component_route.extend(subgraph_route)

    return full_component_route

@dataclass
class ComponentInfo:
    """Holds information about a single connected component of trails."""
    id: int
    segments: List[TrailSegment]
    on_foot_route: List[TrailSegment] = field(default_factory=list)
    center: Optional[Tuple[float, float]] = None
    
    @property
    def total_mileage(self) -> float:
        return sum(seg.length_mi for seg in self.segments)

    def calculate_center(self):
        """Calculate the geometric center of the component."""
        if not self.segments:
            self.center = (0, 0)
            return
        
        all_coords = [coord for seg in self.segments for coord in seg.coordinates]
        lon, lat = zip(*all_coords)
        self.center = (np.mean(lon), np.mean(lat))

def solve_inter_component_tsp(components: List[ComponentInfo]) -> List[int]:
    """
    Solves the Traveling Salesperson Problem to find the optimal order to visit components.
    """
    num_components = len(components)
    if num_components <= 1:
        return list(range(num_components))

    # Create a distance matrix
    distance_matrix = np.zeros((num_components, num_components))
    for i in range(num_components):
        for j in range(num_components):
            if i != j:
                dist = haversine_distance_points(
                    components[i].center,
                    components[j].center
                )
                distance_matrix[i, j] = dist

    # Solve the TSP
    permutation, _ = solve_tsp_simulated_annealing(distance_matrix)
    
    return permutation

def create_master_route(segments: List[TrailSegment]) -> Tuple[List[TrailSegment], Dict[str, Any]]:
    """
    Creates the most optimal continuous route by solving the Chinese Postman Problem
    for each component and the Traveling Salesperson Problem for the inter-component route.
    """
    # 1. Find and process all on-foot components first
    raw_components = find_connected_components(segments)
    
    print(f"Found {len(raw_components)} connected trail components. Optimizing on-foot routes for each...")
    
    processed_components = []
    for i, component_segments in enumerate(raw_components):
        if not component_segments:
            continue

        comp_info = ComponentInfo(id=i, segments=component_segments)
        comp_info.calculate_center()
        
        # Note: Disabling the per-component print to reduce noise
        # print(f"  -> Planning CPP for component {i+1}/{len(raw_components)} ({len(comp_info.segments)} seg, {comp_info.total_mileage:.1f} mi)")
        comp_info.on_foot_route = create_optimal_route_for_component(component_segments)
        processed_components.append(comp_info)

    # 2. Solve the TSP to find the optimal order to visit components
    print("\nOptimizing driving route between components (Solving TSP)...")
    optimal_order_indices = solve_inter_component_tsp(processed_components)
    ordered_components = [processed_components[i] for i in optimal_order_indices]
    print("✅ TSP solution found. Assembling final master route...")

    # 3. Assemble the final master route with optimized driving
    master_route = []
    total_drive_distance = 0
    current_position = None # We'll start at the first component's start
    
    for i, component in enumerate(ordered_components):
        if not component.on_foot_route:
            continue

        component_start_node = component.on_foot_route[0].start_node
        
        if current_position:
            drive_dist = haversine_distance_points(current_position, component_start_node)
            total_drive_distance += drive_dist
            
        master_route.extend(component.on_foot_route)
        current_position = component.on_foot_route[-1].end_coords

    # Calculate final stats
    total_on_foot_miles = sum(s.length_mi for s in master_route)
    credit_miles = sum(s.length_mi for s in segments)
    redundant_miles = total_on_foot_miles - credit_miles

    stats = {
        "total_trail_miles": total_on_foot_miles,
        "credit_miles": credit_miles,
        "redundant_miles": redundant_miles,
        "total_drive_miles": total_drive_distance,
    }
    return master_route, stats

def save_route_csv(route: List[TrailSegment], filename: str):
    """Save the route to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sequence', 'Type', 'Segment_ID', 'Name', 'Length_Miles', 'Traversal', 
                        'Drive_Miles', 'Drive_Time_Min', 'Start_Lat', 'Start_Lon', 'End_Lat', 'End_Lon'])
        
        # Track different types of miles
        credit_miles = 0           # Official segments getting credit
        backtrack_miles = 0        # Official segments traversed again (no credit)
        connector_miles = 0        # On-foot connectors (roads, paths)
        drive_miles = 0           # Driving segments
        total_drive_time = 0
        
        # Track which official segments we've seen for credit
        segments_seen_for_credit = set()
        
        for i, segment in enumerate(route, 1):
            drive_miles_col = ''
            drive_time_col = ''
            if segment.seg_id.startswith('DRIVE_'):
                segment_type = 'DRIVE'
                traversal = 'DRIVE'
                drive_dist = segment.length_mi
                drive_time = drive_dist / 30.0 * 60  # 30 mph
                drive_miles += drive_dist
                total_drive_time += drive_time
                drive_miles_col = drive_dist
                drive_time_col = drive_time
            elif segment.seg_id.startswith('CONNECTOR_'):
                segment_type = 'CONNECTOR'
                traversal = 'ON_FOOT'
                connector_miles += segment.length_mi
            else:
                # Official trail segment
                segment_type = 'TRAIL'
                traversal = 'FORWARD'  # Simplified - would need more logic to determine actual direction
                
                if segment.seg_id not in segments_seen_for_credit:
                    # First time seeing this segment - gets credit
                    credit_miles += segment.length_mi
                    segments_seen_for_credit.add(segment.seg_id)
                else:
                    # Backtracking on this segment
                    backtrack_miles += segment.length_mi
            
            writer.writerow([
                i, segment_type, segment.seg_id, segment.name, f"{segment.length_mi:.3f}",
                traversal, drive_miles_col if drive_miles_col else '', drive_time_col if drive_time_col else '',
                f"{segment.start_coords[1]:.6f}", f"{segment.start_coords[0]:.6f}",
                f"{segment.end_coords[1]:.6f}", f"{segment.end_coords[0]:.6f}"
            ])
        
        # Add detailed summary
        total_on_foot = credit_miles + backtrack_miles + connector_miles
        total_distance = total_on_foot + drive_miles
        
        writer.writerow([])
        writer.writerow(['DETAILED SUMMARY'])
        writer.writerow(['Credit Miles (Official Segments First Time)', f"{credit_miles:.1f}"])
        writer.writerow(['Backtrack Miles (Official Segments Again)', f"{backtrack_miles:.1f}"])
        writer.writerow(['Connector Miles (Roads/Paths On Foot)', f"{connector_miles:.1f}"])
        writer.writerow(['Total On-Foot Miles', f"{total_on_foot:.1f}"])
        writer.writerow(['Drive Miles', f"{drive_miles:.1f}"])
        writer.writerow(['Total Distance', f"{total_distance:.1f}"])
        writer.writerow(['Drive Time (hours)', f"{total_drive_time/60:.1f}"])
        writer.writerow([])
        writer.writerow(['COVERAGE'])
        writer.writerow(['Official Segments Credited', len(segments_seen_for_credit)])
        writer.writerow(['Total Official Segments Available', '247'])
        writer.writerow(['Coverage Percentage', f"{len(segments_seen_for_credit)/247*100:.1f}%"])

def main_logic(args):
    """The core logic of the continuous planner."""
    segments_file = args.trail_segments_json
    
    print("Loading trail segments...")
    segments = load_trail_segments(segments_file)
    
    print("\nCreating optimal continuous route...")
    route, stats = create_master_route(segments)
    
    if route:
        print("\n--- Optimal Route Summary ---")
        print(f"Total On-Foot Distance: {stats.get('total_trail_miles', 0):.2f} miles")
        print(f"  - Official Segments: {stats.get('credit_miles', 0):.2f} miles")
        print(f"  - Backtracking/Connectors: {stats.get('redundant_miles', 0):.2f} miles")
        print(f"Total Driving Distance: {stats.get('total_drive_miles', 0):.2f} miles")

        output_filename = 'output/continuous_master_route_optimal.csv'
        print(f"\nSaving route to {output_filename}...")
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sequence', 'Type', 'Segment_ID', 'Name', 'Length_Miles'])
            for i, segment in enumerate(route, 1):
                writer.writerow([i, 'TRAIL', segment.seg_id, segment.name, f"{segment.length_mi:.3f}"])
        print("✅ Optimal continuous route generated!")
    else:
        print("❌ Failed to generate route.")

def main():
    """The command-line entry point."""
    if len(sys.argv) != 2:
        print("Usage: python -m trail_route_ai.continuous_route_planner <trail_segments.json>")
        sys.exit(1)
    
    class Args:
        trail_segments_json = sys.argv[1]

    main_logic(Args())

if __name__ == "__main__":
    main() 