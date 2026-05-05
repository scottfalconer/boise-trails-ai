#!/usr/bin/env python3
"""
Daily Challenge Planner using a Vehicle Routing Problem (VRP) model.

This planner implements the formal strategy outlined in the project README to
generate an optimal, multi-day hiking plan that covers all required trail
segments of the Boise Trails Challenge.

It models the problem as a Capacitated Arc Routing Problem (CARP) and uses
Google's OR-Tools to find a solution that minimizes total on-foot distance
while respecting daily time/distance constraints.
"""
import os
import sys
import yaml
import json
import networkx as nx
import rasterio
import numpy as np
import csv
import gpxpy
import gpxpy.gpx
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial import KDTree

from .core.models import TrailSegment, PlannerConfig as PlannerConfigBase
from .core.utils import haversine_distance_points

@dataclass
class Hike:
    """Represents a single continuous on-foot loop."""
    hike_number: int
    trailhead_name: str
    segments: List[TrailSegment] = field(default_factory=list)
    total_distance_mi: float = 0.0
    total_elevation_ft: float = 0.0
    drive_to_next_mi: float = 0.0
    
@dataclass
class DailyPlan:
    """Represents a single day's hiking plan, which may contain multiple hikes."""
    day_number: int
    hikes: List[Hike] = field(default_factory=list)
    total_on_foot_mi: float = 0.0
    total_driving_mi: float = 0.0
    total_elevation_ft: float = 0.0

@dataclass
class PlannerConfig(PlannerConfigBase):
    """Loads and holds the configuration for the VRP planner."""
    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

def get_elevation_provider(dem_path: str) -> Optional[rasterio.io.DatasetReader]:
    """Opens the DEM file and returns a rasterio dataset object for querying."""
    if not os.path.exists(dem_path):
        print(f"Warning: DEM file not found at {dem_path}. Elevation data will not be used.")
        return None
    try:
        provider = rasterio.open(dem_path)
        print(f"✅ DEM file loaded successfully from {dem_path}.")
        return provider
    except Exception as e:
        print(f"Warning: Could not load DEM file. Error: {e}. Elevation data will not be used.")
        return None

def get_path_data(graph: nx.DiGraph, path_nodes: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """Retrieves segment and edge data for a given path of nodes."""
    path_data = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        edge_data = graph.get_edge_data(u, v)
        if edge_data and 'segment' in edge_data:
            path_data.append({
                'segment': edge_data['segment'],
                'elev_gain_ft': edge_data.get('elev_gain_ft', 0)
            })
    return path_data

def generate_turn_by_turn(segments: List[TrailSegment], trailhead: str) -> List[str]:
    """Generates human-readable turn-by-turn directions from a list of segments."""
    if not segments:
        return ["No route found."]

    directions = [f"Start by parking at {trailhead}. Begin hike on '{segments[0].name}'."]
    
    for i in range(len(segments) - 1):
        current_seg = segments[i]
        next_seg = segments[i+1]
        
        if current_seg.name == next_seg.name:
            directions.append(f"Continue on '{current_seg.name}'.")
        else:
            directions.append(f"At the end of '{current_seg.name}', proceed to '{next_seg.name}'.")
            
    directions.append(f"After completing '{segments[-1].name}', follow the trail network back to {trailhead}.")
    return directions

def load_trail_data(required_segments_path: str, all_trails_path: str) -> List[TrailSegment]:
    """
    Loads both required and optional trail segments, creating a unified list.
    """
    all_segments = {} # Use dict to handle potential duplicates easily

    # 1. Load the full trail network (optional segments)
    print(f"Loading full trail network from: {all_trails_path}")
    if os.path.exists(all_trails_path):
        with open(all_trails_path, 'r') as f:
            data = json.load(f)
        
        for feature in data['features']:
            props = feature['properties']
            geom = feature.get('geometry')
            if not geom or not geom.get('coordinates'):
                continue # Skip features with no geometry

            coords = geom['coordinates']
            
            # Handle multi-part geometries by flattening the coordinate list
            flat_coords = []
            if any(isinstance(i, list) and len(i) > 0 and isinstance(i[0], list) for i in coords):
                 # This is likely a MultiLineString
                for part in coords:
                    flat_coords.extend(part)
            else:
                flat_coords = coords

            # Standardize a unique ID. Using CART_ID if available, else generate.
            seg_id = props.get('CART_ID') or f"optional_{props.get('OBJECTID')}"

            segment = TrailSegment(
                seg_id=str(seg_id),
                name=props.get('TRAIL_NAME', 'Unnamed Trail'),
                coordinates=[(lon, lat) for lon, lat in flat_coords],
                length_ft=float(props.get('Shape_Length', 0)),
                direction='both', # Assume optional trails are bidirectional
                required=False,
                access_from=props.get('AccessFrom')
            )
            all_segments[segment.seg_id] = segment
    print(f"Loaded {len(all_segments)} total segments from the trail network.")

    # 2. Load the required challenge segments and mark them
    print(f"Loading required segments from: {required_segments_path}")
    if os.path.exists(required_segments_path):
        with open(required_segments_path, 'r') as f:
            data = json.load(f)
        
        required_count = 0
        for seg_data in data['trailSegments']:
            props = seg_data['properties']
            coords = seg_data['geometry']['coordinates']
            seg_id = str(props.get('segId'))

            # If this segment is already in our list, just update it to be required
            if seg_id in all_segments:
                all_segments[seg_id].required = True
                all_segments[seg_id].direction = props.get('direction', 'both')
            else:
                # If not, create it as a new required segment
                segment = TrailSegment(
                    seg_id=seg_id,
                    name=props.get('segName', ''),
                    coordinates=[(c[0], c[1]) for c in coords],
                    length_ft=float(props.get('LengthFt', 0)),
                    direction=props.get('direction', 'both'),
                    required=True,
                    access_from=props.get('AccessFrom')
                )
                all_segments[segment.seg_id] = segment
            required_count += 1
    print(f"Marked {required_count} segments as required for the challenge.")

    return list(all_segments.values())

def build_master_graph(segments: List[TrailSegment], config: PlannerConfig, dem_provider: Optional[rasterio.io.DatasetReader]) -> nx.DiGraph:
    """Builds a master directed graph from all trail segments."""
    G = nx.DiGraph()
    beta = config.cost_model.get('elevation_beta', 10.0)
    print("Building master directed graph with elevation costs...")

    def get_elev(lon, lat):
        if not dem_provider:
            return 0
        try:
            # Note: rasterio expects (x, y) which is (lon, lat)
            return list(dem_provider.sample([(lon, lat)]))[0][0]
        except:
            return 0

    for segment in segments:
        coords = segment.coordinates
        if len(coords) < 2:
            continue

        # For each segment, create one or two directed arcs
        # Arc 1: Forward (start -> end)
        start_node = coords[0]
        end_node = coords[-1]
        dist_mi = segment.length_ft / 5280.0
        
        start_elev = get_elev(start_node[0], start_node[1])
        end_elev = get_elev(end_node[0], end_node[1])
        elev_gain_ft = max(0, end_elev - start_elev)
        
        cost = dist_mi + (elev_gain_ft * beta / 5280.0) # Keep units consistent

        if segment.direction in ['both', 'ascent']:
            G.add_edge(start_node, end_node,
                       cost=cost,
                       required=segment.required,
                       segment=segment,
                       name=segment.name,
                       distance_mi=dist_mi,
                       elev_gain_ft=elev_gain_ft)
        
        # Arc 2: Backward (end -> start)
        elev_gain_ft_rev = max(0, start_elev - end_elev)
        cost_rev = dist_mi + (elev_gain_ft_rev * beta / 5280.0)

        if segment.direction in ['both', 'descent']:
             G.add_edge(end_node, start_node,
                       cost=cost_rev,
                       required=segment.required,
                       segment=segment,
                       name=segment.name,
                       distance_mi=dist_mi,
                       elev_gain_ft=elev_gain_ft_rev)

    print(f"Master graph built. Total nodes: {G.number_of_nodes()}, Total arcs: {G.number_of_edges()}")
    return G

def heal_graph(graph: nx.DiGraph, tolerance_meters: float = 30) -> nx.DiGraph:
    """Connects nearby but disconnected nodes using a KD-tree for efficiency."""
    print("Healing graph by connecting nearby nodes...")
    nodes = list(graph.nodes())
    if not nodes:
        return graph

    tree = KDTree(nodes)
    healed_edges = 0
    
    # Connect components
    components = list(nx.weakly_connected_components(graph))
    if len(components) > 1:
        print(f"Graph has {len(components)} disconnected components. Attempting to bridge them.")
        
        # Convert tolerance to miles for proper distance checks
        tolerance_miles = tolerance_meters / 1609.34
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1_nodes = list(components[i])
                comp2_nodes = list(components[j])
                
                # Use KDTree to find nearest candidates quickly, then validate by haversine
                comp1_tree = KDTree(comp1_nodes)
                # For each node in comp2, find nearest in comp1 (by degree distance as a proxy)
                distances, indices = comp1_tree.query(comp2_nodes, k=1)
                
                # Pick the overall closest pair by actual haversine distance
                min_pair = None
                min_pair_dist_mi = float('inf')
                for idx2, idx1 in enumerate(indices):
                    node2 = comp2_nodes[idx2]
                    node1 = comp1_nodes[idx1]
                    dist_mi = haversine_distance_points(node1, node2)
                    if dist_mi < min_pair_dist_mi:
                        min_pair_dist_mi = dist_mi
                        min_pair = (node1, node2)
                
                if min_pair and min_pair_dist_mi <= tolerance_miles:
                    node1, node2 = min_pair
                    if not graph.has_edge(node1, node2):
                        dist_mi = min_pair_dist_mi
                        segment = TrailSegment(seg_id=f'healed-{healed_edges}', name='Virtual Connector', direction='both', coordinates=[node1, node2], length_ft=dist_mi*5280)
                        graph.add_edge(node1, node2, cost=dist_mi, required=False, segment=segment, name='Virtual Connector', distance_mi=dist_mi, elev_gain_ft=0)
                        graph.add_edge(node2, node1, cost=dist_mi, required=False, segment=segment, name='Virtual Connector', distance_mi=dist_mi, elev_gain_ft=0)
                        healed_edges += 1
 
    if healed_edges > 0:
        print(f"✅ Graph healed. Added {healed_edges} virtual connector edges to bridge components.")
    else:
        print("✅ Graph appears to be fully connected or components are too far apart.")
        
    return graph

def find_nearest_node(graph: nx.DiGraph, lon: float, lat: float, nodes_subset: Optional[List[Tuple[float, float]]] = None) -> Optional[Tuple[float, float]]:
    """Finds the nearest node in the graph to a given lon/lat point."""
    if nodes_subset:
        candidate_nodes = nodes_subset
    else:
        candidate_nodes = list(graph.nodes())

    if not candidate_nodes:
        return None
        
    nodes = np.array(candidate_nodes)
    point = np.array([lon, lat])
    distances = np.sum((nodes - point)**2, axis=1)
    nearest_node_index = np.argmin(distances)
    return tuple(nodes[nearest_node_index])

def compute_shortest_path_cost(graph: nx.DiGraph, start_node: Tuple[float, float], end_node: Tuple[float, float]) -> float:
    """Computes the shortest path cost between two nodes in the graph."""
    try:
        return nx.shortest_path_length(graph, start_node, end_node, weight='cost')
    except nx.NetworkXNoPath:
        # If no path exists on-foot, return a large penalty so the solver avoids this transition
        return 1e6

def prepare_vrp_data(graph: nx.DiGraph, config: PlannerConfig) -> Optional[Dict[str, Any]]:
    """Prepares data for a multi-depot VRP."""
    required_arcs = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get('required')]
    depot_configs = config.trailhead_depots
    depot_nodes = [find_nearest_node(graph, d['lon'], d['lat']) for d in depot_configs]
    
    valid_depots = [(node, cfg['name']) for node, cfg in zip(depot_nodes, depot_configs) if node]
    if not valid_depots:
        print("Error: No valid depots found.")
        return None
        
    depot_nodes, depot_names = zip(*valid_depots)
    
    # Create locations list: depots first, then required arcs
    locations = list(depot_nodes) + [(u, v) for u, v, _ in required_arcs]
    num_locations = len(locations)
    num_depots = len(depot_nodes)

    print(f"Computing cost matrix for {num_locations} locations ({num_depots} depots, {len(required_arcs)} required arcs)...")
    
    # Cost Matrix Calculation
    cost_matrix = np.full((num_locations, num_locations), 999999, dtype=float)
    
    # Set diagonal to 0
    for i in range(num_locations):
        cost_matrix[i][i] = 0
    
    # Compute costs between all pairs of locations
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                loc_i = locations[i]
                loc_j = locations[j]
                
                # For from-location, use depot node directly; for an arc, use the END node of the arc
                if i < num_depots:
                    from_node = loc_i
                else:
                    # loc_i is (u, v)
                    from_node = loc_i[1]
                
                # For to-location, use depot node directly; for an arc, use the START node of the arc
                if j < num_depots:
                    to_node = loc_j
                else:
                    # loc_j is (u, v)
                    to_node = loc_j[0]
                
                cost = compute_shortest_path_cost(graph, from_node, to_node)
                cost_matrix[i][j] = int(cost * 1000)  # Convert to integer for OR-Tools

    # Create vehicle capacities list
    capacities = []
    for day_type, day_config in config.daily_capacities.items():
        for _ in range(day_config['vehicles']):
            capacities.append(int(day_config['capacity'] * 1000))

    return {
        'cost_matrix': cost_matrix.astype(int),
        'num_vehicles': sum(day['vehicles'] for day in config.daily_capacities.values()),
        'depots': list(range(num_depots)),
        'demands': [0] * num_depots + [int(arc[2]['cost'] * 1000) for arc in required_arcs],
        'capacities': capacities,
        'locations': locations,
        'depot_names': depot_names,
        'required_arcs': required_arcs,
    }

def solve_vrp(vrp_data: Dict[str, Any], config: PlannerConfig) -> Tuple[Any, Any, Any]:
    """Solves the VRP using OR-Tools."""
    # For multi-depot VRP, we need to specify start and end depots for each vehicle
    # We'll cycle through the available depots for each vehicle
    num_depots = len(vrp_data['depots'])
    start_depots = []
    end_depots = []
    
    for vehicle_id in range(vrp_data['num_vehicles']):
        depot_idx = vrp_data['depots'][vehicle_id % num_depots]
        start_depots.append(depot_idx)
        end_depots.append(depot_idx)
    
    manager = pywrapcp.RoutingIndexManager(
        len(vrp_data['cost_matrix']), 
        vrp_data['num_vehicles'], 
        start_depots, 
        end_depots
    )
    
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return vrp_data['cost_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return vrp_data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vrp_data['capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.seconds = config.solver_time_limit_seconds

    print("Solving VRP...")
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print("✅ VRP solution found!")
        return manager, routing, solution
    else:
        print("❌ No solution found.")
        return manager, routing, None

def decode_solution(vrp_data: Dict[str, Any], manager: Any, routing: Any, solution: Any, graph: nx.DiGraph, config: PlannerConfig) -> List[DailyPlan]:
    """
    Decodes the VRP solution into daily plans, splitting routes into separate
    hikes based on a driving distance threshold.
    """
    daily_plans = []
    num_depots = len(vrp_data['depot_names'])
    drive_threshold = config.drive_threshold_miles
    depot_nodes = vrp_data['locations'][:num_depots]

    for vehicle_id in range(vrp_data['num_vehicles']):
        # 1. Get the sequence of required arc indices for this vehicle's route.
        index = routing.Start(vehicle_id)
        route_arc_indices = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index >= num_depots:
                arc_idx = node_index - num_depots
                if arc_idx < len(vrp_data['required_arcs']):
                    route_arc_indices.append(arc_idx)
            index = solution.Value(routing.NextVar(index))

        if not route_arc_indices:
            continue
            
        # 2. Reconstruct the full path, splitting into hikes where necessary.
        day_plan = DailyPlan(day_number=len(daily_plans) + 1)
        
        # Get the assigned depot for this vehicle/day
        start_depot_node_idx = manager.IndexToNode(routing.Start(vehicle_id))
        current_trailhead_node = vrp_data['locations'][start_depot_node_idx]
        current_trailhead_name = vrp_data['depot_names'][start_depot_node_idx]
        
        last_physical_node = current_trailhead_node
        current_hike = Hike(hike_number=1, trailhead_name=current_trailhead_name)

        for i, arc_idx in enumerate(route_arc_indices):
            req_u, req_v, req_data = vrp_data['required_arcs'][arc_idx]
            
            # A. Calculate path from last location to the start of the current required arc
            connector_path_nodes = []
            connector_path_data = []
            connector_dist_mi = 0.0
            try:
                connector_path_nodes = nx.shortest_path(graph, source=last_physical_node, target=req_u, weight='cost')
                connector_path_data = get_path_data(graph, connector_path_nodes)
                connector_dist_mi = sum(p['segment'].length_ft for p in connector_path_data) / 5280.0
            except nx.NetworkXNoPath:
                # No on-foot path exists; force a drive split if we already have segments in the current hike
                if current_hike.segments:
                    # Finalize current hike by returning to its trailhead if possible
                    try:
                        return_path = nx.shortest_path(graph, source=last_physical_node, target=current_trailhead_node, weight='cost')
                        current_hike.segments.extend([d['segment'] for d in get_path_data(graph, return_path)])
                    except nx.NetworkXNoPath:
                        pass
                    day_plan.hikes.append(current_hike)
                    
                    # Start a new hike at the nearest depot to the next required arc start
                    next_trailhead_node = find_nearest_node(graph, lon=req_u[0], lat=req_u[1], nodes_subset=depot_nodes)
                    drive_dist = haversine_distance_points(last_physical_node, req_u)
                    current_hike.drive_to_next_mi = drive_dist
                    day_plan.total_driving_mi += drive_dist
                    current_trailhead_node = next_trailhead_node if next_trailhead_node else req_u
                    current_trailhead_name = vrp_data['depot_names'][depot_nodes.index(next_trailhead_node)] if next_trailhead_node in depot_nodes else "Unknown Trailhead"
                    current_hike = Hike(hike_number=len(day_plan.hikes) + 1, trailhead_name=current_trailhead_name)
                    last_physical_node = req_u
                else:
                    # If current hike is empty (at day start) and no connector path exists, just reposition to closest depot
                    next_trailhead_node = find_nearest_node(graph, lon=req_u[0], lat=req_u[1], nodes_subset=depot_nodes)
                    current_trailhead_node = next_trailhead_node if next_trailhead_node else req_u
                    current_trailhead_name = vrp_data['depot_names'][depot_nodes.index(next_trailhead_node)] if next_trailhead_node in depot_nodes else "Unknown Trailhead"
                    last_physical_node = req_u
                    connector_dist_mi = 0.0
                    connector_path_data = []

            # B. Check if a drive is needed due to long connector
            if connector_dist_mi > drive_threshold and current_hike.segments:
                # Finalize the previous hike
                try:
                    return_path = nx.shortest_path(graph, source=last_physical_node, target=current_trailhead_node, weight='cost')
                    current_hike.segments.extend([d['segment'] for d in get_path_data(graph, return_path)])
                except nx.NetworkXNoPath:
                    pass
                
                day_plan.hikes.append(current_hike)

                # Calculate drive distance to the next trailhead
                next_trailhead_node = find_nearest_node(graph, lon=req_u[0], lat=req_u[1], nodes_subset=depot_nodes)
                drive_dist = haversine_distance_points(last_physical_node, req_u)
                current_hike.drive_to_next_mi = drive_dist
                day_plan.total_driving_mi += drive_dist

                # Start a new hike
                current_trailhead_node = next_trailhead_node if next_trailhead_node else req_u
                current_trailhead_name = vrp_data['depot_names'][depot_nodes.index(next_trailhead_node)] if next_trailhead_node in depot_nodes else "Unknown Trailhead"
                current_hike = Hike(hike_number=len(day_plan.hikes) + 1, trailhead_name=current_trailhead_name)
                
                # We are now "at" the start of the required segment, so no new connector needed
                last_physical_node = req_u
            else:
                 # No drive needed (short connector); add the connector path to the current hike
                current_hike.segments.extend([p['segment'] for p in connector_path_data])
            
            # C. Add the required segment itself to the current hike
            current_hike.segments.append(req_data['segment'])
            last_physical_node = req_v

        # 3. Finalize the last hike of the day
        if current_hike.segments:
            try:
                return_path = nx.shortest_path(graph, source=last_physical_node, target=current_trailhead_node, weight='cost')
                current_hike.segments.extend([d['segment'] for d in get_path_data(graph, return_path)])
            except nx.NetworkXNoPath:
                pass
            day_plan.hikes.append(current_hike)
            
        # 4. Calculate final stats for the day
        for hike in day_plan.hikes:
            hike.total_distance_mi = sum(s.length_ft for s in hike.segments) / 5280.0
            day_plan.total_on_foot_mi += hike.total_distance_mi
        
        if day_plan.hikes:
            daily_plans.append(day_plan)
            
    return daily_plans

def save_summary_csv(daily_plans: List[DailyPlan], output_path: str):
    """Saves the daily plans to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Day', 'Hike_Number', 'Trailhead', 'On_Foot_Mi', 'Drive_To_Next_Hike_Mi', 'Segments_Covered']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for plan in daily_plans:
            for hike in plan.hikes:
                segment_names = '; '.join(list(dict.fromkeys([seg.name for seg in hike.segments if seg.required])))
                writer.writerow({
                    'Day': plan.day_number,
                    'Hike_Number': hike.hike_number,
                    'Trailhead': hike.trailhead_name,
                    'On_Foot_Mi': f"{hike.total_distance_mi:.2f}",
                    'Drive_To_Next_Hike_Mi': f"{hike.drive_to_next_mi:.2f}",
                    'Segments_Covered': segment_names
                })
    
    print(f"✅ Summary saved to {output_path}")

def generate_gpx_files(daily_plans: List[DailyPlan], output_dir: str):
    """Generates GPX files for each individual hike."""
    routes_dir = os.path.join(output_dir, 'routes')
    os.makedirs(routes_dir, exist_ok=True)
    
    # Clear old gpx files
    for f in os.listdir(routes_dir):
        if f.endswith('.gpx'):
            os.remove(os.path.join(routes_dir, f))

    for plan in daily_plans:
        for hike in plan.hikes:
            gpx = gpxpy.gpx.GPX()
            gpx_track = gpxpy.gpx.GPXTrack()
            gpx_track.name = f"Day {plan.day_number} Hike {hike.hike_number} - Park at {hike.trailhead_name}"
            gpx.tracks.append(gpx_track)
            
            gpx_segment = gpxpy.gpx.GPXTrackSegment()
            
            # Create a continuous list of coordinates for the entire hike
            full_coords = []
            for segment in hike.segments:
                full_coords.extend(segment.coordinates)

            for lon, lat in full_coords:
                gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))
            
            gpx_track.segments.append(gpx_segment)
            
            filename = f"day_{plan.day_number:02d}_hike_{hike.hike_number:02d}.gpx"
            filepath = os.path.join(routes_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(gpx.to_xml())
    
    print(f"✅ GPX files saved to {routes_dir}")

def main():
    """Main function to run the daily planner."""
    print("🏔️  Boise Trails Challenge - Daily Planner")
    print("=" * 50)
    
    # Load configuration
    config_path = 'config/daily_planner_config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    config = PlannerConfig.from_yaml(config_path)
    print(f"✅ Configuration loaded from {config_path}")
    
    # Load elevation data (optional)
    dem_provider = get_elevation_provider('data/elevation/boise_dem.tif')
    
    # Load trail data
    required_segments_path = 'data/traildata/GETChallengeTrailData_v2.json'
    all_trails_path = 'data/traildata/Boise_Parks_Trails_Open_Data.geojson'
    
    if not os.path.exists(required_segments_path):
        print(f"❌ Required segments file not found: {required_segments_path}")
        return
    
    segments = load_trail_data(required_segments_path, all_trails_path)
    print(f"✅ Loaded {len(segments)} total trail segments")
    
    required_segments = [s for s in segments if s.required]
    print(f"✅ Found {len(required_segments)} required segments for the challenge")
    
    # Build master graph
    master_graph = build_master_graph(segments, config, dem_provider)
    
    # Heal graph to connect nearby nodes
    master_graph = heal_graph(master_graph)
    
    # Prepare VRP data
    vrp_data = prepare_vrp_data(master_graph, config)
    if not vrp_data:
        print("❌ Failed to prepare VRP data")
        return
    
    # Solve VRP
    manager, routing, solution = solve_vrp(vrp_data, config)
    if not solution:
        print("❌ Failed to find VRP solution")
        return
    
    # Decode solution
    daily_plans = decode_solution(vrp_data, manager, routing, solution, master_graph, config)
    
    if not daily_plans:
        print("❌ No valid daily plans generated")
        return
    
    print(f"✅ Generated {len(daily_plans)} daily plans")
    
    # Save outputs
    save_summary_csv(daily_plans, 'output/daily_plan_summary.csv')
    generate_gpx_files(daily_plans, 'output')
    
    # Print summary
    total_distance = sum(plan.total_on_foot_mi for plan in daily_plans)
    total_driving = sum(plan.total_driving_mi for plan in daily_plans)
    
    print("\n📊 Plan Summary:")
    print(f"   • Total Days: {len(daily_plans)}")
    print(f"   • Total On-Foot Distance: {total_distance:.1f} miles")
    print(f"   • Total Driving Distance: {total_driving:.1f} miles")
    
    if dem_provider:
        dem_provider.close()

if __name__ == "__main__":
    main() 