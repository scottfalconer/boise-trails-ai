"""
Functions for analyzing trail network topology, identifying groups, and detecting loops.
"""
from collections import defaultdict
import re
from typing import List, Dict, Set, Tuple

from .planner_utils import Edge # Assuming Edge is in planner_utils

def identify_natural_trail_groups(segments: List[Edge]) -> Dict[str, List[Edge]]:
    """
    Groups trail segments by their natural trail name families.
    For example, "Dry Creek Trail 1", "Dry Creek Trail #2" are grouped under "Dry Creek Trail".
    """
    trail_groups: Dict[str, List[Edge]] = defaultdict(list)

    # Regex to capture the base trail name, removing numbers, #, "No.", etc.
    # It looks for a name part, optionally followed by a space and then common numbering patterns.
    # Example: "Dry Creek Trail No. 1", "Shingle Creek Trail #2", "Trail 3"
    # It aims to extract "Dry Creek Trail", "Shingle Creek Trail", "Trail"
    name_pattern = re.compile(r"^(.*?)(?:\s+(?:No\.?|#)?\s*\d+)?$", re.IGNORECASE)

    for segment in segments:
        if not segment.name:
            # Assign segments with no name to a generic "Unnamed Trails" group
            # or handle them as individual groups if preferred.
            base_name = "Unnamed Trails"
        else:
            match = name_pattern.match(segment.name.strip())
            if match:
                base_name = match.group(1).strip()
                # Further cleanup: remove trailing "Trail" if the base_name itself is just "Trail"
                # to avoid grouping "Trail 1" and "Ridge Road Trail" under "Trail"
                if base_name.upper() == "TRAIL" and segment.name.upper() != "TRAIL":
                    # This is a simple segment like "Trail 1", "Trail 2"
                    # Keep its specific name for now, or group them as "Numbered Trails"
                    # For now, let's use the full name if base is just "Trail" to avoid over-grouping.
                    # This logic can be refined.
                    pass # Keep base_name as "Trail"

                # Remove common suffixes like "Trail", "Path", "Loop" if they are part of the base name
                # to further normalize, e.g. "Dry Creek Trail" and "Dry Creek Path"
                # This is a simple approach; more sophisticated normalization might be needed.
                common_suffixes = [" Trail", " Path", " Loop", " Spur", " Connector"]
                for suffix in common_suffixes:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
            else:
                # If regex doesn't match (e.g. unusual naming), use the full name as base.
                base_name = segment.name.strip()

        trail_groups[base_name].append(segment)

    return dict(trail_groups)

def get_segment_endpoints(segments: List[Edge]) -> Set[Tuple[float, float]]:
    """Extracts all unique start and end points from a list of segments."""
    endpoints: Set[Tuple[float, float]] = set()
    for segment in segments:
        endpoints.add(segment.start)
        endpoints.add(segment.end)
    return endpoints

def find_loops(
    graph: "nx.DiGraph", # type: ignore
    trail_group_segments: List[Edge],
    min_segments: int = 3,
    min_length_mi: float = 1.0,
    max_length_mi: float = float('inf')
) -> List[List[Edge]]:
    """
    Finds potential loops within a given group of trail segments using the graph.

    Args:
        graph: The trail network graph (preferably undirected for loop finding).
        trail_group_segments: A list of Edge objects belonging to a specific trail group.
        min_segments: Minimum number of unique segments a loop must contain.
        min_length_mi: Minimum total length of a loop in miles.
        max_length_mi: Maximum total length of a loop in miles.

    Returns:
        A list of loops, where each loop is represented as a list of Edge objects.
    """
    import networkx as nx # Import networkx here

    if not trail_group_segments:
        return []

    # Create a subgraph containing only the segments from the current trail group
    # and their incident nodes.
    segment_ids_in_group = {s.seg_id for s in trail_group_segments if s.seg_id}

    # We need all nodes that are part of these segments
    nodes_in_group: Set[Tuple[float, float]] = set()
    for seg in trail_group_segments:
        nodes_in_group.add(seg.start)
        nodes_in_group.add(seg.end)

    # Create an undirected subgraph from the main graph, containing only nodes involved in the trail group.
    # This is important because loops can be traversed in either direction.
    # We consider all edges in the main graph that connect nodes within our group.
    subgraph_undirected = nx.Graph()
    for u, v, data in graph.edges(data=True):
        edge_obj = data.get('edge')
        if not edge_obj:
            continue # Should not happen if graph is built correctly

        # Add edge if both its endpoints are in our group of nodes
        # Or if the edge itself is one of the trail_group_segments (important for connector trails that might be part of a loop)
        is_segment_in_group = edge_obj.seg_id in segment_ids_in_group

        if (u in nodes_in_group and v in nodes_in_group) or is_segment_in_group:
            # Ensure we use the canonical Edge object for easy comparison later.
            # The graph might store reversed versions if direction='both'.
            # For loop detection, the specific Edge instance matters.

            # Find the canonical edge from trail_group_segments if this edge_obj matches its ID
            canonical_edge = None
            if edge_obj.seg_id and edge_obj.seg_id in segment_ids_in_group:
                for sgs in trail_group_segments:
                    if sgs.seg_id == edge_obj.seg_id:
                        canonical_edge = sgs
                        break

            if canonical_edge:
                 subgraph_undirected.add_edge(u, v, edge=canonical_edge, weight=data.get('weight', 1.0))
            elif not segment_ids_in_group : # If no specific segments, take any edge between nodes in group
                 subgraph_undirected.add_edge(u, v, edge=edge_obj, weight=data.get('weight', 1.0))


    if subgraph_undirected.number_of_edges() == 0:
        return []

    identified_loops: List[List[Edge]] = []

    # Find all simple cycles in the undirected subgraph.
    # A simple cycle is one with no repeated vertices, except for the start/end vertex.
    # nx.cycle_basis is good for fundamental cycles in an undirected graph.
    # nx.simple_cycles is for directed, but can be used if we construct G_trail_group carefully.
    # For undirected, elementary circuits are what we want.

    # Using nx.cycle_basis for undirected graphs. Each cycle is a list of nodes.
    # We need to convert these node paths back to edge paths.
    cycles_node_paths = list(nx.cycle_basis(subgraph_undirected))

    for node_path in cycles_node_paths:
        if len(node_path) < min_segments: # A cycle of N nodes has N edges
            continue

        current_loop_edges: List[Edge] = []
        current_loop_length_mi = 0.0
        valid_cycle = True

        # Ensure the path is closed by appending the start node if not already present
        # (cycle_basis usually returns them closed)
        path_to_check = node_path + [node_path[0]] if node_path[0] != node_path[-1] else node_path

        if len(path_to_check)-1 < min_segments: # Check actual edge count
             continue

        edge_ids_in_loop: Set[str] = set()

        for i in range(len(path_to_check) - 1):
            u, v = path_to_check[i], path_to_check[i+1]
            if subgraph_undirected.has_edge(u,v):
                # In an undirected graph with multiedges possible (though our build_nx_graph doesn't create them by default for same u,v pair without different keys)
                # we take the first one. If specific edge instances are needed, graph construction needs care.
                # Our subgraph stores the canonical edge.
                edge_data = subgraph_undirected.get_edge_data(u,v)
                loop_edge: Edge = edge_data['edge']

                # Check directionality: if the original segment (from graph, not trail_group_segments)
                # was one-way, we must respect it. Our 'loop_edge' is canonical.
                # This check is complex because graph stores directed edges.
                # For now, assume cycle_basis on Undirected graph means segments are traversable.
                # Refinement: check if graph.has_edge(u,v) or graph.has_edge(v,u) exists for one-ways.

                current_loop_edges.append(loop_edge)
                current_loop_length_mi += loop_edge.length_mi
                if loop_edge.seg_id:
                    edge_ids_in_loop.add(loop_edge.seg_id)
            else:
                valid_cycle = False # Should not happen if cycle_basis is correct
                break

        if not valid_cycle or not current_loop_edges:
            continue

        # Filter by number of unique segments and length
        if len(edge_ids_in_loop) < min_segments: # Count unique segments
            continue
        if not (min_length_mi <= current_loop_length_mi <= max_length_mi):
            continue

        # Avoid duplicate loops (e.g., same set of edges in different order or start point)
        # Sort by segment ID to create a canonical representation of the loop
        canonical_loop_sig = tuple(sorted([e.seg_id for e in current_loop_edges if e.seg_id]))

        is_duplicate = False
        for existing_loop in identified_loops:
            existing_sig = tuple(sorted([e.seg_id for e in existing_loop if e.seg_id]))
            if canonical_loop_sig == existing_sig:
                is_duplicate = True
                break
        if not is_duplicate:
            identified_loops.append(current_loop_edges)

    return identified_loops

def is_cluster_routable(
    graph: "nx.DiGraph", # type: ignore
    cluster_segments: List[Edge],
    allow_virtual_connectors: bool = True
) -> bool:
    """
    Checks if all segments in a potential cluster form a single connected component
    in the provided graph.

    Args:
        graph: The main trail network graph (DiGraph).
        cluster_segments: A list of Edge objects representing the cluster.
        allow_virtual_connectors: If True, considers 'virtual_connector' edges
                                  as valid connections.

    Returns:
        True if the cluster is connected (routable), False otherwise.
    """
    import networkx as nx # Import networkx here

    if not cluster_segments:
        return True # An empty cluster is technically routable (by doing nothing)

    if len(cluster_segments) == 1:
        # A single segment is always routable if its endpoints are in the graph.
        # (build_nx_graph in challenge_planner already adds virtual connectors if endpoints are missing)
        seg = cluster_segments[0]
        return graph.has_node(seg.start) and graph.has_node(seg.end)

    cluster_nodes: Set[Tuple[float, float]] = set()
    for seg in cluster_segments:
        cluster_nodes.add(seg.start)
        cluster_nodes.add(seg.end)

    if not cluster_nodes: # Should not happen if cluster_segments is not empty
        return False

    # Create a subgraph containing only the nodes and edges relevant to this cluster
    # We need to consider edges from the main graph that connect these nodes.

    # We build an UNDIRECTED subgraph for connectivity checking.
    subgraph = nx.Graph()
    subgraph.add_nodes_from(cluster_nodes)

    for u, v, data in graph.edges(data=True):
        edge_obj: Edge = data.get('edge')
        if not edge_obj:
            continue

        # Include the edge if its endpoints are within our cluster_nodes
        if u in cluster_nodes and v in cluster_nodes:
            if edge_obj.kind == 'virtual' and not allow_virtual_connectors:
                continue
            # Add the edge regardless of whether it's one of the cluster_segments itself,
            # as long as it connects nodes within the cluster.
            subgraph.add_edge(u, v, edge=edge_obj)

            # If the original graph is directed and has edges in both directions for a segment,
            # the above will add it once to the undirected subgraph.

    if subgraph.number_of_nodes() == 0: # e.g. if cluster_nodes was empty or nodes not in G
        return False

    # Check if this subgraph is connected
    # Important: is_connected throws an error if the graph has no nodes.
    if not nx.is_connected(subgraph):
        return False

    # Final check: ensure all segments in cluster_segments are actually represented in the connected subgraph.
    # This handles cases where some segments might be isolated from the main connected component
    # of the subgraph, even if the subgraph itself has a large connected piece.
    # (e.g. cluster_segments = [s1, s2, sisolated], s1-s2 connected, sisolated not)

    # Get the largest connected component of the subgraph
    # (If is_connected was true, there's only one component covering all nodes in subgraph)
    # components = list(nx.connected_components(subgraph)) - not needed if is_connected is true
    # largest_component_nodes = max(components, key=len) if components else set()

    # All nodes in the subgraph must be part of the single component if nx.is_connected(subgraph) is true.
    # So, we just need to check if all segment endpoints from cluster_segments are in the subgraph.
    # This is already implicitly handled by adding nodes from cluster_nodes to subgraph.
    # The critical part is that all these nodes must belong to the *same* connected component.

    # A more robust check: each segment in cluster_segments must have its endpoints
    # reachable from each other within the subgraph. If the whole subgraph is connected,
    # and all segment endpoints are in it, this is true.

    # One edge case: what if a segment's nodes are in cluster_nodes, but the segment itself
    # isn't an edge in `graph` (e.g. it's a conceptual segment not physically in graph)?
    # The current function assumes `graph` is the source of truth for connectivity.
    # `cluster_segments` defines *which* segments we care about connecting.

    # If we reach here, the subgraph containing all nodes of the cluster_segments is connected.
    return True
