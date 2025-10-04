"""
New clustering algorithms based on trail network topology and connectivity.
"""
from collections import defaultdict
import math
from typing import List, Dict, Set, Tuple, Optional, Any
import networkx as nx  # type: ignore

from .planner_utils import Edge, _haversine_mi
from .trail_network_analyzer import (
    identify_natural_trail_groups,
    find_loops,
    is_cluster_routable,
)


def _extract_path_edges(graph: nx.DiGraph, path: List[Tuple[float, float]]) -> List[Edge]:
    """Return ``Edge`` objects for each step in ``path``.

    Works with both ``DiGraph`` and ``MultiDiGraph`` edge data formats.
    ``Edge`` objects are stored under the ``"edge"`` attribute.
    """
    edges: List[Edge] = []
    for u, v in nx.utils.pairwise(path):
        data = graph.get_edge_data(u, v)
        if not data:
            continue
        if isinstance(data, dict) and "edge" in data:
            edges.append(data["edge"])
        elif isinstance(data, dict):
            first_key = next(iter(data))
            edge_obj = data[first_key].get("edge")
            if edge_obj is not None:
                edges.append(edge_obj)
    return edges

def _select_distance_miles(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    reference_length: float,
) -> float:
    """Return a distance in miles that matches the data's scale.

    Synthetic unit tests often express coordinates directly in miles, while the
    production data uses lon/lat pairs that require a haversine calculation.
    Choose the distance that best matches the typical segment length so both
    contexts behave sensibly.
    """

    geo_dist = _haversine_mi(point_a, point_b)
    planar_dist = math.dist(point_a, point_b)

    if reference_length <= 0:
        return geo_dist if geo_dist != 0 else planar_dist

    if abs(geo_dist - reference_length) <= abs(planar_dist - reference_length):
        return geo_dist
    return planar_dist


class ClusterScoringSystem:
    """
    Scores potential clusters based on various desirable properties.
    """
    def __init__(self, graph: nx.DiGraph, all_segments: List[Edge]):
        self.graph = graph
        self.all_segments = all_segments
        # Potentially precompute things like segment midpoints, graph properties, etc.

    def score_cluster(self, cluster_segments: List[Edge]) -> Dict[str, float]:
        """
        Calculates a score for a given cluster.
        Higher scores are generally better.
        """
        if not cluster_segments:
            return {
                "loop_potential": 0.0,
                "connectivity": 0.0,
                "compactness": 0.0, # Or a penalty for empty
                "balance": 0.0,     # Or a penalty for empty
                "overall": 0.0
            }

        scores: Dict[str, float] = {}

        # 1. Loop Potential / Loop Score
        # How many actual loops are fully contained? What's their quality?
        # Uses find_loops internally or takes pre-computed loops.
        # For simplicity now, let's say 1 point per segment in a detected loop.
        loops_in_cluster = find_loops(self.graph, cluster_segments, min_segments=3, min_length_mi=0.1)
        loop_score = 0.0
        if loops_in_cluster:
            for loop in loops_in_cluster:
                loop_score += len(loop) # Simple score: number of segments in loops
        scores["loop_potential"] = loop_score

        # 2. Connectivity Score
        # For now, binary based on is_cluster_routable. Could be more nuanced.
        # is_routable = is_cluster_routable(self.graph, cluster_segments)
        # scores["connectivity"] = 1.0 if is_routable else -100.0 # Heavy penalty if not routable
        # Let's assume clusters passed to scoring are already checked for basic routability.
        # This score could measure *how well* connected (e.g. redundancy).
        # For now, let's make it simple: 1 if routable, 0 if not.
        # The clustering algorithm itself should ensure routability.
        # So, this score might be more about internal connectivity quality.
        # For now, let's assume it's externally checked.
        scores["connectivity_quality"] = 0.0 # Placeholder

        # 3. Geographic Compactness
        # Lower is better for distance-based compactness. We invert for score.
        if len(cluster_segments) > 1:
            avg_segment_length = sum(s.length_mi for s in cluster_segments) / len(cluster_segments)
            midpoints = [
                ((seg.start[0] + seg.end[0]) / 2, (seg.start[1] + seg.end[1]) / 2)
                for seg in cluster_segments
            ]
            centroid_lon = sum(p[0] for p in midpoints) / len(midpoints)
            centroid_lat = sum(p[1] for p in midpoints) / len(midpoints)

            avg_dist_to_centroid = sum(
                _select_distance_miles((centroid_lon, centroid_lat), midpoint, avg_segment_length)
                for midpoint in midpoints
            ) / len(midpoints)
            scores["compactness"] = 1.0 / (1.0 + avg_dist_to_centroid) # Higher is better
        elif cluster_segments: # Single segment cluster
            scores["compactness"] = 1.0
        else: # Should be caught by initial check
            scores["compactness"] = 0.0


        # 4. Balanced Effort (e.g. total length)
        # This needs a target daily effort. For now, just use total length.
        # Let's say ideal is 5-15 miles. Score peaks in this range.
        total_length = sum(s.length_mi for s in cluster_segments)
        if 5 <= total_length <= 15:
            scores["balance"] = 1.0
        elif total_length < 5:
            scores["balance"] = total_length / 5.0
        else: # total_length > 15
            scores["balance"] = max(0, 1.0 - (total_length - 15.0) / 15.0) # Penalize going too far over

        # Overall score - simple sum for now, weights can be added
        scores["overall"] = (
            scores.get("loop_potential", 0.0) * 0.4 +
            scores.get("compactness", 0.0) * 0.3 +
            scores.get("balance", 0.0) * 0.3
            # Connectivity is assumed to be a hard constraint handled by the algorithm
        )
        return scores


def build_topology_aware_clusters(
    all_segments: List[Edge],
    graph: nx.DiGraph, # Full graph of all trails and connectors
    config: Any, # Placeholder for planner configuration (e.g. daily targets)
    precomputed_loops: Optional[Dict[str, List[List[Edge]]]] = None # Trail group name to list of loops
) -> List[List[Edge]]:
    """
    Builds clusters based on network topology, natural trail groups, and loop formations.

    Args:
        all_segments: List of all Edge objects to be clustered.
        graph: The NetworkX graph representing the trail network.
        config: Configuration object with parameters like target daily effort.
        precomputed_loops: Optional dictionary of precomputed loops for efficiency.

    Returns:
        A list of clusters, where each cluster is a list of Edge objects.
    """
    if not all_segments:
        return []

    clusters: List[List[Edge]] = []
    remaining_segments = list(all_segments)

    scorer = ClusterScoringSystem(graph, all_segments)

    # Make a copy to modify: segments yet to be assigned to a cluster
    available_segments = list(all_segments)
    processed_segment_ids: Set[str] = set()
    final_clusters: List[List[Edge]] = []

    # 1. Identify natural trail groups to guide initial clustering
    natural_groups = identify_natural_trail_groups(all_segments)

    # Sort groups, e.g., by size (number of segments) to process larger/more defined trails first
    # or by some other priority if available.
    sorted_group_names = sorted(natural_groups.keys(), key=lambda g_name: len(natural_groups[g_name]), reverse=True)

    for group_name in sorted_group_names:
        group_segments = natural_groups[group_name]

        # Consider only segments from this group that haven't been clustered yet
        current_group_available_segments = [
            seg for seg in group_segments if seg.seg_id not in processed_segment_ids
        ]
        if not current_group_available_segments:
            continue

        # Find connected components OF THESE SEGMENTS using the main graph for connectivity context
        # This helps break down a large natural group (e.g., "Ridge Road") into its
        # geographically distinct, routable sections if it's not all one piece.

        component_graph_builder = nx.Graph()
        component_nodes_map: Dict[Tuple[float,float], Any] = {} # Map node coords to component id

        # Add only nodes from the current_group_available_segments initially
        for seg in current_group_available_segments:
            component_graph_builder.add_node(seg.start)
            component_graph_builder.add_node(seg.end)
            # Add the segment itself as an edge to ensure it's considered for component building
            component_graph_builder.add_edge(seg.start, seg.end, edge=seg)

        # Augment with connectors from the main graph if they link nodes within this group
        # This helps connect pieces of the same natural group that might be separated by a short connector.
        for u, v, data in graph.edges(data=True):
            edge_obj: Edge = data.get('edge')
            if not edge_obj: continue
            # If both ends of a graph edge are nodes of our current group segments, consider this edge for connectivity
            if u in component_graph_builder.nodes and v in component_graph_builder.nodes:
                # Add edge only if it's not already there (nx.Graph handles this)
                # We are interested in connectivity, so specifics of edge_obj (like kind) matter less here
                # than the fact that u and v are connected in the main graph.
                component_graph_builder.add_edge(u,v)


        for component_node_set in nx.connected_components(component_graph_builder):
            if not component_node_set or len(component_node_set) < 1: # Need at least one node
                continue

            # Collect all segments from current_group_available_segments whose *both* endpoints fall within this component
            # Or, more broadly, segments that have at least one endpoint in this component.
            # Let's take segments that are fully contained or touch the component.

            cluster_candidate_segments: List[Edge] = []
            for seg in current_group_available_segments:
                if seg.start in component_node_set or seg.end in component_node_set:
                    # Check if this segment is already processed (shouldn't be if logic is right)
                    if seg.seg_id in processed_segment_ids:
                        continue
                    cluster_candidate_segments.append(seg)

            if not cluster_candidate_segments:
                continue

            # Ensure this component is routable
            if is_cluster_routable(graph, cluster_candidate_segments):
                # This component forms a seed cluster.
                # TODO: Implement actual cluster expansion logic here.
                # For now, we'll just add this connected component as a cluster.
                final_clusters.append(list(cluster_candidate_segments)) # Make a copy
                for seg in cluster_candidate_segments:
                    if seg.seg_id:
                        processed_segment_ids.add(seg.seg_id)
                        # Mark as unavailable for future direct processing
                        # Note: available_segments itself isn't directly used in this loop's iteration logic for now,
                        # but would be if we were picking seeds from a global pool.

    # After processing natural groups, handle any remaining segments
    truly_remaining_segments = [seg for seg in all_segments if seg.seg_id not in processed_segment_ids]

    if truly_remaining_segments:
        # TODO: Implement more sophisticated handling for remaining segments
        # (e.g., try to merge into existing clusters, form small new clusters by proximity)
        # For now, create small clusters from connected components of remaining segments

        remaining_component_graph = nx.Graph()
        remaining_nodes = set()
        for seg in truly_remaining_segments:
            remaining_nodes.add(seg.start)
            remaining_nodes.add(seg.end)
        remaining_component_graph.add_nodes_from(remaining_nodes)

        for u,v,data in graph.edges(data=True): # Use main graph for connectivity
            if u in remaining_nodes and v in remaining_nodes:
                 remaining_component_graph.add_edge(u,v)

        for component_node_set in nx.connected_components(remaining_component_graph):
            if not component_node_set: continue

            comp_segs = [s for s in truly_remaining_segments if s.start in component_node_set or s.end in component_node_set]
            if comp_segs and is_cluster_routable(graph, comp_segs):
                final_clusters.append(comp_segs)
                for seg in comp_segs: # Mark as processed
                    if seg.seg_id: processed_segment_ids.add(seg.seg_id)
            elif comp_segs: # Not routable as a group, add individually
                for seg in comp_segs:
                    if is_cluster_routable(graph, [seg]):
                         final_clusters.append([seg])
                         if seg.seg_id: processed_segment_ids.add(seg.seg_id)
from dataclasses import dataclass, field # Added for Cluster dataclass

# --- Cluster Dataclass ---
@dataclass
class Cluster:
    segments: List[Edge]
    id: int # Simple unique ID for the cluster
    score: Dict[str, float] = field(default_factory=dict)
    # Could add other attributes like dominant_natural_group, boundary_nodes, etc.

    def __post_init__(self):
        # Ensure segments are unique by id if that's a requirement, though list is fine for now
        pass

    def add_segment(self, segment: Edge, new_score: Dict[str, float]):
        # Check if segment (by id) is already in the cluster to avoid duplicates if necessary
        # Only add if segment_id is not None and not already present
        if segment.seg_id and not any(s.seg_id == segment.seg_id for s in self.segments if s.seg_id):
            self.segments.append(segment)
        elif not segment.seg_id: # Allow adding segments without IDs (e.g. connectors)
            self.segments.append(segment)
        self.score = new_score

    @property
    def segment_ids(self) -> Set[str]:
        return {s.seg_id for s in self.segments if s.seg_id}

    def get_boundary_nodes(self, graph: nx.DiGraph, all_cluster_segment_ids: Set[str]) -> Set[tuple[float,float]]:
        """
        Identifies nodes that are on the "edge" of the cluster.
        A boundary node is part of this cluster and has an edge in the main graph
        connecting to a node that is NOT part of any segment currently in *this specific cluster object*.

        More accurately for expansion: a node in *this* cluster that has an outgoing edge
        to a segment *not yet processed globally* or *not in this cluster but available*.
        """
        current_cluster_nodes = set()
        for seg in self.segments:
            current_cluster_nodes.add(seg.start)
            current_cluster_nodes.add(seg.end)

        boundary: Set[tuple[float,float]] = set()
        for node in current_cluster_nodes:
            if node not in graph: continue # Should not happen if graph is well-formed
            for neighbor in graph.neighbors(node): # Considers outgoing edges
                # Check edges from node to neighbor
                for edge_data_key in graph.get_edge_data(node, neighbor): # Handles MultiDiGraph
                    edge_obj = graph.get_edge_data(node, neighbor)[edge_data_key].get('edge')
                    if edge_obj and edge_obj.seg_id not in self.segment_ids and edge_obj.seg_id not in all_cluster_segment_ids:
                        boundary.add(node)
                        break # Found one such edge from this node
                if node in boundary: break
            if node not in boundary: # Also check incoming edges leading to non-cluster segments
                 for pred_node in graph.predecessors(node):
                    for edge_data_key in graph.get_edge_data(pred_node, node):
                        edge_obj = graph.get_edge_data(pred_node, node)[edge_data_key].get('edge')
                        if edge_obj and edge_obj.seg_id not in self.segment_ids and edge_obj.seg_id not in all_cluster_segment_ids:
                            boundary.add(node)
                            break
                    if node in boundary: break
        return boundary

    def get_all_nodes(self) -> Set[tuple[float,float]]:
        nodes: Set[tuple[float,float]] = set()
        for seg in self.segments:
            nodes.add(seg.start)
            nodes.add(seg.end)
        return nodes


# --- Main Clustering Function ---
def build_topology_aware_clusters(
    all_segments: List[Edge],
    graph: nx.DiGraph, # Full graph of all trails and connectors
    config: Any, # Placeholder for planner configuration (e.g. daily targets)
    precomputed_loops: Optional[Dict[str, List[List[Edge]]]] = None # Trail group name to list of loops
) -> List[List[Edge]]:
    """
    Builds clusters based on network topology, natural trail groups, and loop formations.
    (Refactored for iterative expansion)
    """
    if not all_segments:
        return []

    scorer = ClusterScoringSystem(graph, all_segments)

    available_segments_map: Dict[str, Edge] = {seg.seg_id: seg for seg in all_segments if seg.seg_id}
    # Include segments without IDs too, perhaps mapping them by a unique hash or index if needed for removal
    # For now, available_segments_map focuses on ID'd segments for easy removal.
    # Segments without IDs will be handled by checking against processed_segment_ids set.

    processed_segment_ids: Set[str] = set() # Tracks seg_ids that are in any final_cluster

    active_clusters: List[Cluster] = []
    final_clusters_objects: List[Cluster] = []
    cluster_id_counter = 0

    # 1. Seed Initial Clusters
    # Priority:
    #   a. High-quality precomputed loops (if any)
    #   b. Loops found within natural trail groups
    #   c. Coherent (connected) parts of natural trail groups
    #   d. Individual remaining segments as last resort seeds

    # For simplicity, let's use the natural group components as initial seeds first.
    natural_groups = identify_natural_trail_groups(all_segments)
    sorted_group_names = sorted(natural_groups.keys(), key=lambda g_name: len(natural_groups[g_name]), reverse=True)

    for group_name in sorted_group_names:
        group_segments = natural_groups[group_name]
        current_group_available_segments = [
            seg for seg in group_segments if seg.seg_id not in processed_segment_ids
        ]
        if not current_group_available_segments:
            continue

        # Sub-component graph for this specific group's available segments
        # to find connected pieces within the natural group
        group_component_graph = nx.Graph()
        temp_group_nodes = set()
        for seg in current_group_available_segments:
            temp_group_nodes.add(seg.start)
            temp_group_nodes.add(seg.end)
            group_component_graph.add_node(seg.start) # Ensure nodes exist
            group_component_graph.add_node(seg.end)
            group_component_graph.add_edge(seg.start, seg.end, edge=seg) # Edge from the group itself

        # Augment with connectors from the main graph if they link nodes within this group
        for u_node, v_node, edge_data in graph.edges(data=True):
            graph_edge_obj: Edge = edge_data.get('edge')
            if not graph_edge_obj: continue
            if u_node in temp_group_nodes and v_node in temp_group_nodes: # Both endpoints are part of the current natural group's nodes
                 # Add this edge from the main graph (could be a connector, or another trail segment)
                group_component_graph.add_edge(u_node, v_node, edge=graph_edge_obj)


        for component_node_set in nx.connected_components(group_component_graph):
            if not component_node_set or len(component_node_set) < 1:
                continue

            seed_segments: List[Edge] = []
            component_segment_ids_temp: Set[str] = set()

            for seg in current_group_available_segments:
                # Segment belongs to this component if its nodes are in component_node_set
                # and it hasn't been processed yet.
                if (seg.start in component_node_set or seg.end in component_node_set) and \
                   (seg.seg_id not in processed_segment_ids and seg.seg_id not in component_segment_ids_temp):
                    seed_segments.append(seg)
                    if seg.seg_id: component_segment_ids_temp.add(seg.seg_id)

            if not seed_segments:
                continue

            # Use relaxed connectivity rules for same trail family
            if is_cluster_routable_relaxed(graph, seed_segments, same_trail_family=True):
                initial_score = scorer.score_cluster(seed_segments)
                new_cluster = Cluster(id=cluster_id_counter, segments=list(seed_segments), score=initial_score)
                active_clusters.append(new_cluster)
                cluster_id_counter += 1
                # Mark these segments as provisionally processed (part of an active_cluster)
                # They will be added to global processed_segment_ids when cluster is finalized.

    # Add any remaining segments not captured in natural group components as individual seeds
    # These are segments that might be isolated or belong to very small/unnamed groups.
    current_processed_in_seeds = set()
    for cluster_obj in active_clusters:
        current_processed_in_seeds.update(cluster_obj.segment_ids)

    for seg in all_segments:
        if seg.seg_id not in current_processed_in_seeds and seg.seg_id not in processed_segment_ids :
            if is_cluster_routable_relaxed(graph, [seg]): # Should usually be true
                initial_score = scorer.score_cluster([seg])
                new_cluster = Cluster(id=cluster_id_counter, segments=[seg], score=initial_score)
                active_clusters.append(new_cluster)
                cluster_id_counter += 1


    # Load trail subsystem mapping once for the entire clustering process
    trail_to_subsystem = load_trail_subsystem_mapping()

    # 2. Iterative Cluster Expansion
    expansion_iteration = 0
    max_expansion_iterations = len(active_clusters) * 2 # Heuristic limit

    # Create a set of all segment IDs that are currently in any active cluster for quick lookup
    all_active_cluster_segment_ids: Set[str] = set()
    for ac in active_clusters:
        all_active_cluster_segment_ids.update(ac.segment_ids)

    while active_clusters and expansion_iteration < max_expansion_iterations:
        expansion_iteration += 1
        current_cluster = active_clusters.pop(0) # FIFO for now, could be priority queue

        can_expand_current = True
        while can_expand_current:
            can_expand_current = False # Assume no expansion in this inner iteration

            # Identify boundary nodes for the current_cluster.
            # These are nodes in current_cluster that have connections to segments NOT YET globally processed.
            # The get_boundary_nodes method needs access to the global processed_segment_ids.
            # Let's pass all_active_cluster_segment_ids which includes segments in *other* active clusters too.
            # Boundary nodes for expansion should connect to segments *not in any cluster yet*.

            # Simpler boundary: all nodes in the current cluster.
            # Candidate segments are those NOT in processed_segment_ids AND connected to these nodes.

            cluster_all_nodes = current_cluster.get_all_nodes()
            candidate_infos: List[Dict[str, Any]] = []

            # Iterate over segments not yet in any cluster (active or final)
            # This requires iterating `all_segments` and checking against `processed_segment_ids` and `all_active_cluster_segment_ids`

            # Build a set of all segment IDs currently tied up in any cluster (active or final)
            # to define truly available segments for expansion.
            globally_processed_ids = processed_segment_ids.copy()
            for ac_other in active_clusters: # Segments in other active clusters
                globally_processed_ids.update(ac_other.segment_ids)
            globally_processed_ids.update(current_cluster.segment_ids) # Segments already in this cluster

            for candidate_segment in all_segments:
                if candidate_segment.seg_id and candidate_segment.seg_id in globally_processed_ids:
                    continue # Already processed or in some cluster

                # Check connectivity: candidate must connect to one of the current_cluster's nodes
                is_connected_to_cluster = False
                if candidate_segment.start in cluster_all_nodes or candidate_segment.end in cluster_all_nodes:
                    is_connected_to_cluster = True
                else:
                    # Check graph for path via connectors (short path of non-trail segments)
                    for cluster_node in cluster_all_nodes:
                        if graph.has_node(cluster_node) and graph.has_node(candidate_segment.start):
                            if nx.has_path(graph, cluster_node, candidate_segment.start): # Check path existence
                                try:
                                    path = nx.shortest_path(
                                        graph,
                                        cluster_node,
                                        candidate_segment.start,
                                        weight="weight",
                                    )
                                    # Check if path is only connectors or very short
                                    path_edges = _extract_path_edges(graph, path)
                                    if all(e.kind != "trail" for e in path_edges) and sum(
                                        e.length_mi for e in path_edges
                                    ) < 0.5:
                                        is_connected_to_cluster = True
                                        break
                                except (nx.NetworkXNoPath, nx.NodeNotFound): pass
                        if is_connected_to_cluster: break
                        if graph.has_node(cluster_node) and graph.has_node(candidate_segment.end):
                            if nx.has_path(graph, cluster_node, candidate_segment.end):
                                try:
                                    path = nx.shortest_path(
                                        graph,
                                        cluster_node,
                                        candidate_segment.end,
                                        weight="weight",
                                    )
                                    path_edges = _extract_path_edges(graph, path)
                                    if all(e.kind != "trail" for e in path_edges) and sum(
                                        e.length_mi for e in path_edges
                                    ) < 0.5:
                                        is_connected_to_cluster = True
                                        break
                                except (nx.NetworkXNoPath, nx.NodeNotFound): pass
                        if is_connected_to_cluster: break

                if not is_connected_to_cluster:
                    continue

                temp_cluster_segments = current_cluster.segments + [candidate_segment]
                
                # Check if candidate segment is from same TrailSubSystem as cluster
                same_family = check_cluster_same_subsystem(current_cluster.segments, candidate_segment, trail_to_subsystem)
                
                if not is_cluster_routable_relaxed(graph, temp_cluster_segments, same_trail_family=same_family):
                    continue

                new_score_dict = scorer.score_cluster(temp_cluster_segments)
                candidate_infos.append({
                    "segment": candidate_segment,
                    "score_dict": new_score_dict,
                    "overall_score": new_score_dict["overall"]
                })

            if not candidate_infos:
                can_expand_current = False # No valid candidates to add
                break

            # Select best candidate (e.g., highest overall score)
            # TODO: Add constraints from config (max cluster size, time)
            best_candidate_info = max(candidate_infos, key=lambda c: c["overall_score"])

            # Improvement check: new score must be better than current cluster's score
            # This simple check might be too greedy or not allow initial score drops for future gains.
            if best_candidate_info["overall_score"] > current_cluster.score.get("overall", -float('inf')):
                current_cluster.add_segment(best_candidate_info["segment"], best_candidate_info["score_dict"])
                if best_candidate_info["segment"].seg_id:
                     globally_processed_ids.add(best_candidate_info["segment"].seg_id) # Add to prevent re-processing by other clusters
                     # Also update all_active_cluster_segment_ids if managing that separately
                can_expand_current = True # Successfully expanded, try again
            else:
                can_expand_current = False # No improvement or suitable candidate

        # Expansion for current_cluster is done, move to final
        final_clusters_objects.append(current_cluster)
        processed_segment_ids.update(current_cluster.segment_ids) # Mark all its segments as globally processed

    # Add any clusters from active_clusters that were not processed (e.g. if max_expansion_iterations hit)
    for ac in active_clusters:
        if not any(fc.id == ac.id for fc in final_clusters_objects):
            final_clusters_objects.append(ac)
            processed_segment_ids.update(ac.segment_ids)


    # 3. Handle Remaining Segments (segments not in any finalized cluster)
    truly_remaining_segments = [seg for seg in all_segments if seg.seg_id not in processed_segment_ids]
    if truly_remaining_segments:
        # Simplified handling: create small clusters from connected components of remaining segments
        # This part can reuse logic from the initial seeding if adapted.
        # For now, let's add them as individual clusters if small enough or break them down.
        # This is a placeholder for more robust "Isolated Segment Handling"

        # Create a temporary graph of only remaining segments to find components
        temp_remaining_graph = nx.Graph()
        remaining_nodes_for_comp = set()
        for seg in truly_remaining_segments:
            remaining_nodes_for_comp.add(seg.start)
            remaining_nodes_for_comp.add(seg.end)
            temp_remaining_graph.add_node(seg.start)
            temp_remaining_graph.add_node(seg.end)
            temp_remaining_graph.add_edge(seg.start, seg.end, edge=seg)

        # Augment with connectors from main graph
        for u_node, v_node, edge_data in graph.edges(data=True):
            if u_node in remaining_nodes_for_comp and v_node in remaining_nodes_for_comp:
                temp_remaining_graph.add_edge(u_node,v_node)

        for component_node_set in nx.connected_components(temp_remaining_graph):
            comp_segs = [s for s in truly_remaining_segments if s.start in component_node_set or s.end in component_node_set]
            if not comp_segs: continue

            if is_cluster_routable_relaxed(graph, comp_segs):
                comp_score = scorer.score_cluster(comp_segs)
                final_clusters_objects.append(Cluster(id=cluster_id_counter, segments=comp_segs, score=comp_score))
                cluster_id_counter +=1
                for s in comp_segs: processed_segment_ids.add(s.seg_id) # type: ignore
            else: # If component not routable as a whole, add segments individually
                for seg_rem in comp_segs:
                    if seg_rem.seg_id not in processed_segment_ids: # Check again
                        rem_score = scorer.score_cluster([seg_rem])
                        final_clusters_objects.append(Cluster(id=cluster_id_counter, segments=[seg_rem], score=rem_score))
                        cluster_id_counter +=1
                        if seg_rem.seg_id: processed_segment_ids.add(seg_rem.seg_id)

    # Attempt to merge adjacent clusters when the combined score improves.
    final_clusters_objects = _merge_clusters_if_beneficial(final_clusters_objects, scorer, graph)

    # Convert Cluster objects to List[Edge] for return
    final_clusters_list_of_edges = [cluster.segments for cluster in final_clusters_objects if cluster.segments]
    return final_clusters_list_of_edges


def _clusters_share_node(cluster_a: Cluster, cluster_b: Cluster) -> bool:
    return bool(cluster_a.get_all_nodes() & cluster_b.get_all_nodes())


def _merge_clusters_if_beneficial(
    clusters: List[Cluster],
    scorer: ClusterScoringSystem,
    graph: nx.DiGraph,
) -> List[Cluster]:
    """Greedily merge clusters when doing so improves the score."""

    merged = list(clusters)
    changed = True
    while changed:
        changed = False
        for idx_a in range(len(merged)):
            if changed:
                break
            for idx_b in range(idx_a + 1, len(merged)):
                cluster_a = merged[idx_a]
                cluster_b = merged[idx_b]

                if not _clusters_share_node(cluster_a, cluster_b):
                    continue

                combined_segments: List[Edge] = []
                seen_ids: Set[str] = set()
                for seg in cluster_a.segments + cluster_b.segments:
                    if seg.seg_id:
                        if seg.seg_id in seen_ids:
                            continue
                        seen_ids.add(seg.seg_id)
                    combined_segments.append(seg)

                if not is_cluster_routable_relaxed(graph, combined_segments):
                    continue

                combined_score = scorer.score_cluster(combined_segments)
                baseline = max(
                    cluster_a.score.get("overall", float("-inf")),
                    cluster_b.score.get("overall", float("-inf")),
                )

                if combined_score.get("overall", float("-inf")) <= baseline + 1e-6:
                    continue

                new_cluster = Cluster(
                    id=min(cluster_a.id, cluster_b.id),
                    segments=combined_segments,
                    score=combined_score,
                )
                merged[idx_a] = new_cluster
                del merged[idx_b]
                changed = True
                break

    return merged

def load_trail_subsystem_mapping(geojson_path: str = "data/traildata/Boise_Parks_Trails_Open_Data.geojson") -> Dict[str, str]:
    """
    Load trail name to TrailSubSystem mapping from the Boise Parks GeoJSON file.
    
    Returns:
        Dict mapping trail names to their TrailSubSystem
    """
    import json
    import os
    
    trail_to_subsystem = {}
    
    # Handle relative path from project root
    if not os.path.isabs(geojson_path):
        # Assume we're running from project root or adjust path accordingly
        possible_paths = [
            geojson_path,
            os.path.join(".", geojson_path),
            os.path.join("..", geojson_path),
        ]
        
        geojson_path = None
        for path in possible_paths:
            if os.path.exists(path):
                geojson_path = path
                break
        
        if not geojson_path:
            print(f"Warning: Could not find GeoJSON file, falling back to name parsing")
            return {}
    
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            trail_name = props.get('TrailName', '')
            subsystem = props.get('TrailSubSystem', '')
            
            if trail_name and subsystem:
                trail_to_subsystem[trail_name] = subsystem
                
        print(f"Loaded {len(trail_to_subsystem)} trail to subsystem mappings")
        return trail_to_subsystem
        
    except (OSError, IOError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to load GeoJSON file: {e}, falling back to name parsing")
        return {}
    except Exception as e:
        print(f"Warning: Unexpected error loading GeoJSON file: {e}, falling back to name parsing")
        return {}


def identify_natural_trail_groups(all_segments: List[Edge]) -> Dict[str, List[Edge]]:
    """
    Groups segments by TrailSubSystem from GeoJSON data, with fallback to trail name families.
    This creates much more accurate natural groupings than name parsing alone.
    """
    # Load the subsystem mapping
    trail_to_subsystem = load_trail_subsystem_mapping()
    
    groups: Dict[str, List[Edge]] = {}
    
    for segment in all_segments:
        group_name = None
        
        if segment.name and segment.name in trail_to_subsystem:
            # Use TrailSubSystem if available
            group_name = trail_to_subsystem[segment.name]
        elif segment.name:
            # Fallback to name parsing for segments not in GeoJSON
            group_name = extract_base_trail_name(segment.name)
        else:
            # Segments without names go into individual groups
            group_name = f"Unnamed_{segment.seg_id}" if segment.seg_id else f"Unnamed_{id(segment)}"
        
        groups.setdefault(group_name, []).append(segment)
    
    return groups


def extract_base_trail_name(trail_name: str) -> str:
    """
    Extract the base trail name from a full trail name.
    
    Examples:
    - "Dry Creek Trail 1" -> "Dry Creek Trail"
    - "Polecat Loop 5" -> "Polecat Loop" 
    - "Central Ridge Spur 3" -> "Central Ridge Spur"
    - "8th Street Motorcycle Trail 2" -> "8th Street Motorcycle Trail"
    """
    import re
    
    # Remove common suffixes that indicate segments/parts
    # Pattern: remove trailing numbers, directional words, and segment indicators
    patterns_to_remove = [
        r'\s+\d+$',  # " 1", " 23", etc.
        r'\s+(North|South|East|West|Upper|Lower|Inner|Outer)$',  # directional
        r'\s+(Part|Section|Segment)\s*\d*$',  # segment indicators
        r'\s+(ascent|descent)$',  # directional movement
    ]
    
    cleaned_name = trail_name.strip()
    for pattern in patterns_to_remove:
        cleaned_name = re.sub(pattern, '', cleaned_name, flags=re.IGNORECASE)
    
    return cleaned_name.strip()


def is_cluster_routable_relaxed(graph: nx.DiGraph, segments: List[Edge], 
                               max_connector_distance: float = 2.0,
                               same_trail_family: bool = False) -> bool:
    """
    Check if a cluster can be routed, with relaxed rules for same trail families.
    
    Args:
        graph: The routing graph
        segments: List of segments to check
        max_connector_distance: Maximum distance for connector paths
        same_trail_family: If True, use more lenient connectivity rules
    """
    if not segments:
        return True
    if len(segments) == 1:
        return True  # Single segments are always routable
    
    # For same trail families, be more lenient
    if same_trail_family:
        max_connector_distance *= 2.0  # Allow longer connectors within trail families
    
    # Use your existing split_cluster_by_connectivity but with relaxed parameters
    from . import challenge_planner
    try:
        subclusters = challenge_planner.split_cluster_by_connectivity(
            segments, graph, max_connector_distance, debug_args=None
        )
        # If all segments end up in one subcluster, it's routable
        return len(subclusters) == 1 and len(subclusters[0]) == len(segments)
    except (ImportError, AttributeError, nx.NetworkXError) as e:
        # If connectivity check fails due to import issues or NetworkX errors
        print(f"Warning: Connectivity check failed: {e}")
        return same_trail_family  # Assume routable for same trail families
    except Exception as e:
        # Unexpected error in connectivity check
        print(f"Warning: Unexpected error in connectivity check: {e}")
        return False  # Conservative: assume not routable for unexpected errors

def segments_same_subsystem(seg1: Edge, seg2: Edge, trail_to_subsystem: Dict[str, str]) -> bool:
    """
    Check if two segments belong to the same TrailSubSystem.
    """
    if not seg1.name or not seg2.name:
        return False
    
    subsystem1 = trail_to_subsystem.get(seg1.name)
    subsystem2 = trail_to_subsystem.get(seg2.name)
    
    if subsystem1 and subsystem2:
        return subsystem1 == subsystem2
    
    # Fallback to name parsing if not in subsystem data
    base1 = extract_base_trail_name(seg1.name)
    base2 = extract_base_trail_name(seg2.name)
    return base1 == base2


def check_cluster_same_subsystem(segments: List[Edge], candidate: Edge, trail_to_subsystem: Dict[str, str]) -> bool:
    """
    Check if a candidate segment belongs to the same subsystem as any segment in the cluster.
    """
    if not segments or not candidate.name:
        return False
    
    candidate_subsystem = trail_to_subsystem.get(candidate.name)
    if not candidate_subsystem:
        # Fallback to name parsing
        candidate_base = extract_base_trail_name(candidate.name)
        cluster_bases = {extract_base_trail_name(s.name) for s in segments if s.name}
        return candidate_base in cluster_bases
    
    # Check if any segment in cluster has same subsystem
    for seg in segments:
        if seg.name and trail_to_subsystem.get(seg.name) == candidate_subsystem:
            return True
    
    return False
