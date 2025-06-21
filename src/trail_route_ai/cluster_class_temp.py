"""
Temporary location for Cluster class, will be merged into clustering.py
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set
from .planner_utils import Edge

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
        if not any(s.seg_id == segment.seg_id for s in self.segments if s.seg_id and segment.seg_id):
            self.segments.append(segment)
        self.score = new_score

    @property
    def segment_ids(self) -> Set[str]:
        return {s.seg_id for s in self.segments if s.seg_id}

    @property
    def boundary_nodes(self) -> Set[tuple[float,float]]:
        """
        Identifies nodes that are on the "edge" of the cluster,
        meaning they are part of a cluster segment but also have connections
        to segments not currently in this cluster (needs graph access).
        Or, simpler: all unique nodes in the cluster's segments.
        """
        nodes: Set[tuple[float,float]] = set()
        for seg in self.segments:
            nodes.add(seg.start)
            nodes.add(seg.end)
        return nodes
