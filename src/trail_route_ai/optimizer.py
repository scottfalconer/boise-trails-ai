from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Set
import logging

from . import planner_utils
from . import challenge_planner

Edge = planner_utils.Edge

logger = logging.getLogger(__name__)

@dataclass
class RouteMetrics:
    total_time: float
    redundancy_ratio: float
    new_segment_time: float
    elevation_gain: float
    connectivity: float


def calculate_route_metrics(
    ctx: planner_utils.PlanningContext,
    route: Iterable[Edge],
    required_ids: Set[str],
    max_foot_road: float,
) -> RouteMetrics:
    """Compute metrics for ``route`` using planning context values."""

    edges = list(route)
    total_time = challenge_planner.total_time(
        edges, ctx.pace, ctx.grade, ctx.road_pace
    )
    efficiency = planner_utils.calculate_route_efficiency_score(edges)
    redundancy_ratio = 1.0 - efficiency

    visited: Set[str] = set()
    new_segment_time = 0.0
    for e in edges:
        sid = str(e.seg_id) if e.seg_id is not None else None
        if sid is not None and sid in required_ids and sid not in visited:
            new_segment_time += planner_utils.estimate_time(
                e, ctx.pace, ctx.grade, ctx.road_pace
            )
            visited.add(sid)

    elevation_gain = sum(e.elev_gain_ft for e in edges)

    connectivity_subs = challenge_planner.split_cluster_by_connectivity(
        [e for e in edges if e.kind != "road"], ctx.graph, max_foot_road
    )
    connectivity = 1.0 / len(connectivity_subs) if connectivity_subs else 0.0

    return RouteMetrics(
        total_time=total_time,
        redundancy_ratio=redundancy_ratio,
        new_segment_time=new_segment_time,
        elevation_gain=elevation_gain,
        connectivity=connectivity,
    )


def is_pareto_improvement(a: RouteMetrics, b: RouteMetrics) -> bool:
    """Return True if ``b`` is not worse than ``a`` on all metrics and
    strictly better on at least one."""

    not_worse = (
        b.total_time <= a.total_time
        and b.redundancy_ratio <= a.redundancy_ratio
        and b.new_segment_time >= a.new_segment_time
        and b.connectivity >= a.connectivity
    )
    strictly_better = (
        b.total_time < a.total_time
        or b.redundancy_ratio < a.redundancy_ratio
        or b.new_segment_time > a.new_segment_time
        or b.connectivity > a.connectivity
    )
    return not_worse and strictly_better


def generate_intelligent_swap_candidates(order: List[Edge], max_candidates: int = 50) -> List[Tuple[int, int]]:
    """Return promising index pairs for 2-opt swaps."""

    n = len(order)
    scores: List[Tuple[float, int, int]] = []
    prefix = [0.0]
    for e in order:
        prefix.append(prefix[-1] + e.length_mi)
    for i in range(n - 1):
        for j in range(i + 2, n + 1):
            sub_len = prefix[j] - prefix[i]
            scores.append((sub_len, i, j))
    scores.sort(reverse=True)
    return [(i, j) for _, i, j in scores[:max_candidates]]


def build_route_from_order(
    ctx: planner_utils.PlanningContext,
    sequence: List[Edge],
    start: Tuple[float, float],
    max_foot_road: float,
    road_threshold: float,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    strict_max_foot_road: bool = False,
) -> List[Edge]:
    """Plan a route following ``sequence`` using the core planner."""

    return challenge_planner._plan_route_for_sequence(
        ctx.graph,
        sequence,
        start,
        ctx.pace,
        ctx.grade,
        ctx.road_pace,
        max_foot_road,
        road_threshold,
        ctx.dist_cache,
        spur_length_thresh=spur_length_thresh,
        spur_road_bonus=spur_road_bonus,
        strict_max_foot_road=strict_max_foot_road,
    )


def advanced_2opt_optimization(
    ctx: planner_utils.PlanningContext,
    order: List[Edge],
    start: Tuple[float, float],
    required_ids: Set[str],
    max_foot_road: float,
    road_threshold: float,
    *,
    strict_max_foot_road: bool = False,
) -> Tuple[List[Edge], List[Edge]]:
    """Perform a multi-objective 2-opt optimization on ``order``."""

    best_order = order[:]
    best_route = build_route_from_order(
        ctx,
        best_order,
        start,
        max_foot_road,
        road_threshold,
        strict_max_foot_road=strict_max_foot_road,
    )
    if not best_route:
        return [], []
    best_metrics = calculate_route_metrics(ctx, best_route, required_ids, max_foot_road)

    improved = True
    max_iterations = max(20, len(best_order) * 5)
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        for i, j in generate_intelligent_swap_candidates(best_order):
            if j - i < 2:
                continue
            new_order = best_order[:i] + best_order[i:j][::-1] + best_order[j:]
            cand_route = build_route_from_order(
                ctx,
                new_order,
                start,
                max_foot_road,
                road_threshold,
                strict_max_foot_road=strict_max_foot_road,
            )
            if not cand_route:
                continue
            cand_metrics = calculate_route_metrics(
                ctx, cand_route, required_ids, max_foot_road
            )
            if is_pareto_improvement(best_metrics, cand_metrics):
                best_metrics = cand_metrics
                best_order = new_order
                best_route = cand_route
                improved = True
                break
        if improved:
            iteration += 1
            continue
    if iteration >= max_iterations:
        logger.warning("2-opt optimization reached iteration limit %d", max_iterations)
    return best_route, best_order
