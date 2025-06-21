"""Utility exports for the trail route planner."""

from .planner_utils import (
    Edge,
    PlanningContext,
    calculate_route_efficiency_score,
    calculate_route_elevation_efficiency_score,
    calculate_overall_efficiency_score,
    optimize_route_for_redundancy,
)
from .optimizer import (
    RouteMetrics,
    is_pareto_improvement,
)
from .challenge_planner import advanced_2opt_optimization
from . import cache_utils

__all__ = [
    "Edge",
    "PlanningContext",
    "calculate_route_efficiency_score",
    "calculate_route_elevation_efficiency_score",
    "calculate_overall_efficiency_score",
    "optimize_route_for_redundancy",
    "RouteMetrics",
    "is_pareto_improvement",
    "advanced_2opt_optimization",
    "cache_utils",
]
