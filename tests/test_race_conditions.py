import multiprocessing as mp
from trail_route_ai import challenge_planner, planner_utils


def build_data():
    edge1 = planner_utils.Edge(
        "A",
        "A",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    edge2 = planner_utils.Edge(
        "B",
        "B",
        (1.0, 0.0),
        (2.0, 0.0),
        1.0,
        0.0,
        [(1.0, 0.0), (2.0, 0.0)],
        "trail",
        "both",
    )
    G = challenge_planner.build_nx_graph([edge1, edge2], pace=10.0, grade=0.0, road_pace=10.0)
    return G, [edge1, edge2]


def plan_worker(_):
    G, edges = build_data()
    return challenge_planner.plan_route(
        G,
        edges,
        edges[0].start,
        pace=10.0,
        grade=0.0,
        road_pace=10.0,
        max_foot_road=0.0,
        road_threshold=0.1,
        use_rpp=False,
    )


def test_parallel_planning_race():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        results = pool.map(plan_worker, range(4))
    lengths = {len(r) for r in results}
    assert len(lengths) == 1
