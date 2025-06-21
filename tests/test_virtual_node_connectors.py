import pytest
from trail_route_ai import challenge_planner, planner_utils


def test_missing_official_nodes_added(monkeypatch):
    # Avoid requiring scipy for this test
    monkeypatch.setattr(challenge_planner, "_HAVE_SCIPY", False)
    edge = planner_utils.Edge(
        "S1",
        "S1",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    extra_node = (2.0, 0.0)
    official = {edge.start, edge.end, extra_node}
    G = challenge_planner.build_nx_graph(
        [edge],
        pace=10.0,
        grade=0.0,
        road_pace=10.0,
        snap_radius_m=25.0,
        official_nodes=official,
    )
    assert extra_node in G.nodes
    outgoing = list(G.edges(extra_node, data=True))
    assert outgoing, "virtual connector not created"
    assert any(d["edge"].kind == "virtual" and pytest.approx(d["weight"], 0.1) for _, _, d in outgoing)

