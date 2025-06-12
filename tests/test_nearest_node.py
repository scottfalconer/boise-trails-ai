import pytest

from trail_route_ai import challenge_planner


def test_nearest_node_kdtree_matches_linear(monkeypatch):
    pytest.importorskip("scipy")
    nodes = [(float(i), float(i % 2)) for i in range(100)]
    pt = (42.3, 0.4)
    tree = challenge_planner.build_kdtree(nodes)
    kd_res = challenge_planner.nearest_node(tree, pt)
    monkeypatch.setattr(challenge_planner, "_HAVE_SCIPY", False)
    lin_res = challenge_planner.nearest_node(nodes, pt)
    assert kd_res == lin_res
