import multiprocessing as mp
import networkx as nx
import pytest

from trail_route_ai.challenge_planner import worker_init_apsp, compute_dijkstra_for_node


def build_graph():
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    G.add_edge(1, 3, weight=5.0)
    return G


def nx_expected(G, source):
    preds, dists = nx.dijkstra_predecessor_and_distance(G, source, weight="weight")
    dist_map = {n: float(d) for n, d in dists.items() if n != source}
    pred_map = {n: preds[n][0] for n in preds if n != source and preds[n]}
    return dist_map, pred_map


def test_compute_dijkstra_pool():
    G = build_graph()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    lock = ctx.Lock()
    with ctx.Pool(processes=2, initializer=worker_init_apsp, initargs=(G, q, lock)) as pool:
        res = dict(pool.map(compute_dijkstra_for_node, [1, 2]))
    q.close()
    q.join_thread()

    for node in [1, 2]:
        dist_map, pred_map = res[node]
        exp_dist, exp_pred = nx_expected(G, node)
        assert dist_map == exp_dist
        assert pred_map == exp_pred


def failing_worker(node):
    raise RuntimeError("boom")


def test_worker_exception_propagates():
    G = build_graph()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    lock = ctx.Lock()
    with ctx.Pool(processes=1, initializer=worker_init_apsp, initargs=(G, q, lock)) as pool:
        with pytest.raises(RuntimeError):
            list(pool.imap_unordered(failing_worker, [1]))
    q.close()
    q.join_thread()

