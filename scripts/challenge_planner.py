import argparse
import csv
import os
import sys
import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
import networkx as nx

# Allow running this file directly without installing the package
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts import planner_utils

# Type aliases
Edge = planner_utils.Edge

def midpoint(edge: Edge) -> Tuple[float, float]:
    sx, sy = edge.start
    ex, ey = edge.end
    return ((sx + ex) / 2.0, (sy + ey) / 2.0)

def total_time(edges: List[Edge], pace: float, grade: float) -> float:
    return sum(planner_utils.estimate_time(e, pace, grade) for e in edges)

def build_nx_graph(edges: List[Edge]) -> nx.Graph:
    G = nx.Graph()
    for e in edges:
        length = e.length_mi
        G.add_edge(e.start, e.end, weight=length, edge=e)
    return G


def nearest_node(nodes: List[Tuple[float, float]], point: Tuple[float, float]):
    return min(nodes, key=lambda n: (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2)


def edges_from_path(G: nx.Graph, path: List[Tuple[float, float]]) -> List[Edge]:
    out = []
    for a, b in zip(path[:-1], path[1:]):
        data = G.get_edge_data(a, b)
        if data:
            out.append(data[0]["edge"] if 0 in data else data["edge"])
    return out


def plan_route(G: nx.Graph, edges: List[Edge], start: Tuple[float, float]) -> List[Edge]:
    remaining = edges[:]
    route: List[Edge] = []
    cur = start
    while remaining:
        best = None
        best_path = None
        best_dist = None
        for e in remaining:
            for end in [e.start, e.end]:
                try:
                    path = nx.shortest_path(G, cur, end, weight="weight")
                    dist = nx.path_weight(G, path, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = (e, end)
                    best_path = path
        if best is None:
            # cannot connect; just append and move on
            e = remaining.pop(0)
            route.append(e)
            cur = e.end
            continue
        e, end = best
        route.extend(edges_from_path(G, best_path))
        if end == e.start:
            route.append(e)
            cur = e.end
        else:
            # reverse orientation
            rev = Edge(e.seg_id, e.name, e.end, e.start, e.length_mi, e.elev_gain_ft, list(reversed(e.coords)))
            route.append(rev)
            cur = rev.end
        remaining.remove(e)
    try:
        path_back = nx.shortest_path(G, cur, start, weight="weight")
        route.extend(edges_from_path(G, path_back))
    except nx.NetworkXNoPath:
        pass
    return route


def parse_remaining(value: str) -> List[str]:
    if os.path.exists(value):
        with open(value) as f:
            text = f.read()
    else:
        text = value
    items = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]
    return items


def cluster_segments(
    edges: List[Edge],
    pace: float,
    grade: float,
    budget: float,
    max_clusters: int,
) -> List[List[Edge]]:
    if not edges:
        return []
    pts = np.array([midpoint(e) for e in edges])
    k = min(max_clusters, len(edges))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(pts)
    initial: Dict[int, List[Edge]] = defaultdict(list)
    for lbl, e in zip(labels, edges):
        initial[lbl].append(e)
    clusters: List[List[Edge]] = []
    for group in initial.values():
        group = sorted(group, key=lambda e: planner_utils.estimate_time(e, pace, grade), reverse=True)
        cur: List[Edge] = []
        t = 0.0
        for e in group:
            et = planner_utils.estimate_time(e, pace, grade)
            if t + et > budget and cur:
                clusters.append(cur)
                cur = [e]
                t = et
            else:
                cur.append(e)
                t += et
        if cur:
            clusters.append(cur)
    while len(clusters) < max_clusters:
        clusters.append([])
    while len(clusters) > max_clusters:
        clusters.sort(key=lambda c: total_time(c, pace, grade))
        small = clusters.pop(0)
        merged = False
        for i, other in enumerate(clusters):
            if total_time(other, pace, grade) + total_time(small, pace, grade) <= budget:
                clusters[i] = other + small
                merged = True
                break
        if not merged:
            clusters.append(small)
            break
    return clusters[:max_clusters]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Challenge route planner")
    parser.add_argument("--start-date", required=True, help="Challenge start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Challenge end date YYYY-MM-DD")
    parser.add_argument("--time", required=True, help="Daily time budget")
    parser.add_argument("--pace", required=True, type=float, help="Base running pace (min/mi)")
    parser.add_argument("--grade", type=float, default=0.0, help="Seconds per 100ft climb")
    parser.add_argument("--segments", default="data/traildata/trail.json")
    parser.add_argument("--perf", default="data/segment_perf.csv")
    parser.add_argument("--year", type=int)
    parser.add_argument("--remaining")
    parser.add_argument("--output", default="challenge_plan.csv")
    parser.add_argument("--gpx-dir", default="gpx")
    args = parser.parse_args(argv)

    start_date = datetime.date.fromisoformat(args.start_date)
    end_date = datetime.date.fromisoformat(args.end_date)
    if end_date < start_date:
        parser.error("--end-date must not be before --start-date")
    num_days = (end_date - start_date).days + 1

    budget = planner_utils.parse_time_budget(args.time)
    edges = planner_utils.load_segments(args.segments)
    completed = planner_utils.load_completed(args.perf, args.year or 0)

    remain_ids = None
    if args.remaining:
        remain_ids = set(parse_remaining(args.remaining))
    if remain_ids is None:
        remain_ids = {str(e.seg_id) for e in edges} - set(completed)
    remaining_edges = [e for e in edges if str(e.seg_id) in remain_ids]

    G = build_nx_graph(edges)
    nodes = list({e.start for e in edges} | {e.end for e in edges})

    clusters = cluster_segments(
        remaining_edges, args.pace, args.grade, budget, num_days
    )

    os.makedirs(args.gpx_dir, exist_ok=True)
    summary_rows = []
    for idx, cluster in enumerate(clusters):
        if not cluster:
            continue
        cur_date = start_date + datetime.timedelta(days=idx)
        centroid = (
            sum(midpoint(e)[0] for e in cluster) / len(cluster),
            sum(midpoint(e)[1] for e in cluster) / len(cluster),
        )
        start = nearest_node(nodes, centroid)
        route = plan_route(G, cluster, start)
        dist = sum(e.length_mi for e in route)
        gain = sum(e.elev_gain_ft for e in route)
        est_time = total_time(route, args.pace, args.grade)
        gpx_path = os.path.join(
            args.gpx_dir, f"{cur_date.strftime('%Y%m%d')}.gpx"
        )
        planner_utils.write_gpx(gpx_path, route)
        summary_rows.append({
            "date": cur_date.isoformat(),
            "segments": " ".join(str(e.seg_id) for e in cluster),
            "plan": " > ".join(e.name or str(e.seg_id) for e in route),
            "distance_mi": round(dist, 2),
            "elev_gain_ft": round(gain, 0),
            "time_min": round(est_time, 1),
        })
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
