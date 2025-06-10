import argparse
import json
import os
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Set


Edge = namedtuple('Edge', ['seg_id', 'name', 'start', 'end', 'length_mi', 'elev_gain_ft'])


def load_segments(path: str) -> List[Edge]:
    with open(path) as f:
        data = json.load(f)
    if 'trailSegments' in data:
        seg_list = data['trailSegments']
    elif 'segments' in data:
        seg_list = data['segments']
    elif 'features' in data:
        seg_list = [f.get('properties', {}) | {'geometry': f['geometry']} for f in data['features']]
    else:
        raise ValueError('Unrecognized segment JSON structure')
    edges = []
    for seg in seg_list:
        props = seg.get('properties', seg)
        coords = seg['geometry']['coordinates'] if 'geometry' in seg else seg['coordinates']
        start = tuple(round(c, 6) for c in coords[0])
        end = tuple(round(c, 6) for c in coords[-1])
        length_ft = float(props.get('LengthFt', 0))
        elev_gain = float(props.get('elevGainFt', 0) or props.get('ElevGainFt', 0) or 0)
        seg_id = props.get('segId') or props.get('id') or props.get('seg_id')
        name = props.get('segName') or props.get('name') or ''
        length_mi = length_ft / 5280.0
        edge = Edge(seg_id, name, start, end, length_mi, elev_gain)
        edges.append(edge)
    return edges


def build_graph(edges: List[Edge]):
    graph: Dict[Tuple[float, float], List[Tuple[Edge, Tuple[float, float]]]] = defaultdict(list)
    for e in edges:
        graph[e.start].append((e, e.end))
        graph[e.end].append((e, e.start))
    return graph


def estimate_time(edge: Edge, pace_min_per_mi: float, grade_factor_sec_per_100ft: float) -> float:
    base = edge.length_mi * pace_min_per_mi
    penalty = (edge.elev_gain_ft / 100.0) * (grade_factor_sec_per_100ft / 60.0)
    return base + penalty


def load_completed(csv_path: str, year: int) -> Set:
    if not os.path.exists(csv_path):
        return set()
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df.year == year]
    return set(df.seg_id.astype(str).unique())


def search_loops(
    graph,
    start,
    pace,
    grade,
    time_budget,
    completed,
    max_segments=5,
):
    """Search for a loop with the most new segments within the time budget.

    The search explores all combinations of up to ``max_segments`` edges using a
    depth-first strategy.  A segment ID may only appear once in a candidate path
    unless it is reused solely to return to the starting node, ensuring loops do
    not traverse the same segment multiple times.
    """

    best = None
    visited: Set[Tuple[str, Tuple[float, float], Tuple[float, float]]] = set()

    def dfs(node, time_so_far, path, used_ids):
        """Recursive search of all feasible paths."""
        nonlocal best

        if node == start and path:
            new_count = len({e.seg_id for e in path if e.seg_id not in completed})
            if best is None or new_count > best['new_count'] or (
                new_count == best['new_count'] and time_so_far < best['time']
            ):
                best = {
                    'path': list(path),
                    'time': time_so_far,
                    'new_count': new_count,
                }
            # continue exploring for possibly better loops

        if len(path) >= max_segments:
            return

        for edge, nxt in graph[node]:
            key = (edge.seg_id, node, nxt)
            if key in visited:
                continue

            # disallow using a segment more than once except to close the loop
            if edge.seg_id in used_ids and nxt != start:
                continue

            seg_time = estimate_time(edge, pace, grade)
            if time_so_far + seg_time > time_budget:
                continue

            visited.add(key)
            path.append(edge)
            added = False
            if edge.seg_id not in used_ids:
                used_ids.add(edge.seg_id)
                added = True

            dfs(nxt, time_so_far + seg_time, path, used_ids)

            if added:
                used_ids.remove(edge.seg_id)
            path.pop()
            visited.remove(key)

    dfs(start, 0.0, [], set())
    return best


def choose_trailhead(
    graph,
    trailheads: Dict[str, Tuple[float, float]],
    pace: float,
    grade: float,
    time_budget: float,
    completed: Set[str],
    max_segments: int = 5,
    choice: str | None = None,
):
    """Evaluate all trailheads and return the best loop.

    Parameters mirror :func:`search_loops`. ``trailheads`` maps a trailhead name
    to the starting node coordinates.  If ``choice`` is provided only that
    trailhead is searched.
    """

    if choice:
        start = trailheads[choice]
        return choice, search_loops(
            graph,
            start,
            pace,
            grade,
            time_budget,
            completed,
            max_segments=max_segments,
        )

    best_name = None
    best_result = None
    for name, start in trailheads.items():
        result = search_loops(
            graph,
            start,
            pace,
            grade,
            time_budget,
            completed,
            max_segments=max_segments,
        )
        if not result:
            continue
        if (
            best_result is None
            or result["new_count"] > best_result["new_count"]
            or (
                result["new_count"] == best_result["new_count"]
                and result["time"] < best_result["time"]
            )
        ):
            best_name = name
            best_result = result
    return best_name, best_result


def write_gpx(path: List[Edge], out_path: str) -> None:
    """Write a GPX file for the provided ``path``.

    Each edge contributes its end coordinate in sequence.  The first start point
    is written once followed by the end of every edge.  Coordinates are treated
    as ``(lon, lat)`` pairs.
    """

    import gpxpy

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    seg = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(seg)

    if not path:
        with open(out_path, "w") as fh:
            fh.write(gpx.to_xml())
        return

    seg.points.append(
        gpxpy.gpx.GPXTrackPoint(path[0].start[1], path[0].start[0])
    )
    for edge in path:
        seg.points.append(
            gpxpy.gpx.GPXTrackPoint(edge.end[1], edge.end[0])
        )

    with open(out_path, "w") as fh:
        fh.write(gpx.to_xml())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Daily route planner")
    parser.add_argument('--time', type=float, required=True, help='Time budget in minutes')
    parser.add_argument('--pace', type=float, required=True, help='Base pace in minutes per mile')
    parser.add_argument('--grade', type=float, default=0.0, help='Additional seconds per 100ft of climb per mile')
    parser.add_argument('--segments', default='data/traildata/trail.json')
    parser.add_argument('--perf', default='data/segment_perf.csv')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--start-seg', type=str, help='Segment ID to start from')
    parser.add_argument('--trailhead', type=str, help='Trailhead name to start from')
    parser.add_argument('--max-segments', type=int, default=5,
                        help='Maximum number of segments to explore')
    args = parser.parse_args(argv)

    edges = load_segments(args.segments)
    graph = build_graph(edges)

    completed = load_completed(args.perf, args.year)

    trailheads: Dict[str, Tuple[float, float]] = {}
    for e in edges:
        if e.start not in trailheads.values():
            trailheads[f"th{len(trailheads)}"] = e.start

    if args.start_seg:
        for e in edges:
            if str(e.seg_id) == args.start_seg:
                trailheads = {"selected": e.start}
                break

    name, result = choose_trailhead(
        graph,
        trailheads,
        args.pace,
        args.grade,
        args.time,
        completed,
        max_segments=args.max_segments,
        choice=args.trailhead,
    )
    if not result:
        print('No loop found within time budget')
        return
    total_distance = sum(e.length_mi for e in result['path'])
    print('Selected segments:')
    for e in result['path']:
        print(f"  {e.seg_id} - {e.name}")
    print(f"Total new segments: {result['new_count']}")
    print(f"Total distance: {total_distance:.2f} mi")
    print(f"Estimated time: {result['time']:.1f} min")


if __name__ == '__main__':
    main()
