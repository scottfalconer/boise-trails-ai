import json
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import networkx as nx


@dataclass
class Edge:
    seg_id: Optional[str]
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_mi: float
    elev_gain_ft: float
    coords: List[Tuple[float, float]]
    kind: str = field(default="trail")  # 'trail' or 'road'


def load_segments(path: str) -> List[Edge]:
    with open(path) as f:
        data = json.load(f)
    if "trailSegments" in data:
        seg_list = data["trailSegments"]
    elif "segments" in data:
        seg_list = data["segments"]
    elif "features" in data:
        seg_list = [
            f.get("properties", {}) | {"geometry": f["geometry"]}
            for f in data["features"]
        ]
    else:
        raise ValueError("Unrecognized segment JSON structure")
    edges: List[Edge] = []
    for seg in seg_list:
        props = seg.get("properties", seg)
        coords = (
            seg["geometry"]["coordinates"]
            if "geometry" in seg
            else seg["coordinates"]
        )
        start = tuple(round(c, 6) for c in coords[0])
        end = tuple(round(c, 6) for c in coords[-1])
        length_ft = float(props.get("LengthFt", 0))
        elev_gain = float(
            props.get("elevGainFt", 0) or props.get("ElevGainFt", 0) or 0
        )
        seg_id = props.get("segId") or props.get("id") or props.get("seg_id")
        name = props.get("segName") or props.get("name") or ""
        length_mi = length_ft / 5280.0
        coords_list = [tuple(pt) for pt in coords]
        edge = Edge(seg_id, name, start, end, length_mi, elev_gain, coords_list)
        edges.append(edge)
    return edges


def load_roads(path: str) -> List[Edge]:
    """Load road segments from a GeoJSON file."""
    with open(path) as f:
        data = json.load(f)
    if "features" not in data:
        raise ValueError("Road GeoJSON must contain features")
    edges: List[Edge] = []
    idx = 0
    for feat in data["features"]:
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in ("LineString", "MultiLineString"):
            continue
        coords_list = (
            [geom["coordinates"]]
            if geom["type"] == "LineString"
            else geom["coordinates"]
        )
        for coords in coords_list:
            start = tuple(round(c, 6) for c in coords[0])
            end = tuple(round(c, 6) for c in coords[-1])
            length = 0.0
            for a, b in zip(coords[:-1], coords[1:]):
                length += _haversine_mi(a, b)
            edges.append(
                Edge(
                    seg_id=f"road_{idx}",
                    name=feat.get("properties", {}).get("name", ""),
                    start=start,
                    end=end,
                    length_mi=length,
                    elev_gain_ft=0.0,
                    coords=[tuple(pt) for pt in coords],
                    kind="road",
                )
            )
            idx += 1
    return edges


def add_elevation_from_dem(edges: List[Edge], dem_path: str) -> None:
    """Populate ``elev_gain_ft`` for each edge using a DEM GeoTIFF."""
    import numpy as np
    import rasterio

    if not os.path.exists(dem_path):
        raise FileNotFoundError(dem_path)

    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        for e in edges:
            if not e.coords:
                e.elev_gain_ft = 0.0
                continue
            samples = list(src.sample([(lon, lat) for lon, lat in e.coords]))
            elevs = [s[0] if s[0] != nodata else np.nan for s in samples]
            gain = 0.0
            prev = elevs[0]
            for val in elevs[1:]:
                if np.isnan(prev):
                    prev = val
                    continue
                if np.isnan(val):
                    continue
                if val > prev:
                    gain += val - prev
                prev = val
            e.elev_gain_ft = float(gain * 3.28084)


def build_graph(edges: List[Edge]):
    graph: Dict[
        Tuple[float, float], List[Tuple[Edge, Tuple[float, float]]]
    ] = defaultdict(list)
    for e in edges:
        graph[e.start].append((e, e.end))
        graph[e.end].append((e, e.start))
    return graph


def estimate_time(
    edge: Edge,
    pace_min_per_mi: float,
    grade_factor_sec_per_100ft: float,
    road_pace_min_per_mi: Optional[float] = None,
) -> float:
    pace = (
        road_pace_min_per_mi if edge.kind == "road" and road_pace_min_per_mi else pace_min_per_mi
    )
    base = edge.length_mi * pace
    penalty = 0.0
    if edge.kind != "road":
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
            new_count = len(
                {e.seg_id for e in path if e.seg_id not in completed}
            )
            if (
                best is None
                or new_count > best["new_count"]
                or (
                    new_count == best["new_count"]
                    and time_so_far < best["time"]
                )
            ):
                best = {
                    "path": list(path),
                    "time": time_so_far,
                    "new_count": new_count,
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


def write_gpx(path: str, edges: List[Edge], mark_road_transitions: bool = False):
    """Write a GPX file for the given sequence of edges.

    If ``mark_road_transitions`` is True, contiguous sections of ``edges`` are
    split into multiple track segments labelled with their ``kind`` (``trail`` or
    ``road``).  Waypoints are also inserted at the start and end of each road
    section so that GPX viewers which do not recognise extensions can still
    highlight road connectors.
    """
    import gpxpy.gpx
    import xml.etree.ElementTree as ET

    if not edges:
        return

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)

    if not mark_road_transitions:
        coords: List[Tuple[float, float]] = []
        for i, e in enumerate(edges):
            seg_coords = [tuple(pt) for pt in e.coords]
            if i == 0:
                coords.extend(seg_coords)
            else:
                last = coords[-1]
                if _close(last, seg_coords[0]):
                    coords.extend(seg_coords[1:])
                elif _close(last, seg_coords[-1]):
                    coords.extend(list(reversed(seg_coords[:-1])))
                else:
                    coords.extend(seg_coords)

        segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(segment)
        for lon, lat in coords:
            segment.points.append(
                gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
            )
    else:
        groups: List[Tuple[str, List[Edge]]] = []
        cur_kind = edges[0].kind
        cur_group: List[Edge] = []
        for e in edges:
            if e.kind != cur_kind:
                groups.append((cur_kind, cur_group))
                cur_group = [e]
                cur_kind = e.kind
            else:
                cur_group.append(e)
        groups.append((cur_kind, cur_group))

        for kind, group_edges in groups:
            coords: List[Tuple[float, float]] = []
            for i, e in enumerate(group_edges):
                seg_coords = [tuple(pt) for pt in e.coords]
                if i == 0:
                    coords.extend(seg_coords)
                else:
                    last = coords[-1]
                    if _close(last, seg_coords[0]):
                        coords.extend(seg_coords[1:])
                    elif _close(last, seg_coords[-1]):
                        coords.extend(list(reversed(seg_coords[:-1])))
                    else:
                        coords.extend(seg_coords)

            segment = gpxpy.gpx.GPXTrackSegment()
            ext = ET.Element("kind")
            ext.text = kind
            segment.extensions.append(ext)
            for lon, lat in coords:
                segment.points.append(
                    gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
                )
            track.segments.append(segment)

            if kind == "road" and coords:
                start_lon, start_lat = coords[0]
                end_lon, end_lat = coords[-1]
                gpx.waypoints.append(
                    gpxpy.gpx.GPXWaypoint(
                        latitude=start_lat,
                        longitude=start_lon,
                        name="Road start",
                        comment="Start of road connector",
                    )
                )
                gpx.waypoints.append(
                    gpxpy.gpx.GPXWaypoint(
                        latitude=end_lat,
                        longitude=end_lon,
                        name="Road end",
                        comment="End of road connector",
                    )
                )

    with open(path, "w") as f:
        f.write(gpx.to_xml())


def _close(
    a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-6
) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _haversine_mi(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return distance in miles between two lon/lat points."""
    import math

    lon1, lat1 = a
    lon2, lat2 = b
    r = 3958.8  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def parse_time_budget(value: str) -> float:
    """Parse a time specification and return minutes.

    Accepts plain minutes ("90"), hours with ``h`` suffix ("1.5h"), or
    ``H:MM`` notation ("1:30").
    """

    text = value.strip().lower()
    if text.endswith("h"):
        return float(text[:-1]) * 60.0
    if ":" in text:
        hrs, mins = text.split(":", 1)
        return float(hrs) * 60.0 + float(mins)
    return float(text)


def estimate_drive_time_minutes(start_coord: Tuple[float, float], end_coord: Tuple[float, float], road_segments: List[Edge], average_speed_mph: float) -> float:
    """Estimates drive time by finding the shortest path on a road graph."""

    def _build_road_nx_graph_for_distance(segments: List[Edge]) -> nx.Graph:
        G = nx.Graph()
        for e in segments:
            G.add_edge(e.start, e.end, length_mi=e.length_mi)
        return G

    def _find_nearest_graph_node(graph_nodes: List[Tuple[float, float]], point: Tuple[float, float]) -> Tuple[float, float]:
        return min(graph_nodes, key=lambda n: (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2)

    road_graph = _build_road_nx_graph_for_distance(road_segments)

    if not road_graph.nodes() or not road_graph.edges():
        return float('inf')

    all_road_nodes = list(road_graph.nodes())
    if not all_road_nodes: # Should be caught by previous check, but good for safety
        return float('inf')

    actual_start_node_on_road = _find_nearest_graph_node(all_road_nodes, start_coord)
    actual_end_node_on_road = _find_nearest_graph_node(all_road_nodes, end_coord)

    if actual_start_node_on_road == actual_end_node_on_road:
        return 0.0

    try:
        distance_miles = nx.shortest_path_length(road_graph, source=actual_start_node_on_road, target=actual_end_node_on_road, weight='length_mi')
    except nx.NetworkXNoPath:
        return float('inf')
    except nx.NodeNotFound: # If one of the nodes is not in graph (e.g. graph is empty, or nearest node logic failed)
        return float('inf')


    if average_speed_mph <= 0:
        return float('inf') # Or raise ValueError("Average speed must be positive")

    return (distance_miles / average_speed_mph) * 60.0
