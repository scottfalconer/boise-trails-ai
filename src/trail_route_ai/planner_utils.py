import json
import os
from collections import defaultdict
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any
import networkx as nx
import re


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
    direction: str = field(default="both")
    access_from: Optional[str] = None


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
        direction = props.get("direction", "both")
        access_from = props.get("AccessFrom") or props.get("access_from") or props.get("accessFrom")
        length_mi = length_ft / 5280.0
        coords_list = [tuple(pt) for pt in coords]
        edge = Edge(
            seg_id,
            name,
            start,
            end,
            length_mi,
            elev_gain,
            coords_list,
            "trail",
            direction,
            access_from,
        )
        edges.append(edge)
    return edges


def bounding_box_from_edges(edges: List[Edge], buffer_km: float = 3.0):
    """Return [min_lon, min_lat, max_lon, max_lat] covering ``edges`` with an optional buffer."""
    if not edges:
        return None
    import numpy as np

    minx = min(min(e.start[0], e.end[0]) for e in edges)
    maxx = max(max(e.start[0], e.end[0]) for e in edges)
    miny = min(min(e.start[1], e.end[1]) for e in edges)
    maxy = max(max(e.start[1], e.end[1]) for e in edges)
    avg_lat = (miny + maxy) / 2
    km_per_deg_lon = 111.32 * abs(np.cos(np.radians(avg_lat)))
    km_per_deg_lat = 111.32
    dx = buffer_km / km_per_deg_lon
    dy = buffer_km / km_per_deg_lat
    return [minx - dx, miny - dy, maxx + dx, maxy + dy]


def load_roads(path: str, bbox: Optional[List[float]] = None) -> List[Edge]:
    """Load road segments from a GeoJSON file or OSM PBF."""

    if path.lower().endswith(".pbf"):
        from pyrosm import OSM

        print(f"Loading road network from PBF: {path}")
        osm = OSM(path, bounding_box=bbox)
        roads = osm.get_network(network_type="driving")
        print("Converting road network to edges...")
        edges: List[Edge] = []
        idx = 0
        for _, row in roads.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                coord_groups = [list(geom.coords)]
            elif geom.geom_type == "MultiLineString":
                coord_groups = [list(g.coords) for g in geom.geoms]
            else:
                continue
            for coords in coord_groups:
                start = tuple(round(c, 6) for c in coords[0])
                end = tuple(round(c, 6) for c in coords[-1])
                length = 0.0
                for a, b in zip(coords[:-1], coords[1:]):
                    length += _haversine_mi(a, b)
                edges.append(
                    Edge(
                        seg_id=f"road_{idx}",
                        name=row.get("name", ""),
                        start=start,
                        end=end,
                        length_mi=length,
                        elev_gain_ft=0.0,
                        coords=[tuple(pt) for pt in coords],
                        kind="road",
                        direction="both",
                    )
                )
                idx += 1
        print(f"Finished processing {len(edges)} road edges")
        return edges

    # default: GeoJSON
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
                    direction="both",
                )
            )
            idx += 1
    return edges


def load_trailheads(path: str) -> Dict[Tuple[float, float], str]:
    """Load trailhead locations from a JSON or CSV file.

    The file may contain a list of objects with ``lat`` and ``lon`` keys or a
    top-level ``trailheads`` list. CSV files should have ``lat`` and ``lon``
    columns.  A ``name`` column or key is optional.
    Returns a mapping of ``(lon, lat)`` tuples to trailhead names (which may be
    an empty string).
    """

    trailheads: Dict[Tuple[float, float], str] = {}

    if path.lower().endswith(".csv"):
        import csv

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row.get("lat") or row.get("latitude"))
                    lon = float(row.get("lon") or row.get("longitude"))
                except (TypeError, ValueError):
                    continue
                name = row.get("name", "")
                trailheads[(round(lon, 6), round(lat, 6))] = name
        return trailheads

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "trailheads" in data:
        entries = data["trailheads"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Unrecognized trailhead file format")

    for item in entries:
        try:
            lat = float(item.get("lat") or item.get("latitude"))
            lon = float(item.get("lon") or item.get("longitude"))
        except (TypeError, ValueError):
            continue
        name = item.get("name", "")
        trailheads[(round(lon, 6), round(lat, 6))] = name

    return trailheads


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
        if e.direction == "both":
            rev = Edge(
                e.seg_id,
                e.name,
                e.end,
                e.start,
                e.length_mi,
                e.elev_gain_ft,
                list(reversed(e.coords)),
                e.kind,
                e.direction,
            )
            graph[e.end].append((rev, rev.end))
        else:
            # one-way segment
            pass
    return graph


def build_road_graph(road_segments: List[Edge]) -> nx.Graph:
    """Return an undirected graph with edge lengths for road segments."""
    G = nx.Graph()
    for e in road_segments:
        G.add_edge(e.start, e.end, length_mi=e.length_mi)
    return G


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


def load_segment_tracking(track_path: str, segments_path: str) -> Dict[str, bool]:
    """Return a mapping of segment IDs to completion status.

    If ``track_path`` does not exist, a file is created containing all
    segment IDs from ``segments_path`` marked as incomplete. The tracking file
    stores additional metadata but this function only returns the completion
    status for use by the planner.
    """

    if os.path.exists(track_path):
        with open(track_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if isinstance(v, bool):
                    result[str(k)] = bool(v)
                elif isinstance(v, dict):
                    result[str(k)] = bool(v.get("completed", False))
                else:
                    raise ValueError(
                        "segment tracking values must be bool or object"
                    )
            return result
        raise ValueError("segment tracking file must be a JSON object")

    segments = load_segments(segments_path)
    tracking = {
        str(e.seg_id): {"completed": False, "name": e.name, "minutes": {}}
        for e in segments
        if e.seg_id is not None
    }
    os.makedirs(os.path.dirname(track_path) or ".", exist_ok=True)
    with open(track_path, "w") as f:
        json.dump(tracking, f, indent=2)
    return {sid: False for sid in tracking}


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


def _segments_from_edges(edges: List[Edge], mark_road_transitions: bool = True):
    """Convert a sequence of edges into GPX track segments and waypoints."""
    import gpxpy.gpx
    import xml.etree.ElementTree as ET

    segments: List[gpxpy.gpx.GPXTrackSegment] = []
    waypoints: List[gpxpy.gpx.GPXWaypoint] = []

    if not edges:
        return segments, waypoints

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
        for lon, lat in coords:
            segment.points.append(
                gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
            )
        segments.append(segment)
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
            segments.append(segment)

            if kind == "road" and coords:
                start_lon, start_lat = coords[0]
                end_lon, end_lat = coords[-1]
                waypoints.append(
                    gpxpy.gpx.GPXWaypoint(
                        latitude=start_lat,
                        longitude=start_lon,
                        name="Road start",
                        comment="Start of road connector",
                    )
                )
                waypoints.append(
                    gpxpy.gpx.GPXWaypoint(
                        latitude=end_lat,
                        longitude=end_lon,
                        name="Road end",
                        comment="End of road connector",
                    )
                )

    return segments, waypoints


def write_gpx(
    path: str,
    edges: List[Edge],
    mark_road_transitions: bool = True,
    start_name: Optional[str] = None,
):
    """Write a GPX file for the given sequence of edges.

    If ``mark_road_transitions`` is True, contiguous sections of ``edges`` are
    split into multiple track segments labelled with their ``kind`` (``trail`` or
    ``road``).  Waypoints are also inserted at the start and end of each road
    section so that GPX viewers which do not recognise extensions can still
    highlight road connectors.
    """
    import gpxpy.gpx

    if not edges:
        return

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)

    segments, waypoints = _segments_from_edges(edges, mark_road_transitions)
    track.segments.extend(segments)
    gpx.waypoints.extend(waypoints)

    if start_name and segments:
        first_pt = segments[0].points[0]
        gpx.waypoints.insert(
            0,
            gpxpy.gpx.GPXWaypoint(
                latitude=first_pt.latitude,
                longitude=first_pt.longitude,
                name=start_name,
            ),
        )

    with open(path, "w") as f:
        f.write(gpx.to_xml())


def write_multiday_gpx(
    path: str,
    daily_plans: List[Dict[str, Any]],
    mark_road_transitions: bool = True,
    colors: Optional[List[str]] = None,
):
    """Write a GPX file containing tracks for all days in ``daily_plans``.

    Each day is written as a separate track named with the ISO date. If
    ``colors`` is provided, track color is set using a simple extension and
    cycled through the list. A waypoint is inserted at the start of each day's
    first activity with the date as the name.
    """
    import gpxpy.gpx
    import xml.etree.ElementTree as ET

    if not daily_plans:
        return

    gpx = gpxpy.gpx.GPX()
    color_list = colors or []

    for idx, day_plan in enumerate(daily_plans):
        activities = day_plan.get("activities", [])
        activity_items = [a for a in activities if a["type"] == "activity"]
        if not activity_items:
            continue

        track = gpxpy.gpx.GPXTrack(name=day_plan["date"].isoformat())
        if color_list:
            ext = ET.Element("color")
            ext.text = color_list[idx % len(color_list)]
            track.extensions.append(ext)
        gpx.tracks.append(track)

        day_waypoint_added = False
        for activity in activity_items:
            edges = activity["route_edges"]
            start_name = activity.get("start_name")
            segments, waypoints = _segments_from_edges(edges, mark_road_transitions)
            if segments and not day_waypoint_added:
                first_pt = segments[0].points[0]
                gpx.waypoints.append(
                    gpxpy.gpx.GPXWaypoint(
                        latitude=first_pt.latitude,
                        longitude=first_pt.longitude,
                        name=day_plan["date"].isoformat(),
                    )
                )
                day_waypoint_added = True
            if start_name and segments:
                first_pt = segments[0].points[0]
                gpx.waypoints.append(
                    gpxpy.gpx.GPXWaypoint(
                        latitude=first_pt.latitude,
                        longitude=first_pt.longitude,
                        name=start_name,
                    )
                )
            track.segments.extend(segments)
            gpx.waypoints.extend(waypoints)

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
    # Formats like "1h30" or "2h15m"
    m = re.match(r"^(\d+(?:\.\d+)?)h(?:([0-5]?\d)(?:m)?)?$", text)
    if m:
        hours = float(m.group(1))
        minutes = float(m.group(2) or 0)
        return hours * 60.0 + minutes
    if text.endswith("h"):
        return float(text[:-1]) * 60.0
    if ":" in text:
        hrs, mins = text.split(":", 1)
        return float(hrs) * 60.0 + float(mins)
    return float(text)


def estimate_drive_time_minutes(
    start_coord: Tuple[float, float],
    end_coord: Tuple[float, float],
    road_graph: nx.Graph,
    average_speed_mph: float,
    return_distance: bool = False,
) -> float | Tuple[float, float]:
    """Estimate driving time between two coords using a prebuilt road graph.

    If ``return_distance`` is ``True`` the function returns ``(time, distance)``.
    """

    def _find_nearest_graph_node(graph_nodes: List[Tuple[float, float]], point: Tuple[float, float]) -> Tuple[float, float]:
        return min(graph_nodes, key=lambda n: (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2)

    if not road_graph.nodes() or not road_graph.edges():
        return (float("inf"), float("inf")) if return_distance else float("inf")

    all_road_nodes = list(road_graph.nodes())
    if not all_road_nodes:  # Should be caught by previous check, but good for safety
        return (float("inf"), float("inf")) if return_distance else float("inf")

    actual_start_node_on_road = _find_nearest_graph_node(all_road_nodes, start_coord)
    actual_end_node_on_road = _find_nearest_graph_node(all_road_nodes, end_coord)

    if actual_start_node_on_road == actual_end_node_on_road:
        return (0.0, 0.0) if return_distance else 0.0

    try:
        distance_miles = nx.shortest_path_length(
            road_graph,
            source=actual_start_node_on_road,
            target=actual_end_node_on_road,
            weight="length_mi",
        )
    except nx.NetworkXNoPath:
        return (float("inf"), float("inf")) if return_distance else float("inf")
    except nx.NodeNotFound:  # If one of the nodes is not in graph
        return (float("inf"), float("inf")) if return_distance else float("inf")


    if average_speed_mph <= 0:
        return (float("inf"), distance_miles) if return_distance else float("inf")

    time_min = (distance_miles / average_speed_mph) * 60.0
    return (time_min, distance_miles) if return_distance else time_min


def collect_route_coords(edges: List[Edge]) -> List[Tuple[float, float]]:
    """Return a continuous sequence of coordinates for ``edges``."""

    coords: List[Tuple[float, float]] = []
    if not edges:
        return coords

    for i, e in enumerate(edges):
        seg_coords = [tuple(pt) for pt in e.coords]
        if not seg_coords:
            continue
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
    return coords


def plot_route_map(coords: List[Tuple[float, float]], out_path: str) -> None:
    """Plot ``coords`` on a simple map and save to ``out_path``."""

    if not coords:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lons, lats = zip(*coords)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(lons, lats, "-", color="blue")
    ax.plot(lons[0], lats[0], "go")
    ax.plot(lons[-1], lats[-1], "ro")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_elevation_profile(
    coords: List[Tuple[float, float]], dem_path: str, out_path: str
) -> None:
    """Save an elevation profile plot for ``coords`` using ``dem_path``."""

    if not coords or not dem_path or not os.path.exists(dem_path):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import numpy as np

    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        samples = list(src.sample([(lon, lat) for lon, lat in coords]))
    elevs = [s[0] if s[0] != nodata else np.nan for s in samples]
    elevs = np.array(elevs, dtype=float) * 3.28084  # meters to feet

    # fill NaNs with previous value
    for i in range(1, len(elevs)):
        if np.isnan(elevs[i]):
            elevs[i] = elevs[i - 1]
    if np.isnan(elevs[0]):
        elevs[0] = elevs[~np.isnan(elevs)][0]

    dists = [0.0]
    for a, b in zip(coords[:-1], coords[1:]):
        dists.append(dists[-1] + _haversine_mi(a, b))

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(dists, elevs, color="green")
    ax.fill_between(dists, elevs, color="lightgreen", alpha=0.5)
    ax.set_xlabel("Distance (mi)")
    ax.set_ylabel("Elevation (ft)")
    ax.set_title("Elevation Profile")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def identify_quick_hits(
    segments: List[Edge],
    pace: float,
    grade: float,
    road_pace: float,
    max_minutes: float = 60.0,
) -> List[dict]:
    """Return simple isolated segments or small loops sorted by time.

    ``segments`` should contain trail segments for the challenge.  The function
    analyzes the connectivity of these segments and identifies two kinds of
    candidates:

    * Connected components whose total estimated time is under ``max_minutes``.
    * Individual segments that attach to the rest of the network at only one
      endpoint (leaf spurs). These are scored using an out-and-back time.

    The returned list contains dictionaries with keys ``name``, ``segments``,
    ``distance_mi``, ``elev_gain_ft`` and ``time_min`` sorted by ``time_min``.
    """

    G = nx.Graph()
    for e in segments:
        G.add_edge(e.start, e.end, edge=e)

    # Pre-compute component membership
    node_component = {}
    components = list(nx.connected_components(G))
    for idx, comp in enumerate(components):
        for n in comp:
            node_component[n] = idx

    comp_edges: List[List[Edge]] = [[] for _ in components]
    for e in segments:
        idx = node_component[e.start]
        comp_edges[idx].append(e)

    candidates = []

    for edges_in_comp in comp_edges:
        total_time = sum(
            estimate_time(e, pace, grade, road_pace) for e in edges_in_comp
        )
        total_dist = sum(e.length_mi for e in edges_in_comp)
        total_gain = sum(e.elev_gain_ft for e in edges_in_comp)
        if total_time <= max_minutes:
            name = edges_in_comp[0].name or str(edges_in_comp[0].seg_id)
            if len(edges_in_comp) > 1:
                name = f"{name} loop"
            candidates.append(
                {
                    "name": name,
                    "segments": [e.seg_id for e in edges_in_comp],
                    "distance_mi": round(total_dist, 2),
                    "elev_gain_ft": round(total_gain, 0),
                    "time_min": round(total_time, 1),
                }
            )

    # Leaf spurs
    degree = dict(G.degree())
    for e in segments:
        if degree[e.start] == 1 or degree[e.end] == 1:
            t = estimate_time(e, pace, grade, road_pace) * 2.0
            dist = e.length_mi * 2.0
            gain = e.elev_gain_ft * 2.0
            if t <= max_minutes:
                candidates.append(
                    {
                        "name": e.name or str(e.seg_id),
                        "segments": [e.seg_id],
                        "distance_mi": round(dist, 2),
                        "elev_gain_ft": round(gain, 0),
                        "time_min": round(t, 1),
                    }
                )

    # Remove duplicates by segment list
    uniq = {}
    for c in candidates:
        key = tuple(sorted(filter(None, c["segments"])))
        if key not in uniq or c["time_min"] < uniq[key]["time_min"]:
            uniq[key] = c

    out = list(uniq.values())
    out.sort(key=lambda x: x["time_min"])
    return out


@dataclass
class PlanningContext:
    """Shared information for route optimization."""

    graph: nx.DiGraph
    pace: float
    grade: float
    road_pace: float
    dist_cache: Optional[dict] = None


def calculate_route_efficiency_score(route: List[Edge]) -> float:
    """Return the ratio of unique to total distance for ``route``."""

    total = sum(e.length_mi for e in route)
    seen_ids: Set[str | None] = set()
    unique = 0.0
    for e in route:
        sid = str(e.seg_id) if e.seg_id is not None else None
        if sid not in seen_ids:
            unique += e.length_mi
            seen_ids.add(sid)
    if total == 0:
        return 1.0
    return unique / total


def calculate_path_efficiency(path: List[Edge]) -> float:
    """Alias for :func:`calculate_route_efficiency_score`."""

    return calculate_route_efficiency_score(path)


def find_next_required_segment(route: List[Edge], index: int, required_ids: Set[str]) -> Optional[int]:
    """Return the index of the next required segment after ``index``."""

    for i in range(index + 1, len(route)):
        sid = str(route[i].seg_id) if route[i].seg_id is not None else None
        if sid in required_ids:
            return i
    return None


def _edges_from_nodes(G: nx.DiGraph, nodes: List[Tuple[float, float]]) -> List[Edge]:
    out: List[Edge] = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        data = G.get_edge_data(a, b)
        if not data:
            continue
        ed = data[0]["edge"] if 0 in data else data["edge"]
        out.append(ed)
    return out


def find_alternative_path(
    context: PlanningContext,
    start: Tuple[float, float],
    end: Tuple[float, float],
    visited_ids: Set[str],
) -> Optional[List[Edge]]:
    """Find a path avoiding ``visited_ids`` if possible."""

    def weight(u, v, data):
        edge = data[0]["edge"] if 0 in data else data["edge"]
        sid = str(edge.seg_id) if edge.seg_id is not None else None
        if sid in visited_ids:
            return math.inf
        return data.get("weight") or estimate_time(edge, context.pace, context.grade, context.road_pace)

    try:
        nodes = nx.dijkstra_path(context.graph, start, end, weight=weight)
    except Exception:
        return None
    return _edges_from_nodes(context.graph, nodes)


def optimize_route_for_redundancy(
    context: PlanningContext,
    route: List[Edge],
    required_ids: Set[str],
    redundancy_threshold: float,
) -> List[Edge]:
    """Attempt to reduce redundant mileage in ``route``."""

    if not route:
        return route

    base_score = calculate_route_efficiency_score(route)
    if 1.0 - base_score < redundancy_threshold:
        return route

    visited: Set[str] = set()
    optimized: List[Edge] = []
    remaining_required: Set[str] = set(required_ids)
    i = 0
    while i < len(route):
        e = route[i]
        sid = str(e.seg_id) if e.seg_id is not None else None
        if sid and sid in visited and optimized:
            nxt = find_next_required_segment(route, i, remaining_required)
            if nxt is None:
                i += 1
                continue
            else:
                alt = find_alternative_path(
                    context,
                    optimized[-1].end,
                    route[nxt].start,
                    visited,
                )
                if alt is not None:
                    optimized.extend(alt)
                    for a in alt:
                        sid_alt = str(a.seg_id) if a.seg_id is not None else None
                        if sid_alt:
                            visited.add(sid_alt)
                            remaining_required.discard(sid_alt)
                    i = nxt + 1
                    continue
        optimized.append(e)
        if sid:
            visited.add(sid)
            remaining_required.discard(sid)
        i += 1

    new_score = calculate_route_efficiency_score(optimized)
    if new_score > base_score:
        return optimized
    return route

