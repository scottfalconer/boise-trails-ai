import json
import os
from collections import defaultdict
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any
import networkx as nx
import re
from tqdm.auto import tqdm


@dataclass
class Edge:
    seg_id: Optional[str]
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_mi: float
    elev_gain_ft: float
    coords: List[
        Tuple[float, float]
    ]  # These are always stored in the canonical direction of the segment
    kind: str = field(default="trail")  # 'trail' or 'road'
    direction: str = field(default="both")
    access_from: Optional[str] = None
    _is_reversed: bool = field(default=False, kw_only=True)  # Internal flag

    def reverse(self) -> "Edge":
        """Return a new ``Edge`` representing traversal in the opposite direction."""
        return Edge(
            self.seg_id,
            self.name,
            self.end,
            self.start,
            self.length_mi,
            self.elev_gain_ft,
            self.coords,
            self.kind,
            self.direction,
            self.access_from,
            _is_reversed=not self._is_reversed,
        )

    @property
    def start_actual(self) -> Tuple[float, float]:
        """Returns the start coordinate based on the traversal direction."""
        return self.end if self._is_reversed else self.start

    @property
    def end_actual(self) -> Tuple[float, float]:
        """Returns the end coordinate based on the traversal direction."""
        return self.start if self._is_reversed else self.end

    @property
    def coords_actual(self) -> List[Tuple[float, float]]:
        """Returns the coordinate list in the direction of traversal."""
        return list(reversed(self.coords)) if self._is_reversed else self.coords


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
        geom = seg.get("geometry") or {
            "type": "LineString",
            "coordinates": seg["coordinates"],
        }
        if geom.get("type") == "LineString":
            coord_groups = [geom["coordinates"]]
        elif geom.get("type") == "MultiLineString":
            coord_groups = geom["coordinates"]
        else:
            continue
        for coords in coord_groups:
            if not coords:
                continue
            start = tuple(round(c, 6) for c in coords[0])
            end = tuple(round(c, 6) for c in coords[-1])
            length_ft = float(props.get("LengthFt", 0))
            if not length_ft:
                length_ft = (
                    sum(_haversine_mi(a, b) for a, b in zip(coords[:-1], coords[1:]))
                    * 5280
                )
            elev_gain = float(
                props.get("elevGainFt", 0) or props.get("ElevGainFt", 0) or 0
            )
            seg_id = props.get("segId") or props.get("id") or props.get("seg_id")
            name = props.get("segName") or props.get("name") or ""
            direction = props.get("direction", "both")
            access_from = (
                props.get("AccessFrom")
                or props.get("access_from")
                or props.get("accessFrom")
            )
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
        try:
            total = len(roads)
        except Exception:
            total = None
        road_iter = roads.iterrows()
        if total:
            from tqdm.auto import tqdm

            road_iter = tqdm(
                road_iter, total=total, desc="Processing road edges", unit="edge"
            )
        for _, row in road_iter:
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
                    lat_raw = row.get("lat")
                    if lat_raw is None:
                        lat_raw = row.get("latitude")
                    lon_raw = row.get("lon")
                    if lon_raw is None:
                        lon_raw = row.get("longitude")
                    lat = float(lat_raw)
                    lon = float(lon_raw)
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
            lat_raw = item.get("lat")
            if lat_raw is None:
                lat_raw = item.get("latitude")
            lon_raw = item.get("lon")
            if lon_raw is None:
                lon_raw = item.get("longitude")
            lat = float(lat_raw)
            lon = float(lon_raw)
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
            # Use coords_actual for elevation calculation
            current_coords = e.coords_actual
            if not current_coords:
                # If an edge is reversed, its elev_gain_ft should ideally be pre-calculated
                # or derived (e.g. from elev_drop_ft of the original).
                # For now, if coords_actual is empty (which implies original coords were empty), set gain to 0.
                # If _is_reversed is True, this calculation might be for the "drop" if elev_gain_ft
                # was defined for the canonical direction. This is a known complexity.
                e.elev_gain_ft = 0.0  # Or specific logic for reversed gain
                continue
            samples = list(src.sample([(lon, lat) for lon, lat in current_coords]))
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




def build_road_graph(road_segments: List[Edge]) -> nx.Graph:
    """Return an undirected graph with edge lengths for road segments."""
    G = nx.Graph()
    for e in road_segments:
        G.add_edge(e.start, e.end, length_mi=e.length_mi)
    return G


def connect_trails_to_roads(
    trail_edges: List[Edge],
    road_edges: List[Edge],
    *,
    threshold_meters: float = 50.0,
) -> List[Edge]:
    """Return short foot connectors between trail ends and nearby road nodes."""

    threshold_mi = threshold_meters / 1609.34
    road_nodes = list({e.start for e in road_edges} | {e.end for e in road_edges})
    connectors: List[Edge] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    idx = 0

    for t in trail_edges:
        for node in (t.start, t.end):
            nearest = None
            nearest_dist = float("inf")
            for rn in road_nodes:
                d = _haversine_mi(node, rn)
                if d < nearest_dist:
                    nearest = rn
                    nearest_dist = d
            if nearest is None or nearest_dist > threshold_mi:
                continue
            pair = (node, nearest)
            if pair in seen:
                continue
            seen.add(pair)
            connectors.append(
                Edge(
                    seg_id=None,
                    name="foot_connector",
                    start=node,
                    end=nearest,
                    length_mi=nearest_dist,
                    elev_gain_ft=0.0,
                    coords=[node, nearest],
                    kind="road",
                    direction="both",
                )
            )
            idx += 1

    return connectors


from functools import lru_cache


@lru_cache(maxsize=100_000)
def _estimate_time_cached(
    length_mi: float,
    elev_gain_ft: float,
    kind: str,
    pace_min_per_mi: float,
    grade_factor_sec_per_100ft: float,
    road_pace_min_per_mi: Optional[float],
) -> float:
    pace = (
        road_pace_min_per_mi
        if kind == "road" and road_pace_min_per_mi
        else pace_min_per_mi
    )
    base = length_mi * pace
    penalty = 0.0
    if kind != "road":
        penalty = (elev_gain_ft / 100.0) * (grade_factor_sec_per_100ft / 60.0)
    return base + penalty


def estimate_time(
    edge: Edge,
    pace_min_per_mi: float,
    grade_factor_sec_per_100ft: float,
    road_pace_min_per_mi: Optional[float] = None,
) -> float:
    """Return estimated time in minutes to traverse ``edge``.

    Results are memoized to improve performance when the same edge and pacing
    parameters are used repeatedly during route planning.
    """

    return _estimate_time_cached(
        edge.length_mi,
        edge.elev_gain_ft,
        edge.kind,
        pace_min_per_mi,
        grade_factor_sec_per_100ft,
        road_pace_min_per_mi,
    )


def load_completed(csv_path: str, year: int) -> Set:
    if not os.path.exists(csv_path):
        return set()
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.loc[df["year"] == year].copy()
    completed_ids = set(df["seg_id"].astype(str).unique())
    return completed_ids


def load_segment_tracking(track_path: str, segments_path: str) -> Dict[str, bool]:
    """Return a mapping of segment IDs to completion status.

    ``track_path`` may be either the legacy ``segment_tracking.json`` format
    (mapping of ID to completion info) or the official Boise Trails dashboard
    export containing ``CompletedSegmentIds``.  If ``track_path`` does not
    exist, a file is created using all segments from ``segments_path`` marked as
    incomplete.
    """

    if os.path.exists(track_path):
        with open(track_path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            # New official dashboard export
            if "CompletedSegmentIds" in data:
                completed_ids = {str(sid) for sid in data.get("CompletedSegmentIds", [])}
                segments = load_segments(segments_path)
                return {
                    str(e.seg_id): str(e.seg_id) in completed_ids
                    for e in segments
                    if e.seg_id is not None
                }

            # Legacy format
            result: Dict[str, bool] = {}
            for k, v in data.items():
                if isinstance(v, bool):
                    result[str(k)] = bool(v)
                elif isinstance(v, dict):
                    result[str(k)] = bool(v.get("completed", False))
                else:
                    raise ValueError("segment tracking values must be bool or object")
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
            seg_coords = [tuple(pt) for pt in e.coords_actual]  # Use actual coords
            if i == 0:
                coords.extend(seg_coords)
            else:
                last = coords[-1]
                # The connection logic relies on geometric proximity.
                # coords_actual provides the correct sequence for the current edge.
                if _close(last, seg_coords[0]):
                    coords.extend(seg_coords[1:])
                elif _close(
                    last, seg_coords[-1]
                ):  # Should not happen if edges are correctly ordered
                    coords.extend(list(reversed(seg_coords[:-1])))
                else:
                    coords.extend(seg_coords)

        segment = gpxpy.gpx.GPXTrackSegment()
        for lon, lat in coords:
            segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon))
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
                seg_coords = [tuple(pt) for pt in e.coords_actual]  # Use actual coords
                if i == 0:
                    coords.extend(seg_coords)
                else:
                    last = coords[-1]
                    if _close(last, seg_coords[0]):
                        coords.extend(seg_coords[1:])
                    elif _close(last, seg_coords[-1]):  # Should not happen
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


def _close(a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-6) -> bool:
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
    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(h))


def parse_time_budget(value: str) -> float:
    """Parse a time specification and return minutes.

    Accepts plain minutes ("90"), hours with ``h`` suffix ("1.5h"), or
    ``H:MM`` notation ("1:30").
    """

    text = value.strip().lower().replace(" ", "")
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


def compute_turn_direction(prev: Edge, nxt: Edge) -> str:
    """Return 'left', 'right', or 'straight' based on turn angle."""
    try:
        a1, a2 = prev.coords_actual[-2], prev.coords_actual[-1]
        b1, b2 = nxt.coords_actual[0], nxt.coords_actual[1]
    except Exception:
        return "straight"

    v1 = (a2[0] - a1[0], a2[1] - a1[1])
    v2 = (b2[0] - b1[0], b2[1] - b1[1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(cross) < 1e-9:
        return "straight"
    return "left" if cross > 0 else "right"


def generate_turn_by_turn(
    edges: List[Edge], challenge_ids: Optional[Set[str]] = None
) -> List[str]:
    """Return human-readable turn-by-turn directions for ``edges``."""

    if not edges:
        return []

    def _path_type(e: Edge) -> str:
        if e.kind == "road":
            return "road"
        if challenge_ids and e.seg_id and str(e.seg_id) in challenge_ids:
            return "official trail"
        return "connector trail"

    lines: List[str] = []
    first = edges[0]
    name = first.name or str(first.seg_id)
    dir_note = f" ({first.direction})" if first.direction != "both" else ""
    lines.append(
        f"Start on {name}{dir_note} ({_path_type(first)}) for {first.length_mi:.1f} mi"
    )

    prev = first
    for e in edges[1:]:
        name = e.name or str(e.seg_id)
        dir_note = f" ({e.direction})" if e.direction != "both" else ""
        turn = compute_turn_direction(prev, e)
        lines.append(
            f"Turn {turn} onto {name}{dir_note} ({_path_type(e)}) for {e.length_mi:.1f} mi"
        )
        prev = e

    return lines


def detect_inefficiencies(edges: List[Edge]) -> List[str]:
    """Return textual flags for redundant or inefficient sections."""
    flags: List[str] = []
    seen: Set[str] = set()
    for e in edges:
        if e.kind == "road" and e.length_mi > 0.3:
            flags.append(f"Road walk {e.length_mi:.1f} mi on {e.name or e.seg_id}")
        sid = str(e.seg_id) if e.seg_id is not None else None
        if sid:
            if sid in seen:
                flags.append(f"Repeated segment {sid}")
            else:
                seen.add(sid)
    return sorted(set(flags))


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

    def _find_nearest_graph_node(
        graph_nodes: List[Tuple[float, float]], point: Tuple[float, float]
    ) -> Tuple[float, float]:
        return min(
            graph_nodes, key=lambda n: (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2
        )

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
        seg_coords = [tuple(pt) for pt in e.coords_actual]  # Use actual coords
        if not seg_coords:
            continue
        if i == 0:
            coords.extend(seg_coords)
        else:
            last = coords[-1]
            if _close(last, seg_coords[0]):
                coords.extend(seg_coords[1:])
            elif _close(last, seg_coords[-1]):  # Should not happen
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




def find_next_required_segment(
    route: List[Edge], index: int, required_ids: Set[str]
) -> Optional[int]:
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
        return data.get("weight") or estimate_time(
            edge, context.pace, context.grade, context.road_pace
        )

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
