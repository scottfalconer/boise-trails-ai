#!/usr/bin/env python3
"""Export a phone-first field packet from the current outing menu map data."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_day_packager import (  # noqa: E402
    build_outing_menu,
    format_miles,
    format_minutes,
    human_route_name,
    outing_time_bucket_sort,
)
from export_execution_gpx import haversine_miles, validate_track_segments  # noqa: E402
from field_activity_review import review_activity_against_segments  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    downsample_coords,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    read_json,
    shortest_connector_path,
)
from field_route_walkthrough_audit import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON as DEFAULT_WALKTHROUGH_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON as DEFAULT_WALKTHROUGH_OFFICIAL_GEOJSON,
    TrailGraph,
    filter_edges_for_track,
    load_graph_edges,
    matched_edge_groups,
    named_nonofficial_group,
    official_group_claimed,
    resample_track_segments_for_matching,
    start_access_text,
    text_mentions_edge,
)
from special_management_rule_audit import (  # noqa: E402
    DEFAULT_R2R_TRAILS_GEOJSON as DEFAULT_SPECIAL_MANAGEMENT_R2R_TRAILS_GEOJSON,
    DEFAULT_RULES_JSON as DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON,
    build_special_management_audit,
)


DEFAULT_CANONICAL_MAP_DATA_JSON = (
    YEAR_DIR
    / "outputs"
    / "private"
    / "2026-outing-menu-map-data.json"
)
DEFAULT_PUBLIC_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_MAP_DATA_JSON = DEFAULT_CANONICAL_MAP_DATA_JSON
DEFAULT_MAP_HTML = REPO_ROOT / "outing-menu-map.html"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_BASENAME = "phone-field-packet"
DEFAULT_CERTIFICATE_JSON = YEAR_DIR / "checkpoints" / "p90-responsible-relaxed-certificate-2026-05-06.json"
DEFAULT_TRAILHEAD_CANDIDATES = (
    YEAR_DIR / "inputs" / "open-data" / "city-parks-facilities-2026-05-04" / "trailhead_candidates.geojson"
)
DEFAULT_FIELD_ROUTE_HINTS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-field-route-hints.json"
DEFAULT_FIELD_MENU_REPLACEMENTS_JSON = (
    YEAR_DIR
    / "inputs"
    / "personal"
    / "private"
    / "2026-field-menu-replacements-v2-multi-start.private.json"
)
DEFAULT_ACCEPTED_REPLACEMENTS_JSON = YEAR_DIR / "inputs" / "accepted-route-replacements-v1.json"
DEFAULT_FIELD_DAY_LAYER_JSON = YEAR_DIR / "checkpoints" / "human-executable-field-day-layer-2026-05-10.json"
DEFAULT_SEGMENT_ELEVATION_JSON = YEAR_DIR / "derived" / "elevation" / "segment-elevation-2026-05-06.json"
DEFAULT_MAX_GAP_MILES = 0.05
DEFAULT_MAX_PARKING_GAP_MILES = 0.35
GPX_ZIP_NAME = "all-field-packet-gpx.zip"
OFFICIAL_GPX_DIR_NAME = "official"
CUE_GPX_DIR_NAME = "cues"
AUDIT_GPX_DIR_NAME = "audit"
FIELD_TOOL_DATA_NAME = "field-tool-data.json"
TIME_FILTER_MINUTES = [60, 90, 120, 180, 240, 360]
COMPLETED_STORAGE_KEY = "fieldPacketCompletedOutings"
ACTIVE_STORAGE_KEY = "fieldPacketActiveOuting"
GPX_PATH_KEYS = ("gpx_path", "cue_gpx_path", "audit_gpx_path")
GPX_HREF_KEYS = ("gpx_href", "cue_gpx_href", "audit_gpx_href")
NON_CREDIT_WAYFINDING_CUE_TYPES = {
    "start_access",
    "official_segment_start",
    "connector_named_trail",
    "connector_road",
    "repeat_official_noncredit",
    "overlap_repeat",
    "exit_access",
    "return_to_car",
}
PRIVATE_LITERAL_PATTERNS = (
    "/Users/scott",
    "outputs/private",
    "GETAthleteDashboard",
    "access_token",
    "refresh_token",
    "client_secret",
)


class FieldPacketCertificationError(ValueError):
    """Raised when an export would publish uncertified field artifacts."""


PRIVATE_REGEX_PATTERNS = (
    re.compile(r"\b911\s+n\.?\s+18th\b", re.IGNORECASE),
    re.compile(r"\b911\s+north\s+18th\b", re.IGNORECASE),
)
BLOCKED_CONNECTOR_TOKENS = (
    "private",
    "access_no",
    "access=no",
    "foot_no",
    "foot=no",
    "no_foot",
    "non_real",
    "graph_artifact",
)
SIGNPOST_TRAIL_NUMBERS = {
    "hippie shake": "50",
    "hippie shake trail": "50",
    "who now loop": "51",
    "who now loop trail": "51",
    "kemper's ridge": "52",
    "kemper's ridge trail": "52",
    "buena vista": "53",
    "buena vista trail": "53",
    "harrison hollow": "57",
    "harrison hollow trail": "57",
    "harrison ridge": "58",
    "harrison ridge trail": "58",
    "lower hulls gulch": "29",
    "lower hulls gulch trail": "29",
    "polecat loop": "81",
    "polecat loop trail": "81",
    "around the mountain": "98",
    "around the mountain trail": "98",
    "bucktail": "20A",
    "bucktail trail": "20A",
}
TRAIL_NUMBER_RE = re.compile(r"#\s*([0-9]+[A-Z]?)\b", re.IGNORECASE)
_FIELD_ROUTE_HINTS_CACHE: dict[str, Any] | None = None

def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:90] or "outing"


def stable_json_sha256(data: dict[str, Any]) -> str:
    """Hash the effective map payload without relying on file formatting."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lookup_text(value: Any) -> str:
    text = str(value or "").replace("’", "'").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


PUBLIC_DISPLAY_REPLACEMENTS = (
    (re.compile(r"\bprivate\s+Strava-derived\s+parking\s+anchor\b", re.IGNORECASE), "prior parking anchor"),
    (re.compile(r"\bprivate\s+Strava\s+parking\s+anchor\b", re.IGNORECASE), "prior parking anchor"),
    (re.compile(r"\bStrava-derived\s+parking\s+anchor\b", re.IGNORECASE), "prior parking anchor"),
    (re.compile(r"\bStrava\s+parking\s+anchor\b", re.IGNORECASE), "prior parking anchor"),
    (re.compile(r"\bStrava\s+anchor\b", re.IGNORECASE), "prior anchor"),
)


def public_display_text(value: Any) -> str:
    text = str(value or "")
    for pattern, replacement in PUBLIC_DISPLAY_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def public_display_value(value: Any) -> Any:
    if isinstance(value, str):
        return public_display_text(value)
    if isinstance(value, list):
        return [public_display_value(item) for item in value]
    if isinstance(value, dict):
        return {key: public_display_value(item) for key, item in value.items()}
    return value


def field_route_hints(path: Path = DEFAULT_FIELD_ROUTE_HINTS_JSON) -> dict[str, Any]:
    global _FIELD_ROUTE_HINTS_CACHE
    if _FIELD_ROUTE_HINTS_CACHE is None:
        _FIELD_ROUTE_HINTS_CACHE = read_json(path) if path.exists() else {}
    return _FIELD_ROUTE_HINTS_CACHE


def match_text(value: Any, expected: Any, *, contains: bool = False) -> bool:
    left = lookup_text(value)
    right = lookup_text(expected)
    if not right:
        return True
    return right in left if contains else left == right


def route_hint_matches(route: dict[str, Any], match: dict[str, Any]) -> bool:
    outing = route.get("outing") or {}
    label = route.get("label") or outing.get("label")
    trailhead = outing.get("trailhead") or route.get("trailhead")
    if match.get("label") and not match_text(label, match["label"]):
        return False
    if match.get("trailhead") and not match_text(trailhead, match["trailhead"], contains=True):
        return False
    return True


def access_hint_for(parking: dict[str, Any], first_trail: Any) -> dict[str, Any] | None:
    trailhead = parking.get("name")
    first = first_trail
    first_key = signpost_key(first_trail) or lookup_text(first_trail)
    for hint in field_route_hints().get("access_hints") or []:
        match = hint.get("match") or {}
        if match.get("trailhead") and not match_text(trailhead, match["trailhead"], contains=True):
            continue
        first_trail_values = match.get("first_trail") or match.get("first_trails") or []
        if isinstance(first_trail_values, str):
            first_trail_values = [first_trail_values]
        if first_trail_values and not any(
            match_text(first, value) or match_text(first_key, value)
            for value in first_trail_values
        ):
            continue
        return hint
    return None


def trail_label_from_id_and_name(trail_id: Any, trail_name: Any) -> str:
    name = str(trail_name or "").strip()
    identifier = str(trail_id or "").strip()
    if identifier and name and not name.startswith("#"):
        return f"#{identifier} {name}"
    if name:
        return name
    if identifier:
        return f"#{identifier}"
    return ""


def public_source_label(source_path: Path | None) -> str:
    if not source_path:
        return "in-memory-map-data"
    return source_path.name


def source_metadata_for_map_data(map_data: dict[str, Any], source_path: Path | None = None) -> dict[str, Any]:
    progress = map_data.get("progress") or {}
    metadata = {
        "canonical_data_role": "2026-outing-menu-map-data",
        "source_label": public_source_label(source_path),
        "map_data_sha256": stable_json_sha256(map_data),
        "package_count": len(map_data.get("packages") or []),
        "component_route_count": int((map_data.get("summary") or {}).get("component_route_count") or 0),
        "completed_segment_count_at_export": len(progress.get("completed_segment_ids") or []),
        "blocked_segment_count_at_export": len(progress.get("blocked_segment_ids") or []),
    }
    if source_path and source_path.exists() and source_path.is_file():
        metadata["source_file_sha256"] = file_sha256(source_path)
    return metadata


def normalized_segment_ids(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    return [str(value) for value in values or [] if value is not None]


def extract_map_data_from_html(html: str) -> dict[str, Any]:
    match = re.search(r"const DATA = (.*?);\nconst map =", html, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find embedded DATA payload in outing map HTML.")
    return json.loads(match.group(1))


def load_map_data(map_html: Path | None, map_data_json: Path | None) -> tuple[dict[str, Any], Path]:
    if map_data_json and map_data_json.exists():
        return read_json(map_data_json), map_data_json
    explicit_html = map_html and map_html != DEFAULT_MAP_HTML
    if explicit_html and map_html.exists():
        return extract_map_data_from_html(map_html.read_text(encoding="utf-8")), map_html
    if DEFAULT_PUBLIC_MAP_DATA_JSON.exists():
        return read_json(DEFAULT_PUBLIC_MAP_DATA_JSON), DEFAULT_PUBLIC_MAP_DATA_JSON
    if map_html and map_html.exists():
        return extract_map_data_from_html(map_html.read_text(encoding="utf-8")), map_html
    raise FileNotFoundError("No map HTML or map-data JSON source exists for the mobile field packet.")


def accepted_replacement_regression_failures(
    map_data: dict[str, Any],
    replacements_json: Path | None = None,
    accepted_replacements_json: Path | None = None,
) -> list[str]:
    failures = []
    replacements_json = replacements_json or DEFAULT_FIELD_MENU_REPLACEMENTS_JSON
    if not replacements_json.exists():
        return failures
    replacements = read_json(replacements_json)
    accepted_replacement_aliases = accepted_replacement_alias_index(accepted_replacements_json or DEFAULT_ACCEPTED_REPLACEMENTS_JSON)
    package_by_number = {
        str(package.get("package_number")): package
        for package in map_data.get("packages") or []
    }
    for entry in replacements.get("overrides") or []:
        replacement = entry.get("replace_package") or {}
        expected_components = [
            component
            for component in replacement.get("components") or []
            if component.get("source") or component.get("multi_start_alternative_id")
        ]
        if not expected_components:
            continue
        package_number = str(entry.get("package_number") or replacement.get("package_number") or "")
        package = package_by_number.get(package_number)
        if not package:
            # A progressed menu may remove a package entirely after its official
            # segment set is already accounted for. Preserve replacement
            # candidates whenever the package still exists in the active menu.
            continue
        expected_block_id = str(entry.get("block_id") or replacement.get("block_id") or "").strip()
        actual_block_id = str(package.get("block_id") or "").strip()
        if expected_block_id and actual_block_id and expected_block_id != actual_block_id:
            continue
        expected_block_name = lookup_text(entry.get("block_name") or replacement.get("block_name"))
        actual_block_name = lookup_text(package.get("block_name"))
        if expected_block_name and actual_block_name and expected_block_name != actual_block_name:
            continue
        actual_candidate_ids = {
            str(component.get("candidate_id"))
            for component in package.get("components") or []
            if component.get("candidate_id")
        }
        actual_components = list(package.get("components") or [])
        missing = [
            str(component.get("candidate_id"))
            for component in expected_components
            if not accepted_candidate_satisfied_by_package(
                str(component.get("candidate_id")),
                actual_candidate_ids,
                actual_components,
                accepted_replacement_aliases,
            )
        ]
        if missing:
            failures.append(
                f"Accepted field-menu replacement package {package_number} is missing candidates: "
                + ", ".join(missing)
            )
    return failures


def accepted_replacement_alias_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    aliases: dict[str, dict[str, Any]] = {}
    for record in payload.get("replacements") or []:
        if record.get("status") not in {"active", "waived"}:
            continue
        keys = {
            str(record.get("replacement_candidate_id") or ""),
            str((record.get("baseline_card_ref") or {}).get("candidate_id") or ""),
        }
        keys.update(str(value) for value in record.get("source_candidate_ids") or [])
        for key in keys:
            if key:
                aliases[key] = record
    return aliases


def accepted_candidate_satisfied_by_package(
    expected_candidate_id: str,
    actual_candidate_ids: set[str],
    actual_components: list[dict[str, Any]],
    accepted_replacement_aliases: dict[str, dict[str, Any]],
) -> bool:
    if expected_candidate_id in actual_candidate_ids:
        return True
    replacement = accepted_replacement_aliases.get(expected_candidate_id)
    if not replacement:
        return False
    replacement_id = replacement.get("replacement_id")
    replacement_candidate_id = str(replacement.get("replacement_candidate_id") or "")
    return any(
        component.get("accepted_replacement_id") == replacement_id
        or str(component.get("candidate_id") or "") == replacement_candidate_id
        for component in actual_components
    )


def field_menu_source_regression_failures(map_data: dict[str, Any]) -> list[str]:
    return accepted_replacement_regression_failures(map_data)


def validate_field_menu_source(map_data: dict[str, Any]) -> None:
    failures = field_menu_source_regression_failures(map_data)
    if failures:
        raise ValueError("Field menu source regression detected:\n" + "\n".join(failures))


def route_parts(feature: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometry = feature.get("geometry") or {}
    geometry_type = geometry.get("type")
    if geometry_type == "LineString":
        raw_parts = [geometry.get("coordinates") or []]
    elif geometry_type == "MultiLineString":
        raw_parts = geometry.get("coordinates") or []
    else:
        return []
    parts: list[list[tuple[float, float]]] = []
    for raw_part in raw_parts:
        part = []
        for coord in raw_part or []:
            if isinstance(coord, list | tuple) and len(coord) >= 2:
                part.append((float(coord[0]), float(coord[1])))
        if len(part) >= 2:
            parts.append(part)
    return parts


def feature_point(feature: dict[str, Any]) -> tuple[float, float] | None:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Point":
        return None
    coords = geometry.get("coordinates") or []
    if len(coords) < 2:
        return None
    return (float(coords[0]), float(coords[1]))


def midpoint(coords: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not coords:
        return None
    return coords[len(coords) // 2]


def coord_key(coord: tuple[float, float]) -> str:
    return f"{coord[0]:.5f},{coord[1]:.5f}"


def first_point(track_segments: list[list[tuple[float, float]]]) -> tuple[float, float] | None:
    for segment in track_segments:
        if segment:
            return segment[0]
    return None


def last_point(track_segments: list[list[tuple[float, float]]]) -> tuple[float, float] | None:
    for segment in reversed(track_segments):
        if segment:
            return segment[-1]
    return None


def track_distance_miles(track_segments: list[list[tuple[float, float]]] | None) -> float:
    total = 0.0
    for segment in track_segments or []:
        for index in range(1, len(segment)):
            total += haversine_miles(segment[index - 1], segment[index])
    return total


def cumulative_track_points(track_segments: list[list[tuple[float, float]]] | None) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    total = 0.0
    for segment in track_segments or []:
        prior: tuple[float, float] | None = None
        for point in segment:
            point = (float(point[0]), float(point[1]))
            if prior is not None:
                total += haversine_miles(prior, point)
            points.append({"point": point, "mile": total})
            prior = point
    return points


def flatten_track_segments(track_segments: list[list[tuple[float, float]]] | None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for segment in track_segments or []:
        for point in segment:
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    return points


def route_miles_near_point(
    route_points: list[dict[str, Any]],
    point: tuple[float, float],
    tolerance_miles: float = 0.03,
) -> list[float]:
    if not route_points:
        return []
    matches = [
        float(item["mile"])
        for item in route_points
        if haversine_miles(point, item["point"]) <= tolerance_miles
    ]
    if matches:
        return matches
    nearest = min(route_points, key=lambda item: haversine_miles(point, item["point"]))
    if haversine_miles(point, nearest["point"]) <= 0.12:
        return [float(nearest["mile"])]
    return []


def official_endpoint_route_miles(
    segment: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
    route_points: list[dict[str, Any]],
) -> list[float]:
    feature = official_index.get(str(segment.get("seg_id") or ""))
    if not feature:
        return []
    miles: list[float] = []
    for part in route_parts(feature):
        if len(part) < 2:
            continue
        miles.extend(route_miles_near_point(route_points, part[0]))
        miles.extend(route_miles_near_point(route_points, part[-1]))
    return miles


def official_feature_endpoint_route_miles(
    feature: dict[str, Any],
    route_points: list[dict[str, Any]],
) -> list[float]:
    miles: list[float] = []
    for part in route_parts(feature):
        if len(part) < 2:
            continue
        miles.extend(route_miles_near_point(route_points, part[0]))
        miles.extend(route_miles_near_point(route_points, part[-1]))
    return sorted({round(float(mile), 3) for mile in miles})


def official_segment_endpoint_miles_for_ids(
    segment_ids: list[Any],
    official_index: dict[str, dict[str, Any]],
    route_points: list[dict[str, Any]],
) -> list[float]:
    miles: list[float] = []
    for segment_id in normalized_segment_ids(segment_ids):
        feature = official_index.get(str(segment_id))
        if feature:
            miles.extend(official_feature_endpoint_route_miles(feature, route_points))
    return sorted({round(float(mile), 3) for mile in miles})


def nearest_route_anchor_mile(
    candidates: list[float],
    *,
    target_mile: float,
    floor_mile: float,
    floor_slop_miles: float = 0.08,
) -> float | None:
    usable = [float(mile) for mile in candidates if float(mile) >= floor_mile - floor_slop_miles]
    if not usable:
        return None
    target = max(float(target_mile or 0), float(floor_mile or 0))
    return min(usable, key=lambda mile: (abs(mile - target), mile))


def route_interval_end_for_official_cue(
    cue: dict[str, Any],
    *,
    start_mile: float,
    fallback_end_mile: float,
    official_index: dict[str, dict[str, Any]],
    route_points: list[dict[str, Any]],
) -> float:
    candidates = official_segment_endpoint_miles_for_ids(cue.get("official_segment_ids") or [], official_index, route_points)
    target = start_mile + float(cue.get("leg_miles") or 0)
    usable = [mile for mile in candidates if mile > start_mile + 0.03]
    if usable:
        return min(usable, key=lambda mile: (abs(mile - target), mile))
    return fallback_end_mile


def set_cue_route_interval(cue: dict[str, Any], start_mile: float, end_mile: float) -> None:
    start = max(0.0, float(start_mile or 0))
    end = max(start, float(end_mile or start))
    cue["route_miles"] = round(start, 3)
    cue["route_leg_miles"] = round(end - start, 3)


def point_at_route_mile(
    route_points: list[dict[str, Any]],
    target_mile: float,
) -> tuple[float, float] | None:
    if not route_points:
        return None
    target = max(0.0, min(float(target_mile or 0), float(route_points[-1].get("mile") or 0)))
    prior = route_points[0]
    for item in route_points[1:]:
        left_mile = float(prior.get("mile") or 0)
        right_mile = float(item.get("mile") or 0)
        if right_mile >= target:
            if right_mile == left_mile:
                return item["point"]
            ratio = (target - left_mile) / (right_mile - left_mile)
            left = prior["point"]
            right = item["point"]
            return (
                left[0] + (right[0] - left[0]) * ratio,
                left[1] + (right[1] - left[1]) * ratio,
            )
        prior = item
    return route_points[-1]["point"]


def route_interval_coordinates(
    route_points: list[dict[str, Any]],
    *,
    start_mile: float,
    end_mile: float,
) -> list[tuple[float, float]]:
    if len(route_points) < 2 or end_mile <= start_mile:
        return []
    coords: list[tuple[float, float]] = []
    start_point = point_at_route_mile(route_points, start_mile)
    end_point = point_at_route_mile(route_points, end_mile)
    if start_point:
        coords.append(start_point)
    coords.extend(
        item["point"]
        for item in route_points
        if start_mile < float(item.get("mile") or 0) < end_mile
    )
    if end_point:
        coords.append(end_point)
    return coords


def cue_route_interval(cue: dict[str, Any]) -> tuple[float, float] | None:
    if cue.get("route_miles") is None and cue.get("cum_miles") is None:
        return None
    start = float(cue.get("route_miles") if cue.get("route_miles") is not None else cue.get("cum_miles") or 0)
    length = float(cue.get("route_leg_miles") if cue.get("route_leg_miles") is not None else cue.get("leg_miles") or 0)
    if length <= 0:
        return None
    return start, start + length


def assign_wayfinding_route_miles(route: dict[str, Any]) -> dict[str, Any]:
    """Attach true GPX route-mile anchors to cues.

    The phone map draws the active leg from GPX geometry, so cue boundaries must
    be based on where the cue actually occurs in that GPX. Card mileage can be
    shorter than GPX mileage when a required trail group contains a connector or
    repeat before the next official cue boundary.
    """

    cues = route.get("wayfinding_cues") or []
    route_points = cumulative_track_points(route.get("_track_segments") or [])
    official_index = route.get("_official_segment_index") or {}
    if len(cues) < 2 or len(route_points) < 2 or not official_index:
        return route
    track_total = float(route_points[-1].get("mile") or 0)
    official_anchor_by_index: dict[int, float] = {}
    floor = 0.0
    for index, cue in enumerate(cues):
        segment_ids = cue.get("official_segment_ids") or []
        if not segment_ids:
            continue
        candidates = official_segment_endpoint_miles_for_ids(segment_ids, official_index, route_points)
        anchor = nearest_route_anchor_mile(
            candidates,
            target_mile=float(cue.get("cum_miles") or 0),
            floor_mile=floor,
        )
        if anchor is None:
            continue
        official_anchor_by_index[index] = anchor
        floor = max(floor, anchor)

    previous_end = 0.0
    for index, cue in enumerate(cues):
        next_official_anchor = None
        for next_index in range(index + 1, len(cues)):
            if next_index in official_anchor_by_index:
                next_official_anchor = official_anchor_by_index[next_index]
                break
        if index in official_anchor_by_index:
            start = max(previous_end, official_anchor_by_index[index])
            next_is_connector = index + 1 < len(cues) and not (cues[index + 1].get("official_segment_ids") or [])
            if next_is_connector or next_official_anchor is None or next_official_anchor <= start + 0.05:
                end = route_interval_end_for_official_cue(
                    cue,
                    start_mile=start,
                    fallback_end_mile=min(track_total, start + float(cue.get("leg_miles") or 0)),
                    official_index=official_index,
                    route_points=route_points,
                )
            else:
                end = next_official_anchor
        else:
            start = previous_end
            if next_official_anchor is not None:
                end = next_official_anchor
            else:
                end = min(track_total, start + float(cue.get("leg_miles") or 0))
        set_cue_route_interval(cue, start, end)
        previous_end = max(previous_end, float(cue.get("route_miles") or 0) + float(cue.get("route_leg_miles") or 0))
    return route


def wayfinding_total_miles(cues: list[dict[str, Any]]) -> float:
    return max(
        (
            float(cue.get("cum_miles") or 0) + float(cue.get("leg_miles") or 0)
            for cue in cues
        ),
        default=0.0,
    )


def route_card_mileage_tolerance(card_miles: float) -> float:
    return max(0.25, float(card_miles or 0) * 0.08)


def reconcile_wayfinding_miles_to_route_card(route: dict[str, Any]) -> dict[str, Any]:
    cues = route.get("wayfinding_cues") or []
    outing = route.get("outing") or {}
    card_miles = float(outing.get("on_foot_miles") or 0)
    cue_miles = wayfinding_total_miles(cues)
    route["wayfinding_mileage_reconciliation"] = {
        "schema": "boise_trails_wayfinding_mileage_reconciliation_v1",
        "status": "already_consistent",
        "card_on_foot_miles": round(card_miles, 2),
        "source_cue_miles": round(cue_miles, 2),
    }
    if not cues or card_miles <= 0 or cue_miles <= 0:
        route["wayfinding_mileage_reconciliation"]["status"] = "not_applicable"
        return route
    tolerance = route_card_mileage_tolerance(card_miles)
    if abs(cue_miles - card_miles) <= tolerance:
        return route

    scale = card_miles / cue_miles
    cum_miles = 0.0
    for cue in cues:
        source_leg = float(cue.get("leg_miles") or 0)
        source_cum = float(cue.get("cum_miles") or 0)
        cue["source_cum_miles"] = round(source_cum, 2)
        cue["source_leg_miles"] = round(source_leg, 2)
        cue["cum_miles"] = round(cum_miles, 2)
        cue["leg_miles"] = round(max(0.0, source_leg * scale), 2)
        cum_miles += max(0.0, source_leg * scale)
    if cues:
        final_total = wayfinding_total_miles(cues)
        delta = round(card_miles - final_total, 2)
        if abs(delta) >= 0.01:
            cues[-1]["leg_miles"] = round(max(0.0, float(cues[-1].get("leg_miles") or 0) + delta), 2)
        for cue in cues:
            refresh_wayfinding_text(cue)
    route["wayfinding_mileage_reconciliation"] = {
        "schema": "boise_trails_wayfinding_mileage_reconciliation_v1",
        "status": "scaled_to_route_card",
        "card_on_foot_miles": round(card_miles, 2),
        "source_cue_miles": round(cue_miles, 2),
        "scale_factor": round(scale, 6),
        "tolerance_miles": round(tolerance, 2),
    }
    return route


def official_segment_label(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    name = props.get("seg_name") or props.get("segName") or props.get("segment_name") or props.get("trail_name")
    return signpost_label(name) or normalized_trail_text(name)


def cue_signed_trail_keys(cue: dict[str, Any]) -> set[str]:
    values = list(cue.get("signed_as") or [])
    if cue.get("target"):
        values.append(cue.get("target"))
    return {signpost_key(value) for value in values if signpost_key(value)}


def off_label_segments_for_cue(
    cue: dict[str, Any],
    *,
    official_index: dict[str, dict[str, Any]],
    route_points: list[dict[str, Any]],
    candidate_segment_ids: set[str] | None = None,
) -> list[str]:
    start = float(cue.get("route_miles") or 0)
    end = start + float(cue.get("route_leg_miles") or 0)
    if end <= start + 0.05:
        return []
    claimed_ids = set(normalized_segment_ids(cue.get("official_segment_ids") or []))
    signed_keys = cue_signed_trail_keys(cue)
    labels = []
    segment_items = (
        ((segment_id, official_index[segment_id]) for segment_id in sorted(candidate_segment_ids) if segment_id in official_index)
        if candidate_segment_ids
        else official_index.items()
    )
    for segment_id, feature in segment_items:
        if segment_id in claimed_ids:
            continue
        props = feature.get("properties") or {}
        trail_name = props.get("trail_name") or props.get("masterTrailName") or props.get("segName") or props.get("seg_name")
        if signpost_key(trail_name) in signed_keys:
            continue
        endpoint_miles = official_feature_endpoint_route_miles(feature, route_points)
        interval_hits = [mile for mile in endpoint_miles if start - 0.04 <= mile <= end + 0.04]
        if len(interval_hits) < 2:
            continue
        length_miles = float(props.get("LengthMi") or props.get("length_miles") or 0)
        if not length_miles:
            length_ft = float(props.get("LengthFt") or 0)
            length_miles = length_ft / 5280 if length_ft else 0
        interval_span_miles = max(interval_hits) - min(interval_hits)
        if length_miles < 0.05 and interval_span_miles < 0.05:
            continue
        label = official_segment_label(feature)
        if label and label not in labels:
            labels.append(label)
    return labels[:3]


def route_off_label_candidate_segment_ids(route: dict[str, Any], official_index: dict[str, dict[str, Any]]) -> set[str]:
    candidates: set[str] = set()
    for cue in route.get("route_cues") or []:
        link_sources = list(cue.get("between_links") or [])
        if cue.get("start_access"):
            link_sources.append(cue.get("start_access") or {})
        if cue.get("return_to_car"):
            link_sources.append(cue.get("return_to_car") or {})
        for link in link_sources:
            for key in ("official_repeat_segment_ids", "connector_segment_ids", "segment_ids"):
                candidates.update(normalized_segment_ids(link.get(key) or []))
    if not candidates and len(official_index) <= 50:
        candidates.update(official_index.keys())
    return candidates


def apply_off_label_route_leg_warnings(route: dict[str, Any]) -> dict[str, Any]:
    cues = route.get("wayfinding_cues") or []
    route_points = cumulative_track_points(route.get("_track_segments") or [])
    official_index = route.get("_official_segment_index") or {}
    if not cues or len(route_points) < 2 or not official_index:
        return route
    candidate_segment_ids = route_off_label_candidate_segment_ids(route, official_index)
    for cue in cues:
        if not cue.get("official_segment_ids"):
            continue
        labels = off_label_segments_for_cue(
            cue,
            official_index=official_index,
            route_points=route_points,
            candidate_segment_ids=candidate_segment_ids,
        )
        if not labels:
            continue
        warning = (
            "This active line also uses "
            + sentence_list(labels)
            + " as connector/repeat mileage; follow the blue line and signs, not only the cue title."
        )
        existing = str(cue.get("field_warning") or "").strip()
        cue["field_warning"] = " ".join(part for part in [existing, warning] if part)
        add_cue_note(cue, warning)
        refresh_wayfinding_text(cue)
    return route


def non_credit_gaps_for_cue(
    cue: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    official_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    segments = cue.get("segments") or []
    route_points = cumulative_track_points(track_segments)
    total_miles = route_points[-1]["mile"] if route_points else 0.0
    result = {
        "start_access_gap_miles": 0.0,
        "return_access_gap_miles": 0.0,
    }
    if not segments or not route_points:
        return result
    first_miles = official_endpoint_route_miles(segments[0], official_index, route_points)
    if first_miles:
        result["start_access_gap_miles"] = round(max(0.0, min(first_miles)), 3)
    last_miles = official_endpoint_route_miles(segments[-1], official_index, route_points)
    if last_miles:
        result["return_access_gap_miles"] = round(max(0.0, total_miles - max(last_miles)), 3)
    return result


def bearing_degrees(start: tuple[float, float], end: tuple[float, float]) -> float | None:
    if start == end:
        return None
    lon1 = math.radians(start[0])
    lat1 = math.radians(start[1])
    lon2 = math.radians(end[0])
    lat2 = math.radians(end[1])
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    if x == 0 and y == 0:
        return None
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def signed_bearing_delta(incoming: float, outgoing: float) -> float:
    return (outgoing - incoming + 540) % 360 - 180


def absolute_bearing_delta(first: float, second: float) -> float:
    return abs(signed_bearing_delta(first, second))


def turn_phrase(delta: float) -> str | None:
    abs_delta = abs(delta)
    if abs_delta < 25:
        return "continue straight"
    if abs_delta < 60:
        return "bear right" if delta > 0 else "bear left"
    if abs_delta < 140:
        return "turn right" if delta > 0 else "turn left"
    if abs_delta < 170:
        return "make a sharp right" if delta > 0 else "make a sharp left"
    return "turn around"


def nearby_track_point(
    points: list[tuple[float, float]],
    start_index: int,
    step: int,
    min_distance_miles: float = 0.01,
) -> tuple[float, float] | None:
    if not points:
        return None
    anchor = points[start_index]
    index = start_index + step
    best: tuple[float, float] | None = None
    while 0 <= index < len(points):
        candidate = points[index]
        if candidate != anchor:
            best = candidate
            if haversine_miles(anchor, candidate) >= min_distance_miles:
                return candidate
        index += step
    return best


def endpoint_candidates(parts: list[list[tuple[float, float]]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for part in parts:
        if len(part) < 2:
            continue
        candidates.append({"point": part[0], "neighbor": part[1]})
        candidates.append({"point": part[-1], "neighbor": part[-2]})
    return candidates


def transition_geometry(
    prior_group: dict[str, Any],
    next_group: dict[str, Any],
    official_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not official_index:
        return None
    prior_segments = prior_group.get("segments") or []
    next_segments = next_group.get("segments") or []
    if not prior_segments or not next_segments:
        return None
    prior_feature = official_index.get(str(prior_segments[-1].get("seg_id") or ""))
    next_feature = official_index.get(str(next_segments[0].get("seg_id") or ""))
    if not prior_feature or not next_feature:
        return None
    prior_candidates = endpoint_candidates(route_parts(prior_feature))
    next_candidates = endpoint_candidates(route_parts(next_feature))
    if not prior_candidates or not next_candidates:
        return None

    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for prior in prior_candidates:
        for next_candidate in next_candidates:
            distance = haversine_miles(prior["point"], next_candidate["point"])
            if best is None or distance < best[0]:
                best = (distance, prior, next_candidate)
    if best is None:
        return None
    _, prior, next_candidate = best
    midpoint_point = (
        (prior["point"][0] + next_candidate["point"][0]) / 2,
        (prior["point"][1] + next_candidate["point"][1]) / 2,
    )
    return {
        "point": midpoint_point,
        "next_bearing": bearing_degrees(next_candidate["point"], next_candidate["neighbor"]),
    }


def turn_phrase_for_transition(
    prior_group: dict[str, Any],
    next_group: dict[str, Any],
    track_segments: list[list[tuple[float, float]]] | None,
    official_index: dict[str, dict[str, Any]] | None,
) -> str | None:
    transition = transition_geometry(prior_group, next_group, official_index)
    points = flatten_track_segments(track_segments)
    if not transition or len(points) < 3:
        return None

    target = transition["point"]
    nearest_index = min(range(len(points)), key=lambda index: haversine_miles(target, points[index]))
    if haversine_miles(target, points[nearest_index]) > 0.12:
        return None

    before = nearby_track_point(points, nearest_index, -1)
    after = nearby_track_point(points, nearest_index, 1)
    if not before or not after:
        return None
    incoming = bearing_degrees(before, points[nearest_index])
    outgoing = bearing_degrees(points[nearest_index], after)
    if incoming is None or outgoing is None:
        return None

    expected_next_bearing = transition.get("next_bearing")
    if expected_next_bearing is not None and absolute_bearing_delta(outgoing, expected_next_bearing) > 70:
        return None
    return turn_phrase(signed_bearing_delta(incoming, outgoing))


def densify_segment(
    coords: list[tuple[float, float]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    dense = [coords[0]]
    target_gap = max_gap_miles * 0.75
    for left, right in zip(coords, coords[1:]):
        gap = haversine_miles(left, right)
        steps = max(1, math.ceil(gap / target_gap)) if target_gap > 0 else 1
        for step in range(1, steps + 1):
            ratio = step / steps
            dense.append(
                (
                    left[0] + (right[0] - left[0]) * ratio,
                    left[1] + (right[1] - left[1]) * ratio,
                )
            )
    return dense


def densify_track_segments(
    track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> list[list[tuple[float, float]]]:
    return [densify_segment(segment, max_gap_miles=max_gap_miles) for segment in track_segments]


def load_default_connector_graph() -> dict[str, Any] | None:
    try:
        official_segments, _metadata = load_official_segments(DEFAULT_OFFICIAL_GEOJSON)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None
    return load_connector_graph(DEFAULT_CONNECTOR_GEOJSON, official_segments=official_segments)


def append_deduped_track_point(target: list[tuple[float, float]], coord: Any) -> None:
    point = (float(coord[0]), float(coord[1]))
    if target and haversine_miles(target[-1], point) < 0.000001:
        return
    target.append(point)


def stitch_inter_segment_track_gaps(
    track_segments: list[list[tuple[float, float]]],
    connector_graph: dict[str, Any] | None,
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
    stitch_snap_tolerance_miles: float = 0.03,
) -> list[list[tuple[float, float]]]:
    """Insert explicit graph connector geometry between GPX parts when possible."""

    if not track_segments or not connector_graph:
        return track_segments
    stitched: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for segment in track_segments:
        if not segment:
            continue
        if not current:
            current = list(segment)
            continue
        gap = haversine_miles(current[-1], segment[0])
        if gap <= max_gap_miles:
            for point in segment:
                append_deduped_track_point(current, point)
            continue
        stitch = shortest_connector_path(
            current[-1],
            segment[0],
            connector_graph,
            stitch_snap_tolerance_miles,
        )
        if not stitch:
            stitched.append(current)
            current = list(segment)
            continue
        for point in stitch.get("path_coordinates") or []:
            append_deduped_track_point(current, point)
        for point in segment:
            append_deduped_track_point(current, point)
    if current:
        stitched.append(current)
    return stitched


def inter_segment_gap_count(
    track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> int:
    count = 0
    for left, right in zip(track_segments, track_segments[1:]):
        if left and right and haversine_miles(left[-1], right[0]) > max_gap_miles:
            count += 1
    return count


def source_gap_repair_summary(
    raw_track_segments: list[list[tuple[float, float]]],
    stitched_track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> dict[str, Any]:
    raw_gap_count = inter_segment_gap_count(raw_track_segments, max_gap_miles=max_gap_miles)
    remaining_gap_count = inter_segment_gap_count(stitched_track_segments, max_gap_miles=max_gap_miles)
    repaired_count = max(0, raw_gap_count - remaining_gap_count)
    return {
        "raw_inter_segment_gap_count": raw_gap_count,
        "repaired_inter_segment_gap_count": repaired_count,
        "remaining_inter_segment_gap_count": remaining_gap_count,
        "repair_method": "graph_connector_stitch" if repaired_count else None,
    }


def indexed_features(
    map_data: dict[str, Any],
    collection_name: str,
    property_name: str,
) -> dict[str, list[dict[str, Any]]]:
    collection = (map_data.get("feature_collections") or {}).get(collection_name) or {}
    index: dict[str, list[dict[str, Any]]] = {}
    for feature in collection.get("features") or []:
        value = (feature.get("properties") or {}).get(property_name)
        if value is None:
            continue
        index.setdefault(str(value), []).append(feature)
    return index


def official_segment_index(map_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    collection = (map_data.get("feature_collections") or {}).get("official_segments") or {}
    index = {}
    for feature in collection.get("features") or []:
        props = feature.get("properties") or {}
        segment_id = props.get("seg_id") or props.get("segment_id")
        if segment_id is not None:
                index[str(segment_id)] = feature
    return index


def official_segment_for_review(feature: dict[str, Any]) -> dict[str, Any] | None:
    props = feature.get("properties") or {}
    segment_id = props.get("seg_id") or props.get("segment_id")
    parts = route_parts(feature)
    if segment_id is None or not parts:
        return None
    coords = parts[0]
    return {
        "seg_id": str(segment_id),
        "seg_name": props.get("segment_name") or props.get("seg_name"),
        "trail_name": props.get("trail_name"),
        "official_miles": float(props.get("official_miles") or 0.0),
        "direction": props.get("direction_rule") or props.get("direction") or "both",
        "coordinates": coords,
        "start": coords[0],
        "end": coords[-1],
    }


def official_segments_for_review(segments_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    segments = []
    for segment_id in sorted(segments_by_id, key=lambda value: (len(value), value)):
        segment = official_segment_for_review(segments_by_id[segment_id])
        if segment:
            segments.append(segment)
    return segments


def coordinate_bbox(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    return min(lons), min(lats), max(lons), max(lats)


def bbox_overlaps(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    lon_buffer: float,
    lat_buffer: float,
) -> bool:
    return not (
        left[2] + lon_buffer < right[0]
        or right[2] + lon_buffer < left[0]
        or left[3] + lat_buffer < right[1]
        or right[3] + lat_buffer < left[1]
    )


def candidate_official_segments_for_route(
    route_coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    planned_ids: set[str],
    threshold_miles: float,
) -> list[dict[str, Any]]:
    if len(route_coords) < 2:
        return official_segments
    route_bbox = coordinate_bbox(route_coords)
    origin_lat = sum(point[1] for point in route_coords) / len(route_coords)
    lat_buffer = threshold_miles / 69.0
    lon_buffer = threshold_miles / max(1e-6, 69.172 * math.cos(math.radians(origin_lat)))
    candidates = []
    for segment in official_segments:
        segment_id = str(segment["seg_id"])
        segment_bbox = coordinate_bbox(segment["coordinates"])
        if segment_id in planned_ids or bbox_overlaps(route_bbox, segment_bbox, lon_buffer, lat_buffer):
            candidates.append(segment)
    return candidates


def route_claim_index(routes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    claims: dict[str, list[dict[str, Any]]] = {}
    for route in routes:
        outing = route.get("outing") or {}
        claim = {
            "outing_id": outing.get("outing_id") or route.get("outing_id"),
            "label": outing.get("label") or route.get("label"),
            "candidate_ids": [str(candidate_id) for candidate_id in outing.get("candidate_ids") or []],
        }
        for segment_id in normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids")):
            claims.setdefault(segment_id, []).append(claim)
    return claims


def segment_review_brief(segment: dict[str, Any], owners: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seg_id": str(segment.get("seg_id")),
        "seg_name": segment.get("seg_name"),
        "trail_name": segment.get("trail_name"),
        "direction": segment.get("direction"),
        "official_miles": round(float(segment.get("official_miles") or 0.0), 2),
        "owned_by_routes": owners,
    }


def apply_segment_ownership_reconciliation(
    routes: list[dict[str, Any]],
    segments_by_id: dict[str, dict[str, Any]],
    *,
    elevation_sampler: Any = None,
    threshold_miles: float = 0.045,
    max_activity_points: int = 1200,
) -> None:
    """Declare official segments a route traverses but another card owns.

    This does not move credit between route cards. It makes the cross-route
    ownership explicit so latent official credit cannot hide behind route-local
    coverage checks.
    """

    official_segments = official_segments_for_review(segments_by_id)
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    claims = route_claim_index(routes)
    all_claimed_ids = set(claims)
    for route in routes:
        outing = route.get("outing") or {}
        route_key = str(outing.get("outing_id") or route.get("outing_id") or "")
        planned_ids = set(normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids")))
        coords = flatten_track_segments(route.get("_track_segments") or [])
        review_coords = downsample_coords(coords, max_points=max_activity_points)
        candidates = candidate_official_segments_for_route(coords, official_segments, planned_ids, threshold_miles)
        review = review_activity_against_segments(
            review_coords,
            candidates,
            planned_segment_ids=planned_ids,
            planned_outing_id=route_key,
            threshold_miles=threshold_miles,
            min_fraction=0.85,
            partial_min_fraction=0.2,
            elevation_sampler=elevation_sampler,
        )
        declared_owned_elsewhere = []
        unclaimed_completed = []
        for segment_id in normalized_segment_ids(review.get("extra_completed_segment_ids") or []):
            other_owners = [
                owner
                for owner in claims.get(segment_id, [])
                if str(owner.get("outing_id") or "") != route_key
            ]
            if other_owners:
                declared_owned_elsewhere.append(segment_review_brief(official_by_id.get(segment_id, {}), other_owners))
            elif segment_id not in all_claimed_ids:
                unclaimed_completed.append(segment_review_brief(official_by_id.get(segment_id, {}), []))
        if declared_owned_elsewhere or unclaimed_completed:
            route["segment_ownership_reconciliation"] = {
                "schema": "boise_trails_segment_ownership_reconciliation_v1",
                "status": "needs_source_repair" if unclaimed_completed else "reconciled",
                "policy": "route_gpx_extra_official_segments_are_declared_against_active_route_owners",
                "declared_owned_elsewhere_segment_ids": normalized_segment_ids(
                    [row["seg_id"] for row in declared_owned_elsewhere]
                ),
                "unclaimed_completed_segment_ids": normalized_segment_ids(
                    [row["seg_id"] for row in unclaimed_completed]
                ),
                "segments_owned_elsewhere": declared_owned_elsewhere,
                "unclaimed_completed_segments": unclaimed_completed,
                "candidate_segment_count": len(candidates),
                "review_point_count": len(review_coords),
            }
        else:
            route["segment_ownership_reconciliation"] = {
                "schema": "boise_trails_segment_ownership_reconciliation_v1",
                "status": "no_latent_official_segments",
                "declared_owned_elsewhere_segment_ids": [],
                "unclaimed_completed_segment_ids": [],
                "segments_owned_elsewhere": [],
                "unclaimed_completed_segments": [],
                "candidate_segment_count": len(candidates),
                "review_point_count": len(review_coords),
            }


def load_trailhead_access_index(path: Path = DEFAULT_TRAILHEAD_CANDIDATES) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = read_json(path)
    index: dict[str, dict[str, Any]] = {}
    for feature in data.get("features") or []:
        props = feature.get("properties") or {}
        point = feature_point(feature)
        name = props.get("facility_name") or props.get("name") or props.get("trailhead")
        open_label = trail_label_from_id_and_name(
            props.get("nearest_open_trail_id"),
            props.get("nearest_open_trail_name"),
        )
        metadata = {
            "nearest_open_trail_id": props.get("nearest_open_trail_id"),
            "nearest_open_trail_name": props.get("nearest_open_trail_name"),
            "nearest_open_trail_label": open_label,
            "nearest_open_trail_distance_miles": props.get("nearest_open_trail_distance_miles"),
            "nearest_official_trail_name": props.get("nearest_official_trail_name"),
            "nearest_official_segment_id": props.get("nearest_official_segment_id"),
            "nearest_official_distance_miles": props.get("nearest_official_distance_miles"),
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, "")}
        if not metadata:
            continue
        if name:
            index[lookup_text(name)] = metadata
        if point:
            index[f"{point[1]:.6f},{point[0]:.6f}"] = metadata
    return index


def trailhead_access_for_parking(
    parking: dict[str, Any],
    trailhead_access_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not trailhead_access_index:
        return {}
    name = parking.get("name")
    if name:
        by_name = trailhead_access_index.get(lookup_text(name))
        if by_name:
            return by_name
    if parking.get("lat") is not None and parking.get("lon") is not None:
        return trailhead_access_index.get(f"{float(parking['lat']):.6f},{float(parking['lon']):.6f}", {})
    return {}


def enrich_parking_with_trailhead_access(
    parking: dict[str, Any],
    trailhead_access_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    metadata = trailhead_access_for_parking(parking, trailhead_access_index)
    if not metadata:
        return parking
    enriched = dict(parking)
    for key, value in metadata.items():
        if enriched.get(key) in (None, ""):
            enriched[key] = value
    return enriched


def parking_for_outing(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    parking_by_candidate: dict[str, list[dict[str, Any]]],
    trailhead_access_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        trailhead = cue.get("trailhead") or {}
        if trailhead.get("lat") is not None and trailhead.get("lon") is not None:
            parking = {
                "name": trailhead.get("name") or outing.get("trailhead") or "Parking/start",
                "lon": float(trailhead["lon"]),
                "lat": float(trailhead["lat"]),
                "has_parking": trailhead.get("has_parking"),
                "has_restroom": trailhead.get("has_restroom"),
                "has_water": trailhead.get("has_water"),
                "water_confidence": trailhead.get("water_confidence"),
                "parking_confidence": trailhead.get("parking_confidence"),
                "source": trailhead.get("source"),
                "field_ready": trailhead.get("field_ready"),
                "nearest_open_trail_name": trailhead.get("nearest_open_trail_name"),
                "nearest_open_trail_label": trailhead.get("nearest_open_trail_label"),
            }
            return enrich_parking_with_trailhead_access(parking, trailhead_access_index)
        for feature in parking_by_candidate.get(str(candidate_id), []):
            point = feature_point(feature)
            if point:
                props = feature.get("properties") or {}
                parking = {
                    "name": props.get("name") or props.get("trailhead") or outing.get("trailhead") or "Parking/start",
                    "lon": point[0],
                    "lat": point[1],
                    "has_parking": props.get("has_parking"),
                    "has_restroom": props.get("has_restroom"),
                    "has_water": props.get("has_water"),
                    "water_confidence": props.get("water_confidence"),
                    "parking_confidence": props.get("parking_confidence"),
                    "source": props.get("source"),
                    "field_ready": props.get("field_ready"),
                    "nearest_open_trail_name": props.get("nearest_open_trail_name"),
                    "nearest_open_trail_label": props.get("nearest_open_trail_label"),
                }
                return enrich_parking_with_trailhead_access(parking, trailhead_access_index)
    return None


def connector_failures_for_outing(outing: dict[str, Any], route_cues: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        links = list(cue.get("between_links") or [])
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            links.append(return_to_car)
        for link in links:
            values = []
            values.extend(str(item).lower() for item in link.get("connector_classes") or [])
            values.extend(str(item).lower() for item in link.get("connector_names") or [])
            joined = " ".join(values)
            if any(token in joined for token in BLOCKED_CONNECTOR_TOKENS):
                failures.append({"code": "blocked_connector_class", "candidate_id": str(candidate_id), "value": joined})
    return failures


def link_declares_field_gap(link: dict[str, Any]) -> bool:
    if not link:
        return False
    if link.get("intentional_repark") or link.get("manual_day_of_access_hold"):
        return True
    if link.get("connector_names") or link.get("signpost_labels"):
        return True
    if link.get("connector_miles") or link.get("road_miles") or link.get("official_repeat_miles"):
        return True
    classes = {str(item) for item in link.get("connector_classes") or []}
    return bool(classes & {"r2r_trail", "osm_path_footway", "osm_public_road", "official_repeat"})


def link_declares_track_break(link: dict[str, Any]) -> bool:
    if not link:
        return False
    return bool(
        link.get("intentional_repark")
        or link.get("multi_start_boundary")
        or link.get("manual_day_of_access_hold")
    )


def declared_gap_links_for_outing(outing: dict[str, Any], route_cues: dict[str, Any]) -> list[dict[str, Any]]:
    declared = []
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        for link in cue.get("between_links") or []:
            if link_declares_track_break(link):
                declared.append(link)
        return_to_car = cue.get("return_to_car") or {}
        if link_declares_track_break(return_to_car):
            declared.append(return_to_car)
    return declared


def track_segments_for_outing(
    outing: dict[str, Any],
    routes_by_candidate: dict[str, list[dict[str, Any]]],
) -> list[list[tuple[float, float]]]:
    segments = []
    for candidate_id in outing.get("candidate_ids") or []:
        for feature in routes_by_candidate.get(str(candidate_id), []):
            segments.extend(route_parts(feature))
    return segments


def turn_waypoints(track_segments: list[list[tuple[float, float]]]) -> list[dict[str, Any]]:
    waypoints = []
    seen = set()
    for segment in track_segments:
        for index in range(1, len(segment) - 1):
            if coord_key(segment[index - 1]) == coord_key(segment[index + 1]):
                key = coord_key(segment[index])
                if key in seen:
                    continue
                seen.add(key)
                waypoints.append(
                    {
                        "name": "TURN",
                        "description": "Double-back or return point. Follow the phone card and route line.",
                        "lon": segment[index][0],
                        "lat": segment[index][1],
                    }
                )
    return waypoints


def segment_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    waypoints = []
    seen = set()
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        for segment in cue.get("segments") or []:
            segment_id = str(segment.get("seg_id"))
            if not segment_id or segment_id in seen:
                continue
            feature = official_index.get(segment_id)
            if not feature:
                continue
            parts = route_parts(feature)
            point = midpoint(parts[0]) if parts else None
            if not point:
                continue
            seen.add(segment_id)
            direction = str(segment.get("direction_rule") or "").lower()
            prefix = "ASCENT" if direction == "ascent" else "SEG"
            order = segment.get("order") or len(waypoints) + 1
            name = f"{prefix} {order} {segment.get('segment_name') or segment.get('trail_name') or segment_id}"
            description = (
                f"{segment.get('direction_cue') or 'Follow route direction.'} "
                f"Official {format_miles(segment.get('official_miles'))} mi."
            )
            effort = segment_effort_sentence(segment)
            if effort:
                description += f" {effort}."
            signpost = signpost_sentence([segment.get("trail_name")], prefix="Signpost")
            if signpost:
                description += f" {signpost}."
            waypoints.append({"name": name, "description": description.strip(), "lon": point[0], "lat": point[1]})
    return waypoints


def cue_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    waypoints = []
    last_trail_key = None
    cue_number = 1
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        for segment in cue.get("segments") or []:
            trail_name = segment.get("trail_name") or "route"
            trail_key = signpost_key(trail_name) or normalized_trail_text(trail_name).lower()
            if trail_key == last_trail_key:
                continue
            segment_id = str(segment.get("seg_id"))
            feature = official_index.get(segment_id)
            if not feature:
                continue
            parts = route_parts(feature)
            point = midpoint(parts[0]) if parts else None
            if not point:
                continue
            signpost = signpost_label(trail_name)
            cue_label = signpost or normalized_trail_text(trail_name) or "route"
            direction = str(segment.get("direction_rule") or "").lower()
            direction_note = segment.get("direction_cue") or "Follow the GPX line."
            if direction == "ascent":
                direction_note = "ASCENT REQUIRED. " + direction_note
            name = f"CUE {cue_number:02d} {cue_label}"
            signpost_note = f" Signpost: {signpost}." if signpost else ""
            description = (
                f"Follow {cue_label}. First official segment here: "
                f"{segment.get('segment_name') or trail_name}.{signpost_note} {direction_note} "
                "This is a navigation cue, not the full segment-credit audit."
            )
            waypoints.append({"name": name, "description": description.strip(), "lon": point[0], "lat": point[1]})
            last_trail_key = trail_key
            cue_number += 1
    return waypoints


def build_navigation_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
    parking: dict[str, Any] | None,
    track_segments: list[list[tuple[float, float]]],
    logistics: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    waypoints = []
    if parking:
        waypoints.append(
            {
                "name": f"PARK/START {public_display_text(parking['name'])}",
                "description": "Park here and start this outing.",
                "lon": parking["lon"],
                "lat": parking["lat"],
            }
        )
    if logistics:
        waypoints.extend(logistics_waypoints(logistics))
    waypoints.extend(cue_waypoints(outing, route_cues, official_index))
    final = last_point(track_segments)
    if final:
        waypoints.append(
            {
                "name": "RETURN TO CAR",
                "description": "Route endpoint / return-to-car point.",
                "lon": final[0],
                "lat": final[1],
            }
        )
    return waypoints


def build_audit_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
    parking: dict[str, Any] | None,
    track_segments: list[list[tuple[float, float]]],
    logistics: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    waypoints = []
    if parking:
        waypoints.append(
            {
                "name": f"PARK/START {public_display_text(parking['name'])}",
                "description": "Park here and start this outing.",
                "lon": parking["lon"],
                "lat": parking["lat"],
            }
        )
    if logistics:
        waypoints.extend(logistics_waypoints(logistics))
    waypoints.extend(turn_waypoints(track_segments))
    final = last_point(track_segments)
    if final:
        waypoints.append(
            {
                "name": "RETURN TO CAR",
                "description": "Route endpoint / return-to-car point.",
                "lon": final[0],
                "lat": final[1],
            }
        )
    waypoints.extend(segment_waypoints(outing, route_cues, official_index))
    return waypoints


def gpx_description(outing: dict[str, Any]) -> str:
    return (
        f"{outing.get('label')}. {public_display_text(outing.get('trailhead'))} | "
        f"Official {format_miles(outing.get('official_miles'))} mi; "
        f"On-foot {format_miles(outing.get('on_foot_miles'))} mi; "
        f"Door-to-door p75 {format_minutes(outing.get('total_minutes'))}; "
        f"Segments left {outing.get('remaining_segment_count')} / {len(outing.get('segment_ids') or [])}. "
        "Check current Ridge to Rivers signage and conditions before leaving."
    )


def segment_effort_sentence(segment: dict[str, Any]) -> str:
    parts = []
    if segment.get("estimated_moving_minutes"):
        parts.append(f"est. {format_minutes(segment.get('estimated_moving_minutes'))}")
    if segment.get("estimated_moving_minutes_p75"):
        parts.append(f"p75 {format_minutes(segment.get('estimated_moving_minutes_p75'))}")
    if segment.get("ascent_ft"):
        parts.append(f"{int(round(float(segment.get('ascent_ft') or 0)))} ft climb")
    return "; ".join(parts)


def render_gpx(
    name: str,
    description: str,
    track_segments: list[list[tuple[float, float]]],
    waypoints: list[dict[str, Any]],
) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="boise-trails-ai" xmlns="http://www.topografix.com/GPX/1/1">',
        "  <metadata>",
        f"    <name>{escape(name)}</name>",
        f"    <desc>{escape(description)}</desc>",
        "  </metadata>",
    ]
    for waypoint in waypoints:
        lines.extend(
            [
                f'  <wpt lat="{waypoint["lat"]:.6f}" lon="{waypoint["lon"]:.6f}">',
                f"    <name>{escape(str(waypoint['name']))}</name>",
                f"    <desc>{escape(str(waypoint.get('description') or ''))}</desc>",
                "  </wpt>",
            ]
        )
    if track_segments:
        lines.extend(["  <trk>", f"    <name>{escape(name)}</name>", f"    <desc>{escape(description)}</desc>"])
        for coords in track_segments:
            lines.append("    <trkseg>")
            for lon, lat in coords:
                lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}" />')
            lines.append("    </trkseg>")
        lines.append("  </trk>")
    lines.extend(["</gpx>", ""])
    return "\n".join(lines)


def render_gpx_readme(routes: list[dict[str, Any]], zip_href: str) -> str:
    return "\n".join(
        [
            "# Field Packet GPX Files",
            "",
            f"- `{OFFICIAL_GPX_DIR_NAME}/` is the clear location for the official user-facing GPX files.",
            "- These are the default car-to-car navigation exports for field use.",
            f"- `{CUE_GPX_DIR_NAME}/` contains cue-only GPX files for debugging.",
            f"- `{AUDIT_GPX_DIR_NAME}/` contains dense segment-audit GPX files for validation.",
            f"- `{Path(zip_href).name}` bundles all GPX flavors for backup.",
            "",
            f"Generated official GPX count: {len(routes)}.",
            "",
        ]
    )


def parking_navigation_url(parking: dict[str, Any] | None) -> str | None:
    if not parking:
        return None
    return f"https://www.google.com/maps/dir/?api=1&destination={parking['lat']:.6f},{parking['lon']:.6f}"


def validate_outing_export(
    outing: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    parking: dict[str, Any] | None,
    route_cues: dict[str, Any],
    max_gap_miles: float,
    max_parking_gap_miles: float,
) -> dict[str, Any]:
    validation = validate_track_segments(track_segments, max_gap_miles=max_gap_miles)
    failures = list(validation.get("failures") or [])
    declared_gap_links = declared_gap_links_for_outing(outing, route_cues)
    declared_gap_index = 0
    declared_gap_count = 0
    for segment_index, (left, right) in enumerate(zip(track_segments, track_segments[1:])):
        if not left or not right:
            continue
        gap = haversine_miles(left[-1], right[0])
        if gap > max_gap_miles:
            if declared_gap_index < len(declared_gap_links):
                declared_gap_index += 1
                declared_gap_count += 1
                continue
            failures.append(
                {
                    "code": "unexplained_inter_segment_gap",
                    "track_segment_index": segment_index,
                    "gap_miles": round(gap, 4),
                    "max_allowed_gap_miles": max_gap_miles,
                }
            )
    if parking:
        parking_point = (parking["lon"], parking["lat"])
        start = first_point(track_segments)
        end = last_point(track_segments)
        start_gap = haversine_miles(parking_point, start) if start else None
        end_gap = haversine_miles(parking_point, end) if end else None
        if start_gap is None or start_gap > max_parking_gap_miles:
            failures.append(
                {
                    "code": "start_not_near_parking",
                    "gap_miles": round(start_gap or 0, 4),
                    "max_allowed_gap_miles": max_parking_gap_miles,
                }
            )
        if end_gap is None or end_gap > max_parking_gap_miles:
            failures.append(
                {
                    "code": "end_not_near_parking",
                    "gap_miles": round(end_gap or 0, 4),
                    "max_allowed_gap_miles": max_parking_gap_miles,
                }
            )
    else:
        failures.append({"code": "missing_parking"})
    failures.extend(connector_failures_for_outing(outing, route_cues))
    return {
        **validation,
        "passed": not failures,
        "failures": failures,
        "max_allowed_parking_gap_miles": max_parking_gap_miles,
        "declared_inter_segment_gap_count": declared_gap_count,
    }


def assert_field_packet_certifiable(manifest: dict[str, Any]) -> None:
    failed_routes = [route for route in manifest.get("routes", []) if not route.get("validation", {}).get("passed")]
    if not failed_routes:
        return
    examples = []
    for route in failed_routes[:5]:
        failures = route.get("validation", {}).get("failures") or []
        codes = ", ".join(str(failure.get("code") or "unknown") for failure in failures[:3]) or "unknown"
        examples.append(f"{route.get('outing_id')} {route.get('label')}: {codes}")
    more = "" if len(failed_routes) <= 5 else f"; +{len(failed_routes) - 5} more"
    raise FieldPacketCertificationError(
        "Field packet is not certifiable: "
        f"{len(failed_routes)} route(s) failed export validation. "
        "Fix the canonical route source, route metadata, or GPX generation before publishing field artifacts. "
        f"Examples: {'; '.join(examples)}{more}"
    )


def html_escape(value: Any) -> str:
    return escape(str(value or ""), {'"': "&quot;"})


def summarized_names(values: list[Any], limit: int = 8) -> str:
    names = unique_nonempty_text(values)
    if not names:
        return ""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", +{len(names) - limit} more"


def unique_nonempty_text(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalized_trail_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("’", "'").strip())


def signpost_key(value: Any) -> str:
    text = normalized_trail_text(value).lower()
    text = TRAIL_NUMBER_RE.sub("", text)
    text = re.sub(r"\btrail\s*(\d+)?\b", "trail", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—,:;")
    return text


def signpost_label(value: Any) -> str:
    text = normalized_trail_text(value)
    if not text:
        return ""
    number_match = TRAIL_NUMBER_RE.search(text)
    number = number_match.group(1).upper() if number_match else SIGNPOST_TRAIL_NUMBERS.get(signpost_key(text))
    if not number:
        return ""
    name = TRAIL_NUMBER_RE.sub("", text).strip(" -–—,:;")
    if not name:
        return f"#{number}"
    return f"#{number} {name}"


def signpost_labels(values: list[Any]) -> list[str]:
    labels = []
    seen = set()
    for value in values:
        label = signpost_label(value)
        key = re.sub(r"\btrail\b", "", label.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def signpost_sentence(values: list[Any], prefix: str = "Signpost cues") -> str:
    labels = signpost_labels(values)
    if not labels:
        return ""
    return f"{prefix}: {'; '.join(labels)}"


def aggregate_logistics(
    cues: list[dict[str, Any]],
    parking: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    car_passes: list[dict[str, Any]] = []
    known_water: list[dict[str, Any]] = []
    for cue in cues:
        logistics = cue.get("logistics") or {}
        car_passes.extend(logistics.get("car_passes") or [])
        known_water.extend(logistics.get("known_water") or [])
    trailhead_names = {
        str((cue.get("trailhead") or {}).get("name"))
        for cue in cues
        if (cue.get("trailhead") or {}).get("name")
    }
    if len(cues) > 1 and len(trailhead_names) == 1 and parking:
        car_passes.insert(
            0,
            {
                "name": "Back at car between route components",
                "inter_component": True,
                "mile_from_start": None,
                "distance_to_car_miles": 0,
                "lon": parking["lon"],
                "lat": parking["lat"],
            },
        )
    if parking and parking.get("has_water") is True:
        known_water.append(
            {
                "name": public_display_text(parking.get("name") or "Parking/start"),
                "location": "parking/start",
                "confidence": parking.get("water_confidence") or "verified",
                "lon": parking["lon"],
                "lat": parking["lat"],
            }
        )
    deduped_water = []
    seen_water = set()
    for item in known_water:
        key = (
            str(item.get("name") or ""),
            str(item.get("location") or ""),
            round(float(item.get("lon") or 0), 5),
            round(float(item.get("lat") or 0), 5),
        )
        if key in seen_water:
            continue
        seen_water.add(key)
        deduped_water.append(item)
    return {"car_passes": car_passes, "known_water": deduped_water}


def car_pass_sentence(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No mid-route car pass detected."
    pieces = []
    for item in items:
        if item.get("inter_component"):
            pieces.append("Back at car between route components.")
        else:
            pieces.append(f"Pass by car again near mile {format_miles(item.get('mile_from_start'))}.")
    return " ".join(pieces)


def water_sentence(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    return "; ".join(
        f"{item.get('name') or 'Water'} · {item.get('location') or 'location'} · {item.get('confidence') or 'verified'}"
        for item in items
    )


def logistics_section_html(logistics: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    car_passes = logistics.get("car_passes") or []
    known_water = logistics.get("known_water") or []
    if car_passes:
        lines.append(f"<b>Car:</b> {html_escape(car_pass_sentence(car_passes))}")
    if known_water:
        lines.append(f"<b>Known water:</b> {html_escape(water_sentence(known_water))}")
    if not lines:
        return ""
    return f"<section><h3>Field logistics</h3><p>{'<br>'.join(lines)}</p></section>"


def logistics_waypoints(logistics: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    waypoints = []
    for index, car_pass in enumerate(logistics.get("car_passes") or [], start=1):
        if car_pass.get("lon") is None or car_pass.get("lat") is None:
            continue
        description = (
            "Back at the parked car between route components."
            if car_pass.get("inter_component")
            else f"Pass by the parked car again near mile {format_miles(car_pass.get('mile_from_start'))}."
        )
        waypoints.append(
            {
                "name": f"CAR PASS {index}",
                "description": description,
                "lon": float(car_pass["lon"]),
                "lat": float(car_pass["lat"]),
            }
        )
    for water in logistics.get("known_water") or []:
        if water.get("lon") is None or water.get("lat") is None:
            continue
        waypoints.append(
            {
                "name": f"WATER {water.get('name') or 'Water'}",
                "description": (
                    f"Known water: {water.get('name') or 'Water'}; "
                    f"{water.get('location') or 'location'}; {water.get('confidence') or 'verified'}."
                ),
                "lon": float(water["lon"]),
                "lat": float(water["lat"]),
            }
        )
    return waypoints


def manual_signpost_notes(route: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for entry in field_route_hints().get("route_signpost_notes") or []:
        if route_hint_matches(route, entry.get("match") or {}):
            notes.extend(str(note) for note in entry.get("notes") or [] if note)
    return notes


def connector_detail(link: dict[str, Any]) -> str:
    pieces = []
    distance = link.get("distance_miles")
    raw_connector_names = unique_nonempty_text([str(name) for name in link.get("connector_names") or []])
    connector_names = summarized_names(raw_connector_names)
    distance_miles = float(distance or 0)
    if len(raw_connector_names) > 2 and distance_miles:
        pieces.append(f"Use the named connector trail/road for {format_miles(distance)} mi")
    elif connector_names and distance_miles:
        pieces.append(f"Use {connector_names} for {format_miles(distance)} mi")
    elif connector_names:
        pieces.append(f"Use {connector_names}")
    elif distance_miles > 0.02:
        pieces.append(f"Short link: {format_miles(distance)} mi")
    signpost = signpost_sentence(
        link.get("connector_names") or [link.get("from_trail"), link.get("to_trail")],
        prefix="Look for signs",
    )
    if signpost:
        pieces.append(signpost)
    return ". ".join(pieces) + ("." if pieces else "")


def display_trail(value: Any) -> str:
    return signpost_label(value) or normalized_trail_text(value) or "the next trail"


def plain_trail(value: Any) -> str:
    return normalized_trail_text(value) or "the previous trail"


def sentence_list(values: list[str]) -> str:
    clean = unique_nonempty_text(values)
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def compact_number_list(numbers: list[str]) -> str:
    if not numbers:
        return ""
    if len(numbers) > 2 and all(number.isdigit() for number in numbers):
        ints = [int(number) for number in numbers]
        if ints == list(range(ints[0], ints[-1] + 1)):
            return f"{ints[0]}-{ints[-1]}"
    return sentence_list(numbers)


def segment_credit_label(segments: list[dict[str, Any]]) -> str:
    names = [str(segment.get("segment_name") or segment.get("trail_name") or "") for segment in segments]
    trail_names = unique_nonempty_text([str(segment.get("trail_name") or "") for segment in segments])
    if len(trail_names) == 1:
        trail_name = trail_names[0]
        if segments and not any(segment.get("segment_name") for segment in segments):
            segment_numbers = [str(segment.get("seg_id") or "") for segment in segments if segment.get("seg_id") is not None]
            if segment_numbers:
                segment_numbers = sorted(segment_numbers, key=lambda value: int(value) if value.isdigit() else value)
                if len(segment_numbers) == 1:
                    return f"{trail_name} official segment {segment_numbers[0]}"
                return f"{trail_name} official segments {compact_number_list(segment_numbers)}"
        numbers = []
        for name in names:
            match = re.search(r"(\d+)$", name.strip())
            if not match:
                break
            base = name[: match.start()].strip()
            if normalized_trail_text(base) != normalized_trail_text(trail_name):
                break
            numbers.append(match.group(1))
        else:
            if numbers:
                if len(numbers) == 1:
                    return f"{trail_name} segment {compact_number_list(numbers)}"
                if len(numbers) == 2:
                    return f"both {trail_name} official segments"
                return f"{trail_name} segments {compact_number_list(numbers)}"
    return sentence_list(names)


def trail_groups(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for segment in segments:
        trail_name = segment.get("trail_name") or "official trail"
        if groups and groups[-1]["trail_name"] == trail_name:
            groups[-1]["segments"].append(segment)
        else:
            groups.append({"trail_name": trail_name, "segments": [segment]})
    return groups


def segment_completion_sentence(
    segments: list[dict[str, Any]],
    *,
    final_group: bool,
    claimed_segment_ids: set[str] | None = None,
) -> str:
    claimed_segments = list(segments)
    repeat_segments: list[dict[str, Any]] = []
    if claimed_segment_ids is not None:
        claimed_segments = [
            segment
            for segment in segments
            if str(segment.get("seg_id") or "") in claimed_segment_ids
        ]
        repeat_segments = [
            segment
            for segment in segments
            if str(segment.get("seg_id") or "") not in claimed_segment_ids
        ]
    parts: list[str] = []
    credit = segment_credit_label(claimed_segments)
    if credit:
        parts.append(f"This earns: {credit}.")
    if repeat_segments:
        repeat = segment_credit_label(repeat_segments)
        if repeat:
            parts.append(f"Official-repeat mileage: {repeat}; do not count as new credit.")
    return " ".join(parts)


def traversal_for_segment(segment: dict[str, Any]) -> str:
    direction_cue = str(segment.get("direction_cue") or "").lower()
    if "opposite official geometry" in direction_cue:
        return "reverse"
    return "forward"


def load_segment_elevation_index(path: Path = DEFAULT_SEGMENT_ELEVATION_JSON) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}
    data = read_json(path)
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for row in data.get("rows") or []:
        segment_id = str(row.get("seg_id") or "")
        traversal = str(row.get("traversal") or "")
        if not segment_id or traversal not in {"forward", "reverse"}:
            continue
        index.setdefault(segment_id, {})[traversal] = row
    return index


def enrich_segment_with_elevation(
    segment: dict[str, Any],
    elevation_index: dict[str, dict[str, dict[str, Any]]],
) -> None:
    segment_id = str(segment.get("seg_id") or "")
    options = elevation_index.get(segment_id) or {}
    if not options:
        return
    traversal = traversal_for_segment(segment)
    selected = options.get(traversal) or {}
    opposite = options.get("reverse" if traversal == "forward" else "forward") or {}
    if selected:
        for key in ("ascent_ft", "descent_ft", "grade_adjusted_miles"):
            if segment.get(key) is None:
                segment[key] = selected.get(key)
        segment.setdefault("elevation_source", "dem")
        segment["elevation_traversal"] = traversal
    if opposite:
        segment["opposite_direction_ascent_ft"] = opposite.get("ascent_ft")
        segment["opposite_direction_descent_ft"] = opposite.get("descent_ft")
        segment["opposite_direction_grade_adjusted_miles"] = opposite.get("grade_adjusted_miles")


def enrich_route_cues_with_segment_elevation(
    route_cues: list[dict[str, Any]],
    elevation_index: dict[str, dict[str, dict[str, Any]]],
) -> None:
    if not elevation_index:
        return
    for cue in route_cues:
        for segment in cue.get("segments") or []:
            enrich_segment_with_elevation(segment, elevation_index)


def group_effort_sentence(segments: list[dict[str, Any]]) -> str:
    official = sum(float(segment.get("official_miles") or 0) for segment in segments)
    ascent = sum(float(segment.get("ascent_ft") or 0) for segment in segments)
    descent = sum(float(segment.get("descent_ft") or 0) for segment in segments)
    minutes = sum(float(segment.get("estimated_moving_minutes_p75") or segment.get("estimated_moving_minutes") or 0) for segment in segments)
    parts = []
    if official:
        parts.append(f"{format_miles(official)} official mi")
    if minutes:
        parts.append(f"~{round(minutes)} min moving")
    if ascent:
        parts.append(f"{round(ascent)} ft climb")
    if descent and descent >= max(200, ascent * 1.5):
        parts.append(f"{round(descent)} ft descent")
    if not parts:
        return ""
    return "Section estimate: " + ", ".join(parts)


def grade_asymmetry_warning_sentence(segments: list[dict[str, Any]]) -> str:
    official = sum(float(segment.get("official_miles") or 0) for segment in segments)
    ascent = sum(float(segment.get("ascent_ft") or 0) for segment in segments)
    opposite_ascent = sum(float(segment.get("opposite_direction_ascent_ft") or 0) for segment in segments)
    if official <= 0 or opposite_ascent < 350 or opposite_ascent < ascent + 200:
        return ""
    return f"Reverse direction would be steep: about {round(opposite_ascent)} ft climb over {format_miles(official)} mi."


def ascent_warning_sentence(segments: list[dict[str, Any]]) -> str:
    ascent_segments = [
        str(segment.get("segment_name") or segment.get("trail_name") or "")
        for segment in segments
        if str(segment.get("direction_rule") or "").lower() == "ascent"
    ]
    if not ascent_segments:
        return ""
    return f"ASCENT REQUIRED on {sentence_list(ascent_segments)}."


def link_for_group_transition(links: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return links[index] if index < len(links) else {}


def turn_title(phrase: str | None, trail_label: str) -> str:
    if not phrase:
        return f"Turn onto {trail_label}"
    if phrase == "continue straight":
        return f"Continue onto {trail_label}"
    return f"{phrase.capitalize()} onto {trail_label}"


def turn_detail_action(phrase: str | None, trail_label: str) -> str:
    if not phrase:
        return f"turn onto {trail_label}"
    if phrase == "continue straight":
        return f"continue straight onto {trail_label}"
    return f"{phrase} onto {trail_label}"


def trail_navigation_steps_for_cue(
    cue: dict[str, Any],
    parking: dict[str, Any],
    track_segments: list[list[tuple[float, float]]] | None = None,
    official_index: dict[str, dict[str, Any]] | None = None,
    claimed_segment_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    segments = cue.get("segments") or []
    groups = trail_groups(segments)
    if not groups:
        return []

    steps: list[dict[str, str]] = []
    links = list(cue.get("between_links") or [])
    trailhead_name = public_display_text(parking.get("name") or (cue.get("trailhead") or {}).get("name") or "the car")

    for index, group in enumerate(groups):
        trail_name = group["trail_name"]
        trail_label = display_trail(trail_name)
        is_first = index == 0
        is_last = index == len(groups) - 1
        next_group = groups[index + 1] if not is_last else None
        prior_group = groups[index - 1] if not is_first else None
        detail_parts: list[str] = []

        if is_first:
            title = f"Take {trail_label}"
            detail_parts.append(f"Follow {trail_label}.")
        else:
            link = link_for_group_transition(links, index - 1)
            from_trail = link.get("from_trail") or (prior_group or {}).get("trail_name") or "previous trail"
            to_trail = link.get("to_trail") or trail_name
            from_plain = plain_trail(from_trail)
            to_plain = plain_trail(to_trail)
            to_label = display_trail(to_trail)
            phrase = turn_phrase_for_transition(prior_group or {}, group, track_segments, official_index)
            title = turn_title(phrase, to_label)
            action = turn_detail_action(phrase, to_label)
            detail_parts.append(f"At the signed junction with {to_plain}, {action}.")
            link_detail = connector_detail(link)
            if link_detail:
                detail_parts.append(link_detail)

        completion = segment_completion_sentence(
            group["segments"],
            final_group=is_last,
            claimed_segment_ids=claimed_segment_ids,
        )
        if completion:
            detail_parts.append(completion)
        effort = group_effort_sentence(group["segments"])
        if effort:
            detail_parts.append(effort + ".")
        grade_warning = grade_asymmetry_warning_sentence(group["segments"])
        if grade_warning:
            detail_parts.append(grade_warning)
        ascent_warning = ascent_warning_sentence(group["segments"])
        if ascent_warning:
            detail_parts.append(ascent_warning)
        if next_group:
            next_label = display_trail(next_group["trail_name"])
            detail_parts.append(f"Next decision: look for signs to {next_label}.")

        steps.append({"kind": "navigate", "title": title, "detail": " ".join(detail_parts)})

    return steps


def access_navigation_step(
    parking: dict[str, Any],
    first_trail: Any,
    access: dict[str, Any],
    start_access_gap_miles: float = 0.0,
) -> dict[str, str]:
    first_trail_label = display_trail(first_trail)
    first_key = signpost_key(first_trail) or lookup_text(first_trail)
    hint = access_hint_for(parking, first_trail)
    if hint:
        return {
            "kind": "access",
            "title": hint["title"],
            "detail": hint["detail"],
        }

    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    if start_label and signpost_key(start_label) != first_key:
        access_miles = max(float(access.get("mapped_access_miles") or 0), float(start_access_gap_miles or 0))
        detail = (
            f"From the car, take {start_label} toward {first_trail_label}. "
            "This access trail/road is part of the route even though it is not official challenge credit."
        )
        if access_miles > 0.05:
            detail += f" Follow the GPX access line for about {format_miles(access_miles)} mi."
        first_signpost = signpost_sentence([start_label, first_trail], prefix="Watch for signs")
        if first_signpost:
            detail += " " + first_signpost + "."
        return {
            "kind": "access",
            "title": f"Start on {start_label}",
            "detail": detail,
        }

    access_detail = f"From the car, head toward {first_trail_label}."
    access_miles = max(float(access.get("mapped_access_miles") or 0), float(start_access_gap_miles or 0))
    if access_miles > 0.05:
        access_detail += (
            f" Follow the GPX access line for about {format_miles(access_miles)} mi before official credit starts."
        )
    first_signpost = signpost_sentence([first_trail], prefix="Look for signs")
    if first_signpost:
        access_detail += " " + first_signpost + "."
    return {
        "kind": "access",
        "title": f"Leave car toward {first_trail_label}",
        "detail": access_detail,
    }


def return_navigation_step(
    parking: dict[str, Any],
    last_trail: Any,
    return_to_car: dict[str, Any],
    return_access_gap_miles: float = 0.0,
) -> dict[str, str]:
    last_key = signpost_key(last_trail) or lookup_text(last_trail)
    last_label = display_trail(last_trail)
    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    return_names = summarized_names(return_to_car.get("connector_names") or [])
    access_miles = max(
        float(return_access_gap_miles or 0),
        float(return_to_car.get("connector_miles") or 0),
        float(return_to_car.get("road_miles") or 0),
    )

    if access_miles > 0.05 or return_names:
        if return_names:
            title = f"Return via {return_names}"
        elif start_label and signpost_key(start_label) != last_key:
            title = f"Return via {start_label}"
        else:
            title = "Follow connector/access back to car"
        parking_name = public_display_text(parking.get("name") or "the parking point")
        detail = f"After {last_label}, you are not fully back at the car. "
        if return_names:
            detail += f"Use {return_names} back toward {parking_name}."
        elif start_label and signpost_key(start_label) != last_key:
            detail += f"Follow the signed connector/access back toward {start_label} and descend to {parking_name}."
        else:
            detail += f"Follow the GPX connector/access line back to {parking_name}."
        if access_miles > 0.05:
            detail += f" Expect about {format_miles(access_miles)} mi after the final official-credit segment."
        detail += " This return leg is part of the route even though it is not official challenge credit."
        return {"kind": "return", "title": title, "detail": detail}

    return_names = summarized_names(return_to_car.get("connector_names") or [])
    detail = return_to_car.get("description") or "Follow the GPX line back to the car."
    if "geometry tolerance" in detail:
        detail = "You should be back at the parking point."
    if return_names:
        detail += f" Return via {return_names}."
    return_parts = []
    if float(return_to_car.get("official_repeat_miles") or 0):
        return_parts.append(f"{format_miles(return_to_car.get('official_repeat_miles'))} mi repeat official")
    if float(return_to_car.get("connector_miles") or 0):
        return_parts.append(f"{format_miles(return_to_car.get('connector_miles'))} mi connector")
    if float(return_to_car.get("road_miles") or 0):
        return_parts.append(f"{format_miles(return_to_car.get('road_miles'))} mi road")
    if return_parts:
        detail += f" Return leg: {sentence_list(return_parts)}."
    return {"kind": "return", "title": "Return to car", "detail": detail}


WAYFINDING_TYPE_LABELS = {
    "start_access": "START/ACCESS",
    "official_segment_start": "OFFICIAL START",
    "follow_official_segment": "FOLLOW",
    "junction_turn": "JCT",
    "connector_named_trail": "CONNECTOR",
    "connector_road": "ROAD",
    "overlap_repeat": "OVERLAP",
    "repeat_official_noncredit": "REPEAT",
    "exit_access": "EXIT",
    "return_to_car": "RETURN",
    "manual_field_check": "HOLD",
}


def cue_type_label(cue_type: str) -> str:
    return WAYFINDING_TYPE_LABELS.get(str(cue_type or ""), str(cue_type or "CUE").upper())


def cue_signed_text(values: list[Any]) -> str:
    return " / ".join(unique_nonempty_text(values))


GENERIC_OSM_CONNECTOR_RE = re.compile(r"^OSM\s+(?:path|footway|road|track)?\s*connector\s+\d+$", re.IGNORECASE)


def field_visible_link_names(link: dict[str, Any]) -> list[str]:
    signposts = unique_nonempty_text(link.get("signpost_labels") or [])
    connector_names = unique_nonempty_text(link.get("connector_names") or [])
    if signposts:
        return signposts
    return [
        name
        for name in connector_names
        if not GENERIC_OSM_CONNECTOR_RE.match(str(name).strip())
    ] or connector_names


def wayfinding_compact(cue: dict[str, Any]) -> str:
    signed = cue_signed_text(cue.get("signed_as") or [])
    label = cue_type_label(str(cue.get("cue_type") or "cue"))
    action = str(cue.get("action") or "FOLLOW").upper()
    show_action = "" if action == label else action
    if signed and show_action:
        signed_part = f" {show_action} {signed}"
    elif signed:
        signed_part = f" {signed}"
    elif show_action:
        signed_part = f" {show_action}"
    else:
        signed_part = ""
    until = f" UNTIL {cue.get('until')}" if cue.get("until") else ""
    return (
        f"{int(cue.get('seq') or 0):02d} {float(cue.get('cum_miles') or 0):.2f} mi "
        f"(+{float(cue.get('leg_miles') or 0):.2f}) {label}"
        f"{signed_part}{until}."
    )


def wayfinding_display_detail(cue: dict[str, Any]) -> str:
    signed = cue_signed_text(cue.get("signed_as") or [])
    action = str(cue.get("action") or "FOLLOW").title()
    target = cue.get("target")
    until = cue.get("until")
    parts = []
    if signed and target:
        parts.append(f"{action} {signed} toward {target}.")
    elif signed:
        parts.append(f"{action} {signed}.")
    elif target:
        parts.append(f"{action} toward {target}.")
    else:
        parts.append(action + ".")
    if until:
        parts.append(f"UNTIL {until}.")
    verify = cue.get("verify")
    if verify:
        parts.append(f"VERIFY: {verify}.")
    avoid = cue.get("avoid") or cue.get("negative_cues") or []
    if avoid:
        parts.append(f"IGNORE: {sentence_list(avoid)}.")
    note = cue.get("note")
    if note:
        parts.append(str(note))
    return " ".join(parts)


def make_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    leg_miles: float,
    cue_type: str,
    action: str,
    signed_as: list[Any] | None = None,
    target: Any = None,
    until: Any = None,
    verify: Any = None,
    avoid: list[Any] | None = None,
    confidence: str = "planner",
    note: Any = None,
    field_warning: Any = None,
    official_segment_ids: list[Any] | None = None,
    official_repeat_segment_ids: list[Any] | None = None,
    official_repeat_miles: Any = None,
) -> dict[str, Any]:
    repeat_segment_ids = normalized_segment_ids(official_repeat_segment_ids or [])
    repeat_miles = round(float(official_repeat_miles or 0), 2)
    if repeat_segment_ids and repeat_miles <= 0:
        repeat_miles = 0.01
    cue = {
        "seq": seq,
        "cum_miles": round(float(cum_miles or 0), 2),
        "leg_miles": round(float(leg_miles or 0), 2),
        "cue_type": cue_type,
        "action": action,
        "signed_as": unique_nonempty_text(signed_as or []),
        "target": normalized_trail_text(target),
        "until": normalized_trail_text(until),
        "verify": normalized_trail_text(verify),
        "avoid": unique_nonempty_text(avoid or []),
        "confidence": confidence,
        "note": normalized_trail_text(note),
        "field_warning": normalized_trail_text(field_warning),
        "official_segment_ids": normalized_segment_ids(official_segment_ids or []),
        "official_repeat_segment_ids": repeat_segment_ids,
        "official_repeat_miles": repeat_miles,
    }
    cue = {key: value for key, value in cue.items() if value not in (None, "", [], 0, 0.0)}
    cue["compact"] = wayfinding_compact(cue)
    cue["display_detail"] = wayfinding_display_detail(cue)
    return cue


def non_credit_repeat_note(
    prefix: str,
    official_repeat_miles: Any,
    official_repeat_segment_ids: list[Any] | None = None,
) -> str:
    miles = float(official_repeat_miles or 0)
    repeat_ids = normalized_segment_ids(official_repeat_segment_ids or [])
    if miles <= 0 and not repeat_ids:
        return prefix
    suffix = (
        f"Includes {format_miles(miles)} mi repeat official; no new credit."
        if miles > 0
        else "Includes repeat official mileage that rounds to 0.00 mi; no new credit."
    )
    return f"{prefix.strip()} {suffix}".strip()


def sort_segment_id(segment_id: str) -> tuple[int, str]:
    text = str(segment_id)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def ordered_segment_ids(values: list[Any] | tuple[Any, ...] | set[Any]) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for segment_id in sorted(normalized_segment_ids(list(values)), key=sort_segment_id):
        if segment_id not in seen:
            ids.append(segment_id)
            seen.add(segment_id)
    return ids


def official_feature_miles(feature: dict[str, Any] | None) -> float:
    if not feature:
        return 0.0
    props = feature.get("properties") or {}
    for key in (
        "official_miles",
        "segment_official_miles",
        "LengthMi",
        "length_miles",
        "distance_miles",
    ):
        try:
            value = float(props.get(key) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    try:
        length_ft = float(props.get("LengthFt") or 0)
    except (TypeError, ValueError):
        length_ft = 0.0
    if length_ft > 0:
        return length_ft / 5280
    total = 0.0
    for part in route_parts(feature):
        for index in range(1, len(part)):
            total += haversine_miles(part[index - 1], part[index])
    return total


def non_credit_repeat_prefix_for_cue(cue: dict[str, Any]) -> str:
    cue_type = str(cue.get("cue_type") or "")
    if cue_type in {"start_access", "official_segment_start"}:
        return "This access leg is not official challenge credit."
    if cue_type in {"exit_access", "return_to_car"}:
        return "Return leg does not count as new official challenge credit."
    if cue_type in {"connector_named_trail", "connector_road"}:
        return "Connector mileage does not count as new official challenge credit."
    if cue_type in {"repeat_official_noncredit", "overlap_repeat"}:
        return "Repeat official mileage does not count as new official challenge credit."
    return str(cue.get("note") or "").strip()


def full_claimed_segments_in_cue_interval(
    cue: dict[str, Any],
    *,
    claimed_segment_ids: set[str],
    official_index: dict[str, dict[str, Any]],
    route_points: list[dict[str, Any]],
    elevation_sampler: Any = None,
    threshold_miles: float = 0.045,
    max_activity_points: int = 600,
) -> set[str]:
    interval = cue_route_interval(cue)
    if not interval or not claimed_segment_ids:
        return set()
    interval_coords = route_interval_coordinates(
        route_points,
        start_mile=interval[0],
        end_mile=interval[1],
    )
    if len(interval_coords) < 2:
        return set()
    candidate_index = {
        segment_id: official_index[segment_id]
        for segment_id in claimed_segment_ids
        if segment_id in official_index
    }
    official_segments = official_segments_for_review(candidate_index)
    if not official_segments:
        return set()
    review = review_activity_against_segments(
        downsample_coords(interval_coords, max_points=max_activity_points),
        official_segments,
        planned_segment_ids=claimed_segment_ids,
        threshold_miles=threshold_miles,
        min_fraction=0.85,
        partial_min_fraction=0.2,
        elevation_sampler=elevation_sampler,
    )
    return set(normalized_segment_ids(review.get("completed_segment_ids") or [])) & claimed_segment_ids


def apply_non_credit_claimed_repeat_declarations(
    route: dict[str, Any],
    *,
    elevation_sampler: Any = None,
) -> dict[str, Any]:
    """Declare official segments fully rerun inside access/connector/return cues.

    The route card may claim a segment once for challenge credit and then use
    the same segment later as exit/access mileage. That has to be visible in
    repeat accounting; otherwise effort looks cheaper than it is.
    """

    cues = route.get("wayfinding_cues") or []
    official_index = route.get("_official_segment_index") or {}
    route_points = cumulative_track_points(route.get("_track_segments") or [])
    outing = route.get("outing") or {}
    claimed_segment_ids = set(
        normalized_segment_ids(
            route.get("segment_ids")
            or outing.get("remaining_segment_ids")
            or outing.get("segment_ids")
            or []
        )
    )
    if not cues or not official_index or len(route_points) < 2 or not claimed_segment_ids:
        return route
    for cue in cues:
        if str(cue.get("cue_type") or "") not in NON_CREDIT_WAYFINDING_CUE_TYPES:
            continue
        existing_ids = set(normalized_segment_ids(cue.get("official_repeat_segment_ids") or []))
        completed_claimed_ids = full_claimed_segments_in_cue_interval(
            cue,
            claimed_segment_ids=claimed_segment_ids,
            official_index=official_index,
            route_points=route_points,
            elevation_sampler=elevation_sampler,
        )
        added_ids = completed_claimed_ids - existing_ids
        if not existing_ids and not added_ids:
            continue
        repeat_miles = float(cue.get("official_repeat_miles") or 0)
        for segment_id in added_ids:
            repeat_miles += official_feature_miles(official_index.get(segment_id))
        all_repeat_ids = ordered_segment_ids(existing_ids | added_ids)
        if all_repeat_ids and repeat_miles <= 0:
            repeat_miles = 0.01
        cue["official_repeat_segment_ids"] = all_repeat_ids
        cue["official_repeat_miles"] = round(max(repeat_miles, 0.01 if all_repeat_ids else 0), 2)
        cue["note"] = non_credit_repeat_note(
            non_credit_repeat_prefix_for_cue(cue),
            cue.get("official_repeat_miles"),
            all_repeat_ids,
        )
        refresh_wayfinding_text(cue)
    return route


def access_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    parking: dict[str, Any],
    first_trail: Any,
    access: dict[str, Any],
    start_access_gap_miles: float,
) -> dict[str, Any]:
    first_trail_label = display_trail(first_trail)
    first_key = signpost_key(first_trail) or lookup_text(first_trail)
    hint = access_hint_for(parking, first_trail)
    access_miles = max(float(access.get("mapped_access_miles") or 0), float(start_access_gap_miles or 0))
    official_repeat_miles = float(access.get("official_repeat_miles") or 0)
    official_repeat_segment_ids = access.get("official_repeat_segment_ids") or []
    if hint:
        return make_wayfinding_cue(
            seq=seq,
            cum_miles=cum_miles,
            leg_miles=access_miles,
            cue_type="start_access",
            action="FOLLOW",
            signed_as=hint.get("signed_as") or [],
            target=hint.get("target") or first_trail_label,
            until=hint.get("until") or f"signed junction with {first_trail_label}",
            verify=f"watch for signs: {cue_signed_text(hint.get('signed_as') or [])}",
            avoid=hint.get("avoid") or [],
            confidence="field_check_needed",
            note=non_credit_repeat_note(
                "This access leg is not official challenge credit.",
                official_repeat_miles,
                official_repeat_segment_ids,
            ),
            official_repeat_segment_ids=official_repeat_segment_ids,
            official_repeat_miles=official_repeat_miles,
        )

    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    if start_label and signpost_key(start_label) != first_key:
        return make_wayfinding_cue(
            seq=seq,
            cum_miles=cum_miles,
            leg_miles=access_miles,
            cue_type="start_access",
            action="FOLLOW",
            signed_as=[start_label],
            target=first_trail_label,
            until=f"signed junction with {first_trail_label}",
            verify=signpost_sentence([start_label], prefix="watch for signs").replace(".", ""),
            confidence="planner",
            note=non_credit_repeat_note(
                "This access leg is not official challenge credit.",
                official_repeat_miles,
                official_repeat_segment_ids,
            ),
            official_repeat_segment_ids=official_repeat_segment_ids,
            official_repeat_miles=official_repeat_miles,
        )

    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=access_miles,
        cue_type="official_segment_start" if access_miles <= 0.05 else "start_access",
        action="FOLLOW",
        signed_as=[first_trail_label],
        target=first_trail_label,
        until=f"signed {first_trail_label} route / first official segment",
        verify=signpost_sentence([first_trail], prefix="watch for signs").replace(".", ""),
        confidence="planner" if access_miles <= 0.05 else "field_check_needed",
        note=non_credit_repeat_note("", official_repeat_miles, official_repeat_segment_ids),
        official_repeat_segment_ids=official_repeat_segment_ids,
        official_repeat_miles=official_repeat_miles,
    )


def link_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    link: dict[str, Any],
    next_trail: Any,
) -> dict[str, Any] | None:
    names = field_visible_link_names(link)
    distance = float(link.get("distance_miles") or link.get("connector_miles") or link.get("road_miles") or 0)
    if not names and distance <= 0.01:
        return None
    classes = {str(item) for item in link.get("connector_classes") or []}
    cue_type = "connector_road" if "osm_public_road" in classes or float(link.get("road_miles") or 0) else "connector_named_trail"
    official_repeat_miles = float(link.get("official_repeat_miles") or 0)
    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=distance,
        cue_type=cue_type,
        action="FOLLOW",
        signed_as=names or ["connector/access"],
        target=display_trail(next_trail),
        until=f"signed junction with {display_trail(next_trail)}",
        verify=signpost_sentence(names, prefix="watch for signs").replace(".", "") if names else "",
        confidence="planner" if names else "field_check_needed",
        note=non_credit_repeat_note(
            "Connector mileage does not count as new official challenge credit.",
            official_repeat_miles,
            link.get("official_repeat_segment_ids") or [],
        ),
        official_repeat_segment_ids=link.get("official_repeat_segment_ids") or [],
        official_repeat_miles=official_repeat_miles,
    )


def official_group_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    group: dict[str, Any],
    next_group: dict[str, Any] | None,
    first_group: bool,
    phrase: str | None = None,
    claimed_segment_ids: set[str] | None = None,
) -> dict[str, Any]:
    trail_label = display_trail(group["trail_name"])
    segments = group.get("segments") or []
    official_miles = sum(float(segment.get("official_miles") or 0) for segment in segments)
    segment_ids = [
        segment.get("seg_id")
        for segment in segments
        if claimed_segment_ids is None or str(segment.get("seg_id") or "") in claimed_segment_ids
    ]
    if next_group:
        until = f"signed junction with {display_trail(next_group['trail_name'])}"
        target = display_trail(next_group["trail_name"])
    else:
        until = f"end of {trail_label} for this route"
        target = "return to car"
    if not segment_ids:
        cue_type = "repeat_official_noncredit"
        action = "FOLLOW" if first_group else (phrase or "FOLLOW").upper()
    elif first_group:
        cue_type = "follow_official_segment"
        action = "FOLLOW"
    else:
        cue_type = "junction_turn"
        action = (phrase or "TAKE").upper()
    effort_note = group_effort_sentence(segments)
    if effort_note and not effort_note.endswith((".", "!", "?")):
        effort_note += "."
    grade_warning = grade_asymmetry_warning_sentence(segments)
    notes = [
        segment_completion_sentence(
            segments,
            final_group=not next_group,
            claimed_segment_ids=claimed_segment_ids,
        ),
        effort_note,
        grade_warning,
        ascent_warning_sentence(segments),
    ]
    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=official_miles,
        cue_type=cue_type,
        action=action,
        signed_as=[trail_label],
        target=target,
        until=until,
        verify=signpost_sentence([trail_label], prefix="watch for signs").replace(".", ""),
        confidence="planner",
        note=" ".join(note for note in notes if note),
        field_warning=grade_warning,
        official_segment_ids=segment_ids,
    )


def return_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    parking: dict[str, Any],
    last_trail: Any,
    return_to_car: dict[str, Any],
    return_access_gap_miles: float,
) -> dict[str, Any]:
    return_names = unique_nonempty_text(return_to_car.get("connector_names") or [])
    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    signed_as = return_names or ([start_label] if start_label else [display_trail(last_trail)])
    access_miles = max(
        float(return_access_gap_miles or 0),
        float(return_to_car.get("connector_miles") or 0),
        float(return_to_car.get("road_miles") or 0),
        float(return_to_car.get("official_repeat_miles") or 0),
    )
    official_repeat_miles = float(return_to_car.get("official_repeat_miles") or 0)
    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=access_miles,
        cue_type="return_to_car" if access_miles <= 0.05 else "exit_access",
        action="FOLLOW",
        signed_as=signed_as,
        target=public_display_text(parking.get("name") or "parked car"),
        until="parked car / trailhead",
        verify=f"finish at {public_display_text(parking.get('name') or 'the parked car')}",
        confidence="planner" if return_names or start_label else "field_check_needed",
        note=non_credit_repeat_note(
            "Return leg does not count as new official challenge credit." if access_miles > 0.05 else "",
            official_repeat_miles,
            return_to_car.get("official_repeat_segment_ids") or [],
        ),
        official_repeat_segment_ids=return_to_car.get("official_repeat_segment_ids") or [],
        official_repeat_miles=official_repeat_miles,
    )


def refresh_wayfinding_text(cue: dict[str, Any]) -> None:
    cue["compact"] = wayfinding_compact(cue)
    cue["display_detail"] = wayfinding_display_detail(cue)


def add_cue_note(cue: dict[str, Any], note: str) -> None:
    existing = str(cue.get("note") or "").strip()
    if note not in existing:
        cue["note"] = " ".join(part for part in [existing, note] if part)


def add_cue_avoid(cue: dict[str, Any], avoid: str) -> None:
    cue["avoid"] = unique_nonempty_text((cue.get("avoid") or []) + [avoid])


def route_card_total_miles(route: dict[str, Any], cues: list[dict[str, Any]]) -> float:
    outing = route.get("outing") or {}
    candidates = [
        outing.get("on_foot_miles"),
        route.get("on_foot_miles"),
        max(
            (
                float(cue.get("cum_miles") or 0) + float(cue.get("leg_miles") or 0)
                for cue in cues
            ),
            default=0.0,
        ),
    ]
    for candidate in candidates:
        value = float(candidate or 0)
        if value > 0:
            return value
    return 0.0


def card_miles_to_track_miles(card_miles: float, card_total_miles: float, track_total_miles: float) -> float:
    if card_total_miles <= 0 or track_total_miles <= 0:
        return max(0.0, float(card_miles or 0))
    return max(0.0, min(float(card_miles or 0) * track_total_miles / card_total_miles, track_total_miles))


def cue_start_track_mile(cue: dict[str, Any], card_total_miles: float, track_total_miles: float) -> float:
    if cue.get("route_miles") is not None:
        return max(0.0, min(float(cue.get("route_miles") or 0), track_total_miles))
    return card_miles_to_track_miles(float(cue.get("cum_miles") or 0), card_total_miles, track_total_miles)


def cue_end_track_mile(
    cues: list[dict[str, Any]],
    index: int,
    card_total_miles: float,
    track_total_miles: float,
) -> float:
    cue = cues[index]
    if index + 1 < len(cues):
        next_cue = cues[index + 1]
        if next_cue.get("route_miles") is not None:
            return max(0.0, min(float(next_cue.get("route_miles") or 0), track_total_miles))
        return card_miles_to_track_miles(float(next_cue.get("cum_miles") or 0), card_total_miles, track_total_miles)
    start = cue_start_track_mile(cue, card_total_miles, track_total_miles)
    if cue.get("route_leg_miles") is not None:
        return max(start, min(start + float(cue.get("route_leg_miles") or 0), track_total_miles))
    return max(start, min(start + float(cue.get("leg_miles") or 0), track_total_miles))


def point_at_track_mile(route_points: list[dict[str, Any]], target_mile: float) -> tuple[float, float] | None:
    if not route_points:
        return None
    target = max(0.0, min(float(target_mile or 0), float(route_points[-1].get("mile") or 0)))
    prior = route_points[0]
    for item in route_points[1:]:
        item_mile = float(item.get("mile") or 0)
        prior_mile = float(prior.get("mile") or 0)
        if item_mile >= target:
            span = item_mile - prior_mile
            if span <= 0:
                return item["point"]
            ratio = (target - prior_mile) / span
            left = prior["point"]
            right = item["point"]
            return (
                left[0] + (right[0] - left[0]) * ratio,
                left[1] + (right[1] - left[1]) * ratio,
            )
        prior = item
    return route_points[-1]["point"]


def sample_track_interval(
    route_points: list[dict[str, Any]],
    start_mile: float,
    end_mile: float,
    *,
    step_miles: float = 0.03,
) -> list[dict[str, Any]]:
    start = max(0.0, min(float(start_mile or 0), float(end_mile or 0)))
    end = max(float(start_mile or 0), float(end_mile or 0))
    if end - start < 0.05:
        point = point_at_track_mile(route_points, start)
        return [{"mile": start, "point": point}] if point else []
    sample_count = max(2, int(math.ceil((end - start) / step_miles)) + 1)
    samples = []
    for index in range(sample_count):
        mile = start + (end - start) * index / (sample_count - 1)
        point = point_at_track_mile(route_points, mile)
        if point:
            samples.append({"mile": mile, "point": point})
    return samples


def cue_track_intervals(route: dict[str, Any]) -> list[dict[str, Any]]:
    cues = route.get("wayfinding_cues") or []
    route_points = cumulative_track_points(route.get("_track_segments") or [])
    if len(cues) < 2 or len(route_points) < 2:
        return []
    track_total = float(route_points[-1].get("mile") or 0)
    card_total = route_card_total_miles(route, cues)
    intervals = []
    for index, cue in enumerate(cues):
        start_track = cue_start_track_mile(cue, card_total, track_total)
        end_track = cue_end_track_mile(cues, index, card_total, track_total)
        if end_track <= start_track + 0.05:
            continue
        samples = sample_track_interval(route_points, start_track, end_track)
        if len(samples) < 2:
            continue
        intervals.append(
            {
                "cue_index": index,
                "cue": cue,
                "start_mile": start_track,
                "end_mile": end_track,
                "samples": samples,
            }
        )
    return intervals


def nearest_interval_sample(
    point: tuple[float, float],
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], float] | None:
    if not samples:
        return None
    nearest = min(samples, key=lambda sample: haversine_miles(point, sample["point"]))
    return nearest, haversine_miles(point, nearest["point"])


def overlapping_interval_match(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance_miles: float = 0.025,
    min_fraction: float = 0.65,
    min_overlap_miles: float = 0.2,
) -> dict[str, Any] | None:
    matches = []
    for sample in current.get("samples") or []:
        nearest = nearest_interval_sample(sample["point"], candidate.get("samples") or [])
        if not nearest:
            continue
        candidate_sample, distance_miles = nearest
        if distance_miles <= tolerance_miles:
            matches.append((sample, candidate_sample, distance_miles))
    sample_count = len(current.get("samples") or [])
    if not sample_count:
        return None
    matched_fraction = len(matches) / sample_count
    current_length = float(current.get("end_mile") or 0) - float(current.get("start_mile") or 0)
    matched_miles = matched_fraction * current_length
    if matched_fraction < min_fraction or matched_miles < min_overlap_miles:
        return None
    candidate_miles = [match[1]["mile"] for match in matches]
    direction = "opposite" if candidate_miles[-1] < candidate_miles[0] - 0.05 else "same"
    return {
        "candidate_cue_index": candidate.get("cue_index"),
        "candidate_seq": candidate.get("cue", {}).get("seq"),
        "matched_fraction": round(matched_fraction, 2),
        "matched_miles": round(matched_miles, 2),
        "direction": direction,
    }


def detect_wayfinding_overlap_matches(route: dict[str, Any]) -> dict[int, dict[str, Any]]:
    intervals = cue_track_intervals(route)
    matches_by_index: dict[int, dict[str, Any]] = {}
    for current in intervals:
        best: dict[str, Any] | None = None
        current_index = int(current.get("cue_index") or 0)
        for candidate in intervals:
            candidate_index = int(candidate.get("cue_index") or 0)
            if candidate_index >= current_index:
                continue
            match = overlapping_interval_match(current, candidate)
            if not match:
                continue
            if best is None or match["matched_miles"] > best["matched_miles"]:
                best = match
        if best:
            matches_by_index[current_index] = best
    return matches_by_index


def apply_geometry_overlap_wayfinding_cautions(route: dict[str, Any]) -> dict[str, Any]:
    cues = route.get("wayfinding_cues") or []
    matches = detect_wayfinding_overlap_matches(route)
    for cue_index, match in matches.items():
        cue = cues[cue_index]
        cue_type = str(cue.get("cue_type") or "")
        is_non_credit_leg = not cue.get("official_segment_ids")
        is_connector_like = cue_type in {"connector_named_trail", "connector_road", "repeat_official_noncredit", "overlap_repeat"}
        is_access_like = cue_type in {"start_access", "exit_access", "return_to_car"}
        if match.get("direction") != "opposite" or not (is_connector_like or (is_non_credit_leg and is_access_like)):
            continue
        direction_label = "Double-back overlap" if match.get("direction") == "opposite" else "Same-corridor overlap"
        compared_seq = match.get("candidate_seq")
        compared_text = f" from cue {compared_seq}" if compared_seq else ""
        until = cue.get("until") or cue.get("target") or "the next signed decision point"
        warning = (
            f"{direction_label}: this leg reuses GPS line{compared_text}. "
            f"Follow the active blue leg and arrows until {until}."
        )
        if not cue.get("field_warning"):
            cue["field_warning"] = warning
        cue["overlap_match"] = {
            key: value
            for key, value in {
                "matched_cue_seq": compared_seq,
                "matched_fraction": match.get("matched_fraction"),
                "matched_miles": match.get("matched_miles"),
                "direction": match.get("direction"),
            }.items()
            if value not in (None, "")
        }
        if is_non_credit_leg and is_connector_like:
            cue["cue_type"] = "overlap_repeat"
            if match.get("direction") == "opposite":
                cue["action"] = "DOUBLE BACK"
        add_cue_avoid(cue, "do not read the overlapping full-route line as a separate trail")
        add_cue_note(
            cue,
            "Overlap warning: this cue leg reuses another part of the route line.",
        )
        refresh_wayfinding_text(cue)
    return route


def apply_overlap_exit_wayfinding_cautions(route: dict[str, Any]) -> dict[str, Any]:
    """Mark the signed exit after a same-corridor double-back leg."""

    cues = route.get("wayfinding_cues") or []
    for index, cue in enumerate(cues[:-1]):
        match = cue.get("overlap_match") or {}
        if match.get("direction") != "opposite":
            continue
        next_cue = cues[index + 1]
        signed = cue_signed_text(next_cue.get("signed_as") or [])
        target = next_cue.get("target") or next_cue.get("until") or "the next signed decision point"
        if signed:
            warning = f"Exit the overlap here: follow {signed} toward {target} after the repeated stretch."
        else:
            warning = f"Exit the overlap here: follow the next active cue toward {target} after the repeated stretch."
        if not next_cue.get("field_warning"):
            next_cue["field_warning"] = warning
        add_cue_note(next_cue, "This is the exit from the overlapping repeated route line.")
        refresh_wayfinding_text(next_cue)
    return route


def add_names_to_wayfinding_cue(cue: dict[str, Any], names: list[str], note_prefix: str) -> None:
    clean_names = unique_nonempty_text(names)
    if not clean_names:
        return
    cue["signed_as"] = unique_nonempty_text((cue.get("signed_as") or []) + clean_names)
    note = str(cue.get("note") or "").strip()
    addition = f"{note_prefix}: {sentence_list(clean_names)}."
    if addition not in note:
        cue["note"] = " ".join(part for part in [note, addition] if part)
    refresh_wayfinding_text(cue)


def append_names_to_step(step: dict[str, str], names: list[str], prefix: str) -> None:
    clean_names = unique_nonempty_text(names)
    if not clean_names:
        return
    addition = f"{prefix}: {sentence_list(clean_names)}."
    detail = str(step.get("detail") or "")
    if addition not in detail:
        step["detail"] = " ".join(part for part in [detail, addition] if part)


def first_cue_index_before_official(route: dict[str, Any]) -> int | None:
    cues = route.get("wayfinding_cues") or []
    for index, cue in enumerate(cues):
        if cue.get("official_segment_ids"):
            return 0 if index else None
        if str(cue.get("cue_type") or "") in {"start_access", "official_segment_start"}:
            return index
    return 0 if cues else None


def connector_cue_index_for_mile(route: dict[str, Any], route_miles: float | None) -> int | None:
    cues = route.get("wayfinding_cues") or []
    if not cues:
        return None
    connector_indexes = [
        index
        for index, cue in enumerate(cues)
        if str(cue.get("cue_type") or "")
        in {"connector_named_trail", "connector_road", "overlap_repeat", "repeat_official_noncredit"}
    ]
    if connector_indexes:
        if route_miles is None:
            return connector_indexes[0]
        return min(
            connector_indexes,
            key=lambda index: abs(
                float(cues[index].get("cum_miles") or 0)
                + float(cues[index].get("leg_miles") or 0) / 2
                - float(route_miles or 0)
            ),
        )
    if route_miles is not None:
        return min(
            range(len(cues)),
            key=lambda index: abs(float(cues[index].get("cum_miles") or 0) - float(route_miles or 0)),
        )
    return 0


def enrich_route_with_walkthrough_edge_names(
    route: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    graph_edges: list[Any],
    *,
    snap_tolerance_miles: float = 0.015,
) -> dict[str, Any]:
    if not track_segments or not graph_edges or not route.get("wayfinding_cues"):
        return route
    local_edges = filter_edges_for_track(graph_edges, track_segments)
    if not local_edges:
        return route
    all_text = " ".join(
        [
            str(step.get("title") or "") + " " + str(step.get("detail") or "")
            for step in route.get("turn_by_turn_steps") or []
        ]
        + [
            " ".join(
                [
                    str(cue.get("compact") or ""),
                    str(cue.get("display_detail") or ""),
                    str(cue.get("target") or ""),
                    str(cue.get("until") or ""),
                    " ".join(str(item) for item in cue.get("signed_as") or []),
                ]
            )
            for cue in route.get("wayfinding_cues") or []
        ]
    )
    groups, _unmatched = matched_edge_groups(
        resample_track_segments_for_matching(track_segments),
        TrailGraph(local_edges),
        snap_tolerance_miles,
        preferred_text=all_text,
    )
    claimed_segment_ids = {
        str(item)
        for item in (
            route.get("segment_ids")
            or (route.get("outing") or {}).get("remaining_segment_ids")
            or (route.get("outing") or {}).get("segment_ids")
            or []
        )
    }
    first_official_index = next(
        (index for index, group in enumerate(groups) if official_group_claimed(group, claimed_segment_ids)),
        None,
    )
    last_official_index = None
    for index, group in enumerate(groups):
        if official_group_claimed(group, claimed_segment_ids):
            last_official_index = index

    start_names: list[str] = []
    if first_official_index is not None:
        route_start_text = start_access_text(route)
        for group in groups[:first_official_index]:
            edge = group.get("_edge")
            if edge and named_nonofficial_group(group) and not text_mentions_edge(route_start_text, edge):
                start_names.append(edge.name)
    if start_names:
        cue_index = first_cue_index_before_official(route)
        if cue_index is not None:
            add_names_to_wayfinding_cue(
                route["wayfinding_cues"][cue_index],
                start_names,
                "Access also follows",
            )
        for step in route.get("turn_by_turn_steps") or []:
            if str(step.get("kind") or "").lower() == "access":
                append_names_to_step(step, start_names, "Access also follows")
                break

    if first_official_index is not None and last_official_index is not None:
        for group_index, group in enumerate(groups):
            edge = group.get("_edge")
            is_between_claimed_officials = first_official_index < group_index < last_official_index
            if not edge or not is_between_claimed_officials or not named_nonofficial_group(group):
                continue
            if text_mentions_edge(all_text, edge):
                continue
            cue_index = connector_cue_index_for_mile(route, group.get("route_miles_start"))
            if cue_index is not None:
                add_names_to_wayfinding_cue(
                    route["wayfinding_cues"][cue_index],
                    [edge.name],
                    "Connector also uses",
                )
            for step in route.get("turn_by_turn_steps") or []:
                if str(step.get("kind") or "").lower() == "navigate":
                    append_names_to_step(step, [edge.name], "Connector also uses")
                    break
            all_text += " " + edge.name
    return route


def build_wayfinding_cues(route: dict[str, Any]) -> list[dict[str, Any]]:
    parking = route.get("parking") or {}
    cue_list = route.get("route_cues") or []
    track_segments = route.get("_track_segments") or []
    official_index = route.get("_official_segment_index") or {}
    outing = route.get("outing") or {}
    claimed_segment_ids = set(
        normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids"))
    )
    cues: list[dict[str, Any]] = []
    seq = 1
    cum_miles = 0.0
    for cue in cue_list:
        segments = cue.get("segments") or []
        groups = trail_groups(segments)
        if not groups:
            continue
        first_trail = groups[0]["trail_name"]
        gaps = non_credit_gaps_for_cue(cue, track_segments, official_index)
        access_cue = access_wayfinding_cue(
            seq=seq,
            cum_miles=cum_miles,
            parking=parking,
            first_trail=first_trail,
            access=cue.get("start_access") or {},
            start_access_gap_miles=float(gaps.get("start_access_gap_miles") or 0),
        )
        cues.append(access_cue)
        seq += 1
        cum_miles += float(access_cue.get("leg_miles") or 0)
        links = list(cue.get("between_links") or [])
        for index, group in enumerate(groups):
            is_first = index == 0
            prior_group = groups[index - 1] if not is_first else None
            if not is_first:
                link = link_for_group_transition(links, index - 1)
                connector_cue = link_wayfinding_cue(seq=seq, cum_miles=cum_miles, link=link, next_trail=group["trail_name"])
                if connector_cue:
                    cues.append(connector_cue)
                    seq += 1
                    cum_miles += float(connector_cue.get("leg_miles") or 0)
            next_group = groups[index + 1] if index + 1 < len(groups) else None
            phrase = turn_phrase_for_transition(prior_group or {}, group, track_segments, official_index) if prior_group else None
            follow_cue = official_group_wayfinding_cue(
                seq=seq,
                cum_miles=cum_miles,
                group=group,
                next_group=next_group,
                first_group=is_first,
                phrase=phrase,
                claimed_segment_ids=claimed_segment_ids,
            )
            cues.append(follow_cue)
            seq += 1
            cum_miles += float(follow_cue.get("leg_miles") or 0)
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            last_trail = groups[-1]["trail_name"]
            return_cue = return_wayfinding_cue(
                seq=seq,
                cum_miles=cum_miles,
                parking=parking,
                last_trail=last_trail,
                return_to_car=return_to_car,
                return_access_gap_miles=float(gaps.get("return_access_gap_miles") or 0),
            )
            cues.append(return_cue)
            seq += 1
            cum_miles += float(return_cue.get("leg_miles") or 0)
    return cues


def build_turn_by_turn_steps(route: dict[str, Any]) -> list[dict[str, str]]:
    outing = route["outing"]
    parking = route.get("parking") or {}
    cues = route.get("route_cues") or []
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    track_segments = route.get("_track_segments") or []
    official_index = route.get("_official_segment_index") or {}
    claimed_segment_ids = set(
        normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids"))
    )
    steps = [
        {
            "kind": "park",
            "title": f"Park/start at {public_display_text(parking.get('name') or outing.get('trailhead') or 'planned parking')}",
            "detail": "Start the GPX before leaving the car. The track should begin and end at this parking point.",
        }
    ]
    notes = manual_signpost_notes(route)
    if notes:
        steps.append({"kind": "checkpoint", "title": "Key checkpoints", "detail": " ".join(notes)})
    for cue in cues:
        segments = cue.get("segments") or []
        first_segment = segments[0] if segments else {}
        first_trail = first_segment.get("trail_name") or cue.get("title") or "the first official segment"
        access = cue.get("start_access") or {}
        gaps = non_credit_gaps_for_cue(cue, track_segments, official_index)
        steps.append(access_navigation_step(parking, first_trail, access, gaps.get("start_access_gap_miles") or 0))
        steps.extend(
            trail_navigation_steps_for_cue(
                cue,
                parking,
                track_segments,
                official_index,
                claimed_segment_ids,
            )
        )
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            last_segment = segments[-1] if segments else {}
            last_trail = last_segment.get("trail_name") or cue.get("title") or "route"
            steps.append(return_navigation_step(parking, last_trail, return_to_car, gaps.get("return_access_gap_miles") or 0))
    return steps


def navigation_quality_for_route(route: dict[str, Any]) -> dict[str, Any]:
    cues = route.get("route_cues") or []
    track_segments = route.get("_track_segments") or []
    official_index = route.get("_official_segment_index") or {}
    gap_records = [
        non_credit_gaps_for_cue(cue, track_segments, official_index)
        for cue in cues
    ]
    max_start_gap = max((float(record.get("start_access_gap_miles") or 0) for record in gap_records), default=0.0)
    max_return_gap = max((float(record.get("return_access_gap_miles") or 0) for record in gap_records), default=0.0)
    return {
        "start_access_gap_miles": round(max_start_gap, 3),
        "return_access_gap_miles": round(max_return_gap, 3),
        "non_credit_start_access_required": max_start_gap > 0.05,
        "non_credit_return_access_required": max_return_gap > 0.05,
    }


def segment_ownership_reconciliation_html(route: dict[str, Any]) -> str:
    reconciliation = route.get("segment_ownership_reconciliation") or {}
    segments = reconciliation.get("segments_owned_elsewhere") or []
    unclaimed = reconciliation.get("unclaimed_completed_segments") or []
    if not segments and not unclaimed:
        return ""
    items = []
    for segment in segments:
        owner_labels = ", ".join(
            str(owner.get("label") or owner.get("outing_id"))
            for owner in segment.get("owned_by_routes") or []
            if owner.get("label") or owner.get("outing_id")
        )
        label = segment.get("trail_name") or segment.get("seg_name") or f"segment {segment.get('seg_id')}"
        detail = f"Also traverses official segment {segment.get('seg_id')} ({label}); planned owner: {owner_labels or 'another route card'}."
        items.append(f"<li>{html_escape(detail)}</li>")
    for segment in unclaimed:
        label = segment.get("trail_name") or segment.get("seg_name") or f"segment {segment.get('seg_id')}"
        detail = f"Also traverses unclaimed official segment {segment.get('seg_id')} ({label}); source repair required."
        items.append(f"<li>{html_escape(detail)}</li>")
    return (
        "<section><h3>Cross-route segment ownership</h3>"
        "<p>This GPX traverses official trail already assigned to another active route card. "
        "Use the segment-owner card for planned new-credit accounting unless activity review changes the active plan.</p>"
        f"<ul>{''.join(items)}</ul></section>"
    )


def outing_route_name(outing: dict[str, Any]) -> str:
    return str(
        outing.get("route_name")
        or human_route_name(outing.get("trails") or [], outing.get("trailhead") or "").get("route_name")
        or outing.get("label")
        or "Route"
    )


def route_code_text(outing: dict[str, Any]) -> str:
    code = str(outing.get("route_code") or outing.get("label") or "").strip()
    name = outing_route_name(outing)
    if code and code != name:
        return code
    return ""


def special_management_failure_summary(failure: dict[str, Any]) -> str:
    code = str(failure.get("code") or "special_management_rule_failed")
    rule_id = str(failure.get("rule_id") or "unknown_rule")
    if failure.get("segment_id") is not None:
        target = f"segment {failure.get('segment_id')}"
    elif failure.get("matched_miles") is not None:
        target = f"matched {float(failure.get('matched_miles') or 0):.2f} mi"
    else:
        target = "route traversal"
    message = str(failure.get("message") or "").strip()
    return f"{code} {rule_id} {target}" + (f": {message}" if message else "")


def route_special_management_status(route: dict[str, Any]) -> dict[str, Any]:
    status = route.get("special_management") or (route.get("outing") or {}).get("special_management") or {}
    return status if isinstance(status, dict) else {}


def route_special_management_failures(route: dict[str, Any]) -> list[dict[str, Any]]:
    return list(route_special_management_status(route).get("failures") or [])


def route_is_special_management_blocked(route: dict[str, Any]) -> bool:
    status = str(route.get("field_readiness_status") or "")
    return status == "blocked_special_management" or bool(route_special_management_failures(route))


def route_has_blocked_status(route: dict[str, Any]) -> bool:
    status = str(route.get("field_readiness_status") or (route.get("outing") or {}).get("route_card_status") or "")
    return status.startswith("blocked_")


def route_is_field_ready(route: dict[str, Any]) -> bool:
    if route_has_blocked_status(route) or route_is_special_management_blocked(route):
        return False
    return (route.get("validation") or {}).get("passed") is not False


def cue_anchor_mismatch_failure_summary(failure: dict[str, Any]) -> str:
    code = str(failure.get("code") or "navigation_source_mismatch")
    cue_seq = failure.get("cue_seq")
    cue_part = f"cue {cue_seq}" if cue_seq is not None else "route cue"
    cue_miles = float(failure.get("cue_miles") or 0)
    map_miles = float(failure.get("map_miles") or 0)
    car_passes = failure.get("car_pass_miles") or []
    car_text = ", ".join(f"{float(value):.2f}" for value in car_passes)
    detail = f"{cue_part} spans {map_miles:.2f} GPX mi for {cue_miles:.2f} cue mi"
    if car_text:
        detail += f" and crosses parked-car pass at mi {car_text}"
    return f"{code}: {detail}"


def route_navigation_source_failures(route: dict[str, Any]) -> list[dict[str, Any]]:
    car_passes = [
        item
        for item in (route.get("logistics") or {}).get("car_passes") or []
        if not item.get("inter_component")
        and item.get("mile_from_start") is not None
        and float(item.get("distance_to_car_miles") or 0) <= 0.05
    ]
    if not car_passes:
        return []
    failures: list[dict[str, Any]] = []
    for cue in route.get("wayfinding_cues") or []:
        cue_miles = float(cue.get("leg_miles") or 0)
        map_miles_value = cue.get("route_leg_miles")
        if map_miles_value is None or cue_miles <= 0.05:
            continue
        map_miles = float(map_miles_value or 0)
        if map_miles <= max((cue_miles * 2.0) + 0.25, cue_miles + 1.0):
            continue
        start_miles = float(cue.get("route_miles") if cue.get("route_miles") is not None else cue.get("cum_miles") or 0)
        end_miles = start_miles + map_miles
        crossed_car_passes = [
            float(item.get("mile_from_start") or 0)
            for item in car_passes
            if start_miles - 0.05 <= float(item.get("mile_from_start") or 0) <= end_miles + 0.05
        ]
        if not crossed_car_passes:
            continue
        failures.append(
            {
                "code": "mid_route_car_pass_anchor_mismatch",
                "cue_seq": cue.get("seq"),
                "cue_type": cue.get("cue_type"),
                "cue_miles": round(cue_miles, 3),
                "map_miles": round(map_miles, 3),
                "route_miles_start": round(start_miles, 3),
                "route_miles_end": round(end_miles, 3),
                "car_pass_miles": [round(value, 3) for value in crossed_car_passes],
                "signed_as": cue.get("signed_as") or [],
                "target": cue.get("target"),
            }
        )
    return failures


def route_blocked_message(route: dict[str, Any]) -> str:
    status = str(route.get("field_readiness_status") or (route.get("outing") or {}).get("route_card_status") or "")
    if status == "blocked_special_management" or route_special_management_failures(route):
        return (
            "This route is held because it has a published trail-management rule violation. "
            "Do not run it from this field packet until redesigned and recertified."
        )
    if status == "blocked_navigation_source":
        return (
            "This route is held because the source route, cue anchors, and live-map geometry disagree. "
            "Do not run it from this field packet until the canonical route is repaired and recertified."
        )
    return "This route is held because it is not field-ready. Do not run it from this field packet until recertified."


def apply_navigation_source_audit_to_routes(routes: list[dict[str, Any]]) -> None:
    for route in routes:
        failures = route_navigation_source_failures(route)
        route["navigation_source_audit"] = {
            "status": "failed" if failures else "passed",
            "failures": failures,
        }
        if not failures:
            continue
        summaries = [cue_anchor_mismatch_failure_summary(failure) for failure in failures]
        route["field_readiness_status"] = "blocked_navigation_source"
        route["route_card_audit_blockers"] = append_unique_text(
            route.get("route_card_audit_blockers") or [],
            summaries,
        )
        outing = route.get("outing") or {}
        outing["route_card_status"] = "blocked_navigation_source"
        outing["packet_visibility"] = "blocked_not_field_ready"
        outing["certified_route_card"] = False
        outing["requires_field_walkthrough"] = True
        outing["route_card_audit_blockers"] = append_unique_text(
            outing.get("route_card_audit_blockers") or [],
            summaries,
        )


def normalized_route_audit_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "status": "not_checked",
            "failures": [],
            "checked_segments": [],
            "date_context_required": [],
        }
    failures = list(report.get("failures") or [])
    if failures:
        status = "failed"
    elif report.get("passed") is True:
        status = "passed"
    else:
        status = "not_checked"
    return {
        "status": status,
        "failures": failures,
        "checked_segments": list(report.get("checked_segments") or []),
        "date_context_required": list(report.get("date_context_required") or []),
    }


def route_audit_match_keys(route: dict[str, Any]) -> set[str]:
    outing = route.get("outing") or {}
    keys = {
        str(route.get("outing_id") or ""),
        str(route.get("label") or ""),
        str(route.get("route_code") or ""),
        str(outing.get("outing_id") or ""),
        str(outing.get("label") or ""),
        str(outing.get("route_code") or ""),
    }
    return {key for key in keys if key}


def append_unique_text(values: list[Any], additions: list[str]) -> list[str]:
    output = [str(value) for value in values or [] if str(value)]
    seen = set(output)
    for addition in additions:
        if addition and addition not in seen:
            output.append(addition)
            seen.add(addition)
    return output


def apply_special_management_audit_to_routes(
    routes: list[dict[str, Any]],
    audit: dict[str, Any] | None,
) -> None:
    reports_by_key: dict[str, dict[str, Any]] = {}
    for report in (audit or {}).get("routes") or []:
        for key in (str(report.get("outing_id") or ""), str(report.get("label") or "")):
            if key:
                reports_by_key[key] = report
    for route in routes:
        report = next(
            (reports_by_key[key] for key in route_audit_match_keys(route) if key in reports_by_key),
            None,
        )
        status = normalized_route_audit_status(report)
        route["special_management"] = status
        outing = route.get("outing") or {}
        outing["special_management"] = status
        failure_summaries = [
            special_management_failure_summary(failure)
            for failure in status.get("failures") or []
        ]
        if failure_summaries:
            route["field_readiness_status"] = "blocked_special_management"
            route["route_card_audit_blockers"] = append_unique_text(
                route.get("route_card_audit_blockers") or outing.get("route_card_audit_blockers") or [],
                failure_summaries,
            )
            outing["route_card_status"] = "blocked_special_management"
            outing["packet_visibility"] = "blocked_not_field_ready"
            outing["certified_route_card"] = False
            outing["requires_field_walkthrough"] = True
            outing["route_card_audit_blockers"] = append_unique_text(
                outing.get("route_card_audit_blockers") or [],
                failure_summaries,
            )
        elif not route_has_blocked_status(route):
            route["field_readiness_status"] = "field_ready" if route_is_field_ready(route) else "needs_review"


def build_special_management_audit_for_routes(
    routes: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any] | None:
    if not routes:
        return None
    if not DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON.exists() or not Path(DEFAULT_OFFICIAL_GEOJSON).exists():
        return None
    field_tool_data = {
        "routes": [
            route_field_tool_record(route, route.get("completion_safety"))
            for route in routes
        ]
    }
    return build_special_management_audit(
        field_tool_data=field_tool_data,
        official_geojson=read_json(Path(DEFAULT_OFFICIAL_GEOJSON)),
        rules_config=read_json(DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON),
        packet_dir=output_dir,
        open_trails_geojson=(
            read_json(DEFAULT_SPECIAL_MANAGEMENT_R2R_TRAILS_GEOJSON)
            if DEFAULT_SPECIAL_MANAGEMENT_R2R_TRAILS_GEOJSON.exists()
            else None
        ),
    )


def render_card(route: dict[str, Any]) -> str:
    outing = route["outing"]
    parking = route.get("parking") or {}
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    effort = route_effort_summary(route)
    time_estimates = route_time_estimate_summary(route)
    completion_safe = (route.get("completion_safety") or {}).get(
        "normal_completion_preserves_remaining_menu_coverage"
    )
    field_ready = route_is_field_ready(route)
    completion_safe_value = "false" if completion_safe is False or not field_ready else "true"
    field_ready_value = "true" if field_ready else "false"
    segment_ids = " ".join(normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids")))
    nav_url = route.get("parking_navigation_url")
    if field_ready:
        nav_link = (
            f'<a class="secondary" href="{html_escape(nav_url)}">Open parking in Google Maps</a>'
            if nav_url
            else '<span class="secondary disabled">Parking navigation unavailable</span>'
        )
    else:
        nav_link = '<span class="secondary disabled">Parking held</span>'
    route_name = outing_route_name(outing)
    code = route_code_text(outing)
    trailhead_detail = public_display_text(outing["trailhead"])
    code_detail = f"{code} · {trailhead_detail}" if code else trailhead_detail
    wayfinding_cues = route.get("wayfinding_cues") or build_wayfinding_cues(route)
    steps_html = "".join(
        f'<li class="{html_escape(cue.get("cue_type"))}" tabindex="0">'
        f'<b><span class="cue-code">{int(cue.get("seq") or 0):02d} {float(cue.get("cum_miles") or 0):.2f} mi '
        f'(+{float(cue.get("leg_miles") or 0):.2f})</span> {html_escape(cue_type_label(str(cue.get("cue_type") or "")))}</b>'
        f'<span>{html_escape(public_display_text(cue.get("display_detail") or wayfinding_display_detail(cue)))}</span></li>'
        for cue in wayfinding_cues
    )
    logistics_html = logistics_section_html(logistics)
    ownership_html = segment_ownership_reconciliation_html(route)
    blockers = append_unique_text(
        outing.get("route_card_audit_blockers") or route.get("route_card_audit_blockers") or [],
        [special_management_failure_summary(failure) for failure in route_special_management_failures(route)],
    )
    blocked_banner = ""
    if not field_ready:
        blocker_items = "".join(f"<li>{html_escape(blocker)}</li>" for blocker in blockers)
        blocked_banner = (
            '<section class="route-blocked-banner"><h3>NOT FIELD READY</h3>'
            f"<p>{html_escape(route_blocked_message(route))}</p>"
            f"<ul>{blocker_items}</ul></section>"
        )
    if field_ready:
        action_html = f"""
        <a href="{html_escape(route['gpx_href'])}" download>Open Field GPX</a>
        <a class="secondary" href="live-map.html?outing={html_escape(outing['outing_id'])}">Open Live Map</a>
        {nav_link}
        <button type="button" class="active-button" data-active-action="pin">Pin active</button>
        <button type="button" class="done-button" data-complete-action="mark">Mark done</button>
        <button type="button" class="undo-button" data-complete-action="undo">Undo done</button>
        """
    else:
        action_html = f"""
        <span class="disabled">Field GPX held</span>
        <span class="secondary disabled">Live map held</span>
        {nav_link}
        """
    return f"""
    <article class="card{' blocked-route' if not field_ready else ''}" id="{html_escape(outing['outing_id'])}" data-outing-id="{html_escape(outing['outing_id'])}" data-minutes="{int(outing.get('total_minutes') or 0)}" data-completion-safe="{completion_safe_value}" data-segment-ids="{html_escape(segment_ids)}" data-field-ready="{field_ready_value}">
      <div class="card-head">
        <span>{html_escape(code_detail)}</span>
        <h2>{html_escape(route_name)}</h2>
      </div>
      {blocked_banner}
      <div class="stats">
        <div><b>Door to door p75</b><strong>{html_escape(format_minutes(outing.get('total_minutes')))}</strong></div>
        <div><b>On foot</b><strong>{html_escape(format_miles(outing.get('on_foot_miles')))} mi</strong></div>
        <div><b>Official</b><strong>{html_escape(format_miles(outing.get('official_miles')))} mi</strong></div>
        <div><b>Door to door p90</b><strong>{html_escape(format_minutes(time_estimates.get('door_to_door_minutes_p90')))}</strong></div>
        <div><b>Segments</b><strong>{html_escape(outing.get('remaining_segment_count'))} / {len(outing.get('segment_ids') or [])}</strong></div>
        <div><b>Climb</b><strong>{int(round(effort.get('ascent_ft') or 0))} ft</strong></div>
      </div>
      <div class="actions">
        {action_html}
      </div>
      <section><h3>PARK/START</h3><p>Park/start at {html_escape(public_display_text(parking.get('name') or outing.get('trailhead')))}</p></section>
      {logistics_html}
      <section><h3>Trails</h3><p>{html_escape(', '.join(outing.get('trails') or []))}</p></section>
      {ownership_html}
      <section><h3>Field Cue Sheet</h3><p class="cue-help">What to do next: Tap the cue you are working on to keep your place.</p><ol class="steps decision-cards">{steps_html}</ol></section>
    </article>
    """


def status_label(value: Any) -> str:
    return str(value or "unknown").replace("_", " ").strip()


def pluralize(count: Any, singular: str, plural: str | None = None) -> str:
    number = int(count or 0)
    label = singular if number == 1 else (plural or f"{singular}s")
    return f"{number} {label}"


def render_field_day_loop(loop: dict[str, Any]) -> str:
    route_ref = loop.get("route_card_ref") or {}
    trails = summarized_names(loop.get("trail_names") or [], limit=4)
    status = status_label(loop.get("certification_status"))
    route_name = loop.get("route_name") or loop.get("label") or loop.get("candidate_id") or "Loop"
    code = str(loop.get("route_code") or loop.get("label") or "").strip()
    code_detail = f"{code} · " if code and code != route_name else ""
    blockers = loop.get("route_card_audit_blockers") or route_ref.get("certification_blockers") or []
    blocked = (
        str(loop.get("certification_status") or "") == "blocked_special_management"
        or str(route_ref.get("field_readiness_status") or "") == "blocked_special_management"
        or bool(blockers)
    )
    blocker_html = ""
    if blockers:
        blocker_html = f'<em>Audit blockers: {html_escape(", ".join(str(blocker) for blocker in blockers))}</em>'
    route_card_actions = ""
    if route_ref.get("outing_id"):
        route_card_label = route_ref.get("route_name") or route_ref.get("label") or route_ref.get("outing_id")
        route_card_actions += (
            f'<a href="#{html_escape(route_ref.get("outing_id"))}">'
            f'Route card {html_escape(route_card_label)}</a>'
        )
    else:
        route_card_actions += '<span class="field-day-muted">Route card needed</span>'
    if route_ref.get("gpx_href") and not blocked:
        route_card_actions += f'<a class="secondary" href="{html_escape(route_ref.get("gpx_href"))}" download>GPX</a>'
    elif blocked:
        route_card_actions += '<span class="field-day-muted">GPX held</span>'
    return f"""
      <li class="field-day-loop {html_escape(str(loop.get('certification_status') or 'unknown'))}">
        <div>
          <b>{html_escape(route_name)}</b>
          <span>{html_escape(code_detail)}{html_escape(public_display_text(loop.get("trailhead")))} · {html_escape(trails)}</span>
          <em>{html_escape(status)}</em>
          {blocker_html}
        </div>
        <div class="field-day-loop-stats">
          <span>{html_escape(format_minutes(loop.get("p75_minutes")))} p75</span>
          <span>{html_escape(format_miles(loop.get("on_foot_miles")))} mi</span>
          <span>{html_escape(loop.get("segment_count"))} seg</span>
        </div>
        <div class="field-day-loop-actions">{route_card_actions}</div>
      </li>
    """


def render_field_day_card(day: dict[str, Any]) -> str:
    loops = day.get("loops") or []
    loop_html = "\n".join(render_field_day_loop(loop) for loop in loops)
    constraints = ", ".join(str(value) for value in day.get("constraints") or [])
    constraint_html = f'<p class="field-day-constraints">{html_escape(constraints)}</p>' if constraints else ""
    reserve_html = (
        '<p class="field-day-empty">Reserve / buffer day - no route planned.</p>'
        if not loops
        else ""
    )
    return f"""
    <article class="field-day-card" data-field-day-id="{html_escape(day.get("field_day_id"))}" data-day-status="{html_escape(day.get("execution_status"))}">
      <div class="field-day-head">
        <h2>{html_escape(day.get("weekday_name"))}, {html_escape(day.get("date"))}</h2>
        <span>{html_escape(status_label(day.get("execution_status")))}</span>
      </div>
      <div class="field-day-stats">
        <div><b>Door to door p75</b><strong>{html_escape(format_minutes(day.get("p75_minutes")))}</strong></div>
        <div><b>Door to door p90</b><strong>{html_escape(format_minutes(day.get("p90_minutes")))}</strong></div>
        <div><b>On foot</b><strong>{html_escape(format_miles(day.get("on_foot_miles")))} mi</strong></div>
        <div><b>Segments</b><strong>{html_escape(day.get("segment_count"))}</strong></div>
        <div><b>Loops</b><strong>{html_escape(day.get("loop_count"))}</strong></div>
        <div><b>Re-park drive</b><strong>{html_escape(format_minutes(day.get("between_drive_minutes")))}</strong></div>
      </div>
      {constraint_html}
      {reserve_html}
      <ol class="field-day-loops">{loop_html}</ol>
    </article>
    """


def render_field_day_view(field_day_layer: dict[str, Any] | None) -> str:
    if not field_day_layer:
        return ""
    summary = field_day_layer.get("summary") or {}
    field_days = field_day_layer.get("field_days") or []
    certified = int(summary.get("certified_route_card_loop_count") or 0)
    needs_audit_fix = int(summary.get("needs_route_card_audit_fix_loop_count") or 0)
    needs_promotion = int(summary.get("needs_route_card_promotion_loop_count") or 0)
    calendar_day_count = int(summary.get("calendar_day_count") or summary.get("field_day_count") or len(field_days))
    reserve_day_count = int(summary.get("reserve_day_count") or 0)
    active_execution_day_count = int(
        summary.get("active_execution_day_count")
        if summary.get("active_execution_day_count") is not None
        else max(0, calendar_day_count - reserve_day_count)
    )
    summary_text = (
        f"{pluralize(calendar_day_count, 'calendar day')} · "
        f"{pluralize(active_execution_day_count, 'active execution day')} · "
        f"{pluralize(reserve_day_count, 'reserve day')} · "
        f"{pluralize(summary.get('loop_count'), 'loop')} · "
        f"{pluralize(summary.get('multi_start_day_count'), 'multi-start day')} · "
        f"{summary.get('covered_segment_count')}/{summary.get('official_segment_count')} official segments"
    )
    route_card_text = (
        f"{pluralize(certified, 'certified loop')} · "
        f"{needs_audit_fix} needs route-card audit fix · "
        f"{needs_promotion} needs route-card promotion"
    )
    cards = "\n".join(render_field_day_card(day) for day in field_days)
    return f"""
  <section id="field-day-view" class="field-day-view" aria-label="Field Days">
    <div class="field-day-summary">
      <h2>Field Days</h2>
      <p>{html_escape(summary_text)}</p>
      <p>{html_escape(route_card_text)} · {html_escape(status_label(field_day_layer.get("publication_status")))}</p>
    </div>
    <div class="field-day-list">{cards}</div>
  </section>
    """


def render_index(manifest: dict[str, Any]) -> str:
    cards = "\n".join(render_card(route) for route in manifest["routes"])
    field_day_layer = manifest.get("field_day_layer")
    field_day_view = render_field_day_view(field_day_layer)
    default_view = "field-days" if field_day_view else "routes"
    field_days_tab_class = ' class="active"' if default_view == "field-days" else ""
    routes_tab_class = ' class="active"' if default_view == "routes" else ""
    view_tabs = (
        f"""
      <div class="view-tabs" role="tablist" aria-label="Field guide views">
        <button type="button"{field_days_tab_class} data-view="field-days">Field Days</button>
        <button type="button"{routes_tab_class} data-view="routes">Route Cards</button>
      </div>
        """
        if field_day_view
        else ""
    )
    manual_count = manifest["summary"]["manual_hold_count"]
    zip_href = manifest["summary"].get("gpx_zip_href") or f"gpx/{GPX_ZIP_NAME}"
    all_segment_ids = {
        segment_id
        for route in manifest["routes"]
        for segment_id in normalized_segment_ids(
            route.get("outing", {}).get("remaining_segment_ids") or route.get("outing", {}).get("segment_ids")
        )
    }
    field_ready_routes = [route for route in manifest["routes"] if route_is_field_ready(route)]
    held_routes = [route for route in manifest["routes"] if not route_is_field_ready(route)]
    certified = manifest.get("certified_baseline") or {}
    official_segment_count = certified.get("official_segment_count") or len(all_segment_ids)
    menu_on_foot_miles = sum(
        float((route.get("outing") or route).get("on_foot_miles") or 0)
        for route in field_ready_routes
    )
    gpx_status = "GPX passed" if manifest["summary"].get("gpx_validation_passed") else "GPX needs review"
    special_management_status = (
        "special-management blocks present" if held_routes else "special-management passed"
    )
    field_menu_text = (
        f"{len(field_ready_routes)} runnable outings · "
        f"{len(held_routes)} held · "
        f"{len(all_segment_ids)}/{official_segment_count} official segments · "
        f"{format_miles(menu_on_foot_miles)} runnable on foot · {gpx_status} · {special_management_status}"
    )
    filter_labels = {60: "&le;1h", 90: "&le;90m", 120: "&le;2h", 180: "&le;3h", 240: "&le;4h", 360: "&le;6h"}
    filter_buttons = "\n      ".join(
        ["""<button type="button" class="active" data-filter="all">All</button>"""]
        + [
            f'<button type="button" data-filter="{minutes}">{filter_labels.get(minutes, f"&le;{minutes}m")}</button>'
            for minutes in TIME_FILTER_MINUTES
        ]
    )
    manual_note = (
        f"<p>{manual_count} manual-design outing(s) are intentionally hidden from GPX export until the route is redesigned.</p>"
        if manual_count
        else ""
    )
    quick_list_heading = "Route-card options" if field_day_view else "Today&apos;s best options"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#111827">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Trails Packet">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <link rel="manifest" href="manifest.webmanifest">
  <link rel="apple-touch-icon" href="icons/icon-192.png">
  <link rel="icon" href="icons/icon-192.png">
  <title>Phone Field Packet</title>
  <style>
    body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f7f2; color:#111827; }}
    header {{ padding:14px 14px 10px; background:rgba(255,255,255,.97); border-bottom:1px solid #d7ddd4; }}
    h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:0; }}
    p {{ margin:4px 0; color:#475467; line-height:1.4; }}
    .top-grid {{ display:grid; grid-template-columns:1fr; gap:8px; margin-top:10px; }}
    .status-panel {{ border:1px solid #d7ddd4; border-radius:8px; padding:8px; background:#f9fafb; }}
    .status-panel b {{ color:#111827; }}
    .filters {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; margin-top:10px; }}
    button {{ min-height:34px; border:1px solid #d7ddd4; border-radius:6px; background:#fff; font-weight:700; }}
    button.active {{ background:#111827; color:#fff; border-color:#111827; }}
    .view-tabs {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:10px; }}
    .view-tabs button {{ min-height:38px; }}
    .utility-actions {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; }}
    .utility-actions a,.utility-actions button {{ display:flex; align-items:center; justify-content:center; min-height:38px; padding:0 10px; border-radius:6px; border:1px solid #d7ddd4; background:#fff; color:#111827; font-weight:800; text-decoration:none; }}
    .quick-list {{ margin:10px 0 0; padding:8px; border:1px solid #d7ddd4; border-radius:8px; background:#fff; }}
    .quick-list h2 {{ margin:0 0 4px; color:#111827; font-size:14px; }}
    main {{ padding:10px; display:grid; gap:10px; }}
    body.view-field-days .filters, body.view-field-days .quick-list, body.view-field-days #route-cards, body.view-routes #field-day-view {{ display:none !important; }}
    .card {{ overflow:hidden; border:1px solid #d7ddd4; border-radius:8px; background:#fff; box-shadow:0 1px 4px rgba(15,23,42,.08); }}
    .card.active-outing {{ border:2px solid #2563eb; box-shadow:0 0 0 3px rgba(37,99,235,.12); }}
    .card.blocked-route {{ border-color:#dc2626; background:#fffafa; }}
    .card.blocked-route .card-head {{ background:#7f1d1d; }}
    .card.completed {{ opacity:.48; }}
    body.hide-completed .card.completed {{ display:none !important; }}
    .card-head {{ padding:12px; background:#111827; color:#fff; }}
    .card.active-outing .card-head {{ background:#1d4ed8; }}
    .card-head span {{ display:block; color:#cbd5e1; font-size:12px; text-transform:uppercase; font-weight:800; }}
    h2 {{ margin:3px 0 0; font-size:19px; line-height:1.15; letter-spacing:0; }}
    .stats {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; padding:10px 12px; }}
    .stats div {{ border:1px solid #e5e7eb; border-radius:6px; padding:7px; background:#f9fafb; }}
    .stats b {{ display:block; color:#667085; font-size:11px; text-transform:uppercase; }}
    .stats strong {{ display:block; margin-top:2px; font-size:15px; }}
    .actions {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 12px 12px; }}
    .actions a,.actions span,.actions button {{ display:flex; align-items:center; justify-content:center; min-height:40px; border-radius:6px; font-weight:800; text-decoration:none; border:1px solid #d7ddd4; }}
    .actions a:first-child {{ background:#2563eb; color:#fff; }}
    .done-button {{ background:#166534; color:#fff; border-color:#166534 !important; }}
    .active-button {{ background:#dbeafe; color:#1e3a8a; border-color:#93c5fd !important; }}
    .card.active-outing .active-button {{ background:#1d4ed8; color:#fff; border-color:#1d4ed8 !important; }}
    .undo-button {{ display:none !important; background:#fff; color:#111827; }}
    .card.completed .done-button {{ display:none !important; }}
    .card.completed .undo-button {{ display:flex !important; }}
    .secondary {{ border:1px solid #d7ddd4; color:#111827; background:#fff; }}
    .disabled {{ color:#667085; }}
    .route-blocked-banner {{ border-top:0; background:#fef2f2; color:#7f1d1d; }}
    .route-blocked-banner h3 {{ color:#7f1d1d; }}
    .route-blocked-banner p {{ color:#7f1d1d; font-weight:800; }}
    .route-blocked-banner ul {{ margin:6px 0 0; padding-left:18px; color:#7f1d1d; font-size:12px; line-height:1.35; }}
    section {{ padding:10px 12px; border-top:1px solid #eef0ed; }}
    h3 {{ margin:0 0 6px; font-size:12px; text-transform:uppercase; letter-spacing:0; color:#111827; }}
    .segment {{ border:1px solid #e5e7eb; border-radius:6px; padding:7px; margin:5px 0; background:#fff; }}
    .segment.ascent {{ border-color:#b45309; background:#fff7ed; }}
    .segment b {{ display:block; font-size:13px; }}
    .segment span {{ display:block; margin-top:2px; color:#475467; font-size:12px; line-height:1.35; }}
    .steps {{ margin:0; padding:0; list-style:none; counter-reset:step; display:grid; gap:7px; }}
    .steps li {{ counter-increment:step; position:relative; padding:8px 8px 8px 36px; border:1px solid #e5e7eb; border-radius:6px; background:#fff; }}
    .steps li::before {{ content:counter(step); position:absolute; left:8px; top:8px; display:grid; place-items:center; width:20px; height:20px; border-radius:999px; background:#111827; color:#fff; font-size:11px; font-weight:900; }}
    .steps li.connector {{ background:#f0f9ff; border-color:#bae6fd; }}
    .steps li.ascent {{ background:#fff7ed; border-color:#fdba74; }}
    .steps li.return {{ background:#f0fdf4; border-color:#bbf7d0; }}
    .steps li.car-pass {{ background:#fefce8; border-color:#fde68a; }}
    .steps li.water {{ background:#eff6ff; border-color:#bfdbfe; }}
    .steps li.checkpoint {{ background:#fff7ed; border-color:#fdba74; }}
    .steps li.current-step {{ border-color:#2563eb; box-shadow:0 0 0 3px rgba(37,99,235,.14); background:#eff6ff; }}
    .steps b {{ display:block; font-size:13px; }}
    .steps span {{ display:block; margin-top:2px; color:#475467; font-size:12px; line-height:1.35; }}
    .decision-cards li b {{ font-size:15px; }}
    .decision-cards li span {{ font-size:13px; }}
    .cue-code {{ display:block; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:#6b7280; margin-bottom:3px; }}
    .cue-help {{ margin:0 0 8px; font-size:13px; color:#475467; }}
    .signpost-notes {{ margin:6px 0 0; padding-left:18px; color:#475467; font-size:12px; line-height:1.35; }}
    .field-day-view {{ padding:10px; display:grid; gap:10px; }}
    .field-day-summary {{ border:1px solid #d7ddd4; border-radius:8px; padding:10px; background:#fff; }}
    .field-day-summary h2 {{ margin:0 0 4px; font-size:18px; }}
    .field-day-list {{ display:grid; gap:10px; }}
    .field-day-card {{ overflow:hidden; border:1px solid #d7ddd4; border-radius:8px; background:#fff; box-shadow:0 1px 4px rgba(15,23,42,.08); }}
    .field-day-head {{ padding:12px; display:flex; justify-content:space-between; gap:8px; background:#17324d; color:#fff; }}
    .field-day-head h2 {{ margin:0; font-size:18px; line-height:1.15; }}
    .field-day-head span {{ align-self:start; border:1px solid rgba(255,255,255,.4); border-radius:999px; padding:3px 7px; color:#e5edf5; font-size:11px; font-weight:900; text-transform:uppercase; white-space:nowrap; }}
    .field-day-stats {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; padding:10px 12px; }}
    .field-day-stats div {{ border:1px solid #e5e7eb; border-radius:6px; padding:7px; background:#f9fafb; }}
    .field-day-stats b {{ display:block; color:#667085; font-size:11px; text-transform:uppercase; }}
    .field-day-stats strong {{ display:block; margin-top:2px; font-size:15px; }}
    .field-day-constraints {{ margin:0 12px 10px; padding:7px; border-left:4px solid #2563eb; background:#eff6ff; color:#1e3a8a; font-size:12px; }}
    .field-day-empty {{ margin:0 12px 12px; padding:8px; border:1px solid #d7ddd4; border-radius:6px; background:#f8faf8; color:#475467; font-size:13px; font-weight:700; }}
    .field-day-loops {{ margin:0; padding:0 12px 12px; list-style:none; display:grid; gap:7px; }}
    .field-day-loop {{ border:1px solid #e5e7eb; border-radius:6px; padding:8px; background:#fff; display:grid; gap:6px; }}
    .field-day-loop.needs_route_card_promotion,.field-day-loop.needs_route_card_audit_fix {{ border-color:#fdba74; background:#fff7ed; }}
    .field-day-loop.blocked_special_management {{ border-color:#dc2626; background:#fef2f2; }}
    .field-day-loop b {{ display:block; font-size:14px; }}
    .field-day-loop span,.field-day-loop em {{ display:block; color:#475467; font-size:12px; line-height:1.35; font-style:normal; }}
    .field-day-loop em {{ color:#7c2d12; font-weight:800; }}
    .field-day-loop-stats {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:5px; }}
    .field-day-loop-stats span {{ border:1px solid #e5e7eb; border-radius:5px; padding:5px; color:#111827; background:#fff; font-weight:800; text-align:center; }}
    .field-day-loop-actions {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; }}
    .field-day-loop-actions a,.field-day-loop-actions span {{ display:flex; align-items:center; justify-content:center; min-height:34px; padding:0 8px; border:1px solid #d7ddd4; border-radius:6px; background:#fff; color:#111827; font-size:12px; font-weight:900; text-decoration:none; }}
    .field-day-loop-actions a:first-child {{ background:#2563eb; color:#fff; border-color:#2563eb; }}
    .field-day-muted {{ color:#667085 !important; background:#f9fafb !important; }}
    .warning {{ margin:10px 12px; padding:8px; border-left:4px solid #b45309; background:#fff7ed; color:#7c2d12; }}
    body.screenshot header,.screenshot .utility-actions,.screenshot .filters,.screenshot .actions {{ display:none !important; }}
    body.screenshot main {{ padding:0; }}
    body.screenshot .card {{ display:none !important; border:0; border-radius:0; box-shadow:none; }}
    body.screenshot .card.active-outing:not(.completed), body.screenshot .card:not(.completed):first-of-type {{ display:block !important; }}
    @media (min-width:760px) {{ main {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} header {{ position:static; }} }}
  </style>
</head>
<body class="view-{default_view}">
  <header>
    <h1>Phone Field Packet</h1>
    <p>Run the field day first; open each certified route card for GPX, parking, cue, and return-to-car detail.</p>
    <div class="top-grid">
      <div class="status-panel"><b>Offline-ready</b> after the first full load. In Safari, Share &rarr; Add to Home Screen for the app-style launcher. <span id="offline-status">Checking offline cache...</span></div>
      <div class="status-panel"><b>Field menu</b> <span id="field-menu-summary">{html_escape(field_menu_text)}</span></div>
      <div class="status-panel"><b>Progress</b> <span id="remaining-segment-count">{len(all_segment_ids)}</span> of <span id="total-segment-count">{len(all_segment_ids)}</span> official segments remain in this field menu. <span id="completed-outing-count">0</span> outing(s) marked done on this phone.</div>
      <div class="utility-actions">
        <a href="live-map.html">Live GPS map</a>
        <button type="button" id="completed-toggle">Hide completed</button>
        <button type="button" id="screenshot-toggle">Screenshot mode</button>
        <button type="button" id="clear-active">Clear active</button>
        <button type="button" id="export-progress">Export progress</button>
        <button type="button" id="reset-completed">Reset progress</button>
      </div>
    </div>
    {view_tabs}
    <div class="quick-list"><h2>{quick_list_heading}</h2><p id="best-today-copy">Use the time buttons to pick what fits the door-to-door window. Mark completed outings so they disappear from the active field list.</p></div>
    {manual_note}
    <div class="filters">
      {filter_buttons}
    </div>
  </header>
  <main id="route-cards">{cards}</main>
  {field_day_view}
  <script>
    const STORAGE_KEY = "{COMPLETED_STORAGE_KEY}";
    const ACTIVE_KEY = "{ACTIVE_STORAGE_KEY}";
    const DEFAULT_VIEW = "{default_view}";
    const buttons = [...document.querySelectorAll("button[data-filter]")];
    const viewButtons = [...document.querySelectorAll("button[data-view]")];
    const cards = [...document.querySelectorAll(".card")];
    const cardContainer = document.getElementById("route-cards");
    const completedToggle = document.getElementById("completed-toggle");
    const screenshotToggle = document.getElementById("screenshot-toggle");
    const clearActive = document.getElementById("clear-active");
    const exportProgress = document.getElementById("export-progress");
    const resetCompleted = document.getElementById("reset-completed");
    const offlineStatus = document.getElementById("offline-status");
    const remainingSegmentCount = document.getElementById("remaining-segment-count");
    const totalSegmentCount = document.getElementById("total-segment-count");
    const completedOutingCount = document.getElementById("completed-outing-count");
    const bestTodayCopy = document.getElementById("best-today-copy");

    function setView(view) {{
      const fieldDaysActive = view === "field-days";
      document.body.classList.toggle("view-field-days", fieldDaysActive);
      document.body.classList.toggle("view-routes", !fieldDaysActive);
      viewButtons.forEach(button => button.classList.toggle("active", button.dataset.view === view));
    }}

    viewButtons.forEach(button => button.addEventListener("click", () => setView(button.dataset.view || DEFAULT_VIEW)));
    const requestedView = new URLSearchParams(window.location.search).get("view");
    if (requestedView === "field-days" || requestedView === "routes") {{
      setView(requestedView);
    }} else if (window.location.hash === "#field-days") {{
      setView("field-days");
    }} else {{
      setView(DEFAULT_VIEW);
    }}

    function segmentIdsForCard(card) {{
      return (card.dataset.segmentIds || "").split(" ").filter(Boolean);
    }}
    function fieldReady(card) {{
      return card.dataset.fieldReady !== "false";
    }}

    function allSegmentSet() {{
      const ids = new Set();
      cards.filter(fieldReady).forEach(card => segmentIdsForCard(card).forEach(id => ids.add(id)));
      return ids;
    }}

    function completedSet() {{
      try {{
        return new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));
      }} catch (error) {{
        return new Set();
      }}
    }}

    function saveCompleted(set) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...set]));
    }}

    function completedSegmentSet() {{
      const completed = completedSet();
      const ids = new Set();
      cards.forEach(card => {{
        if (fieldReady(card) && completed.has(card.dataset.outingId)) {{
          segmentIdsForCard(card).forEach(id => ids.add(id));
        }}
      }});
      return ids;
    }}

    function activeOutingId() {{
      return localStorage.getItem(ACTIVE_KEY) || "";
    }}

    function saveActive(id) {{
      if (id) {{
        localStorage.setItem(ACTIVE_KEY, id);
      }} else {{
        localStorage.removeItem(ACTIVE_KEY);
      }}
    }}

    function syncCompletedState() {{
      const completed = completedSet();
      cards.forEach(card => card.classList.toggle("completed", completed.has(card.dataset.outingId)));
      applyFilter();
    }}

    function syncActiveState(scrollToActive = false) {{
      let activeId = activeOutingId();
      const activeIsHeld = activeId && cards.some(card => card.dataset.outingId === activeId && !fieldReady(card));
      if (activeIsHeld) {{
        saveActive("");
        activeId = "";
      }}
      let activeCard = null;
      cards.forEach(card => {{
        const isActive = activeId && card.dataset.outingId === activeId;
        card.classList.toggle("active-outing", Boolean(isActive));
        const activeButton = card.querySelector('[data-active-action="pin"]');
        if (activeButton) {{
          activeButton.textContent = isActive ? "Active" : "Pin active";
        }}
        if (isActive) {{
          activeCard = card;
        }}
      }});
      if (activeCard) {{
        cardContainer.prepend(activeCard);
      }}
      if (clearActive) {{
        clearActive.disabled = !activeId;
      }}
      applyFilter();
      if (scrollToActive && activeCard) {{
        activeCard.scrollIntoView({{ block: "start" }});
      }}
    }}

    function activeFilter() {{
      return document.querySelector("button[data-filter].active")?.dataset.filter || "all";
    }}
    function activeFilterLabel() {{
      return document.querySelector("button[data-filter].active")?.textContent?.trim() || "All";
    }}

    function updateDailyStatus() {{
      const completed = completedSet();
      const allSegments = allSegmentSet();
      const completedSegments = completedSegmentSet();
      const remaining = Math.max(allSegments.size - completedSegments.size, 0);
      if (remainingSegmentCount) {{
        remainingSegmentCount.textContent = String(remaining);
      }}
      if (totalSegmentCount) {{
        totalSegmentCount.textContent = String(allSegments.size);
      }}
      if (completedOutingCount) {{
        completedOutingCount.textContent = String(cards.filter(card => fieldReady(card) && completed.has(card.dataset.outingId)).length);
      }}
      const filter = activeFilter();
      const candidates = cards
        .filter(fieldReady)
        .filter(card => !completed.has(card.dataset.outingId))
        .filter(card => filter === "all" || Number(card.dataset.minutes || 0) <= Number(filter))
        .map(card => {{
          const newSegments = segmentIdsForCard(card).filter(id => !completedSegments.has(id)).length;
          return {{
            card,
            completionSafe: card.dataset.completionSafe !== "false",
            newSegments,
            minutes: Number(card.dataset.minutes || 0),
            title: card.querySelector("h2")?.textContent?.trim() || card.dataset.outingId,
          }};
        }})
        .filter(item => item.newSegments > 0)
        .sort((left, right) => Number(right.completionSafe) - Number(left.completionSafe) || right.newSegments - left.newSegments || left.minutes - right.minutes);
      if (!bestTodayCopy) {{
        return;
      }}
      if (!candidates.length) {{
        bestTodayCopy.textContent = filter === "all"
          ? "No remaining runnable outings in this packet."
          : `No remaining runnable outings fit the ${{activeFilterLabel()}} door-to-door window.`;
        return;
      }}
      const best = candidates[0];
      const windowText = filter === "all" ? "the current list" : `${{activeFilterLabel()}} door to door`;
      const safetyText = best.completionSafe ? "completion-safe in the current menu" : "needs recertification review";
      bestTodayCopy.textContent = `Best today for ${{windowText}}: ${{best.title}} · ${{best.minutes}} min p75 · ${{best.newSegments}} new official segment(s) · ${{safetyText}}. Pin it before you start.`;
    }}

    function applyFilter() {{
      const filter = activeFilter();
      cards.forEach(card => {{
        const minutes = Number(card.dataset.minutes || 0);
        const isActive = card.classList.contains("active-outing");
        card.style.display = isActive || filter === "all" || minutes <= Number(filter) ? "" : "none";
      }});
      updateDailyStatus();
    }}

    buttons.forEach(button => button.addEventListener("click", () => {{
      buttons.forEach(item => item.classList.toggle("active", item === button));
      applyFilter();
    }}));

    cards.forEach(card => {{
      card.querySelector('[data-complete-action="mark"]')?.addEventListener("click", () => {{
        const completed = completedSet();
        completed.add(card.dataset.outingId);
        saveCompleted(completed);
        syncCompletedState();
      }});
      card.querySelector('[data-complete-action="undo"]')?.addEventListener("click", () => {{
        const completed = completedSet();
        completed.delete(card.dataset.outingId);
        saveCompleted(completed);
        syncCompletedState();
      }});
      card.querySelector('[data-active-action="pin"]')?.addEventListener("click", () => {{
        saveActive(card.dataset.outingId);
        syncActiveState(true);
      }});
      card.querySelectorAll(".steps li").forEach(step => {{
        step.addEventListener("click", () => {{
          card.querySelectorAll(".steps li.current-step").forEach(item => item.classList.remove("current-step"));
          step.classList.add("current-step");
        }});
        step.addEventListener("keydown", event => {{
          if (event.key === "Enter" || event.key === " ") {{
            event.preventDefault();
            step.click();
          }}
        }});
      }});
    }});

    clearActive.addEventListener("click", () => {{
      saveActive("");
      syncActiveState();
    }});

    completedToggle.addEventListener("click", () => {{
      document.body.classList.toggle("hide-completed");
      completedToggle.textContent = document.body.classList.contains("hide-completed") ? "Show completed" : "Hide completed";
    }});

    screenshotToggle.addEventListener("click", () => {{
      document.body.classList.toggle("screenshot");
    }});

    resetCompleted.addEventListener("click", () => {{
      localStorage.removeItem(STORAGE_KEY);
      syncCompletedState();
    }});

    exportProgress.addEventListener("click", () => {{
      const payload = {{
        schema: "boise_trails_phone_progress_v1",
        exported_at: new Date().toISOString(),
        completed_outing_ids: [...completedSet()].sort(),
        completed_segment_ids: [],
        provisional_completed_segment_ids: [...completedSegmentSet()].sort((left, right) => left.length - right.length || left.localeCompare(right)),
        missed_segment_ids: [],
        note: "Phone card taps are provisional UX state. Apply validated completed_segment_ids only after activity geometry proves endpoint-to-endpoint segment coverage."
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "boise-trails-progress.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }});

    if ("serviceWorker" in navigator) {{
      navigator.serviceWorker.register("service-worker.js").then(registration => {{
        offlineStatus.textContent = "Offline cache installed.";
        return registration;
      }}).catch(() => {{
        offlineStatus.textContent = "Offline cache unavailable; keep GPX downloaded.";
      }});
    }} else {{
      offlineStatus.textContent = "Offline cache unsupported; keep GPX downloaded.";
    }}

    window.addEventListener("online", () => offlineStatus.textContent = "Online.");
    window.addEventListener("offline", () => offlineStatus.textContent = "Offline. Cached cards and GPX remain available.");
    syncCompletedState();
    syncActiveState();
  </script>
</body>
</html>
"""


def render_live_map_html(asset_version: str = "") -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#0f172a">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Trail Live Map">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <link rel="manifest" href="manifest.webmanifest">
  <link rel="apple-touch-icon" href="icons/icon-192.png">
  <link rel="icon" href="icons/icon-192.png">
  <title>Live GPS Route Map</title>
  <style>
    :root { color-scheme: light; }
    html { height:100%; overflow:hidden; }
    * { box-sizing: border-box; }
    body { margin:0; height:100%; overflow:hidden; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#e8ece4; color:#111827; }
    .app { height:100dvh; min-height:100vh; overflow:hidden; display:grid; grid-template-rows:auto minmax(0,1fr) auto; }
    header { padding:calc(10px + env(safe-area-inset-top)) 12px 10px; background:#0f172a; color:#fff; box-shadow:0 2px 12px rgba(15,23,42,.18); z-index:2; }
    h1 { margin:0 0 8px; font-size:19px; letter-spacing:0; }
    .controls { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; }
    select,button { min-height:38px; border:1px solid #cbd5e1; border-radius:7px; background:#fff; color:#111827; font-weight:800; font-size:14px; }
    select { width:100%; min-width:0; padding:0 8px; }
    button { padding:0 10px; }
    .button-row { margin-top:8px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
    .button-row button.active { background:#2563eb; color:#fff; border-color:#2563eb; }
    .map-shell { position:relative; min-height:0; overflow:hidden; background:#f7f6ef; }
    svg { width:100%; height:100%; display:block; touch-action:none; }
    .basemap-tile { opacity:.56; }
    .grid-line { stroke:#d8ded2; stroke-width:1; vector-effect:non-scaling-stroke; }
    .route-halo { fill:none; stroke:#fff; stroke-width:18; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .route-line { fill:none; stroke:#2563eb; stroke-width:8; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .route-context { fill:none; stroke:#fff; stroke-width:14; stroke-linecap:round; stroke-linejoin:round; opacity:.7; vector-effect:non-scaling-stroke; }
    .route-context-gradient { opacity:.52; }
    .active-halo { fill:none; stroke:#fff; stroke-width:22; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .active-line { fill:none; stroke:#2563eb; stroke-width:10; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .route-slice { fill:none; stroke-width:9; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .cue-leg { fill:none; stroke-width:9; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .chevron { fill:none; stroke:#111827; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .direction-arrow { fill:#111827; stroke:#fff; stroke-width:2.5; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .cue-dot { fill:#111827; stroke:#fff; stroke-width:3; vector-effect:non-scaling-stroke; }
    .cue-dot.active { fill:#2563eb; }
    .cue-dot.next { fill:#16a34a; }
    .cue-dot.context-marker { fill:#64748b; opacity:.28; stroke-width:2; }
    .cue-anchor { fill:#fff; stroke:#111827; stroke-width:2.2; vector-effect:non-scaling-stroke; }
    .cue-anchor.active { stroke:#2563eb; }
    .cue-anchor.next { stroke:#16a34a; }
    .cue-callout-line { stroke:#fff; stroke-width:7; stroke-linecap:round; vector-effect:non-scaling-stroke; }
    .cue-callout-line.dark { stroke:#111827; stroke-width:1.5; }
    .cue-label { fill:#fff; font-size:13px; font-weight:900; text-anchor:middle; dominant-baseline:central; pointer-events:none; }
    .cue-label.context-label { display:none; }
    .leg-tag { fill:#111827; font-size:15px; font-weight:950; paint-order:stroke; stroke:#fff; stroke-width:6; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .endpoint-anchor { fill:#fff; stroke:#111827; stroke-width:2.2; vector-effect:non-scaling-stroke; }
    .endpoint-callout-line { stroke:#fff; stroke-width:7; stroke-linecap:round; vector-effect:non-scaling-stroke; }
    .endpoint-callout-line.dark { stroke:#111827; stroke-width:1.5; }
    .parking-dot { fill:#166534; stroke:#fff; stroke-width:4; vector-effect:non-scaling-stroke; }
    .finish-dot { fill:#b91c1c; stroke:#fff; stroke-width:4; vector-effect:non-scaling-stroke; }
    .marker-tag { fill:#111827; font-size:13px; font-weight:900; paint-order:stroke; stroke:#fff; stroke-width:5; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .user-accuracy { fill:#2563eb; opacity:.12; stroke:#2563eb; stroke-width:1; vector-effect:non-scaling-stroke; }
    .user-dot { fill:#0ea5e9; stroke:#fff; stroke-width:5; vector-effect:non-scaling-stroke; }
    .user-heading { fill:#0f172a; stroke:#fff; stroke-width:2; vector-effect:non-scaling-stroke; }
    .user-offscreen { fill:#0ea5e9; stroke:#fff; stroke-width:4; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .user-offscreen-label { fill:#0f172a; font-size:13px; font-weight:900; paint-order:stroke; stroke:#fff; stroke-width:5; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .map-leg-banner { position:absolute; left:10px; right:10px; top:10px; z-index:1; border:1px solid #bfdbfe; border-radius:8px; background:rgba(239,246,255,.94); color:#111827; padding:8px 46px 8px 10px; font-size:13px; font-weight:850; line-height:1.25; box-shadow:0 2px 10px rgba(15,23,42,.10); }
    .map-leg-banner b { color:#1d4ed8; }
    .map-leg-banner .leg-warning { display:block; margin-top:4px; color:#9a3412; font-weight:900; }
    .map-leg-banner-close { position:absolute; top:6px; right:6px; width:32px; min-width:32px; height:32px; min-height:32px; padding:0; border-color:#bfdbfe; border-radius:999px; background:#fff; color:#0f172a; font-size:14px; font-weight:950; line-height:1; box-shadow:0 1px 4px rgba(15,23,42,.12); }
    .route-blocked-warning { margin-top:8px; border:1px solid #fecaca; border-radius:8px; background:#fef2f2; color:#7f1d1d; padding:8px; font-size:13px; font-weight:850; line-height:1.3; }
    .gap-warning { position:absolute; left:10px; right:10px; top:10px; z-index:1; border:1px solid #fed7aa; border-radius:8px; background:#fff7ed; color:#9a3412; padding:8px 10px; font-size:13px; font-weight:800; box-shadow:0 2px 10px rgba(15,23,42,.12); }
    .map-tools { position:absolute; right:10px; bottom:calc(10px + env(safe-area-inset-bottom)); display:grid; gap:6px; }
    .map-tools button { min-width:44px; min-height:44px; box-shadow:0 2px 8px rgba(15,23,42,.16); }
    .map-tools button.active { background:#2563eb; color:#fff; border-color:#2563eb; }
    .tile-attribution { position:absolute; left:8px; bottom:calc(8px + env(safe-area-inset-bottom)); max-width:52%; padding:3px 5px; border-radius:5px; background:rgba(255,255,255,.78); color:#334155; font-size:10px; line-height:1.2; box-shadow:0 1px 5px rgba(15,23,42,.12); }
    .tile-attribution a { color:#1d4ed8; font-weight:800; }
    footer { padding:8px 12px calc(8px + env(safe-area-inset-bottom)); background:rgba(255,255,255,.96); border-top:1px solid #d7ddd4; }
    .cue-card { min-height:46px; border:1px solid #d7ddd4; border-radius:8px; padding:8px; background:#fff; }
    .cue-card b { display:block; font-size:12px; text-transform:uppercase; color:#475467; }
    .cue-card span { display:block; margin-top:2px; color:#111827; font-size:14px; font-weight:800; line-height:1.25; }
    .cue-controls { margin-top:8px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
    .cue-controls button { min-height:36px; font-size:13px; }
    .overview-link { min-height:44px; display:flex; align-items:center; justify-content:center; margin-top:8px; border:1px solid #bfdbfe; border-radius:8px; background:#eff6ff; color:#1d4ed8; font-size:14px; font-weight:950; text-align:center; text-decoration:none; }
    .overview-link:focus-visible,.overview-link:hover { text-decoration:underline; background:#dbeafe; }
    .note { margin-top:6px; color:#667085; font-size:12px; line-height:1.35; }
    @media (min-width:800px) {
      .app { grid-template-columns:380px 1fr; grid-template-rows:1fr auto; }
      header { grid-row:1 / span 2; }
      footer { grid-column:2; }
    }
    @media (max-width:900px) {
      header { padding:calc(8px + env(safe-area-inset-top)) 10px 8px; }
      h1 { font-size:17px; margin-bottom:6px; }
      select,button { min-height:34px; font-size:13px; }
      .button-row { margin-top:6px; gap:5px; }
      .note { display:none; }
      .map-leg-banner { font-size:12px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Live GPS Route Map</h1>
      <div class="controls">
        <select id="route-select" aria-label="Choose outing"></select>
        <button type="button" id="locate-button">Start GPS</button>
      </div>
      <div id="route-blocked-warning" class="route-blocked-warning" hidden></div>
      <p class="note">Field cue-leg map. The blue ribbon is the active cue-to-cue leg; muted lines are surrounding route context. Use the GitHub Pages HTTPS URL or a local web server; direct file:// cannot load GPX data.</p>
    </header>
    <main class="map-shell">
      <svg id="map-svg" role="img" aria-label="Live route map">
        <g id="tile-layer"></g>
        <g id="grid-layer"></g>
        <g id="route-layer"></g>
        <g id="marker-layer"></g>
        <g id="user-layer"></g>
      </svg>
      <div id="map-leg-banner" class="map-leg-banner" hidden>
        <button type="button" id="map-leg-banner-close" class="map-leg-banner-close" aria-label="Hide cue banner">X</button>
        <div id="map-leg-banner-content"></div>
      </div>
      <div id="tile-attribution" class="tile-attribution" hidden></div>
      <div class="map-tools">
        <button type="button" id="basemap-button" class="active" aria-pressed="true">OSM</button>
        <button type="button" id="fit-button">Fit</button>
        <button type="button" id="zoom-in">+</button>
        <button type="button" id="zoom-out">-</button>
      </div>
    </main>
    <footer>
      <div class="cue-card">
        <b>Active leg / nearest cue</b>
        <span id="nearest-cue">Load an outing, then start GPS.</span>
        <div class="cue-controls">
          <button type="button" id="previous-cue">Prev cue</button>
          <button type="button" id="fit-leg">Fit leg</button>
          <button type="button" id="next-cue">Next cue</button>
        </div>
      </div>
      <div class="button-row" aria-label="Route display">
        <button type="button" class="active" data-style="ribbon">Ribbon</button>
        <button type="button" data-style="cue-legs">Cue legs</button>
        <button type="button" id="show-all-route" aria-pressed="false">Show all</button>
      </div>
      <a class="overview-link" href="index.html">Return to overview</a>
    </footer>
  </div>
  <script>
    const FIELD_DATA_URL = "__FIELD_TOOL_DATA__";
    const DEFAULT_ASSET_VERSION = "__ASSET_VERSION__";
    const pageParams = new URLSearchParams(window.location.search);
    const ASSET_VERSION = pageParams.get("v") || DEFAULT_ASSET_VERSION;
    const ACTIVE_KEY = "__ACTIVE_KEY__";
    const state = {
      routes: [],
      route: null,
      trackSegments: [],
      waypoints: [],
      displayedSegments: [],
      contextSegments: [],
      contextLaneSpacingM: null,
      contextStyle: "",
      projectedSegments: [],
      projected: [],
      routePositions: [],
      cumulativeM: [],
      totalRouteM: 0,
      viewBox: null,
      baseViewBox: null,
      style: "ribbon",
      showAllRoute: false,
      basemap: "osm",
      geoBounds: null,
      geoScale: null,
      activeCueIndex: 0,
      dismissedMapLegBannerKey: "",
      watchId: null,
      user: null
    };
    const TILE_BASEMAPS = {
      osm: {
        label: "OSM",
        attribution: '<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap contributors</a>',
        urlForTile: (z, x, y) => `https://tile.openstreetmap.org/${z}/${x}/${y}.png`
      },
      r2r: {
        label: "R2R",
        attribution: "R2R / Ada County imagery",
        urlForTile: (z, x, y) => `https://tiles.arcgis.com/tiles/ocXK9Gg6fvIH5GTf/arcgis/rest/services/FoothillsMosaic2025/MapServer/tile/${z}/${y}/${x}`
      }
    };
    const BASEMAP_SEQUENCE = ["osm", "r2r", null];
    const activePointers = new Map();
    const gesture = { mode: null, lastPoint: null, lastCenter: null, lastDistance: 0 };
    const svg = document.getElementById("map-svg");
    const routeSelect = document.getElementById("route-select");
    const routeLayer = document.getElementById("route-layer");
    const markerLayer = document.getElementById("marker-layer");
    const userLayer = document.getElementById("user-layer");
    const tileLayer = document.getElementById("tile-layer");
    const gridLayer = document.getElementById("grid-layer");
    const mapLegBanner = document.getElementById("map-leg-banner");
    const mapLegBannerContent = document.getElementById("map-leg-banner-content");
    const mapLegBannerClose = document.getElementById("map-leg-banner-close");
    const routeBlockedWarning = document.getElementById("route-blocked-warning");
    const tileAttribution = document.getElementById("tile-attribution");
    const nearestCue = document.getElementById("nearest-cue");
    const locateButton = document.getElementById("locate-button");
    const previousCue = document.getElementById("previous-cue");
    const nextCue = document.getElementById("next-cue");
    const fitLegButton = document.getElementById("fit-leg");
    const fitButton = document.getElementById("fit-button");
    const basemapButton = document.getElementById("basemap-button");
    const showAllRouteButton = document.getElementById("show-all-route");

    function miles(meters) { return meters / 1609.344; }
    function metersFromMiles(value) { return Number(value || 0) * 1609.344; }
    function fmtDistance(meters) {
      if (!Number.isFinite(meters)) return "--";
      if (meters < 160) return `${Math.round(meters)} m`;
      return `${miles(meters).toFixed(2)} mi`;
    }
    function cardRouteTotalM() {
      return metersFromMiles(state.route?.on_foot_miles) || state.totalRouteM || 0;
    }
    function routeMToCardM(routeM) {
      const geometryTotal = state.totalRouteM || 0;
      const cardTotal = cardRouteTotalM();
      if (!geometryTotal || !cardTotal) return 0;
      return Math.max(0, Math.min(routeM, geometryTotal)) * cardTotal / geometryTotal;
    }
    function cardMilesToRouteM(value) {
      const cardM = metersFromMiles(value);
      const cardTotal = cardRouteTotalM();
      const geometryTotal = state.totalRouteM || 0;
      if (!cardTotal || !geometryTotal) return Math.max(0, Math.min(cardM, geometryTotal));
      return Math.max(0, Math.min(cardM * geometryTotal / cardTotal, geometryTotal));
    }
    function fmtProgress(routeM) {
      const total = cardRouteTotalM();
      if (!total) return "--";
      return `${miles(routeMToCardM(routeM)).toFixed(2)} / ${miles(total).toFixed(2)} mi`;
    }
    function escapeText(value) {
      return String(value ?? "").replace(/[&<>"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
    }
    function routeHeld(route) {
      return String(route?.field_readiness_status || route?.route_card_status || "").startsWith("blocked_") ||
        Boolean((route?.special_management?.failures || []).length);
    }
    function routeHeldMessage(route) {
      const status = String(route?.field_readiness_status || route?.route_card_status || "");
      const blockers = route?.route_card_audit_blockers || [];
      if (status === "blocked_navigation_source") {
        return `Route held: source route/cue geometry mismatch. Do not run this route from the field packet until the canonical route is repaired and recertified. ${blockers[0] || ""}`.trim();
      }
      const failures = route?.special_management?.failures || [];
      const first = failures[0] || {};
      const code = first.code || "special-management rule violation";
      const rule = first.rule_id ? ` ${first.rule_id}` : "";
      const target = first.segment_id ? ` segment ${first.segment_id}` : "";
      if (status === "blocked_special_management" || failures.length) {
        return `Route held: special-management rule violation. Do not run this route from the field packet until redesigned and recertified. ${code}${rule}${target}`.trim();
      }
      return `Route held: not field-ready. Do not run this route from the field packet until repaired and recertified. ${blockers[0] || ""}`.trim();
    }
    function updateRouteHeldState() {
      const held = routeHeld(state.route);
      locateButton.disabled = routeHeld(state.route);
      locateButton.textContent = held ? "Route held" : (state.watchId === null ? "Start GPS" : "Stop GPS");
      routeBlockedWarning.hidden = !held;
      routeBlockedWarning.textContent = held ? routeHeldMessage(state.route) : "";
      if (held && state.watchId !== null) {
        navigator.geolocation.clearWatch(state.watchId);
        state.watchId = null;
      }
    }
    function versionedAssetUrl(href) {
      if (!ASSET_VERSION) return href;
      const url = new URL(href, window.location.href);
      url.searchParams.set("v", ASSET_VERSION);
      return url.href;
    }
    function gpxNodes(xml, localName) {
      return [...xml.getElementsByTagNameNS("*", localName)];
    }
    function parseGpx(text) {
      const xml = new DOMParser().parseFromString(text, "application/xml");
      const trackSegments = gpxNodes(xml, "trkseg").map(segment => (
        [...segment.getElementsByTagNameNS("*", "trkpt")].map(node => ({
          lat: Number(node.getAttribute("lat")),
          lon: Number(node.getAttribute("lon"))
        })).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon))
      )).filter(segment => segment.length > 0);
      if (!trackSegments.length) {
        const fallbackTrack = [...xml.getElementsByTagNameNS("*", "trkpt")].map(node => ({
          lat: Number(node.getAttribute("lat")),
          lon: Number(node.getAttribute("lon"))
        })).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon));
        if (fallbackTrack.length) trackSegments.push(fallbackTrack);
      }
      const waypoints = gpxNodes(xml, "wpt").map(node => ({
        lat: Number(node.getAttribute("lat")),
        lon: Number(node.getAttribute("lon")),
        name: node.getElementsByTagNameNS("*", "name")[0]?.textContent || "",
        desc: node.getElementsByTagNameNS("*", "desc")[0]?.textContent || ""
      })).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon));
      return { trackSegments, waypoints };
    }
    function boundsFor(points) {
      const lats = points.map(point => point.lat);
      const lons = points.map(point => point.lon);
      return {
        minLat: Math.min(...lats),
        maxLat: Math.max(...lats),
        minLon: Math.min(...lons),
        maxLon: Math.max(...lons)
      };
    }
    function makeProjector(points) {
      const bounds = boundsFor(points);
      const lat0 = ((bounds.minLat + bounds.maxLat) / 2) * Math.PI / 180;
      const cosLat = Math.max(Math.cos(lat0), 0.01);
      state.geoBounds = bounds;
      state.geoScale = { lon: 111320 * cosLat, lat: 110540 };
      return point => ({
        x: (point.lon - bounds.minLon) * state.geoScale.lon,
        y: (bounds.maxLat - point.lat) * state.geoScale.lat
      });
    }
    function latLonForPoint(point) {
      if (!state.geoBounds || !state.geoScale || !point) return null;
      return {
        lat: state.geoBounds.maxLat - point.y / state.geoScale.lat,
        lon: state.geoBounds.minLon + point.x / state.geoScale.lon
      };
    }
    function pathFor(points) {
      if (!points.length) return "";
      return points.map((point, index) => `${index ? "L" : "M"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
    }
    function smoothPathFor(points) {
      if (points.length <= 2) return pathFor(points);
      const commands = [`M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`];
      for (let index = 1; index < points.length - 1; index += 1) {
        const current = points[index];
        const next = points[index + 1];
        const midX = (current.x + next.x) / 2;
        const midY = (current.y + next.y) / 2;
        commands.push(`Q ${current.x.toFixed(1)} ${current.y.toFixed(1)} ${midX.toFixed(1)} ${midY.toFixed(1)}`);
      }
      const last = points[points.length - 1];
      commands.push(`L ${last.x.toFixed(1)} ${last.y.toFixed(1)}`);
      return commands.join(" ");
    }
    function quadraticPoint(start, control, end, t) {
      const inverse = 1 - t;
      return {
        x: inverse * inverse * start.x + 2 * inverse * t * control.x + t * t * end.x,
        y: inverse * inverse * start.y + 2 * inverse * t * control.y + t * t * end.y,
        routeM: inverse * inverse * (start.routeM || 0) + 2 * inverse * t * (control.routeM || 0) + t * t * (end.routeM || 0)
      };
    }
    function smoothPolylinePoints(points, steps = 6) {
      if (points.length <= 2) return points.map(point => ({ ...point }));
      const output = [{ ...points[0] }];
      let cursor = points[0];
      for (let index = 1; index < points.length - 1; index += 1) {
        const control = points[index];
        const next = points[index + 1];
        const end = {
          x: (control.x + next.x) / 2,
          y: (control.y + next.y) / 2,
          routeM: ((control.routeM || 0) + (next.routeM || 0)) / 2
        };
        for (let step = 1; step <= steps; step += 1) {
          output.push(quadraticPoint(cursor, control, end, step / steps));
        }
        cursor = end;
      }
      const last = points[points.length - 1];
      for (let step = 1; step <= Math.max(2, Math.floor(steps / 2)); step += 1) {
        const t = step / Math.max(2, Math.floor(steps / 2));
        output.push({
          x: cursor.x + (last.x - cursor.x) * t,
          y: cursor.y + (last.y - cursor.y) * t,
          routeM: (cursor.routeM || 0) + ((last.routeM || 0) - (cursor.routeM || 0)) * t
        });
      }
      return output;
    }
    function smoothSegmentsForDisplay(segments) {
      return segments.map(segment => smoothPolylinePoints(segment));
    }
    function pathForSegments(segments, options = {}) {
      const renderer = options.smooth ? smoothPathFor : pathFor;
      return segments.map(renderer).filter(Boolean).join(" ");
    }
    function distance(a, b) {
      return Math.hypot(a.x - b.x, a.y - b.y);
    }
    function isBacktrackingTurn(previous, current, next) {
      const ax = current.x - previous.x;
      const ay = current.y - previous.y;
      const bx = next.x - current.x;
      const by = next.y - current.y;
      const aLength = Math.hypot(ax, ay);
      const bLength = Math.hypot(bx, by);
      if (aLength < 4 || bLength < 4) return false;
      const cosine = (ax * bx + ay * by) / (aLength * bLength);
      const turnbackWidth = distance(previous, next);
      return cosine < -0.78 && turnbackWidth <= Math.max(18, Math.min(aLength, bLength) * 0.75);
    }
    function splitBacktrackingDisplaySegments(segments) {
      return segments.flatMap(segment => {
        if (segment.length <= 2) return [segment];
        const parts = [];
        let part = [segment[0]];
        for (let index = 1; index < segment.length - 1; index += 1) {
          const previous = segment[index - 1];
          const current = segment[index];
          const next = segment[index + 1];
          part.push(current);
          if (isBacktrackingTurn(previous, current, next)) {
            parts.push(part);
            part = [current];
          }
        }
        part.push(segment[segment.length - 1]);
        if (part.length > 1) parts.push(part);
        return parts;
      });
    }
    function perpendicularDistance(point, lineStart, lineEnd) {
      const dx = lineEnd.x - lineStart.x;
      const dy = lineEnd.y - lineStart.y;
      const len2 = dx * dx + dy * dy;
      if (!len2) return distance(point, lineStart);
      const t = Math.max(0, Math.min(1, ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / len2));
      return distance(point, { x: lineStart.x + dx * t, y: lineStart.y + dy * t });
    }
    function simplifyPolyline(points, tolerance = 6) {
      if (points.length <= 2) return points;
      let maxDistance = 0;
      let splitIndex = 0;
      for (let index = 1; index < points.length - 1; index += 1) {
        const candidateDistance = perpendicularDistance(points[index], points[0], points[points.length - 1]);
        if (candidateDistance > maxDistance) {
          maxDistance = candidateDistance;
          splitIndex = index;
        }
      }
      if (maxDistance <= tolerance) return [points[0], points[points.length - 1]];
      const left = simplifyPolyline(points.slice(0, splitIndex + 1), tolerance);
      const right = simplifyPolyline(points.slice(splitIndex), tolerance);
      return [...left.slice(0, -1), ...right];
    }
    const SCHEMATIC_CORRIDOR_KEY_M = 6;
    const SCHEMATIC_LANE_SPACING_PX = 9;
    const SCHEMATIC_MIN_LANE_SPACING_M = 1;
    const SCHEMATIC_MAX_LANE_SPACING_M = 10;
    const SCHEMATIC_CONTEXT_TOLERANCE_M = 14;
    const SCHEMATIC_COLOR_STEP_M = 80;
    const SCHEMATIC_COLOR_OVERLAP_M = 8;
    function canonicalIntervalVector(a, b) {
      const forward = a.x < b.x || (a.x === b.x && a.y <= b.y);
      const start = forward ? a : b;
      const end = forward ? b : a;
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const length = Math.hypot(dx, dy) || 1;
      return { start, end, forward, dx, dy, length };
    }
    function corridorKey(a, b) {
      const midX = Math.round(((a.x + b.x) / 2) / SCHEMATIC_CORRIDOR_KEY_M);
      const midY = Math.round(((a.y + b.y) / 2) / SCHEMATIC_CORRIDOR_KEY_M);
      const vector = canonicalIntervalVector(a, b);
      const angle = Math.atan2(vector.dy, vector.dx);
      const bucket = Math.round(angle / (Math.PI / 12));
      return `${midX}:${midY}:${bucket}`;
    }
    function offsetRepeatedCorridors(projectedSegments, laneSpacingM = SCHEMATIC_MAX_LANE_SPACING_M) {
      const intervalRecords = [];
      const counts = new Map();
      const baselines = new Map();
      projectedSegments.forEach((segment, segmentIndex) => {
        for (let pointIndex = 1; pointIndex < segment.length; pointIndex += 1) {
          const start = segment[pointIndex - 1];
          const end = segment[pointIndex];
          const key = corridorKey(start, end);
          const vector = canonicalIntervalVector(start, end);
          const record = { segmentIndex, pointIndex, key, vector };
          intervalRecords.push(record);
          counts.set(key, (counts.get(key) || 0) + 1);
          const baseline = baselines.get(key) || { startX: 0, startY: 0, endX: 0, endY: 0, count: 0 };
          baseline.startX += vector.start.x;
          baseline.startY += vector.start.y;
          baseline.endX += vector.end.x;
          baseline.endY += vector.end.y;
          baseline.count += 1;
          baselines.set(key, baseline);
        }
      });
      const seen = new Map();
      const targets = projectedSegments.map(segment => segment.map(() => ({ x: 0, y: 0, count: 0 })));
      function addTarget(segmentIndex, pointIndex, target) {
        const pointTarget = targets[segmentIndex][pointIndex];
        pointTarget.x += target.x;
        pointTarget.y += target.y;
        pointTarget.count += 1;
      }
      intervalRecords.forEach(record => {
        const total = counts.get(record.key) || 0;
        if (total <= 1) return;
        const baseline = baselines.get(record.key);
        if (!baseline?.count) return;
        const canonicalStart = { x: baseline.startX / baseline.count, y: baseline.startY / baseline.count };
        const canonicalEnd = { x: baseline.endX / baseline.count, y: baseline.endY / baseline.count };
        const baselineDx = canonicalEnd.x - canonicalStart.x;
        const baselineDy = canonicalEnd.y - canonicalStart.y;
        const baselineLength = Math.hypot(baselineDx, baselineDy) || record.vector.length;
        const occurrence = seen.get(record.key) || 0;
        seen.set(record.key, occurrence + 1);
        const laneOffset = (occurrence - (total - 1) / 2) * laneSpacingM;
        const nx = -baselineDy / baselineLength;
        const ny = baselineDx / baselineLength;
        const dx = nx * laneOffset;
        const dy = ny * laneOffset;
        const alignedStart = record.vector.forward ? canonicalStart : canonicalEnd;
        const alignedEnd = record.vector.forward ? canonicalEnd : canonicalStart;
        addTarget(record.segmentIndex, record.pointIndex - 1, { x: alignedStart.x + dx, y: alignedStart.y + dy });
        addTarget(record.segmentIndex, record.pointIndex, { x: alignedEnd.x + dx, y: alignedEnd.y + dy });
      });
      return projectedSegments.map((segment, segmentIndex) => segment.map((point, pointIndex) => {
        const target = targets[segmentIndex][pointIndex];
        if (!target.count) return { ...point };
        return {
          ...point,
          x: target.x / target.count,
          y: target.y / target.count
        };
      }));
    }
    function currentSchematicLaneSpacingM() {
      const rect = svg.getBoundingClientRect();
      const metersPerPixel = rect?.width ? (state.viewBox?.w || 1) / rect.width : SCHEMATIC_MAX_LANE_SPACING_M / SCHEMATIC_LANE_SPACING_PX;
      return Math.max(SCHEMATIC_MIN_LANE_SPACING_M, Math.min(SCHEMATIC_MAX_LANE_SPACING_M, metersPerPixel * SCHEMATIC_LANE_SPACING_PX));
    }
    function refreshContextSegments(force = false) {
      const laneSpacingM = currentSchematicLaneSpacingM();
      if (!force && state.contextStyle === state.style && Number.isFinite(state.contextLaneSpacingM) && Math.abs(state.contextLaneSpacingM - laneSpacingM) < 0.25) {
        return;
      }
      state.contextLaneSpacingM = laneSpacingM;
      state.contextStyle = state.style;
      state.contextSegments = offsetRepeatedCorridors(state.projectedSegments, laneSpacingM).map(segment => (
        simplifyPolyline(segment, SCHEMATIC_CONTEXT_TOLERANCE_M)
      ));
    }
    function refreshDisplaySegments() {
      state.displayedSegments = state.projectedSegments.map(segment => (
        simplifyPolyline(segment, 5)
      ));
      refreshContextSegments(true);
    }
    function buildSegmentCumulative(projectedSegments) {
      const segmentCumulative = [];
      const routePositions = [];
      let total = 0;
      for (let segmentIndex = 0; segmentIndex < projectedSegments.length; segmentIndex += 1) {
        const segment = projectedSegments[segmentIndex];
        const values = [];
        for (let pointIndex = 0; pointIndex < segment.length; pointIndex += 1) {
          if (pointIndex > 0) total += distance(segment[pointIndex - 1], segment[pointIndex]);
          values.push(total);
          segment[pointIndex].routeM = total;
          segment[pointIndex].segmentIndex = segmentIndex;
          segment[pointIndex].pointIndex = pointIndex;
          routePositions.push({ ...segment[pointIndex] });
        }
        segmentCumulative.push(values);
      }
      return { segmentCumulative, routePositions, totalRouteM: total };
    }
    function positionForRouteM(routeM) {
      if (!state.projectedSegments.length) return null;
      const target = Math.max(0, Math.min(routeM, state.totalRouteM || 0));
      for (let segmentIndex = 0; segmentIndex < state.projectedSegments.length; segmentIndex += 1) {
        const segment = state.projectedSegments[segmentIndex];
        const cumulative = state.cumulativeM[segmentIndex] || [];
        if (!segment.length) continue;
        if (target <= cumulative[cumulative.length - 1] || segmentIndex === state.projectedSegments.length - 1) {
          for (let pointIndex = 1; pointIndex < segment.length; pointIndex += 1) {
            if (cumulative[pointIndex] >= target) {
              const before = cumulative[pointIndex - 1];
              const after = cumulative[pointIndex];
              const t = after === before ? 0 : (target - before) / (after - before);
              const a = segment[pointIndex - 1];
              const b = segment[pointIndex];
              return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, routeM: target, segmentIndex, pointIndex };
            }
          }
          const last = segment[segment.length - 1];
          return { ...last, routeM: target, segmentIndex, pointIndex: segment.length - 1 };
        }
      }
      const lastSegment = state.projectedSegments[state.projectedSegments.length - 1];
      const last = lastSegment[lastSegment.length - 1];
      return { ...last, routeM: target, segmentIndex: state.projectedSegments.length - 1, pointIndex: lastSegment.length - 1 };
    }
    function displayedRoutePositionForM(routeM, options = {}) {
      const displaySource = options.segments || (options.context ? state.contextSegments : state.displayedSegments);
      if (!displaySource.length) return null;
      const target = Math.max(0, Math.min(routeM, state.totalRouteM || 0));
      let fallback = null;
      for (let segmentIndex = 0; segmentIndex < displaySource.length; segmentIndex += 1) {
        const segment = displaySource[segmentIndex];
        if (!segment.length) continue;
        if (!fallback) fallback = { ...segment[0], angle: 0 };
        for (let pointIndex = 1; pointIndex < segment.length; pointIndex += 1) {
          const a = segment[pointIndex - 1];
          const b = segment[pointIndex];
          const aM = a.routeM || 0;
          const bM = b.routeM || 0;
          if (target < Math.min(aM, bM) || target > Math.max(aM, bM)) continue;
          const span = bM - aM;
          const t = span === 0 ? 0 : (target - aM) / span;
          return {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            routeM: target,
            angle: Math.atan2(b.y - a.y, b.x - a.x),
            segmentIndex,
            pointIndex
          };
        }
        fallback = { ...segment[segment.length - 1], angle: fallback.angle || 0 };
      }
      return fallback;
    }
    function tangentForRouteM(routeM) {
      const position = positionForRouteM(routeM);
      if (!position) return null;
      const segment = state.projectedSegments[position.segmentIndex] || [];
      const next = segment[Math.min(position.pointIndex + 1, segment.length - 1)];
      const previous = segment[Math.max(position.pointIndex - 1, 0)];
      if (!next || !previous) return null;
      return Math.atan2(next.y - previous.y, next.x - previous.x);
    }
    function interpolateRoutePoint(a, b, routeM) {
      const span = (b.routeM || 0) - (a.routeM || 0);
      const t = span === 0 ? 0 : (routeM - (a.routeM || 0)) / span;
      return {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        routeM
      };
    }
    function projectPointToRoute(point) {
      if (!state.projectedSegments.length) return null;
      const projectedUser = state.project(point);
      let best = null;
      for (let segmentIndex = 0; segmentIndex < state.projectedSegments.length; segmentIndex += 1) {
        const segment = state.projectedSegments[segmentIndex];
        const cumulative = state.cumulativeM[segmentIndex] || [];
        for (let pointIndex = 1; pointIndex < segment.length; pointIndex += 1) {
          const a = segment[pointIndex - 1];
          const b = segment[pointIndex];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const len2 = dx * dx + dy * dy || 1;
          const t = Math.max(0, Math.min(1, ((projectedUser.x - a.x) * dx + (projectedUser.y - a.y) * dy) / len2));
          const snap = { x: a.x + t * dx, y: a.y + t * dy };
          const dist = distance(projectedUser, snap);
          if (!best || dist < best.distanceM) {
            const legM = distance(a, b) * t;
            best = { point: projectedUser, snap, distanceM: dist, routeM: cumulative[pointIndex - 1] + legM, segmentIndex, pointIndex };
          }
        }
      }
      return best;
    }
    function cueForRouteM(routeM) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return null;
      const next = cues.find(cue => metersFromMiles(cue.cum_miles) >= routeM - 30);
      return next || cues[cues.length - 1];
    }
    function cueLabel(cue) {
      if (!cue) return "No cue data for this route.";
      const seq = String(cue.seq || "").padStart(2, "0");
      const typeLabels = { overlap_repeat: "OVERLAP" };
      const rawType = String(cue.cue_type || "cue");
      const type = typeLabels[rawType] || rawType.replaceAll("_", " ").toUpperCase();
      const action = cue.action ? `${cue.action}: ` : "";
      const target = cue.target ? ` toward ${cue.target}` : "";
      return `${seq} ${type} - ${action}${(cue.signed_as || []).join(" / ")}${target}`;
    }
    function cueWarning(cue) {
      return String(cue?.field_warning || "").trim();
    }
    function cueSeq(cue, fallbackIndex) {
      return String(cue?.seq || fallbackIndex + 1).padStart(2, "0");
    }
    function cueRouteM(cue) {
      const routeMiles = Number(cue?.route_miles);
      if (Number.isFinite(routeMiles)) return Math.max(0, Math.min(metersFromMiles(routeMiles), state.totalRouteM || 0));
      return cardMilesToRouteM(cue?.cum_miles);
    }
    function cueEndRouteM(cue) {
      const startM = cueRouteM(cue);
      const routeLegMiles = Number(cue?.route_leg_miles);
      if (Number.isFinite(routeLegMiles)) return Math.max(startM, Math.min(startM + metersFromMiles(routeLegMiles), state.totalRouteM || 0));
      return Math.max(startM, Math.min(startM + metersFromMiles(cue?.leg_miles || 0), state.totalRouteM || 0));
    }
    function activeLegRange(index = state.activeCueIndex) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return { startM: 0, endM: state.totalRouteM || 0, cue: null, nextCue: null, index: 0, nextIndex: null };
      const clamped = Math.max(0, Math.min(index, cues.length - 1));
      const cue = cues[clamped];
      const startM = cueRouteM(cue);
      let endM = cueEndRouteM(cue) || state.totalRouteM || 0;
      let nextIndex = null;
      for (let candidateIndex = clamped + 1; candidateIndex < cues.length; candidateIndex += 1) {
        const candidateM = cueRouteM(cues[candidateIndex]);
        if (candidateM > startM + 8) {
          endM = candidateM;
          nextIndex = candidateIndex;
          break;
        }
      }
      if (endM <= startM + 8 && (state.totalRouteM || 0) > startM + 8) endM = state.totalRouteM;
      return { startM, endM, cue, nextCue: nextIndex === null ? null : cues[nextIndex], index: clamped, nextIndex };
    }
    function visibleRouteStartM() {
      return state.showAllRoute ? 0 : activeContextStartM();
    }
    function cueIndexForRouteM(routeM) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return 0;
      let active = 0;
      for (let index = 0; index < cues.length; index += 1) {
        if (cueRouteM(cues[index]) <= routeM + 25) active = index;
      }
      return active;
    }
    function requestedCueIndex() {
      const cues = state.route?.wayfinding_cues || [];
      const requested = Number(pageParams.get("cue"));
      if (!cues.length || !Number.isFinite(requested)) return null;
      const seqIndex = cues.findIndex(cue => Number(cue.seq) === requested);
      if (seqIndex >= 0) return seqIndex;
      if (requested >= 1 && requested <= cues.length) return requested - 1;
      return null;
    }
    function nextDistinctCueIndex(index = state.activeCueIndex) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return 0;
      const leg = activeLegRange(index);
      if (leg.nextIndex !== null) return leg.nextIndex;
      return Math.min(index + 1, cues.length - 1);
    }
    function previousDistinctCueIndex(index = state.activeCueIndex) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return 0;
      const currentM = cueRouteM(cues[Math.max(0, Math.min(index, cues.length - 1))]);
      for (let candidateIndex = Math.min(index - 1, cues.length - 1); candidateIndex >= 0; candidateIndex -= 1) {
        if (cueRouteM(cues[candidateIndex]) < currentM - 8) return candidateIndex;
      }
      return index;
    }
    function activeContextStartM(index = state.activeCueIndex) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return 0;
      const clamped = Math.max(0, Math.min(index, cues.length - 1));
      const previousIndex = previousDistinctCueIndex(clamped);
      return cueRouteM(cues[previousIndex] || cues[clamped]);
    }
    function activeLegMilesText(leg) {
      const routeMiles = miles(Math.max(0, leg.endM - leg.startM));
      const cueMiles = Number(leg?.cue?.leg_miles);
      if (!Number.isFinite(cueMiles) || cueMiles <= 0) return `+${routeMiles.toFixed(2)} mi`;
      const tolerance = Math.max(0.15, cueMiles * 0.2);
      if (Math.abs(routeMiles - cueMiles) <= tolerance) return `+${cueMiles.toFixed(2)} mi`;
      return `+${cueMiles.toFixed(2)} mi cue · map +${routeMiles.toFixed(2)} mi`;
    }
    function nextCueIndexAfterRouteM(routeM) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return null;
      for (let index = 0; index < cues.length; index += 1) {
        if (cueRouteM(cues[index]) > routeM + 25) return index;
      }
      return null;
    }
    function cuePointForIndex(index) {
      const cues = state.route?.wayfinding_cues || [];
      if (index === null || !cues[index]) return null;
      const cueM = cueRouteM(cues[index]);
      return displayedRoutePositionForM(cueM, { context: true }) || displayedRoutePositionForM(cueM) || positionForRouteM(cueM);
    }
    function updateActiveCuePanel() {
      if (routeHeld(state.route)) {
        nearestCue.textContent = routeHeldMessage(state.route);
        previousCue.disabled = true;
        nextCue.disabled = true;
        updateRouteHeldState();
        updateMapLegBanner();
        return;
      }
      const cues = state.route?.wayfinding_cues || [];
      const leg = activeLegRange();
      if (!cues.length || !leg.cue) {
        nearestCue.textContent = "No cue data for this route.";
      } else {
        const currentSeq = cueSeq(leg.cue, leg.index);
        const nextSeq = leg.nextCue ? cueSeq(leg.nextCue, leg.nextIndex) : "finish";
        const legMiles = activeLegMilesText(leg);
        const warning = cueWarning(leg.cue);
        nearestCue.textContent = `Cue ${currentSeq} -> ${nextSeq} · ${legMiles} · ${cueLabel(leg.cue)}${warning ? ` · ${warning}` : ""}`;
      }
      previousCue.disabled = !cues.length || previousDistinctCueIndex() === state.activeCueIndex;
      nextCue.disabled = !cues.length || nextDistinctCueIndex() === state.activeCueIndex;
      updateMapLegBanner();
    }
    function mapLegBannerKey(leg) {
      if (!leg?.cue) return "";
      const fromSeq = cueSeq(leg.cue, leg.index);
      const toSeq = leg.nextCue ? cueSeq(leg.nextCue, leg.nextIndex) : "finish";
      return `${state.route?.outing_id || ""}:${fromSeq}:${toSeq}`;
    }
    function updateMapLegBanner() {
      const leg = activeLegRange();
      if (!leg.cue) {
        mapLegBanner.hidden = true;
        mapLegBannerContent.textContent = "";
        return;
      }
      const bannerKey = mapLegBannerKey(leg);
      if (state.dismissedMapLegBannerKey === bannerKey) {
        mapLegBanner.hidden = true;
        mapLegBannerContent.textContent = "";
        return;
      }
      const fromSeq = cueSeq(leg.cue, leg.index);
      const toSeq = leg.nextCue ? cueSeq(leg.nextCue, leg.nextIndex) : "finish";
      const signedAs = (leg.cue.signed_as || []).join(" / ") || "active route";
      const target = leg.cue.target ? ` to ${escapeText(leg.cue.target)}` : "";
      const until = leg.cue.until ? ` until ${escapeText(leg.cue.until)}` : "";
      const bannerAction = escapeText(String(leg.cue.action || "FOLLOW").toUpperCase());
      const warning = cueWarning(leg.cue);
      const warningHtml = warning ? ` <span class="leg-warning">${escapeText(warning)}</span>` : "";
      mapLegBanner.hidden = false;
      mapLegBannerContent.innerHTML = `<b>${bannerAction} ${fromSeq} -> ${toSeq}</b>: ${escapeText(signedAs)}${target}${until}.${warningHtml}`;
    }
    function setActiveCueIndex(index, options = {}) {
      const cues = state.route?.wayfinding_cues || [];
      state.activeCueIndex = cues.length ? Math.max(0, Math.min(index, cues.length - 1)) : 0;
      updateActiveCuePanel();
      if (options.fit) fitActiveLeg(Boolean(state.user));
      if (options.render !== false) render();
    }
    function setViewBox(box) {
      state.viewBox = box;
      svg.setAttribute("viewBox", `${box.x} ${box.y} ${box.w} ${box.h}`);
    }
    function svgPointFromClient(clientX, clientY) {
      const box = state.viewBox || state.baseViewBox;
      const rect = svg.getBoundingClientRect();
      if (!box || !rect.width || !rect.height) return null;
      return {
        x: box.x + ((clientX - rect.left) / rect.width) * box.w,
        y: box.y + ((clientY - rect.top) / rect.height) * box.h
      };
    }
    function panViewBox(deltaX, deltaY) {
      const box = state.viewBox || state.baseViewBox;
      if (!box) return;
      setViewBox({ x: box.x + deltaX, y: box.y + deltaY, w: box.w, h: box.h });
    }
    function zoomAt(scale, center) {
      const box = state.viewBox || state.baseViewBox;
      if (!box || !center || !Number.isFinite(scale) || scale <= 0) return;
      const nextW = Math.max(40, Math.min(box.w * scale, 20000));
      const nextH = Math.max(40, Math.min(box.h * scale, 20000));
      const widthScale = nextW / box.w;
      const heightScale = nextH / box.h;
      setViewBox({
        x: center.x - (center.x - box.x) * widthScale,
        y: center.y - (center.y - box.y) * heightScale,
        w: nextW,
        h: nextH
      });
    }
    function pointerDistance(a, b) {
      return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    }
    function pointerCenter(a, b) {
      return { clientX: (a.clientX + b.clientX) / 2, clientY: (a.clientY + b.clientY) / 2 };
    }
    function resetGestureFromPointers() {
      const pointers = [...activePointers.values()];
      if (pointers.length === 1) {
        gesture.mode = "pan";
        gesture.lastPoint = svgPointFromClient(pointers[0].clientX, pointers[0].clientY);
        gesture.lastCenter = null;
        gesture.lastDistance = 0;
      } else if (pointers.length >= 2) {
        const center = pointerCenter(pointers[0], pointers[1]);
        gesture.mode = "pinch";
        gesture.lastPoint = null;
        gesture.lastCenter = svgPointFromClient(center.clientX, center.clientY);
        gesture.lastDistance = pointerDistance(pointers[0], pointers[1]);
      } else {
        gesture.mode = null;
        gesture.lastPoint = null;
        gesture.lastCenter = null;
        gesture.lastDistance = 0;
      }
    }
    function mapUnitsPerPixel() {
      const box = state.viewBox || state.baseViewBox;
      const width = svg.clientWidth || 1;
      const height = svg.clientHeight || 1;
      if (!box) return 1;
      return Math.max(box.w / width, box.h / height);
    }
    function pointInViewBox(point, marginPixels = 0) {
      const box = state.viewBox || state.baseViewBox;
      if (!box || !point) return false;
      const margin = marginPixels * mapUnitsPerPixel();
      return point.x >= box.x + margin && point.x <= box.x + box.w - margin &&
        point.y >= box.y + margin && point.y <= box.y + box.h - margin;
    }
    function offscreenUserIndicator(point, nearest) {
      const box = state.viewBox || state.baseViewBox;
      if (!box || !point) return "";
      const unit = mapUnitsPerPixel();
      const margin = 26 * unit;
      const marker = {
        x: Math.max(box.x + margin, Math.min(box.x + box.w - margin, point.x)),
        y: Math.max(box.y + margin, Math.min(box.y + box.h - margin, point.y))
      };
      const angle = Math.atan2(point.y - marker.y, point.x - marker.x);
      const tip = { x: marker.x + Math.cos(angle) * 13 * unit, y: marker.y + Math.sin(angle) * 13 * unit };
      const left = { x: marker.x + Math.cos(angle + 2.45) * 10 * unit, y: marker.y + Math.sin(angle + 2.45) * 10 * unit };
      const right = { x: marker.x + Math.cos(angle - 2.45) * 10 * unit, y: marker.y + Math.sin(angle - 2.45) * 10 * unit };
      const label = nearest ? `GPS off map · ${fmtDistance(nearest.distanceM)} to route` : "GPS off map";
      return `<g><title>${escapeText(label)}</title>` +
        `<path class="user-offscreen" d="M ${tip.x.toFixed(1)} ${tip.y.toFixed(1)} L ${left.x.toFixed(1)} ${left.y.toFixed(1)} L ${right.x.toFixed(1)} Z" />` +
        `<text class="user-offscreen-label" x="${(marker.x + 18 * unit).toFixed(1)}" y="${(marker.y + 5 * unit).toFixed(1)}" font-size="${(13 * unit).toFixed(1)}" stroke-width="${(5 * unit).toFixed(1)}">GPS off map</text>` +
        `</g>`;
    }
    function updateFitButtonLabel() {
      fitButton.textContent = state.user ? "Fit GPS" : "Fit";
    }
    function fitPoints(points, minSize = 120, padFactor = 0.18, padBase = 80) {
      const usable = points.filter(Boolean);
      if (!usable.length) return;
      const xs = usable.map(point => point.x);
      const ys = usable.map(point => point.y);
      const width = Math.max(Math.max(...xs) - Math.min(...xs), minSize);
      const height = Math.max(Math.max(...ys) - Math.min(...ys), minSize);
      const pad = Math.max(width, height) * padFactor + padBase;
      const box = { x: Math.min(...xs) - pad, y: Math.min(...ys) - pad, w: width + pad * 2, h: height + pad * 2 };
      setViewBox(box);
    }
    function fitRoute(includeUser = false) {
      const points = [...state.projected];
      if (includeUser && state.user) points.push(state.project(state.user));
      if (!points.length) return;
      fitPoints(points, 120, 0.18, 80);
      const box = state.viewBox;
      state.baseViewBox = box;
    }
    function fitActiveLeg(includeUser = false) {
      const leg = activeLegRange();
      const contextStartM = activeContextStartM(leg.index);
      const legSegments = segmentsForRouteRange(contextStartM, leg.endM, { context: true });
      const points = legSegments.flat();
      if (includeUser && state.user) points.push(state.project(state.user));
      if (!points.length) return fitRoute(includeUser);
      fitPoints(points, 70, 0.35, 70);
    }
    function fitGpsToNextCue() {
      if (!state.user) return fitRoute(false);
      const userPoint = state.project(state.user);
      const nearest = projectPointToRoute(state.user);
      const nextIndex = nearest ? nextCueIndexAfterRouteM(nearest.routeM) : activeLegRange().nextIndex;
      const nextCuePoint = cuePointForIndex(nextIndex);
      const finishPoint = displayedRoutePositionForM(state.totalRouteM, { context: true }) || displayedRoutePositionForM(state.totalRouteM) || positionForRouteM(state.totalRouteM);
      fitPoints([userPoint, nextCuePoint || finishPoint], 90, 0.42, 85);
    }
    function zoom(factor) {
      const box = state.viewBox || state.baseViewBox;
      if (!box) return;
      zoomAt(factor, { x: box.x + box.w / 2, y: box.y + box.h / 2 });
    }
    function clampTileLat(lat) {
      return Math.max(-85.05112878, Math.min(85.05112878, lat));
    }
    function tileXYForLatLon(lat, lon, zoomLevel) {
      const n = 2 ** zoomLevel;
      const latRad = clampTileLat(lat) * Math.PI / 180;
      return {
        x: Math.floor(((lon + 180) / 360) * n),
        y: Math.floor(((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n)
      };
    }
    function latLonForTileXY(x, y, zoomLevel) {
      const n = 2 ** zoomLevel;
      const lon = x / n * 360 - 180;
      const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n)));
      return { lat: latRad * 180 / Math.PI, lon };
    }
    function tileZoomForView(lonSpan) {
      const targetTileCount = Math.max(2, Math.min(8, (svg.clientWidth || 640) / 220));
      return Math.max(11, Math.min(17, Math.round(Math.log2((360 * targetTileCount) / Math.max(lonSpan, 0.0001)))));
    }
    function updateBasemapButton() {
      const basemap = state.basemap ? TILE_BASEMAPS[state.basemap] : null;
      basemapButton.textContent = basemap ? basemap.label : "No map";
      basemapButton.classList.toggle("active", Boolean(basemap));
      basemapButton.setAttribute("aria-pressed", basemap ? "true" : "false");
    }
    function cycleBasemap() {
      const index = BASEMAP_SEQUENCE.indexOf(state.basemap);
      state.basemap = BASEMAP_SEQUENCE[(index + 1) % BASEMAP_SEQUENCE.length];
      updateBasemapButton();
      render();
    }
    function drawTiles() {
      tileLayer.innerHTML = "";
      const basemap = state.basemap ? TILE_BASEMAPS[state.basemap] : null;
      tileAttribution.hidden = true;
      if (!basemap || !state.geoBounds || !state.geoScale) return;
      const box = state.viewBox || state.baseViewBox;
      if (!box) return;
      const northwest = latLonForPoint({ x: box.x, y: box.y });
      const southeast = latLonForPoint({ x: box.x + box.w, y: box.y + box.h });
      if (!northwest || !southeast) return;
      const minLon = Math.max(-180, Math.min(northwest.lon, southeast.lon));
      const maxLon = Math.min(180, Math.max(northwest.lon, southeast.lon));
      const minLat = clampTileLat(Math.min(northwest.lat, southeast.lat));
      const maxLat = clampTileLat(Math.max(northwest.lat, southeast.lat));
      let zoomLevel = tileZoomForView(maxLon - minLon);
      let minTile;
      let maxTile;
      const maxTileImages = 72;
      while (zoomLevel >= 11) {
        minTile = tileXYForLatLon(maxLat, minLon, zoomLevel);
        maxTile = tileXYForLatLon(minLat, maxLon, zoomLevel);
        const tileCount = (maxTile.x - minTile.x + 1) * (maxTile.y - minTile.y + 1);
        if (tileCount <= maxTileImages || zoomLevel === 11) break;
        zoomLevel -= 1;
      }
      const images = [];
      for (let x = minTile.x; x <= maxTile.x; x += 1) {
        for (let y = minTile.y; y <= maxTile.y; y += 1) {
          const tileNorthwest = latLonForTileXY(x, y, zoomLevel);
          const tileSoutheast = latLonForTileXY(x + 1, y + 1, zoomLevel);
          const start = state.project(tileNorthwest);
          const end = state.project(tileSoutheast);
          images.push(`<image class="basemap-tile" href="${basemap.urlForTile(zoomLevel, x, y)}" x="${start.x.toFixed(1)}" y="${start.y.toFixed(1)}" width="${(end.x - start.x).toFixed(1)}" height="${(end.y - start.y).toFixed(1)}" preserveAspectRatio="none" />`);
        }
      }
      tileLayer.innerHTML = images.join("");
      tileAttribution.innerHTML = basemap.attribution;
      tileAttribution.hidden = !images.length;
      updateBasemapButton();
    }
    function drawGrid() {
      const box = state.viewBox || state.baseViewBox;
      if (!box) return;
      const spacing = Math.max(100, Math.round(Math.max(box.w, box.h) / 6 / 50) * 50);
      const lines = [];
      for (let x = Math.floor(box.x / spacing) * spacing; x < box.x + box.w; x += spacing) {
        lines.push(`<line class="grid-line" x1="${x}" y1="${box.y}" x2="${x}" y2="${box.y + box.h}" />`);
      }
      for (let y = Math.floor(box.y / spacing) * spacing; y < box.y + box.h; y += spacing) {
        lines.push(`<line class="grid-line" x1="${box.x}" y1="${y}" x2="${box.x + box.w}" y2="${y}" />`);
      }
      gridLayer.innerHTML = lines.join("");
    }
    const MAX_OVERVIEW_CHEVRONS = 18;
    function chevrons(maxCount = MAX_OVERVIEW_CHEVRONS, startM = 0, endM = state.totalRouteM) {
      const total = state.totalRouteM || 0;
      const span = Math.max(0, Math.min(endM, total) - Math.max(0, startM));
      if (!span) return "";
      const spacing = Math.max(120, span / Math.max(maxCount, 1));
      const items = [];
      for (let target = startM + spacing; target < endM; target += spacing) {
        const center = positionForRouteM(target);
        const angle = tangentForRouteM(target);
        if (!center || angle === null) continue;
        const size = 14;
        const wing = Math.PI * 0.78;
        const p1 = { x: center.x - Math.cos(angle - wing) * size, y: center.y - Math.sin(angle - wing) * size };
        const p2 = { x: center.x - Math.cos(angle + wing) * size, y: center.y - Math.sin(angle + wing) * size };
        items.push(`<path class="chevron" d="M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} L ${center.x.toFixed(1)} ${center.y.toFixed(1)} L ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}" />`);
      }
      return items.join("");
    }
    function displaySegmentLength(segment) {
      let length = 0;
      for (let index = 1; index < segment.length; index += 1) {
        length += distance(segment[index - 1], segment[index]);
      }
      return length;
    }
    function displayPointAtDistance(segment, targetDistance) {
      let walked = 0;
      for (let index = 1; index < segment.length; index += 1) {
        const previous = segment[index - 1];
        const current = segment[index];
        const span = distance(previous, current);
        if (!span) continue;
        if (walked + span >= targetDistance) {
          const local = Math.max(0, Math.min(1, (targetDistance - walked) / span));
          return {
            x: previous.x + (current.x - previous.x) * local,
            y: previous.y + (current.y - previous.y) * local,
            angle: Math.atan2(current.y - previous.y, current.x - previous.x)
          };
        }
        walked += span;
      }
      const last = segment[segment.length - 1];
      const prior = segment[Math.max(0, segment.length - 2)];
      return last && prior ? { x: last.x, y: last.y, angle: Math.atan2(last.y - prior.y, last.x - prior.x) } : null;
    }
    function directionArrowPath(center, angle, unit) {
      const size = 15 * unit;
      const baseAngle = angle - Math.PI;
      const tip = { x: center.x + Math.cos(angle) * size * 0.72, y: center.y + Math.sin(angle) * size * 0.72 };
      const left = { x: center.x + Math.cos(baseAngle - 0.48) * size, y: center.y + Math.sin(baseAngle - 0.48) * size };
      const right = { x: center.x + Math.cos(baseAngle + 0.48) * size, y: center.y + Math.sin(baseAngle + 0.48) * size };
      return `<path class="direction-arrow" d="M ${tip.x.toFixed(1)} ${tip.y.toFixed(1)} L ${left.x.toFixed(1)} ${left.y.toFixed(1)} L ${right.x.toFixed(1)} ${right.y.toFixed(1)} Z" />`;
    }
    function activeLegArrows(startM, endM, options = {}) {
      const span = Math.max(0, endM - startM);
      if (span < 80) return "";
      const unit = mapUnitsPerPixel();
      const displaySegments = options.segments || [];
      if (displaySegments.length) {
        const items = [];
        const targetSpacing = 130 * unit;
        for (const segment of displaySegments) {
          if (segment.length < 2) continue;
          const length = displaySegmentLength(segment);
          if (length < 55 * unit) continue;
          const arrowCount = Math.max(1, Math.min(8, Math.floor(length / Math.max(targetSpacing, 1))));
          for (let index = 1; index <= arrowCount && items.length < 36; index += 1) {
            const sample = displayPointAtDistance(segment, (length * index) / (arrowCount + 1));
            if (!sample || sample.angle === null) continue;
            items.push(directionArrowPath(sample, sample.angle, unit));
          }
        }
        return items.join("");
      }
      const arrowSpacing = 145;
      const inset = Math.min(90, span * 0.16);
      const items = [];
      let count = 0;
      for (let target = startM + inset; target < endM - inset && count < 28; target += arrowSpacing) {
        const sample = displayedRoutePositionForM(target, options);
        if (!sample || sample.angle === null) continue;
        const center = sample;
        const angle = sample.angle;
        items.push(directionArrowPath(center, angle, unit));
        count += 1;
      }
      return items.join("");
    }
    const ROUTE_GRADIENT_STOPS = [
      { at: 0, color: [220, 38, 38] },
      { at: 0.33, color: [234, 179, 8] },
      { at: 0.66, color: [22, 163, 74] },
      { at: 1, color: [21, 128, 61] }
    ];
    function routeColorAt(progress) {
      const p = Math.max(0, Math.min(1, progress));
      for (let index = 1; index < ROUTE_GRADIENT_STOPS.length; index += 1) {
        const left = ROUTE_GRADIENT_STOPS[index - 1];
        const right = ROUTE_GRADIENT_STOPS[index];
        if (p <= right.at) {
          const local = right.at === left.at ? 0 : (p - left.at) / (right.at - left.at);
          const color = left.color.map((value, channel) => Math.round(value + (right.color[channel] - value) * local));
          return `rgb(${color[0]} ${color[1]} ${color[2]})`;
        }
      }
      const last = ROUTE_GRADIENT_STOPS[ROUTE_GRADIENT_STOPS.length - 1].color;
      return `rgb(${last[0]} ${last[1]} ${last[2]})`;
    }
    function segmentsForRouteRange(startM, endM, options = {}) {
      const output = [];
      const displaySource = options.context ? state.contextSegments : state.displayedSegments;
      for (const segment of displaySource) {
        const slice = [];
        for (let index = 1; index < segment.length; index += 1) {
          const a = segment[index - 1];
          const b = segment[index];
          const aM = a.routeM || 0;
          const bM = b.routeM || 0;
          if (bM < startM || aM > endM) continue;
          const start = Math.max(aM, startM);
          const end = Math.min(bM, endM);
          const startPoint = start === aM ? a : interpolateRoutePoint(a, b, start);
          const endPoint = end === bM ? b : interpolateRoutePoint(a, b, end);
          if (!slice.length || distance(slice[slice.length - 1], startPoint) > 0.1) slice.push(startPoint);
          slice.push(endPoint);
        }
        if (slice.length > 1) output.push(slice);
      }
      return splitBacktrackingDisplaySegments(output);
    }
    function drawProgressRibbon(startAtM = 0) {
      refreshContextSegments();
      const total = state.totalRouteM || 1;
      const slices = [];
      const stepM = SCHEMATIC_COLOR_STEP_M;
      const firstM = Math.max(0, Math.min(startAtM, state.totalRouteM || 0));
      for (let startM = firstM; startM < state.totalRouteM; startM += stepM) {
        const endM = Math.min(state.totalRouteM, startM + stepM + SCHEMATIC_COLOR_OVERLAP_M);
        const mid = Math.min(state.totalRouteM, startM + stepM / 2);
        const chunkSegments = segmentsForRouteRange(startM, endM, { context: true });
        const chunkPath = pathForSegments(chunkSegments, { smooth: true });
        if (!chunkPath) continue;
        slices.push(`<path class="route-slice" stroke="${routeColorAt(mid / total)}" d="${chunkPath}" />`);
      }
      return slices.join("");
    }
    function drawRoute() {
      refreshContextSegments();
      const cueLegColors = ["#2563eb", "#06b6d4", "#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7", "#0f766e"];
      const visibleStartM = visibleRouteStartM();
      const visibleSegments = segmentsForRouteRange(visibleStartM, state.totalRouteM, { context: true });
      const fullPath = pathForSegments(visibleSegments, { smooth: true });
      let routeHtml = fullPath ? `<path class="route-context" d="${fullPath}" />` : "";
      if (state.style === "cue-legs") {
        const cueStops = (state.route?.wayfinding_cues || [])
          .map(cue => cueRouteM(cue))
          .filter(value => value >= visibleStartM - 8);
        const stops = [...new Set([visibleStartM, ...cueStops, state.totalRouteM])]
          .filter(value => Number.isFinite(value) && value >= visibleStartM - 8 && value <= state.totalRouteM)
          .sort((a, b) => a - b);
        routeHtml += `<g class="route-context-gradient">`;
        for (let index = 1; index < stops.length; index += 1) {
          const legSegments = segmentsForRouteRange(stops[index - 1], stops[index], { context: true });
          routeHtml += legSegments.map(leg => `<path class="cue-leg" stroke="${cueLegColors[index % cueLegColors.length]}" d="${smoothPathFor(leg)}" />`).join("");
        }
        routeHtml += `</g>`;
      } else if (state.style === "ribbon") {
        routeHtml += `<g class="route-context-gradient">${drawProgressRibbon(visibleStartM)}</g>`;
      }
      const leg = activeLegRange();
      const activeSegments = smoothSegmentsForDisplay(segmentsForRouteRange(leg.startM, leg.endM, { context: true }));
      const activePath = pathForSegments(activeSegments);
      if (activePath) {
        routeHtml += `<path class="active-halo" d="${activePath}" /><path class="active-line" d="${activePath}" />`;
      }
      routeLayer.innerHTML = routeHtml + activeLegArrows(leg.startM, leg.endM, { segments: activeSegments });
    }
    function drawMarkers() {
      const placed = [];
      const leg = activeLegRange();
      const visibleStartM = visibleRouteStartM();
      const unit = mapUnitsPerPixel();
      const cueMarkers = (state.route?.wayfinding_cues || [])
        .map((cue, index) => {
          const cueM = cueRouteM(cue);
          const isActive = index === state.activeCueIndex;
          const isNext = index === leg.nextIndex;
          if (!state.showAllRoute && cueM < visibleStartM - 8 && !isActive && !isNext) return "";
          const point = displayedRoutePositionForM(cueM, { context: isActive || isNext }) || displayedRoutePositionForM(cueM) || positionForRouteM(cueM);
          if (!point) return "";
          const nearby = placed.filter(existing => distance(existing, point) < 30).length;
          placed.push(point);
          const number = String(cue.seq || index + 1).padStart(2, "0");
          const title = escapeText(cue.compact || cueLabel(cue));
          const isCallout = isActive || isNext;
          const calloutAngle = (point.angle || 0) + (isActive ? -Math.PI / 2 : Math.PI / 2);
          const calloutDistance = isCallout ? 44 * unit : nearby ? 24 * unit : 0;
          const fallbackAngle = nearby * Math.PI * 0.75;
          const markerAngle = isCallout ? calloutAngle : fallbackAngle;
          const marker = { x: point.x + Math.cos(markerAngle) * calloutDistance, y: point.y + Math.sin(markerAngle) * calloutDistance };
          const roleClass = isActive ? " active" : isNext ? " next" : " context-marker";
          const labelClass = isActive || isNext ? "cue-label" : "cue-label context-label";
          const anchorClass = isActive ? " active" : isNext ? " next" : "";
          const radiusForCue = (isActive || isNext ? 16 : 6) * unit;
          const anchorRadius = (isActive || isNext ? 5 : 0) * unit;
          const labelFontSize = (isActive || isNext ? 16 : 10) * unit;
          const tagFontSize = 18 * unit;
          const tagStrokeWidth = 6 * unit;
          const legTag = isActive || isNext
            ? `<text class="leg-tag" x="${(marker.x + 22 * unit).toFixed(1)}" y="${(marker.y - 21 * unit).toFixed(1)}" font-size="${tagFontSize.toFixed(1)}" stroke-width="${tagStrokeWidth.toFixed(1)}">${isActive ? "FROM" : "NEXT"} ${escapeText(number)}</text>`
            : "";
          const anchor = anchorRadius
            ? `<circle class="cue-anchor${anchorClass}" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${anchorRadius.toFixed(1)}" />`
            : "";
          return `<g><title>${title}</title>${anchor}<line class="cue-callout-line" x1="${point.x.toFixed(1)}" y1="${point.y.toFixed(1)}" x2="${marker.x.toFixed(1)}" y2="${marker.y.toFixed(1)}" /><line class="cue-callout-line dark" x1="${point.x.toFixed(1)}" y1="${point.y.toFixed(1)}" x2="${marker.x.toFixed(1)}" y2="${marker.y.toFixed(1)}" /><circle class="cue-dot${roleClass}" cx="${marker.x.toFixed(1)}" cy="${marker.y.toFixed(1)}" r="${radiusForCue.toFixed(1)}" /><text class="${labelClass}" x="${marker.x.toFixed(1)}" y="${marker.y.toFixed(1)}" font-size="${labelFontSize.toFixed(1)}">${escapeText(number)}</text>${legTag}</g>`;
        });
      const first = displayedRoutePositionForM(0) || positionForRouteM(0);
      const last = displayedRoutePositionForM(state.totalRouteM) || positionForRouteM(state.totalRouteM);
      const sameStartFinish = first && last && distance(first, last) < 25;
      function endpointCallout(point, label, kind, dx, dy) {
        const endpointAnchorRadius = 5 * unit;
        const endpointMarkerRadius = 11 * unit;
        const marker = { x: point.x + dx * unit, y: point.y + dy * unit };
        const textX = marker.x + 17 * unit;
        const textY = marker.y + 5 * unit;
        const tagFontSize = 14 * unit;
        const tagStrokeWidth = 5 * unit;
        const mainClass = kind === "finish" ? "finish-dot" : "parking-dot";
        const finishDot = kind === "both"
          ? `<circle class="finish-dot" cx="${(marker.x + 15 * unit).toFixed(1)}" cy="${(marker.y + 3 * unit).toFixed(1)}" r="${(endpointMarkerRadius * 0.78).toFixed(1)}"><title>FINISH</title></circle>`
          : "";
        return `<g><title>${escapeText(label)}</title>` +
          `<circle class="endpoint-anchor" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${endpointAnchorRadius.toFixed(1)}" />` +
          `<line class="endpoint-callout-line" x1="${point.x.toFixed(1)}" y1="${point.y.toFixed(1)}" x2="${marker.x.toFixed(1)}" y2="${marker.y.toFixed(1)}" />` +
          `<line class="endpoint-callout-line dark" x1="${point.x.toFixed(1)}" y1="${point.y.toFixed(1)}" x2="${marker.x.toFixed(1)}" y2="${marker.y.toFixed(1)}" />` +
          `<circle class="${mainClass}" cx="${marker.x.toFixed(1)}" cy="${marker.y.toFixed(1)}" r="${endpointMarkerRadius.toFixed(1)}" />` +
          finishDot +
          `<text class="marker-tag" x="${textX.toFixed(1)}" y="${textY.toFixed(1)}" font-size="${tagFontSize.toFixed(1)}" stroke-width="${tagStrokeWidth.toFixed(1)}">${escapeText(label)}</text>` +
          `</g>`;
      }
      const endpointMarkers = sameStartFinish
        ? [endpointCallout(first, "START/FINISH", "both", 58, 42)]
        : [
            first ? endpointCallout(first, "START", "start", 42, 24) : "",
            last ? endpointCallout(last, "FINISH", "finish", 42, -24) : ""
          ];
      markerLayer.innerHTML = [
        ...endpointMarkers,
        ...cueMarkers
      ].join("");
    }
    function drawUser() {
      userLayer.innerHTML = "";
      updateFitButtonLabel();
      if (!state.user) return;
      const point = state.project(state.user);
      const nearest = projectPointToRoute(state.user);
      if (!pointInViewBox(point, 18)) {
        userLayer.innerHTML = offscreenUserIndicator(point, nearest);
        return;
      }
      const unit = mapUnitsPerPixel();
      const accuracy = Math.max(Number(state.user.accuracy || 0), 8);
      const heading = Number.isFinite(state.user.heading) ? state.user.heading : null;
      const headingTriangle = heading === null ? "" : (() => {
        const angle = (heading - 90) * Math.PI / 180;
        const tip = { x: point.x + Math.cos(angle) * 22 * unit, y: point.y + Math.sin(angle) * 22 * unit };
        const left = { x: point.x + Math.cos(angle + 2.5) * 13 * unit, y: point.y + Math.sin(angle + 2.5) * 13 * unit };
        const right = { x: point.x + Math.cos(angle - 2.5) * 13 * unit, y: point.y + Math.sin(angle - 2.5) * 13 * unit };
        return `<path class="user-heading" d="M ${tip.x.toFixed(1)} ${tip.y.toFixed(1)} L ${left.x.toFixed(1)} ${left.y.toFixed(1)} L ${right.x.toFixed(1)} ${right.y.toFixed(1)} Z" />`;
      })();
      userLayer.innerHTML = `<circle class="user-accuracy" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${accuracy.toFixed(1)}" />${headingTriangle}<circle class="user-dot" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${(10 * unit).toFixed(1)}" />`;
    }
    function render() {
      drawTiles();
      drawGrid();
      drawRoute();
      drawMarkers();
      drawUser();
    }
    async function loadRoute(route) {
      state.route = route;
      state.dismissedMapLegBannerKey = "";
      localStorage.setItem(ACTIVE_KEY, route.outing_id);
      routeSelect.value = route.outing_id;
      updateRouteHeldState();
      nearestCue.textContent = "Loading GPX...";
      const response = await fetch(versionedAssetUrl(route.gpx_href), { cache: "no-cache" });
      const gpx = parseGpx(await response.text());
      state.trackSegments = gpx.trackSegments;
      state.waypoints = gpx.waypoints;
      const flatTrack = state.trackSegments.flat();
      const points = [...flatTrack, ...state.waypoints];
      if (!flatTrack.length || !points.length) {
        nearestCue.textContent = "No track geometry found for this outing.";
        return;
      }
      state.project = makeProjector(points);
      state.projectedSegments = state.trackSegments.map(segment => segment.map(state.project));
      const metrics = buildSegmentCumulative(state.projectedSegments);
      state.cumulativeM = metrics.segmentCumulative;
      state.routePositions = metrics.routePositions;
      state.totalRouteM = metrics.totalRouteM;
      refreshDisplaySegments();
      state.activeCueIndex = 0;
      state.projected = state.routePositions;
      setActiveCueIndex(requestedCueIndex() ?? cueIndexForRouteM(0), { render: false });
      fitActiveLeg(false);
      updateRouteHeldState();
      render();
    }
    async function boot() {
      const response = await fetch(versionedAssetUrl(FIELD_DATA_URL), { cache: "no-cache" });
      const data = await response.json();
      state.routes = data.routes || [];
      routeSelect.innerHTML = state.routes.map(route => {
        const name = route.route_name || route.label || "Route";
        const code = route.route_code || route.label || "";
        const codeText = code && code !== name ? ` (${code})` : "";
        const heldText = routeHeld(route) ? "[HELD] " : "";
        return `<option value="${escapeText(route.outing_id)}">${escapeText(heldText)}${escapeText(name)}${escapeText(codeText)} · ${escapeText(route.trailhead_display || route.trailhead)}</option>`;
      }).join("");
      const preferred = pageParams.get("outing") || localStorage.getItem(ACTIVE_KEY) || state.routes[0]?.outing_id;
      const route = state.routes.find(item => item.outing_id === preferred) || state.routes[0];
      if (route) await loadRoute(route);
    }
    routeSelect.addEventListener("change", () => {
      const route = state.routes.find(item => item.outing_id === routeSelect.value);
      if (route) loadRoute(route);
    });
    document.querySelectorAll("[data-style]").forEach(button => button.addEventListener("click", () => {
      document.querySelectorAll("[data-style]").forEach(item => item.classList.toggle("active", item === button));
      state.style = button.dataset.style;
      refreshDisplaySegments();
      render();
    }));
    showAllRouteButton.addEventListener("click", () => {
      state.showAllRoute = !state.showAllRoute;
      showAllRouteButton.classList.toggle("active", state.showAllRoute);
      showAllRouteButton.setAttribute("aria-pressed", state.showAllRoute ? "true" : "false");
      render();
    });
    basemapButton.addEventListener("click", cycleBasemap);
    fitButton.addEventListener("click", () => { state.user ? fitGpsToNextCue() : fitRoute(false); render(); });
    fitLegButton.addEventListener("click", () => { fitActiveLeg(Boolean(state.user)); render(); });
    mapLegBannerClose.addEventListener("click", () => {
      const leg = activeLegRange();
      state.dismissedMapLegBannerKey = mapLegBannerKey(leg);
      mapLegBanner.hidden = true;
      mapLegBannerContent.textContent = "";
    });
    previousCue.addEventListener("click", () => setActiveCueIndex(previousDistinctCueIndex(), { fit: true }));
    nextCue.addEventListener("click", () => setActiveCueIndex(nextDistinctCueIndex(), { fit: true }));
    document.getElementById("zoom-in").addEventListener("click", () => { zoom(0.72); render(); });
    document.getElementById("zoom-out").addEventListener("click", () => { zoom(1.38); render(); });
    svg.addEventListener("pointerdown", event => {
      svg.setPointerCapture(event.pointerId);
      activePointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
      resetGestureFromPointers();
      event.preventDefault();
    });
    svg.addEventListener("pointermove", event => {
      if (!activePointers.has(event.pointerId)) return;
      activePointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
      const pointers = [...activePointers.values()];
      if (pointers.length === 1 && gesture.mode === "pan") {
        const current = svgPointFromClient(event.clientX, event.clientY);
        if (current && gesture.lastPoint) {
          panViewBox(gesture.lastPoint.x - current.x, gesture.lastPoint.y - current.y);
          gesture.lastPoint = svgPointFromClient(event.clientX, event.clientY);
          render();
        }
      } else if (pointers.length >= 2) {
        const centerClient = pointerCenter(pointers[0], pointers[1]);
        const currentDistance = pointerDistance(pointers[0], pointers[1]);
        const center = svgPointFromClient(centerClient.clientX, centerClient.clientY);
        if (gesture.mode !== "pinch" || !gesture.lastDistance || !gesture.lastCenter) {
          resetGestureFromPointers();
        } else if (currentDistance > 12 && center) {
          zoomAt(gesture.lastDistance / currentDistance, center);
          const recentered = svgPointFromClient(centerClient.clientX, centerClient.clientY);
          if (recentered && gesture.lastCenter) {
            panViewBox(gesture.lastCenter.x - recentered.x, gesture.lastCenter.y - recentered.y);
          }
          gesture.lastDistance = currentDistance;
          gesture.lastCenter = svgPointFromClient(centerClient.clientX, centerClient.clientY);
          render();
        }
      }
      event.preventDefault();
    });
    svg.addEventListener("pointerup", event => {
      activePointers.delete(event.pointerId);
      resetGestureFromPointers();
      event.preventDefault();
    });
    svg.addEventListener("pointercancel", event => {
      activePointers.delete(event.pointerId);
      resetGestureFromPointers();
      event.preventDefault();
    });
    svg.addEventListener("wheel", event => {
      const center = svgPointFromClient(event.clientX, event.clientY);
      zoomAt(event.deltaY < 0 ? 0.82 : 1.22, center);
      render();
      event.preventDefault();
    }, { passive: false });
    locateButton.addEventListener("click", () => {
      if (routeHeld(state.route)) {
        nearestCue.textContent = routeHeldMessage(state.route);
        updateRouteHeldState();
        return;
      }
      if (!navigator.geolocation) {
        nearestCue.textContent = "Geolocation is not available in this browser.";
        return;
      }
      if (state.watchId !== null) {
        navigator.geolocation.clearWatch(state.watchId);
        state.watchId = null;
        locateButton.textContent = "Start GPS";
        return;
      }
      locateButton.textContent = "Stop GPS";
      state.watchId = navigator.geolocation.watchPosition(position => {
        state.user = {
          lat: position.coords.latitude,
          lon: position.coords.longitude,
          accuracy: position.coords.accuracy,
          heading: position.coords.heading
        };
        if (!pointInViewBox(state.project(state.user), 18)) {
          nearestCue.textContent = "GPS acquired; tap Fit GPS to include your dot.";
        }
        render();
      }, error => {
        nearestCue.textContent = `GPS unavailable: ${error.message}`;
        locateButton.textContent = "Start GPS";
      }, { enableHighAccuracy: true, maximumAge: 4000, timeout: 15000 });
    });
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("service-worker.js").catch(() => {});
    }
    boot().catch(error => { nearestCue.textContent = `Unable to load field data: ${error.message}`; });
  </script>
</body>
</html>
""".replace("__FIELD_TOOL_DATA__", FIELD_TOOL_DATA_NAME).replace("__ACTIVE_KEY__", ACTIVE_STORAGE_KEY).replace(
        "__ASSET_VERSION__",
        asset_version,
    )


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", binascii.crc32(tag + payload) & 0xFFFFFFFF)
    )


def solid_png_bytes(size: int, rgb: tuple[int, int, int] = (17, 24, 39)) -> bytes:
    """Build a simple standards-compliant RGB PNG without optional dependencies."""
    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )


def render_web_manifest() -> str:
    manifest = {
        "name": "Boise Trails Field Packet",
        "short_name": "Trails Packet",
        "description": "Offline field cards and GPX routes for Boise Trails Challenge outings.",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "orientation": "portrait",
        "theme_color": "#111827",
        "background_color": "#f5f7f2",
        "icons": [
            {
                "src": "icons/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable",
            },
            {
                "src": "icons/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable",
            },
        ],
    }
    return json.dumps(manifest, indent=2) + "\n"


def render_service_worker(precache_urls: list[str], cache_name: str) -> str:
    urls = ["./"] + unique_nonempty_text(precache_urls)
    return f"""const CACHE_NAME = "{cache_name}";
const PRECACHE_URLS = {json.dumps(urls, indent=2)};
const NETWORK_FIRST_URLS = new Set([
  "index.html",
  "live-map.html",
  "field-tool-data.json",
  "manifest.json",
  "manifest.webmanifest"
]);

function normalizedCacheKey(request) {{
  const requestUrl = new URL(request.url);
  requestUrl.search = '';
  return requestUrl.href;
}}

function shouldUseNetworkFirst(request) {{
  const requestUrl = new URL(request.url);
  const filename = requestUrl.pathname.split('/').pop() || 'index.html';
  return NETWORK_FIRST_URLS.has(filename) || requestUrl.pathname.includes('/gpx/');
}}

self.addEventListener('install', event => {{
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
}});

self.addEventListener('activate', event => {{
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
    )).then(() => self.clients.claim())
  );
}});

self.addEventListener('fetch', event => {{
  if (event.request.method !== 'GET') {{
    return;
  }}
  const cacheKey = normalizedCacheKey(event.request);
  if (shouldUseNetworkFirst(event.request)) {{
    event.respondWith(
      fetch(event.request).then(response => {{
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(cacheKey, copy));
        return response;
      }}).catch(() => caches.match(cacheKey).then(cached => cached || caches.match('./index.html')))
    );
    return;
  }}
  event.respondWith(
    caches.match(event.request).then(cached => {{
      if (cached) {{
        return cached;
      }}
      return fetch(event.request).then(response => {{
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        return response;
      }}).catch(() => caches.match('./index.html'));
    }})
  );
}});
"""


def write_pwa_assets(
    output_dir: Path,
    routes: list[dict[str, Any]],
    zip_href: str,
    extra_precache_urls: list[str] | None = None,
) -> list[Path]:
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    icon_192 = icons_dir / "icon-192.png"
    icon_512 = icons_dir / "icon-512.png"
    icon_192.write_bytes(solid_png_bytes(192))
    icon_512.write_bytes(solid_png_bytes(512))
    web_manifest_path = output_dir / "manifest.webmanifest"
    web_manifest_path.write_text(render_web_manifest(), encoding="utf-8")
    precache_urls = [
        "index.html",
        "manifest.json",
        "manifest.webmanifest",
        "icons/icon-192.png",
        "icons/icon-512.png",
        zip_href,
    ]
    precache_urls.extend(extra_precache_urls or [])
    for route in routes:
        precache_urls.extend(route[key] for key in GPX_HREF_KEYS)
    service_worker_path = output_dir / "service-worker.js"
    digest = hashlib.sha256()
    for route in routes:
        for key in GPX_PATH_KEYS:
            digest.update(Path(route[key]).read_bytes())
    digest.update((output_dir / zip_href).read_bytes())
    for href in extra_precache_urls or []:
        extra_path = output_dir / href
        if extra_path.exists() and extra_path.is_file():
            digest.update(extra_path.read_bytes())
    cache_suffix = digest.hexdigest().replace("911", "nineoneone")[:18]
    cache_name = f"boise-trails-field-packet-v{len(routes)}-{cache_suffix}"
    service_worker_path.write_text(
        strip_trailing_whitespace(render_service_worker(precache_urls, cache_name)),
        encoding="utf-8",
    )
    return [icon_192, icon_512, web_manifest_path, service_worker_path]


def write_gpx_zip(gpx_dir: Path, routes: list[dict[str, Any]]) -> Path:
    zip_path = gpx_dir / GPX_ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for route in routes:
            for key in GPX_PATH_KEYS:
                gpx_path = Path(route[key])
                archive.write(gpx_path, arcname=str(gpx_path.relative_to(gpx_dir)))
    return zip_path


def public_safety_check(output_dir: Path, extra_forbidden: list[str] | None = None) -> list[str]:
    forbidden = list(PRIVATE_LITERAL_PATTERNS)
    forbidden.extend(extra_forbidden or [])
    failures = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token and token in text:
                failures.append(f"{path}: contains {token}")
        for pattern in PRIVATE_REGEX_PATTERNS:
            if pattern.search(text):
                failures.append(f"{path}: contains private address pattern")
    return failures


def load_default_certificate_data(path: Path = DEFAULT_CERTIFICATE_JSON) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def load_default_field_day_layer(path: Path = DEFAULT_FIELD_DAY_LAYER_JSON) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def apply_progress_to_map_data(map_data: dict[str, Any], progress_data: dict[str, Any] | None) -> dict[str, Any]:
    if not progress_data:
        return map_data
    updated = json.loads(json.dumps(map_data))
    completed_ids = {
        int(seg_id)
        for seg_id in (updated.get("progress") or {}).get("completed_segment_ids") or []
        if seg_id is not None
    }
    completed_ids.update(
        int(seg_id)
        for seg_id in progress_data.get("completed_segment_ids") or []
        if seg_id is not None
    )
    completed_ids.update(
        int(seg_id)
        for seg_id in progress_data.get("extra_completed_segment_ids") or []
        if seg_id is not None
    )
    missed_ids = {int(seg_id) for seg_id in progress_data.get("missed_segment_ids") or [] if seg_id is not None}
    blocked_ids = {int(seg_id) for seg_id in progress_data.get("blocked_segment_ids") or [] if seg_id is not None}
    completed_ids.difference_update(missed_ids)
    completed_ids.difference_update(blocked_ids)
    progress = dict(updated.get("progress") or {})
    progress["completed_segment_ids"] = sorted(completed_ids)
    progress["blocked_segment_ids"] = sorted(set(progress.get("blocked_segment_ids") or []) | blocked_ids)
    if missed_ids:
        progress["missed_segment_ids"] = sorted(missed_ids)
    provisional_outing_ids = normalized_segment_ids(progress_data.get("completed_outing_ids"))
    if provisional_outing_ids:
        progress["provisional_completed_outing_ids"] = provisional_outing_ids
    if progress_data.get("provisional_completed_segment_ids"):
        progress["provisional_completed_segment_ids"] = normalized_segment_ids(
            progress_data.get("provisional_completed_segment_ids")
        )
    updated["progress"] = progress
    return updated


def certified_baseline_from_certificate(
    certificate_data: dict[str, Any] | None,
    fallback_segment_count: int,
) -> dict[str, Any]:
    if not certificate_data:
        return {
            "status": "not_loaded",
            "profile_id": None,
            "official_segment_count": fallback_segment_count,
            "covered_segment_count": fallback_segment_count,
            "missing_segment_count": None,
            "field_day_count": None,
            "total_p75_minutes": None,
            "total_on_foot_miles": None,
            "max_on_foot_miles": None,
            "max_p90_minutes": None,
            "day_gpx_validation_passed": None,
            "actual_max_day_trackpoint_gap_miles": None,
        }
    segment_set = certificate_data.get("segment_set") or {}
    field_days = certificate_data.get("field_days") or {}
    gpx_validation = certificate_data.get("gpx_validation") or {}
    profile = certificate_data.get("profile") or {}
    bounds = profile.get("bounds") or {}
    return {
        "status": certificate_data.get("certificate_status"),
        "profile_id": profile.get("profile_id"),
        "official_segment_count": segment_set.get("official_segment_count", fallback_segment_count),
        "covered_segment_count": segment_set.get("selected_calendar_segment_count")
        or segment_set.get("selected_plan_segment_count"),
        "missing_segment_count": segment_set.get("missing_segment_count"),
        "field_day_count": field_days.get("field_day_count"),
        "total_p75_minutes": field_days.get("total_p75_minutes"),
        "total_on_foot_miles": field_days.get("total_on_foot_miles"),
        "max_on_foot_miles": field_days.get("max_on_foot_miles"),
        "max_p90_minutes": field_days.get("max_p90_minutes"),
        "weekday_p90_minutes": bounds.get("weekday_p90_minutes"),
        "weekend_p90_minutes": bounds.get("weekend_p90_minutes"),
        "max_on_foot_miles_per_field_day": bounds.get("max_on_foot_miles_per_field_day"),
        "day_gpx_validation_passed": gpx_validation.get("day_track_validation_passed"),
        "actual_max_day_trackpoint_gap_miles": gpx_validation.get("actual_max_day_trackpoint_gap_miles"),
    }


def route_effort_summary(route: dict[str, Any]) -> dict[str, Any]:
    segment_ascent_ft = 0.0
    segment_descent_ft = 0.0
    segment_grade_adjusted_miles = 0.0
    segment_moving_p50 = 0.0
    segment_moving_p75 = 0.0
    cue_ascent_ft = 0.0
    cue_descent_ft = 0.0
    cue_grade_adjusted_miles = 0.0
    cue_moving_p50 = 0.0
    cue_moving_p75 = 0.0
    seen_segments: set[str] = set()
    used_segment_effort = False
    for cue in route.get("route_cues") or []:
        effort = cue.get("effort") or {}
        estimates = cue.get("time_estimates_minutes") or {}
        cue_ascent_ft += float(effort.get("ascent_ft") or 0)
        cue_descent_ft += float(effort.get("descent_ft") or 0)
        cue_grade_adjusted_miles += float(effort.get("grade_adjusted_miles") or 0)
        cue_moving_p50 += float(estimates.get("moving_effort_p50") or estimates.get("moving_p50") or 0)
        cue_moving_p75 += float(estimates.get("moving_effort_p75") or estimates.get("moving_p75") or 0)
        for segment in cue.get("segments") or []:
            segment_id = str(segment.get("seg_id") or segment.get("segment_id") or "")
            if segment_id and segment_id in seen_segments:
                continue
            if segment_id:
                seen_segments.add(segment_id)
            if segment.get("ascent_ft") is not None or segment.get("descent_ft") is not None:
                used_segment_effort = True
            segment_ascent_ft += float(segment.get("ascent_ft") or 0)
            segment_descent_ft += float(segment.get("descent_ft") or 0)
            segment_grade_adjusted_miles += float(segment.get("grade_adjusted_miles") or 0)
            segment_moving_p50 += float(segment.get("estimated_moving_minutes") or 0)
            segment_moving_p75 += float(segment.get("estimated_moving_minutes_p75") or 0)
    ascent_ft = segment_ascent_ft if used_segment_effort else cue_ascent_ft
    descent_ft = segment_descent_ft if used_segment_effort else cue_descent_ft
    grade_adjusted_miles = segment_grade_adjusted_miles if used_segment_effort else cue_grade_adjusted_miles
    moving_p50 = cue_moving_p50 or segment_moving_p50
    moving_p75 = cue_moving_p75 or segment_moving_p75
    elevation_source = "dem" if ascent_ft or descent_ft or grade_adjusted_miles else "unavailable"
    return {
        "ascent_ft": int(round(ascent_ft)),
        "descent_ft": int(round(descent_ft)),
        "grade_adjusted_miles": round(grade_adjusted_miles, 2) if grade_adjusted_miles else None,
        "estimated_moving_minutes_p50": int(round(moving_p50)) if moving_p50 else None,
        "estimated_moving_minutes_p75": int(round(moving_p75)) if moving_p75 else None,
        "elevation_source": elevation_source,
    }


def route_time_estimate_summary(route: dict[str, Any]) -> dict[str, Any]:
    outing = route.get("outing") or {}
    p75 = int(round(float(outing.get("total_minutes") or 0)))
    p90_values = []
    for cue in route.get("route_cues") or []:
        estimates = cue.get("time_estimates_minutes") or {}
        if estimates.get("door_to_door_p90") is not None:
            p90_values.append(float(estimates["door_to_door_p90"]))
    if len(p90_values) == 1:
        p90 = int(round(p90_values[0]))
    elif p90_values:
        p90 = max(p75, int(round(sum(p90_values))))
    else:
        p90 = int(math.ceil(p75 * 1.12)) if p75 else None
    return {
        "door_to_door_minutes_p75": p75 or None,
        "door_to_door_minutes_p90": p90,
    }


def route_start_justification(outing: dict[str, Any], parking: dict[str, Any]) -> str | None:
    if outing.get("start_justification"):
        return public_display_text(outing["start_justification"])
    label = public_display_text(parking.get("name") or outing.get("trailhead"))
    if not label:
        return None
    confidence = parking.get("parking_confidence") or "field-packet parking evidence"
    source = parking.get("source") or "the canonical field-packet source"
    return (
        f"Chosen because {label} is the current field-packet parking/start anchor for this exact "
        f"official segment set, with {confidence} parking evidence from {source}; no active "
        "accepted same-credit replacement is recorded for this route at export time."
    )


def route_field_tool_record(route: dict[str, Any], completion_safety: dict[str, Any] | None = None) -> dict[str, Any]:
    outing = route["outing"]
    parking = route.get("parking") or {}
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    validation = route.get("validation") or {}
    segment_ids = normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids"))
    time_estimates = route_time_estimate_summary(route)
    cue_generation_mode = outing.get("cue_generation_mode") or next(
        (
            cue.get("cue_generation_mode")
            for cue in route.get("route_cues") or []
            if cue.get("cue_generation_mode")
        ),
        None,
    )
    return {
        "outing_id": outing.get("outing_id"),
        "label": outing.get("label"),
        "route_code": outing.get("route_code") or outing.get("label"),
        "route_name": outing_route_name(outing),
        "route_name_source": outing.get("route_name_source"),
        "route_name_area": outing.get("route_name_area"),
        "route_name_primary_trail": outing.get("route_name_primary_trail"),
        "block_name": outing.get("block_name"),
        "trailhead": public_display_text(outing.get("trailhead")),
        "trailhead_display": public_display_text(outing.get("trailhead")),
        "candidate_ids": [str(candidate_id) for candidate_id in outing.get("candidate_ids") or []],
        "trails": outing.get("trails") or [],
        "segment_ids": segment_ids,
        "remaining_segment_count": len(segment_ids),
        "official_miles": outing.get("official_miles"),
        "on_foot_miles": outing.get("on_foot_miles"),
        "door_to_door_minutes_p75": time_estimates.get("door_to_door_minutes_p75"),
        "door_to_door_minutes_p90": time_estimates.get("door_to_door_minutes_p90"),
        "effort": route_effort_summary(route),
        "time_bucket": outing.get("time_bucket"),
        "accepted_replacement_id": outing.get("accepted_replacement_id"),
        "start_justification": route_start_justification(outing, parking),
        "route_card_status": outing.get("route_card_status"),
        "packet_visibility": outing.get("packet_visibility"),
        "certified_route_card": outing.get("certified_route_card"),
        "requires_field_walkthrough": outing.get("requires_field_walkthrough"),
        "field_readiness_status": route.get("field_readiness_status"),
        "field_ready": route_is_field_ready(route),
        "special_management": route.get("special_management") or outing.get("special_management") or {},
        "route_card_audit_blockers": append_unique_text(
            route.get("route_card_audit_blockers") or [],
            outing.get("route_card_audit_blockers") or [],
        ),
        "cue_generation_mode": cue_generation_mode,
        "anchor_to_credit_endpoint_distance_miles": outing.get("anchor_to_credit_endpoint_distance_miles"),
        "credit_endpoint_used": outing.get("credit_endpoint_used"),
        "gpx_href": route.get("gpx_href"),
        "parking_navigation_url": route.get("parking_navigation_url"),
        "parking": {
            "name": public_display_text(parking.get("name")),
            "display_name": public_display_text(parking.get("name")),
            "lat": parking.get("lat"),
            "lon": parking.get("lon"),
            "has_parking": parking.get("has_parking"),
            "has_restroom": parking.get("has_restroom"),
            "has_water": parking.get("has_water"),
            "water_confidence": parking.get("water_confidence"),
            "parking_confidence": parking.get("parking_confidence"),
            "source": public_display_text(parking.get("source")),
            "field_ready": parking.get("field_ready"),
            "nearest_open_trail_name": public_display_text(parking.get("nearest_open_trail_name")),
            "nearest_open_trail_label": public_display_text(parking.get("nearest_open_trail_label")),
        },
        "logistics": {
            "car_pass_count": len(logistics.get("car_passes") or []),
            "known_water_count": len(logistics.get("known_water") or []),
            "has_car_pass": bool(logistics.get("car_passes")),
            "has_known_water": bool(logistics.get("known_water")),
        },
        "validation": {
            "passed": validation.get("passed"),
            "max_trackpoint_gap_miles": validation.get("max_trackpoint_gap_miles"),
            "max_allowed_gap_miles": validation.get("max_allowed_gap_miles"),
            "max_allowed_parking_gap_miles": validation.get("max_allowed_parking_gap_miles"),
            "failures": validation.get("failures") or [],
        },
        "navigation_quality": route.get("navigation_quality") or {},
        "navigation_source_audit": route.get("navigation_source_audit") or {},
        "source_gap_repair": route.get("source_gap_repair") or {},
        "wayfinding_mileage_reconciliation": route.get("wayfinding_mileage_reconciliation") or {},
        "segment_ownership_reconciliation": route.get("segment_ownership_reconciliation") or {},
        "completion_safety": completion_safety or route.get("completion_safety") or {},
        "segment_direction_evidence": route.get("segment_direction_evidence") or {},
        "turn_by_turn_steps": public_display_value(route.get("turn_by_turn_steps") or []),
        "wayfinding_cues": public_display_value(route.get("wayfinding_cues") or []),
    }


def completion_safety_by_outing(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_segment_ids = {
        segment_id
        for route in routes
        for segment_id in normalized_segment_ids(
            route.get("outing", {}).get("remaining_segment_ids") or route.get("outing", {}).get("segment_ids")
        )
    }
    by_outing: dict[str, dict[str, Any]] = {}
    for route in routes:
        outing = route.get("outing") or {}
        outing_id = str(outing.get("outing_id") or "")
        completed_by_this_route = set(
            normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids"))
        )
        remaining_after = all_segment_ids - completed_by_this_route
        available_after = set()
        for other in routes:
            other_outing = other.get("outing") or {}
            if str(other_outing.get("outing_id") or "") == outing_id:
                continue
            available_after.update(
                normalized_segment_ids(other_outing.get("remaining_segment_ids") or other_outing.get("segment_ids"))
            )
        missing_after = sorted(remaining_after - available_after, key=lambda value: (len(value), value))
        by_outing[outing_id] = {
            "normal_completion_preserves_remaining_menu_coverage": not missing_after,
            "missing_remaining_segment_ids_after_completion": missing_after,
        }
    return by_outing


def public_field_day_route_ref(route_ref: dict[str, Any] | None) -> dict[str, Any] | None:
    if not route_ref:
        return None
    return {
        "outing_id": route_ref.get("outing_id"),
        "label": route_ref.get("label"),
        "route_code": route_ref.get("route_code") or route_ref.get("label"),
        "route_name": route_ref.get("route_name"),
        "candidate_ids": [str(value) for value in route_ref.get("candidate_ids") or []],
        "gpx_href": route_ref.get("gpx_href"),
        "validation_passed": route_ref.get("validation_passed"),
        "route_card_quality_passed": route_ref.get("route_card_quality_passed"),
        "field_readiness_status": route_ref.get("field_readiness_status"),
        "route_card_status": route_ref.get("route_card_status"),
        "packet_visibility": route_ref.get("packet_visibility"),
        "certified_route_card": route_ref.get("certified_route_card"),
        "special_management": route_ref.get("special_management") or {},
        "certification_blockers": [str(value) for value in route_ref.get("certification_blockers") or []],
    }


def public_field_day_loop(loop: dict[str, Any]) -> dict[str, Any]:
    route_ref = loop.get("route_card_ref") or {}
    route_name = (
        loop.get("route_name")
        or route_ref.get("route_name")
        or human_route_name(loop.get("trail_names") or [], loop.get("trailhead") or "").get("route_name")
    )
    return {
        "loop_id": loop.get("loop_id"),
        "source": loop.get("source"),
        "candidate_id": loop.get("candidate_id"),
        "label": loop.get("label"),
        "route_code": loop.get("route_code") or loop.get("label"),
        "route_name": route_name,
        "trailhead": public_display_text(loop.get("trailhead")),
        "trailhead_display": public_display_text(loop.get("trailhead")),
        "trail_names": [str(value) for value in loop.get("trail_names") or []],
        "segment_ids": normalized_segment_ids(loop.get("segment_ids")),
        "segment_count": loop.get("segment_count"),
        "official_miles": loop.get("official_miles"),
        "on_foot_miles": loop.get("on_foot_miles"),
        "p75_minutes": loop.get("p75_minutes"),
        "p90_minutes": loop.get("p90_minutes"),
        "field_day_schedule_p75_minutes": loop.get("field_day_schedule_p75_minutes"),
        "field_day_schedule_p90_minutes": loop.get("field_day_schedule_p90_minutes"),
        "route_card_door_to_door_p75_minutes": loop.get("route_card_door_to_door_p75_minutes"),
        "route_card_door_to_door_p90_minutes": loop.get("route_card_door_to_door_p90_minutes"),
        "timing_source": loop.get("timing_source"),
        "validation_passed": loop.get("validation_passed"),
        "manual_design_hold": loop.get("manual_design_hold"),
        "certification_status": loop.get("certification_status"),
        "field_readiness_status": loop.get("field_readiness_status"),
        "accepted_replacement_id": loop.get("accepted_replacement_id"),
        "route_card_status": loop.get("route_card_status"),
        "packet_visibility": loop.get("packet_visibility"),
        "certified_route_card": loop.get("certified_route_card"),
        "requires_field_walkthrough": loop.get("requires_field_walkthrough"),
        "special_management": loop.get("special_management") or {},
        "anchor_to_credit_endpoint_distance_miles": loop.get("anchor_to_credit_endpoint_distance_miles"),
        "credit_endpoint_used": loop.get("credit_endpoint_used"),
        "route_card_audit_blockers": [str(value) for value in loop.get("route_card_audit_blockers") or []],
        "route_card_ref": public_field_day_route_ref(loop.get("route_card_ref")),
    }


def public_field_day_record(day: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": day.get("date"),
        "weekday_name": day.get("weekday_name"),
        "day_type": day.get("day_type"),
        "constraints": [str(value) for value in day.get("constraints") or []],
        "draft_day_number": day.get("draft_day_number"),
        "field_day_id": day.get("field_day_id"),
        "p75_minutes": day.get("p75_minutes"),
        "p90_minutes": day.get("p90_minutes"),
        "field_day_schedule_p75_minutes": day.get("field_day_schedule_p75_minutes"),
        "field_day_schedule_p90_minutes": day.get("field_day_schedule_p90_minutes"),
        "route_card_door_to_door_p75_sum": day.get("route_card_door_to_door_p75_sum"),
        "route_card_door_to_door_p90_sum": day.get("route_card_door_to_door_p90_sum"),
        "legacy_recomputed_p75_minutes": day.get("legacy_recomputed_p75_minutes"),
        "legacy_recomputed_p90_minutes": day.get("legacy_recomputed_p90_minutes"),
        "timing_authority": day.get("timing_authority"),
        "timing_repair": day.get("timing_repair"),
        "single_loop_timing_mismatch": day.get("single_loop_timing_mismatch"),
        "route_card_timing_double_count_risk": day.get("route_card_timing_double_count_risk"),
        "p90_bound_minutes": day.get("p90_bound_minutes"),
        "available_minutes_p90": day.get("available_minutes_p90"),
        "availability_source": day.get("availability_source"),
        "day_type_capacity_proxy_used": day.get("day_type_capacity_proxy_used"),
        "stress": day.get("stress"),
        "drive_minutes": day.get("drive_minutes"),
        "between_drive_minutes": day.get("between_drive_minutes"),
        "loop_count": day.get("loop_count"),
        "transfer_count": day.get("transfer_count"),
        "official_miles": day.get("official_miles"),
        "on_foot_miles": day.get("on_foot_miles"),
        "segment_count": day.get("segment_count"),
        "segment_ids": normalized_segment_ids(day.get("segment_ids")),
        "schedule_integrity": day.get("schedule_integrity"),
        "execution_status": day.get("execution_status"),
        "loops": [public_field_day_loop(loop) for loop in day.get("loops") or []],
    }


def public_field_day_layer_record(field_day_layer_data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not field_day_layer_data:
        return None
    summary_keys = {
        "field_day_count",
        "calendar_day_count",
        "active_execution_day_count",
        "reserve_day_count",
        "reserve_dates",
        "loop_count",
        "multi_start_day_count",
        "total_p75_minutes",
        "max_p90_minutes",
        "total_between_drive_minutes",
        "schedule_authority",
        "route_card_door_to_door_timing_policy",
        "single_loop_timing_repair_count",
        "single_loop_timing_repairs",
        "single_loop_timing_mismatch_unrepaired_count",
        "single_loop_timing_mismatches_unrepaired",
        "availability_bound_authority",
        "day_type_capacity_proxy_used",
        "weekday_weekend_labels_role",
        "day_gpx_validation_passed",
        "schedule_p90_violation_day_count",
        "schedule_p90_violation_days",
        "certified_route_card_loop_count",
        "needs_route_card_audit_fix_loop_count",
        "needs_route_card_promotion_loop_count",
        "accepted_replacement_blocker_count",
        "skipped_source_loop_count",
        "official_segment_count",
        "covered_segment_count",
        "missing_segment_count",
        "assignment_audit_passed",
        "field_tool_baseline_status",
    }
    source_files = field_day_layer_data.get("source_files") or {}
    execution_model = field_day_layer_data.get("execution_model") or {}
    return {
        "schema": field_day_layer_data.get("schema"),
        "generated_at": field_day_layer_data.get("generated_at"),
        "publication_status": field_day_layer_data.get("publication_status"),
        "execution_model": {
            "primary_execution_artifact": execution_model.get("primary_execution_artifact") or "field_day_layer",
            "proof_unit": execution_model.get("proof_unit") or "certified_route_card",
            "default_phone_view": execution_model.get("default_phone_view") or "field-days",
            "route_card_role": execution_model.get("route_card_role") or "certification_and_navigation_unit",
            "promotion_gap_policy": execution_model.get("promotion_gap_policy")
            or "Loops without an audit-clean route_card_ref, or with route-card audit blockers, stay visible as promotion/audit gaps.",
            "timing_authority": execution_model.get("timing_authority"),
        },
        "source_files": {
            key: source_files.get(key)
            for key in ("calendar_assignment", "field_tool_data", "route_card_promotion")
            if source_files.get(key)
        },
        "summary": {
            key: value
            for key, value in (field_day_layer_data.get("summary") or {}).items()
            if key in summary_keys
        },
        "field_days": [public_field_day_record(day) for day in field_day_layer_data.get("field_days") or []],
    }


def apply_route_names_to_field_day_layer(field_day_layer: dict[str, Any] | None, routes: list[dict[str, Any]]) -> None:
    if not field_day_layer:
        return
    by_outing_id = {
        str(route.get("outing_id") or ""): route
        for route in routes
        if route.get("outing_id")
    }
    by_candidate_id: dict[str, dict[str, Any]] = {}
    for route in routes:
        outing = route.get("outing") or {}
        for candidate_id in outing.get("candidate_ids") or []:
            by_candidate_id[str(candidate_id)] = route
    for day in field_day_layer.get("field_days") or []:
        for loop in day.get("loops") or []:
            route_ref = loop.get("route_card_ref") or {}
            source = by_outing_id.get(str(route_ref.get("outing_id") or ""))
            if not source:
                source = by_candidate_id.get(str(loop.get("candidate_id") or ""))
            if not source:
                continue
            outing = source.get("outing") or {}
            route_name = outing_route_name(outing)
            route_code = outing.get("route_code") or outing.get("label")
            field_readiness_status = source.get("field_readiness_status")
            blockers = append_unique_text(
                loop.get("route_card_audit_blockers") or [],
                source.get("route_card_audit_blockers") or outing.get("route_card_audit_blockers") or [],
            )
            loop["route_name"] = route_name
            loop["route_code"] = route_code
            loop["field_readiness_status"] = field_readiness_status
            loop["route_card_status"] = outing.get("route_card_status")
            loop["packet_visibility"] = outing.get("packet_visibility")
            loop["certified_route_card"] = outing.get("certified_route_card")
            loop["requires_field_walkthrough"] = outing.get("requires_field_walkthrough")
            loop["special_management"] = source.get("special_management") or outing.get("special_management") or {}
            loop["route_card_audit_blockers"] = blockers
            if field_readiness_status == "blocked_special_management":
                loop["certification_status"] = "blocked_special_management"
            if route_ref:
                route_ref["route_name"] = route_name
                route_ref["route_code"] = route_code
                route_ref["field_readiness_status"] = field_readiness_status
                route_ref["route_card_status"] = outing.get("route_card_status")
                route_ref["packet_visibility"] = outing.get("packet_visibility")
                route_ref["certified_route_card"] = outing.get("certified_route_card")
                route_ref["special_management"] = source.get("special_management") or outing.get("special_management") or {}
                route_ref["certification_blockers"] = append_unique_text(
                    route_ref.get("certification_blockers") or [],
                    blockers,
                )
                if source.get("gpx_href"):
                    route_ref["gpx_href"] = source.get("gpx_href")
    recalculate_field_day_layer_route_card_summary(field_day_layer)


def recalculate_field_day_layer_route_card_summary(field_day_layer: dict[str, Any]) -> None:
    summary = field_day_layer.setdefault("summary", {})
    certified = 0
    needs_audit_fix = 0
    needs_promotion = 0
    blocked_special_management = 0
    for day in field_day_layer.get("field_days") or []:
        day_blocked = False
        for loop in day.get("loops") or []:
            status = str(loop.get("certification_status") or "")
            blockers = loop.get("route_card_audit_blockers") or []
            if status == "blocked_special_management":
                blocked_special_management += 1
                needs_audit_fix += 1
                day_blocked = True
            elif status == "certified_route_card" and not blockers:
                certified += 1
            elif status == "needs_route_card_promotion":
                needs_promotion += 1
            elif status == "needs_route_card_audit_fix" or blockers:
                needs_audit_fix += 1
        if day_blocked:
            day["execution_status"] = "blocked_special_management"
    summary["certified_route_card_loop_count"] = certified
    summary["needs_route_card_audit_fix_loop_count"] = needs_audit_fix
    summary["needs_route_card_promotion_loop_count"] = needs_promotion
    if blocked_special_management:
        summary["blocked_special_management_loop_count"] = blocked_special_management
        field_day_layer["publication_status"] = "blocked_by_special_management"


def build_field_tool_data(
    manifest: dict[str, Any],
    certificate_data: dict[str, Any] | None = None,
    map_data: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
    field_day_layer_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_segment_ids = sorted(
        {
            segment_id
            for route in manifest["routes"]
            for segment_id in normalized_segment_ids(
                route.get("outing", {}).get("remaining_segment_ids") or route.get("outing", {}).get("segment_ids")
            )
        },
        key=lambda value: (len(value), value),
    )
    progress = (map_data or {}).get("progress") or {}
    completed_segment_ids = normalized_segment_ids(progress.get("completed_segment_ids"))
    blocked_segment_ids = normalized_segment_ids(progress.get("blocked_segment_ids"))
    safety_by_outing = completion_safety_by_outing(manifest["routes"])
    field_ready_routes = [route for route in manifest["routes"] if route_is_field_ready(route)]
    held_routes = [route for route in manifest["routes"] if not route_is_field_ready(route)]
    special_management_failed_routes = [
        route for route in manifest["routes"] if route_is_special_management_blocked(route)
    ]
    field_day_layer = public_field_day_layer_record(field_day_layer_data)
    apply_route_names_to_field_day_layer(field_day_layer, manifest["routes"])
    default_view = "field-days" if field_day_layer else "routes"
    payload = {
        "schema": "boise_trails_field_tool_data_v1",
        "source": source_metadata or source_metadata_for_map_data(map_data or {}),
        "execution_model": {
            "primary_execution_artifact": "field_day_layer" if field_day_layer else "route_cards",
            "default_phone_view": default_view,
            "route_card_role": "certification_and_navigation_unit",
            "route_cards_are_proof_units": True,
            "field_days_publication_status": field_day_layer.get("publication_status") if field_day_layer else None,
        },
        "time_filters_minutes": TIME_FILTER_MINUTES,
        "certified_baseline": certified_baseline_from_certificate(certificate_data, len(all_segment_ids)),
        "progress": {
            "completed_segment_ids_at_export": completed_segment_ids,
            "blocked_segment_ids_at_export": blocked_segment_ids,
            "remaining_segment_count_at_start": len(set(all_segment_ids) - set(blocked_segment_ids)),
        },
        "summary": {
            "runnable_outing_count": manifest["summary"]["runnable_outing_count"],
            "manual_hold_count": manifest["summary"]["manual_hold_count"],
            "gpx_validation_passed": manifest["summary"]["gpx_validation_passed"],
            "field_ready_route_count": len(field_ready_routes),
            "held_route_count": len(held_routes),
            "special_management_failed_route_count": len(special_management_failed_routes),
            "segment_count_in_field_menu": len(all_segment_ids),
            "gpx_zip_href": manifest["summary"].get("gpx_zip_href"),
        },
        "routes": [
            route_field_tool_record(route, safety_by_outing.get(str(route.get("outing_id"))))
            for route in manifest["routes"]
        ],
        "manual_holds": manifest.get("manual_holds") or [],
    }
    if field_day_layer:
        payload["field_day_layer"] = field_day_layer
    return payload


def load_default_walkthrough_graph_edges() -> list[Any]:
    if not DEFAULT_WALKTHROUGH_OFFICIAL_GEOJSON.exists():
        return []
    official_geojson = read_json(DEFAULT_WALKTHROUGH_OFFICIAL_GEOJSON)
    connector_geojson = (
        read_json(DEFAULT_WALKTHROUGH_CONNECTOR_GEOJSON)
        if DEFAULT_WALKTHROUGH_CONNECTOR_GEOJSON.exists()
        else {"features": []}
    )
    graph_edges, _official_index = load_graph_edges(official_geojson, connector_geojson)
    return graph_edges


def segment_direction_evidence_for_route(route: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    outing = route.get("outing") or {}
    claimed_segment_ids = set(
        normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids"))
    )
    for cue in route.get("route_cues") or []:
        for segment in cue.get("segments") or []:
            segment_id = str(segment.get("seg_id") or "")
            if not segment_id:
                continue
            if claimed_segment_ids and segment_id not in claimed_segment_ids:
                continue
            direction_rule = str(segment.get("direction_rule") or "")
            direction_cue = str(segment.get("direction_cue") or "")
            allowed_geometry_direction = None
            if "opposite official geometry" in direction_cue.lower():
                allowed_geometry_direction = "reverse"
            elif direction_rule.lower() in {"ascent", "oneway", "forward"}:
                allowed_geometry_direction = "forward"
            if direction_rule or direction_cue or allowed_geometry_direction:
                evidence[segment_id] = {
                    key: value
                    for key, value in {
                        "direction_rule": direction_rule,
                        "direction_cue": direction_cue,
                        "allowed_geometry_direction": allowed_geometry_direction,
                    }.items()
                    if value
                }
    return evidence


def export_field_packet(
    map_data: dict[str, Any],
    output_dir: Path,
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
    max_parking_gap_miles: float = DEFAULT_MAX_PARKING_GAP_MILES,
    require_certifiable: bool = False,
    certificate_data: dict[str, Any] | None = None,
    progress_data: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
    field_day_layer_data: dict[str, Any] | None = None,
    trailhead_access_index: dict[str, dict[str, Any]] | None = None,
    walkthrough_graph_edges: list[Any] | None = None,
) -> dict[str, Any]:
    map_data = apply_progress_to_map_data(map_data, progress_data)
    computed_source_metadata = source_metadata_for_map_data(map_data)
    effective_source_metadata = {**computed_source_metadata, **(source_metadata or {})}
    effective_source_metadata["map_data_sha256"] = computed_source_metadata["map_data_sha256"]
    effective_source_metadata["completed_segment_count_at_export"] = computed_source_metadata[
        "completed_segment_count_at_export"
    ]
    effective_source_metadata["blocked_segment_count_at_export"] = computed_source_metadata[
        "blocked_segment_count_at_export"
    ]
    validate_field_menu_source(map_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_manifest in output_dir.glob("*-artifact-manifest.json"):
        stale_manifest.unlink()
    gpx_dir = output_dir / "gpx"
    gpx_dir.mkdir(parents=True, exist_ok=True)
    for stale_gpx in gpx_dir.rglob("*.gpx"):
        stale_gpx.unlink()
    for stale_zip in gpx_dir.glob("*.zip"):
        stale_zip.unlink()
    official_gpx_dir = gpx_dir / OFFICIAL_GPX_DIR_NAME
    cue_gpx_dir = gpx_dir / CUE_GPX_DIR_NAME
    audit_gpx_dir = gpx_dir / AUDIT_GPX_DIR_NAME
    for directory in (official_gpx_dir, cue_gpx_dir, audit_gpx_dir):
        directory.mkdir(parents=True, exist_ok=True)
    routes_by_candidate = indexed_features(map_data, "routes", "candidate_id")
    parking_by_candidate = indexed_features(map_data, "parking", "candidate_id")
    if trailhead_access_index is None:
        trailhead_access_index = {}
    segments_by_id = official_segment_index(map_data)
    route_cues = map_data.get("route_cues") or {}
    if walkthrough_graph_edges is None:
        walkthrough_graph_edges = load_default_walkthrough_graph_edges()
    segment_elevation_index = load_segment_elevation_index()
    dem_context = load_dem_context(DEFAULT_DEM_TIF, DEFAULT_DEM_SUMMARY_JSON)
    connector_graph = load_default_connector_graph()
    outings = build_outing_menu(map_data)
    runnable = [outing for outing in outings if not outing.get("manual_design_hold") and outing.get("remaining_segment_ids")]
    manual_holds = [outing for outing in outings if outing.get("manual_design_hold") and outing.get("remaining_segment_ids")]
    routes = []
    for outing in sorted(
        runnable,
        key=lambda item: (
            outing_time_bucket_sort(item["time_bucket"]),
            int(item.get("total_minutes") or 0),
            str(item.get("label") or ""),
        ),
    ):
        raw_track_segments = track_segments_for_outing(outing, routes_by_candidate)
        stitched_track_segments = stitch_inter_segment_track_gaps(
            raw_track_segments,
            connector_graph,
            max_gap_miles=max_gap_miles,
        )
        source_gap_repair = source_gap_repair_summary(
            raw_track_segments,
            stitched_track_segments,
            max_gap_miles=max_gap_miles,
        )
        track_segments = densify_track_segments(
            stitched_track_segments,
            max_gap_miles=max_gap_miles,
        )
        parking = parking_for_outing(outing, route_cues, parking_by_candidate, trailhead_access_index)
        validation = validate_outing_export(
            outing,
            track_segments,
            parking,
            route_cues,
            max_gap_miles=max_gap_miles,
            max_parking_gap_miles=max_parking_gap_miles,
        )
        route_name = outing_route_name(outing)
        title = f"{route_name} ({outing['label']})" if route_code_text(outing) else route_name
        slug = slugify(f"{route_name}-{outing['label']}")
        description = gpx_description(outing)
        cue_list = [route_cues[str(candidate_id)] for candidate_id in outing.get("candidate_ids") or [] if str(candidate_id) in route_cues]
        enrich_route_cues_with_segment_elevation(cue_list, segment_elevation_index)
        logistics = aggregate_logistics(cue_list, parking)
        navigation_waypoints = build_navigation_waypoints(
            outing,
            route_cues,
            segments_by_id,
            parking,
            track_segments,
            logistics,
        )
        cue_only_waypoints = []
        if parking:
            cue_only_waypoints.append(
                {
                    "name": f"PARK/START {public_display_text(parking['name'])}",
                    "description": "Park here and start this outing.",
                    "lon": parking["lon"],
                    "lat": parking["lat"],
                }
            )
        cue_only_waypoints.extend(logistics_waypoints(logistics))
        cue_only_waypoints.extend(cue_waypoints(outing, route_cues, segments_by_id))
        final = last_point(track_segments)
        if final:
            cue_only_waypoints.append(
                {
                    "name": "RETURN TO CAR",
                    "description": "Route endpoint / return-to-car point.",
                    "lon": final[0],
                    "lat": final[1],
                }
            )
        audit_waypoints = build_audit_waypoints(
            outing,
            route_cues,
            segments_by_id,
            parking,
            track_segments,
            logistics,
        )
        gpx_path = official_gpx_dir / f"{slug}.gpx"
        cue_gpx_path = cue_gpx_dir / f"{slug}.gpx"
        audit_gpx_path = audit_gpx_dir / f"{slug}.gpx"
        gpx_path.write_text(
            strip_trailing_whitespace(render_gpx(title, description, track_segments, navigation_waypoints)),
            encoding="utf-8",
        )
        cue_gpx_path.write_text(
            strip_trailing_whitespace(render_gpx(f"{title} cues", description, [], cue_only_waypoints)),
            encoding="utf-8",
        )
        audit_gpx_path.write_text(
            strip_trailing_whitespace(render_gpx(f"{title} audit", description, track_segments, audit_waypoints)),
            encoding="utf-8",
        )
        route = {
            "outing_id": outing["outing_id"],
            "label": outing["label"],
            "route_code": outing.get("route_code") or outing["label"],
            "route_name": route_name,
            "outing": outing,
            "parking": parking,
            "logistics": logistics,
            "parking_navigation_url": parking_navigation_url(parking),
            "gpx_path": str(gpx_path),
            "gpx_href": f"gpx/{OFFICIAL_GPX_DIR_NAME}/{gpx_path.name}",
            "cue_gpx_path": str(cue_gpx_path),
            "cue_gpx_href": f"gpx/{CUE_GPX_DIR_NAME}/{cue_gpx_path.name}",
            "audit_gpx_path": str(audit_gpx_path),
            "audit_gpx_href": f"gpx/{AUDIT_GPX_DIR_NAME}/{audit_gpx_path.name}",
            "validation": validation,
            "route_cues": cue_list,
            "waypoint_count": len(navigation_waypoints),
            "navigation_waypoint_count": len(navigation_waypoints),
            "cue_waypoint_count": len(cue_only_waypoints),
            "audit_waypoint_count": len(audit_waypoints),
            "track_segment_count": len(track_segments),
            "source_gap_repair": source_gap_repair,
            "_track_segments": track_segments,
            "_official_segment_index": segments_by_id,
        }
        route["navigation_quality"] = navigation_quality_for_route(route)
        route["turn_by_turn_steps"] = build_turn_by_turn_steps(route)
        route["wayfinding_cues"] = build_wayfinding_cues(route)
        assign_wayfinding_route_miles(route)
        apply_non_credit_claimed_repeat_declarations(
            route,
            elevation_sampler=dem_context.get("sampler"),
        )
        reconcile_wayfinding_miles_to_route_card(route)
        apply_off_label_route_leg_warnings(route)
        route["segment_direction_evidence"] = segment_direction_evidence_for_route(route)
        enrich_route_with_walkthrough_edge_names(route, track_segments, walkthrough_graph_edges)
        apply_geometry_overlap_wayfinding_cautions(route)
        apply_overlap_exit_wayfinding_cautions(route)
        routes.append(route)
    apply_segment_ownership_reconciliation(
        routes,
        segments_by_id,
        elevation_sampler=dem_context.get("sampler"),
    )
    apply_navigation_source_audit_to_routes(routes)
    safety_by_outing = completion_safety_by_outing(routes)
    for route in routes:
        route["completion_safety"] = safety_by_outing.get(str(route.get("outing_id")), {})
    special_management_audit = build_special_management_audit_for_routes(routes, output_dir)
    apply_special_management_audit_to_routes(routes, special_management_audit)
    zip_path = write_gpx_zip(gpx_dir, routes)
    zip_href = f"gpx/{zip_path.name}"
    gpx_readme_path = gpx_dir / "README.md"
    gpx_readme_path.write_text(render_gpx_readme(routes, zip_href), encoding="utf-8")
    manifest = {
        "summary": {
            "runnable_outing_count": len(runnable),
            "manual_hold_count": len(manual_holds),
            "gpx_count": len(routes) * len(GPX_PATH_KEYS),
            "official_gpx_count": len(routes),
            "official_gpx_href": f"gpx/{OFFICIAL_GPX_DIR_NAME}/",
            "navigation_gpx_count": len(routes),
            "cue_gpx_count": len(routes),
            "audit_gpx_count": len(routes),
            "gpx_zip_href": zip_href,
            "gpx_validation_passed": all(route["validation"]["passed"] for route in routes),
            "failed_gpx_count": len([route for route in routes if not route["validation"]["passed"]]),
            "field_ready_route_count": len([route for route in routes if route_is_field_ready(route)]),
            "held_route_count": len([route for route in routes if not route_is_field_ready(route)]),
            "special_management_failed_route_count": len(
                [route for route in routes if route_is_special_management_blocked(route)]
            ),
            "max_gap_miles": max_gap_miles,
            "max_parking_gap_miles": max_parking_gap_miles,
        },
        "routes": routes,
        "manual_holds": manual_holds,
    }
    if special_management_audit:
        manifest["special_management_audit"] = special_management_audit
    if require_certifiable:
        assert_field_packet_certifiable(manifest)
    if certificate_data is None:
        certificate_data = load_default_certificate_data()
    field_tool_data = build_field_tool_data(
        manifest,
        certificate_data=certificate_data,
        map_data=map_data,
        source_metadata=effective_source_metadata,
        field_day_layer_data=field_day_layer_data,
    )
    manifest["certified_baseline"] = field_tool_data["certified_baseline"]
    manifest["field_day_layer"] = field_tool_data.get("field_day_layer")
    manifest["summary"]["field_tool_data_href"] = FIELD_TOOL_DATA_NAME
    manifest["summary"]["map_data_sha256"] = field_tool_data["source"]["map_data_sha256"]
    (output_dir / "index.html").write_text(strip_trailing_whitespace(render_index(manifest)), encoding="utf-8")
    field_tool_json = json.dumps(field_tool_data, indent=2) + "\n"
    (output_dir / FIELD_TOOL_DATA_NAME).write_text(field_tool_json, encoding="utf-8")
    asset_digest = hashlib.sha256()
    asset_digest.update(field_tool_json.encode("utf-8"))
    asset_digest.update(zip_path.read_bytes())
    asset_version = asset_digest.hexdigest()[:16]
    live_map_path = output_dir / "live-map.html"
    live_map_path.write_text(strip_trailing_whitespace(render_live_map_html(asset_version)), encoding="utf-8")
    pwa_paths = write_pwa_assets(
        output_dir,
        routes,
        zip_href,
        extra_precache_urls=[FIELD_TOOL_DATA_NAME, "live-map.html"],
    )
    public_manifest = {
        **manifest,
        "routes": [
            public_display_value({
                **{
                    key: value
                    for key, value in route.items()
                    if key not in {"route_cues", "_track_segments", "_official_segment_index", *GPX_PATH_KEYS}
                },
                "gpx_path": route["gpx_href"],
                "cue_gpx_path": route["cue_gpx_href"],
                "audit_gpx_path": route["audit_gpx_href"],
            })
            for route in routes
        ],
    }
    public_manifest["summary"]["pwa_artifacts"] = [
        "manifest.webmanifest",
        "service-worker.js",
        "icons/icon-192.png",
        "icons/icon-512.png",
        FIELD_TOOL_DATA_NAME,
        "live-map.html",
    ]
    (output_dir / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    safety_failures = public_safety_check(output_dir)
    if safety_failures:
        raise ValueError("Public safety check failed:\n" + "\n".join(safety_failures))
    manifest["pwa_paths"] = [str(live_map_path), *[str(path) for path in pwa_paths]]
    manifest["gpx_zip_path"] = str(zip_path)
    manifest["gpx_readme_path"] = str(gpx_readme_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-html", type=Path, default=DEFAULT_MAP_HTML)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--field-day-layer-json", type=Path, default=DEFAULT_FIELD_DAY_LAYER_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-gap-miles", type=float, default=DEFAULT_MAX_GAP_MILES)
    parser.add_argument("--max-parking-gap-miles", type=float, default=DEFAULT_MAX_PARKING_GAP_MILES)
    parser.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="Write diagnostic field artifacts even when route export validation fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_data, source_path = load_map_data(args.map_html, args.map_data_json)
    try:
        manifest = export_field_packet(
            map_data,
            args.output_dir,
            max_gap_miles=args.max_gap_miles,
            max_parking_gap_miles=args.max_parking_gap_miles,
            require_certifiable=not args.allow_uncertified,
            progress_data=read_json(args.progress_json) if args.progress_json else None,
            source_metadata=source_metadata_for_map_data(map_data, source_path),
            field_day_layer_data=load_default_field_day_layer(args.field_day_layer_json),
            trailhead_access_index=load_trailhead_access_index(),
        )
    except FieldPacketCertificationError as error:
        print(str(error), file=sys.stderr)
        print(
            "Certification stopped before writing a new field guide, live map, PWA manifest, or field-tool data.",
            file=sys.stderr,
        )
        print("Use --allow-uncertified only for local diagnostics, never for field publication.", file=sys.stderr)
        return 1
    artifact_manifest_dir = YEAR_DIR / "outputs" / "private" / "field-packet"
    artifact_manifest_dir.mkdir(parents=True, exist_ok=True)
    artifact_manifest_path = artifact_manifest_dir / f"{DEFAULT_BASENAME}-artifact-manifest.json"
    outputs = [
        args.output_dir / "index.html",
        args.output_dir / "manifest.json",
        args.output_dir / FIELD_TOOL_DATA_NAME,
        args.output_dir / "manifest.webmanifest",
        args.output_dir / "service-worker.js",
        args.output_dir / "icons" / "icon-192.png",
        args.output_dir / "icons" / "icon-512.png",
        args.output_dir / "gpx" / GPX_ZIP_NAME,
        Path(manifest["gpx_readme_path"]),
    ]
    outputs.extend(Path(route["gpx_path"]) for route in manifest["routes"])
    inputs = [source_path]
    if args.progress_json:
        inputs.append(args.progress_json)
    if args.field_day_layer_json and args.field_day_layer_json.exists():
        inputs.append(args.field_day_layer_json)
    write_manifest(
        artifact_manifest_path,
        build_artifact_manifest(
            run_id=str(map_data.get("run_id") or DEFAULT_BASENAME),
            inputs=inputs,
            outputs=outputs,
            command="export_mobile_field_packet.py",
            metadata={
                "runnable_outing_count": manifest["summary"]["runnable_outing_count"],
                "gpx_validation_passed": manifest["summary"]["gpx_validation_passed"],
            },
        ),
    )
    print(f"Wrote {manifest['summary']['gpx_count']} GPX files to {args.output_dir / 'gpx'}")
    print(f"Wrote official GPX files to {args.output_dir / 'gpx' / OFFICIAL_GPX_DIR_NAME}")
    print(f"Wrote {args.output_dir / 'index.html'}")
    print(f"Wrote {args.output_dir / 'manifest.json'}")
    print(f"Wrote {artifact_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
