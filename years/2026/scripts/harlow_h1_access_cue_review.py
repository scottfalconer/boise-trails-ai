#!/usr/bin/env python3
"""Review H1 Avimor-native Harlow/Spring connector access and cueability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from freestone_cluster_route_generation_experiment import display_path, float_value, write_json  # noqa: E402
from harlow_h1_gate_repair_audit import H1_REPLACE_ROUTE_LABELS, sort_id  # noqa: E402


DEFAULT_H1_AUDIT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-gate-repair-audit-2026-05-12.json"
DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_CONNECTOR_GEOJSON = (
    YEAR_DIR / "inputs" / "open-data" / "routing-connectors-2026-05-04" / "combined_r2r_osm_connectors.geojson"
)
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-access-cue-review-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "harlow-h1-access-cue-review-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-access-cue-review-2026-05-12-manifest.json"

AVIMOR_TRAILS_LAYER_URL = (
    "https://services7.arcgis.com/k4RHn3Cq6S9Ti9zx/arcgis/rest/services/"
    "Avimor_Trails_View/FeatureServer/1"
)

PUBLIC_SOURCE_REFERENCES = [
    {
        "source_id": "avimor_official_trails_page",
        "url": "https://www.avimor.com/trails-and-outdoors",
        "role": "Official Avimor trail names, seasonal-closure framing, trail rules, and public-facing map link.",
    },
    {
        "source_id": "avimor_arcgis_hike_bike_layer",
        "url": AVIMOR_TRAILS_LAYER_URL,
        "role": "Public ArcGIS Hike/Bike layer used for named-trail open/closed and hike-bike eligibility checks.",
    },
    {
        "source_id": "alltrails_spring_valley_creek",
        "url": "https://www.alltrails.com/trail/us/idaho/spring-valley-creek",
        "role": "Public route/parking corroboration for the Avimor Spring Valley Creek start.",
    },
    {
        "source_id": "mtbproject_twisted_spring_connector",
        "url": "https://www.mtbproject.com/trail/7001983/twisted-springs-connector",
        "role": "Public map/source corroboration that Twisted Spring connects the trailhead area to Spring Valley Creek.",
    },
    {
        "source_id": "trailforks_burnt_car_draw",
        "url": "https://www.trailforks.com/trails/burnt-car-draw/",
        "role": "Public map/source corroboration for Burnt Car Draw, Hike/Run use, direction, and nearby trailhead context.",
    },
]

LEG_DEFINITIONS = {
    "access_to_twisted_spring": {
        "title": "Access to Twisted Spring",
        "targets": {"1687"},
        "field_cue_name": "Avimor Spring Valley Creek parking -> McLeod Way Greenbelt/Twisted Spring Trail #8",
        "landmark": "Leave the lot on the paved greenbelt/park path, then pick up the signed Twisted Spring Trail #8 / Spring Valley Creek trailhead area.",
        "fallback": "If the Twisted Spring trailhead/bridge is not obvious from the lot, do not improvise through homes; return to the car and use H2 or the current microcards.",
    },
    "ricochet_shooting_whistling": {
        "title": "Ricochet / Shooting Range / Whistling Pig connector",
        "targets": {"1657"},
        "field_cue_name": "North Smokeys Draw Place -> Ricochet #2 -> Shooting Range #5 -> Whistling Pig #3",
        "landmark": "Use the signed Ricochet/Shooting/Whistling trail junctions near the neighborhood edge; North Smokeys Draw Place is a public-road connector cue, not challenge credit.",
        "fallback": "If the Ricochet/Shooting junction is unclear, stay on known signed Avimor trails and do not count the missing Shooting Range credit until field repair.",
    },
    "spring_creek_connector": {
        "title": "Spring Creek connector",
        "targets": {"1661"},
        "field_cue_name": "Whistling Pig #3 / Twisted Spring #8 -> Spring Creek #9",
        "landmark": "At the Whistling Pig / Spring Creek junction, follow Spring Creek #9 for the next official credit segment.",
        "fallback": "If the Spring Creek turn is missed, continue only until a signed junction; backtrack on signed trail rather than cutting across social paths.",
    },
    "harlows_connector": {
        "title": "Harlow's connector via Burnt Car / Cartwright / The Wall",
        "targets": {"1704"},
        "field_cue_name": "Burnt Car Draw #10 -> Cartwright Road #20 -> The Wall #29 -> Harlow's Hollows #16/#16a",
        "landmark": "Use the signed Burnt Car Draw / Cartwright Road / The Wall chain to reach Harlow's Hollows; keep the Harlow credit start distinct from the approach.",
        "fallback": "If The Wall or Harlow's entry is not signed/open, abort H1 promotion and use H2 or existing cards; do not substitute Broken Horn/Fisher access.",
    },
    "return_to_avimor": {
        "title": "Return to Avimor",
        "targets": {"return_to_car"},
        "field_cue_name": "Spring Creek #9 / Twisted Spring #8 / McLeod Way Greenbelt back to Avimor Spring Valley Creek parking",
        "landmark": "After the Harlow chain, return on signed Spring Creek/Twisted Spring corridors to the same paved path and parking lot.",
        "fallback": "If the return junction is confusing, follow signed Spring Creek/Twisted Spring back toward the Avimor neighborhood and avoid unmarked neighborhood cut-throughs.",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_arcgis_hike_bike_features(layer_url: str = AVIMOR_TRAILS_LAYER_URL) -> dict[str, Any]:
    query = urlencode(
        {
            "f": "geojson",
            "where": "1=1",
            "outFields": "FID,Type,Name,Hike_Bike,Equestrian,OHV,Bike_Dir,Leashing,Difficulty,Length,Seasonal,Visible,Closed,TrailNum,Owner",
            "returnGeometry": "true",
            "outSR": 4326,
        }
    )
    with urlopen(f"{layer_url}/query?{query}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_line_parts(geometry: dict[str, Any]) -> list[list[list[float]]]:
    if not geometry:
        return []
    if geometry.get("type") == "LineString":
        return [geometry.get("coordinates") or []]
    if geometry.get("type") == "MultiLineString":
        return geometry.get("coordinates") or []
    return []


def iter_points(geometry: dict[str, Any]) -> list[list[float]]:
    points: list[list[float]] = []
    for part in iter_line_parts(geometry):
        points.extend(part)
    return points


def haversine_miles(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    lon1, lat1 = math.radians(float(a[0])), math.radians(float(a[1]))
    lon2, lat2 = math.radians(float(b[0])), math.radians(float(b[1]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def line_length_miles(geometry: dict[str, Any]) -> float:
    total = 0.0
    for part in iter_line_parts(geometry):
        total += sum(haversine_miles(a, b) for a, b in zip(part, part[1:]))
    return round(total, 4)


def normalize_trail_name(name: str | None) -> str:
    if not name:
        return ""
    value = str(name).replace("’", "'").strip()
    value = value.replace(" - #", " #")
    if " #" in value:
        value = value.split(" #", 1)[0]
    if value == "Spring Creek":
        return "Spring Creek"
    if value == "Harlow Hollows":
        return "Harlow's Hollows"
    return value


def connector_feature_index(connector_geojson: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for feature in connector_geojson.get("features") or []:
        props = feature.get("properties") or {}
        name = props.get("TrailName") or props.get("Name") or props.get("SystemName") or f"connector_{props.get('OBJECTID')}"
        index[str(name)] = feature
    return index


def open_arcgis_features(arcgis_geojson: dict[str, Any]) -> list[dict[str, Any]]:
    open_features = []
    for feature in arcgis_geojson.get("features") or []:
        props = feature.get("properties") or {}
        if props.get("Hike_Bike") == "Y" and props.get("Visible") == "Y" and props.get("Closed") == "N":
            open_features.append(feature)
    return open_features


def arcgis_name_index(arcgis_features: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for feature in arcgis_features:
        name = normalize_trail_name((feature.get("properties") or {}).get("Name"))
        if name:
            index.setdefault(name, []).append(feature)
    return index


def nearest_open_arcgis_trail(connector_feature: dict[str, Any], arcgis_features: list[dict[str, Any]]) -> dict[str, Any] | None:
    connector_points = iter_points(connector_feature.get("geometry") or {})
    if not connector_points:
        return None
    best: tuple[float, dict[str, Any] | None] = (math.inf, None)
    for feature in arcgis_features:
        for connector_point in connector_points:
            for arcgis_point in iter_points(feature.get("geometry") or {}):
                distance = haversine_miles(connector_point, arcgis_point)
                if distance < best[0]:
                    best = (distance, feature)
    if best[1] is None:
        return None
    props = dict(best[1].get("properties") or {})
    return {
        "distance_miles": round(best[0], 4),
        "name": props.get("Name"),
        "trail_num": props.get("TrailNum"),
        "owner": props.get("Owner"),
        "hike_bike": props.get("Hike_Bike"),
        "closed": props.get("Closed"),
        "seasonal": props.get("Seasonal"),
        "difficulty": props.get("Difficulty"),
    }


def connector_source_review(
    connector_name: str,
    *,
    connector_index: dict[str, dict[str, Any]],
    arcgis_open_features: list[dict[str, Any]],
    arcgis_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    feature = connector_index.get(connector_name)
    props = dict((feature or {}).get("properties") or {})
    normalized = normalize_trail_name(connector_name)
    exact_arcgis = arcgis_by_name.get(normalized) or []
    nearest = nearest_open_arcgis_trail(feature, arcgis_open_features) if feature else None
    unsafe = []
    for key in ("access", "foot"):
        value = props.get(key)
        if value is not None and str(value).lower() in {"no", "private"}:
            unsafe.append(f"{key}={value}")

    if connector_name.startswith("OSM "):
        if unsafe:
            status = "blocked_private_or_no_foot"
        elif nearest and nearest["distance_miles"] <= 0.01:
            status = "passed_resolved_to_nearby_open_named_trail"
        elif nearest and nearest["distance_miles"] <= 0.03:
            status = "needs_manual_map_review_near_open_named_trail"
        else:
            status = "blocked_unresolved_opaque_osm_connector"
        cue_name = (
            f"{nearest['name']} {nearest.get('trail_num') or ''}".strip()
            if nearest and nearest.get("name")
            else "opaque OSM connector"
        )
    elif exact_arcgis:
        if unsafe:
            status = "blocked_private_or_no_foot"
        else:
            status = "passed_named_open_hike_bike_trail"
        cue_name = connector_name
    elif props.get("source") == "openstreetmap" and props.get("highway") in {"residential", "service", "footway", "path"} and not unsafe:
        status = "passed_osm_public_road_or_path_no_access_blocker"
        cue_name = connector_name
    else:
        status = "needs_manual_map_review_no_named_open_source"
        cue_name = connector_name

    return {
        "connector_name": connector_name,
        "normalized_name": normalized,
        "status": status,
        "field_cue_name": cue_name,
        "can_cue_without_opaque_osm_id": not connector_name.startswith("OSM ") or status == "passed_resolved_to_nearby_open_named_trail",
        "connector_source": props.get("source"),
        "highway": props.get("highway"),
        "surface": props.get("surface"),
        "access": props.get("access"),
        "foot": props.get("foot"),
        "length_miles": line_length_miles((feature or {}).get("geometry") or {}),
        "unsafe_reasons": unsafe,
        "exact_arcgis_open_feature_count": len(exact_arcgis),
        "nearest_open_arcgis_trail": nearest,
        "public_legal_evidence": public_legal_evidence(connector_name, props, exact_arcgis, nearest, status),
    }


def public_legal_evidence(
    connector_name: str,
    props: dict[str, Any],
    exact_arcgis: list[dict[str, Any]],
    nearest: dict[str, Any] | None,
    status: str,
) -> list[str]:
    evidence = []
    if exact_arcgis:
        sample = exact_arcgis[0].get("properties") or {}
        evidence.append(
            f"Avimor ArcGIS layer marks {sample.get('Name')} ({sample.get('TrailNum')}) Hike_Bike={sample.get('Hike_Bike')} Closed={sample.get('Closed')} Seasonal={sample.get('Seasonal')} Owner={sample.get('Owner')}."
        )
    if connector_name.startswith("OSM ") and nearest:
        evidence.append(
            f"Opaque OSM connector is within {nearest['distance_miles']} mi of Avimor ArcGIS open Hike/Bike trail {nearest.get('name')} ({nearest.get('trail_num')})."
        )
    if props.get("source") == "openstreetmap":
        evidence.append(
            f"OSM connector tags are highway={props.get('highway')} surface={props.get('surface')} access={props.get('access')} foot={props.get('foot')}."
        )
    if "blocked" not in status and "manual" not in status:
        evidence.append("No access=no, access=private, foot=no, or foot=private tag was present in the local connector source.")
    return evidence


def leg_id_for_row(row: dict[str, Any]) -> str:
    target = str(row.get("to_segment_id"))
    for leg_id, definition in LEG_DEFINITIONS.items():
        if target in definition["targets"]:
            return leg_id
    return f"connector_to_{target}"


def leg_reviews(h1_report: dict[str, Any], source_reviews: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = h1_report["repaired_candidate"]["link_rows"]
    reviews = []
    for row in rows:
        connector_names = [str(name) for name in row.get("connector_names") or []]
        if not connector_names and not row.get("official_repeat_segment_ids") and row.get("to_segment_id") != "return_to_car":
            continue
        leg_id = leg_id_for_row(row)
        definition = LEG_DEFINITIONS.get(
            leg_id,
            {
                "title": f"Connector to {row.get('to_segment_name')}",
                "field_cue_name": ", ".join(connector_names),
                "landmark": "Use signed trail/road junctions where available.",
                "fallback": "If unsigned, hold promotion until field/source review.",
            },
        )
        reviews_for_leg = [source_reviews[name] for name in connector_names if name in source_reviews]
        failures = [item for item in reviews_for_leg if item["status"].startswith("blocked")]
        manual = [item for item in reviews_for_leg if item["status"].startswith("needs_manual")]
        opaque = [item for item in reviews_for_leg if not item["can_cue_without_opaque_osm_id"]]
        seasonal = [
            item
            for item in reviews_for_leg
            if ((item.get("nearest_open_arcgis_trail") or {}).get("seasonal") == "Y")
            or any("Seasonal=Y" in e for e in item.get("public_legal_evidence") or [])
        ]
        status = "passed"
        if failures:
            status = "blocked"
        elif manual or opaque:
            status = "needs_manual_map_review"
        elif seasonal:
            status = "passed_with_day_of_seasonal_condition_check"
        reviews.append(
            {
                "leg_id": leg_id,
                "title": definition["title"],
                "to_segment_id": row.get("to_segment_id"),
                "to_segment_name": row.get("to_segment_name"),
                "link_track_miles": row.get("link_track_miles"),
                "connector_names": connector_names,
                "field_cue_name": definition["field_cue_name"],
                "public_legal_evidence": [
                    evidence
                    for item in reviews_for_leg
                    for evidence in item.get("public_legal_evidence") or []
                ][:8],
                "can_cue_without_opaque_osm_ids": not opaque and not manual and not failures,
                "expected_field_landmark_or_junction": definition["landmark"],
                "fallback_if_missed": definition["fallback"],
                "official_repeat_segment_ids": row.get("official_repeat_segment_ids") or [],
                "official_repeat_miles": row.get("official_repeat_miles"),
                "status": status,
                "status_reason": leg_status_reason(status, failures, manual, opaque, seasonal),
            }
        )
    return reviews


def leg_status_reason(
    status: str,
    failures: list[dict[str, Any]],
    manual: list[dict[str, Any]],
    opaque: list[dict[str, Any]],
    seasonal: list[dict[str, Any]],
) -> str:
    if status == "blocked":
        return "Blocked by private/no-foot connector evidence: " + ", ".join(item["connector_name"] for item in failures)
    if status == "needs_manual_map_review":
        names = [item["connector_name"] for item in manual or opaque]
        return "One or more connectors remain opaque or lack a nearby named open-trail resolution: " + ", ".join(names)
    if status == "passed_with_day_of_seasonal_condition_check":
        names = sorted({(item.get("nearest_open_arcgis_trail") or {}).get("name") or item["connector_name"] for item in seasonal})
        return "Cueable from public named-trail evidence; seasonal/day-of open-condition check remains for " + ", ".join(names)
    return "All non-official connector pieces can be cued as named trail, public road, or nearby open Hike/Bike trail without relying on opaque OSM IDs."


def graph_sanity_rows(source_reviews: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for review in sorted(source_reviews.values(), key=lambda row: row["connector_name"]):
        rows.append(
            {
                "connector_name": review["connector_name"],
                "is_real_trail_road_or_path": review.get("highway") in {"path", "footway", "residential", "track", "service"}
                or review["exact_arcgis_open_feature_count"] > 0,
                "visible_on_public_map_source": review["exact_arcgis_open_feature_count"] > 0
                or (review.get("nearest_open_arcgis_trail") or {}).get("distance_miles", 99) <= 0.01
                or review.get("connector_source") == "openstreetmap",
                "allowed_for_foot_travel": not review.get("unsafe_reasons"),
                "private_or_no_access_geometry": review.get("unsafe_reasons") or [],
                "driveway_or_private_neighborhood_risk": driveway_risk(review),
                "field_cue_name": review["field_cue_name"],
                "status": review["status"],
            }
        )
    return rows


def driveway_risk(review: dict[str, Any]) -> str:
    name = review["connector_name"]
    if name.startswith("OSM footway connector"):
        return "low_greenbelt_or_park_path_context"
    if name == "North Smokeys Draw Place":
        return "public_residential_road_connector_not_driveway"
    if review["status"].startswith("needs_manual"):
        return "unknown_until_manual_map_review"
    return "low_named_trail_or_public_road"


def readiness_checks(leg_review_rows: list[dict[str, Any]], h1_report: dict[str, Any]) -> list[dict[str, Any]]:
    by_leg = {row["leg_id"]: row for row in leg_review_rows}
    checks = [
        {
            "check": "start parking",
            "status": "passed",
            "evidence": h1_report.get("parking_source_sync") or {},
            "blocker": None,
        },
        {
            "check": "access to Twisted Spring",
            "status": by_leg["access_to_twisted_spring"]["status"],
            "evidence": by_leg["access_to_twisted_spring"]["status_reason"],
            "blocker": blocker_for_status(by_leg["access_to_twisted_spring"]["status"]),
        },
        {
            "check": "Ricochet/Shooting/Whistling connector",
            "status": by_leg["ricochet_shooting_whistling"]["status"],
            "evidence": by_leg["ricochet_shooting_whistling"]["status_reason"],
            "blocker": blocker_for_status(by_leg["ricochet_shooting_whistling"]["status"]),
        },
        {
            "check": "Spring Creek connector",
            "status": by_leg["spring_creek_connector"]["status"],
            "evidence": by_leg["spring_creek_connector"]["status_reason"],
            "blocker": blocker_for_status(by_leg["spring_creek_connector"]["status"]),
        },
        {
            "check": "Harlow connector via Burnt Car / Cartwright / The Wall",
            "status": by_leg["harlows_connector"]["status"],
            "evidence": by_leg["harlows_connector"]["status_reason"],
            "blocker": blocker_for_status(by_leg["harlows_connector"]["status"]),
        },
        {
            "check": "return to Avimor",
            "status": by_leg["return_to_avimor"]["status"],
            "evidence": by_leg["return_to_avimor"]["status_reason"],
            "blocker": blocker_for_status(by_leg["return_to_avimor"]["status"]),
        },
    ]
    return checks


def blocker_for_status(status: str) -> str | None:
    if status in {"blocked", "needs_manual_map_review"}:
        return "needs_public_safe_cueable_access_review"
    return None


def replacement_segment_diff(h1_report: dict[str, Any], field_tool_data: dict[str, Any]) -> dict[str, Any]:
    replaced_routes = [route for route in field_tool_data.get("routes") or [] if route.get("label") in H1_REPLACE_ROUTE_LABELS]
    old_ids = sorted({str(segment_id) for route in replaced_routes for segment_id in route.get("segment_ids") or []}, key=sort_id)
    new_ids = sorted({str(segment_id) for segment_id in h1_report["repaired_candidate"]["official_segment_ids"]}, key=sort_id)
    current_scope = h1_report["current_scope"]
    new_candidate = h1_report["repaired_candidate"]
    return {
        "claimed_ids": new_ids,
        "replaced_old_claimed_ids": old_ids,
        "missing_ids": sorted(set(old_ids) - set(new_ids), key=sort_id),
        "extra_ids": sorted(set(new_ids) - set(old_ids), key=sort_id),
        "old_route_cards_removed": H1_REPLACE_ROUTE_LABELS,
        "old_official_miles": round(sum(float_value(route.get("official_miles")) for route in replaced_routes), 2),
        "new_official_miles": new_candidate.get("official_miles"),
        "old_on_foot_miles": current_scope.get("on_foot_miles"),
        "new_on_foot_miles": new_candidate.get("track_miles"),
        "old_p75_minutes": current_scope.get("p75_minutes"),
        "new_p75_minutes": new_candidate["dem_pricing"]["time_estimates_minutes"]["door_to_door_p75"],
        "old_p90_minutes": current_scope.get("p90_minutes"),
        "new_p90_minutes": new_candidate["dem_pricing"]["time_estimates_minutes"]["door_to_door_p90"],
    }


def build_report(args: argparse.Namespace, *, arcgis_geojson: dict[str, Any] | None = None) -> dict[str, Any]:
    h1_report = read_json(args.h1_audit_json)
    field_tool_data = read_json(args.field_tool_data_json)
    connector_geojson = read_json(args.connector_geojson)
    if arcgis_geojson is None:
        arcgis_geojson = fetch_arcgis_hike_bike_features(args.avimor_trails_layer_url)
    connector_index = connector_feature_index(connector_geojson)
    arcgis_open = open_arcgis_features(arcgis_geojson)
    by_name = arcgis_name_index(arcgis_open)
    connector_names = sorted(
        {
            str(name)
            for row in h1_report["repaired_candidate"]["link_rows"]
            for name in row.get("connector_names") or []
        }
    )
    source_reviews = {
        name: connector_source_review(
            name,
            connector_index=connector_index,
            arcgis_open_features=arcgis_open,
            arcgis_by_name=by_name,
        )
        for name in connector_names
    }
    leg_review_rows = leg_reviews(h1_report, source_reviews)
    graph_rows = graph_sanity_rows(source_reviews)
    promotion_checks = readiness_checks(leg_review_rows, h1_report)
    hard_blockers = sorted({check["blocker"] for check in promotion_checks if check.get("blocker")})
    segment_diff = replacement_segment_diff(h1_report, field_tool_data)
    if hard_blockers:
        decision = "keep_gated_needs_access_repair"
        access_status = "failed"
    else:
        decision = "access_cue_review_passed_keep_unpromoted"
        access_status = "passed_with_day_of_condition_check"
    return {
        "schema": "boise_trails_harlow_h1_access_cue_review_v1",
        "generated_at": now_iso(),
        "objective": "Decide whether H1's repaired connector graph is public-safe and cueable enough to clear the access/cue blocker, without promoting route cards.",
        "candidate_id": h1_report["candidate_id"],
        "decision": decision,
        "access_cue_review_status": access_status,
        "source_files": {
            "h1_audit_json": display_path(args.h1_audit_json),
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "connector_geojson": display_path(args.connector_geojson),
        },
        "public_source_references": PUBLIC_SOURCE_REFERENCES,
        "arcgis_source_summary": {
            "layer_url": args.avimor_trails_layer_url,
            "feature_count": len(arcgis_geojson.get("features") or []),
            "open_hike_bike_feature_count": len(arcgis_open),
        },
        "leg_by_leg_cueability": leg_review_rows,
        "graph_to_field_sanity": graph_rows,
        "promotion_readiness": {
            "status": "access_gate_clear_keep_unpromoted" if not hard_blockers else "access_gate_blocked",
            "checks": promotion_checks,
            "remaining_blockers_after_access_review": [
                *hard_blockers,
                "needs_field_packet_route_card_promotion",
                "needs_field_packet_recertification",
            ],
            "not_promotion_note": "This clears or classifies the access/cue review only. It does not remove the existing Harlow/Avimor route cards from the active field packet.",
        },
        "h1_replacement_segment_set_diff": segment_diff,
        "frame_shift": {
            "artifact_contract": "Public-safe access/cue review for the candidate route, not a route promotion.",
            "downstream_question": "Can a runner leave the Avimor parking lot, follow named/signed public routes without opaque OSM IDs, and return to the car without hidden private/no-foot geometry?",
            "negative_space": "This does not prove day-of trail conditions, live signage, or full field-packet readiness after route-card replacement.",
            "decision": "keep-gated until route-card promotion and recertification even when the access/cue gate passes.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Harlow / Avimor H1 Access Cue Review",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Summary",
        "",
        f"- Access/cue review status: `{report['access_cue_review_status']}`",
        f"- Avimor ArcGIS features checked: {report['arcgis_source_summary']['feature_count']} total / {report['arcgis_source_summary']['open_hike_bike_feature_count']} open Hike/Bike visible",
        f"- Promotion status: `{report['promotion_readiness']['status']}`",
        f"- Remaining blockers after this review: {', '.join(report['promotion_readiness']['remaining_blockers_after_access_review'])}",
        "",
        "## Leg-by-leg cueability",
        "",
        "| Leg | Status | Field cue name | Public/legal evidence | Opaque OSM-free? | Landmark / junction | Fallback |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["leg_by_leg_cueability"]:
        evidence = "; ".join(row.get("public_legal_evidence") or []) or "none"
        lines.append(
            f"| {row['title']} | `{row['status']}` | {row['field_cue_name']} | {evidence} | {str(row['can_cue_without_opaque_osm_ids']).lower()} | {row['expected_field_landmark_or_junction']} | {row['fallback_if_missed']} |"
        )
    lines.extend(
        [
            "",
            "## Graph-to-field sanity",
            "",
            "| Connector | Status | Field cue name | Real path/road? | Public map/source visible | Foot allowed | Private/no-access risk | Driveway/neighborhood risk |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in report["graph_to_field_sanity"]:
        lines.append(
            f"| {row['connector_name']} | `{row['status']}` | {row['field_cue_name']} | {str(row['is_real_trail_road_or_path']).lower()} | {str(row['visible_on_public_map_source']).lower()} | {str(row['allowed_for_foot_travel']).lower()} | {', '.join(row['private_or_no_access_geometry']) or 'none'} | {row['driveway_or_private_neighborhood_risk']} |"
        )
    lines.extend(
        [
            "",
            "## Promotion readiness",
            "",
            "| Check | Status | Blocker | Evidence |",
            "|---|---|---|---|",
        ]
    )
    for row in report["promotion_readiness"]["checks"]:
        evidence = row.get("evidence")
        if not isinstance(evidence, str):
            evidence = json.dumps(evidence, sort_keys=True)
        lines.append(f"| {row['check']} | `{row['status']}` | {row.get('blocker') or 'none'} | {evidence} |")
    diff = report["h1_replacement_segment_set_diff"]
    lines.extend(
        [
            "",
            "## Replacement segment-set diff",
            "",
            f"- Claimed ids: `{', '.join(diff['claimed_ids'])}`",
            f"- Replaced old claimed ids: `{', '.join(diff['replaced_old_claimed_ids'])}`",
            f"- Missing ids: {diff['missing_ids']}",
            f"- Extra ids: {diff['extra_ids']}",
            f"- Old route cards removed: {', '.join(diff['old_route_cards_removed'])}",
            f"- Old official miles / new official miles: {diff['old_official_miles']} / {diff['new_official_miles']}",
            f"- Old on-foot / new on-foot: {diff['old_on_foot_miles']} / {diff['new_on_foot_miles']}",
            f"- Old p75 / new p75: {diff['old_p75_minutes']} / {diff['new_p75_minutes']}",
            f"- Old p90 / new p90: {diff['old_p90_minutes']} / {diff['new_p90_minutes']}",
            "",
            "## Public source references",
            "",
        ]
    )
    for source in report["public_source_references"]:
        lines.append(f"- `{source['source_id']}`: {source['url']} - {source['role']}")
    lines.extend(
        [
            "",
            "## Not Promoted",
            "",
            "This artifact reviews whether the H1 access and connector cues are source-backed and field-readable. It does not remove FD27A, FD27B, FD27C, FD24A, or FD30A, and it does not regenerate the active route-card set.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1-audit-json", type=Path, default=DEFAULT_H1_AUDIT_JSON)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--avimor-trails-layer-url", default=AVIMOR_TRAILS_LAYER_URL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="harlow_h1_access_cue_review",
        inputs=[args.h1_audit_json, args.field_tool_data_json, args.connector_geojson],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/harlow_h1_access_cue_review.py",
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.manifest_json}")
    print(json.dumps(report["promotion_readiness"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
