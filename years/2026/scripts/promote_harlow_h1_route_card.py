#!/usr/bin/env python3
"""Promote the repaired H1 Harlow/Avimor route into the canonical field menu."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_route_candidate_pass import PALETTE, multiline_feature, point_feature  # noqa: E402
from export_execution_gpx import validate_track_segments  # noqa: E402
from harlow_h1_gate_repair_audit import (  # noqa: E402
    H1_BUNDLE_ID,
    H1_REPLACE_ROUTE_LABELS,
    H1_ROUTE_LABEL,
    H1_SEGMENT_IDS,
    sort_id,
)
from human_loop_plan import (  # noqa: E402
    render_outing_menu_markdown,
    render_package_map_html,
    sync_official_segment_features,
)
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments, round_miles  # noqa: E402


DEFAULT_BASE_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_OUTPUT_MAP_DATA_JSON = DEFAULT_BASE_MAP_DATA_JSON
DEFAULT_OUTPUT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTPUT_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_H1_AUDIT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-gate-repair-audit-2026-05-12.json"
DEFAULT_H1_ACCESS_REVIEW_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-access-cue-review-2026-05-12.json"
DEFAULT_FIELD_DAY_LAYER_JSON = YEAR_DIR / "checkpoints" / "human-executable-field-day-layer-2026-05-10.json"
DEFAULT_PRIOR_FIELD_DAY_PROMOTION_JSON = YEAR_DIR / "checkpoints" / "field-day-loop-promotion-2026-05-11.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-route-card-promotion-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "harlow-h1-route-card-promotion-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-route-card-promotion-2026-05-12-manifest.json"
DEFAULT_OFFICIAL_SEGMENTS_GEOJSON = DEFAULT_OFFICIAL_GEOJSON
DEFAULT_ASSIGNED_DATE = "2026-07-04"
DEFAULT_ASSIGNED_DRAFT_DAY_NUMBER = 27
DEFAULT_WEEKEND_P90_BOUND_MINUTES = 360
H1_FIELD_MENU_LABEL = "H1"
H1_LOOP_ID = "harlow_h1_route_card_replacement::2026-07-04::avimor"
H1_PACKAGE_NUMBER = 127
H1_BLOCK_ID = "field-day-27-h1-harlow-avimor"
H1_BLOCK_NAME = "Field Day 27 H1 Harlow/Avimor replacement"
H1_TRAILHEAD = "Avimor Spring Valley Creek parking"
H1_TRAIL_NAMES = [
    "Twisted Spring",
    "Ricochet",
    "Shooting Range",
    "Whistling Pig",
    "Spring Creek",
    "Harlow's Hollows",
    "Harlow's Hollows Connector",
]
CONNECTOR_NAME_OVERRIDES = {
    "1687": ["McLeod Way Greenbelt", "Twisted Spring Trail - #8"],
    "1626": ["Ricochet - #2"],
    "1657": ["North Smokeys Draw Place", "Ricochet - #2", "Shooting Range - #5"],
    "1696": ["Whistling Pig - #3"],
    "1661": ["Whistling Pig - #3", "Twisted Spring Trail - #8", "Spring Creek - #9"],
    "1704": ["Burnt Car Draw - #10", "Cartwright Road - #20", "The Wall - #29", "Harlow's Hollows - #16"],
    "return_to_car": ["Spring Creek - #9", "Twisted Spring Trail - #8", "McLeod Way Greenbelt"],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=sort_id)


def parse_gpx_track(path: Path) -> list[tuple[float, float]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    coords: list[tuple[float, float]] = []
    for trkpt in root.findall(".//{*}trkpt"):
        lat = trkpt.get("lat")
        lon = trkpt.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lon), float(lat)))
    if len(coords) < 2:
        raise ValueError(f"H1 GPX has no usable track: {path}")
    return coords


def route_labels(map_data: dict[str, Any]) -> list[str]:
    return [
        str(component.get("field_menu_label") or "")
        for package in map_data.get("packages") or []
        for component in package.get("components") or []
    ]


def route_components_by_label(map_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(component.get("field_menu_label")): component
        for package in map_data.get("packages") or []
        for component in package.get("components") or []
        if component.get("field_menu_label")
    }


def removed_components_from_field_day_layer(path: Path = DEFAULT_FIELD_DAY_LAYER_JSON) -> list[dict[str, Any]]:
    components = []
    if path.exists():
        layer = read_json(path)
        for day in layer.get("field_days") or []:
            for loop in day.get("loops") or []:
                label = str(loop.get("label") or "")
                if label not in set(H1_REPLACE_ROUTE_LABELS):
                    continue
                components.append(
                    {
                        "field_menu_label": label,
                        "candidate_id": (loop.get("candidate_ids") or [loop.get("candidate_id")])[0],
                        "field_menu_group_id": loop.get("loop_id"),
                        "source_loop_id": loop.get("loop_id"),
                        "source": loop.get("source"),
                        "trailhead": loop.get("trailhead"),
                        "official_miles": loop.get("official_miles"),
                        "on_foot_miles": loop.get("on_foot_miles"),
                        "total_minutes": loop.get("p75_minutes"),
                        "time_estimates_minutes": {"door_to_door_p90": loop.get("p90_minutes")},
                        "segment_ids": loop.get("segment_ids") or [],
                    }
                )
    if not components and DEFAULT_OUTPUT_JSON.exists():
        prior_h1 = read_json(DEFAULT_OUTPUT_JSON)
        for promotion in prior_h1.get("promotions") or []:
            label = str(promotion.get("label") or "")
            if label not in set(H1_REPLACE_ROUTE_LABELS):
                continue
            components.append(
                {
                    "field_menu_label": label,
                    "candidate_id": promotion.get("source_candidate_id"),
                    "field_menu_group_id": promotion.get("loop_id"),
                    "source_loop_id": promotion.get("loop_id"),
                    "source": promotion.get("source"),
                    "trailhead": promotion.get("trailhead"),
                    "official_miles": promotion.get("official_miles"),
                    "on_foot_miles": promotion.get("on_foot_miles"),
                    "total_minutes": promotion.get("p75_minutes"),
                    "time_estimates_minutes": {"door_to_door_p90": promotion.get("p90_minutes")},
                    "segment_ids": [],
                }
            )
    return sorted(components, key=lambda item: H1_REPLACE_ROUTE_LABELS.index(str(item.get("field_menu_label"))))


def existing_official_feature_index(map_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = ((map_data.get("feature_collections") or {}).get("official_segments") or {}).get("features") or []
    return {
        str((feature.get("properties") or {}).get("seg_id")): feature
        for feature in features
        if (feature.get("properties") or {}).get("seg_id") is not None
    }


def segment_rows(map_data: dict[str, Any], order: list[str]) -> list[dict[str, Any]]:
    index = existing_official_feature_index(map_data)
    rows = []
    for seq, segment_id in enumerate(order, start=1):
        feature = index.get(str(segment_id))
        props = copy.deepcopy((feature or {}).get("properties") or {})
        if not props:
            raise ValueError(f"Missing official segment feature for H1 segment {segment_id}")
        rows.append(
            {
                "order": seq,
                "seg_id": int(segment_id),
                "segment_name": props.get("segment_name") or props.get("seg_name") or props.get("trail_name"),
                "trail_name": props.get("trail_name"),
                "signpost_label": props.get("signpost_label") or "",
                "official_miles": props.get("segment_official_miles") or props.get("official_miles"),
                "direction_rule": props.get("direction_rule") or props.get("direction") or "both",
                "direction_cue": props.get("direction_cue") or "Either direction allowed; follow map arrows.",
            }
        )
    return rows


def names_without_osm(names: list[Any]) -> list[str]:
    result = []
    for name in names:
        text = str(name or "").strip()
        if not text or text.lower().startswith("osm "):
            continue
        if text not in result:
            result.append(text)
    return result


def signpost_labels(names: list[str]) -> list[str]:
    return [name for name in names if "#" in name]


def cue_link(row: dict[str, Any], *, use_mapped_access_key: bool = False) -> dict[str, Any]:
    target = str(row.get("to_segment_id"))
    names = CONNECTOR_NAME_OVERRIDES.get(target) or names_without_osm(row.get("connector_names") or [])
    link_miles = float(row.get("link_track_miles") or row.get("link_distance_miles") or 0)
    result = {
        "from_trail": row.get("from_trail"),
        "to_trail": row.get("to_trail_name"),
        "distance_miles": round_miles(link_miles),
        "connector_miles": round_miles(link_miles),
        "official_repeat_miles": row.get("official_repeat_miles"),
        "official_repeat_segment_ids": normalized_ids(row.get("official_repeat_segment_ids") or []),
        "connector_names": names,
        "signpost_labels": signpost_labels(names),
        "connector_classes": row.get("connector_classes") or [],
        "graph_validated": row.get("path_source") != "direct_gap_fallback",
    }
    if use_mapped_access_key:
        result["mapped_access_miles"] = result.pop("distance_miles")
        result["access_class"] = "named_public_access"
        result["confidence"] = "high"
    if target == "return_to_car":
        result["strategy"] = "mapped_mixed_loop"
        result["description"] = (
            "Return to Avimor Spring Valley Creek parking on signed Spring Creek, "
            "Twisted Spring, and McLeod Way Greenbelt connectors."
        )
    return {key: value for key, value in result.items() if value not in (None, "", [])}


def h1_between_links(repaired: dict[str, Any]) -> list[dict[str, Any]]:
    rows = {str(row.get("to_segment_id")): row for row in repaired.get("link_rows") or []}
    transitions = ["1626", "1657", "1696", "1661", "1704", "1708", "1706"]
    links = []
    for target in transitions:
        link = cue_link(rows[target])
        if target == "1626":
            link["from_trail"] = "Twisted Spring"
            link["to_trail"] = "Ricochet"
        if target == "1696":
            link["from_trail"] = "Shooting Range"
            link["to_trail"] = "Whistling Pig"
        links.append(link)
    return links


def h1_route_cue(map_data: dict[str, Any], h1_audit: dict[str, Any]) -> dict[str, Any]:
    repaired = h1_audit["repaired_candidate"]
    rows = {str(row.get("to_segment_id")): row for row in repaired.get("link_rows") or []}
    pricing = repaired["dem_pricing"]
    parking = copy.deepcopy(repaired["parking"])
    parking.setdefault("parking_minutes", 8)
    return {
        "candidate_id": H1_BUNDLE_ID,
        "title": H1_ROUTE_LABEL,
        "trailhead": parking,
        "segments": segment_rows(map_data, list(repaired.get("traversed_segment_order") or H1_SEGMENT_IDS)),
        "start_access": cue_link(rows["1687"], use_mapped_access_key=True),
        "between_links": h1_between_links(repaired),
        "return_to_car": cue_link(rows["return_to_car"]),
        "time_estimates_minutes": pricing["time_estimates_minutes"],
        "effort": pricing["effort"],
        "official_miles": repaired["official_miles"],
        "on_foot_miles": repaired["track_miles"],
        "route_status": "promoted_harlow_h1_cluster_replacement",
        "promotion_notes": [
            "Promoted as the all-or-nothing Harlow/Avimor replacement for FD27A, FD27B, FD27C, FD24A, and FD30A.",
            "Phone-visible connector names are the access-review field names, not opaque connector ids.",
        ],
    }


def h1_component(h1_audit: dict[str, Any]) -> dict[str, Any]:
    repaired = h1_audit["repaired_candidate"]
    pricing = repaired["dem_pricing"]
    official = float(repaired["official_miles"])
    on_foot = float(repaired["track_miles"])
    return {
        "route_number": 25,
        "candidate_id": H1_BUNDLE_ID,
        "field_menu_group_id": H1_LOOP_ID,
        "field_menu_label": H1_FIELD_MENU_LABEL,
        "trail_names": H1_TRAIL_NAMES,
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2),
        "total_minutes": int(pricing["time_estimates_minutes"]["door_to_door_p75"]),
        "raw_total_minutes": int(pricing["time_estimates_minutes"]["door_to_door_raw"]),
        "trailhead": H1_TRAILHEAD,
        "less_optimal_flags": [],
        "segment_ids": [int(segment_id) for segment_id in normalized_ids(H1_SEGMENT_IDS)],
        "time_breakdown_minutes": None,
        "time_estimates_minutes": pricing["time_estimates_minutes"],
        "effort": pricing["effort"],
        "route_status": "promoted_harlow_h1_cluster_replacement",
        "source": "harlow_h1_route_card_promotion",
        "source_loop_id": H1_LOOP_ID,
        "source_candidate_id": H1_BUNDLE_ID,
    }


def package_for_h1(component: dict[str, Any]) -> dict[str, Any]:
    return {
        "package_number": H1_PACKAGE_NUMBER,
        "block_id": H1_BLOCK_ID,
        "block_name": H1_BLOCK_NAME,
        "source_field_day_id": "harlow-h1-route-card-replacement",
        "source_date": DEFAULT_ASSIGNED_DATE,
        "component_route_count": 1,
        "component_candidate_ids": [component["candidate_id"]],
        "trail_names": component["trail_names"],
        "official_miles": component["official_miles"],
        "on_foot_miles": component["on_foot_miles"],
        "ratio": component["ratio"],
        "trailheads": [H1_TRAILHEAD],
        "trailhead_count": 1,
        "primary_trailhead": H1_TRAILHEAD,
        "total_minutes_components": component["total_minutes"],
        "component_routes_under_1_official_mile": 0,
        "component_routes_under_2_official_miles": 0,
        "segment_ids": component["segment_ids"],
        "components": [component],
        "planning_status": "harlow_h1_cluster_bundle_promoted",
        "planning_reasons": [
            "h1_access_cue_gate_cleared",
            "h1_segment_set_equals_removed_harlow_avimor_union",
            "h1_reduces_cluster_on_foot_and_p75_cost",
            "weekend_p90_bound_passes",
        ],
    }


def recalculate_summary(map_data: dict[str, Any], before_summary: dict[str, Any], h1_access: dict[str, Any]) -> None:
    packages = map_data.get("packages") or []
    segment_ids = sorted({int(seg_id) for package in packages for seg_id in package.get("segment_ids") or []})
    total_official = round_miles(sum(float(package.get("official_miles") or 0) for package in packages))
    total_on_foot = round_miles(sum(float(package.get("on_foot_miles") or 0) for package in packages))
    summary = copy.deepcopy(before_summary)
    summary.update(
        {
            "source": "harlow_h1_route_card_promotion",
            "package_count": len(packages),
            "component_route_count": sum(len(package.get("components") or []) for package in packages),
            "covered_segment_count": len(segment_ids),
            "official_miles": total_official,
            "total_on_foot_miles": total_on_foot,
            "planwide_on_foot_to_official_ratio": round(total_on_foot / total_official, 2) if total_official else None,
            "harlow_h1_cluster_replacement": h1_access.get("h1_replacement_segment_set_diff") or {},
        }
    )
    map_data["summary"] = summary


def remove_old_features(features: list[dict[str, Any]], removed_candidate_ids: set[str]) -> list[dict[str, Any]]:
    return [
        feature
        for feature in features
        if str((feature.get("properties") or {}).get("candidate_id")) not in removed_candidate_ids
    ]


def build_h1_features(
    *,
    h1_audit: dict[str, Any],
    coords: list[tuple[float, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    component = h1_component(h1_audit)
    props = {
        "kind": "route",
        "route_number": component["route_number"],
        "package_number": H1_PACKAGE_NUMBER,
        "candidate_id": H1_BUNDLE_ID,
        "block_name": H1_BLOCK_NAME,
        "title": H1_ROUTE_LABEL,
        "official_miles": component["official_miles"],
        "on_foot_miles": component["on_foot_miles"],
        "trailhead": H1_TRAILHEAD,
        "color": PALETTE[(component["route_number"] - 1) % len(PALETTE)],
        "field_menu_label": H1_FIELD_MENU_LABEL,
    }
    route = multiline_feature([coords], props)
    if route is None:
        raise ValueError("H1 route feature could not be built")
    parking = h1_audit["repaired_candidate"]["parking"]
    parking_feature = point_feature(
        float(parking["lon"]),
        float(parking["lat"]),
        {
            **props,
            "kind": "parking",
            "name": parking.get("name") or H1_TRAILHEAD,
            "trailhead": H1_TRAILHEAD,
            "has_parking": True,
            "has_restroom": parking.get("has_restroom"),
            "has_water": parking.get("has_water"),
            "water_confidence": parking.get("water_confidence"),
            "parking_minutes": parking.get("parking_minutes") or 8,
            "source": parking.get("source"),
            "parking_confidence": parking.get("parking_confidence"),
            "field_ready": parking.get("field_ready"),
            "nearest_open_trail_name": parking.get("nearest_open_trail_name"),
            "nearest_open_trail_label": parking.get("nearest_open_trail_label"),
        },
    )
    return route, parking_feature


def promote_map_data(
    *,
    base_map_data: dict[str, Any],
    h1_audit: dict[str, Any],
    h1_access: dict[str, Any],
    official_segments: list[dict[str, Any]],
    h1_gpx_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = copy.deepcopy(base_map_data)
    before_summary = copy.deepcopy(before.get("summary") or {})
    by_label = route_components_by_label(before)
    missing_old = [label for label in H1_REPLACE_ROUTE_LABELS if label not in by_label]
    already_promoted = missing_old and H1_FIELD_MENU_LABEL in set(route_labels(before))
    if missing_old and not already_promoted:
        raise ValueError(f"Cannot promote H1; old route labels missing from canonical source: {missing_old}")

    replacement_diff = h1_access.get("h1_replacement_segment_set_diff") or {}
    if normalized_ids(replacement_diff.get("missing_ids")) or normalized_ids(replacement_diff.get("extra_ids")):
        raise ValueError("Cannot promote H1; replacement segment-set diff is not exact")
    if (h1_access.get("promotion_readiness") or {}).get("status") != "access_gate_clear_keep_unpromoted":
        raise ValueError("Cannot promote H1; access/cue gate is not clear")

    h1_segment_ids = normalized_ids(H1_SEGMENT_IDS)
    removed_components = (
        removed_components_from_field_day_layer()
        if already_promoted
        else [by_label[label] for label in H1_REPLACE_ROUTE_LABELS]
    )
    if len(removed_components) != len(H1_REPLACE_ROUTE_LABELS):
        raise ValueError("Cannot promote H1; old route component details are missing from the field-day layer")
    removed_candidate_ids = {str(component.get("candidate_id")) for component in removed_components}
    removed_loop_ids = [str(component.get("field_menu_group_id") or component.get("source_loop_id")) for component in removed_components]
    removed_segment_ids = normalized_ids(
        [segment_id for component in removed_components for segment_id in component.get("segment_ids") or []]
    )
    if already_promoted and not removed_segment_ids:
        removed_segment_ids = normalized_ids(replacement_diff.get("replaced_old_claimed_ids") or [])
    if removed_segment_ids != h1_segment_ids:
        raise ValueError(f"H1 segment ids do not match removed union: {removed_segment_ids} != {h1_segment_ids}")

    coords = parse_gpx_track(h1_gpx_path)
    validation = validate_track_segments([coords], max_gap_miles=0.1)
    if not validation.get("passed"):
        raise ValueError(f"H1 GPX is not continuous: {validation.get('failures')}")

    promoted = copy.deepcopy(base_map_data)
    promoted["schema"] = "boise_trails_harlow_h1_promoted_route_card_source_v1"
    promoted["run_id"] = "harlow-h1-route-card-promotion-2026-05-12"
    promoted["generated_at"] = now_iso()
    h1_pkg = package_for_h1(h1_component(h1_audit))
    if not already_promoted:
        packages = []
        for package in promoted.get("packages") or []:
            kept_components = [
                component
                for component in package.get("components") or []
                if str(component.get("field_menu_label")) not in set(H1_REPLACE_ROUTE_LABELS)
            ]
            if int(package.get("package_number") or 0) == H1_PACKAGE_NUMBER:
                packages.append(h1_pkg)
                continue
            if not kept_components:
                continue
            if len(kept_components) != len(package.get("components") or []):
                updated = copy.deepcopy(package)
                updated["components"] = kept_components
                updated["component_route_count"] = len(kept_components)
                updated["component_candidate_ids"] = [component["candidate_id"] for component in kept_components]
                updated["segment_ids"] = sorted({int(seg_id) for component in kept_components for seg_id in component.get("segment_ids") or []})
                updated["official_miles"] = round_miles(sum(float(component.get("official_miles") or 0) for component in kept_components))
                updated["on_foot_miles"] = round_miles(sum(float(component.get("on_foot_miles") or 0) for component in kept_components))
                updated["total_minutes_components"] = sum(int(component.get("total_minutes") or 0) for component in kept_components)
                packages.append(updated)
            else:
                packages.append(package)
        promoted["packages"] = packages
        route_feature, parking_feature = build_h1_features(h1_audit=h1_audit, coords=coords)
        collections = promoted.setdefault("feature_collections", {})
        routes = collections.setdefault("routes", {"type": "FeatureCollection", "features": []})
        parking = collections.setdefault("parking", {"type": "FeatureCollection", "features": []})
        logistics = collections.setdefault("logistics", {"type": "FeatureCollection", "features": []})
        routes["features"] = remove_old_features(routes.get("features") or [], removed_candidate_ids) + [route_feature]
        parking["features"] = remove_old_features(parking.get("features") or [], removed_candidate_ids) + [parking_feature]
        logistics["features"] = remove_old_features(logistics.get("features") or [], removed_candidate_ids)

        sync_official_segment_features(promoted, official_segments)

    route_cues = promoted.setdefault("route_cues", {})
    for candidate_id in removed_candidate_ids:
        route_cues.pop(candidate_id, None)
    route_cues[H1_BUNDLE_ID] = h1_route_cue(base_map_data, h1_audit)

    route_validations = (promoted.setdefault("map_validation", {}).setdefault("route_validations", []))
    route_validations[:] = [
        row
        for row in route_validations
        if str(row.get("candidate_id")) not in removed_candidate_ids | {H1_BUNDLE_ID}
    ]
    route_validations.append(
        {
            "candidate_id": H1_BUNDLE_ID,
            "source_loop_id": H1_LOOP_ID,
            "track_source": display_path(h1_gpx_path),
            "source_gap_warning": False,
            "source_max_gap_miles": validation.get("max_trackpoint_gap_miles"),
            "rendered_passed": validation.get("passed"),
            "rendered_failures": validation.get("failures") or [],
        }
    )
    promoted["source_files"] = {
        **(promoted.get("source_files") or {}),
        "harlow_h1_gate_repair_audit": display_path(DEFAULT_H1_AUDIT_JSON),
        "harlow_h1_access_cue_review": display_path(DEFAULT_H1_ACCESS_REVIEW_JSON),
        "harlow_h1_repaired_gpx": display_path(h1_gpx_path),
    }
    recalculate_summary(promoted, before_summary, h1_access)
    promotion_state = {
        "removed_components": removed_components,
        "removed_candidate_ids": sorted(removed_candidate_ids),
        "removed_loop_ids": removed_loop_ids,
        "removed_segment_ids": removed_segment_ids,
        "h1_segment_ids": h1_segment_ids,
        "old_route_count": (
            int(before_summary.get("component_route_count") or 0) + len(H1_REPLACE_ROUTE_LABELS) - 1
            if already_promoted
            else int(before_summary.get("component_route_count") or 0)
        ),
        "new_route_count": int((promoted.get("summary") or {}).get("component_route_count") or 0),
        "old_total_on_foot_miles": float(before_summary.get("total_on_foot_miles") or 0),
        "new_total_on_foot_miles": float((promoted.get("summary") or {}).get("total_on_foot_miles") or 0),
        "h1_gpx_validation": validation,
    }
    return promoted, promotion_state


def h1_loop_row(h1_audit: dict[str, Any]) -> dict[str, Any]:
    component = h1_component(h1_audit)
    estimates = component["time_estimates_minutes"]
    return {
        "loop_id": H1_LOOP_ID,
        "source": "harlow_h1_route_card_promotion",
        "candidate_id": H1_BUNDLE_ID,
        "label": H1_FIELD_MENU_LABEL,
        "trailhead": H1_TRAILHEAD,
        "trail_names": H1_TRAIL_NAMES,
        "segment_ids": component["segment_ids"],
        "segment_count": len(component["segment_ids"]),
        "official_miles": component["official_miles"],
        "on_foot_miles": component["on_foot_miles"],
        "p75_minutes": estimates["door_to_door_p75"],
        "p90_minutes": estimates["door_to_door_p90"],
        "validation_passed": True,
    }


def field_day_replacement_payload(h1_audit: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    loop = h1_loop_row(h1_audit)
    return {
        "match": {"date": DEFAULT_ASSIGNED_DATE, "draft_day_number": DEFAULT_ASSIGNED_DRAFT_DAY_NUMBER},
        "field_day": {
            "draft_day_number": DEFAULT_ASSIGNED_DRAFT_DAY_NUMBER,
            "field_day_id": "weekend-harlow-h1-avimor-native-route-card-replacement",
            "p75_minutes": loop["p75_minutes"],
            "p90_minutes": loop["p90_minutes"],
            "p90_bound_minutes": DEFAULT_WEEKEND_P90_BOUND_MINUTES,
            "stress": round(loop["p90_minutes"] / DEFAULT_WEEKEND_P90_BOUND_MINUTES, 3),
            "drive_minutes": 60,
            "between_drive_minutes": 0,
            "loop_count": 1,
            "segment_summary": {
                "segment_count": loop["segment_count"],
                "segment_ids": loop["segment_ids"],
                "official_miles": loop["official_miles"],
            },
            "on_foot_miles": loop["on_foot_miles"],
        },
        "loops": [loop],
        "replaces_loop_ids": state["removed_loop_ids"],
        "replaces_route_labels": H1_REPLACE_ROUTE_LABELS,
        "empties_dates": ["2026-06-21", "2026-07-12"],
    }


def merged_prior_promotions(
    prior_payload: dict[str, Any] | None,
    *,
    existing_promotions: list[dict[str, Any]],
    removed_loop_ids: list[str],
) -> list[dict[str, Any]]:
    if not prior_payload:
        return []
    removed = {str(loop_id) for loop_id in removed_loop_ids}
    seen = {
        (str(promotion.get("loop_id") or ""), str(promotion.get("mode") or ""), str(promotion.get("route_card_candidate_id") or ""))
        for promotion in existing_promotions
    }
    merged = []
    for promotion in prior_payload.get("promotions") or []:
        if str(promotion.get("loop_id") or "") in removed:
            continue
        key = (
            str(promotion.get("loop_id") or ""),
            str(promotion.get("mode") or ""),
            str(promotion.get("route_card_candidate_id") or ""),
        )
        if key in seen:
            continue
        clone = copy.deepcopy(promotion)
        clone["rank"] = len(existing_promotions) + len(merged) + 1
        clone.setdefault("source_promotion_payload", display_path(DEFAULT_PRIOR_FIELD_DAY_PROMOTION_JSON))
        merged.append(clone)
        seen.add(key)
    return merged


def promotion_payload(
    h1_audit: dict[str, Any],
    h1_access: dict[str, Any],
    state: dict[str, Any],
    prior_promotion_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    component = h1_component(h1_audit)
    promotions = [
        {
            "rank": 1,
            "date": DEFAULT_ASSIGNED_DATE,
            "draft_day_number": DEFAULT_ASSIGNED_DRAFT_DAY_NUMBER,
            "loop_id": H1_LOOP_ID,
            "source": "harlow_h1_route_card_promotion",
            "source_candidate_id": H1_BUNDLE_ID,
            "route_card_candidate_id": H1_BUNDLE_ID,
            "label": H1_FIELD_MENU_LABEL,
            "trailhead": H1_TRAILHEAD,
            "official_miles": component["official_miles"],
            "on_foot_miles": component["on_foot_miles"],
            "p75_minutes": component["time_estimates_minutes"]["door_to_door_p75"],
            "p90_minutes": component["time_estimates_minutes"]["door_to_door_p90"],
            "mode": "cluster_bundle_route_card_replacement",
            "replaces_route_labels": H1_REPLACE_ROUTE_LABELS,
            "replaces_loop_ids": state["removed_loop_ids"],
        }
    ]
    for index, old in enumerate(state["removed_components"], start=2):
        promotions.append(
            {
                "rank": index,
                "date": None,
                "loop_id": old.get("field_menu_group_id") or old.get("source_loop_id"),
                "source": old.get("source"),
                "source_candidate_id": old.get("candidate_id"),
                "route_card_candidate_id": H1_BUNDLE_ID,
                "label": old.get("field_menu_label"),
                "trailhead": old.get("trailhead"),
                "official_miles": old.get("official_miles"),
                "on_foot_miles": old.get("on_foot_miles"),
                "p75_minutes": old.get("total_minutes"),
                "p90_minutes": (old.get("time_estimates_minutes") or {}).get("door_to_door_p90"),
                "mode": "removed_source_loop_after_cluster_bundle_replacement",
                "skipped_route_card_source": True,
                "reassigned_to_candidate_id": H1_BUNDLE_ID,
                "reasons": ["h1_claims_exact_removed_segment_union", "h1_cluster_bundle_promoted"],
            }
        )
    prior_promotions = merged_prior_promotions(
        prior_promotion_payload,
        existing_promotions=promotions,
        removed_loop_ids=state["removed_loop_ids"],
    )
    promotions.extend(prior_promotions)
    diff = h1_access.get("h1_replacement_segment_set_diff") or {}
    return {
        "schema": "boise_trails_harlow_h1_route_card_promotion_v1",
        "generated_at": now_iso(),
        "objective": "Promote repaired H1 as the all-or-nothing Harlow/Avimor active route-card replacement.",
        "decision": "promoted_to_canonical_route_card_source_pending_recertification",
        "candidate_id": H1_BUNDLE_ID,
        "assigned_date": DEFAULT_ASSIGNED_DATE,
        "assigned_draft_day_number": DEFAULT_ASSIGNED_DRAFT_DAY_NUMBER,
        "summary": {
            "old_route_card_count": state["old_route_count"],
            "new_route_card_count": state["new_route_count"],
            "expected_active_route_cards_after_export": 44,
            "old_harlow_avimor_on_foot_miles": diff.get("old_on_foot_miles"),
            "new_h1_on_foot_miles": diff.get("new_on_foot_miles"),
            "saved_on_foot_miles": round_miles(float(diff.get("old_on_foot_miles") or 0) - float(diff.get("new_on_foot_miles") or 0)),
            "old_harlow_avimor_p75_minutes": diff.get("old_p75_minutes"),
            "new_h1_p75_minutes": diff.get("new_p75_minutes"),
            "saved_p75_minutes": int(diff.get("old_p75_minutes") or 0) - int(diff.get("new_p75_minutes") or 0),
            "old_harlow_avimor_p90_minutes": diff.get("old_p90_minutes"),
            "new_h1_p90_minutes": diff.get("new_p90_minutes"),
            "saved_p90_minutes": int(diff.get("old_p90_minutes") or 0) - int(diff.get("new_p90_minutes") or 0),
            "official_segment_coverage_after_source_promotion": len(state["h1_segment_ids"]),
            "prior_promotion_rows_preserved": len(prior_promotions),
        },
        "source_files": {
            "harlow_h1_gate_repair_audit": display_path(DEFAULT_H1_AUDIT_JSON),
            "harlow_h1_access_cue_review": display_path(DEFAULT_H1_ACCESS_REVIEW_JSON),
            "prior_field_day_loop_promotion": display_path(DEFAULT_PRIOR_FIELD_DAY_PROMOTION_JSON),
            "canonical_map_data": display_path(DEFAULT_OUTPUT_MAP_DATA_JSON),
        },
        "replacement_segment_set_diff": diff,
        "field_day_replacements": [field_day_replacement_payload(h1_audit, state)],
        "promotions": promotions,
        "promotion_assertions": promotion_assertions(h1_audit, h1_access, state),
        "before_after_field_day_diff": before_after_field_day_diff(h1_audit, state),
        "remaining_gates": [
            "regenerate_mobile_field_packet",
            "regenerate_field_day_layer_from_new_route_card",
            "rerun_mobile_field_packet_with_updated_field_day_layer",
            "run_repeat_progress_recertification_completion_walkthrough_and_pytest_gates",
        ],
    }


def promotion_assertions(h1_audit: dict[str, Any], h1_access: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    repaired = h1_audit["repaired_candidate"]
    repeat = h1_audit["route_repeat_optimization_audit_for_h1"]["candidate_specific_repeat_accounting"]
    route_cue = h1_route_cue({"feature_collections": read_json(DEFAULT_BASE_MAP_DATA_JSON)["feature_collections"]}, h1_audit)
    cue_text = json.dumps(route_cue, sort_keys=True)
    component = h1_component(h1_audit)
    checks = [
        ("old_route_labels_removed_from_source", not (set(route_labels(read_json(DEFAULT_BASE_MAP_DATA_JSON))) & set(H1_REPLACE_ROUTE_LABELS))),
        ("expected_source_route_count_44", state["new_route_count"] == 44),
        ("h1_claimed_segment_set_equals_removed_union", state["removed_segment_ids"] == state["h1_segment_ids"]),
        ("h1_p90_fits_assigned_weekend_bound", component["time_estimates_minutes"]["door_to_door_p90"] <= DEFAULT_WEEKEND_P90_BOUND_MINUTES),
        ("h1_has_no_direct_gap_fallback", float(repaired.get("direct_gap_fallback_miles") or 0) == 0),
        ("h1_has_no_hidden_self_repeat", not repeat.get("hidden_self_repeat_ids")),
        ("h1_repeat_mileage_priced", float(repaired.get("official_repeat_miles") or 0) > 0 and not repeat.get("unpriced_repeat_ids")),
        ("h1_parking_metadata_present", bool((repaired.get("parking") or {}).get("parking_confidence") and (repaired.get("parking") or {}).get("source"))),
        ("h1_access_cue_gate_cleared", (h1_access.get("promotion_readiness") or {}).get("status") == "access_gate_clear_keep_unpromoted"),
        ("h1_phone_cues_use_named_features_not_osm_ids", "OSM " not in cue_text and "connector 22098" not in cue_text),
    ]
    return [{"assertion": name, "passed": bool(passed)} for name, passed in checks]


def before_after_field_day_diff(h1_audit: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    component = h1_component(h1_audit)
    return [
        {
            "date": "2026-06-21",
            "before_labels": ["FD24A"],
            "after_labels": [],
            "after_status": "reusable_empty_field_day",
            "reason": "FD24A segment ownership moves into H1 on 2026-07-04.",
        },
        {
            "date": DEFAULT_ASSIGNED_DATE,
            "before_labels": ["FD27A", "FD27B", "FD27C"],
            "after_labels": [H1_FIELD_MENU_LABEL],
            "after_p75_minutes": component["time_estimates_minutes"]["door_to_door_p75"],
            "after_p90_minutes": component["time_estimates_minutes"]["door_to_door_p90"],
            "p90_bound_minutes": DEFAULT_WEEKEND_P90_BOUND_MINUTES,
            "after_status": "executable_route_card",
            "reason": "H1 is assigned to the existing Avimor-heavy weekend slot.",
        },
        {
            "date": "2026-07-12",
            "before_labels": ["FD30A"],
            "after_labels": [],
            "after_status": "reusable_empty_field_day",
            "reason": "FD30A segment ownership moves into H1 on 2026-07-04.",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Harlow / Avimor H1 Route-Card Promotion",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Summary",
        "",
        f"- Active route cards after source promotion: {summary['old_route_card_count']} -> {summary['new_route_card_count']} (expected field packet: {summary['expected_active_route_cards_after_export']})",
        f"- Harlow/Avimor on-foot: {summary['old_harlow_avimor_on_foot_miles']} -> {summary['new_h1_on_foot_miles']} mi",
        f"- Harlow/Avimor p75: {summary['old_harlow_avimor_p75_minutes']} -> {summary['new_h1_p75_minutes']} min",
        f"- Harlow/Avimor p90: {summary['old_harlow_avimor_p90_minutes']} -> {summary['new_h1_p90_minutes']} min",
        f"- Assigned date: {report['assigned_date']} (weekend p90 bound {DEFAULT_WEEKEND_P90_BOUND_MINUTES} min)",
        "",
        "## Segment Set",
        "",
        f"- Claimed ids: `{', '.join(report['replacement_segment_set_diff'].get('claimed_ids') or [])}`",
        f"- Missing ids: `{', '.join(report['replacement_segment_set_diff'].get('missing_ids') or []) or 'none'}`",
        f"- Extra ids: `{', '.join(report['replacement_segment_set_diff'].get('extra_ids') or []) or 'none'}`",
        "",
        "## Field-Day Diff",
        "",
        "| Date | Before | After | Status |",
        "|---|---|---|---|",
    ]
    for row in report["before_after_field_day_diff"]:
        lines.append(
            f"| {row['date']} | {', '.join(row['before_labels']) or '-'} | {', '.join(row['after_labels']) or '-'} | {row['after_status']} |"
        )
    lines.extend(["", "## Promotion Assertions", "", "| Assertion | Status |", "|---|---|"])
    for assertion in report["promotion_assertions"]:
        lines.append(f"| `{assertion['assertion']}` | {'pass' if assertion['passed'] else 'FAIL'} |")
    lines.extend(["", "## Remaining Gates", ""])
    lines.extend(f"- `{gate}`" for gate in report["remaining_gates"])
    lines.append("")
    return "\n".join(lines)


def write_map_views(map_data: dict[str, Any], map_html_path: Path, menu_md_path: Path) -> None:
    map_html_path.parent.mkdir(parents=True, exist_ok=True)
    menu_md_path.parent.mkdir(parents=True, exist_ok=True)
    map_html_path.write_text(render_package_map_html(map_data), encoding="utf-8")
    menu_md_path.write_text(render_outing_menu_markdown(map_data, map_html_path), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-map-data-json", type=Path, default=DEFAULT_BASE_MAP_DATA_JSON)
    parser.add_argument("--output-map-data-json", type=Path, default=DEFAULT_OUTPUT_MAP_DATA_JSON)
    parser.add_argument("--output-map-html", type=Path, default=DEFAULT_OUTPUT_MAP_HTML)
    parser.add_argument("--output-menu-md", type=Path, default=DEFAULT_OUTPUT_MENU_MD)
    parser.add_argument("--h1-audit-json", type=Path, default=DEFAULT_H1_AUDIT_JSON)
    parser.add_argument("--h1-access-review-json", type=Path, default=DEFAULT_H1_ACCESS_REVIEW_JSON)
    parser.add_argument("--prior-field-day-promotion-json", type=Path, default=DEFAULT_PRIOR_FIELD_DAY_PROMOTION_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_map_data = read_json(args.base_map_data_json)
    h1_audit = read_json(args.h1_audit_json)
    h1_access = read_json(args.h1_access_review_json)
    prior_promotion_payload = read_json(args.prior_field_day_promotion_json) if args.prior_field_day_promotion_json.exists() else None
    h1_gpx_path = REPO_ROOT / h1_audit["repaired_candidate"]["gpx_path"]
    official_segments, _meta = load_official_segments(args.official_geojson)
    promoted, state = promote_map_data(
        base_map_data=base_map_data,
        h1_audit=h1_audit,
        h1_access=h1_access,
        official_segments=official_segments,
        h1_gpx_path=h1_gpx_path,
    )
    write_json(args.output_map_data_json, promoted)
    write_map_views(promoted, args.output_map_html, args.output_menu_md)
    report = promotion_payload(h1_audit, h1_access, state, prior_promotion_payload)
    if not all(row["passed"] for row in report["promotion_assertions"]):
        failed = [row["assertion"] for row in report["promotion_assertions"] if not row["passed"]]
        raise ValueError(f"H1 source promotion assertions failed: {failed}")
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="harlow_h1_route_card_promotion",
        inputs=[
            args.base_map_data_json,
            args.h1_audit_json,
            args.h1_access_review_json,
            args.prior_field_day_promotion_json,
            args.official_geojson,
            h1_gpx_path,
        ],
        outputs=[args.output_map_data_json, args.output_map_html, args.output_menu_md, args.output_json, args.output_md],
        command="python years/2026/scripts/promote_harlow_h1_route_card.py",
        metadata={
            "schema": report["schema"],
            "new_route_card_count": report["summary"]["new_route_card_count"],
            "saved_on_foot_miles": report["summary"]["saved_on_foot_miles"],
        },
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_map_data_json)}")
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
