#!/usr/bin/env python3
"""Reconcile canonical outing-menu metrics from generated field-packet routes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from block_day_packager import render_html, render_outing_menu_markdown  # noqa: E402


DEFAULT_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTING_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def round_miles(value: float) -> float:
    return round(float(value or 0), 2)


def ratio(official_miles: float, on_foot_miles: float) -> float | None:
    if official_miles <= 0:
        return None
    return round(float(on_foot_miles) / float(official_miles), 2)


def route_records_by_candidate(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for route in field_tool_data.get("routes") or []:
        candidate_ids = [str(value) for value in route.get("candidate_ids") or []]
        if len(candidate_ids) != 1:
            continue
        records[candidate_ids[0]] = route
    return records


def update_component_from_route(component: dict[str, Any], route: dict[str, Any]) -> bool:
    changed = False
    field_map = {
        "official_miles": route.get("official_miles"),
        "on_foot_miles": route.get("on_foot_miles"),
        "total_minutes": route.get("door_to_door_minutes_p75"),
        "segment_ids": [
            int(segment_id) if str(segment_id).isdigit() else str(segment_id)
            for segment_id in route.get("segment_ids") or []
        ],
    }
    for key, value in field_map.items():
        if value is None:
            continue
        if key.endswith("miles"):
            value = round_miles(float(value))
        if component.get(key) != value:
            if key == "on_foot_miles" and component.get(key) is not None:
                component.setdefault("source_card_on_foot_miles", component.get(key))
            component[key] = value
            changed = True
    return changed


def update_route_feature_from_route(feature: dict[str, Any], route: dict[str, Any]) -> bool:
    props = feature.get("properties") or {}
    feature["properties"] = props
    changed = False
    for key, route_key in (
        ("official_miles", "official_miles"),
        ("on_foot_miles", "on_foot_miles"),
        ("total_minutes", "door_to_door_minutes_p75"),
    ):
        value = route.get(route_key)
        if value is None:
            continue
        if key.endswith("miles"):
            value = round_miles(float(value))
        if props.get(key) != value:
            if key == "on_foot_miles" and props.get(key) is not None:
                props.setdefault("source_card_on_foot_miles", props.get(key))
            props[key] = value
            changed = True
    return changed


def recompute_package(package: dict[str, Any]) -> None:
    components = package.get("components") or []
    official = sum(float(component.get("official_miles") or 0) for component in components)
    on_foot = sum(float(component.get("on_foot_miles") or 0) for component in components)
    package["component_route_count"] = len(components)
    package["official_miles"] = round_miles(official)
    package["on_foot_miles"] = round_miles(on_foot)
    package["ratio"] = ratio(official, on_foot)
    package["total_minutes_components"] = sum(int(round(float(component.get("total_minutes") or 0))) for component in components)
    package["component_routes_under_1_official_mile"] = sum(
        1 for component in components if float(component.get("official_miles") or 0) < 1
    )
    package["component_routes_under_2_official_miles"] = sum(
        1 for component in components if float(component.get("official_miles") or 0) < 2
    )


def recompute_summary(map_data: dict[str, Any]) -> None:
    packages = map_data.get("packages") or []
    summary = map_data.setdefault("summary", {})
    official = sum(float(package.get("official_miles") or 0) for package in packages)
    on_foot = sum(float(package.get("on_foot_miles") or 0) for package in packages)
    summary["package_count"] = len(packages)
    summary["component_route_count"] = sum(int(package.get("component_route_count") or 0) for package in packages)
    summary["official_miles"] = round_miles(official)
    summary["total_on_foot_miles"] = round_miles(on_foot)
    summary["planwide_on_foot_to_official_ratio"] = ratio(official, on_foot)
    summary["component_routes_under_1_official_mile"] = sum(
        int(package.get("component_routes_under_1_official_mile") or 0) for package in packages
    )
    summary["component_routes_under_2_official_miles"] = sum(
        int(package.get("component_routes_under_2_official_miles") or 0) for package in packages
    )


def reconcile_map_data(map_data: dict[str, Any], field_tool_data: dict[str, Any]) -> dict[str, Any]:
    routes_by_candidate = route_records_by_candidate(field_tool_data)
    changed_candidate_ids = []
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            candidate_id = str(component.get("candidate_id") or "")
            route = routes_by_candidate.get(candidate_id)
            if route and update_component_from_route(component, route):
                changed_candidate_ids.append(candidate_id)
        recompute_package(package)

    route_features = ((map_data.get("feature_collections") or {}).get("routes") or {}).get("features") or []
    for feature in route_features:
        candidate_id = str((feature.get("properties") or {}).get("candidate_id") or "")
        route = routes_by_candidate.get(candidate_id)
        if route:
            update_route_feature_from_route(feature, route)

    recompute_summary(map_data)
    map_data["field_packet_metric_reconciliation"] = {
        "schema": "boise_trails_field_packet_metric_reconciliation_v1",
        "source_field_tool_data": str(DEFAULT_FIELD_TOOL_DATA_JSON),
        "updated_candidate_count": len(sorted(set(changed_candidate_ids))),
        "updated_candidate_ids": sorted(set(changed_candidate_ids)),
    }
    return map_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--output-map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--output-map-html", type=Path, default=DEFAULT_MAP_HTML)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTING_MENU_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_data = reconcile_map_data(read_json(args.map_data_json), read_json(args.field_tool_data_json))
    write_json(args.output_map_data_json, map_data)
    args.output_map_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_map_html.write_text(render_html(map_data), encoding="utf-8")
    args.output_md.write_text(render_outing_menu_markdown(map_data, args.output_map_html), encoding="utf-8")
    reconciliation = map_data.get("field_packet_metric_reconciliation") or {}
    print(
        "Updated "
        f"{reconciliation.get('updated_candidate_count', 0)} route component(s) from "
        f"{args.field_tool_data_json}"
    )
    print(f"Wrote {args.output_map_data_json}")
    print(f"Wrote {args.output_map_html}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
