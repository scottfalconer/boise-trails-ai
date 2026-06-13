#!/usr/bin/env python3
"""Apply official trail-list drift repairs to current route source artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from block_day_packager import render_html, render_outing_menu_markdown  # noqa: E402
from human_loop_plan import sync_official_segment_features  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments, round_miles  # noqa: E402


DEFAULT_PRIVATE_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_PRIVATE_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_PRIVATE_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_FIELD_MENU_REPLACEMENTS_JSON = (
    YEAR_DIR / "inputs" / "personal" / "private" / "2026-field-menu-replacements-v2-multi-start.private.json"
)
DEFAULT_MANUAL_DESIGN_JSON = YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json"
DEFAULT_COMPARE_OFFICIAL_GEOJSON = (
    YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
)
DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON = YEAR_DIR / "inputs" / "open-data" / "special-management-rules-2026.json"
DEFAULT_REPORT_JSON = YEAR_DIR / "checkpoints" / "official-data-drift-repair-2026-06-13.json"
DEFAULT_REPORT_MD = YEAR_DIR / "checkpoints" / "official-data-drift-repair-2026-06-13.md"

# June 2026 official update: old 1664 geometry is now official segment 1762;
# old 1663 is no longer in the official on-foot list.
DEFAULT_SEGMENT_ID_REMAP = {"1664": "1762"}
DEFAULT_REMOVED_SEGMENT_IDS = {"1663"}

ID_LIST_KEYS = {
    "segment_ids",
    "required_segment_ids",
    "remaining_segment_ids",
    "official_repeat_segment_ids",
    "claimed_ids",
    "new_segment_ids",
    "completed_segment_ids",
    "blocked_segment_ids",
    "extra_completed_segment_ids",
    "missed_segment_ids",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def preserve_type(new_id: str, old_value: Any) -> Any:
    if isinstance(old_value, int) and new_id.isdigit():
        return int(new_id)
    return new_id


def normalize_id(value: Any, id_remap: dict[str, str], removed_ids: set[str]) -> Any | None:
    value_id = str(value)
    if value_id in removed_ids:
        return None
    new_id = id_remap.get(value_id, value_id)
    return preserve_type(new_id, value)


def repair_id_list(values: list[Any], id_remap: dict[str, str], removed_ids: set[str]) -> tuple[list[Any], bool]:
    repaired = []
    seen = set()
    changed = False
    for value in values:
        normalized = normalize_id(value, id_remap, removed_ids)
        if normalized is None:
            changed = True
            continue
        key = str(normalized)
        if key in seen:
            changed = True
            continue
        seen.add(key)
        repaired.append(normalized)
        if key != str(value):
            changed = True
    if len(repaired) != len(values):
        changed = True
    return repaired, changed


def segment_row_from_official(segment_id: str, official_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    official = official_by_id[segment_id]
    direction = str(official.get("direction") or "both")
    row: dict[str, Any] = {
        "seg_id": int(segment_id) if segment_id.isdigit() else segment_id,
        "seg_name": official.get("seg_name"),
        "trail_name": official.get("trail_name"),
        "official_miles": round_miles(official.get("official_miles")),
        "direction": direction,
        "direction_rule": direction,
        "start": list(official.get("start") or []),
        "end": list(official.get("end") or []),
        "coordinates": [list(coord) for coord in official.get("coordinates") or []],
    }
    if direction == "ascent":
        row["direction_cue"] = "Ascent-only segment; verify signed uphill direction."
    else:
        row["direction_cue"] = "Either direction allowed; follow map arrows."
    return row


def repair_segment_rows(
    segments: list[Any],
    official_by_id: dict[str, dict[str, Any]],
    id_remap: dict[str, str],
    removed_ids: set[str],
) -> tuple[list[Any], bool]:
    repaired = []
    changed = False
    seen = set()
    for segment in segments:
        if not isinstance(segment, dict) or "seg_id" not in segment:
            repaired.append(segment)
            continue
        old_id = str(segment.get("seg_id"))
        new_id = id_remap.get(old_id, old_id)
        if old_id in removed_ids:
            changed = True
            continue
        if new_id in seen:
            changed = True
            continue
        seen.add(new_id)
        if new_id in official_by_id:
            synced = segment_row_from_official(new_id, official_by_id)
            # Preserve fields that are not official metadata.
            for key, value in segment.items():
                if key not in synced:
                    synced[key] = value
            repaired.append(synced)
            if synced != segment:
                changed = True
        else:
            repaired.append(segment)
    return repaired, changed


def repair_feature_list(
    features: list[Any],
    official_by_id: dict[str, dict[str, Any]],
    id_remap: dict[str, str],
    removed_ids: set[str],
) -> tuple[list[Any], bool]:
    repaired = []
    changed = False
    for feature in features:
        if not isinstance(feature, dict):
            repaired.append(feature)
            continue
        props = feature.get("properties") or {}
        segment_id = props.get("seg_id") or props.get("segment_id") or props.get("segId")
        if segment_id is not None:
            old_id = str(segment_id)
            if old_id in removed_ids:
                changed = True
                continue
            new_id = id_remap.get(old_id, old_id)
            if new_id != old_id:
                feature = copy.deepcopy(feature)
                props = feature.setdefault("properties", {})
                if "seg_id" in props:
                    props["seg_id"] = preserve_type(new_id, props["seg_id"])
                if "segment_id" in props:
                    props["segment_id"] = preserve_type(new_id, props["segment_id"])
                if "segId" in props:
                    props["segId"] = preserve_type(new_id, props["segId"])
                changed = True
            if new_id in official_by_id and props.get("kind") == "official_segment":
                official = official_by_id[new_id]
                feature = copy.deepcopy(feature)
                feature["geometry"] = {
                    "type": "LineString",
                    "coordinates": [list(coord) for coord in official.get("coordinates") or []],
                }
                props = feature.setdefault("properties", {})
                props.update(
                    {
                        "seg_id": int(new_id) if new_id.isdigit() else new_id,
                        "seg_name": official.get("seg_name"),
                        "trail_name": official.get("trail_name"),
                        "official_miles": round_miles(official.get("official_miles")),
                        "direction": official.get("direction"),
                    }
                )
                changed = True
        repaired.append(feature)
    return repaired, changed


def repair_connector_edges(
    edges: list[Any],
    id_remap: dict[str, str],
    removed_ids: set[str],
) -> tuple[list[Any], bool]:
    repaired = []
    changed = False
    for edge in edges:
        if not isinstance(edge, dict):
            repaired.append(edge)
            continue
        segment_id = edge.get("seg_id")
        if segment_id is None:
            repaired.append(edge)
            continue
        old_id = str(segment_id)
        edge = copy.deepcopy(edge)
        if old_id in removed_ids:
            edge["seg_id"] = None
            edge["edge_type"] = "connector"
            edge["connector_class"] = "retired_official_connector"
            edge["source_before_official_drift_repair"] = edge.get("source")
            edge["source"] = "retired_official_challenge_geometry"
            edge["official_traversal_direction"] = None
            edge["official_list_status"] = "removed_in_2026_06_13_pull"
            changed = True
        else:
            new_id = id_remap.get(old_id, old_id)
            if new_id != old_id:
                edge["seg_id"] = preserve_type(new_id, segment_id)
                changed = True
        repaired.append(edge)
    return repaired, changed


def edge_miles(edge: dict[str, Any]) -> float:
    try:
        return float(edge.get("distance_miles") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def repair_connector_link_metrics(link: dict[str, Any]) -> bool:
    edges = [edge for edge in link.get("connector_edges") or [] if isinstance(edge, dict)]
    if not edges:
        return False
    repeat_edges = [
        edge
        for edge in edges
        if edge.get("seg_id") is not None
        and (edge.get("edge_type") == "official_repeat" or edge.get("connector_class") == "official_repeat")
    ]
    connector_edges = [edge for edge in edges if edge not in repeat_edges]
    official_repeat_ids = []
    seen_repeat_ids = set()
    for edge in repeat_edges:
        segment_id = str(edge.get("seg_id"))
        if segment_id in seen_repeat_ids:
            continue
        seen_repeat_ids.add(segment_id)
        official_repeat_ids.append(int(segment_id) if segment_id.isdigit() else segment_id)

    connector_names = []
    seen_names = set()
    for edge in connector_edges:
        name = edge.get("name")
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        connector_names.append(name)

    connector_classes = []
    seen_classes = set()
    for edge in edges:
        connector_class = edge.get("connector_class")
        if not connector_class or connector_class in seen_classes:
            continue
        seen_classes.add(connector_class)
        connector_classes.append(connector_class)

    before = {
        "connector_miles": link.get("connector_miles"),
        "official_repeat_miles": link.get("official_repeat_miles"),
        "official_repeat_segment_ids": link.get("official_repeat_segment_ids"),
        "connector_names": link.get("connector_names"),
        "connector_classes": link.get("connector_classes"),
    }
    link["connector_miles"] = round_miles(sum(edge_miles(edge) for edge in connector_edges))
    link["official_repeat_miles"] = round_miles(sum(edge_miles(edge) for edge in repeat_edges))
    link["official_repeat_segment_ids"] = official_repeat_ids
    link["connector_names"] = connector_names
    link["connector_classes"] = connector_classes
    after = {
        "connector_miles": link.get("connector_miles"),
        "official_repeat_miles": link.get("official_repeat_miles"),
        "official_repeat_segment_ids": link.get("official_repeat_segment_ids"),
        "connector_names": link.get("connector_names"),
        "connector_classes": link.get("connector_classes"),
    }
    return before != after


def repair_nested_ids(
    value: Any,
    official_by_id: dict[str, dict[str, Any]],
    id_remap: dict[str, str],
    removed_ids: set[str],
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        result = {}
        for key, child in value.items():
            if key in ID_LIST_KEYS and isinstance(child, list):
                result[key], child_changed = repair_id_list(child, id_remap, removed_ids)
            elif key == "segments" and isinstance(child, list):
                result[key], child_changed = repair_segment_rows(child, official_by_id, id_remap, removed_ids)
            elif key == "features" and isinstance(child, list):
                result[key], child_changed = repair_feature_list(child, official_by_id, id_remap, removed_ids)
            elif key == "connector_edges" and isinstance(child, list):
                result[key], child_changed = repair_connector_edges(child, id_remap, removed_ids)
            else:
                result[key], child_changed = repair_nested_ids(child, official_by_id, id_remap, removed_ids)
            changed = changed or child_changed
        if "connector_edges" in result:
            changed = repair_connector_link_metrics(result) or changed
        return result, changed
    if isinstance(value, list):
        changed = False
        result = []
        for child in value:
            repaired, child_changed = repair_nested_ids(child, official_by_id, id_remap, removed_ids)
            result.append(repaired)
            changed = changed or child_changed
        return result, changed
    return value, False


def official_miles_for_ids(segment_ids: list[Any], official_by_id: dict[str, dict[str, Any]]) -> float:
    return round_miles(sum(float(official_by_id[str(segment_id)]["official_miles"]) for segment_id in segment_ids if str(segment_id) in official_by_id))


def recompute_component(component: dict[str, Any], official_by_id: dict[str, dict[str, Any]]) -> None:
    segment_ids = component.get("segment_ids") or []
    official_miles = official_miles_for_ids(segment_ids, official_by_id)
    component["official_miles"] = official_miles
    on_foot = float(component.get("on_foot_miles") or 0.0)
    component["ratio"] = round(on_foot / official_miles, 2) if official_miles else None
    if "remaining_segment_count" in component:
        component["remaining_segment_count"] = len(segment_ids)


def recompute_package(package: dict[str, Any], official_by_id: dict[str, dict[str, Any]]) -> None:
    components = package.get("components") or []
    for component in components:
        if isinstance(component, dict):
            recompute_component(component, official_by_id)
    package["component_route_count"] = len(components)
    package["component_candidate_ids"] = [
        component.get("candidate_id") for component in components if component.get("candidate_id")
    ]
    package["official_miles"] = round_miles(sum(float(component.get("official_miles") or 0.0) for component in components))
    package["on_foot_miles"] = round_miles(sum(float(component.get("on_foot_miles") or 0.0) for component in components))
    package["ratio"] = (
        round(float(package["on_foot_miles"]) / float(package["official_miles"]), 2)
        if float(package["official_miles"] or 0.0)
        else None
    )
    package["component_routes_under_1_official_mile"] = [
        component.get("candidate_id")
        for component in components
        if component.get("candidate_id") and float(component.get("official_miles") or 0.0) < 1.0
    ]
    package["component_routes_under_2_official_miles"] = [
        component.get("candidate_id")
        for component in components
        if component.get("candidate_id") and float(component.get("official_miles") or 0.0) < 2.0
    ]


def recompute_route_cue(cue: dict[str, Any], official_by_id: dict[str, dict[str, Any]]) -> None:
    segment_ids = [segment.get("seg_id") for segment in cue.get("segments") or [] if isinstance(segment, dict)]
    cue["official_miles"] = official_miles_for_ids(segment_ids, official_by_id)


def recompute_map_data(map_data: dict[str, Any], official_by_id: dict[str, dict[str, Any]], official_segments: list[dict[str, Any]]) -> None:
    for package in map_data.get("packages") or []:
        recompute_package(package, official_by_id)
    for cue in (map_data.get("route_cues") or {}).values():
        if isinstance(cue, dict):
            recompute_route_cue(cue, official_by_id)
    packages = map_data.get("packages") or []
    unique_segment_ids = {
        str(segment_id)
        for package in packages
        for component in package.get("components") or []
        for segment_id in component.get("segment_ids") or []
    }
    summary = map_data.setdefault("summary", {})
    summary["package_count"] = len(packages)
    summary["component_route_count"] = sum(len(package.get("components") or []) for package in packages)
    summary["covered_segment_count"] = len(unique_segment_ids)
    summary["official_miles"] = round_miles(sum(float(package.get("official_miles") or 0.0) for package in packages))
    summary["total_on_foot_miles"] = round_miles(sum(float(package.get("on_foot_miles") or 0.0) for package in packages))
    summary["planwide_on_foot_to_official_ratio"] = (
        round(float(summary["total_on_foot_miles"]) / float(summary["official_miles"]), 2)
        if float(summary["official_miles"] or 0.0)
        else None
    )
    sync_official_segment_features(map_data, official_segments)


def recompute_replacement_source(data: dict[str, Any], official_by_id: dict[str, dict[str, Any]]) -> None:
    for override in data.get("overrides") or []:
        replace_package = override.get("replace_package")
        if isinstance(replace_package, dict):
            recompute_package(replace_package, official_by_id)
        for cue in (override.get("route_cues") or {}).values():
            if isinstance(cue, dict):
                recompute_route_cue(cue, official_by_id)


def repair_stack_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: repair_stack_text(child) for key, child in value.items()}
    if isinstance(value, list):
        return [repair_stack_text(child) for child in value]
    if not isinstance(value, str):
        return value
    text = value
    text = text.replace("official segments 1663 and 1664", "official segment 1762")
    text = text.replace("segments 1663 and 1664", "segment 1762")
    text = text.replace("1663 and 1664", "1762")
    text = text.replace("3.49 official mi", "1.0 official mi")
    text = text.replace("3.50 official mi", "1.0 official mi")
    return text


def repair_map_data(data: dict[str, Any], official_segments: list[dict[str, Any]], official_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    repaired, changed = repair_nested_ids(data, official_by_id, DEFAULT_SEGMENT_ID_REMAP, DEFAULT_REMOVED_SEGMENT_IDS)
    repaired = repair_stack_text(repaired)
    recompute_map_data(repaired, official_by_id, official_segments)
    return repaired, changed


def repair_replacement_source(data: dict[str, Any], official_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    repaired, changed = repair_nested_ids(data, official_by_id, DEFAULT_SEGMENT_ID_REMAP, DEFAULT_REMOVED_SEGMENT_IDS)
    repaired = repair_stack_text(repaired)
    recompute_replacement_source(repaired, official_by_id)
    return repaired, changed


def repair_manual_design(data: dict[str, Any], official_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    repaired, changed = repair_nested_ids(data, official_by_id, DEFAULT_SEGMENT_ID_REMAP, DEFAULT_REMOVED_SEGMENT_IDS)
    for area in repaired.get("areas") or []:
        for alternative in area.get("alternatives") or []:
            required_ids = alternative.get("required_segment_ids") or []
            if required_ids and alternative.get("target_official_miles") is not None:
                alternative["target_official_miles"] = official_miles_for_ids(required_ids, official_by_id)
    return repaired, changed


def load_official_feature_coordinates(path: Path) -> dict[str, list[list[float]]]:
    if not path.exists():
        return {}
    collection = read_json(path)
    by_id = {}
    for feature in collection.get("features") or []:
        props = feature.get("properties") or {}
        segment_id = props.get("seg_id") or props.get("segId") or props.get("segment_id")
        coords = ((feature.get("geometry") or {}).get("coordinates") or [])
        if segment_id is not None and coords:
            by_id[str(segment_id)] = coords
    return by_id


def reversed_official_segment_ids(old_geojson: Path, new_geojson: Path) -> set[str]:
    old_by_id = load_official_feature_coordinates(old_geojson)
    new_by_id = load_official_feature_coordinates(new_geojson)
    return {
        segment_id
        for segment_id, old_coords in old_by_id.items()
        if segment_id in new_by_id and old_coords == list(reversed(new_by_id[segment_id]))
    }


def flipped_direction(direction: Any) -> Any:
    if direction == "forward":
        return "reverse"
    if direction == "reverse":
        return "forward"
    return direction


def repair_special_management_rules(data: dict[str, Any], reversed_segment_ids: set[str]) -> tuple[dict[str, Any], bool]:
    repaired = copy.deepcopy(data)
    changed = False
    for rule in repaired.get("rules") or []:
        if rule.get("rule_type") != "directional_segment_traversal":
            continue
        overrides = rule.get("segment_direction_overrides") or {}
        for segment_id in sorted(reversed_segment_ids):
            if segment_id not in overrides:
                continue
            old_allowed = list(overrides[segment_id])
            new_allowed = [flipped_direction(direction) for direction in old_allowed]
            if new_allowed != old_allowed:
                overrides[segment_id] = new_allowed
                changed = True
    return repaired, changed


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Official Data Drift Repair - 2026-06-13",
        "",
        f"- Official source: `{report['official_geojson']}`",
        f"- ID remap: `{report['segment_id_remap']}`",
        f"- Removed segment ids: `{report['removed_segment_ids']}`",
        f"- Reversed official geometry ids: `{report['reversed_official_segment_ids']}`",
        "",
        "## Files",
        "",
    ]
    for row in report["files"]:
        lines.append(f"- `{row['path']}`: {'changed' if row['changed'] else 'unchanged'}")
    lines.extend(
        [
            "",
            "## Route Impact",
            "",
            "- `16C-1` now claims official segment `1762` instead of removed `1663` / old `1664`.",
            "- Sweet Connie `1667` official mileage is synchronized to the latest official geometry.",
            "- Special-management direction overrides were flipped for exact official geometry reversals.",
            "- Private map/menu artifacts were re-rendered from the repaired canonical source.",
            "",
            "The downstream public map/menu and phone packet still need regeneration and certification.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--private-map-data-json", type=Path, default=DEFAULT_PRIVATE_MAP_DATA_JSON)
    parser.add_argument("--private-map-html", type=Path, default=DEFAULT_PRIVATE_MAP_HTML)
    parser.add_argument("--private-menu-md", type=Path, default=DEFAULT_PRIVATE_MENU_MD)
    parser.add_argument("--field-menu-replacements-json", type=Path, default=DEFAULT_FIELD_MENU_REPLACEMENTS_JSON)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGN_JSON)
    parser.add_argument("--compare-official-geojson", type=Path, default=DEFAULT_COMPARE_OFFICIAL_GEOJSON)
    parser.add_argument("--special-management-rules-json", type=Path, default=DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def repair_file(path: Path, repair_fn) -> dict[str, Any]:
    if not path.exists():
        return {"path": display_path(path), "changed": False, "missing": True}
    original = read_json(path)
    repaired, changed = repair_fn(original)
    if changed or repaired != original:
        write_json(path, repaired)
        changed = True
    return {"path": display_path(path), "changed": changed, "missing": False}


def main() -> int:
    args = parse_args()
    official_segments, _metadata = load_official_segments(args.official_geojson)
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    reversed_segment_ids = reversed_official_segment_ids(args.compare_official_geojson, args.official_geojson)

    files: list[dict[str, Any]] = []
    files.append(
        repair_file(
            args.special_management_rules_json,
            lambda data: repair_special_management_rules(data, reversed_segment_ids),
        )
    )
    files.append(
        repair_file(
            args.field_menu_replacements_json,
            lambda data: repair_replacement_source(data, official_by_id),
        )
    )
    files.append(
        repair_file(
            args.manual_design_json,
            lambda data: repair_manual_design(data, official_by_id),
        )
    )
    files.append(
        repair_file(
            args.private_map_data_json,
            lambda data: repair_map_data(data, official_segments, official_by_id),
        )
    )

    map_data = read_json(args.private_map_data_json)
    args.private_map_html.parent.mkdir(parents=True, exist_ok=True)
    args.private_map_html.write_text(render_html(map_data), encoding="utf-8")
    files.append({"path": display_path(args.private_map_html), "changed": True, "missing": False})
    args.private_menu_md.parent.mkdir(parents=True, exist_ok=True)
    args.private_menu_md.write_text(
        render_outing_menu_markdown(map_data, map_html_path=args.private_map_html),
        encoding="utf-8",
    )
    files.append({"path": display_path(args.private_menu_md), "changed": True, "missing": False})

    report = {
        "schema": "boise_trails_official_data_drift_repair_v1",
        "official_geojson": display_path(args.official_geojson),
        "compare_official_geojson": display_path(args.compare_official_geojson),
        "segment_id_remap": DEFAULT_SEGMENT_ID_REMAP,
        "removed_segment_ids": sorted(DEFAULT_REMOVED_SEGMENT_IDS),
        "reversed_official_segment_ids": sorted(reversed_segment_ids),
        "files": files,
    }
    write_json(args.report_json, report)
    args.report_md.write_text(render_report(report), encoding="utf-8")
    print(f"Wrote {display_path(args.report_json)}")
    print(f"Wrote {display_path(args.report_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
