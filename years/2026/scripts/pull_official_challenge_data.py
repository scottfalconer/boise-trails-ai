#!/usr/bin/env python3
"""Pull and diff current Boise Trails Challenge official trail data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]

DEFAULT_TRAILS_URL = "https://boisetrailschallenge.com/api/trails"
DEFAULT_LEADERBOARD_URL = "https://boisetrailschallenge.com/api/leaderboard"
DEFAULT_COMPARE_TO = YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04"
DEFAULT_FIELD_TOOL_DATA = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
FOOT_ACTIVITY_TYPES = {"both", "foot"}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def fetch_json(url: str, timeout: int = 30) -> Any:
    request = Request(url, headers={"User-Agent": "boise-trails-ai-planner/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def segment_id(feature: dict[str, Any]) -> str:
    return str((feature.get("properties") or {}).get("segId"))


def sort_id(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(text):012d}")
    except ValueError:
        return (1, text)


def length_miles(feature: dict[str, Any]) -> float:
    props = feature.get("properties") or {}
    return float(props.get("LengthFt") or 0.0) / 5280.0


def is_foot_activity(row: dict[str, Any]) -> bool:
    return row.get("activity_type") in FOOT_ACTIVITY_TYPES


def foot_segments(trails_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        feature
        for feature in trails_payload.get("trailSegments") or []
        if is_foot_activity(feature.get("properties") or {})
    ]


def foot_master_trails(trails_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [trail for trail in trails_payload.get("masterTrails") or [] if is_foot_activity(trail)]


def official_foot_segments_collection(trails_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "lastUpdatedUTC": trails_payload.get("lastUpdatedUTC"),
        "features": foot_segments(trails_payload),
    }


def counter_dict(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def official_foot_summary(trails_payload: dict[str, Any]) -> dict[str, Any]:
    all_trails = trails_payload.get("masterTrails") or []
    all_segments = trails_payload.get("trailSegments") or []
    foot_trails = foot_master_trails(trails_payload)
    foot_features = foot_segments(trails_payload)
    bike_only_trails = [trail for trail in all_trails if trail.get("activity_type") == "bike"]
    bike_only_segments = [
        feature for feature in all_segments if (feature.get("properties") or {}).get("activity_type") == "bike"
    ]
    return {
        "official_foot_trails_count": len(foot_trails),
        "official_foot_segments_count": len(foot_features),
        "official_foot_distance_miles": round(sum(length_miles(feature) for feature in foot_features), 2),
        "official_foot_direction_counts": counter_dict(
            [(feature.get("properties") or {}).get("direction") for feature in foot_features]
        ),
        "all_trails_count": len(all_trails),
        "all_segments_count": len(all_segments),
        "all_distance_miles": round(sum(length_miles(feature) for feature in all_segments), 2),
        "bike_only_trails_count": len(bike_only_trails),
        "bike_only_segments_count": len(bike_only_segments),
        "bike_only_distance_miles": round(sum(length_miles(feature) for feature in bike_only_segments), 2),
        "trail_activity_counts": counter_dict([trail.get("activity_type") for trail in all_trails]),
        "segment_activity_counts": counter_dict(
            [(feature.get("properties") or {}).get("activity_type") for feature in all_segments]
        ),
        "trails_last_updated_utc": trails_payload.get("lastUpdatedUTC"),
    }


def build_pull_summary(
    trails_payload: dict[str, Any],
    challenge_metadata: dict[str, Any] | None,
    *,
    fetched_at_utc: str,
    trails_url: str,
    leaderboard_url: str | None,
) -> dict[str, Any]:
    return {
        "fetched_at_utc": fetched_at_utc,
        "trails_url": trails_url,
        "leaderboard_url": leaderboard_url,
        "trails_last_updated_utc": trails_payload.get("lastUpdatedUTC"),
        "master_trails_count": len(trails_payload.get("masterTrails") or []),
        "trail_segments_count": len(trails_payload.get("trailSegments") or []),
        "official_foot_summary": official_foot_summary(trails_payload),
        "challenge_metadata": challenge_metadata,
    }


def geometry_hash(feature: dict[str, Any]) -> str:
    raw = json.dumps(feature.get("geometry"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compact_segment(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties") or {}
    return {
        "segId": props.get("segId"),
        "segName": props.get("segName"),
        "LengthFt": props.get("LengthFt"),
        "direction": props.get("direction"),
        "specInst": props.get("specInst"),
        "activity_type": props.get("activity_type"),
    }


def feature_index_from_collection(collection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {segment_id(feature): feature for feature in collection.get("features") or []}


def property_changes(old_feature: dict[str, Any], new_feature: dict[str, Any]) -> dict[str, dict[str, Any]]:
    old_props = old_feature.get("properties") or {}
    new_props = new_feature.get("properties") or {}
    changes = {}
    for key in sorted(set(old_props) | set(new_props)):
        if old_props.get(key) != new_props.get(key):
            changes[key] = {"old": old_props.get(key), "new": new_props.get(key)}
    return changes


def route_impacts(field_tool_data: dict[str, Any] | None, ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    impacts: dict[str, list[dict[str, Any]]] = {seg_id: [] for seg_id in sorted(ids, key=sort_id)}
    if not field_tool_data:
        return impacts
    for route in field_tool_data.get("routes") or []:
        route_ids = {str(value) for value in route.get("segment_ids") or []}
        for seg_id in route_ids & ids:
            impacts.setdefault(seg_id, []).append(
                {
                    "label": route.get("label"),
                    "route_name": route.get("route_name"),
                    "field_readiness_status": route.get("field_readiness_status"),
                    "official_miles": route.get("official_miles"),
                    "on_foot_miles": route.get("on_foot_miles"),
                }
            )
    return impacts


def field_packet_segment_ids(field_tool_data: dict[str, Any] | None) -> set[str]:
    if not field_tool_data:
        return set()
    ids: set[str] = set()
    for route in field_tool_data.get("routes") or []:
        ids.update(str(value) for value in route.get("segment_ids") or [])
    return ids


def build_drift_report(
    old_collection: dict[str, Any],
    new_collection: dict[str, Any],
    *,
    old_pull_dir: Path,
    new_pull_dir: Path,
    field_tool_data: dict[str, Any] | None = None,
    generated_at_utc: str,
) -> dict[str, Any]:
    old_index = feature_index_from_collection(old_collection)
    new_index = feature_index_from_collection(new_collection)
    old_ids = set(old_index)
    new_ids = set(new_index)
    removed_ids = old_ids - new_ids
    added_ids = new_ids - old_ids
    changed_rows = []
    for seg_id in sorted(old_ids & new_ids, key=sort_id):
        old_feature = old_index[seg_id]
        new_feature = new_index[seg_id]
        props = property_changes(old_feature, new_feature)
        geometry_changed = geometry_hash(old_feature) != geometry_hash(new_feature)
        if props or geometry_changed:
            changed_rows.append(
                {
                    "segId": int(seg_id) if seg_id.isdigit() else seg_id,
                    "old": compact_segment(old_feature),
                    "new": compact_segment(new_feature),
                    "property_changes": props,
                    "geometry_changed": geometry_changed,
                    "old_geometry_sha256": geometry_hash(old_feature),
                    "new_geometry_sha256": geometry_hash(new_feature),
                }
            )

    impacted_ids = removed_ids | {str(row["segId"]) for row in changed_rows}
    active_ids = field_packet_segment_ids(field_tool_data)
    report = {
        "schema": "boise_trails_official_data_drift_v1",
        "generated_at_utc": generated_at_utc,
        "old_pull_dir": display_path(old_pull_dir),
        "new_pull_dir": display_path(new_pull_dir),
        "old_trails_last_updated_utc": old_collection.get("lastUpdatedUTC"),
        "new_trails_last_updated_utc": new_collection.get("lastUpdatedUTC"),
        "summary": {
            "old_foot_segments": len(old_ids),
            "new_foot_segments": len(new_ids),
            "delta_foot_segments": len(new_ids) - len(old_ids),
            "old_foot_miles": round(sum(length_miles(feature) for feature in old_index.values()), 2),
            "new_foot_miles": round(sum(length_miles(feature) for feature in new_index.values()), 2),
            "delta_foot_miles": round(
                sum(length_miles(feature) for feature in new_index.values())
                - sum(length_miles(feature) for feature in old_index.values()),
                2,
            ),
            "added_segment_count": len(added_ids),
            "removed_segment_count": len(removed_ids),
            "changed_common_segment_count": len(changed_rows),
        },
        "removed_foot_segments": [compact_segment(old_index[seg_id]) for seg_id in sorted(removed_ids, key=sort_id)],
        "added_foot_segments": [compact_segment(new_index[seg_id]) for seg_id in sorted(added_ids, key=sort_id)],
        "changed_foot_segments": changed_rows,
        "active_field_packet_impacts": {
            "field_tool_data_path": display_path(DEFAULT_FIELD_TOOL_DATA) if field_tool_data else None,
            "unique_claimed_segment_count": len(active_ids) if field_tool_data else None,
            "routes_claiming_removed_segments": route_impacts(field_tool_data, removed_ids),
            "routes_claiming_changed_segments": route_impacts(
                field_tool_data, {str(row["segId"]) for row in changed_rows}
            ),
            "new_official_segments_not_claimed_by_active_packet": sorted(new_ids - active_ids, key=sort_id)
            if field_tool_data
            else [],
            "old_claimed_segments_no_longer_official": sorted(active_ids - new_ids, key=sort_id)
            if field_tool_data
            else [],
        },
        "recommended_next_step": (
            "Treat this as a route-list change event: update the canonical official input path, "
            "repair affected route cards, regenerate the field packet, and run the field-packet "
            "certification chain before using the active menu."
        ),
    }
    return report


def render_drift_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Official Segment Drift Report - {report['generated_at_utc'][:10]}",
        "",
        "## Summary",
        "",
        f"- Previous pull: `{report['old_pull_dir']}` ({report['old_trails_last_updated_utc']})",
        f"- New pull: `{report['new_pull_dir']}` ({report['new_trails_last_updated_utc']})",
        f"- Foot segments: {summary['old_foot_segments']} -> {summary['new_foot_segments']} ({summary['delta_foot_segments']:+d})",
        f"- Foot distance: {summary['old_foot_miles']} mi -> {summary['new_foot_miles']} mi ({summary['delta_foot_miles']:+.2f} mi)",
        f"- Added / removed / changed common segments: {summary['added_segment_count']} / {summary['removed_segment_count']} / {summary['changed_common_segment_count']}",
        "",
        "## Removed Foot Segments",
        "",
    ]
    if report["removed_foot_segments"]:
        lines.extend(segment_table(report["removed_foot_segments"]))
    else:
        lines.append("None.")
    lines.extend(["", "## Added Foot Segments", ""])
    if report["added_foot_segments"]:
        lines.extend(segment_table(report["added_foot_segments"]))
    else:
        lines.append("None.")
    lines.extend(["", "## Changed Common Segments", ""])
    if report["changed_foot_segments"]:
        lines.append("| Segment | Name | Changes | Geometry |")
        lines.append("| --- | --- | --- | --- |")
        for row in report["changed_foot_segments"]:
            change_keys = ", ".join(row["property_changes"].keys()) or "properties unchanged"
            geom = "changed" if row["geometry_changed"] else "unchanged"
            lines.append(f"| {row['segId']} | {row['new'].get('segName')} | {change_keys} | {geom} |")
    else:
        lines.append("None.")

    impacts = report["active_field_packet_impacts"]
    lines.extend(["", "## Active Field Packet Impact", ""])
    old_stale = impacts.get("old_claimed_segments_no_longer_official") or []
    new_missing = impacts.get("new_official_segments_not_claimed_by_active_packet") or []
    lines.append(
        f"- Old claimed segments no longer official: {', '.join(old_stale) if old_stale else 'none'}"
    )
    lines.append(
        f"- New official segments not claimed by active packet: {', '.join(new_missing) if new_missing else 'none'}"
    )
    for title, key in [
        ("Routes Claiming Removed Segments", "routes_claiming_removed_segments"),
        ("Routes Claiming Changed Segments", "routes_claiming_changed_segments"),
    ]:
        lines.extend(["", f"### {title}", ""])
        impacts_by_segment = impacts.get(key) or {}
        wrote = False
        for seg_id, routes in impacts_by_segment.items():
            if not routes:
                continue
            wrote = True
            route_text = "; ".join(
                f"{route.get('label')} ({route.get('route_name')}, {route.get('field_readiness_status')})"
                for route in routes
            )
            lines.append(f"- `{seg_id}`: {route_text}")
        if not wrote:
            lines.append("None.")

    lines.extend(["", "## Next Step", "", report["recommended_next_step"], ""])
    return "\n".join(lines)


def segment_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Segment | Name | Miles | Direction | Activity |", "| --- | --- | ---: | --- | --- |"]
    for row in rows:
        miles = round(float(row.get("LengthFt") or 0.0) / 5280.0, 2)
        lines.append(
            f"| {row.get('segId')} | {row.get('segName')} | {miles:.2f} | {row.get('direction')} | {row.get('activity_type')} |"
        )
    return lines


def render_readme(pull_dir: Path, summary: dict[str, Any]) -> str:
    foot = summary["official_foot_summary"]
    challenge = summary.get("challenge_metadata") or {}
    direction_rules = ", ".join(
        f"{direction}: {count}" for direction, count in foot["official_foot_direction_counts"].items()
    )
    return "\n".join(
        [
            f"# Official 2026 Challenge API Pull - {pull_dir.name.removeprefix('api-pull-')}",
            "",
            f"Fetched at: {summary['fetched_at_utc']}",
            f"Trail data last changed: {summary['trails_last_updated_utc']}",
            "",
            "Public read-only endpoints used:",
            "",
            f"- `{summary['trails_url']}`",
            f"- `{summary['leaderboard_url']}` for challenge metadata only",
            "",
            "Files in this pull:",
            "",
            "- `trails.json` - raw public trail payload.",
            "- `challenge_metadata.json` - `ChallengeData[0]` only, without raw participant leaderboard rows.",
            "- `official_foot_segments.geojson` - foot/both segment FeatureCollection.",
            "- `official_foot_master_trails.json` - foot/both master trail list.",
            "- `official_foot_summary.json` - derived counts and distance summary.",
            "- `official_foot_drift_report.json` / `.md` - comparison to the prior official pull.",
            "",
            "Current on-foot challenge metrics from this pull:",
            "",
            f"- Official on-foot trails: {foot['official_foot_trails_count']}",
            f"- Official on-foot segments: {foot['official_foot_segments_count']}",
            f"- Official on-foot distance: {foot['official_foot_distance_miles']} miles",
            f"- Direction rules: {direction_rules}",
            f"- Challenge metadata foot segments: {challenge.get('num_trail_segs_foot')}",
            f"- Challenge metadata trail data change: {challenge.get('last_trail_data_change')}",
            "",
            "Raw leaderboard data is intentionally not saved here because it includes participant identifiers.",
            "",
        ]
    )


def load_optional_json(path: Path | None) -> Any | None:
    if not path:
        return None
    if not path.exists():
        return None
    return read_json(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-date", default=today_utc(), help="Date label for api-pull-YYYY-MM-DD output.")
    parser.add_argument("--output-root", type=Path, default=YEAR_DIR / "inputs" / "official")
    parser.add_argument("--trails-url", default=DEFAULT_TRAILS_URL)
    parser.add_argument("--leaderboard-url", default=DEFAULT_LEADERBOARD_URL)
    parser.add_argument("--trails-json", type=Path, help="Use an already fetched trails payload instead of network.")
    parser.add_argument("--leaderboard-json", type=Path, help="Use an already fetched leaderboard payload instead of network.")
    parser.add_argument("--compare-to", type=Path, default=DEFAULT_COMPARE_TO)
    parser.add_argument("--field-tool-data", type=Path, default=DEFAULT_FIELD_TOOL_DATA)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fetched_at = now_utc()
    trails_payload = read_json(args.trails_json) if args.trails_json else fetch_json(args.trails_url, args.timeout)
    leaderboard_payload = (
        read_json(args.leaderboard_json)
        if args.leaderboard_json
        else fetch_json(args.leaderboard_url, args.timeout)
    )
    challenge_rows = leaderboard_payload.get("ChallengeData") or []
    challenge_metadata = challenge_rows[0] if challenge_rows else None

    pull_dir = args.output_root / f"api-pull-{args.pull_date}"
    pull_dir.mkdir(parents=True, exist_ok=True)
    foot_collection = official_foot_segments_collection(trails_payload)
    summary = build_pull_summary(
        trails_payload,
        challenge_metadata,
        fetched_at_utc=fetched_at,
        trails_url=args.trails_url,
        leaderboard_url=args.leaderboard_url,
    )

    write_json(pull_dir / "trails.json", trails_payload)
    if challenge_metadata:
        write_json(pull_dir / "challenge_metadata.json", challenge_metadata)
    write_json(pull_dir / "official_foot_segments.geojson", foot_collection)
    write_json(pull_dir / "official_foot_master_trails.json", foot_master_trails(trails_payload))
    write_json(pull_dir / "official_foot_summary.json", summary["official_foot_summary"])
    write_json(pull_dir / "summary.json", summary)

    field_tool_data = load_optional_json(args.field_tool_data)
    old_collection_path = args.compare_to / "official_foot_segments.geojson"
    if old_collection_path.exists():
        drift_report = build_drift_report(
            read_json(old_collection_path),
            foot_collection,
            old_pull_dir=args.compare_to,
            new_pull_dir=pull_dir,
            field_tool_data=field_tool_data,
            generated_at_utc=fetched_at,
        )
        write_json(pull_dir / "official_foot_drift_report.json", drift_report)
        (pull_dir / "official_foot_drift_report.md").write_text(
            render_drift_markdown(drift_report), encoding="utf-8"
        )
    (pull_dir / "README.md").write_text(render_readme(pull_dir, summary), encoding="utf-8")

    print(f"Wrote official pull to {display_path(pull_dir)}")
    print(
        "Foot segments: "
        f"{summary['official_foot_summary']['official_foot_segments_count']} "
        f"({summary['official_foot_summary']['official_foot_distance_miles']} mi)"
    )
    if old_collection_path.exists():
        print(f"Drift report: {display_path(pull_dir / 'official_foot_drift_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
