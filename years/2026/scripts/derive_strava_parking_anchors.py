#!/usr/bin/env python3
"""Derive private reusable parking/start anchors from Strava activity endpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_STRAVA_PULL_DIR = YEAR_DIR / "inputs" / "strava" / "api-pulls" / "2026-05-03"
DEFAULT_STATE_PATH = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_GEOJSON = YEAR_DIR / "inputs" / "personal" / "private" / "strava-parking-anchors-v1.geojson"
DEFAULT_OUTPUT_SUMMARY = YEAR_DIR / "derived" / "parking" / "strava-parking-anchors-summary-2026-05-06.json"

ON_FOOT_TYPES = {"Run", "TrailRun", "Hike", "Walk"}
MILES_PER_METER = 0.000621371


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


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    radius_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def parse_local_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def challenge_windows_from_pull_summary(pull_summary: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    raw = (((pull_summary.get("detail_selection") or {}).get("challenge_windows")) or {})
    for label, payload in sorted(raw.items()):
        start = payload.get("start")
        end = payload.get("end")
        if not start or not end:
            continue
        windows.append(
            {
                "label": str(label),
                "start": date.fromisoformat(str(start)),
                "end": date.fromisoformat(str(end)),
            }
        )
    return windows


def activity_window(activity: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    local = activity.get("start_date_local")
    if not local:
        return None
    activity_date = parse_local_date(str(local))
    for window in windows:
        if window["start"] <= activity_date <= window["end"]:
            return window
    return None


def home_point_from_state(state: dict[str, Any]) -> tuple[float, float] | None:
    drive_model = state.get("drive_model") or {}
    lat = drive_model.get("origin_lat")
    lon = drive_model.get("origin_lon")
    if lat is None or lon is None:
        return None
    return (float(lon), float(lat))


def endpoint_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    lat, lon = value[:2]
    if lat is None or lon is None:
        return None
    if float(lat) == 0 and float(lon) == 0:
        return None
    return (float(lon), float(lat))


def candidate_endpoint_records(
    activities: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    *,
    home_point: tuple[float, float] | None = None,
    exclude_home_radius_miles: float = 0.25,
    min_activity_miles: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    counters: dict[str, int] = defaultdict(int)
    for activity in activities:
        counters["activities_seen"] += 1
        sport = activity.get("sport_type") or activity.get("type")
        if sport not in ON_FOOT_TYPES:
            counters["not_on_foot"] += 1
            continue
        window = activity_window(activity, windows)
        if not window:
            counters["outside_challenge_windows"] += 1
            continue
        distance_miles = float(activity.get("distance") or 0.0) * MILES_PER_METER
        if distance_miles < min_activity_miles:
            counters["too_short"] += 1
            continue
        for endpoint_kind in ["start", "end"]:
            point = endpoint_point(activity.get(f"{endpoint_kind}_latlng"))
            if not point:
                counters[f"missing_{endpoint_kind}_latlng"] += 1
                continue
            if home_point and haversine_miles(point, home_point) <= exclude_home_radius_miles:
                counters["excluded_home_proximate_endpoint"] += 1
                continue
            records.append(
                {
                    "activity_id": int(activity["id"]),
                    "activity_name": activity.get("name"),
                    "activity_date": str(parse_local_date(str(activity["start_date_local"]))),
                    "window_label": window["label"],
                    "endpoint_kind": endpoint_kind,
                    "lon": point[0],
                    "lat": point[1],
                    "activity_distance_miles": round(distance_miles, 2),
                }
            )
    counters["candidate_endpoint_count"] = len(records)
    return records, dict(counters)


def cluster_endpoint_records(
    records: list[dict[str, Any]],
    cluster_radius_miles: float = 0.08,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for record in records:
        point = (float(record["lon"]), float(record["lat"]))
        best: tuple[int, float] | None = None
        for index, cluster in enumerate(clusters):
            distance = haversine_miles(point, (cluster["lon"], cluster["lat"]))
            if distance <= cluster_radius_miles and (best is None or distance < best[1]):
                best = (index, distance)
        if best is None:
            clusters.append(
                {
                    "lon": point[0],
                    "lat": point[1],
                    "records": [record],
                }
            )
            continue
        cluster = clusters[best[0]]
        cluster["records"].append(record)
        cluster["lon"] = sum(float(item["lon"]) for item in cluster["records"]) / len(cluster["records"])
        cluster["lat"] = sum(float(item["lat"]) for item in cluster["records"]) / len(cluster["records"])

    for index, cluster in enumerate(clusters, start=1):
        records_for_cluster = cluster["records"]
        activity_ids = sorted({int(item["activity_id"]) for item in records_for_cluster})
        dates = sorted({str(item["activity_date"]) for item in records_for_cluster})
        windows = sorted({str(item["window_label"]) for item in records_for_cluster})
        endpoint_counts = defaultdict(int)
        for item in records_for_cluster:
            endpoint_counts[str(item["endpoint_kind"])] += 1
        max_cluster_radius = max(
            haversine_miles((cluster["lon"], cluster["lat"]), (float(item["lon"]), float(item["lat"])))
            for item in records_for_cluster
        )
        cluster.update(
            {
                "anchor_id": f"strava-parking-anchor-{index:02d}",
                "name": f"Strava parking anchor {index:02d}",
                "evidence_endpoint_count": len(records_for_cluster),
                "evidence_activity_count": len(activity_ids),
                "activity_ids": activity_ids,
                "activity_dates": dates,
                "challenge_windows": windows,
                "endpoint_counts": dict(sorted(endpoint_counts.items())),
                "max_cluster_radius_miles": round(max_cluster_radius, 4),
            }
        )
    return sorted(
        clusters,
        key=lambda item: (-int(item["evidence_activity_count"]), item["name"]),
    )


def confidence_for_cluster(cluster: dict[str, Any]) -> str:
    activity_count = int(cluster["evidence_activity_count"])
    endpoint_count = int(cluster["evidence_endpoint_count"])
    if activity_count >= 3:
        return "strava_reused_prior_challenge_window"
    if activity_count >= 2 or endpoint_count >= 2:
        return "strava_seen_prior_challenge_window"
    return "strava_single_prior_challenge_window"


def clusters_to_geojson(clusters: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    features = []
    for rank, cluster in enumerate(clusters, start=1):
        confidence = confidence_for_cluster(cluster)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [cluster["lon"], cluster["lat"]]},
                "properties": {
                    "facility_id": cluster["anchor_id"],
                    "facility_name": cluster["name"],
                    "name": cluster["name"],
                    "rank": rank,
                    "source": "strava_activity_endpoint_cluster",
                    "has_parking": True,
                    "parking_confidence": confidence,
                    "parking_minutes": 8,
                    "evidence_endpoint_count": cluster["evidence_endpoint_count"],
                    "evidence_activity_count": cluster["evidence_activity_count"],
                    "activity_dates": cluster["activity_dates"],
                    "challenge_windows": cluster["challenge_windows"],
                    "endpoint_counts": cluster["endpoint_counts"],
                    "max_cluster_radius_miles": cluster["max_cluster_radius_miles"],
                    "privacy": "private_exact_coordinates",
                    "planning_note": (
                        "Derived from the user's prior challenge-window Strava start/end points; "
                        "use as a known practical parked start, not as a public facility claim."
                    ),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "Private Strava-derived parking anchors",
        "summary": summary,
        "features": features,
    }


def public_safe_summary(clusters: list[dict[str, Any]], summary: dict[str, Any], output_geojson: Path) -> dict[str, Any]:
    by_confidence = defaultdict(int)
    for cluster in clusters:
        by_confidence[confidence_for_cluster(cluster)] += 1
    return {
        "dataset": "strava-parking-anchors-v1",
        "private_geojson_path": display_path(output_geojson),
        "privacy": "Exact coordinates and activity ids are private and ignored by git.",
        "summary": summary,
        "anchor_count": len(clusters),
        "confidence_counts": dict(sorted(by_confidence.items())),
        "top_anchor_evidence_counts": [
            {
                "rank": index,
                "evidence_activity_count": cluster["evidence_activity_count"],
                "evidence_endpoint_count": cluster["evidence_endpoint_count"],
                "challenge_windows": cluster["challenge_windows"],
                "parking_confidence": confidence_for_cluster(cluster),
            }
            for index, cluster in enumerate(clusters[:12], start=1)
        ],
    }


def derive_parking_anchors(
    activities_summary_json: Path,
    pull_summary_json: Path,
    state_path: Path | None,
    *,
    cluster_radius_miles: float,
    exclude_home_radius_miles: float,
    min_activity_miles: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    activities = read_json(activities_summary_json)
    pull_summary = read_json(pull_summary_json)
    state = read_json(state_path) if state_path and state_path.exists() else {}
    windows = challenge_windows_from_pull_summary(pull_summary)
    home_point = home_point_from_state(state)
    records, counters = candidate_endpoint_records(
        activities,
        windows,
        home_point=home_point,
        exclude_home_radius_miles=exclude_home_radius_miles,
        min_activity_miles=min_activity_miles,
    )
    clusters = cluster_endpoint_records(records, cluster_radius_miles=cluster_radius_miles)
    summary = {
        "source_activities_summary_json": display_path(activities_summary_json),
        "source_pull_summary_json": display_path(pull_summary_json),
        "challenge_windows": [
            {"label": item["label"], "start": item["start"].isoformat(), "end": item["end"].isoformat()}
            for item in windows
        ],
        "filters": {
            "on_foot_types": sorted(ON_FOOT_TYPES),
            "cluster_radius_miles": cluster_radius_miles,
            "exclude_home_radius_miles": exclude_home_radius_miles,
            "min_activity_miles": min_activity_miles,
            "home_origin_used_for_exclusion": bool(home_point),
        },
        "counters": counters,
        "anchor_count": len(clusters),
    }
    return clusters_to_geojson(clusters, summary), public_safe_summary(clusters, summary, DEFAULT_OUTPUT_GEOJSON)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strava-pull-dir", type=Path, default=DEFAULT_STRAVA_PULL_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output-geojson", type=Path, default=DEFAULT_OUTPUT_GEOJSON)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--cluster-radius-miles", type=float, default=0.08)
    parser.add_argument("--exclude-home-radius-miles", type=float, default=0.25)
    parser.add_argument("--min-activity-miles", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geojson, summary = derive_parking_anchors(
        args.strava_pull_dir / "activities_summary.json",
        args.strava_pull_dir / "pull_summary.json",
        args.state,
        cluster_radius_miles=args.cluster_radius_miles,
        exclude_home_radius_miles=args.exclude_home_radius_miles,
        min_activity_miles=args.min_activity_miles,
    )
    write_json(args.output_geojson, geojson)
    summary["private_geojson_path"] = display_path(args.output_geojson)
    write_json(args.output_summary, summary)
    print(f"Wrote {args.output_geojson}")
    print(f"Wrote {args.output_summary}")
    print(f"Anchors: {len(geojson['features'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
