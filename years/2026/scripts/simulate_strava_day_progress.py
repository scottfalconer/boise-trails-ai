#!/usr/bin/env python3
"""Replay prior Strava challenge-window days against the 2026 planner.

This is a simulation tool, not official completion evidence. It uses prior
Strava GPS geometry to infer which 2026 official segments would have been
knocked out, then reruns the 2026 route menu cumulatively after each simulated
progress day. It also snapshots the current outing-map state so we can see
which runnable outing cards disappear as progress accrues.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_day_packager import build_outing_menu  # noqa: E402
from human_loop_plan import (  # noqa: E402
    DEFAULT_MANUAL_DESIGN_JSON,
    DEFAULT_MANUAL_DESIGN_REPORT_JSON,
    merge_manual_design_report,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STATE_PATH,
    DEFAULT_STRAVA_DETAILS_DIR,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    METERS_PER_MILE,
    ON_FOOT_SPORTS,
    activity_geometry,
    build_direction_validation,
    build_plan,
    haversine_miles,
    load_dem_context,
    load_official_segments,
    load_state,
    match_activity_geometry_to_segments,
    read_json,
    round_miles,
    write_json,
)


DEFAULT_STRAVA_PULL_SUMMARY = (
    YEAR_DIR / "inputs" / "strava" / "api-pulls" / "2026-05-03" / "pull_summary.json"
)
DEFAULT_PACKAGE_MAP_JSON = (
    YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1-map-data.json"
)
DEFAULT_OUTPUT_DIR = YEAR_DIR / "experiments" / "2026-05-05-strava-two-year-simulation"
BASELINE_2025_OFFICIAL_MILES = 68.90


def parse_local_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value[:10])


def date_range(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def parse_source_years(value: str) -> list[int]:
    years = []
    for item in value.replace(" ", "").split(","):
        if item:
            years.append(int(item))
    return years


def challenge_windows_from_pull_summary(path: Path, source_years: list[int]) -> list[dict[str, Any]]:
    summary = read_json(path) if path.exists() else {}
    windows = ((summary.get("date_window") or {}).get("challenge_windows") or {})
    by_year: dict[int, dict[str, Any]] = {}
    for key, value in windows.items():
        if not isinstance(value, dict) or not value.get("start") or not value.get("end"):
            continue
        year = int(str(value["start"])[:4])
        by_year[year] = {
            "source_year": year,
            "window_name": key,
            "start": value["start"],
            "end": value["end"],
        }
    fallbacks = {
        2024: {"window_name": "2024_proxy_window", "start": "2024-06-19", "end": "2024-07-19"},
        2025: {"window_name": "2025_challenge_window", "start": "2025-06-19", "end": "2025-07-19"},
    }
    result = []
    for year in source_years:
        window = by_year.get(year) or {**fallbacks[year], "source_year": year}
        result.append(
            {
                **window,
                "start_date": date.fromisoformat(window["start"]),
                "end_date": date.fromisoformat(window["end"]),
            }
        )
    return result


def nearest_activity_index(
    point: tuple[float, float], activity_coords: list[tuple[float, float]]
) -> int | None:
    if not activity_coords:
        return None
    best_index = None
    best_distance = None
    for index, coord in enumerate(activity_coords):
        distance = haversine_miles(point, coord)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def ascent_direction_counts(
    segment: dict[str, Any],
    activity_coords: list[tuple[float, float]],
    elevation_sampler,
) -> dict[str, Any]:
    validation = build_direction_validation([segment], elevation_sampler=elevation_sampler)
    planned = validation["planned_traversal_direction"].get(str(segment["seg_id"]))
    start_index = nearest_activity_index(segment["start"], activity_coords)
    end_index = nearest_activity_index(segment["end"], activity_coords)
    if start_index is None or end_index is None or planned == "unknown":
        direction_ok = False
    elif planned == "official_geometry_start_to_end":
        direction_ok = start_index < end_index
    else:
        direction_ok = end_index < start_index
    return {
        "direction_ok": direction_ok,
        "planned_traversal_direction": planned,
        "activity_start_index": start_index,
        "activity_end_index": end_index,
        "direction_validation": validation,
    }


def match_activity(
    activity: dict[str, Any],
    official_segments: list[dict[str, Any]],
    segment_by_id: dict[int, dict[str, Any]],
    elevation_sampler,
    threshold_miles: float,
    min_fraction: float,
) -> dict[str, Any] | None:
    if activity.get("sport_type") not in ON_FOOT_SPORTS:
        return None
    local_date = parse_local_date(activity.get("start_date_local"))
    if not local_date:
        return None
    coords = activity_geometry(activity)
    try:
        distance_miles = float(activity.get("distance") or 0) / METERS_PER_MILE
    except (TypeError, ValueError):
        distance_miles = 0.0
    if len(coords) < 2:
        return {
            "activity_id": activity.get("id"),
            "activity_name": activity.get("name"),
            "date": str(local_date),
            "sport_type": activity.get("sport_type"),
            "distance_miles": round_miles(distance_miles),
            "matched_segments": [],
            "credited_segments": [],
            "skipped_reason": "missing_activity_geometry",
        }
    matches = match_activity_geometry_to_segments(
        coords,
        official_segments,
        threshold_miles=threshold_miles,
        min_fraction=min_fraction,
    )
    credited = []
    matched = []
    for match in matches:
        segment = segment_by_id[match["seg_id"]]
        direction_detail = None
        counts = segment["direction"] == "both"
        if segment["direction"] == "ascent":
            direction_detail = ascent_direction_counts(segment, coords, elevation_sampler)
            counts = direction_detail["direction_ok"]
        detail = {
            **match,
            "counts_for_simulation": counts,
            "ascent_direction_detail": direction_detail,
        }
        matched.append(detail)
        if counts:
            credited.append(segment["seg_id"])
    return {
        "activity_id": activity.get("id"),
        "activity_name": activity.get("name"),
        "date": str(local_date),
        "sport_type": activity.get("sport_type"),
        "distance_miles": round_miles(distance_miles),
        "moving_time_minutes": round((float(activity.get("moving_time") or 0) / 60), 1),
        "total_elevation_gain_ft": round(float(activity.get("total_elevation_gain") or 0) * 3.28084, 0),
        "matched_segments": matched,
        "credited_segments": sorted(set(credited)),
    }


def summarize_primary(plan: dict[str, Any], bucket_name: str) -> dict[str, Any] | None:
    candidate = plan["route_menu"]["primary_candidates_by_bucket"].get(bucket_name)
    if not candidate:
        return None
    return {
        "candidate_id": candidate["candidate_id"],
        "trail_names": candidate["trail_names"],
        "route_status": candidate["route_status"],
        "total_minutes": candidate["total_minutes"],
        "official_miles": candidate["official_new_miles"],
    }


def primary_label(value: dict[str, Any] | None) -> str:
    if not value:
        return "none"
    return (
        f"{', '.join(value['trail_names'])} "
        f"[{value['route_status']}] "
        f"({value['total_minutes']}m, {value['official_miles']}mi)"
    )


def primary_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "under_1_hour": summarize_primary(plan, "under_1_hour"),
        "one_to_two_hours": summarize_primary(plan, "one_to_two_hours"),
        "two_to_three_hours": summarize_primary(plan, "two_to_three_hours"),
        "three_to_four_hours": summarize_primary(plan, "three_to_four_hours"),
        "four_plus_hours": summarize_primary(plan, "four_plus_hours"),
    }


def load_package_map_with_manual_design(
    package_map_json: Path,
    manual_design_json: Path,
    manual_design_report_json: Path,
) -> dict[str, Any]:
    package_map = read_json(package_map_json)
    if manual_design_json.exists():
        manual_design = read_json(manual_design_json)
        report = read_json(manual_design_report_json) if manual_design_report_json.exists() else {}
        package_map["manual_design"] = merge_manual_design_report(manual_design, report)
    return package_map


def outing_state_for_completed(
    package_map: dict[str, Any],
    completed_segment_ids: set[int],
) -> dict[str, Any]:
    map_data = copy.deepcopy(package_map)
    map_data["progress"] = {
        **(map_data.get("progress") or {}),
        "completed_segment_ids": sorted(completed_segment_ids),
    }
    outings = build_outing_menu(map_data)
    runnable = [outing for outing in outings if not outing.get("manual_design_hold")]
    manual_holds = [outing for outing in outings if outing.get("manual_design_hold")]
    bucket_counts: dict[str, int] = defaultdict(int)
    for outing in runnable:
        bucket_counts[outing["time_bucket"]] += 1
    next_by_bucket = {}
    for bucket in ["2 hours or less", "2-3 hours", "3-4 hours", "4+ hours"]:
        bucket_outings = [outing for outing in runnable if outing["time_bucket"] == bucket]
        if bucket_outings:
            outing = bucket_outings[0]
            next_by_bucket[bucket] = {
                "label": outing["label"],
                "trailhead": outing["trailhead"],
                "block_name": outing["block_name"],
                "trails": outing["trails"],
                "total_minutes": outing["total_minutes"],
                "official_miles": outing["official_miles"],
                "on_foot_miles": outing["on_foot_miles"],
                "candidate_ids": outing["candidate_ids"],
            }
        else:
            next_by_bucket[bucket] = None
    return {
        "open_runnable_outing_count": len(runnable),
        "manual_hold_count": len(manual_holds),
        "time_bucket_counts": dict(bucket_counts),
        "open_candidate_ids": sorted({candidate_id for outing in runnable for candidate_id in outing["candidate_ids"]}),
        "open_outing_labels": [outing["label"] for outing in runnable],
        "next_by_time_bucket": next_by_bucket,
    }


def collect_activity_matches(
    strava_details_dir: Path,
    window: dict[str, Any],
    official_segments: list[dict[str, Any]],
    segment_by_id: dict[int, dict[str, Any]],
    elevation_sampler,
    threshold_miles: float,
    min_fraction: float,
) -> list[dict[str, Any]]:
    matches = []
    for path in sorted(strava_details_dir.glob("*.json")):
        activity = read_json(path)
        local_date = parse_local_date(activity.get("start_date_local"))
        if not local_date or local_date < window["start_date"] or local_date > window["end_date"]:
            continue
        match = match_activity(
            activity,
            official_segments,
            segment_by_id,
            elevation_sampler,
            threshold_miles=threshold_miles,
            min_fraction=min_fraction,
        )
        if match:
            matches.append(match)
    return matches


def build_progress_plan(
    base_state: dict[str, Any],
    completed_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    replay_state = dict(base_state)
    replay_state["completed_segment_ids"] = sorted(completed_ids)
    return build_plan(
        official_geojson=args.official,
        state=replay_state,
        strava_activity_details_dir=args.strava_details_dir,
        activity_summary_csv=args.activity_summary_csv,
        activity_detail_summary_csv=args.activity_detail_summary_csv,
        segment_perf_csv=args.segment_perf_csv,
        connector_geojson=args.connector_geojson,
        trailheads_geojson=args.trailheads_geojson,
        dem_tif=args.dem_tif,
        dem_summary_json=args.dem_summary_json,
    )


def simulate_window(
    window: dict[str, Any],
    args: argparse.Namespace,
    official_segments: list[dict[str, Any]],
    segment_by_id: dict[int, dict[str, Any]],
    elevation_sampler,
    package_map: dict[str, Any],
    base_state: dict[str, Any],
) -> dict[str, Any]:
    activity_matches = collect_activity_matches(
        args.strava_details_dir,
        window,
        official_segments,
        segment_by_id,
        elevation_sampler,
        args.threshold_miles,
        args.min_fraction,
    )
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in activity_matches:
        by_day[match["date"]].append(match)

    completed_ids: set[int] = set()
    plan = build_progress_plan(base_state, completed_ids, args)
    previous_primary = primary_snapshot(plan)
    initial_outing_state = outing_state_for_completed(package_map, completed_ids)
    previous_open_outings: set[str] = set(initial_outing_state["open_outing_labels"])
    days = []
    progress_day_count = 0
    target_start = date.fromisoformat(args.target_start_date)

    for source_date in date_range(window["start_date"], window["end_date"]):
        source_key = str(source_date)
        day_matches = by_day.get(source_key) or []
        day_ids = {
            seg_id
            for match in day_matches
            for seg_id in match.get("credited_segments", [])
        }
        new_ids = sorted(day_ids - completed_ids)
        if new_ids:
            progress_day_count += 1
            completed_ids.update(new_ids)
            plan = build_progress_plan(base_state, completed_ids, args)

        outing_state = outing_state_for_completed(package_map, completed_ids)
        primary = primary_snapshot(plan)
        open_outings = set(outing_state["open_outing_labels"])
        hidden_outings = sorted(previous_open_outings - open_outings)
        primary_changed = primary != previous_primary
        previous_primary = copy.deepcopy(primary)
        previous_open_outings = open_outings

        new_trails = sorted({segment_by_id[seg_id]["trail_name"] for seg_id in new_ids})
        new_miles = sum(segment_by_id[seg_id]["official_miles"] for seg_id in new_ids)
        source_day_index = (source_date - window["start_date"]).days
        target_date = target_start + timedelta(days=source_day_index)
        completed_miles = sum(
            segment["official_miles"]
            for segment in official_segments
            if segment["seg_id"] in completed_ids
        )
        days.append(
            {
                "source_date": source_key,
                "target_2026_date": str(target_date),
                "challenge_day": source_day_index + 1,
                "activity_count": len(day_matches),
                "activity_distance_miles": round_miles(sum(match.get("distance_miles") or 0 for match in day_matches)),
                "activity_moving_minutes": round(sum(match.get("moving_time_minutes") or 0 for match in day_matches), 1),
                "new_segment_ids": new_ids,
                "new_segment_count": len(new_ids),
                "new_official_miles": round_miles(new_miles),
                "new_trail_names": new_trails,
                "cumulative_segment_count": len(completed_ids),
                "cumulative_official_miles": round_miles(completed_miles),
                "remaining_official_miles": plan["summary"]["remaining_available_official_miles"],
                "coverage_valid": plan["coverage_validation"]["valid"],
                "goals": {
                    "beat_2025_68_90_miles": completed_miles >= BASELINE_2025_OFFICIAL_MILES,
                    "half_complete": completed_miles >= (plan["summary"]["official_miles"] * 0.5),
                    "full_complete": len(completed_ids) == len(official_segments),
                },
                "route_menu_primary": primary,
                "route_menu_primary_changed": primary_changed,
                "outing_map_state": outing_state,
                "outing_cards_hidden_since_previous_day": hidden_outings,
            }
        )

    completed_miles = sum(
        segment["official_miles"] for segment in official_segments if segment["seg_id"] in completed_ids
    )
    days_to_baseline = next(
        (day for day in days if day["goals"]["beat_2025_68_90_miles"]),
        None,
    )
    hidden_event_count = sum(len(day["outing_cards_hidden_since_previous_day"]) for day in days)
    distinct_bucket_states = {
        json.dumps(day["outing_map_state"]["next_by_time_bucket"], sort_keys=True)
        for day in days
    }
    return {
        "source_year": window["source_year"],
        "window_name": window["window_name"],
        "source_window": {"start": window["start"], "end": window["end"]},
        "target_2026_window": {
            "start": args.target_start_date,
            "end": str(date.fromisoformat(args.target_start_date) + timedelta(days=len(days) - 1)),
        },
        "summary": {
            "calendar_days": len(days),
            "activity_days": sum(1 for day in days if day["activity_count"]),
            "activities_considered": len(activity_matches),
            "activities_with_geometry_matches": sum(1 for match in activity_matches if match.get("matched_segments")),
            "progress_days": progress_day_count,
            "completed_segments": len(completed_ids),
            "completed_official_miles": round_miles(completed_miles),
            "remaining_official_miles": round_miles(
                sum(segment["official_miles"] for segment in official_segments) - completed_miles
            ),
            "coverage_valid_after_every_rerun": all(day["coverage_valid"] for day in days),
            "beat_2025_baseline_day": days_to_baseline["challenge_day"] if days_to_baseline else None,
            "beat_2025_baseline_source_date": days_to_baseline["source_date"] if days_to_baseline else None,
            "initial_open_runnable_outings": initial_outing_state["open_runnable_outing_count"],
            "final_open_runnable_outings": days[-1]["outing_map_state"]["open_runnable_outing_count"] if days else None,
            "open_runnable_outings_removed": (
                initial_outing_state["open_runnable_outing_count"]
                - days[-1]["outing_map_state"]["open_runnable_outing_count"]
                if days
                else 0
            ),
            "outing_hidden_events": hidden_event_count,
            "days_with_hidden_outings": sum(
                1 for day in days if day["outing_cards_hidden_since_previous_day"]
            ),
            "route_menu_primary_changed_days": sum(1 for day in days if day["route_menu_primary_changed"]),
            "distinct_time_bucket_recommendation_states": len(distinct_bucket_states),
        },
        "activity_matches": activity_matches,
        "days": days,
        "highest_impact_days": sorted(
            [day for day in days if day["new_segment_count"]],
            key=lambda item: (-item["new_segment_count"], -item["new_official_miles"], item["source_date"]),
        )[:8],
    }


def render_markdown(simulation: dict[str, Any]) -> str:
    lines = [
        "# 2026 Two-Year Strava Replay Simulation",
        "",
        "This replays 2024 and 2025 challenge-window Strava GPS geometry against the 2026 official segment set. Each source-year day is mapped onto the 2026 challenge calendar by day number, then the planner and outing-map state are recalculated cumulatively.",
        "",
        "Caveat: this is not official Boise Trails Challenge completion evidence. It is a stress test for whether the current menu/map responds sanely as real-ish activities knock out segments.",
        "",
        "## Summary",
        "",
        "| Source year | Activity days | Activities | Progress days | Completed segs | Completed official mi | Remaining mi | Beat 2025 baseline day | Final open outings | Coverage valid |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for scenario in simulation["scenarios"]:
        summary = scenario["summary"]
        baseline = (
            f"day {summary['beat_2025_baseline_day']} ({summary['beat_2025_baseline_source_date']})"
            if summary["beat_2025_baseline_day"]
            else "not reached"
        )
        lines.append(
            f"| {scenario['source_year']} | {summary['activity_days']} | {summary['activities_considered']} | "
            f"{summary['progress_days']} | {summary['completed_segments']} | {summary['completed_official_miles']} | "
            f"{summary['remaining_official_miles']} | {baseline} | {summary['final_open_runnable_outings']} | "
            f"{summary['coverage_valid_after_every_rerun']} |"
        )
    lines.extend(
        [
            "",
            "## Adaptation Check",
            "",
            "This is the core validation for the current workflow: as simulated activities finish official segments, the planner reruns, map/list outing cards disappear when fully covered, and time-bucket recommendations can change.",
            "",
            "| Source year | Open outings start -> end | Outings removed | Hidden outing events | Days with hidden outings | Primary recommendation changed days | Distinct time-bucket states | Coverage valid after reruns |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in simulation["scenarios"]:
        summary = scenario["summary"]
        lines.append(
            f"| {scenario['source_year']} | "
            f"{summary.get('initial_open_runnable_outings', 'n/a')} -> {summary.get('final_open_runnable_outings', 'n/a')} | "
            f"{summary.get('open_runnable_outings_removed', 0)} | "
            f"{summary.get('outing_hidden_events', 0)} | "
            f"{summary.get('days_with_hidden_outings', 0)} | "
            f"{summary.get('route_menu_primary_changed_days', 0)} | "
            f"{summary.get('distinct_time_bucket_recommendation_states', 0)} | "
            f"{summary['coverage_valid_after_every_rerun']} |"
        )
    lines.extend(
        [
            "",
            "## Day Replay",
            "",
            "| Source | 2026 date | Day | Runs | New segs | Cum segs | Cum mi | Remaining mi | Open outings | Next <=2h outing | Next 4+h outing | Recalc changed |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for scenario in simulation["scenarios"]:
        for day in scenario["days"]:
            next_short = day["outing_map_state"]["next_by_time_bucket"].get("2 hours or less")
            next_long = day["outing_map_state"]["next_by_time_bucket"].get("4+ hours")
            lines.append(
                f"| {scenario['source_year']} | {day['target_2026_date']} | {day['challenge_day']} | "
                f"{day['activity_count']} | {day['new_segment_count']} | {day['cumulative_segment_count']} | "
                f"{day['cumulative_official_miles']} | {day['remaining_official_miles']} | "
                f"{day['outing_map_state']['open_runnable_outing_count']} | "
                f"{outing_label(next_short)} | {outing_label(next_long)} | "
                f"{day['route_menu_primary_changed']} |"
            )
    lines.extend(["", "## Highest-Impact Historical Days", ""])
    for scenario in simulation["scenarios"]:
        lines.append(f"### {scenario['source_year']}")
        if not scenario["highest_impact_days"]:
            lines.append("- No matched progress days.")
        for day in scenario["highest_impact_days"]:
            lines.append(
                f"- {day['source_date']} -> {day['target_2026_date']}: "
                f"{day['new_segment_count']} new segments, {day['new_official_miles']} official mi; "
                f"trails: {', '.join(day['new_trail_names'])}"
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Interactive replay map: `{simulation['outputs']['html']}`",
            f"- JSON: `{simulation['outputs']['json']}`",
            f"- Markdown: `{simulation['outputs']['markdown']}`",
            "",
        ]
    )
    return "\n".join(lines)


def outing_label(outing: dict[str, Any] | None) -> str:
    if not outing:
        return "none"
    return (
        f"{outing['label']} {outing['trailhead']} "
        f"({outing['total_minutes']}m, {outing['official_miles']}mi)"
    )


def render_html(simulation: dict[str, Any], package_map: dict[str, Any]) -> str:
    payload = {
        "simulation": simulation,
        "feature_collections": package_map.get("feature_collections") or {},
    }
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2026 Strava Replay Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#1f2933; }}
    .app {{ display:grid; grid-template-columns:420px minmax(0,1fr); min-height:100vh; }}
    aside {{ overflow:auto; border-right:1px solid #d7ddd4; background:#fff; }}
    header {{ padding:14px 16px; border-bottom:1px solid #d7ddd4; }}
    h1 {{ margin:0 0 5px; font-size:20px; }}
    p {{ margin:0; color:#667085; font-size:13px; line-height:1.4; }}
    .controls, .metrics, .list {{ padding:12px 16px; border-bottom:1px solid #d7ddd4; }}
    label {{ display:block; margin:0 0 8px; color:#344054; font-size:12px; font-weight:700; }}
    select {{ width:100%; min-height:34px; margin-top:4px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .metric {{ border:1px solid #d7ddd4; border-radius:6px; padding:8px; }}
    .metric span {{ display:block; color:#667085; font-size:11px; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:3px; font-size:16px; }}
    .item {{ border:1px solid #d7ddd4; border-radius:6px; padding:8px; margin:0 0 8px; background:#fff; }}
    .item strong {{ display:block; font-size:13px; line-height:1.3; }}
    .item span {{ color:#667085; font-size:12px; line-height:1.35; }}
    #map {{ min-height:100vh; }}
    .parking {{ width:22px; height:22px; border-radius:50%; background:#111827; color:#fff; border:2px solid #fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; box-shadow:0 2px 8px rgba(15,23,42,.32); }}
    @media (max-width:860px) {{ .app {{ grid-template-columns:1fr; }} #map {{ min-height:55vh; }} }}
  </style>
</head>
<body>
<div class="app">
  <aside>
    <header>
      <h1>Strava Replay Map</h1>
      <p>Pick a historical challenge-window day to see simulated 2026 progress and which outing-map routes remain open.</p>
    </header>
    <div class="controls">
      <label>Source year <select id="scenario"></select></label>
      <label>Replay day <select id="day"></select></label>
    </div>
    <div class="metrics" id="metrics"></div>
    <div class="list" id="details"></div>
  </aside>
  <main><div id="map"></div></main>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = {data};
const map = L.map("map", {{ preferCanvas:true }});
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{ maxZoom:19, attribution:"&copy; OpenStreetMap contributors" }}).addTo(map);
const routeLayer = L.layerGroup().addTo(map);
const parkingLayer = L.layerGroup().addTo(map);
function esc(value) {{ return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch])); }}
function fmt(value) {{ return value === null || value === undefined ? "n/a" : value; }}
function scenario() {{ return DATA.simulation.scenarios[Number(document.getElementById("scenario").value || 0)]; }}
function day() {{ return scenario().days[Number(document.getElementById("day").value || 0)]; }}
function routeParts(feature) {{
  const geom = feature.geometry || {{}};
  if (geom.type === "LineString") return [geom.coordinates || []];
  if (geom.type === "MultiLineString") return geom.coordinates || [];
  return [];
}}
function parkingIcon() {{
  return L.divIcon({{ className:"", iconSize:[26,26], iconAnchor:[13,13], html:'<div class="parking">P</div>' }});
}}
function popup(props) {{
  return `<strong>${{esc(props.title || props.name || props.trailhead)}}</strong><br>${{esc(props.block_name || "")}}<br>${{fmt(props.official_miles)}} official mi · ${{fmt(props.on_foot_miles)}} on-foot mi`;
}}
function populateControls() {{
  const scenarioSelect = document.getElementById("scenario");
  scenarioSelect.innerHTML = DATA.simulation.scenarios.map((item, index) => `<option value="${{index}}">${{item.source_year}}</option>`).join("");
  scenarioSelect.addEventListener("change", () => {{ populateDays(); draw(); }});
  populateDays();
}}
function populateDays() {{
  const daySelect = document.getElementById("day");
  daySelect.innerHTML = scenario().days.map((item, index) => `<option value="${{index}}">Day ${{item.challenge_day}} · ${{item.target_2026_date}} · +${{item.new_segment_count}} segs</option>`).join("");
  daySelect.addEventListener("change", draw);
}}
function draw() {{
  const current = day();
  const openIds = new Set(current.outing_map_state.open_candidate_ids || []);
  routeLayer.clearLayers();
  parkingLayer.clearLayers();
  const layers = [];
  (DATA.feature_collections.routes?.features || []).forEach(feature => {{
    if (!openIds.has(String(feature.properties.candidate_id))) return;
    const color = feature.properties.color || "#2563eb";
    routeParts(feature).forEach(part => {{
      const latlngs = part.map(pt => [pt[1], pt[0]]);
      const layer = L.polyline(latlngs, {{ color, weight:4, opacity:.72, lineCap:"round", lineJoin:"round" }}).bindPopup(popup(feature.properties || {{}})).addTo(routeLayer);
      layers.push(layer);
    }});
  }});
  (DATA.feature_collections.parking?.features || []).forEach(feature => {{
    if (!openIds.has(String(feature.properties.candidate_id))) return;
    const coords = feature.geometry.coordinates;
    const layer = L.marker([coords[1], coords[0]], {{ icon:parkingIcon() }}).bindPopup(popup(feature.properties || {{}})).addTo(parkingLayer);
    layers.push(layer);
  }});
  document.getElementById("metrics").innerHTML = [
    ["Cum mi", current.cumulative_official_miles],
    ["Remaining", current.remaining_official_miles],
    ["Cum segs", current.cumulative_segment_count],
    ["Open outings", current.outing_map_state.open_runnable_outing_count],
    ["Runs", current.activity_count],
    ["New segs", current.new_segment_count],
  ].map(([a,b]) => `<div class="metric"><span>${{a}}</span><strong>${{esc(b)}}</strong></div>`).join("");
  const next = Object.entries(current.outing_map_state.next_by_time_bucket || {{}})
    .map(([bucket, outing]) => outing ? `<div class="item"><strong>${{esc(bucket)}}: ${{esc(outing.label)}} ${{esc(outing.trailhead)}}</strong><span>${{esc(outing.total_minutes)}}m · ${{esc(outing.official_miles)}} official · ${{esc((outing.trails || []).join(", "))}}</span></div>` : "")
    .join("");
  document.getElementById("details").innerHTML = `<div class="item"><strong>${{esc(scenario().source_year)}} source ${{esc(current.source_date)}} -> 2026 ${{esc(current.target_2026_date)}}</strong><span>${{esc(current.new_trail_names.join(", ") || "no new matched trails")}}</span></div>${{next}}`;
  if (layers.length) map.fitBounds(L.featureGroup(layers).getBounds(), {{ padding:[24,24], maxZoom:13 }});
}}
populateControls();
draw();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--strava-details-dir", type=Path, default=DEFAULT_STRAVA_DETAILS_DIR)
    parser.add_argument("--activity-summary-csv", type=Path, default=DEFAULT_ACTIVITY_SUMMARY_CSV)
    parser.add_argument("--activity-detail-summary-csv", type=Path, default=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV)
    parser.add_argument("--segment-perf-csv", type=Path, default=DEFAULT_SEGMENT_PERF_CSV)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--strava-pull-summary", type=Path, default=DEFAULT_STRAVA_PULL_SUMMARY)
    parser.add_argument("--package-map-json", type=Path, default=DEFAULT_PACKAGE_MAP_JSON)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGN_JSON)
    parser.add_argument("--manual-design-report-json", type=Path, default=DEFAULT_MANUAL_DESIGN_REPORT_JSON)
    parser.add_argument("--source-years", default="2024,2025")
    parser.add_argument("--from-date", default=None, help="Optional single custom source window start date.")
    parser.add_argument("--to-date", default=None, help="Optional single custom source window end date.")
    parser.add_argument("--target-start-date", default="2026-06-18")
    parser.add_argument("--threshold-miles", type=float, default=0.045)
    parser.add_argument("--min-fraction", type=float, default=0.55)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "simulation.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_DIR / "simulation.md")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_DIR / "simulation_replay_map.html")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_segments, _ = load_official_segments(args.official)
    segment_by_id = {segment["seg_id"]: segment for segment in official_segments}
    dem_context = load_dem_context(args.dem_tif, args.dem_summary_json)
    elevation_sampler = dem_context["sampler"]
    base_state = load_state(args.state)
    package_map = load_package_map_with_manual_design(
        args.package_map_json,
        args.manual_design_json,
        args.manual_design_report_json,
    )

    if args.from_date and args.to_date:
        start = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date)
        windows = [
            {
                "source_year": start.year,
                "window_name": "custom_window",
                "start": args.from_date,
                "end": args.to_date,
                "start_date": start,
                "end_date": end,
            }
        ]
    else:
        windows = challenge_windows_from_pull_summary(
            args.strava_pull_summary,
            parse_source_years(args.source_years),
        )

    scenarios = []
    for window in windows:
        print(f"Simulating {window['source_year']} ({window['start']} to {window['end']})")
        scenarios.append(
            simulate_window(
                window,
                args,
                official_segments,
                segment_by_id,
                elevation_sampler,
                package_map,
                base_state,
            )
        )

    simulation = {
        "planning_status": "strava_two_year_day_replay",
        "summary": {
            "official_segments": len(official_segments),
            "official_miles": round_miles(sum(segment["official_miles"] for segment in official_segments)),
            "source_years": [scenario["source_year"] for scenario in scenarios],
            "scenario_count": len(scenarios),
            "match_threshold_miles": args.threshold_miles,
            "min_segment_sample_fraction": args.min_fraction,
            "dem_loaded": bool(elevation_sampler),
        },
        "source_datasets": {
            "official_geojson": str(args.official),
            "state": str(args.state),
            "strava_details_dir": str(args.strava_details_dir),
            "strava_pull_summary": str(args.strava_pull_summary),
            "connector_geojson": str(args.connector_geojson),
            "trailheads_geojson": str(args.trailheads_geojson),
            "package_map_json": str(args.package_map_json),
            "manual_design_json": str(args.manual_design_json),
            "manual_design_report_json": str(args.manual_design_report_json),
            "dem": dem_context["metadata"],
        },
        "scenarios": scenarios,
        "outputs": {
            "json": str(args.output_json),
            "markdown": str(args.output_md),
            "html": str(args.output_html),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, simulation)
    args.output_md.write_text(render_markdown(simulation), encoding="utf-8")
    args.output_html.write_text(render_html(simulation, package_map), encoding="utf-8")
    manifest_path = args.output_json.parent / "simulation-artifact-manifest.json"
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id="2026-05-05-strava-two-year-simulation",
            inputs=[
                args.official,
                args.state,
                args.strava_pull_summary,
                args.package_map_json,
                args.manual_design_json,
                args.manual_design_report_json,
            ],
            outputs=[args.output_json, args.output_md, args.output_html],
            command="simulate_strava_day_progress.py",
            metadata=simulation["summary"],
        ),
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_html}")
    print(f"Wrote {manifest_path}")
    for scenario in scenarios:
        print(
            f"{scenario['source_year']}: "
            f"{scenario['summary']['completed_segments']} segments, "
            f"{scenario['summary']['completed_official_miles']} official miles, "
            f"{scenario['summary']['final_open_runnable_outings']} open outings remaining"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
