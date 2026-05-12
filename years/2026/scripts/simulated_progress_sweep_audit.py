#!/usr/bin/env python3
"""Rank routes and field days by simulated progress impact on the remaining menu."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from cluster_level_repricing_audit import (  # noqa: E402
    display_path,
    float_value,
    int_value,
    normalized_ids,
    official_segment_miles,
    route_index,
    route_key,
    route_label,
    route_metrics,
    rounded,
    sort_id,
    write_json,
)
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_ROUTE_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "simulated-progress-sweep-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "simulated-progress-sweep-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "simulated-progress-sweep-audit-2026-05-12-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def base_completed_segment_ids(field_tool_data: dict[str, Any]) -> set[str]:
    progress = field_tool_data.get("progress") or {}
    completed = set(
        normalized_ids(
            progress.get("completed_segment_ids")
            or progress.get("completed_segment_ids_at_export")
            or []
        )
    )
    completed.update(normalized_ids(progress.get("extra_completed_segment_ids") or []))
    completed.difference_update(normalized_ids(progress.get("missed_segment_ids") or []))
    completed.difference_update(normalized_ids(progress.get("blocked_segment_ids") or []))
    return completed


def route_repeat_index(route_repeat_audit: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = route_index(routes)
    rows: dict[str, dict[str, Any]] = {}
    for row in route_repeat_audit.get("routes") or []:
        route = lookup.get(str(row.get("outing_id") or "")) or lookup.get(str(row.get("label") or ""))
        if route:
            rows[route_key(route)] = row
    return rows


def route_claimed_ids(route: dict[str, Any]) -> set[str]:
    return set(normalized_ids(route.get("segment_ids") or []))


def route_completed_ids(route: dict[str, Any], repeat_rows_by_route: dict[str, dict[str, Any]]) -> set[str]:
    claimed = route_claimed_ids(route)
    repeat_row = repeat_rows_by_route.get(route_key(route)) or {}
    actual_full = set(normalized_ids(repeat_row.get("actual_full_segment_ids") or []))
    return actual_full | claimed


def sum_metrics(routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "route_count": len(routes),
        "on_foot_miles": rounded(sum(float_value(route.get("on_foot_miles")) for route in routes)),
        "door_to_door_minutes_p75": sum(int_value(route.get("door_to_door_minutes_p75")) for route in routes),
        "door_to_door_minutes_p90": sum(int_value(route.get("door_to_door_minutes_p90")) for route in routes),
    }


def official_miles_for(segment_ids: set[str], segment_miles: dict[str, float]) -> float:
    return rounded(sum(segment_miles.get(segment_id, 0.0) for segment_id in segment_ids))


def route_ref_key(ref: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> str | None:
    if not ref:
        return None
    route = lookup.get(str(ref.get("outing_id") or "")) or lookup.get(str(ref.get("label") or ""))
    return route_key(route) if route else None


def subject_rows(field_tool_data: dict[str, Any], routes_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookup = route_index(list(routes_by_key.values()))
    for route in routes_by_key.values():
        key = route_key(route)
        rows.append(
            {
                "subject_type": "route_card",
                "route_id": key,
                "label": route_label(route),
                "date": None,
                "subject_route_keys": [key],
                "subject_metrics": {
                    "on_foot_miles": rounded(route.get("on_foot_miles")),
                    "door_to_door_minutes_p75": int_value(route.get("door_to_door_minutes_p75")),
                    "door_to_door_minutes_p90": int_value(route.get("door_to_door_minutes_p90")),
                },
            }
        )
    for day in (field_tool_data.get("field_day_layer") or {}).get("field_days") or []:
        route_keys = []
        loop_labels = []
        for loop in day.get("loops") or []:
            key = route_ref_key(loop.get("route_card_ref") or {}, lookup)
            if key and key not in route_keys:
                route_keys.append(key)
            loop_labels.append(str(loop.get("label") or key or loop.get("loop_id") or "unknown-loop"))
        if not route_keys:
            continue
        rows.append(
            {
                "subject_type": "field_day",
                "route_id": str(day.get("field_day_id") or day.get("date") or "|".join(route_keys)),
                "label": f"{day.get('date') or 'unscheduled'}: " + " + ".join(loop_labels),
                "date": day.get("date"),
                "subject_route_keys": route_keys,
                "subject_metrics": {
                    "on_foot_miles": rounded(day.get("on_foot_miles")),
                    "door_to_door_minutes_p75": int_value(
                        day.get("field_day_schedule_p75_minutes") or day.get("p75_minutes")
                    ),
                    "door_to_door_minutes_p90": int_value(
                        day.get("field_day_schedule_p90_minutes") or day.get("p90_minutes")
                    ),
                },
                "field_day": {
                    "date": day.get("date"),
                    "day_type": day.get("day_type"),
                    "loop_count": day.get("loop_count"),
                    "transfer_count": day.get("transfer_count"),
                    "execution_status": day.get("execution_status"),
                    "timing_authority": day.get("timing_authority"),
                },
            }
        )
    return rows


def route_status_rows(
    routes: list[dict[str, Any]],
    completed_ids: set[str],
    segment_miles: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    removed = []
    shrunk = []
    open_routes = []
    for route in routes:
        claimed = route_claimed_ids(route)
        newly_completed = claimed & completed_ids
        remaining = claimed - completed_ids
        if claimed and not remaining:
            removed.append(
                {
                    "route_key": route_key(route),
                    "route": route_metrics(route),
                    "completed_claimed_segment_ids": normalized_ids(newly_completed),
                    "remaining_segment_ids": [],
                    "completed_claimed_official_miles": official_miles_for(newly_completed, segment_miles),
                    "status": "removed_all_claimed_credit_completed",
                }
            )
        elif newly_completed:
            shrunk.append(
                {
                    "route_key": route_key(route),
                    "route": route_metrics(route),
                    "completed_claimed_segment_ids": normalized_ids(newly_completed),
                    "remaining_segment_ids": normalized_ids(remaining),
                    "completed_claimed_official_miles": official_miles_for(newly_completed, segment_miles),
                    "remaining_official_miles": official_miles_for(remaining, segment_miles),
                    "status": "shrunk_needs_route_card_reprice",
                }
            )
            open_routes.append(route)
        else:
            open_routes.append(route)
    return {"removed": removed, "shrunk": shrunk, "open": open_routes}


def simulate_subject(
    subject: dict[str, Any],
    *,
    routes: list[dict[str, Any]],
    routes_by_key: dict[str, dict[str, Any]],
    repeat_rows_by_route: dict[str, dict[str, Any]],
    base_completed_ids: set[str],
    baseline_metrics: dict[str, Any],
    segment_miles: dict[str, float],
) -> dict[str, Any]:
    subject_route_keys = set(subject["subject_route_keys"])
    simulated_ids = set(base_completed_ids)
    subject_claimed_ids: set[str] = set()
    subject_physical_ids: set[str] = set()
    for key in subject_route_keys:
        route = routes_by_key[key]
        subject_claimed_ids.update(route_claimed_ids(route))
        subject_physical_ids.update(route_completed_ids(route, repeat_rows_by_route))
    simulated_ids.update(subject_physical_ids)
    newly_credited = subject_physical_ids - base_completed_ids
    latent_completed = newly_credited - subject_claimed_ids
    status_rows = route_status_rows(routes, simulated_ids, segment_miles)
    removed_rows = status_rows["removed"]
    shrunk_rows = status_rows["shrunk"]
    open_routes = status_rows["open"]
    future_removed = [row for row in removed_rows if row["route_key"] not in subject_route_keys]
    future_shrunk = [row for row in shrunk_rows if row["route_key"] not in subject_route_keys]
    removed_routes = [routes_by_key[row["route_key"]] for row in removed_rows]
    future_removed_routes = [routes_by_key[row["route_key"]] for row in future_removed]
    remaining_metrics = sum_metrics(open_routes)
    removed_metrics = sum_metrics(removed_routes)
    future_removed_metrics = sum_metrics(future_removed_routes)
    shrink_official_miles = rounded(sum(float_value(row["completed_claimed_official_miles"]) for row in future_shrunk))
    return {
        **subject,
        "segments_newly_credited": normalized_ids(newly_credited),
        "newly_credited_segment_count": len(newly_credited),
        "newly_credited_official_miles": official_miles_for(newly_credited, segment_miles),
        "claimed_segment_ids": normalized_ids(subject_claimed_ids),
        "latent_completed_segment_ids": normalized_ids(latent_completed),
        "latent_completed_segment_count": len(latent_completed),
        "latent_completed_official_miles": official_miles_for(latent_completed, segment_miles),
        "future_routes_removed": future_removed,
        "future_routes_shrunk": future_shrunk,
        "all_removed_routes": removed_rows,
        "all_shrunk_routes": shrunk_rows,
        "remaining_menu_after_simulation": remaining_metrics,
        "net_remaining_menu_saved": {
            "on_foot_miles": rounded(baseline_metrics["on_foot_miles"] - remaining_metrics["on_foot_miles"]),
            "door_to_door_minutes_p75": baseline_metrics["door_to_door_minutes_p75"]
            - remaining_metrics["door_to_door_minutes_p75"],
            "door_to_door_minutes_p90": baseline_metrics["door_to_door_minutes_p90"]
            - remaining_metrics["door_to_door_minutes_p90"],
            "route_count": baseline_metrics["route_count"] - remaining_metrics["route_count"],
        },
        "removed_route_card_savings": removed_metrics,
        "future_collapse_savings": future_removed_metrics,
        "future_shrink_unpriced": {
            "route_count": len(future_shrunk),
            "completed_claimed_official_miles": shrink_official_miles,
            "note": "Partial route-card shrink is not priced as on-foot/p75 savings until regenerated route cards exist.",
        },
        "priority_score": {
            "future_collapse_on_foot_miles": future_removed_metrics["on_foot_miles"],
            "future_collapse_p75_minutes": future_removed_metrics["door_to_door_minutes_p75"],
            "future_shrink_official_miles_unpriced": shrink_official_miles,
            "newly_credited_official_miles": official_miles_for(newly_credited, segment_miles),
        },
    }


def sort_sweeps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float_value(row["priority_score"]["future_collapse_on_foot_miles"]),
            -int_value(row["priority_score"]["future_collapse_p75_minutes"]),
            -float_value(row["priority_score"]["future_shrink_official_miles_unpriced"]),
            -float_value(row["newly_credited_official_miles"]),
            row["label"],
        ),
    )


def build_simulated_progress_sweep_audit(
    field_tool_data: dict[str, Any],
    route_repeat_audit: dict[str, Any],
    official_segments: list[dict[str, Any]],
    *,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = {route_key(route): route for route in routes}
    repeat_rows_by_route = route_repeat_index(route_repeat_audit, routes)
    base_completed_ids = base_completed_segment_ids(field_tool_data)
    segment_miles = official_segment_miles(official_segments, field_tool_data)
    baseline_metrics = sum_metrics(routes)
    subjects = subject_rows(field_tool_data, routes_by_key)
    sweeps = [
        simulate_subject(
            subject,
            routes=routes,
            routes_by_key=routes_by_key,
            repeat_rows_by_route=repeat_rows_by_route,
            base_completed_ids=base_completed_ids,
            baseline_metrics=baseline_metrics,
            segment_miles=segment_miles,
        )
        for subject in subjects
    ]
    route_sweeps = sort_sweeps([row for row in sweeps if row["subject_type"] == "route_card"])
    field_day_sweeps = sort_sweeps([row for row in sweeps if row["subject_type"] == "field_day"])
    all_ranked = sort_sweeps(sweeps)
    sweeps_with_future_removed = [
        row for row in sweeps if int_value(row["future_collapse_savings"]["route_count"]) > 0
    ]
    sweeps_with_future_shrunk = [
        row for row in sweeps if int_value(row["future_shrink_unpriced"]["route_count"]) > 0
    ]
    status = (
        "simulated_progress_priority_found"
        if sweeps_with_future_removed or sweeps_with_future_shrunk
        else "no_future_collapse_found"
    )
    return {
        "schema": "boise_trails_simulated_progress_sweep_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "parameters": {
            "simulated_completion_source": "route-repeat audit actual_full_segment_ids plus route-card segment_ids",
            "baseline_completed_segment_count": len(base_completed_ids),
            "pricing_policy": "full future route removals priced as on-foot/p75/p90 savings; partial shrinks reported but unpriced",
            "priority_ranking": "future_collapse_on_foot_miles, then future_collapse_p75_minutes, then unpriced shrink official miles",
        },
        "scope": {
            "proves": [
                "which route cards or field days remove later route cards if completed first",
                "which later route cards become partial-shrink candidates after simulated progress",
                "a route/day priority queue over the existing certified field-packet menu",
            ],
            "does_not_prove": [
                "official BTC app progress before challenge-window activity validation",
                "field-ready replacement cards for partial shrink rows",
                "a regenerated global route menu over newly generated loop candidates",
            ],
        },
        "baseline_remaining_menu": baseline_metrics,
        "summary": {
            "route_count": len(routes),
            "field_day_count": len((field_tool_data.get("field_day_layer") or {}).get("field_days") or []),
            "subject_count": len(sweeps),
            "route_sweep_count": len(route_sweeps),
            "field_day_sweep_count": len(field_day_sweeps),
            "sweeps_with_future_removed_route_count": len(sweeps_with_future_removed),
            "sweeps_with_future_shrunk_route_count": len(sweeps_with_future_shrunk),
            "top_route_card_by_future_collapse": route_sweeps[0]["label"] if route_sweeps else None,
            "top_route_card_future_collapse_on_foot_miles": route_sweeps[0]["future_collapse_savings"]["on_foot_miles"]
            if route_sweeps
            else 0.0,
            "top_field_day_by_future_collapse": field_day_sweeps[0]["label"] if field_day_sweeps else None,
            "top_field_day_future_collapse_on_foot_miles": field_day_sweeps[0]["future_collapse_savings"]["on_foot_miles"]
            if field_day_sweeps
            else 0.0,
        },
        "route_sweeps_ranked": route_sweeps,
        "field_day_sweeps_ranked": field_day_sweeps,
        "all_sweeps_ranked": all_ranked,
    }


def route_summary_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return ", ".join(row["route"]["label"] for row in rows)


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Simulated Progress Sweep Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Field days audited: {summary['field_day_count']}",
        f"- Sweeps with future route removals: {summary['sweeps_with_future_removed_route_count']}",
        f"- Sweeps with future route shrinks: {summary['sweeps_with_future_shrunk_route_count']}",
        f"- Top route-card collapse: {summary['top_route_card_by_future_collapse']} ({summary['top_route_card_future_collapse_on_foot_miles']:.2f} future on-foot mi)",
        f"- Top field-day collapse: {summary['top_field_day_by_future_collapse']} ({summary['top_field_day_future_collapse_on_foot_miles']:.2f} future on-foot mi)",
        "",
        "## Route Priority",
        "",
        "| Rank | Route | New credit ids | Latent ids | Future removed | Future shrunk | Future collapse mi | Future p75 | Net saved mi | Net saved p75 | Shrink official mi |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(audit.get("route_sweeps_ranked", [])[:20], start=1):
        if (
            float_value(row["future_collapse_savings"]["on_foot_miles"]) <= 0
            and int_value(row["future_shrink_unpriced"]["route_count"]) <= 0
        ):
            continue
        lines.append(
            f"| {index} | {row['label']} | {row['newly_credited_segment_count']} | {row['latent_completed_segment_count']} | "
            f"{route_summary_list(row['future_routes_removed']) or ''} | {row['future_shrink_unpriced']['route_count']} | "
            f"{float_value(row['future_collapse_savings']['on_foot_miles']):.2f} | "
            f"{row['future_collapse_savings']['door_to_door_minutes_p75']} | "
            f"{float_value(row['net_remaining_menu_saved']['on_foot_miles']):.2f} | "
            f"{row['net_remaining_menu_saved']['door_to_door_minutes_p75']} | "
            f"{float_value(row['future_shrink_unpriced']['completed_claimed_official_miles']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Field-Day Priority",
            "",
            "| Rank | Field day | New credit ids | Latent ids | Future removed | Future shrunk | Future collapse mi | Future p75 | Net saved mi | Net saved p75 | Shrink official mi |",
            "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(audit.get("field_day_sweeps_ranked", [])[:20], start=1):
        if (
            float_value(row["future_collapse_savings"]["on_foot_miles"]) <= 0
            and int_value(row["future_shrink_unpriced"]["route_count"]) <= 0
        ):
            continue
        lines.append(
            f"| {index} | {row['label']} | {row['newly_credited_segment_count']} | {row['latent_completed_segment_count']} | "
            f"{route_summary_list(row['future_routes_removed']) or ''} | {row['future_shrink_unpriced']['route_count']} | "
            f"{float_value(row['future_collapse_savings']['on_foot_miles']):.2f} | "
            f"{row['future_collapse_savings']['door_to_door_minutes_p75']} | "
            f"{float_value(row['net_remaining_menu_saved']['on_foot_miles']):.2f} | "
            f"{row['net_remaining_menu_saved']['door_to_door_minutes_p75']} | "
            f"{float_value(row['future_shrink_unpriced']['completed_claimed_official_miles']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- This is a route-priority simulator, not a progress proof layer.",
            "- Simulated completion uses route-card claims plus GPX-derived full official segment coverage from the route-repeat audit.",
            "- Future full route removals are priced as saved on-foot/p75/p90 work.",
            "- Partial future route shrinks are reported as unpriced official-credit pressure until replacement route cards are generated and recertified.",
            "- Official BTC progress still requires challenge-window activity validation.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--route-repeat-audit-json", type=Path, default=DEFAULT_ROUTE_REPEAT_AUDIT_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    audit = build_simulated_progress_sweep_audit(
        read_json(args.field_tool_data_json),
        read_json(args.route_repeat_audit_json),
        official_segments,
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "route_repeat_audit_json": display_path(args.route_repeat_audit_json),
            "official_geojson": display_path(args.official_geojson),
            "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="simulated-progress-sweep-audit-2026-05-12",
        inputs=[args.field_tool_data_json, args.route_repeat_audit_json, args.official_geojson],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/simulated_progress_sweep_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] == "no_future_collapse_found" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
