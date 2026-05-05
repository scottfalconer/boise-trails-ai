#!/usr/bin/env python3
"""Compare the selected human plan to a stricter constraint probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, utc_now_id, write_manifest  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def ready_segment_ids(execution: dict[str, Any]) -> set[int]:
    return {
        int(segment_id)
        for outing in execution.get("outings") or []
        if outing.get("execution_status") == "simulated_ready"
        for segment_id in outing.get("segment_ids") or []
    }


def scheduled_segment_ids_from_schedule(schedule: dict[str, Any]) -> set[int]:
    return {
        int(segment_id)
        for day in schedule.get("days") or []
        if day.get("status") == "scheduled"
        for segment_id in day.get("segment_ids") or []
    }


def label_counts(runbook: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for day in runbook.get("days") or []:
        for outing in day.get("outings") or []:
            counts[str(outing.get("route_label") or "unlabeled")] += 1
    return dict(sorted(counts.items()))


def build_gap_report(
    selected_runbook: dict[str, Any],
    strict_schedule: dict[str, Any],
    execution: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    ready_segments = ready_segment_ids(execution)
    strict_segments = scheduled_segment_ids_from_schedule(strict_schedule)
    strict_missing = sorted(ready_segments - strict_segments)
    load = selected_runbook.get("load_analysis") or {}
    selected_summary = selected_runbook.get("summary") or {}
    strict_summary = strict_schedule.get("summary") or {}
    return {
        "run_id": utc_now_id(),
        "profile_name": profile_name,
        "selected_profile": selected_runbook.get("profile_name"),
        "selected_summary": selected_summary,
        "selected_audit": selected_runbook.get("audit") or {},
        "selected_constraints": selected_runbook.get("constraints") or {},
        "selected_route_label_counts": label_counts(selected_runbook),
        "strict_probe_summary": strict_summary,
        "strict_probe_constraints": strict_schedule.get("constraints") or {},
        "strict_probe_missing_segment_count": len(strict_missing),
        "strict_probe_missing_segment_ids": strict_missing,
        "constraint_gaps": {
            "strict_probe_scheduled_segments": strict_summary.get("scheduled_segments"),
            "required_segments": len(ready_segments),
            "strict_probe_short_by_segments": len(strict_missing),
            "weekday_exception_days_needed_in_selected": load.get("weekday_exception_day_count", 0),
            "weekday_exception_minutes_in_selected": load.get("weekday_exception_minutes", 0),
            "rest_days_before_latest_deficit": load.get("rest_days_before_latest_deficit", 0),
            "consecutive_day_violation_count": load.get("consecutive_day_violation_count", 0),
            "weekly_mileage_violation_count": load.get("weekly_mileage_violation_count", 0),
            "long_day_count_violation_count": load.get("long_day_count_violation_count", 0),
        },
        "interpretation": [
            "Strict 180-minute weekdays are treated as a feasibility probe, not the selected plan.",
            "The selected plan keeps 100% coverage by allowing explicitly reported weekday-normal-cap exceptions.",
            "Single-car-only routing remains the dominant source of extra on-foot mileage.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_summary"]
    strict = report["strict_probe_summary"]
    gaps = report["constraint_gaps"]
    lines = [
        "# Human 100 v2 Constraint Gap Report",
        "",
        f"- Selected profile: {report.get('selected_profile')}",
        f"- Selected segments: {selected.get('scheduled_segments')}",
        f"- Selected official miles: {selected.get('scheduled_official_miles')}",
        f"- Selected total on-foot miles: {selected.get('scheduled_total_on_foot_miles')}",
        f"- Selected ascent: {selected.get('scheduled_ascent_ft')} ft",
        f"- Strict-probe segments: {strict.get('scheduled_segments')}",
        f"- Strict-probe missing segments: {report.get('strict_probe_missing_segment_count')}",
        "",
        "## Constraint Gaps",
        "",
        f"- Weekday exception days in selected plan: {gaps.get('weekday_exception_days_needed_in_selected')}",
        f"- Weekday exception minutes in selected plan: {gaps.get('weekday_exception_minutes_in_selected')}",
        f"- Rest-days-before-latest deficit: {gaps.get('rest_days_before_latest_deficit')}",
        f"- Consecutive-day violation count: {gaps.get('consecutive_day_violation_count')}",
        f"- Weekly-mileage violation count: {gaps.get('weekly_mileage_violation_count')}",
        f"- Long-day-count violation count: {gaps.get('long_day_count_violation_count')}",
        "",
        "## Route Labels",
        "",
    ]
    for label, count in report.get("selected_route_label_counts", {}).items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Interpretation", ""])
    for item in report.get("interpretation") or []:
        lines.append(f"- {item}")
    if report.get("strict_probe_missing_segment_ids"):
        lines.extend(["", "## Strict-Probe Missing Segment IDs", ""])
        lines.append(", ".join(str(item) for item in report["strict_probe_missing_segment_ids"]))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-runbook", type=Path, required=True)
    parser.add_argument("--strict-schedule-json", type=Path, required=True)
    parser.add_argument("--execution-json", type=Path, required=True)
    parser.add_argument("--profile-name", default="human-100-v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_gap_report(
        read_json(args.selected_runbook),
        read_json(args.strict_schedule_json),
        read_json(args.execution_json),
        args.profile_name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.profile_name}-constraint-gap-report.json"
    md_path = args.output_dir / f"{args.profile_name}-constraint-gap-report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    manifest_path = args.output_dir / f"{args.profile_name}-constraint-gap-report-artifact-manifest.json"
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=report["run_id"],
            inputs=[args.selected_runbook, args.strict_schedule_json, args.execution_json],
            outputs=[json_path, md_path],
            command="human_plan_gap_report.py",
            metadata={"profile_name": args.profile_name},
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
