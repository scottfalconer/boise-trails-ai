#!/usr/bin/env python3
"""Summarize car-hop quality for the relaxed-drive near-miss solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_INPUT_JSON = YEAR_DIR / "checkpoints" / "p90-near-miss-pressure-audit-drive45-n40-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-relaxed-drive-solution-quality-2026-05-06"


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


def summarize_days(days: list[dict[str, Any]]) -> dict[str, Any]:
    if not days:
        return {
            "field_day_count": 0,
            "multi_start_days": 0,
            "total_between_drive_minutes": 0,
            "max_between_drive_minutes": 0,
            "days_with_between_drive_over_20": 0,
            "days_with_p90_over_340": 0,
        }
    return {
        "field_day_count": len(days),
        "multi_start_days": sum(1 for day in days if int(day.get("loop_count") or 0) > 1),
        "total_between_drive_minutes": int(sum(int(day.get("between_drive_minutes") or 0) for day in days)),
        "max_between_drive_minutes": int(max(int(day.get("between_drive_minutes") or 0) for day in days)),
        "days_with_between_drive_over_20": sum(1 for day in days if int(day.get("between_drive_minutes") or 0) > 20),
        "days_with_p90_over_340": sum(1 for day in days if int(day.get("p90_minutes") or 0) > 340),
    }


def short_day(day: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_day_id": day["field_day_id"],
        "day_type": day["day_type"],
        "p75_minutes": day["p75_minutes"],
        "p90_minutes": day["p90_minutes"],
        "loop_count": day["loop_count"],
        "segment_count": len(day["segment_ids"]),
        "between_drive_minutes": day.get("between_drive_minutes") or 0,
        "drive_minutes": day.get("drive_minutes") or 0,
        "loop_ids": day["loop_ids"],
    }


def build_report(solution_json: dict[str, Any], *, limit: int) -> dict[str, Any]:
    solution = (
        solution_json.get("p75_min_full_cover")
        or solution_json.get("actual_day_count_full_cover")
        or {}
    )
    days = solution.get("selected_field_days") or []
    return {
        "objective": "summarize car-hop quality for the relaxed 292/360 drive45 field-day solution",
        "source_files": {
            "relaxed_drive_solution_json": display_path(DEFAULT_INPUT_JSON),
        },
        "solution_summary": {
            "solution_source": "p75_min_full_cover" if solution_json.get("p75_min_full_cover") else "actual_day_count_full_cover",
            "success": solution.get("success"),
            "field_day_count": solution.get("field_day_count"),
            "weekday_field_day_count": solution.get("weekday_field_day_count"),
            "weekend_field_day_count": solution.get("weekend_field_day_count"),
            "total_p75_minutes": solution.get("total_p75_minutes"),
            "max_p90_minutes": solution.get("max_p90_minutes"),
        },
        "car_hop_summary": summarize_days(days),
        "top_between_drive_days": [
            short_day(day)
            for day in sorted(days, key=lambda day: int(day.get("between_drive_minutes") or 0), reverse=True)[:limit]
        ],
        "longest_p90_days": [
            short_day(day)
            for day in sorted(days, key=lambda day: int(day.get("p90_minutes") or 0), reverse=True)[:limit]
        ],
        "interpretation": [
            "This sensitivity solution is not car-hop-free, but the selected plan has only one day with more than 20 minutes of between-start driving.",
            "The relaxed-drive proof still needs human review because loop grouping may be less enjoyable than the outing-first field menu.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["solution_summary"]
    hop = report["car_hop_summary"]
    lines = [
        "# P90 Relaxed-Drive Solution Quality",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Solution success: {summary['success']}",
        f"- Solution source: {summary['solution_source']}",
        f"- Field days: {summary['field_day_count']} ({summary['weekday_field_day_count']} weekday / {summary['weekend_field_day_count']} weekend)",
        f"- Total p75 minutes: {summary['total_p75_minutes']}",
        f"- Max p90 minutes: {summary['max_p90_minutes']}",
        f"- Multi-start days: {hop['multi_start_days']}",
        f"- Total between-start drive minutes: {hop['total_between_drive_minutes']}",
        f"- Max between-start drive minutes: {hop['max_between_drive_minutes']}",
        f"- Days with between-start drive >20 min: {hop['days_with_between_drive_over_20']}",
        f"- Days with p90 >340 min: {hop['days_with_p90_over_340']}",
        "",
        "## Highest Between-Start Drive Days",
        "",
        "| Between drive | P90 | P75 | Loops | Segments | Field day |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for day in report["top_between_drive_days"]:
        lines.append(
            f"| {day['between_drive_minutes']} | {day['p90_minutes']} | {day['p75_minutes']} | "
            f"{day['loop_count']} | {day['segment_count']} | {day['field_day_id']} |"
        )
    lines.extend(["", "## Longest P90 Days", "", "| P90 | P75 | Between drive | Loops | Field day |", "|---:|---:|---:|---:|---|"])
    for day in report["longest_p90_days"]:
        lines.append(
            f"| {day['p90_minutes']} | {day['p75_minutes']} | {day['between_drive_minutes']} | "
            f"{day['loop_count']} | {day['field_day_id']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(read_json(args.input_json), limit=args.limit)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["car_hop_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
