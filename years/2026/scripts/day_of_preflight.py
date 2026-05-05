#!/usr/bin/env python3
"""Build day-of route preflight checks for executable recommendations."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any


YEAR_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXECUTION_JSON = (
    YEAR_DIR
    / "experiments"
    / "2026-05-04-outing-execution-simulation"
    / "outing_execution.json"
)
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "preflight"


LOW_FOOTHILLS_TRAILS = {
    "8th Street Motorcycle Trail",
    "Bob Smylie",
    "Central Ridge Trail",
    "Connection (Eagle Ridge)",
    "Doe Ridge",
    "Eagle Ridge Trail",
    "Full Sail Trail",
    "Gold Finch",
    "Harrison Hollow",
    "Harrison Ridge",
    "Hippie Shake Trail",
    "Hull's Gulch Interpretive",
    "Kemper's Ridge Trail",
    "Kestral Trail",
    "Lower Hull's Gulch Trail",
    "Mountain Cove",
    "Owl's Roost",
    "Polecat Loop",
    "Quarry Trail - Castle Rock",
    "Red Cliffs",
    "Table Rock Trail",
    "Tram Trail",
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def route_trail_names(route: dict[str, Any]) -> set[str]:
    return {str(name) for name in route.get("trail_names") or []}


def evaluate_route_preflight(
    route: dict[str, Any],
    run_date: str,
    start_time: str,
) -> dict[str, Any]:
    planned_date = parse_date(run_date)
    planned_start = parse_time(start_time)
    trails = route_trail_names(route)
    blocking_reasons: list[str] = []
    manual_checks_required = ["current_r2r_conditions"]
    warnings: list[str] = []

    if "Lower Hull's Gulch Trail" in trails and planned_date.day % 2 == 1:
        blocking_reasons.append("lower_hulls_odd_day_closed_to_foot")
    if "Polecat Loop" in trails:
        manual_checks_required.append("polecat_current_directional_signage")
    if "Around the Mountain Trail" in trails or "Around the Mountain" in trails:
        manual_checks_required.append("around_the_mountain_current_directional_signage")
    if "Bucktail Trail" in trails or "Bucktail" in trails:
        manual_checks_required.append("bucktail_pedestrian_accommodation")
    if trails & LOW_FOOTHILLS_TRAILS and planned_start >= time(10, 0):
        warnings.append("late_start_low_foothills_heat_risk")

    return {
        "route_id": route.get("id") or route.get("combo_id") or route.get("candidate_id"),
        "recommendation_type": route.get("recommendation_type"),
        "trail_names": sorted(trails),
        "run_date": run_date,
        "start_time": start_time,
        "field_status": "blocked" if blocking_reasons else "needs_day_of_check",
        "blocking_reasons": blocking_reasons,
        "manual_checks_required": sorted(set(manual_checks_required)),
        "warnings": warnings,
        "notes": [
            "Static rules are not a substitute for current Ridge to Rivers conditions, closures, and signage.",
        ],
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def selected_recommendations(execution: dict[str, Any], menu: str) -> dict[str, Any]:
    if menu == "single-car":
        return execution["single_car_menu"]["recommended_by_bucket"]
    if menu == "best-executable":
        return execution["best_executable_menu"]["recommended_by_bucket"]
    raise ValueError(f"Unsupported menu: {menu}")


def build_preflight(
    execution: dict[str, Any],
    run_date: str,
    start_time: str,
    menu: str,
) -> dict[str, Any]:
    recommendations = selected_recommendations(execution, menu)
    route_checks = {
        bucket: evaluate_route_preflight(route, run_date, start_time)
        for bucket, route in recommendations.items()
    }
    return {
        "run_date": run_date,
        "start_time": start_time,
        "menu": menu,
        "route_checks": route_checks,
        "summary": {
            "blocked": sum(1 for check in route_checks.values() if check["field_status"] == "blocked"),
            "needs_day_of_check": sum(
                1 for check in route_checks.values() if check["field_status"] == "needs_day_of_check"
            ),
            "warnings": sum(len(check["warnings"]) for check in route_checks.values()),
        },
        "dynamic_sources_to_check": [
            "Ridge to Rivers current conditions / interactive map",
            "Ridge to Rivers trail news / closures",
            "ACHD RITA roadwork if driving to trailheads",
            "Forecast heat/smoke/AQI for planned start time",
        ],
    }


def render_markdown(preflight: dict[str, Any]) -> str:
    lines = [
        "# Day-Of Route Preflight",
        "",
        f"- Run date: {preflight['run_date']}",
        f"- Start time: {preflight['start_time']}",
        f"- Menu: {preflight['menu']}",
        f"- Blocked by static rules: {preflight['summary']['blocked']}",
        f"- Needs day-of checks: {preflight['summary']['needs_day_of_check']}",
        "",
        "| Bucket | Status | Route | Blocking reasons | Manual checks | Warnings |",
        "|---|---|---|---|---|---|",
    ]
    for bucket, check in preflight["route_checks"].items():
        lines.append(
            f"| {bucket} "
            f"| {check['field_status']} "
            f"| {', '.join(check['trail_names'])} "
            f"| {', '.join(check['blocking_reasons']) or 'none'} "
            f"| {', '.join(check['manual_checks_required']) or 'none'} "
            f"| {', '.join(check['warnings']) or 'none'} |"
        )
    lines.extend(["", "## Dynamic Sources To Check", ""])
    for source in preflight["dynamic_sources_to_check"]:
        lines.append(f"- {source}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-json", type=Path, default=DEFAULT_EXECUTION_JSON)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--start-time", default="07:00")
    parser.add_argument("--menu", choices=["single-car", "best-executable"], default="single-car")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execution = load_json(args.execution_json)
    preflight = build_preflight(execution, args.run_date, args.start_time, args.menu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run_date}-{args.start_time.replace(':', '')}-{args.menu}"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(preflight, indent=2) + "\n")
    md_path.write_text(render_markdown(preflight))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        "Preflight: "
        f"{preflight['summary']['blocked']} blocked, "
        f"{preflight['summary']['needs_day_of_check']} need day-of checks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
