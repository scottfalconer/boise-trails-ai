#!/usr/bin/env python3
"""Summarize the current decision gate blocking the strict p90 completion proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_COMPLETION_AUDIT_JSON = YEAR_DIR / "checkpoints" / "field-day-p90-completion-audit-2026-05-06.json"
DEFAULT_SHINGLE_PROBE_JSON = YEAR_DIR / "checkpoints" / "p90-shingle-1656-anchor-exhaustive-probe-2026-05-06.json"
DEFAULT_PROFILE_ACCEPTANCE_JSON = YEAR_DIR / "checkpoints" / "p90-profile-acceptance-audit-2026-05-06.json"
DEFAULT_RELAXED_DRAFT_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-draft-field-day-plan-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-completion-decision-gate-2026-05-06"


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


def strict_status(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_strict_completion_passed": False,
        "strict_max_coverage_segments": int(metrics["wide_joint_optimizer_strict_max_coverage_segments"]),
        "strict_missing_segments": int(metrics["wide_joint_optimizer_strict_max_coverage_missing_segments"]),
        "strict_missing_official_miles": round(float(metrics["wide_joint_optimizer_strict_max_coverage_missing_segments"]), 2)
        if "wide_joint_optimizer_strict_max_coverage_missing_official_miles" in metrics
        else float(metrics.get("strict_profile_max_coverage_plan_missing_official_miles") or 0.0),
        "strict_blocker_segment_ids": list(metrics.get("wide_joint_optimizer_strict_missing_segment_ids") or []),
    }


def shingle_status(shingle_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": 1656,
        "strict_success": bool(shingle_summary["strict_success"]),
        "anchors_tested": int(shingle_summary["anchor_count"]),
        "under_bound_graph_track_rows": int(shingle_summary["under_bound_graph_track_count"]),
        "best_field_ready_anchor": shingle_summary["best_field_ready_anchor"],
        "best_field_ready_p90_minutes": int(shingle_summary["best_field_ready_p90_minutes"]),
        "best_field_ready_p75_minutes": int(shingle_summary["best_field_ready_p75_minutes"]),
        "minutes_over_active_bound": int(shingle_summary["minutes_over_bound_best_field_ready"]),
    }


def relaxed_status(
    metrics: dict[str, Any],
    profile_summary: dict[str, Any],
    relaxed_draft: dict[str, Any],
) -> dict[str, Any]:
    config = relaxed_draft.get("config") or {}
    summary = relaxed_draft.get("summary") or {}
    return {
        "full_clear_draft_exists": bool(metrics.get("near_miss_relaxed_drive_45_neighbor_40_feasible")),
        "accepted_as_active_personal_plan": bool(profile_summary["accepted_as_active_personal_plan"]),
        "candidate_weekday_bound_minutes": int(config.get("weekday_bound_minutes") or 0),
        "candidate_weekend_bound_minutes": int(config.get("weekend_bound_minutes") or 0),
        "candidate_inter_trailhead_drive_minutes": int(config.get("inter_trailhead_drive_minutes") or 0),
        "field_days": int(summary.get("field_day_count") or metrics.get("near_miss_relaxed_drive_45_neighbor_40_field_days") or 0),
        "total_p75_minutes": int(summary.get("total_p75_minutes") or metrics.get("near_miss_relaxed_drive_45_neighbor_40_total_p75_minutes") or 0),
        "day_level_gpx_validated": bool(metrics.get("near_miss_relaxed_drive_draft_day_level_gpx_validated")),
        "current_bound_p90_violation_count": int(profile_summary["current_bound_p90_violation_count"]),
        "inter_trailhead_drive_violation_count": int(profile_summary["inter_trailhead_drive_violation_count"]),
        "max_minutes_over_current_p90_bound": int(profile_summary["max_minutes_over_current_p90_bound"]),
        "max_minutes_over_current_inter_trailhead_drive_bound": int(
            profile_summary["max_minutes_over_current_inter_trailhead_drive_bound"]
        ),
    }


def decision_options(metrics: dict[str, Any], shingle: dict[str, Any], relaxed: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "option": "keep_active_strict_bounds",
            "completion_plan_available": False,
            "result": (
                f"Best current strict schedule covers {metrics['wide_joint_optimizer_strict_max_coverage_segments']}/251 "
                f"segments; Shingle 1656 has no under-bound anchor."
            ),
            "when_to_choose": "Use this if the 260 weekday / 180 weekend p90 profile is non-negotiable.",
            "next_work": "Treat the plan as adaptive partial-coverage until route redesign or field calibration changes the Shingle/time evidence.",
        },
        {
            "option": "accept_shingle_only_292_exception",
            "completion_plan_available": False,
            "result": (
                "Not enough by itself. The non-compliant Shingle-292 scenario still fails field-day packing "
                f"and needs at least {metrics['wide_joint_optimizer_shingle_292_relaxed_min_field_days']} field days "
                f"or {metrics['wide_joint_optimizer_shingle_292_min_weekdays_with_9_weekends']} weekdays with 9 weekends."
            ),
            "when_to_choose": "Only useful as one ingredient if the rest of the schedule is also relaxed or redesigned.",
            "next_work": "Continue route consolidation around weekday-only pressure.",
        },
        {
            "option": "accept_relaxed_292_360_drive45_profile",
            "completion_plan_available": bool(relaxed["full_clear_draft_exists"]),
            "result": (
                f"Existing draft covers 251/251 in {relaxed['field_days']} days with {relaxed['total_p75_minutes']} p75 minutes, "
                f"but violates current strict bounds on {relaxed['current_bound_p90_violation_count']} days."
            ),
            "when_to_choose": "Use this if 6-hour weekends and the wider inter-start drive allowance are realistic enough for the challenge.",
            "next_work": "Promote this profile into the active personal state, then rerun calendar/GPX/field-packet generation from that canonical profile.",
        },
        {
            "option": "field_calibrate_or_redesign_shingle",
            "completion_plan_available": False,
            "result": (
                f"Current best Shingle row is {shingle['best_field_ready_p90_minutes']} p90, "
                f"{shingle['minutes_over_active_bound']} minutes over the active bound."
            ),
            "when_to_choose": "Use this if the profile should stay strict but Shingle may be faster in real life than modeled.",
            "next_work": "Field-test or manually validate a shorter legal Shingle connector/return; do not mark complete without new evidence.",
        },
    ]


def build_report(
    *,
    completion_audit: dict[str, Any],
    shingle_probe: dict[str, Any],
    profile_acceptance: dict[str, Any],
    relaxed_draft: dict[str, Any],
) -> dict[str, Any]:
    metrics = completion_audit["metrics"]
    shingle = shingle_status(shingle_probe["summary"])
    relaxed = relaxed_status(metrics, profile_acceptance["summary"], relaxed_draft)
    strict = strict_status(metrics)
    return {
        "objective": "state the active completion-plan decision gate with current evidence",
        "source_files": {
            "completion_audit_json": display_path(DEFAULT_COMPLETION_AUDIT_JSON),
            "shingle_probe_json": display_path(DEFAULT_SHINGLE_PROBE_JSON),
            "profile_acceptance_json": display_path(DEFAULT_PROFILE_ACCEPTANCE_JSON),
            "relaxed_draft_json": display_path(DEFAULT_RELAXED_DRAFT_JSON),
        },
        "verdict": "decision_required_not_complete",
        "strict_status": strict,
        "shingle_status": shingle,
        "relaxed_completion_draft_status": relaxed,
        "decision_options": decision_options(metrics, shingle, relaxed),
        "recommended_next_step": (
            "Ask the user to choose whether the active profile remains strict or whether to promote the "
            "292 weekday / 360 weekend / 45-minute inter-start-drive profile for the 2026 completion plan."
        ),
    }


def render_md(report: dict[str, Any]) -> str:
    strict = report["strict_status"]
    shingle = report["shingle_status"]
    relaxed = report["relaxed_completion_draft_status"]
    lines = [
        "# P90 Completion Decision Gate",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Strict completion passed: {strict['active_strict_completion_passed']}",
        f"- Strict max coverage: {strict['strict_max_coverage_segments']}/251",
        f"- Strict missing segments: {strict['strict_missing_segments']}",
        "",
        "## Shingle Gate",
        "",
        f"- Anchors tested: {shingle['anchors_tested']}",
        f"- Under-bound graph/track-valid rows: {shingle['under_bound_graph_track_rows']}",
        f"- Best field-ready anchor: {shingle['best_field_ready_anchor']}",
        f"- Best field-ready p90/p75: {shingle['best_field_ready_p90_minutes']} / {shingle['best_field_ready_p75_minutes']} min",
        f"- Minutes over active bound: {shingle['minutes_over_active_bound']}",
        "",
        "## Existing Full-Clear Draft",
        "",
        f"- Full-clear draft exists: {relaxed['full_clear_draft_exists']}",
        f"- Accepted as active plan: {relaxed['accepted_as_active_personal_plan']}",
        f"- Candidate profile: {relaxed['candidate_weekday_bound_minutes']} weekday / {relaxed['candidate_weekend_bound_minutes']} weekend / {relaxed['candidate_inter_trailhead_drive_minutes']} min inter-start drive",
        f"- Field days: {relaxed['field_days']}",
        f"- Total p75: {relaxed['total_p75_minutes']} min",
        f"- Current-bound p90 violations: {relaxed['current_bound_p90_violation_count']}",
        "",
        "## Decision Options",
        "",
        "| Option | Completion available | Result | Next work |",
        "|---|---|---|---|",
    ]
    for option in report["decision_options"]:
        lines.append(
            f"| `{option['option']}` | {option['completion_plan_available']} | "
            f"{option['result']} | {option['next_work']} |"
        )
    lines.extend(["", "## Recommended Next Step", "", report["recommended_next_step"], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-audit-json", type=Path, default=DEFAULT_COMPLETION_AUDIT_JSON)
    parser.add_argument("--shingle-probe-json", type=Path, default=DEFAULT_SHINGLE_PROBE_JSON)
    parser.add_argument("--profile-acceptance-json", type=Path, default=DEFAULT_PROFILE_ACCEPTANCE_JSON)
    parser.add_argument("--relaxed-draft-json", type=Path, default=DEFAULT_RELAXED_DRAFT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        completion_audit=read_json(args.completion_audit_json),
        shingle_probe=read_json(args.shingle_probe_json),
        profile_acceptance=read_json(args.profile_acceptance_json),
        relaxed_draft=read_json(args.relaxed_draft_json),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps({"verdict": report["verdict"], "recommended_next_step": report["recommended_next_step"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
