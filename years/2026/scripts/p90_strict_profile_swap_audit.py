#!/usr/bin/env python3
"""Measure strict-profile opportunity cost for forcing missing segments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_joint_field_day_optimizer as joint_optimizer  # noqa: E402
import p90_strict_profile_gap_recovery_targets as gap_targets  # noqa: E402
from p90_completion_gap_analyzer import official_segments  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_OPTIMIZER_JSON = gap_targets.DEFAULT_OPTIMIZER_JSON
DEFAULT_GAP_JSON = YEAR_DIR / "checkpoints" / "p90-strict-profile-gap-recovery-targets-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-strict-profile-swap-audit-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def covered_ids(target_ids: list[int], solution: dict[str, Any]) -> set[int]:
    return set(target_ids) - {int(seg_id) for seg_id in solution.get("missing_segment_ids") or []}


def classify_forced_swap(
    *,
    forced_solution: dict[str, Any],
    baseline_solution: dict[str, Any],
    newly_recovered: set[int],
    lost_previously_covered: set[int],
) -> str:
    if not forced_solution.get("success"):
        if forced_solution.get("reason") == "required_segments_not_coverable_by_field_day_candidates":
            return "no_strict_field_day_candidate"
        return "force_infeasible"
    if not newly_recovered:
        return "force_did_not_recover_segment"
    delta = int(forced_solution.get("covered_segment_count") or 0) - int(
        baseline_solution.get("covered_segment_count") or 0
    )
    if delta > 0:
        return "improves_max_coverage"
    if delta == 0:
        if lost_previously_covered:
            return "one_for_one_swap"
        return "recovers_without_coverage_loss"
    return "coverage_loss_swap"


def build_swap_row(
    seg_id: int,
    *,
    official_row: dict[str, Any],
    target_ids: list[int],
    baseline_solution: dict[str, Any],
    forced_solution: dict[str, Any],
    option_count: int,
) -> dict[str, Any]:
    baseline_covered = covered_ids(target_ids, baseline_solution)
    if forced_solution.get("success"):
        forced_covered = covered_ids(target_ids, forced_solution)
        newly_recovered = forced_covered - baseline_covered
        lost_previously_covered = baseline_covered - forced_covered
        p75_delta = int(forced_solution.get("total_p75_minutes") or 0) - int(
            baseline_solution.get("total_p75_minutes") or 0
        )
        covered_delta = int(forced_solution.get("covered_segment_count") or 0) - int(
            baseline_solution.get("covered_segment_count") or 0
        )
    else:
        forced_covered = set()
        newly_recovered = set()
        lost_previously_covered = set()
        p75_delta = None
        covered_delta = None
    classification = classify_forced_swap(
        forced_solution=forced_solution,
        baseline_solution=baseline_solution,
        newly_recovered=newly_recovered,
        lost_previously_covered=lost_previously_covered,
    )
    return {
        **official_row,
        "forced_segment_id": int(seg_id),
        "strict_option_count": int(option_count),
        "force_success": bool(forced_solution.get("success")),
        "force_failure_reason": forced_solution.get("reason") or forced_solution.get("message"),
        "classification": classification,
        "forced_covered_segment_count": forced_solution.get("covered_segment_count"),
        "covered_segment_delta": covered_delta,
        "p75_delta_minutes": p75_delta,
        "forced_total_p75_minutes": forced_solution.get("total_p75_minutes"),
        "newly_recovered_missing_segments": sorted(newly_recovered),
        "lost_previously_covered_segments": sorted(lost_previously_covered),
        "lost_previously_covered_count": len(lost_previously_covered),
        "forced_missing_segment_ids": forced_solution.get("missing_segment_ids") or [],
    }


def official_segment_rows() -> tuple[list[int], dict[int, dict[str, Any]], dict[int, float]]:
    segment_index = official_segments(read_json(DEFAULT_OFFICIAL_GEOJSON))
    target_ids = sorted(int(seg_id) for seg_id in segment_index)
    rows = {}
    miles = {}
    for seg_id in target_ids:
        segment = segment_index[seg_id]
        official_miles = round(float(segment["official_miles"]), 2)
        rows[seg_id] = {
            "seg_id": seg_id,
            "seg_name": segment["seg_name"],
            "trail_name": segment["trail_name"],
            "official_miles": official_miles,
        }
        miles[seg_id] = float(segment["official_miles"])
    return target_ids, rows, miles


def sort_rows_for_review(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classification_rank = {
        "recovers_without_coverage_loss": 0,
        "one_for_one_swap": 1,
        "coverage_loss_swap": 2,
        "force_infeasible": 3,
        "no_strict_field_day_candidate": 4,
    }
    return sorted(
        rows,
        key=lambda row: (
            classification_rank.get(str(row["classification"]), 99),
            -(row.get("covered_segment_delta") or -999),
            row.get("p75_delta_minutes") if row.get("p75_delta_minutes") is not None else 999999,
            row.get("lost_previously_covered_count") or 0,
            row["seg_id"],
        ),
    )


def build_report(optimizer: dict[str, Any], gap_report: dict[str, Any]) -> dict[str, Any]:
    scenario = gap_targets.select_strict_scenario(optimizer)
    baseline = scenario["max_coverage_solution"]
    target_ids, official_rows, official_miles_by_segment = official_segment_rows()
    field_days = gap_targets.strict_field_day_candidates(
        weekday_bound=int(scenario["weekday_bound_minutes"]),
        weekend_bound=int(scenario["weekend_bound_minutes"]),
        max_combo_size=int((optimizer.get("config") or {}).get("max_combo_size") or 3),
        neighbor_limit=int((optimizer.get("config") or {}).get("neighbor_limit") or 40),
    )
    gap_rows = {int(row["seg_id"]): row for row in gap_report.get("rows") or []}
    swap_rows = []
    for seg_id in [int(seg_id) for seg_id in baseline.get("missing_segment_ids") or []]:
        forced = joint_optimizer.solve_direct_field_day_max_coverage(
            field_days,
            target_ids,
            scenario["date_counts"],
            official_miles_by_segment=official_miles_by_segment,
            required_segment_ids=[seg_id],
        )
        swap_rows.append(
            build_swap_row(
                seg_id,
                official_row=official_rows[seg_id],
                target_ids=target_ids,
                baseline_solution=baseline,
                forced_solution=forced,
                option_count=int((gap_rows.get(seg_id) or {}).get("candidate_option_count") or 0),
            )
        )
    sorted_rows = sort_rows_for_review(swap_rows)
    counts = Counter(row["classification"] for row in swap_rows)
    return {
        "objective": "measure opportunity cost of forcing each strict-profile missing segment into the 31-day max-coverage schedule",
        "source_files": {
            "joint_optimizer_json": display_path(DEFAULT_OPTIMIZER_JSON),
            "gap_recovery_json": display_path(DEFAULT_GAP_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
        },
        "profile": {
            "scenario": scenario["scenario"],
            "weekday_bound_minutes": scenario["weekday_bound_minutes"],
            "weekend_bound_minutes": scenario["weekend_bound_minutes"],
            "weekday_count": scenario["date_counts"]["weekday"],
            "weekend_count": scenario["date_counts"]["weekend"],
        },
        "baseline": {
            "covered_segment_count": baseline["covered_segment_count"],
            "missing_segment_count": baseline["missing_segment_count"],
            "covered_official_miles": baseline["covered_official_miles"],
            "missing_official_miles": baseline["missing_official_miles"],
            "field_day_count": baseline["field_day_count"],
            "total_p75_minutes": baseline["total_p75_minutes"],
        },
        "summary": {
            "forced_segment_count": len(swap_rows),
            "classification_counts": dict(sorted(counts.items())),
            "best_review_rows": sorted_rows[:12],
        },
        "rows": sorted_rows,
        "interpretation": [
            "one_for_one_swap means the segment can be forced without reducing total segment count, but another currently covered segment drops out.",
            "coverage_loss_swap means forcing the segment lowers total strict-profile max coverage and needs better grouping or a larger bound.",
            "no_strict_field_day_candidate means route/access/time redesign is needed before the segment can even compete in the strict schedule.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    baseline = report["baseline"]
    profile = report["profile"]
    lines = [
        "# Strict Profile Swap Audit",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Scenario: `{profile['scenario']}`",
        f"- Bounds: {profile['weekday_bound_minutes']} weekday / {profile['weekend_bound_minutes']} weekend",
        f"- Day counts: {profile['weekday_count']} weekday / {profile['weekend_count']} weekend",
        f"- Baseline max coverage: {baseline['covered_segment_count']} covered / {baseline['missing_segment_count']} missing",
        f"- Baseline official miles: {baseline['covered_official_miles']} covered / {baseline['missing_official_miles']} missing",
        f"- Forced missing segments tested: {summary['forced_segment_count']}",
        f"- Classification counts: `{summary['classification_counts']}`",
        "",
        "## Best Review Rows",
        "",
        "| Segment | Trail | Classification | Delta covered | Delta p75 | Recovers | Loses |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for row in summary["best_review_rows"]:
        lines.append(
            f"| {row['seg_id']} | {row['seg_name']} | {row['classification']} | "
            f"{row.get('covered_segment_delta') if row.get('covered_segment_delta') is not None else ''} | "
            f"{row.get('p75_delta_minutes') if row.get('p75_delta_minutes') is not None else ''} | "
            f"{', '.join(str(seg_id) for seg_id in row['newly_recovered_missing_segments'])} | "
            f"{', '.join(str(seg_id) for seg_id in row['lost_previously_covered_segments'][:6])} |"
        )
    lines.extend(
        [
            "",
            "## All Forced-Segment Rows",
            "",
            "| Segment | Trail | Official mi | Options | Classification | Delta covered | Delta p75 | Lost covered count |",
            "|---:|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            f"| {row['seg_id']} | {row['seg_name']} | {row['official_miles']} | "
            f"{row['strict_option_count']} | {row['classification']} | "
            f"{row.get('covered_segment_delta') if row.get('covered_segment_delta') is not None else ''} | "
            f"{row.get('p75_delta_minutes') if row.get('p75_delta_minutes') is not None else ''} | "
            f"{row['lost_previously_covered_count']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-json", type=Path, default=DEFAULT_OPTIMIZER_JSON)
    parser.add_argument("--gap-json", type=Path, default=DEFAULT_GAP_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(read_json(args.optimizer_json), read_json(args.gap_json))
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["summary"], indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
