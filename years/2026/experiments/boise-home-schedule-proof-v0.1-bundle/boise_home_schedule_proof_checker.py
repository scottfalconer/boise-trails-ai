#!/usr/bin/env python3
"""Independent checker skeleton for Boise Trails Challenge home-to-home proof instances.

This checker intentionally separates policy/instance validation from optimization. It can:

1. verify that frozen input file hashes still match the proof instance,
2. verify that required private parameters are present when provided,
3. validate a proposed selected-home-schedule.json against the formal home-to-home
   / single-car / p90-bounds / coverage policy.

It does not prove optimality by itself; it verifies feasibility and objective totals.
The exact optimizer must provide the lower-bound / zero-gap certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


BOUND_FIELD_TO_DAY_METRIC = {
    "max_daily_on_foot_miles": "daily_on_foot_miles",
    "max_daily_grade_adjusted_miles": "daily_grade_adjusted_miles",
    "max_daily_ascent_ft": "daily_ascent_ft",
    "max_daily_moving_p90_minutes": "daily_moving_p90_minutes",
    "max_daily_door_to_door_p90_minutes": "daily_door_to_door_p90_minutes",
    "max_parking_starts_per_day": "daily_parking_starts",
    "max_run_loops_per_day": "daily_run_loops",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def issue(severity: str, code: str, message: str, path: str = "") -> Dict[str, str]:
    return {"severity": severity, "code": code, "path": path, "message": message}


def required_segment_ids(instance: Dict[str, Any]) -> Set[int]:
    ids = instance.get("sets", {}).get("T", {}).get("required_segment_ids", [])
    return {int(x) for x in ids}


def personal_bounds(instance: Dict[str, Any], private_params: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    bounds = dict(instance.get("sets", {}).get("B", {}).get("values", {}) or {})
    if private_params:
        bounds.update(private_params.get("personal_daily_bounds", {}) or {})
    return bounds


def verify_manifest_hashes(instance: Dict[str, Any], base_dir: Path) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    files = instance.get("data_manifest", {}).get("input_files", {})
    for key, meta in files.items():
        rel = meta.get("path")
        expected = meta.get("sha256")
        if not rel or not expected:
            issues.append(issue("error", "manifest_missing_path_or_hash", f"Manifest entry {key!r} lacks path or sha256.", f"data_manifest.input_files.{key}"))
            continue
        path = base_dir / rel
        if not path.exists():
            issues.append(issue("error", "manifest_file_missing", f"Expected input file does not exist: {path}", f"data_manifest.input_files.{key}.path"))
            continue
        actual = sha256_file(path)
        if actual != expected:
            issues.append(issue("error", "manifest_hash_mismatch", f"Hash mismatch for {rel}: expected {expected}, actual {actual}", f"data_manifest.input_files.{key}.sha256"))
    return issues


def validate_instance(instance: Dict[str, Any], private_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    if instance.get("schema_version") != "boise_home_schedule_proof_instance_v0.1":
        issues.append(issue("error", "bad_schema_version", "Unexpected or missing schema_version.", "schema_version"))

    t = instance.get("sets", {}).get("T", {})
    if not t.get("required_segment_ids"):
        issues.append(issue("error", "missing_required_segment_ids", "Official segment set T has no required_segment_ids.", "sets.T.required_segment_ids"))

    h = instance.get("sets", {}).get("H", {})
    private_home = (private_params or {}).get("home_vertex_id") or h.get("value")
    if h.get("required") and not private_home:
        issues.append(issue("warning", "home_vertex_required", "Home vertex is required for optimization but is intentionally absent from public instance.", "sets.H"))

    bounds = personal_bounds(instance, private_params)
    null_bounds = [k for k, v in bounds.items() if v is None]
    if null_bounds:
        issues.append(issue("warning", "daily_bounds_required", "Personal daily bounds are required before proving a final home schedule: " + ", ".join(null_bounds), "sets.B.values"))

    objective = instance.get("objective", {})
    if objective.get("primary") != "total_p75_home_to_home_minutes":
        issues.append(issue("error", "bad_primary_objective", "Primary objective must be total_p75_home_to_home_minutes.", "objective.primary"))
    return issues


def validate_schedule(instance: Dict[str, Any], schedule: Dict[str, Any], private_params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    issues: List[Dict[str, str]] = []
    derived: Dict[str, Any] = {
        "covered_segment_ids": [],
        "missing_segment_ids": [],
        "totals": {
            "total_p75_home_to_home_minutes": 0.0,
            "total_p90_home_to_home_minutes": 0.0,
            "total_grade_adjusted_miles": 0.0,
            "total_on_foot_miles": 0.0,
            "field_day_count": 0,
            "total_parking_risk_score": 0.0,
            "max_daily_stress_ratio": 0.0,
        },
    }
    bounds = personal_bounds(instance, private_params)
    required_ids = required_segment_ids(instance)
    covered_ids: Set[int] = set()

    field_days = schedule.get("field_days", [])
    if not isinstance(field_days, list) or not field_days:
        issues.append(issue("error", "no_field_days", "Schedule must contain a non-empty field_days list.", "field_days"))
        return issues, derived

    for d_idx, day in enumerate(field_days):
        d_path = f"field_days[{d_idx}]"
        derived["totals"]["field_day_count"] += 1

        if day.get("starts_at") != "home" or day.get("ends_at") != "home":
            issues.append(issue("error", "field_day_not_home_to_home", "Every field day must start at home and end at home.", d_path))

        loops = day.get("run_loops", [])
        if not loops:
            issues.append(issue("error", "field_day_has_no_run_loops", "Every selected field day must include at least one run loop.", f"{d_path}.run_loops"))

        parking_starts = set()
        for l_idx, loop in enumerate(loops):
            l_path = f"{d_path}.run_loops[{l_idx}]"
            start = loop.get("start_parking_id")
            end = loop.get("end_parking_id")
            if not start or not end or start != end:
                issues.append(issue("error", "run_loop_not_single_car_loop", "Run loop must start and end at the same parking id.", l_path))
            if start:
                parking_starts.add(start)

            if not loop.get("legal_edges_only", False):
                issues.append(issue("error", "run_loop_uses_unverified_edges", "Run loop must assert legal_edges_only=true after ledger validation.", f"{l_path}.legal_edges_only"))
            if not loop.get("gpx_continuity_passed", False):
                issues.append(issue("error", "run_loop_gpx_not_continuous", "Run loop must pass GPX continuity.", f"{l_path}.gpx_continuity_passed"))
            if not loop.get("direction_rules_passed", False):
                issues.append(issue("error", "run_loop_direction_rules_not_verified", "Run loop must pass official direction rules.", f"{l_path}.direction_rules_passed"))

            for sid in loop.get("covered_segment_ids", []) or []:
                covered_ids.add(int(sid))

        metrics = day.get("metrics", {})
        # Sum declared metrics. If absent, checker records an error rather than guessing.
        for total_key, metric_key in [
            ("total_p75_home_to_home_minutes", "daily_door_to_door_p75_minutes"),
            ("total_p90_home_to_home_minutes", "daily_door_to_door_p90_minutes"),
            ("total_grade_adjusted_miles", "daily_grade_adjusted_miles"),
            ("total_on_foot_miles", "daily_on_foot_miles"),
            ("total_parking_risk_score", "daily_parking_risk_score"),
        ]:
            value = metrics.get(metric_key)
            if value is None:
                issues.append(issue("error", "missing_day_metric", f"Missing required day metric {metric_key}.", f"{d_path}.metrics.{metric_key}"))
            else:
                derived["totals"][total_key] += float(value)

        # p90 bounds and daily stress ratio.
        day_stress = 0.0
        for bound_key, metric_key in BOUND_FIELD_TO_DAY_METRIC.items():
            bound = bounds.get(bound_key)
            if bound is None:
                continue
            metric = metrics.get(metric_key)
            if metric is None:
                issues.append(issue("error", "missing_bound_metric", f"Daily bound {bound_key} requires metric {metric_key}.", f"{d_path}.metrics.{metric_key}"))
                continue
            metric_float = float(metric)
            bound_float = float(bound)
            if metric_float > bound_float + 1e-9:
                issues.append(issue("error", "daily_bound_exceeded", f"{metric_key}={metric_float} exceeds {bound_key}={bound_float}.", f"{d_path}.metrics.{metric_key}"))
            if bound_float > 0:
                day_stress = max(day_stress, metric_float / bound_float)
        derived["totals"]["max_daily_stress_ratio"] = max(derived["totals"]["max_daily_stress_ratio"], day_stress)

        # Optional sanity check: parking starts metric should match actual loop starts if provided.
        if metrics.get("daily_parking_starts") is not None and int(metrics["daily_parking_starts"]) != len(parking_starts):
            issues.append(issue("warning", "parking_start_count_mismatch", f"daily_parking_starts={metrics['daily_parking_starts']} but loops use {len(parking_starts)} unique starts.", f"{d_path}.metrics.daily_parking_starts"))

    missing = sorted(required_ids - covered_ids)
    extra = sorted(covered_ids - required_ids)
    derived["covered_segment_ids"] = sorted(covered_ids)
    derived["missing_segment_ids"] = missing
    derived["extra_segment_ids"] = extra
    if missing:
        issues.append(issue("error", "official_segments_missing", f"Schedule is missing {len(missing)} required official segments.", "field_days"))
    if extra:
        issues.append(issue("warning", "extra_nonrequired_segments", f"Schedule reports {len(extra)} segment ids not in T.", "field_days"))

    # Compare declared schedule objective if present.
    declared = schedule.get("objective_totals", {})
    for key, actual in derived["totals"].items():
        if key in declared and not math.isclose(float(declared[key]), float(actual), rel_tol=1e-6, abs_tol=1e-6):
            issues.append(issue("error", "objective_total_mismatch", f"Declared {key}={declared[key]} but derived {actual}.", f"objective_totals.{key}"))

    return issues, derived


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Boise home-to-home proof instance and optional schedule.")
    parser.add_argument("instance", type=Path, help="boise-home-schedule-proof-instance-v0.1.json")
    parser.add_argument("--base-dir", type=Path, default=None, help="Directory containing frozen input files. Defaults to instance directory.")
    parser.add_argument("--private-params", type=Path, default=None, help="Optional private JSON with home_vertex_id and personal_daily_bounds.")
    parser.add_argument("--schedule", type=Path, default=None, help="Optional selected-home-schedule.json to validate.")
    args = parser.parse_args()

    instance = load_json(args.instance)
    base_dir = args.base_dir or args.instance.parent
    private_params = load_json(args.private_params) if args.private_params else None

    issues = []
    issues.extend(validate_instance(instance, private_params))
    issues.extend(verify_manifest_hashes(instance, base_dir))

    derived = None
    if args.schedule:
        schedule = load_json(args.schedule)
        sched_issues, derived = validate_schedule(instance, schedule, private_params)
        issues.extend(sched_issues)

    error_count = sum(1 for x in issues if x["severity"] == "error")
    warning_count = sum(1 for x in issues if x["severity"] == "warning")
    report = {
        "status": "passed" if error_count == 0 else "failed",
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "derived_schedule_totals": derived,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
