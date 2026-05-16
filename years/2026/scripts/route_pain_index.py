#!/usr/bin/env python3
"""Rank current field-guide routes by real human cost and optimization leverage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402


DEFAULT_FIELD_TOOL_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_MULTI_START_AUDIT_JSON = YEAR_DIR / "checkpoints" / "multi-start-alternative-audit-2026-05-08.json"
DEFAULT_COMPLETION_AUDIT_JSON = YEAR_DIR / "checkpoints" / "field-tool-completion-audit-2026-05-06.json"
DEFAULT_ACCESS_VERIFICATION_JSON = YEAR_DIR / "checkpoints" / "10a-ms-08-access-verification-2026-05-10.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "route-pain-index-2026-05-10.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "route-pain-index-2026-05-10.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "route-pain-index-2026-05-10-manifest.json"

BASE_LABEL_RE = re.compile(r"^(\d+[A-Z]?)(?:-\d+)?$")
ACTIONABLE_ALTERNATIVE_STATUSES = {"promising", "needs_parking_check"}


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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def round2(value: float) -> float:
    return round(float(value), 2)


def base_route_label(label: Any) -> str:
    value = str(label or "").strip()
    match = BASE_LABEL_RE.match(value)
    return match.group(1) if match else value


def missing_verified_start_labels(completion_audit: dict[str, Any] | None) -> set[str]:
    missing: set[str] = set()
    for check in (completion_audit or {}).get("checks") or []:
        evidence = str(check.get("evidence") or "")
        if "missing verified parked start" not in evidence:
            continue
        for entry in evidence.split(";"):
            text = entry.strip()
            if "missing verified parked start" not in text:
                continue
            label = text.split(" ", 1)[0].strip()
            if label:
                missing.add(label)
    return missing


def active_split_bases(field_tool_payload: dict[str, Any]) -> set[str]:
    labels = [str(route.get("label") or "") for route in field_tool_payload.get("routes") or []]
    exact_labels = set(labels)
    base_counts = Counter(base_route_label(label) for label in labels if base_route_label(label) != label)
    return {base for base, count in base_counts.items() if count >= 2 and base not in exact_labels}


def best_alternatives_by_label(multi_start_audit: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for outing in (multi_start_audit or {}).get("outings") or []:
        label = str(outing.get("label") or "")
        candidates = []
        for alternative in outing.get("alternatives") or []:
            savings = as_float(alternative.get("on_foot_savings_miles"))
            if savings <= 0:
                continue
            status = str(alternative.get("status") or "")
            elapsed_delta = as_int(alternative.get("elapsed_delta_minutes"))
            candidates.append(
                {
                    "alternative_id": alternative.get("alternative_id"),
                    "status": status,
                    "on_foot_savings_miles": round2(savings),
                    "elapsed_delta_minutes": elapsed_delta,
                    "parking_blockers": list(alternative.get("parking_blockers") or []),
                }
            )
        if not candidates:
            continue
        candidates.sort(
            key=lambda alternative: (
                1 if alternative["status"] in ACTIONABLE_ALTERNATIVE_STATUSES else 0,
                alternative["on_foot_savings_miles"],
                -alternative["elapsed_delta_minutes"],
            ),
            reverse=True,
        )
        best[label] = candidates[0]
    return best


def access_decisions_by_alternative(access_verification: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for decision in (access_verification or {}).get("access_decisions") or []:
        alternative_id = str(decision.get("alternative_id") or "")
        if not alternative_id:
            continue
        decisions[alternative_id] = {
            "decision": decision.get("decision"),
            "field_certifiable": decision.get("field_certifiable"),
            "replacement_ready": decision.get("replacement_ready"),
            "next_action": decision.get("next_action"),
            "evidence_file": decision.get("evidence_file"),
        }
    return decisions


def apply_access_decisions(
    alternatives_by_label: dict[str, dict[str, Any]],
    access_decisions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    annotated: dict[str, dict[str, Any]] = {}
    for label, alternative in alternatives_by_label.items():
        copy = dict(alternative)
        decision = access_decisions.get(str(alternative.get("alternative_id") or ""))
        if decision:
            copy["access_decision"] = decision
        annotated[label] = copy
    return annotated


def route_warning_count(route: dict[str, Any]) -> int:
    total = 0
    for cue in route.get("wayfinding_cues") or []:
        if cue.get("field_warning"):
            total += 1
    return total


def max_wayfinding_leg_miles(route: dict[str, Any]) -> float:
    return max((as_float(cue.get("leg_miles")) for cue in route.get("wayfinding_cues") or []), default=0.0)


def route_pain_score(
    *,
    official_miles: float,
    on_foot_miles: float,
    p75_minutes: int,
    p90_minutes: int,
    has_car_pass: bool,
    has_known_water: bool,
    has_parking: bool,
    missing_verified_start: bool,
    warning_count: int,
    max_leg_miles: float,
) -> float:
    overhead_miles = max(on_foot_miles - official_miles, 0.0)
    score = overhead_miles * 10.0
    score += max(p75_minutes - 240, 0) / 10.0
    score += max(p90_minutes - 300, 0) / 10.0
    if not has_car_pass:
        score += 8.0
    if not has_known_water:
        score += 4.0
    if not has_parking:
        score += 12.0
    if missing_verified_start:
        score += 18.0
    score += warning_count * 2.0
    score += max(max_leg_miles - 3.0, 0.0) * 2.0
    return round2(score)


def route_optimization_status(
    *,
    route_label: str,
    base_label: str,
    active_split_base_labels: set[str],
    best_alternative: dict[str, Any] | None,
    pain_score: float,
    has_parking: bool,
    missing_verified_start: bool,
) -> str:
    if best_alternative and base_label in active_split_base_labels and route_label != base_label:
        if missing_verified_start or not has_parking:
            return "parked_start_certification"
        return "already_split_active"

    if best_alternative and (best_alternative.get("access_decision") or {}).get("field_certifiable") is False:
        return "certifiable_anchor_redesign"

    if best_alternative and best_alternative["status"] == "needs_parking_check":
        if best_alternative["on_foot_savings_miles"] >= 2.0 and best_alternative["elapsed_delta_minutes"] < 0:
            return "access_verification_sprint"

    if best_alternative and best_alternative["status"] == "promising":
        return "promote_or_recertify_split"

    if missing_verified_start or not has_parking:
        return "parked_start_certification"

    if pain_score >= 130:
        return "human_cost_research"

    return "monitor"


def action_priority(status: str, pain_score: float, best_alternative: dict[str, Any] | None) -> float:
    if status == "access_verification_sprint" and best_alternative:
        time_saved = max(-as_int(best_alternative.get("elapsed_delta_minutes")), 0)
        return round2(1000.0 + best_alternative["on_foot_savings_miles"] * 100.0 + time_saved * 2.0 + pain_score / 10.0)
    if status == "promote_or_recertify_split" and best_alternative:
        time_saved = max(-as_int(best_alternative.get("elapsed_delta_minutes")), 0)
        return round2(800.0 + best_alternative["on_foot_savings_miles"] * 100.0 + time_saved + pain_score / 10.0)
    if status == "certifiable_anchor_redesign" and best_alternative:
        time_saved = max(-as_int(best_alternative.get("elapsed_delta_minutes")), 0)
        return round2(700.0 + best_alternative["on_foot_savings_miles"] * 80.0 + time_saved + pain_score / 10.0)
    if status == "parked_start_certification":
        return round2(300.0 + pain_score)
    if status == "human_cost_research":
        return round2(200.0 + pain_score)
    return 0.0


def route_row(
    route: dict[str, Any],
    *,
    missing_start_labels: set[str],
    active_split_base_labels: set[str],
    alternative_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    label = str(route.get("label") or "")
    base_label = base_route_label(label)
    official_miles = as_float(route.get("official_miles"))
    on_foot_miles = as_float(route.get("on_foot_miles"))
    p75_minutes = as_int(route.get("door_to_door_minutes_p75"))
    p90_minutes = as_int(route.get("door_to_door_minutes_p90"))
    parking = route.get("parking") or {}
    logistics = route.get("logistics") or {}
    has_parking = parking.get("has_parking") is True
    has_car_pass = logistics.get("has_car_pass") is True
    has_known_water = logistics.get("has_known_water") is True
    missing_verified_start = label in missing_start_labels
    warning_count = route_warning_count(route)
    max_leg_miles = max_wayfinding_leg_miles(route)
    pain_score = route_pain_score(
        official_miles=official_miles,
        on_foot_miles=on_foot_miles,
        p75_minutes=p75_minutes,
        p90_minutes=p90_minutes,
        has_car_pass=has_car_pass,
        has_known_water=has_known_water,
        has_parking=has_parking,
        missing_verified_start=missing_verified_start,
        warning_count=warning_count,
        max_leg_miles=max_leg_miles,
    )
    best_alternative = alternative_index.get(base_label)
    status = route_optimization_status(
        route_label=label,
        base_label=base_label,
        active_split_base_labels=active_split_base_labels,
        best_alternative=best_alternative,
        pain_score=pain_score,
        has_parking=has_parking,
        missing_verified_start=missing_verified_start,
    )
    if status in {"already_split_active", "parked_start_certification"} and base_label in active_split_base_labels:
        best_alternative = None

    return {
        "label": label,
        "base_label": base_label,
        "trailhead": route.get("trailhead"),
        "official_miles": round2(official_miles),
        "on_foot_miles": round2(on_foot_miles),
        "overhead_miles": round2(max(on_foot_miles - official_miles, 0.0)),
        "door_to_door_minutes_p75": p75_minutes,
        "door_to_door_minutes_p90": p90_minutes,
        "has_car_pass": has_car_pass,
        "has_known_water": has_known_water,
        "has_parking": has_parking,
        "missing_verified_start": missing_verified_start,
        "wayfinding_warning_count": warning_count,
        "max_wayfinding_leg_miles": round2(max_leg_miles),
        "validation_passed": bool((route.get("validation") or {}).get("passed")),
        "pain_score": pain_score,
        "optimization_status": status,
        "action_priority": action_priority(status, pain_score, best_alternative),
        "best_alternative": best_alternative,
    }


def primary_recommendation(actionable_rows: list[dict[str, Any]], pain_rows: list[dict[str, Any]]) -> str:
    if not actionable_rows:
        return "No route-card optimization with measurable current savings surfaced from the current inputs."

    top = actionable_rows[0]
    alternative = top.get("best_alternative") or {}
    if top["optimization_status"] == "access_verification_sprint" and alternative:
        highest_pain = pain_rows[0] if pain_rows else {}
        return (
            f"Run an access-verification sprint for {top['label']} before another global reroute. "
            f"The retained {alternative.get('alternative_id')} option saves "
            f"{alternative.get('on_foot_savings_miles')} on-foot miles and "
            f"{abs(as_int(alternative.get('elapsed_delta_minutes')))} p75 minutes if parking proves legal and cueable. "
            f"The raw highest-pain route is {highest_pain.get('label')}, but the current multi-start audit found no "
            "worthwhile split alternative for that class of route."
        )

    if top["optimization_status"] == "certifiable_anchor_redesign" and alternative:
        return (
            f"Redesign {top['label']} around a certifiable parked start before another global reroute. "
            f"The best paper candidate, {alternative.get('alternative_id')}, saves "
            f"{alternative.get('on_foot_savings_miles')} on-foot miles and "
            f"{abs(as_int(alternative.get('elapsed_delta_minutes')))} p75 minutes, but access verification "
            "blocked the exact residential/probe anchors. The route-mapping optimization is now an outward "
            "search for legal, repeatable parking plus recalculated connector cost, not promotion of the paper split."
        )

    return f"Work the highest-priority current blocker: {top['label']} ({top['optimization_status']})."


def build_pain_index(
    field_tool_payload: dict[str, Any],
    *,
    multi_start_audit: dict[str, Any] | None = None,
    completion_audit: dict[str, Any] | None = None,
    access_verification: dict[str, Any] | None = None,
    generated_at: str | None = None,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    missing_start_labels = missing_verified_start_labels(completion_audit)
    active_split_base_labels = active_split_bases(field_tool_payload)
    access_decisions = access_decisions_by_alternative(access_verification)
    alternative_index = apply_access_decisions(best_alternatives_by_label(multi_start_audit), access_decisions)
    rows = [
        route_row(
            route,
            missing_start_labels=missing_start_labels,
            active_split_base_labels=active_split_base_labels,
            alternative_index=alternative_index,
        )
        for route in field_tool_payload.get("routes") or []
    ]
    rows_by_pain = sorted(rows, key=lambda row: row["pain_score"], reverse=True)
    actionable_rows = sorted(
        [row for row in rows if row["optimization_status"] not in {"monitor", "already_split_active"}],
        key=lambda row: row["action_priority"],
        reverse=True,
    )
    known_savings_rows = [
        row
        for row in rows
        if row.get("best_alternative")
        and row["optimization_status"] in {"access_verification_sprint", "promote_or_recertify_split"}
    ]
    blocked_savings_rows = [
        row
        for row in rows
        if row.get("best_alternative") and row["optimization_status"] == "certifiable_anchor_redesign"
    ]

    report = {
        "schema": "boise_trails_route_pain_index_v1",
        "generated_at": generated_at or utc_now(),
        "objective": "Rank current field-guide routes by human cost and actionable route-mapping leverage.",
        "source_files": source_files or {},
        "summary": {
            "route_count": len(rows),
            "active_split_base_count": len(active_split_base_labels),
            "missing_verified_start_count": sum(1 for row in rows if row["missing_verified_start"]),
            "access_verification_sprint_count": sum(
                1 for row in rows if row["optimization_status"] == "access_verification_sprint"
            ),
            "known_actionable_on_foot_savings_miles": round2(
                sum((row.get("best_alternative") or {}).get("on_foot_savings_miles", 0.0) for row in known_savings_rows)
            ),
            "blocked_paper_on_foot_savings_miles": round2(
                sum((row.get("best_alternative") or {}).get("on_foot_savings_miles", 0.0) for row in blocked_savings_rows)
            ),
            "top_actionable_label": actionable_rows[0]["label"] if actionable_rows else None,
            "top_pain_label": rows_by_pain[0]["label"] if rows_by_pain else None,
        },
        "primary_recommendation": primary_recommendation(actionable_rows, rows_by_pain),
        "active_split_base_labels": sorted(active_split_base_labels),
        "missing_verified_start_labels": sorted(missing_start_labels),
        "top_actionable_optimizations": actionable_rows[:10],
        "top_pain_routes": rows_by_pain[:10],
        "route_rows_by_label": {row["label"]: row for row in rows},
        "method_notes": [
            "Pain score weights excess on-foot miles, p75/p90 pressure, missing car-pass and water support, parked-start proof, wayfinding warnings, and unusually long cue legs.",
            "Promising alternatives for bases already represented by split active route cards are treated as stale savings, not rediscovered optimization opportunities.",
            "Access-verification sprint means do not replace the route card until parking/access is public, legal, repeatable, cueable, and rerun through the field-packet certification chain.",
        ],
    }
    return report


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for title, _key in columns) + " |",
        "| " + " | ".join("---" for _title, _key in columns) + " |",
    ]
    for row in rows:
        values = []
        for _title, key in columns:
            value = row
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Route Pain Index - 2026-05-10",
        "",
        "## Objective",
        "",
        report["objective"],
        "",
        "## Primary Recommendation",
        "",
        report["primary_recommendation"],
        "",
        "## Summary",
        "",
        f"- Routes ranked: {summary['route_count']}",
        f"- Active split bases treated as already-promoted savings: {summary['active_split_base_count']}",
        f"- Missing verified parked-start route cards: {summary['missing_verified_start_count']}",
        f"- Access-verification sprint candidates: {summary['access_verification_sprint_count']}",
        f"- Known actionable on-foot savings still not promoted: {summary['known_actionable_on_foot_savings_miles']} mi",
        f"- Blocked paper savings needing redesign: {summary['blocked_paper_on_foot_savings_miles']} mi",
        "",
        "## Top Actionable Optimizations",
        "",
    ]
    lines.extend(
        markdown_table(
            report["top_actionable_optimizations"][:8],
            [
                ("Label", "label"),
                ("Status", "optimization_status"),
                ("Priority", "action_priority"),
                ("Pain", "pain_score"),
                ("Savings Mi", "best_alternative.on_foot_savings_miles"),
                ("P75 Delta", "best_alternative.elapsed_delta_minutes"),
                ("Alternative", "best_alternative.alternative_id"),
                ("Access Decision", "best_alternative.access_decision.decision"),
            ],
        )
    )
    lines.extend(["", "## Highest Pain Route Cards", ""])
    lines.extend(
        markdown_table(
            report["top_pain_routes"][:8],
            [
                ("Label", "label"),
                ("Pain", "pain_score"),
                ("Official Mi", "official_miles"),
                ("On-Foot Mi", "on_foot_miles"),
                ("Overhead Mi", "overhead_miles"),
                ("P75", "door_to_door_minutes_p75"),
                ("P90", "door_to_door_minutes_p90"),
                ("Status", "optimization_status"),
            ],
        )
    )
    lines.extend(["", "## Method Notes", ""])
    lines.extend(f"- {note}" for note in report["method_notes"])
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-json", type=Path, default=DEFAULT_FIELD_TOOL_JSON)
    parser.add_argument("--multi-start-audit-json", type=Path, default=DEFAULT_MULTI_START_AUDIT_JSON)
    parser.add_argument("--completion-audit-json", type=Path, default=DEFAULT_COMPLETION_AUDIT_JSON)
    parser.add_argument("--access-verification-json", type=Path, default=DEFAULT_ACCESS_VERIFICATION_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    field_tool_payload = read_json(args.field_tool_json)
    multi_start_audit = read_json(args.multi_start_audit_json) if args.multi_start_audit_json.exists() else None
    completion_audit = read_json(args.completion_audit_json) if args.completion_audit_json.exists() else None
    access_verification = read_json(args.access_verification_json) if args.access_verification_json.exists() else None
    source_files = {
        "field_tool_json": display_path(args.field_tool_json),
        "multi_start_audit_json": display_path(args.multi_start_audit_json),
        "completion_audit_json": display_path(args.completion_audit_json),
        "access_verification_json": display_path(args.access_verification_json),
    }
    report = build_pain_index(
        field_tool_payload,
        multi_start_audit=multi_start_audit,
        completion_audit=completion_audit,
        access_verification=access_verification,
        source_files=source_files,
    )
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="route-pain-index-2026-05-10",
        inputs=[args.field_tool_json, args.multi_start_audit_json, args.completion_audit_json, args.access_verification_json],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/route_pain_index.py",
        metadata={"schema": report["schema"], "top_actionable_label": report["summary"]["top_actionable_label"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
