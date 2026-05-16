#!/usr/bin/env python3
"""Audit whether the current route set is proven efficient enough to trust."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_FIELD_PACKET_JSON = REPO_ROOT / "docs" / "field-packet" / "manifest.json"
DEFAULT_HUMAN_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "human-loop-plan-v1.json"
DEFAULT_PACKAGE16_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "package16-manual-route-design-v1.json"
DEFAULT_ALTERNATIVE_CHALLENGE_JSON = YEAR_DIR / "checkpoints" / "route-alternative-challenge-ratio-gap-2026-05-06.json"
DEFAULT_BOUNDARY_CHALLENGE_JSONS = [
    YEAR_DIR / "checkpoints" / "route-boundary-challenge-p02-p13-2026-05-06.json",
    YEAR_DIR / "checkpoints" / "route-boundary-challenge-p06-p15-p16-2026-05-06.json",
    YEAR_DIR / "checkpoints" / "route-boundary-challenge-p17-p18-2026-05-06.json",
    YEAR_DIR / "checkpoints" / "route-boundary-challenge-p19-2026-05-06.json",
]
DEFAULT_GLOBAL_OPTIMIZER_JSON = YEAR_DIR / "checkpoints" / "route-global-optimizer-challenge-2026-05-06.json"
DEFAULT_ROUTE_PROOF_JSONS = [
    YEAR_DIR / "checkpoints" / "route-local-map-proof-2026-05-06.json",
    YEAR_DIR / "checkpoints" / "adversarial-route-disproof-2026-05-16.json",
]
DEFAULT_MANUAL_CHALLENGE_JSONS = [
    YEAR_DIR / "outputs" / "private" / "route-blocks" / "harlow-spring-manual-route-design-v1.json"
]
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "route-efficiency-audit-2026-05-06"
PREFERRED_PLAN_RATIO = 1.6
ACCEPTED_PROOF_RATIO_LIMIT = 1.65


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def component_rows(map_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            official = float(component.get("official_miles") or 0.0)
            on_foot = float(component.get("on_foot_miles") or 0.0)
            rows.append(
                {
                    "label": str(
                        component.get("field_menu_label")
                        or component.get("label")
                        or package.get("package_number")
                        or ""
                    ),
                    "package_number": package.get("package_number"),
                    "block_name": package.get("block_name") or "",
                    "planning_status": package.get("planning_status"),
                    "route_design_status": component.get("route_design_status"),
                    "trailhead": component.get("trailhead") or "",
                    "trails": component.get("trail_names") or [],
                    "official_miles": round(official, 2),
                    "on_foot_miles": round(on_foot, 2),
                    "overhead_miles": round(on_foot - official, 2),
                    "ratio": round(on_foot / official, 2) if official else None,
                    "total_minutes": component.get("total_minutes") or component.get("door_to_door_minutes"),
                    "candidate_ids": component.get("candidate_ids") or [component.get("candidate_id")],
                }
            )
    return rows


def is_manually_challenged(row: dict[str, Any]) -> bool:
    return row.get("route_design_status") == "gpx_generated_parking_manual" or row.get("planning_status") in {
        "accepted_manual_split_parking_manual",
        "accepted_manual_override",
    }


def route_proof_is_accepted(proof: dict[str, Any]) -> bool:
    checks = proof.get("checks") or {}
    return (
        proof.get("status") == "accepted_current"
        and checks.get("gpx_continuity_passed") is True
        and checks.get("current_route_has_p75_time") is True
        and checks.get("current_route_has_dem_effort") is True
        and checks.get("no_better_exact_generated_candidate") is True
        and checks.get("no_dominant_boundary_recombination") is True
        and checks.get("no_dominant_global_optimizer_replacement") is True
    )


def route_proof_index(route_proofs: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for registry in route_proofs or []:
        for proof in registry.get("proofs") or []:
            if not route_proof_is_accepted(proof):
                continue
            for candidate_id in proof.get("candidate_ids") or []:
                index[str(candidate_id)] = proof
            if proof.get("candidate_id"):
                index[str(proof["candidate_id"])] = proof
    return index


def is_proven_by_route_proof(row: dict[str, Any], accepted_proofs: dict[str, dict[str, Any]]) -> bool:
    return any(str(candidate_id) in accepted_proofs for candidate_id in row.get("candidate_ids") or [])


def route_rows(field_packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for route in field_packet.get("routes") or []:
        outing = route.get("outing") or {}
        official = float(outing.get("official_miles") or 0.0)
        on_foot = float(outing.get("on_foot_miles") or 0.0)
        rows.append(
            {
                "label": str(outing.get("label") or ""),
                "trailhead": outing.get("trailhead") or "",
                "trails": outing.get("trails") or [],
                "official_miles": round(official, 2),
                "on_foot_miles": round(on_foot, 2),
                "overhead_miles": round(on_foot - official, 2),
                "ratio": round(on_foot / official, 2) if official else None,
                    "total_minutes": outing.get("total_minutes"),
                    "segment_ids": [
                        int(segment_id)
                        for segment_id in outing.get("segment_ids") or outing.get("remaining_segment_ids") or []
                        if str(segment_id).isdigit()
                    ],
                    "has_segment_ids": bool(outing.get("segment_ids") or outing.get("remaining_segment_ids")),
            }
        )
    return rows


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    official = sum(float(row.get("official_miles") or 0.0) for row in rows)
    on_foot = sum(float(row.get("on_foot_miles") or 0.0) for row in rows)
    has_segment_ids = any(row.get("has_segment_ids") or row.get("segment_ids") for row in rows)
    segment_ids = {
        int(segment_id)
        for row in rows
        for segment_id in row.get("segment_ids") or []
        if str(segment_id).isdigit()
    }
    return {
        "official_miles": round(official, 2),
        "on_foot_miles": round(on_foot, 2),
        "ratio": round(on_foot / official, 3) if official else None,
        "count": len(rows),
        "covered_segment_count": len(segment_ids) if has_segment_ids else None,
    }


def top_rows(rows: list[dict[str, Any]], key: str, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key) or 0.0), reverse=True)[:limit]


def number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalized_effort(effort: dict[str, Any] | None) -> dict[str, Any]:
    effort = effort or {}
    return {
        "ascent_ft": number_or_none(effort.get("ascent_ft")),
        "grade_adjusted_miles": number_or_none(effort.get("grade_adjusted_miles")),
        "elevation_source": effort.get("elevation_source"),
    }


def has_dem_effort(effort: dict[str, Any]) -> bool:
    return (
        effort["ascent_ft"] is not None
        and effort["grade_adjusted_miles"] is not None
        and effort["elevation_source"] == "dem"
    )


def segment_effort_fields(segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not segments:
        return {"ascent_ft": None, "grade_adjusted_miles": None, "elevation_source": None}
    ascent_values = [number_or_none(segment.get("ascent_ft")) for segment in segments]
    grade_values = [number_or_none(segment.get("grade_adjusted_miles")) for segment in segments]
    return {
        "ascent_ft": sum(value for value in ascent_values if value is not None) if any(value is not None for value in ascent_values) else None,
        "grade_adjusted_miles": sum(value for value in grade_values if value is not None) if any(value is not None for value in grade_values) else None,
        "elevation_source": "dem" if any(segment.get("elevation_source") == "dem" for segment in segments) else None,
    }


def cue_effort_fields(cue: dict[str, Any] | None, component: dict[str, Any]) -> dict[str, Any]:
    cue = cue or {}
    for effort_source in (cue.get("effort"), component.get("effort")):
        effort = normalized_effort(effort_source)
        if has_dem_effort(effort):
            return effort

    segment_effort = segment_effort_fields(cue.get("segments") or component.get("segments") or [])
    if has_dem_effort(segment_effort):
        return segment_effort

    for effort_source in (cue.get("effort"), component.get("effort")):
        effort = normalized_effort(effort_source)
        if effort["ascent_ft"] is not None or effort["grade_adjusted_miles"] is not None:
            return effort
    return segment_effort


def time_estimate_quality(map_data: dict[str, Any]) -> dict[str, Any]:
    route_cues = map_data.get("route_cues") or {}
    rows = []
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            candidate_id = str(component.get("candidate_id") or "")
            cue = route_cues.get(candidate_id)
            time_estimates = (
                (cue or {}).get("time_estimates_minutes")
                or component.get("time_estimates_minutes")
                or {}
            )
            total = number_or_none(component.get("total_minutes") or component.get("door_to_door_minutes"))
            p75 = number_or_none(time_estimates.get("door_to_door_p75"))
            moving_p75 = number_or_none(time_estimates.get("moving_effort_p75"))
            effort = cue_effort_fields(cue, component)
            stale_delta = abs(p75 - total) if p75 is not None and total is not None else None
            stale_threshold = max(10.0, total * 0.15) if total is not None else 10.0
            rows.append(
                {
                    "label": str(
                        component.get("field_menu_label")
                        or component.get("label")
                        or package.get("package_number")
                        or ""
                    ),
                    "candidate_id": candidate_id,
                    "missing_cue": cue is None,
                    "missing_total": total is None,
                    "missing_p75": p75 is None,
                    "missing_moving_p75": moving_p75 is None,
                    "missing_effort": effort["ascent_ft"] is None
                    or effort["grade_adjusted_miles"] is None
                    or effort["elevation_source"] != "dem",
                    "stale_p75": stale_delta is not None and stale_delta > stale_threshold,
                    "stale_delta_minutes": round(stale_delta) if stale_delta is not None else None,
                    "component_total_minutes": int(round(total)) if total is not None else None,
                    "door_to_door_p75_minutes": int(round(p75)) if p75 is not None else None,
                    "moving_effort_p75_minutes": int(round(moving_p75)) if moving_p75 is not None else None,
                    "ascent_ft": int(round(effort["ascent_ft"])) if effort["ascent_ft"] is not None else None,
                    "grade_adjusted_miles": round(effort["grade_adjusted_miles"], 2)
                    if effort["grade_adjusted_miles"] is not None
                    else None,
                }
            )
    problem_rows = [
        row
        for row in rows
        if row["missing_cue"]
        or row["missing_total"]
        or row["missing_p75"]
        or row["missing_moving_p75"]
        or row["missing_effort"]
        or row["stale_p75"]
    ]
    return {
        "component_count": len(rows),
        "problem_count": len(problem_rows),
        "missing_cue_count": sum(1 for row in rows if row["missing_cue"]),
        "missing_total_count": sum(1 for row in rows if row["missing_total"]),
        "missing_p75_count": sum(1 for row in rows if row["missing_p75"]),
        "missing_moving_p75_count": sum(1 for row in rows if row["missing_moving_p75"]),
        "missing_effort_count": sum(1 for row in rows if row["missing_effort"]),
        "stale_p75_count": sum(1 for row in rows if row["stale_p75"]),
        "problem_candidate_ids": [row["candidate_id"] for row in problem_rows],
        "problems": problem_rows[:12],
    }


def package16_summary(package16: dict[str, Any]) -> dict[str, Any]:
    areas = package16.get("areas") or []
    if not areas:
        return {}
    area = areas[0]
    current = area.get("current_placeholder") or {}
    best = area.get("default_split_probe") or area.get("current_good_route") or {}
    return {
        "status": area.get("status"),
        "decision": area.get("decision"),
        "held_official_miles": current.get("official_miles") or area.get("current_demoted_official_miles"),
        "held_on_foot_miles": current.get("on_foot_miles") or area.get("current_demoted_on_foot_miles"),
        "accepted_split_official_miles": best.get("official_miles"),
        "accepted_split_on_foot_miles": best.get("on_foot_miles"),
        "improvement_miles": best.get("improvement_vs_current_on_foot_miles"),
        "remaining_blocker": (area.get("current_good_route") or {}).get("remaining_blocker"),
    }


def alternative_challenge_summary(alternative_challenge: dict[str, Any] | None) -> dict[str, Any]:
    if not alternative_challenge:
        return {
            "available": False,
            "target_count": 0,
            "challenged_candidate_ids": [],
            "better_exact_candidate_count": None,
            "better_superset_candidate_count": None,
            "manual_map_review_still_required_count": None,
        }
    summary = alternative_challenge.get("summary") or {}
    return {
        "available": True,
        "target_count": summary.get("target_count") or 0,
        "challenged_candidate_ids": summary.get("challenged_candidate_ids") or [],
        "better_exact_candidate_count": summary.get("better_exact_candidate_count") or 0,
        "better_superset_candidate_count": summary.get("better_superset_candidate_count") or 0,
        "manual_map_review_still_required_count": summary.get("manual_map_review_still_required_count") or 0,
    }


def manual_challenge_summary(
    manual_challenges: list[dict[str, Any]] | None,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    active_candidate_ids = {
        str(candidate_id)
        for component in components
        for candidate_id in (component.get("candidate_ids") or [])
        if candidate_id
    }
    accepted = []
    for challenge in manual_challenges or []:
        for area in challenge.get("areas") or []:
            current_good = area.get("current_good_route") or {}
            if not current_good:
                continue
            demoted_ids = [str(value) for value in area.get("demote_candidate_ids") or []]
            still_active = sorted(set(demoted_ids) & active_candidate_ids)
            improvement = (
                float(area.get("current_demoted_on_foot_miles") or 0.0)
                - float(current_good.get("on_foot_miles") or 0.0)
            )
            accepted.append(
                {
                    "area_id": area.get("area_id"),
                    "title": area.get("title"),
                    "demote_candidate_ids": demoted_ids,
                    "still_active_candidate_ids": still_active,
                    "official_miles": current_good.get("official_miles"),
                    "on_foot_miles": current_good.get("on_foot_miles"),
                    "improvement_miles": round(improvement, 2),
                    "remaining_blocker": current_good.get("remaining_blocker"),
                }
            )
    pending = [area for area in accepted if area["still_active_candidate_ids"]]
    return {
        "available": bool(manual_challenges),
        "accepted_manual_improvement_count": len(accepted),
        "pending_integration_count": len(pending),
        "potential_on_foot_savings_miles": round(sum(float(area["improvement_miles"] or 0.0) for area in pending), 2),
        "pending": pending,
    }


def boundary_challenge_summary(boundary_challenges: list[dict[str, Any]] | None) -> dict[str, Any]:
    challenges = boundary_challenges or []
    package_numbers: set[int] = set()
    for challenge in challenges:
        for package_number in challenge.get("package_numbers") or []:
            try:
                package_numbers.add(int(package_number))
            except (TypeError, ValueError):
                continue
    summaries = [challenge.get("summary") or {} for challenge in challenges]
    return {
        "available": bool(challenges),
        "challenge_count": len(challenges),
        "challenged_package_numbers": sorted(package_numbers),
        "generated_combo_beats_current_count": sum(
            1 for summary in summaries if summary.get("generated_combo_beats_current") is True
        ),
        "better_generated_metric_count": sum(int(summary.get("better_generated_metric_count") or 0) for summary in summaries),
        "all_have_elevation_metrics": bool(challenges)
        and all(summary.get("all_covering_combos_include_elevation") is True for summary in summaries),
        "all_have_p75_time": bool(challenges)
        and all(summary.get("all_covering_combos_include_p75_time") is True for summary in summaries),
    }


def global_optimizer_summary(global_optimizer: dict[str, Any] | None) -> dict[str, Any]:
    if not global_optimizer:
        return {
            "available": False,
            "global_optimizer_beats_current": None,
            "dominant_solution_count": None,
            "best_dominant_solution": None,
        }
    summary = global_optimizer.get("summary") or {}
    best = global_optimizer.get("best_dominant_solution")
    return {
        "available": True,
        "global_optimizer_beats_current": bool(summary.get("global_optimizer_beats_current")),
        "dominant_solution_count": int(summary.get("dominant_solution_count") or 0),
        "best_dominant_solution": {
            "materially_better_metrics": (best or {}).get("materially_better_metrics") or [],
            "materially_worse_metrics": (best or {}).get("materially_worse_metrics") or [],
            "deltas": (best or {}).get("deltas") or {},
            "candidate_ids": ((best or {}).get("combo") or {}).get("candidate_ids") or [],
        }
        if best
        else None,
    }


def route_proof_summary(route_proofs: list[dict[str, Any]] | None, components: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = route_proof_index(route_proofs)
    active_candidate_ids = {
        str(candidate_id)
        for component in components
        for candidate_id in (component.get("candidate_ids") or [])
        if candidate_id
    }
    accepted_active = sorted(set(accepted) & active_candidate_ids)
    return {
        "available": bool(route_proofs),
        "registry_count": len(route_proofs or []),
        "accepted_proof_count": len({id(proof) for proof in accepted.values()}),
        "accepted_candidate_ids": sorted(accepted),
        "accepted_active_candidate_ids": accepted_active,
    }


def build_audit(
    map_data: dict[str, Any],
    field_packet: dict[str, Any],
    human_plan: dict[str, Any],
    package16: dict[str, Any],
    alternative_challenge: dict[str, Any] | None = None,
    manual_challenges: list[dict[str, Any]] | None = None,
    boundary_challenges: list[dict[str, Any]] | None = None,
    global_optimizer: dict[str, Any] | None = None,
    route_proofs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    components = component_rows(map_data)
    runnable = route_rows(field_packet)
    map_summary = map_data.get("summary") or {}
    field_summary = field_packet.get("summary") or {}
    human_summary = human_plan.get("summary") or {}
    manual_holds = field_packet.get("manual_holds") or []
    component_totals = totals(components)
    runnable_totals = totals(runnable)
    accepted_route_proofs = route_proof_index(route_proofs)
    high_ratio_components = [
        row
        for row in components
        if float(row.get("ratio") or 0.0) > 2.0
        and not is_manually_challenged(row)
        and not is_proven_by_route_proof(row, accepted_route_proofs)
    ]
    high_overhead_components = [
        row
        for row in components
        if float(row.get("overhead_miles") or 0.0) >= 6.0
        and not is_manually_challenged(row)
        and not is_proven_by_route_proof(row, accepted_route_proofs)
    ]
    challenge_summary = alternative_challenge_summary(alternative_challenge)
    manual_summary = manual_challenge_summary(manual_challenges, components)
    boundary_summary = boundary_challenge_summary(boundary_challenges)
    optimizer_summary = global_optimizer_summary(global_optimizer)
    route_proofs_summary = route_proof_summary(route_proofs, components)
    time_quality = time_estimate_quality(map_data)
    challenged_ids = set(challenge_summary["challenged_candidate_ids"])
    high_target_ids = {
        candidate_id
        for row in high_ratio_components + high_overhead_components
        for candidate_id in (row.get("candidate_ids") or [])
        if candidate_id
    }
    candidate_challenge_covers_current_targets = not high_target_ids or high_target_ids <= challenged_ids
    component_ratio = float(component_totals.get("ratio") or 99.0)
    ratio_overage_miles = max(
        0.0,
        float(component_totals.get("on_foot_miles") or 0.0)
        - float(component_totals.get("official_miles") or 0.0) * PREFERRED_PLAN_RATIO,
    )
    challenged_candidate_ids = set(challenge_summary["challenged_candidate_ids"])
    proofed_candidate_ids = set(route_proofs_summary["accepted_candidate_ids"])
    challenged_targets_have_proofs = bool(challenged_candidate_ids) and challenged_candidate_ids <= proofed_candidate_ids
    preferred_ratio_met = component_ratio <= PREFERRED_PLAN_RATIO
    accepted_ratio_proof_met = (
        component_ratio <= ACCEPTED_PROOF_RATIO_LIMIT
        and challenge_summary["available"]
        and int(challenge_summary["better_exact_candidate_count"] or 0) == 0
        and int(challenge_summary["better_superset_candidate_count"] or 0) == 0
        and challenged_targets_have_proofs
        and optimizer_summary["available"]
        and not optimizer_summary["global_optimizer_beats_current"]
    )
    runnable_segment_count = runnable_totals.get("covered_segment_count")
    target_segment_count = int(map_summary.get("covered_segment_count") or 0)
    runnable_coverage_passed = not manual_holds and (
        runnable_segment_count is None or int(runnable_segment_count or 0) == target_segment_count
    )
    gates = [
        {
            "gate": "Full official coverage is represented",
            "status": "passed" if int(map_summary.get("covered_segment_count") or 0) == 251 else "failed",
            "evidence": f"map_data covered_segment_count={map_summary.get('covered_segment_count')}; official_miles={map_summary.get('official_miles')}",
        },
        {
            "gate": "Runnable field packet covers all official work",
            "status": "passed" if runnable_coverage_passed else "failed",
            "evidence": (
                f"runnable_segments={runnable_segment_count}; "
                f"target_segments={target_segment_count}; "
                f"runnable_official_miles={runnable_totals['official_miles']}; manual_holds={len(manual_holds)}"
            ),
        },
        {
            "gate": "Planwide on-foot/official ratio is within preferred target or accepted proof tolerance",
            "status": "passed" if preferred_ratio_met or accepted_ratio_proof_met else "failed",
            "evidence": (
                f"current_all_component_ratio={component_totals['ratio']}; "
                f"runnable_ratio={runnable_totals['ratio']}; "
                f"preferred={PREFERRED_PLAN_RATIO}; accepted_proof_limit={ACCEPTED_PROOF_RATIO_LIMIT}; "
                f"over_preferred_by_miles={round(ratio_overage_miles, 2)}; "
                f"challenged_targets={len(challenged_candidate_ids)}; "
                f"proofed_challenged_targets={challenged_targets_have_proofs}"
            ),
        },
        {
            "gate": "No unresolved manual route-design area remains",
            "status": "failed" if int(human_summary.get("manual_design_area_count") or 0) else "passed",
            "evidence": f"manual_design_area_count={human_summary.get('manual_design_area_count')}; package16={package16_summary(package16).get('status')}",
        },
        {
            "gate": "No route exceeds 2.0x without manual/local-map proof",
            "status": "failed" if high_ratio_components else "passed",
            "evidence": f"unchallenged_components_over_2x={len(high_ratio_components)}",
        },
        {
            "gate": "Largest overhead routes have been manually challenged",
            "status": "failed" if high_overhead_components else "passed",
            "evidence": f"unchallenged_components_with_6+_overhead_miles={len(high_overhead_components)}",
        },
        {
            "gate": "Generated candidate universe has been checked for better exact alternatives",
            "status": "passed"
            if (
                not high_target_ids
                or challenge_summary["available"]
                and candidate_challenge_covers_current_targets
                and int(challenge_summary["better_exact_candidate_count"] or 0) == 0
            )
            else "failed",
            "evidence": (
                f"challenge_available={challenge_summary['available']}; "
                f"targets={challenge_summary['target_count']}; "
                f"better_exact={challenge_summary['better_exact_candidate_count']}; "
                f"covers_current_targets={candidate_challenge_covers_current_targets}"
            ),
        },
        {
            "gate": "Boundary recombination checks include elevation and p75 time",
            "status": "passed"
            if (
                not high_overhead_components
                or boundary_summary["available"]
                and boundary_summary["generated_combo_beats_current_count"] == 0
                and boundary_summary["all_have_elevation_metrics"]
                and boundary_summary["all_have_p75_time"]
            )
            else "failed",
            "evidence": (
                f"boundary_challenges={boundary_summary['challenge_count']}; "
                f"packages={boundary_summary['challenged_package_numbers']}; "
                f"better_metric_count={boundary_summary['better_generated_metric_count']}; "
                f"beats_current_count={boundary_summary['generated_combo_beats_current_count']}; "
                f"elevation={boundary_summary['all_have_elevation_metrics']}; "
                f"p75_time={boundary_summary['all_have_p75_time']}"
            ),
        },
        {
            "gate": "Global executable set-cover optimizer has no dominant replacement",
            "status": "passed"
            if optimizer_summary["available"] and not optimizer_summary["global_optimizer_beats_current"]
            else "failed",
            "evidence": (
                f"available={optimizer_summary['available']}; "
                f"beats_current={optimizer_summary['global_optimizer_beats_current']}; "
                f"dominant_solutions={optimizer_summary['dominant_solution_count']}"
            ),
        },
        {
            "gate": "Runnable outings have current p75 time and DEM effort estimates",
            "status": "passed" if time_quality["problem_count"] == 0 else "failed",
            "evidence": (
                f"components={time_quality['component_count']}; "
                f"missing_p75={time_quality['missing_p75_count']}; "
                f"missing_moving={time_quality['missing_moving_p75_count']}; "
                f"missing_effort={time_quality['missing_effort_count']}; "
                f"stale_p75={time_quality['stale_p75_count']}"
            ),
        },
        {
            "gate": "Accepted manual improvements have been integrated or explicitly rejected",
            "status": "failed" if manual_summary["pending_integration_count"] else "passed",
            "evidence": (
                f"accepted_manual_improvements={manual_summary['accepted_manual_improvement_count']}; "
                f"pending_integration={manual_summary['pending_integration_count']}; "
                f"potential_savings_miles={manual_summary['potential_on_foot_savings_miles']}"
            ),
        },
    ]
    achieved = all(gate["status"] == "passed" for gate in gates)
    next_required_work = []
    package16 = package16_summary(package16)
    package16_pending = bool(manual_holds) or bool(manual_summary["pending_integration_count"]) or bool(
        int(human_summary.get("manual_design_area_count") or 0)
    )
    if package16_pending:
        next_required_work.append("Integrate or explicitly reject the Package 16 accepted split probe in the runnable field packet.")
    if float(map_summary.get("planwide_on_foot_to_official_ratio") or 99.0) > 1.6 or high_ratio_components or high_overhead_components:
        manual_target_names = []
        for row in sorted(high_overhead_components, key=lambda item: float(item.get("overhead_miles") or 0.0), reverse=True):
            name = row.get("block_name") or row.get("label")
            if name and name not in manual_target_names:
                manual_target_names.append(str(name))
        if manual_summary["pending_integration_count"]:
            next_required_work.append(
                "Integrate or explicitly reject accepted manual improvements before calling the current field menu efficient."
            )
        if high_ratio_components or high_overhead_components:
            if challenge_summary["available"] and int(challenge_summary["better_exact_candidate_count"] or 0) == 0:
                next_required_work.append(
                    "The generated candidate universe does not contain a better exact route for the high-overhead targets; next proof step is manual/local-map GPX challenge for those areas."
                )
            elif challenge_summary["available"]:
                next_required_work.append(
                    "Implement or reject the better exact route alternatives found by route-alternative-challenge before trusting the menu."
                )
        if not preferred_ratio_met and not accepted_ratio_proof_met:
            next_required_work.append(
                f"Planwide ratio is {round(ratio_overage_miles, 2)} on-foot miles above the preferred {PREFERRED_PLAN_RATIO}x target; challenge and proof the next ratio-gap candidates or lower the route mileage."
            )
        if manual_target_names:
            next_required_work.append(
                "Manually challenge the remaining highest-overhead routes first: "
                + ", ".join(manual_target_names)
                + "."
            )
        if high_ratio_components or high_overhead_components or (not preferred_ratio_met and not accepted_ratio_proof_met):
            next_required_work.append(
                "For each challenged route, record the best alternative found, its official miles, on-foot miles, parking/start, and why the current route wins or loses."
            )
    if optimizer_summary["global_optimizer_beats_current"]:
        next_required_work.append(
            "Implement or explicitly reject the dominant global optimizer replacement before calling the menu route-optimal."
        )
    if not next_required_work:
        next_required_work.append(
            "No unresolved efficiency work remains under the current single-car, public-road-allowed, p75-time-aware proof gates."
        )
    else:
        next_required_work.append("Only then re-run this audit and consider tightening the preferred ratio gate below 1.7x.")
    return {
        "objective": "prove the current Boise Trails Challenge route set is as efficient as practical under the user's constraints",
        "verdict": "proven" if achieved else "not_proven",
        "achieved": achieved,
        "summary": {
            "all_component_totals": component_totals,
            "runnable_field_packet_totals": runnable_totals,
            "manual_hold_count": len(manual_holds),
            "human_loop_plan_ratio": human_summary.get("planwide_on_foot_to_official_ratio"),
            "human_loop_plan_on_foot_miles": human_summary.get("total_on_foot_miles"),
            "alternative_challenge": challenge_summary,
            "manual_challenges": manual_summary,
            "boundary_challenges": boundary_summary,
            "global_optimizer": optimizer_summary,
            "route_proofs": route_proofs_summary,
            "time_estimate_quality": time_quality,
        },
        "gates": gates,
        "package16": package16,
        "worst_ratio_components": top_rows(components, "ratio"),
        "worst_overhead_components": top_rows(components, "overhead_miles"),
        "longest_components": top_rows(components, "total_minutes"),
        "next_required_work": next_required_work,
    }


def render_md(audit: dict[str, Any]) -> str:
    lines = [
        "# Route Efficiency Audit",
        "",
        f"Objective: {audit['objective']}",
        "",
        f"Verdict: {audit['verdict']}",
        f"Achieved: {audit['achieved']}",
        "",
        "## Summary",
        "",
    ]
    summary = audit["summary"]
    lines.extend(
        [
            f"- All-component plan: {summary['all_component_totals']['official_miles']} official mi / {summary['all_component_totals']['on_foot_miles']} on-foot mi / {summary['all_component_totals']['ratio']}x",
            f"- Runnable field packet: {summary['runnable_field_packet_totals']['official_miles']} official mi / {summary['runnable_field_packet_totals']['on_foot_miles']} on-foot mi / {summary['runnable_field_packet_totals']['ratio']}x",
            f"- Manual holds: {summary['manual_hold_count']}",
            f"- Human-loop plan: {summary['human_loop_plan_on_foot_miles']} on-foot mi / {summary['human_loop_plan_ratio']}x",
            f"- Alternative challenge: available={summary.get('alternative_challenge', {}).get('available')}; targets={summary.get('alternative_challenge', {}).get('target_count')}; better exact={summary.get('alternative_challenge', {}).get('better_exact_candidate_count')}",
            f"- Boundary challenges: available={summary.get('boundary_challenges', {}).get('available')}; count={summary.get('boundary_challenges', {}).get('challenge_count')}; packages={summary.get('boundary_challenges', {}).get('challenged_package_numbers')}; better metrics={summary.get('boundary_challenges', {}).get('better_generated_metric_count')}",
            f"- Global optimizer: available={summary.get('global_optimizer', {}).get('available')}; beats current={summary.get('global_optimizer', {}).get('global_optimizer_beats_current')}; dominant solutions={summary.get('global_optimizer', {}).get('dominant_solution_count')}",
            f"- Route proofs: available={summary.get('route_proofs', {}).get('available')}; accepted active={len(summary.get('route_proofs', {}).get('accepted_active_candidate_ids') or [])}",
            f"- Time estimate quality: problems={summary.get('time_estimate_quality', {}).get('problem_count')}; missing p75={summary.get('time_estimate_quality', {}).get('missing_p75_count')}; stale p75={summary.get('time_estimate_quality', {}).get('stale_p75_count')}; missing effort={summary.get('time_estimate_quality', {}).get('missing_effort_count')}",
            f"- Manual improvements: accepted={summary.get('manual_challenges', {}).get('accepted_manual_improvement_count')}; pending integration={summary.get('manual_challenges', {}).get('pending_integration_count')}; potential savings={summary.get('manual_challenges', {}).get('potential_on_foot_savings_miles')} mi",
            "",
            "## Gates",
            "",
            "| Gate | Status | Evidence |",
            "|---|---|---|",
        ]
    )
    for gate in audit["gates"]:
        lines.append(f"| {gate['gate']} | {gate['status']} | {gate['evidence']} |")
    lines.extend(["", "## Package 16", ""])
    for key, value in audit.get("package16", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Worst Ratio Components", "", "| Label | Trailhead | Official | On-foot | Ratio | Trails |", "|---|---|---:|---:|---:|---|"])
    for row in audit["worst_ratio_components"]:
        lines.append(
            f"| {row['label']} | {row['trailhead']} | {row['official_miles']} | {row['on_foot_miles']} | {row['ratio']} | {', '.join(row['trails'])} |"
        )
    lines.extend(["", "## Worst Overhead Components", "", "| Label | Trailhead | Official | On-foot | Overhead | Ratio | Trails |", "|---|---|---:|---:|---:|---:|---|"])
    for row in audit["worst_overhead_components"]:
        lines.append(
            f"| {row['label']} | {row['trailhead']} | {row['official_miles']} | {row['on_foot_miles']} | {row['overhead_miles']} | {row['ratio']} | {', '.join(row['trails'])} |"
        )
    lines.extend(["", "## Next Required Work", ""])
    lines.extend(f"- {step}" for step in audit["next_required_work"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--field-packet-json", type=Path, default=DEFAULT_FIELD_PACKET_JSON)
    parser.add_argument("--human-plan-json", type=Path, default=DEFAULT_HUMAN_PLAN_JSON)
    parser.add_argument("--package16-json", type=Path, default=DEFAULT_PACKAGE16_JSON)
    parser.add_argument("--alternative-challenge-json", type=Path, default=DEFAULT_ALTERNATIVE_CHALLENGE_JSON)
    parser.add_argument("--global-optimizer-json", type=Path, default=DEFAULT_GLOBAL_OPTIMIZER_JSON)
    parser.add_argument("--boundary-challenge-json", action="append", type=Path, dest="boundary_challenge_jsons")
    parser.add_argument("--manual-challenge-json", action="append", type=Path, dest="manual_challenge_jsons")
    parser.add_argument("--route-proof-json", action="append", type=Path, dest="route_proof_jsons")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manual_challenge_paths = args.manual_challenge_jsons if args.manual_challenge_jsons is not None else DEFAULT_MANUAL_CHALLENGE_JSONS
    boundary_challenge_paths = (
        args.boundary_challenge_jsons if args.boundary_challenge_jsons is not None else DEFAULT_BOUNDARY_CHALLENGE_JSONS
    )
    route_proof_paths = args.route_proof_jsons if args.route_proof_jsons is not None else DEFAULT_ROUTE_PROOF_JSONS
    manual_challenges = [
        payload
        for payload in (read_optional_json(path) for path in manual_challenge_paths)
        if payload is not None
    ]
    boundary_challenges = [
        payload
        for payload in (read_optional_json(path) for path in boundary_challenge_paths)
        if payload is not None
    ]
    route_proofs = [
        payload
        for payload in (read_optional_json(path) for path in route_proof_paths)
        if payload is not None
    ]
    audit = build_audit(
        read_json(args.map_data_json),
        read_json(args.field_packet_json),
        read_json(args.human_plan_json),
        read_json(args.package16_json),
        read_optional_json(args.alternative_challenge_json),
        manual_challenges,
        boundary_challenges,
        read_optional_json(args.global_optimizer_json),
        route_proofs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.basename}.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"{args.basename}.md").write_text(render_md(audit), encoding="utf-8")
    print(f"Wrote {args.output_dir / f'{args.basename}.json'}")
    print(f"Wrote {args.output_dir / f'{args.basename}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
