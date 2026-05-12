#!/usr/bin/env python3
"""Rank cluster-level route archetype mismatch, bundle, access, and dominance candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_TEMPLATE_CANDIDATES_JSON = YEAR_DIR / "checkpoints" / "common-route-template-candidates-2026-05-12.json"
DEFAULT_ROUTE_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_REPEAT_PRODUCTIVITY_JSON = YEAR_DIR / "checkpoints" / "repeat-productivity-audit-2026-05-12.json"
DEFAULT_SIMULATED_PROGRESS_JSON = YEAR_DIR / "checkpoints" / "simulated-progress-sweep-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "cluster-route-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "cluster-route-optimization-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "cluster-route-optimization-audit-2026-05-12-manifest.json"


ACCESS_CUE_TYPES = {"start_access", "official_segment_start"}
RETURN_CUE_TYPES = {"exit_access", "return_to_car"}


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


def sort_id(value: str) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def normalized_ids(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def int_value(value: Any) -> int:
    return int(round(float_value(value)))


def rounded(value: Any, digits: int = 2) -> float:
    return round(float_value(value), digits)


def normalize_text(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("route_key") or route.get("label") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def route_metrics(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_key": route_key(route),
        "outing_id": route.get("outing_id"),
        "label": route_label(route),
        "trailhead": route.get("trailhead"),
        "segment_ids": normalized_ids(route.get("segment_ids") or []),
        "official_miles": rounded(route.get("official_miles")),
        "on_foot_miles": rounded(route.get("on_foot_miles")),
        "door_to_door_minutes_p75": int_value(route.get("door_to_door_minutes_p75")),
        "door_to_door_minutes_p90": int_value(route.get("door_to_door_minutes_p90")),
    }


def route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index = {}
    for route in routes:
        for key in [route_key(route), str(route.get("outing_id") or ""), str(route.get("label") or ""), route_label(route)]:
            if key:
                index[key] = route
    return index


def repeat_index(route_repeat_audit: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = route_index(routes)
    rows = {}
    for row in route_repeat_audit.get("routes") or []:
        route = lookup.get(str(row.get("outing_id") or "")) or lookup.get(str(row.get("label") or ""))
        if route:
            rows[route_key(route)] = row
    return rows


def route_physical_ids(route: dict[str, Any], repeat_rows: dict[str, dict[str, Any]]) -> set[str]:
    key = route_key(route)
    repeat_row = repeat_rows.get(key) or {}
    ids = set(normalized_ids(route.get("segment_ids") or []))
    ids.update(normalized_ids(repeat_row.get("actual_full_segment_ids") or []))
    return ids


def productivity_index(repeat_productivity_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in repeat_productivity_audit.get("routes") or []:
        key = str(row.get("route_key") or row.get("outing_id") or "")
        if key:
            rows[key] = row
    return rows


def simulated_sweep_index(simulated_progress_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in simulated_progress_audit.get("route_sweeps_ranked") or []:
        for key in [str(row.get("route_id") or ""), *(str(item) for item in row.get("subject_route_keys") or [])]:
            if key:
                rows[key] = row
    return rows


def normal_start_matches(route_trailhead: Any, normal_start: Any) -> bool:
    trailhead = normalize_text(route_trailhead)
    starts = [normalize_text(item) for item in re.split(r"\bor\b|/|,", str(normal_start or "")) if item.strip()]
    if not trailhead or not starts:
        return False
    return any(trailhead in start or start in trailhead for start in starts)


def route_has_unresolved_direction(route: dict[str, Any], overlap_ids: set[str]) -> bool:
    evidence = route.get("segment_direction_evidence") or {}
    for segment_id in overlap_ids:
        row = evidence.get(str(segment_id)) if isinstance(evidence, dict) else None
        if row and row.get("direction_rule") == "ascent" and not row.get("allowed_geometry_direction"):
            return True
    return False


def archetype_mismatch_rows(template_candidates: dict[str, Any], routes_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for template in template_candidates.get("templates_ranked") or []:
        matched_routes = template.get("matched_current_routes") or []
        matched_count = len(matched_routes)
        unusual_routes = [
            row for row in matched_routes
            if not normal_start_matches(row.get("trailhead"), template.get("normal_start"))
        ]
        pressure = template.get("current_route_pressure") or {}
        candidate_official = float_value(template.get("candidate_official_miles"))
        old_on_foot = float_value(pressure.get("matched_current_on_foot_miles"))
        route_fragment_penalty = max(0, matched_count - 1) * 1.5
        unusual_start_penalty = len(unusual_routes) * 1.0
        high_repeat_without_future_savings_penalty = max(
            0.0,
            float_value(pressure.get("dead_repeat_candidate_miles"))
            - float_value(pressure.get("future_collapse_on_foot_miles")),
        )
        noncredit_burden_vs_public_route_penalty = max(0.0, old_on_foot - candidate_official) * 0.25
        unresolved_direction_count = 0
        for row in matched_routes:
            route = routes_by_key.get(str(row.get("route_key") or ""))
            overlap_ids = set(normalized_ids(row.get("overlap_segment_ids") or []))
            if route and route_has_unresolved_direction(route, overlap_ids):
                unresolved_direction_count += 1
        opposite_common_direction_penalty = unresolved_direction_count * 1.0
        components = {
            "unusual_start_penalty": rounded(unusual_start_penalty),
            "route_fragments_common_loop_penalty": rounded(route_fragment_penalty),
            "high_repeat_without_future_savings_penalty": rounded(high_repeat_without_future_savings_penalty),
            "opposite_common_direction_penalty": rounded(opposite_common_direction_penalty),
            "noncredit_burden_vs_public_route_penalty": rounded(noncredit_burden_vs_public_route_penalty),
        }
        mismatch_score = rounded(sum(float_value(value) for value in components.values()))
        rows.append(
            {
                "template_id": template.get("template_id"),
                "target_area": template.get("target_area"),
                "mismatch_score": mismatch_score,
                "score_components": components,
                "matched_route_count": matched_count,
                "matched_routes": matched_routes,
                "unusual_start_routes": [row["label"] for row in unusual_routes],
                "candidate_official_miles": rounded(candidate_official),
                "matched_current_on_foot_miles": rounded(old_on_foot),
                "warnings": template.get("warnings") or [],
                "investigation_status": "ranked_investigation_target" if mismatch_score > 0 else "low_mismatch_pressure",
                "direction_scoring_note": (
                    "Only unresolved ascent-direction evidence is scored; common-route direction geometry is still a promotion proof gap."
                ),
            }
        )
    return sorted(rows, key=lambda row: (-float_value(row["mismatch_score"]), str(row["template_id"])))


def bundle_replacement_rows(template_candidates: dict[str, Any]) -> list[dict[str, Any]]:
    bundles = []
    for template in template_candidates.get("templates_ranked") or []:
        matched_routes = template.get("matched_current_routes") or []
        contained = [row for row in matched_routes if row.get("match_type") == "contained_route_card"]
        touched = [row for row in matched_routes if row.get("match_type") != "contained_route_card"]
        all_current_ids: set[str] = set()
        for row in matched_routes:
            all_current_ids.update(normalized_ids(row.get("segment_ids") or []))
        candidate_ids = set(normalized_ids(template.get("candidate_segment_ids") or []))
        uncovered_current = all_current_ids - candidate_ids
        old_on_foot = sum(float_value(row.get("on_foot_miles")) for row in matched_routes)
        old_official = sum(float_value(row.get("official_miles")) for row in matched_routes)
        old_p75 = sum(int_value(row.get("door_to_door_minutes_p75")) for row in matched_routes)
        old_p90 = sum(int_value(row.get("door_to_door_minutes_p90")) for row in matched_routes)
        lower_bound_new_on_foot = float_value(template.get("candidate_official_miles"))
        bundles.append(
            {
                "replacement_type": "cluster_bundle",
                "template_id": template.get("template_id"),
                "target_area": template.get("target_area"),
                "replacement_status": "needs_additional_loops" if uncovered_current else "needs_route_geometry_and_p75",
                "replaces_routes": [row["label"] for row in contained],
                "replaces_route_ids": [str(row.get("outing_id") or row.get("route_key")) for row in contained],
                "touches_routes": [row["label"] for row in touched],
                "touches_route_ids": [str(row.get("outing_id") or row.get("route_key")) for row in touched],
                "new_loops": [
                    {
                        "loop_id": f"template:{template.get('template_id')}",
                        "source_template_id": template.get("template_id"),
                        "normal_start": template.get("normal_start"),
                        "normal_direction": template.get("normal_direction"),
                        "trail_sequence": template.get("trail_sequence") or [],
                        "official_segment_ids": normalized_ids(candidate_ids),
                    }
                ],
                "segment_coverage": normalized_ids(candidate_ids),
                "current_replaced_segment_ids": normalized_ids(all_current_ids),
                "uncovered_current_segment_ids": normalized_ids(uncovered_current),
                "extra_candidate_segment_ids": normalized_ids(candidate_ids - all_current_ids),
                "total_old_official_miles": rounded(old_official),
                "total_old_on_foot": rounded(old_on_foot),
                "total_old_p75_minutes": old_p75,
                "total_old_p90_minutes": old_p90,
                "total_new_on_foot": None,
                "total_new_p75_minutes": None,
                "total_new_p90_minutes": None,
                "total_new_on_foot_lower_bound": rounded(lower_bound_new_on_foot),
                "savings": None,
                "lower_bound_savings_if_no_connector_overhead": {
                    "on_foot_miles": rounded(old_on_foot - lower_bound_new_on_foot),
                    "note": "Lower bound only; not a replacement savings claim until route geometry, access, p75/p90, and field cues are generated.",
                },
                "promotion_gates": [
                    "add any uncovered current official segments to explicit loops or preserve their existing route cards",
                    "derive continuous car-to-car GPX for each new loop",
                    "validate official segment coverage and ascent-only direction",
                    "verify parking/access/current trail legality",
                    "price p75/p90 and DEM effort",
                    "recertify field-day preservation before replacing active cards",
                ],
            }
        )
    return sorted(
        bundles,
        key=lambda row: (
            -float_value(row["lower_bound_savings_if_no_connector_overhead"]["on_foot_miles"]),
            len(row["uncovered_current_segment_ids"]),
            str(row["template_id"]),
        ),
    )


def cue_name(cue: dict[str, Any]) -> str:
    signed = " / ".join(str(item) for item in cue.get("signed_as") or [])
    return normalize_text(f"{signed} {cue.get('target') or ''}")


def cue_signed_name(cue: dict[str, Any]) -> str:
    signed = " / ".join(str(item) for item in cue.get("signed_as") or [])
    return normalize_text(signed or cue.get("target") or "")


def cue_leg_miles(cue: dict[str, Any]) -> float:
    return float_value(cue.get("route_leg_miles") if cue.get("route_leg_miles") is not None else cue.get("leg_miles"))


def corridor_signature(route: dict[str, Any], cue: dict[str, Any], kind: str) -> str:
    ids = normalized_ids(route.get("segment_ids") or [])
    endpoint_segment = ids[0] if kind == "access" and ids else ids[-1] if ids else "none"
    return "|".join(
        [
            kind,
            normalize_text(route.get("trailhead")),
            cue_name(cue),
            str(endpoint_segment),
        ]
    )


def corridor_family_signature(route: dict[str, Any], cue: dict[str, Any], kind: str) -> str:
    return "|".join(
        [
            kind,
            normalize_text(route.get("trailhead")),
            cue_signed_name(cue),
        ]
    )


def corridor_rows(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for route in routes:
        cues = route.get("wayfinding_cues") or []
        for kind, cue_types in [("access", ACCESS_CUE_TYPES), ("return", RETURN_CUE_TYPES)]:
            cue = next((item for item in cues if item.get("cue_type") in cue_types), None)
            if not cue and kind == "return":
                cue = next((item for item in reversed(cues) if item.get("cue_type") in RETURN_CUE_TYPES), None)
            if not cue:
                continue
            rows.append(
                {
                    "signature": corridor_signature(route, cue, kind),
                    "family_signature": corridor_family_signature(route, cue, kind),
                    "kind": kind,
                    "route_key": route_key(route),
                    "route": route_metrics(route),
                    "cue_seq": cue.get("seq"),
                    "target": cue.get("target"),
                    "signed_as": cue.get("signed_as") or [],
                    "leg_miles": rounded(cue_leg_miles(cue), 3),
                    "official_repeat_segment_ids": normalized_ids(cue.get("official_repeat_segment_ids") or []),
                    "official_repeat_miles": rounded(cue.get("official_repeat_miles")),
                }
            )
    return rows


def field_day_index(field_tool_data: dict[str, Any], routes_by_key: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    by_key = {route_key(route): route_key(route) for route in routes_by_key.values()}
    for route in routes_by_key.values():
        if route.get("outing_id"):
            by_key[str(route.get("outing_id"))] = route_key(route)
        if route.get("label"):
            by_key[str(route.get("label"))] = route_key(route)
    day_ids: dict[str, set[str]] = {}
    for day in (field_tool_data.get("field_day_layer") or {}).get("field_days") or []:
        day_id = str(day.get("field_day_id") or day.get("date") or "")
        if not day_id:
            continue
        for loop in day.get("loops") or []:
            ref = loop.get("route_card_ref") or {}
            key = by_key.get(str(ref.get("outing_id") or "")) or by_key.get(str(ref.get("label") or ""))
            if key:
                day_ids.setdefault(key, set()).add(day_id)
    return day_ids


def sweep_relation(source_key: str, target_key: str, sweeps_by_route: dict[str, dict[str, Any]]) -> str | None:
    sweep = sweeps_by_route.get(source_key) or {}
    if any(row.get("route_key") == target_key for row in sweep.get("future_routes_removed") or []):
        return "removes_future_route"
    if any(row.get("route_key") == target_key for row in sweep.get("future_routes_shrunk") or []):
        return "shrinks_future_route"
    return None


def access_corridor_groups(
    field_tool_data: dict[str, Any],
    routes: list[dict[str, Any]],
    routes_by_key: dict[str, dict[str, Any]],
    sweeps_by_route: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in corridor_rows(routes):
        grouped.setdefault(str(row["family_signature"]), []).append(row)
    day_ids = field_day_index(field_tool_data, routes_by_key)
    result = []
    for family_signature, rows in grouped.items():
        route_keys = [str(row["route_key"]) for row in rows]
        if len(set(route_keys)) < 2:
            continue
        route_day_sets = [day_ids.get(key, set()) for key in route_keys]
        common_days = set.intersection(*route_day_sets) if route_day_sets and all(route_day_sets) else set()
        relations = []
        for left in route_keys:
            for right in route_keys:
                if left == right:
                    continue
                relation = sweep_relation(left, right, sweeps_by_route)
                if relation:
                    relations.append({"source_route_key": left, "target_route_key": right, "relation": relation})
        trailheads = sorted({str(row["route"].get("trailhead") or "") for row in rows})
        result.append(
            {
                "family_signature": family_signature,
                "strict_signatures": sorted({str(row["signature"]) for row in rows}),
                "kind": rows[0]["kind"],
                "route_count": len(set(route_keys)),
                "routes": rows,
                "total_corridor_leg_miles": rounded(sum(float_value(row.get("leg_miles")) for row in rows)),
                "same_trailhead": len(trailheads) == 1,
                "same_day_bundle_possible": bool(common_days) or len(trailheads) == 1,
                "common_field_day_ids": sorted(common_days),
                "latent_or_shrink_relations": relations,
                "can_one_route_shrink_after_other": bool(relations),
                "investigation_prompt": "same access/return corridor is paid by multiple cards; test same-day bundle or post-completion shrink",
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -float_value(row["total_corridor_leg_miles"]),
            -int_value(row["route_count"]),
            row["family_signature"],
        ),
    )


def post_progress_dominance_rows(sweeps_by_route: dict[str, dict[str, Any]], routes_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source_key, sweep in sweeps_by_route.items():
        source = routes_by_key.get(source_key)
        if not source:
            continue
        for removed in sweep.get("future_routes_removed") or []:
            target_key = str(removed.get("route_key") or "")
            rows.append(
                {
                    "dominance_type": "post_progress_route_removal",
                    "dominant_route": route_metrics(source),
                    "dominated_route": removed.get("route") or {"route_key": target_key},
                    "reason": "simulated completion of dominant route fully completes the dominated route's claimed official segments",
                    "action": "remove_after_validated_completion",
                }
            )
        for shrunk in sweep.get("future_routes_shrunk") or []:
            target_key = str(shrunk.get("route_key") or "")
            rows.append(
                {
                    "dominance_type": "post_progress_route_shrink",
                    "dominant_route": route_metrics(source),
                    "dominated_route": shrunk.get("route") or {"route_key": target_key},
                    "completed_claimed_segment_ids": normalized_ids(shrunk.get("completed_claimed_segment_ids") or []),
                    "reason": "simulated completion of dominant route completes some official segments currently claimed by the target route",
                    "action": "reprice_target_after_validated_completion",
                }
            )
    return rows


def current_route_dominance_rows(routes: list[dict[str, Any]], repeat_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left in routes:
        left_physical = route_physical_ids(left, repeat_rows)
        for right in routes:
            if route_key(left) == route_key(right):
                continue
            right_claimed = set(normalized_ids(right.get("segment_ids") or []))
            if not right_claimed or not right_claimed <= left_physical:
                continue
            if (
                float_value(left.get("on_foot_miles")) <= float_value(right.get("on_foot_miles"))
                and int_value(left.get("door_to_door_minutes_p75")) <= int_value(right.get("door_to_door_minutes_p75"))
                and int_value(left.get("door_to_door_minutes_p90")) <= int_value(right.get("door_to_door_minutes_p90"))
            ):
                rows.append(
                    {
                        "dominance_type": "current_route_dominates_current_route",
                        "dominant_route": route_metrics(left),
                        "dominated_route": route_metrics(right),
                        "covered_segment_ids": normalized_ids(right_claimed),
                        "reason": "dominant route physically covers a superset of target claimed segments with no higher on-foot, p75, or p90 cost",
                        "action": "investigate_stale_card_or_keep_explicit_backup_reason",
                    }
                )
    return rows


def lower_bound_bundle_dominance_rows(bundles: list[dict[str, Any]], routes_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for bundle in bundles:
        coverage = set(normalized_ids(bundle.get("segment_coverage") or []))
        for route_id in bundle.get("replaces_route_ids") or []:
            route = routes_by_key.get(str(route_id))
            if not route:
                continue
            claimed = set(normalized_ids(route.get("segment_ids") or []))
            if claimed and claimed <= coverage and float_value(bundle.get("total_new_on_foot_lower_bound")) <= float_value(route.get("on_foot_miles")):
                rows.append(
                    {
                        "dominance_type": "cluster_bundle_lower_bound_candidate",
                        "bundle_template_id": bundle.get("template_id"),
                        "dominated_route": route_metrics(route),
                        "covered_segment_ids": normalized_ids(claimed),
                        "reason": "bundle official-mile lower bound covers the route's claims with fewer miles, but p75/access/cue status is unpriced",
                        "action": "generate and price bundle before deleting route",
                    }
                )
    return rows


def dominance_rows(
    routes: list[dict[str, Any]],
    repeat_rows: dict[str, dict[str, Any]],
    sweeps_by_route: dict[str, dict[str, Any]],
    routes_by_key: dict[str, dict[str, Any]],
    bundles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    rows.extend(current_route_dominance_rows(routes, repeat_rows))
    rows.extend(post_progress_dominance_rows(sweeps_by_route, routes_by_key))
    rows.extend(lower_bound_bundle_dominance_rows(bundles, routes_by_key))
    return sorted(rows, key=lambda row: (row["dominance_type"], row.get("bundle_template_id") or "", row["dominated_route"].get("label") or ""))


def build_cluster_route_optimization_audit(
    field_tool_data: dict[str, Any],
    template_candidates: dict[str, Any],
    route_repeat_audit: dict[str, Any],
    repeat_productivity_audit: dict[str, Any],
    simulated_progress_audit: dict[str, Any],
    *,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = {route_key(route): route for route in routes}
    repeat_rows = repeat_index(route_repeat_audit, routes)
    sweeps_by_route = simulated_sweep_index(simulated_progress_audit)
    mismatch = archetype_mismatch_rows(template_candidates, routes_by_key)
    bundles = bundle_replacement_rows(template_candidates)
    access_groups = access_corridor_groups(field_tool_data, routes, routes_by_key, sweeps_by_route)
    dominance = dominance_rows(routes, repeat_rows, sweeps_by_route, routes_by_key, bundles)
    status = "cluster_optimization_targets_found" if mismatch or bundles or access_groups or dominance else "no_cluster_targets_found"
    return {
        "schema": "boise_trails_cluster_route_optimization_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "parameters": {
            "mismatch_score": "unusual_start + route_fragments + dead_repeat_not_future_saving + unresolved_direction + noncredit_burden_proxy",
            "bundle_policy": "cluster bundles are unpromoted candidate replacements until route geometry, p75/p90, access, cue, and recertification gates pass",
            "access_corridor_policy": "start/return cue signatures identify repeated paid access corridors for same-day bundle or post-completion shrink investigation",
            "dominance_policy": "hard dominance requires known on-foot/p75/p90 no worse; lower-bound bundle dominance is only an investigation target",
        },
        "scope": {
            "proves": [
                "which clusters diverge most from captured common-route templates",
                "which route-card groups should be priced as bundle replacements instead of one-card swaps",
                "which access and return corridors are paid by multiple current cards",
                "which current or simulated-progress relationships can remove or shrink stale cards",
            ],
            "does_not_prove": [
                "a generated bundle is runnable",
                "a route can be deleted before validated completion or recertification",
                "public route direction/legality is current",
                "lower-bound savings become real p75/on-foot savings",
            ],
        },
        "summary": {
            "template_cluster_count": len(template_candidates.get("templates_ranked") or []),
            "mismatch_target_count": len([row for row in mismatch if float_value(row["mismatch_score"]) > 0]),
            "top_mismatch_template": mismatch[0]["template_id"] if mismatch else None,
            "top_mismatch_score": mismatch[0]["mismatch_score"] if mismatch else 0.0,
            "cluster_bundle_count": len(bundles),
            "bundle_needs_additional_loops_count": len([row for row in bundles if row["replacement_status"] == "needs_additional_loops"]),
            "repeated_access_corridor_count": len(access_groups),
            "dominance_candidate_count": len(dominance),
            "post_progress_dominance_count": len([row for row in dominance if row["dominance_type"].startswith("post_progress")]),
        },
        "archetype_mismatch_ranked": mismatch,
        "cluster_bundle_replacements": bundles,
        "already_paid_access_corridors": access_groups,
        "dominance_checks": dominance,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Cluster Route Optimization Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Template clusters scored: {summary['template_cluster_count']}",
        f"- Mismatch investigation targets: {summary['mismatch_target_count']}",
        f"- Top mismatch: {summary['top_mismatch_template']} ({summary['top_mismatch_score']:.2f})",
        f"- Cluster bundle candidates: {summary['cluster_bundle_count']}",
        f"- Bundle candidates needing additional loops: {summary['bundle_needs_additional_loops_count']}",
        f"- Repeated paid access corridors: {summary['repeated_access_corridor_count']}",
        f"- Dominance candidates: {summary['dominance_candidate_count']}",
        "",
        "## Archetype Mismatch",
        "",
        "| Rank | Template | Score | Matched cards | Old on-foot | Template official | Components |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(audit.get("archetype_mismatch_ranked") or [], start=1):
        components = row["score_components"]
        component_text = (
            f"start {components['unusual_start_penalty']:.2f}; fragments {components['route_fragments_common_loop_penalty']:.2f}; "
            f"dead {components['high_repeat_without_future_savings_penalty']:.2f}; direction {components['opposite_common_direction_penalty']:.2f}; "
            f"burden {components['noncredit_burden_vs_public_route_penalty']:.2f}"
        )
        lines.append(
            f"| {index} | {row['template_id']} | {row['mismatch_score']:.2f} | {row['matched_route_count']} | "
            f"{row['matched_current_on_foot_miles']:.2f} | {row['candidate_official_miles']:.2f} | {component_text} |"
        )
    lines.extend(
        [
            "",
            "## Bundle Replacement Candidates",
            "",
            "| Rank | Template | Status | Replaces | Touches | Old on-foot | New lower bound | Uncovered current ids |",
            "|---:|---|---|---|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(audit.get("cluster_bundle_replacements") or [], start=1):
        lines.append(
            f"| {index} | {row['template_id']} | `{row['replacement_status']}` | {', '.join(row['replaces_routes']) or 'none'} | "
            f"{', '.join(row['touches_routes']) or 'none'} | {row['total_old_on_foot']:.2f} | "
            f"{row['total_new_on_foot_lower_bound']:.2f} | {len(row['uncovered_current_segment_ids'])} |"
        )
    lines.extend(
        [
            "",
            "## Already-Paid Access Corridors",
            "",
            "| Rank | Kind | Route count | Corridor miles paid | Same-day possible | Can shrink after another | Routes |",
            "|---:|---|---:|---:|---|---|---|",
        ]
    )
    for index, row in enumerate((audit.get("already_paid_access_corridors") or [])[:25], start=1):
        route_labels = ", ".join(item["route"]["label"] for item in row["routes"])
        lines.append(
            f"| {index} | {row['kind']} | {row['route_count']} | {row['total_corridor_leg_miles']:.2f} | "
            f"{'yes' if row['same_day_bundle_possible'] else 'no'} | {'yes' if row['can_one_route_shrink_after_other'] else 'no'} | {route_labels} |"
        )
    lines.extend(
        [
            "",
            "## Dominance Checks",
            "",
            "| Type | Dominant / bundle | Dominated | Action |",
            "|---|---|---|---|",
        ]
    )
    for row in (audit.get("dominance_checks") or [])[:30]:
        dominant = row.get("bundle_template_id") or (row.get("dominant_route") or {}).get("label") or ""
        dominated = (row.get("dominated_route") or {}).get("label") or (row.get("dominated_route") or {}).get("route_key") or ""
        lines.append(f"| {row['dominance_type']} | {dominant} | {dominated} | {row['action']} |")
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- This audit ranks investigation targets and bundle candidates. It does not promote route cards.",
            "- Bundle lower-bound savings use official miles only and are not p75/on-foot savings claims.",
            "- A route remains in the menu until validated completion, regenerated replacements, and recertification prove it can be removed or shrunk.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--template-candidates-json", type=Path, default=DEFAULT_TEMPLATE_CANDIDATES_JSON)
    parser.add_argument("--route-repeat-audit-json", type=Path, default=DEFAULT_ROUTE_REPEAT_AUDIT_JSON)
    parser.add_argument("--repeat-productivity-json", type=Path, default=DEFAULT_REPEAT_PRODUCTIVITY_JSON)
    parser.add_argument("--simulated-progress-json", type=Path, default=DEFAULT_SIMULATED_PROGRESS_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_cluster_route_optimization_audit(
        read_json(args.field_tool_data_json),
        read_json(args.template_candidates_json),
        read_json(args.route_repeat_audit_json),
        read_json(args.repeat_productivity_json),
        read_json(args.simulated_progress_json),
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "template_candidates_json": display_path(args.template_candidates_json),
            "route_repeat_audit_json": display_path(args.route_repeat_audit_json),
            "repeat_productivity_json": display_path(args.repeat_productivity_json),
            "simulated_progress_json": display_path(args.simulated_progress_json),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="cluster-route-optimization-audit-2026-05-12",
        inputs=[
            args.field_tool_data_json,
            args.template_candidates_json,
            args.route_repeat_audit_json,
            args.repeat_productivity_json,
            args.simulated_progress_json,
        ],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/cluster_route_optimization_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] == "no_cluster_targets_found" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
