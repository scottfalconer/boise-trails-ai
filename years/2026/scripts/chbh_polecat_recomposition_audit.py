#!/usr/bin/env python3
"""Compare 36th/CHBH/Polecat 5 route ownership alternatives.

This is a private route-generation audit, not a field-packet promotion. It
uses the existing connector graph and generated-GPX validation path to answer
whether CHBH Connector 1 and Polecat Loop 5 belong with the current Peggy's /
Cartwright card, the Polecat core card, or a smaller 36th Street Chute card.
"""

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
from export_execution_gpx import render_gpx_segments  # noqa: E402
from freestone_cluster_route_generation_experiment import build_generated_route  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    load_connector_graph,
    load_official_segments,
    round_miles,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "chbh-polecat-recomposition-2026-06-23"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "chbh-polecat-recomposition-audit.json"
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_DIR / "chbh-polecat-recomposition-audit.md"
DEFAULT_MANIFEST_JSON = DEFAULT_OUTPUT_DIR / "chbh-polecat-recomposition-audit-manifest.json"

ROUTE_CODES = ("1A-1", "5B", "6")
CURRENT_SEGMENT_SET = [
    "1482",
    "1599",
    "1602",
    "1600",
    "1598",
    "1601",
    "1603",
    "1604",
    "1541",
    "1610",
    "1597",
    "1519",
    "1520",
    "1521",
    "1709",
    "1509",
    "1508",
    "1516",
]

VARIANTS = [
    {
        "base_variant_id": "lower36-36th-only",
        "title": "Lower 36th: 36th Street Chute only",
        "parking_route_code": "1A-1",
        "ordered_segment_ids": ["1482"],
        "reason": "Control card for the current Chute-only official credit.",
    },
    {
        "base_variant_id": "lower36-36th-chbh",
        "title": "Lower 36th: 36th Street Chute plus CHBH",
        "parking_route_code": "1A-1",
        "ordered_segment_ids": ["1482", "1516"],
        "reason": "Move CHBH out of the Peggy's / Cartwright card into the nearby Chute outing.",
    },
    {
        "base_variant_id": "lower36-36th-chbh-polecat5",
        "title": "Lower 36th: 36th Street Chute plus CHBH and Polecat 5",
        "parking_route_code": "1A-1",
        "ordered_segment_ids": ["1482", "1516", "1601"],
        "reason": "User-suggested Chute-side ownership for CHBH and Polecat Loop 5.",
    },
    {
        "base_variant_id": "cartwright-5b-current",
        "title": "Cartwright: current Polecat core 5B segment set",
        "parking_route_code": "5B",
        "ordered_segment_ids": ["1599", "1602", "1600", "1598", "1601", "1603", "1604", "1541", "1610"],
        "reason": "Generated control for the current Polecat core ownership.",
    },
    {
        "base_variant_id": "cartwright-5b-minus-polecat5",
        "title": "Cartwright: Polecat core without Polecat 5",
        "parking_route_code": "5B",
        "ordered_segment_ids": ["1599", "1602", "1600", "1598", "1603", "1604", "1541", "1610"],
        "reason": "Residual Polecat core if Polecat Loop 5 moves to the Chute-side outing.",
    },
    {
        "base_variant_id": "cartwright-5b-plus-chbh",
        "title": "Cartwright: Polecat core plus CHBH",
        "parking_route_code": "5B",
        "ordered_segment_ids": ["1599", "1602", "1600", "1598", "1601", "1603", "1604", "1541", "1610", "1516"],
        "reason": "Move CHBH to the Polecat core instead of leaving it in Peggy's / Cartwright.",
    },
    {
        "base_variant_id": "cartwright-chbh-polecat5",
        "title": "Cartwright: CHBH plus Polecat 5 mini-card",
        "parking_route_code": "5B",
        "ordered_segment_ids": ["1516", "1601"],
        "reason": "Small split card if CHBH and Polecat 5 should stay north but not bloat Peggy's.",
    },
    {
        "base_variant_id": "cartwright-6-current",
        "title": "Cartwright: current Peggy's / Cartwright segment set",
        "parking_route_code": "6",
        "ordered_segment_ids": ["1597", "1519", "1520", "1521", "1709", "1509", "1508", "1516"],
        "reason": "Generated control for the current Route 6 ownership.",
    },
    {
        "base_variant_id": "cartwright-6-minus-chbh",
        "title": "Cartwright: Peggy's / Cartwright without CHBH",
        "parking_route_code": "6",
        "ordered_segment_ids": ["1597", "1519", "1520", "1521", "1709", "1509", "1508"],
        "reason": "Residual Peggy's / Cartwright card after CHBH moves out.",
    },
]

PLAN_DEFS = [
    {
        "plan_id": "active-field-packet-baseline",
        "title": "Active packet: 1A-1 + 5B + 6",
        "parts": [{"field_route_code": "1A-1"}, {"field_route_code": "5B"}, {"field_route_code": "6"}],
        "notes": "Current phone-packet route ownership.",
    },
    {
        "plan_id": "generated-current-partition",
        "title": "Generated control: 36th only + current 5B + current 6",
        "parts": [
            {"candidate_base_variant_id": "lower36-36th-only"},
            {"candidate_base_variant_id": "cartwright-5b-current"},
            {"candidate_base_variant_id": "cartwright-6-current"},
        ],
        "notes": "Apples-to-apples generated control using the current ownership split.",
    },
    {
        "plan_id": "move-chbh-to-chute",
        "title": "Move CHBH to the 36th Street Chute card",
        "parts": [
            {"candidate_base_variant_id": "lower36-36th-chbh"},
            {"candidate_base_variant_id": "cartwright-5b-current"},
            {"candidate_base_variant_id": "cartwright-6-minus-chbh"},
        ],
        "notes": "Keeps Polecat 5 in 5B and removes CHBH from the mega Peggy's / Cartwright card.",
    },
    {
        "plan_id": "move-chbh-and-polecat5-to-chute",
        "title": "Move CHBH and Polecat 5 to the 36th Street Chute card",
        "parts": [
            {"candidate_base_variant_id": "lower36-36th-chbh-polecat5"},
            {"candidate_base_variant_id": "cartwright-5b-minus-polecat5"},
            {"candidate_base_variant_id": "cartwright-6-minus-chbh"},
        ],
        "notes": "User-suggested split that avoids leaving CHBH and Polecat 5 in Peggy's.",
    },
    {
        "plan_id": "move-chbh-to-polecat-core",
        "title": "Move CHBH into the Polecat core card",
        "parts": [
            {"candidate_base_variant_id": "lower36-36th-only"},
            {"candidate_base_variant_id": "cartwright-5b-plus-chbh"},
            {"candidate_base_variant_id": "cartwright-6-minus-chbh"},
        ],
        "notes": "Keeps a three-route plan but makes the north route own CHBH.",
    },
    {
        "plan_id": "separate-cartwright-chbh-polecat5-card",
        "title": "Separate CHBH + Polecat 5 Cartwright mini-card",
        "parts": [
            {"candidate_base_variant_id": "lower36-36th-only"},
            {"candidate_base_variant_id": "cartwright-chbh-polecat5"},
            {"candidate_base_variant_id": "cartwright-5b-minus-polecat5"},
            {"candidate_base_variant_id": "cartwright-6-minus-chbh"},
        ],
        "notes": "Avoids one mega track by splitting CHBH/Polecat 5 into its own Cartwright card.",
    },
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def sort_id(value: Any) -> tuple[int, str]:
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


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "route"


def scaled_minutes(candidate_miles: float, current_miles: float, current_minutes: int) -> int:
    if current_miles <= 0:
        return 0
    return int(round(candidate_miles * (current_minutes / current_miles)))


def route_by_code(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    routes = field_tool_data.get("routes") or []
    return {str(route.get("route_code") or route.get("label")): route for route in routes}


def field_route_metrics(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "field_route",
        "id": str(route.get("route_code") or route.get("label")),
        "title": f"Active route {route.get('route_code') or route.get('label')}",
        "track_miles": round_miles(float_value(route.get("on_foot_miles"))),
        "official_miles": round_miles(float_value(route.get("official_miles"))),
        "p75_minutes": int_value(route.get("door_to_door_minutes_p75")),
        "p90_minutes": int_value(route.get("door_to_door_minutes_p90")),
        "segment_ids": normalized_ids(route.get("segment_ids") or []),
        "official_repeat_miles": 0.0,
        "direct_gap_fallback_miles": 0.0,
        "source": "active_field_packet",
    }


def sanitize_parking(parking: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": parking.get("name") or parking.get("display_name"),
        "display_name": parking.get("display_name"),
        "source": parking.get("source"),
        "has_parking": parking.get("has_parking"),
        "parking_confidence": parking.get("parking_confidence"),
        "nearest_open_trail_name": parking.get("nearest_open_trail_name"),
        "nearest_open_trail_label": parking.get("nearest_open_trail_label"),
    }


def render_route_gpx(output_dir: Path, candidate: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slugify(candidate['variant_id'])}.gpx"
    path.write_text(
        render_gpx_segments(
            f"CHBH/Polecat recomposition audit: {candidate['variant_id']}",
            [candidate["track_coordinates"]],
        ),
        encoding="utf-8",
    )
    return path


def candidate_rank_key(candidate: dict[str, Any]) -> tuple[int, int, float, float]:
    validation = candidate.get("gpx_validation") or {}
    gpx_passed = validation.get("passed") or candidate.get("gpx_status") == "passed"
    return (
        0 if gpx_passed else 1,
        0 if float_value(candidate.get("direct_gap_fallback_miles")) == 0 else 1,
        float_value(candidate.get("track_miles")),
        float_value(candidate.get("official_repeat_miles")),
    )


def summarize_candidate(
    candidate: dict[str, Any],
    variant_def: dict[str, Any],
    gpx_path: Path,
    current_scope: dict[str, Any],
) -> dict[str, Any]:
    p75 = scaled_minutes(candidate["track_miles"], current_scope["track_miles"], current_scope["p75_minutes"])
    p90 = scaled_minutes(candidate["track_miles"], current_scope["track_miles"], current_scope["p90_minutes"])
    validation = candidate.get("gpx_validation") or {}
    repeat_ids = normalized_ids(candidate.get("non_template_repeat_segment_ids") or [])
    self_repeat_ids = normalized_ids(candidate.get("self_repeat_segment_ids") or [])
    return {
        "base_variant_id": variant_def["base_variant_id"],
        "variant_id": candidate["variant_id"],
        "title": variant_def["title"],
        "reason": variant_def["reason"],
        "strategy": candidate["strategy"],
        "status": candidate["status"],
        "track_miles": candidate["track_miles"],
        "official_miles": candidate["official_miles"],
        "connector_miles": candidate["connector_miles"],
        "official_repeat_miles": candidate["official_repeat_miles"],
        "direct_gap_fallback_miles": candidate["direct_gap_fallback_miles"],
        "p75_minutes_scaled": p75,
        "p90_minutes_scaled": p90,
        "pricing_status": "scaled_from_current_1a1_5b_6_scope_needs_dem_and_field_calibration",
        "official_segment_ids": candidate["official_segment_ids"],
        "traversed_segment_order": candidate["traversed_segment_order"],
        "coverage_status": (candidate.get("coverage_validation") or {}).get("status"),
        "gpx_status": "passed" if validation.get("passed") else "failed",
        "ascent_direction_validation": candidate.get("ascent_direction_validation"),
        "self_repeat_segment_ids": self_repeat_ids,
        "non_template_repeat_segment_ids": repeat_ids,
        "repeat_review_status": "needs_repeat_credit_review" if repeat_ids or self_repeat_ids else "no_hidden_repeat_detected",
        "cue_complexity": candidate.get("cue_complexity"),
        "gpx_path": display_path(gpx_path),
    }


def candidate_part_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "generated_candidate",
        "id": summary["base_variant_id"],
        "variant_id": summary["variant_id"],
        "title": summary["title"],
        "strategy": summary["strategy"],
        "track_miles": summary["track_miles"],
        "official_miles": summary["official_miles"],
        "p75_minutes": summary["p75_minutes_scaled"],
        "p90_minutes": summary["p90_minutes_scaled"],
        "segment_ids": summary["official_segment_ids"],
        "official_repeat_miles": summary["official_repeat_miles"],
        "direct_gap_fallback_miles": summary["direct_gap_fallback_miles"],
        "source": "generated_private_audit",
    }


def build_plan_summary(
    plan_def: dict[str, Any],
    active_routes: dict[str, dict[str, Any]],
    best_candidates: dict[str, dict[str, Any]],
    official_by_id: dict[str, dict[str, Any]],
    active_baseline: dict[str, Any] | None,
    generated_control: dict[str, Any] | None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for part in plan_def["parts"]:
        route_code = part.get("field_route_code")
        candidate_id = part.get("candidate_base_variant_id")
        if route_code:
            parts.append(field_route_metrics(active_routes[route_code]))
        elif candidate_id:
            parts.append(candidate_part_metrics(best_candidates[candidate_id]))
        else:
            raise ValueError(f"Unknown plan part: {part}")
    segment_ids = normalized_ids(segment_id for part in parts for segment_id in part["segment_ids"])
    current_ids = set(CURRENT_SEGMENT_SET)
    segment_set = set(segment_ids)
    total = {
        "plan_id": plan_def["plan_id"],
        "title": plan_def["title"],
        "notes": plan_def["notes"],
        "route_count": len(parts),
        "track_miles": round_miles(sum(float_value(part["track_miles"]) for part in parts)),
        "official_miles": round_miles(
            sum(float_value(official_by_id[segment_id]["official_miles"]) for segment_id in segment_ids)
        ),
        "p75_minutes": sum(int_value(part["p75_minutes"]) for part in parts),
        "p90_minutes": sum(int_value(part["p90_minutes"]) for part in parts),
        "official_repeat_miles": round_miles(sum(float_value(part["official_repeat_miles"]) for part in parts)),
        "direct_gap_fallback_miles": round_miles(sum(float_value(part["direct_gap_fallback_miles"]) for part in parts)),
        "segment_ids": segment_ids,
        "segment_count": len(segment_ids),
        "missing_current_segment_ids": sorted(current_ids - segment_set, key=sort_id),
        "extra_segment_ids": sorted(segment_set - current_ids, key=sort_id),
        "parts": parts,
    }
    if active_baseline:
        total["delta_vs_active_field_packet_miles"] = round_miles(total["track_miles"] - active_baseline["track_miles"])
        total["delta_vs_active_field_packet_p75_minutes"] = total["p75_minutes"] - active_baseline["p75_minutes"]
    else:
        total["delta_vs_active_field_packet_miles"] = 0.0
        total["delta_vs_active_field_packet_p75_minutes"] = 0
    if generated_control:
        total["delta_vs_generated_current_partition_miles"] = round_miles(total["track_miles"] - generated_control["track_miles"])
        total["delta_vs_generated_current_partition_p75_minutes"] = total["p75_minutes"] - generated_control["p75_minutes"]
    else:
        total["delta_vs_generated_current_partition_miles"] = None
        total["delta_vs_generated_current_partition_p75_minutes"] = None
    total["promotion_status"] = (
        "generated_candidate_needs_full_route_certification"
        if any(part["kind"] == "generated_candidate" for part in parts)
        else "active_field_packet_baseline"
    )
    total["coverage_status"] = "covers_current_scope" if not total["missing_current_segment_ids"] else "missing_current_scope_segments"
    total["continuity_status"] = (
        "needs_direct_gap_review" if float_value(total["direct_gap_fallback_miles"]) > 0 else "no_direct_gap_fallbacks"
    )
    return total


def build_report(
    field_tool_data: dict[str, Any],
    official_segments: list[dict[str, Any]],
    connector_graph: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    active_routes = route_by_code(field_tool_data)
    missing_routes = [route_code for route_code in ROUTE_CODES if route_code not in active_routes]
    if missing_routes:
        raise ValueError(f"Missing active routes in field-tool data: {', '.join(missing_routes)}")
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    current_scope = {
        "track_miles": round_miles(sum(float_value(active_routes[code].get("on_foot_miles")) for code in ROUTE_CODES)),
        "official_miles": round_miles(sum(float_value(active_routes[code].get("official_miles")) for code in ROUTE_CODES)),
        "p75_minutes": sum(int_value(active_routes[code].get("door_to_door_minutes_p75")) for code in ROUTE_CODES),
        "p90_minutes": sum(int_value(active_routes[code].get("door_to_door_minutes_p90")) for code in ROUTE_CODES),
    }
    generated_summaries: list[dict[str, Any]] = []
    generated_full: list[dict[str, Any]] = []
    by_base: dict[str, list[dict[str, Any]]] = {}
    gpx_dir = output_dir / "gpx"
    for variant_def in VARIANTS:
        parking_route = active_routes[variant_def["parking_route_code"]]
        parking = dict(parking_route.get("parking") or {})
        for strategy in ("template_sequence_greedy", "nearest_segment_greedy"):
            candidate = build_generated_route(
                variant_id=f"{variant_def['base_variant_id']}--{strategy}",
                strategy=strategy,
                ordered_segment_ids=variant_def["ordered_segment_ids"],
                parking=parking,
                official_by_id=official_by_id,
                connector_graph=connector_graph,
            )
            gpx_path = render_route_gpx(gpx_dir, candidate)
            summary = summarize_candidate(candidate, variant_def, gpx_path, current_scope)
            generated_summaries.append(summary)
            by_base.setdefault(variant_def["base_variant_id"], []).append(summary)
            candidate["gpx_path"] = display_path(gpx_path)
            candidate["parking"] = sanitize_parking(candidate.get("parking") or {})
            candidate.pop("track_coordinates", None)
            generated_full.append(candidate)
    best_candidates = {
        base_id: min(rows, key=candidate_rank_key)
        for base_id, rows in by_base.items()
    }
    active_baseline = None
    generated_control = None
    plan_summaries: list[dict[str, Any]] = []
    for plan_def in PLAN_DEFS:
        plan = build_plan_summary(
            plan_def,
            active_routes,
            best_candidates,
            official_by_id,
            active_baseline,
            generated_control,
        )
        if plan["plan_id"] == "active-field-packet-baseline":
            active_baseline = plan
            plan["delta_vs_active_field_packet_miles"] = 0.0
            plan["delta_vs_active_field_packet_p75_minutes"] = 0
        if plan["plan_id"] == "generated-current-partition":
            generated_control = plan
            plan["delta_vs_generated_current_partition_miles"] = 0.0
            plan["delta_vs_generated_current_partition_p75_minutes"] = 0
        if active_baseline and plan["plan_id"] != "active-field-packet-baseline":
            plan["delta_vs_active_field_packet_miles"] = round_miles(plan["track_miles"] - active_baseline["track_miles"])
            plan["delta_vs_active_field_packet_p75_minutes"] = plan["p75_minutes"] - active_baseline["p75_minutes"]
        if generated_control and plan["plan_id"] not in {"active-field-packet-baseline", "generated-current-partition"}:
            plan["delta_vs_generated_current_partition_miles"] = round_miles(
                plan["track_miles"] - generated_control["track_miles"]
            )
            plan["delta_vs_generated_current_partition_p75_minutes"] = plan["p75_minutes"] - generated_control["p75_minutes"]
        plan_summaries.append(plan)
    generated_plans = [
        plan
        for plan in plan_summaries
        if plan["plan_id"] not in {"active-field-packet-baseline", "generated-current-partition"}
        and plan["coverage_status"] == "covers_current_scope"
    ]
    promotion_comparable_plans = [
        plan
        for plan in generated_plans
        if plan["continuity_status"] == "no_direct_gap_fallbacks"
    ]
    ranked_pool = promotion_comparable_plans or generated_plans
    best_plan = min(
        ranked_pool,
        key=lambda row: (
            0 if row["continuity_status"] == "no_direct_gap_fallbacks" else 1,
            float_value(row["direct_gap_fallback_miles"]),
            float_value(row["track_miles"]),
            row["route_count"],
        ),
    )
    chbh_owner = next(
        route.get("route_code")
        for route in active_routes.values()
        if "1516" in normalized_ids(route.get("segment_ids") or [])
    )
    summary = {
        "current_chbh_owner_route_code": chbh_owner,
        "current_scope_track_miles": current_scope["track_miles"],
        "current_scope_p75_minutes": current_scope["p75_minutes"],
        "best_generated_plan_id": best_plan["plan_id"],
        "best_generated_plan_track_miles": best_plan["track_miles"],
        "best_generated_plan_delta_vs_active_miles": best_plan["delta_vs_active_field_packet_miles"],
        "best_generated_plan_route_count": best_plan["route_count"],
        "promotion_comparable_plan_count": len(promotion_comparable_plans),
        "ranked_plan_basis": "promotion_comparable_no_direct_gap"
        if promotion_comparable_plans
        else "review_seed_all_generated_plans_have_direct_gap_fallbacks",
        "user_suggested_plan_id": "move-chbh-and-polecat5-to-chute",
        "user_suggested_plan": next(
            plan for plan in plan_summaries if plan["plan_id"] == "move-chbh-and-polecat5-to-chute"
        ),
        "recommendation": (
            "move_chbh_out_of_route_6_but_do_not_promote_until_direct_gap_and_full_route_certification"
        ),
    }
    return {
        "schema": "boise_trails_chbh_polecat_recomposition_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "private_generated_comparison_not_active_route_card",
        "source_files": {
            "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
            "official_segments_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "connector_geojson": display_path(DEFAULT_CONNECTOR_GEOJSON),
        },
        "scope": {
            "route_codes_compared": list(ROUTE_CODES),
            "segment_ids_in_current_scope": CURRENT_SEGMENT_SET,
            "privacy": "Generated GPX files live under years/2026/outputs/private and may include private parking-anchor coordinates.",
            "promotion_policy": "Do not promote any generated candidate without route-card cues, access/current-condition checks, special-management review, and field-packet recertification.",
        },
        "current_active_routes": {
            route_code: {
                "label": active_routes[route_code].get("label"),
                "trailhead": active_routes[route_code].get("trailhead"),
                "official_miles": active_routes[route_code].get("official_miles"),
                "on_foot_miles": active_routes[route_code].get("on_foot_miles"),
                "p75_minutes": active_routes[route_code].get("door_to_door_minutes_p75"),
                "p90_minutes": active_routes[route_code].get("door_to_door_minutes_p90"),
                "segment_ids": active_routes[route_code].get("segment_ids"),
            }
            for route_code in ROUTE_CODES
        },
        "summary": summary,
        "plan_summaries": plan_summaries,
        "best_candidate_by_base_variant": best_candidates,
        "candidate_summaries": generated_summaries,
        "generated_candidates": generated_full,
        "outputs": {
            "gpx_paths": [summary["gpx_path"] for summary in generated_summaries],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    user_plan = summary["user_suggested_plan"]
    lines = [
        "# CHBH / Polecat Recomposition Audit",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Readout",
        "",
        f"- Current CHBH owner: route `{summary['current_chbh_owner_route_code']}`.",
        f"- Active 1A-1 + 5B + 6 packet scope: {summary['current_scope_track_miles']} mi / {summary['current_scope_p75_minutes']} scaled p75 minutes.",
        f"- Best generated recomposition seed: `{summary['best_generated_plan_id']}` at {summary['best_generated_plan_track_miles']} mi ({summary['best_generated_plan_delta_vs_active_miles']:+.2f} mi vs active packet metrics).",
        f"- User-suggested CHBH + Polecat 5 to Chute seed: {user_plan['track_miles']} mi ({user_plan['delta_vs_active_field_packet_miles']:+.2f} mi vs active packet metrics), {user_plan['route_count']} routes.",
        "",
        "These are generated candidates, not promoted field cards. Timing is scaled from the current 1A-1/5B/6 packet scope and still needs DEM/field calibration.",
        "",
        "## Plan Comparison",
        "",
        "| Plan | Routes | Miles | Delta vs active | Scaled p75 | Delta p75 | Direct gap | Repeat | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for plan in report["plan_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{plan['plan_id']}`",
                    str(plan["route_count"]),
                    str(plan["track_miles"]),
                    f"{plan['delta_vs_active_field_packet_miles']:+.2f}",
                    str(plan["p75_minutes"]),
                    f"{plan['delta_vs_active_field_packet_p75_minutes']:+d}",
                    str(plan["direct_gap_fallback_miles"]),
                    str(plan["official_repeat_miles"]),
                    plan["coverage_status"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best Candidate Per Segment Set",
            "",
            "| Variant | Strategy | Miles | Official | Repeat | Direct gap | Ascent | GPX |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for base_id, row in report["best_candidate_by_base_variant"].items():
        ascent = row.get("ascent_direction_validation") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{base_id}`",
                    f"`{row['strategy']}`",
                    str(row["track_miles"]),
                    str(row["official_miles"]),
                    str(row["official_repeat_miles"]),
                    str(row["direct_gap_fallback_miles"]),
                    ascent.get("status") or "",
                    row["gpx_path"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate Notes",
            "",
            "- Generated GPX is private evidence only; the active route packet is unchanged.",
            "- Any candidate with Polecat 5 still needs a field-facing special-management/signage check before promotion.",
            "- Current conditions, closures, heat, water, and parking were not refreshed in this audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_data = read_json(args.field_tool_data_json)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    report = build_report(field_tool_data, official_segments, connector_graph, args.output_dir)
    report["source_files"] = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "official_segments_geojson": display_path(args.official_geojson),
        "connector_geojson": display_path(args.connector_geojson),
    }
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    output_paths = [args.output_json, args.output_md] + [Path(path) for path in report["outputs"]["gpx_paths"]]
    manifest = build_artifact_manifest(
        run_id="chbh_polecat_recomposition_audit",
        command="python years/2026/scripts/chbh_polecat_recomposition_audit.py",
        inputs=[args.field_tool_data_json, args.official_geojson, args.connector_geojson],
        outputs=output_paths,
        metadata={"status": report.get("status"), "summary": report.get("summary")},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
