#!/usr/bin/env python3
"""Prove a planning-level net-effort reduction for a proposed route repair."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]

DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_SOURCE_ACTIVITY_REVIEW_JSON = (
    YEAR_DIR / "checkpoints" / "15a-1-latent-shingle-credit-review-2026-05-11.json"
)
DEFAULT_REPLACEMENT_PROBE_JSON = (
    YEAR_DIR / "checkpoints" / "manual-access-anchor-probe-shingle-lower-2026-05-06.json"
)
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "15a-16a-net-effort-reduction-proof-2026-05-11.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "15a-16a-net-effort-reduction-proof-2026-05-11.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=lambda item: (len(item), item))


def as_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def rounded(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def find_route(routes: list[dict[str, Any]], *, outing_id: str, label: str | None = None) -> dict[str, Any]:
    for route in routes:
        if str(route.get("outing_id")) == outing_id:
            return route
    if label:
        for route in routes:
            if str(route.get("label")) == label:
                return route
    raise ValueError(f"Missing route {outing_id or label}")


def find_segment_review(activity_review: dict[str, Any], segment_id: str) -> dict[str, Any] | None:
    for row in activity_review.get("segment_reviews") or []:
        if str(row.get("seg_id")) == str(segment_id):
            return row
    return None


def find_probe_row(probe: dict[str, Any], segment_id: str) -> dict[str, Any] | None:
    for row in probe.get("probe_rows") or []:
        if str(row.get("seg_id")) == str(segment_id):
            return row
    return None


def route_ids(route: dict[str, Any]) -> list[str]:
    return normalized_ids(route.get("segment_ids") or [])


def route_totals(routes: list[dict[str, Any]]) -> dict[str, Any]:
    assigned_ids = [segment_id for route in routes for segment_id in route_ids(route)]
    unique_ids = sorted(set(assigned_ids), key=lambda item: (len(item), item))
    duplicates = sorted(
        {segment_id for segment_id in assigned_ids if assigned_ids.count(segment_id) > 1},
        key=lambda item: (len(item), item),
    )
    return {
        "route_count": len(routes),
        "unique_segment_count": len(unique_ids),
        "duplicate_segment_ids": duplicates,
        "segment_ids": unique_ids,
        "on_foot_miles": rounded(sum(as_number(route.get("on_foot_miles")) for route in routes)),
        "door_to_door_minutes_p75": int(
            sum(as_number(route.get("door_to_door_minutes_p75")) for route in routes)
        ),
        "door_to_door_minutes_p90": int(
            sum(as_number(route.get("door_to_door_minutes_p90")) for route in routes)
        ),
    }


def build_gate(name: str, passed: bool, detail: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "evidence": evidence or {},
    }


def has_timing_and_effort(row: dict[str, Any]) -> bool:
    return all(
        row.get(key) is not None
        for key in (
            "door_to_door_p75_minutes",
            "door_to_door_p90_minutes",
            "moving_effort_p75_minutes",
            "ascent_ft",
            "grade_adjusted_miles",
        )
    )


def build_proposed_routes(
    routes: list[dict[str, Any]],
    *,
    source_outing_id: str,
    latent_segment_id: str,
    replacement_outing_id: str,
    retained_segment_id: str,
    retained_probe_row: dict[str, Any],
) -> list[dict[str, Any]]:
    proposed = deepcopy(routes)
    source = find_route(proposed, outing_id=source_outing_id)
    replacement = find_route(proposed, outing_id=replacement_outing_id)

    source_ids = route_ids(source)
    if latent_segment_id not in source_ids:
        source["segment_ids"] = normalized_ids([*source_ids, latent_segment_id])
        source["remaining_segment_count"] = len(source["segment_ids"])

    replacement["segment_ids"] = [str(retained_segment_id)]
    replacement["remaining_segment_count"] = 1
    replacement["trails"] = [retained_probe_row.get("trail_name") or replacement.get("trails", ["Unknown"])[0]]
    replacement["official_miles"] = rounded(as_number(retained_probe_row.get("official_miles")))
    replacement["on_foot_miles"] = rounded(as_number(retained_probe_row.get("on_foot_miles")))
    replacement["door_to_door_minutes_p75"] = int(
        as_number(retained_probe_row.get("door_to_door_p75_minutes"))
    )
    replacement["door_to_door_minutes_p90"] = int(
        as_number(retained_probe_row.get("door_to_door_p90_minutes"))
    )
    replacement["effort"] = {
        "ascent_ft": retained_probe_row.get("ascent_ft"),
        "grade_adjusted_miles": retained_probe_row.get("grade_adjusted_miles"),
        "estimated_moving_minutes_p75": retained_probe_row.get("moving_effort_p75_minutes"),
        "elevation_source": "dem",
    }
    replacement["proof_note"] = "Proposed proof-only replacement from manual access anchor probe; active packet not promoted."
    return proposed


def build_net_effort_reduction_proof(
    field_tool_data: dict[str, Any],
    source_activity_review: dict[str, Any],
    replacement_probe: dict[str, Any],
    *,
    source_outing_id: str = "15-1",
    source_label: str = "15A-1",
    latent_segment_id: str = "1656",
    replacement_outing_id: str = "16-2",
    replacement_label: str = "16A-2",
    retained_segment_id: str = "1653",
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    source_route = find_route(routes, outing_id=source_outing_id, label=source_label)
    replacement_route = find_route(routes, outing_id=replacement_outing_id, label=replacement_label)
    retained_probe_row = find_probe_row(replacement_probe, retained_segment_id)
    if retained_probe_row is None:
        raise ValueError(f"Missing probe row for retained segment {retained_segment_id}")
    latent_review = find_segment_review(source_activity_review, latent_segment_id)

    proposed_routes = build_proposed_routes(
        routes,
        source_outing_id=source_outing_id,
        latent_segment_id=str(latent_segment_id),
        replacement_outing_id=replacement_outing_id,
        retained_segment_id=str(retained_segment_id),
        retained_probe_row=retained_probe_row,
    )

    current_totals = route_totals(routes)
    proposed_totals = route_totals(proposed_routes)
    expected_segment_count = int(
        field_tool_data.get("summary", {}).get("segment_count_in_field_menu")
        or current_totals["unique_segment_count"]
    )
    current_segment_set = set(current_totals["segment_ids"])
    proposed_segment_set = set(proposed_totals["segment_ids"])
    missing_after_repair = sorted(current_segment_set - proposed_segment_set, key=lambda item: (len(item), item))
    added_after_repair = sorted(proposed_segment_set - current_segment_set, key=lambda item: (len(item), item))

    on_foot_delta = rounded(proposed_totals["on_foot_miles"] - current_totals["on_foot_miles"])
    p75_delta = proposed_totals["door_to_door_minutes_p75"] - current_totals["door_to_door_minutes_p75"]
    p90_delta = proposed_totals["door_to_door_minutes_p90"] - current_totals["door_to_door_minutes_p90"]

    source_extra_ids = set(normalized_ids(source_activity_review.get("extra_completed_segment_ids") or []))
    source_missed_ids = normalized_ids(source_activity_review.get("missed_segment_ids") or [])
    gates = [
        build_gate(
            "active_packet_segment_count_matches_summary",
            current_totals["unique_segment_count"] == expected_segment_count,
            f"Current active menu has {current_totals['unique_segment_count']} unique segments; expected {expected_segment_count}.",
            {"current_unique_segment_count": current_totals["unique_segment_count"], "expected_segment_count": expected_segment_count},
        ),
        build_gate(
            "source_route_already_covers_latent_segment",
            str(latent_segment_id) in source_extra_ids,
            f"{source_label} activity review lists {latent_segment_id} as extra completed credit.",
            {"extra_completed_segment_ids": normalized_ids(source_extra_ids)},
        ),
        build_gate(
            "source_route_latent_segment_full_credit",
            bool(
                latent_review
                and latent_review.get("completion_status") == "completed"
                and latent_review.get("endpoints_ok") is True
                and latent_review.get("direction_ok") is True
                and as_number(latent_review.get("match_fraction")) >= 0.85
            ),
            f"{source_label} review must cover {latent_segment_id} end-to-end and in the required direction.",
            latent_review or {},
        ),
        build_gate(
            "source_route_has_no_missed_planned_segments",
            not source_missed_ids,
            f"{source_label} review has no missed planned segments.",
            {"missed_segment_ids": source_missed_ids},
        ),
        build_gate(
            "replacement_probe_track_valid",
            bool(retained_probe_row.get("validation_passed") and retained_probe_row.get("track_validation_passed")),
            f"Sheep-only retained segment {retained_segment_id} has graph and track validation.",
            {
                "validation_passed": retained_probe_row.get("validation_passed"),
                "track_validation_passed": retained_probe_row.get("track_validation_passed"),
                "track_validation": retained_probe_row.get("track_validation"),
            },
        ),
        build_gate(
            "replacement_probe_has_timing_and_effort",
            has_timing_and_effort(retained_probe_row),
            "Sheep-only probe includes p75/p90 timing and DEM effort fields.",
            {
                key: retained_probe_row.get(key)
                for key in (
                    "door_to_door_p75_minutes",
                    "door_to_door_p90_minutes",
                    "moving_effort_p75_minutes",
                    "ascent_ft",
                    "grade_adjusted_miles",
                    "on_foot_miles",
                )
            },
        ),
        build_gate(
            "proposed_assignment_preserves_current_unique_coverage",
            not missing_after_repair and not added_after_repair,
            "Moving Shingle to 15A-1 and retaining Sheep in 16A-2 preserves the current official segment set.",
            {
                "missing_after_repair": missing_after_repair,
                "added_after_repair": added_after_repair,
                "proposed_duplicate_segment_ids": proposed_totals["duplicate_segment_ids"],
            },
        ),
        build_gate(
            "proposed_assignment_has_no_duplicates",
            not proposed_totals["duplicate_segment_ids"],
            "Proposed route ownership has no duplicate official segment assignments.",
            {"duplicate_segment_ids": proposed_totals["duplicate_segment_ids"]},
        ),
        build_gate(
            "proposed_total_on_foot_lower",
            on_foot_delta < 0,
            f"Proposed full menu changes total on-foot miles by {on_foot_delta}.",
            {"current_on_foot_miles": current_totals["on_foot_miles"], "proposed_on_foot_miles": proposed_totals["on_foot_miles"]},
        ),
        build_gate(
            "proposed_total_p75_lower",
            p75_delta < 0,
            f"Proposed full menu changes total p75 minutes by {p75_delta}.",
            {
                "current_p75_minutes": current_totals["door_to_door_minutes_p75"],
                "proposed_p75_minutes": proposed_totals["door_to_door_minutes_p75"],
            },
        ),
    ]
    status = "proved_planning_net_effort_reduction" if all(gate["passed"] for gate in gates) else "not_proven"

    current_replacement = {
        "outing_id": replacement_route.get("outing_id"),
        "label": replacement_route.get("label"),
        "segment_ids": route_ids(replacement_route),
        "official_miles": replacement_route.get("official_miles"),
        "on_foot_miles": replacement_route.get("on_foot_miles"),
        "door_to_door_minutes_p75": replacement_route.get("door_to_door_minutes_p75"),
        "door_to_door_minutes_p90": replacement_route.get("door_to_door_minutes_p90"),
        "effort": replacement_route.get("effort"),
        "validation": replacement_route.get("validation"),
    }
    proposed_replacement = find_route(proposed_routes, outing_id=replacement_outing_id)
    source_repair = find_route(proposed_routes, outing_id=source_outing_id)

    return {
        "schema": "boise_trails_net_effort_reduction_proof_v1",
        "proved_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "proof_level": "planning_menu_repricing",
        "source_files": source_files or {},
        "scope": {
            "proves": [
                "current active full-menu p75/on-foot baseline",
                "proposed segment ownership preserves all current official segment coverage",
                "proposed 16A-2 Sheep-only replacement lowers full-menu on-foot miles and p75/p90 time",
            ],
            "does_not_prove": [
                "official BTC app credit before a real challenge-window activity is validated",
                "field-packet promotion or cue readiness for the proposed replacement",
                "day-of legality, closures, heat, mud, water, or signage status",
            ],
        },
        "summary": {
            "status": status,
            "current_total_on_foot_miles": current_totals["on_foot_miles"],
            "proposed_total_on_foot_miles": proposed_totals["on_foot_miles"],
            "reduction_on_foot_miles": rounded(-on_foot_delta),
            "current_total_p75_minutes": current_totals["door_to_door_minutes_p75"],
            "proposed_total_p75_minutes": proposed_totals["door_to_door_minutes_p75"],
            "reduction_p75_minutes": -p75_delta,
            "current_total_p90_minutes": current_totals["door_to_door_minutes_p90"],
            "proposed_total_p90_minutes": proposed_totals["door_to_door_minutes_p90"],
            "reduction_p90_minutes": -p90_delta,
            "current_unique_segment_count": current_totals["unique_segment_count"],
            "proposed_unique_segment_count": proposed_totals["unique_segment_count"],
            "route_count": current_totals["route_count"],
        },
        "route_change": {
            "source_route": {
                "outing_id": source_route.get("outing_id"),
                "label": source_route.get("label"),
                "current_segment_ids": route_ids(source_route),
                "proposed_segment_ids": route_ids(source_repair),
                "on_foot_miles_change": 0.0,
                "p75_minutes_change": 0,
                "evidence": {
                    "latent_segment_id": str(latent_segment_id),
                    "latent_segment_review": latent_review,
                },
            },
            "replacement_route": {
                "current": current_replacement,
                "proposed": {
                    "outing_id": proposed_replacement.get("outing_id"),
                    "label": proposed_replacement.get("label"),
                    "segment_ids": route_ids(proposed_replacement),
                    "official_miles": proposed_replacement.get("official_miles"),
                    "on_foot_miles": proposed_replacement.get("on_foot_miles"),
                    "door_to_door_minutes_p75": proposed_replacement.get("door_to_door_minutes_p75"),
                    "door_to_door_minutes_p90": proposed_replacement.get("door_to_door_minutes_p90"),
                    "official_repeat_miles": retained_probe_row.get("official_repeat_miles"),
                    "connector_miles": retained_probe_row.get("connector_miles"),
                    "road_miles": retained_probe_row.get("road_miles"),
                    "effort": proposed_replacement.get("effort"),
                    "candidate_id": retained_probe_row.get("candidate_id"),
                },
                "delta": {
                    "on_foot_miles": rounded(
                        as_number(proposed_replacement.get("on_foot_miles"))
                        - as_number(replacement_route.get("on_foot_miles"))
                    ),
                    "door_to_door_minutes_p75": int(
                        as_number(proposed_replacement.get("door_to_door_minutes_p75"))
                        - as_number(replacement_route.get("door_to_door_minutes_p75"))
                    ),
                    "door_to_door_minutes_p90": int(
                        as_number(proposed_replacement.get("door_to_door_minutes_p90"))
                        - as_number(replacement_route.get("door_to_door_minutes_p90"))
                    ),
                },
            },
        },
        "coverage": {
            "missing_after_repair": missing_after_repair,
            "added_after_repair": added_after_repair,
            "current_duplicate_segment_ids": current_totals["duplicate_segment_ids"],
            "proposed_duplicate_segment_ids": proposed_totals["duplicate_segment_ids"],
        },
        "gates": gates,
        "promotion_status": {
            "active_packet_promoted": False,
            "required_before_promotion": [
                "turn this proof into a real route-card replacement source",
                "regenerate the field packet from source",
                "run GPX human-validity review and field-packet audits",
                "do day-of Ridge to Rivers / closure / signage / weather checks",
            ],
        },
    }


def render_markdown(proof: dict[str, Any]) -> str:
    summary = proof["summary"]
    route_delta = proof["route_change"]["replacement_route"]["delta"]
    lines = [
        "# 15A / 16A net effort reduction proof",
        "",
        f"- Status: `{proof['status']}`",
        f"- Proof level: `{proof['proof_level']}`",
        f"- Full-menu on-foot miles: {summary['current_total_on_foot_miles']} -> {summary['proposed_total_on_foot_miles']} ({summary['reduction_on_foot_miles']} mi saved)",
        f"- Full-menu p75 minutes: {summary['current_total_p75_minutes']} -> {summary['proposed_total_p75_minutes']} ({summary['reduction_p75_minutes']} min saved)",
        f"- Full-menu p90 minutes: {summary['current_total_p90_minutes']} -> {summary['proposed_total_p90_minutes']} ({summary['reduction_p90_minutes']} min saved)",
        f"- Official segment coverage: {summary['current_unique_segment_count']} -> {summary['proposed_unique_segment_count']}",
        "",
        "## Route Change",
        "",
        "- `15A-1` keeps the same GPX/on-foot effort but claims Shingle Creek segment `1656`, which its current GPX already covers end-to-end in ascent direction.",
        "- `16A-2` changes from Shingle Creek + Sheep Camp to the Sheep Camp-only probe for segment `1653`.",
        f"- `16A-2` local delta: {route_delta['on_foot_miles']} on-foot miles, {route_delta['door_to_door_minutes_p75']} p75 minutes, {route_delta['door_to_door_minutes_p90']} p90 minutes.",
        "",
        "## Gates",
        "",
    ]
    for gate in proof["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- {status}: `{gate['name']}` - {gate['detail']}")
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- This proves a planning-level net human-effort reduction against the current full active menu.",
            "- It does not prove official BTC credit until a real challenge-window BTC activity validates the segment.",
            "- It does not promote the active field packet; promotion still needs a source route-card replacement, regeneration, human-validity review, and day-of condition/access checks.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--source-activity-review-json", type=Path, default=DEFAULT_SOURCE_ACTIVITY_REVIEW_JSON)
    parser.add_argument("--replacement-probe-json", type=Path, default=DEFAULT_REPLACEMENT_PROBE_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proof = build_net_effort_reduction_proof(
        read_json(args.field_tool_data_json),
        read_json(args.source_activity_review_json),
        read_json(args.replacement_probe_json),
        source_files={
            "field_tool_data_json": str(args.field_tool_data_json),
            "source_activity_review_json": str(args.source_activity_review_json),
            "replacement_probe_json": str(args.replacement_probe_json),
        },
    )
    write_json(args.output_json, proof)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(proof), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": proof["status"], **proof["summary"]}, indent=2))
    return 0 if proof["status"] == "proved_planning_net_effort_reduction" else 1


if __name__ == "__main__":
    raise SystemExit(main())
