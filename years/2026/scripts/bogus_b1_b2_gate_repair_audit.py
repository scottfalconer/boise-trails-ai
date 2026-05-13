#!/usr/bin/env python3
"""Run a first-pass gate repair audit for Bogus B1/B2 candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from freestone_cluster_route_generation_experiment import (  # noqa: E402
    display_path,
    float_value,
    normalized_ids,
    round_miles,
    sort_id,
    write_json,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_TEMPLATE_CANDIDATES_JSON = YEAR_DIR / "checkpoints" / "template-route-candidates-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "bogus-b1-b2-gate-repair-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "bogus-b1-b2-gate-repair-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "bogus-b1-b2-gate-repair-audit-2026-05-12-manifest.json"

BOGUS_BUNDLE_IDS = ["B1-simplot-side-bogus-day", "B2-pioneer-mores-side-day"]

DIRECT_GAP_REPAIR_CUE_REUSE = {
    ("B1-simplot-side-bogus-day", "1713"): {
        "source_route_label": "FD07A",
        "cue_type": "start_access",
        "reason": "Reuse the certified Simplot-to-Sunshine start-access cue instead of an opaque direct parking snap.",
    },
    ("B1-simplot-side-bogus-day", "return_to_car"): {
        "source_route_label": "FD25A",
        "cue_type": "exit_access",
        "reason": "Reuse the certified Elk Meadows-to-Simplot return cue instead of an opaque direct return snap.",
    },
    ("B2-pioneer-mores-side-day", "1732"): {
        "source_route_label": "18",
        "cue_type": "connector_road",
        "target_contains": "Mores Mtn Interpretive",
        "reason": "Reuse the certified Lodge/Mores connector cue from route 18 instead of an opaque direct snap.",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def route_by_label(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(route.get("label") or ""): route for route in field_tool_data.get("routes") or []}


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("id") or route.get("label") or "")


def segment_owner_index(field_tool_data: dict[str, Any]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for route in field_tool_data.get("routes") or []:
        label = str(route.get("label") or route_key(route))
        for segment_id in normalized_ids(route.get("segment_ids") or []):
            owners.setdefault(segment_id, []).append(label)
    return owners


def bundle_by_id(template_candidates: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    for bundle in template_candidates.get("bundles") or []:
        if bundle.get("bundle_id") == bundle_id:
            return bundle
    raise ValueError(f"Missing template bundle {bundle_id}")


def find_reusable_cue(
    *,
    routes_by_label: dict[str, dict[str, Any]],
    source_route_label: str,
    cue_type: str | None = None,
    target_contains: str | None = None,
) -> dict[str, Any] | None:
    route = routes_by_label.get(source_route_label)
    if not route:
        return None
    for cue in route.get("wayfinding_cues") or []:
        if cue_type and cue.get("cue_type") != cue_type:
            continue
        if target_contains and target_contains not in json.dumps(cue, sort_keys=True):
            continue
        return cue
    return None


def direct_gap_repairs(
    *,
    bundle_id: str,
    loop: dict[str, Any],
    routes_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    repair_delta_miles = 0.0
    direct_gap_count = 0
    for link in loop.get("link_rows") or []:
        if link.get("path_source") != "direct_gap_fallback":
            continue
        direct_gap_count += 1
        to_segment_id = str(link.get("to_segment_id"))
        mapping = DIRECT_GAP_REPAIR_CUE_REUSE.get((bundle_id, to_segment_id))
        cue = None
        if mapping:
            cue = find_reusable_cue(
                routes_by_label=routes_by_label,
                source_route_label=str(mapping.get("source_route_label")),
                cue_type=mapping.get("cue_type"),
                target_contains=mapping.get("target_contains"),
            )
        original_miles = float_value(link.get("link_distance_miles"))
        replacement_miles = float_value((cue or {}).get("leg_miles")) if cue else None
        if replacement_miles is not None:
            repair_delta_miles += replacement_miles - original_miles
        rows.append(
            {
                "to_segment_id": to_segment_id,
                "to_segment_name": link.get("to_segment_name"),
                "original_direct_gap_miles": round_miles(original_miles),
                "repair_status": "named_route_card_cue_found_but_gpx_not_rebuilt" if cue else "no_named_repair_cue_found",
                "source_route_label": (mapping or {}).get("source_route_label"),
                "source_cue_seq": (cue or {}).get("seq"),
                "source_cue_type": (cue or {}).get("cue_type"),
                "source_cue_signed_as": (cue or {}).get("signed_as"),
                "source_cue_target": (cue or {}).get("target"),
                "replacement_leg_miles": round_miles(replacement_miles) if replacement_miles is not None else None,
                "delta_miles_vs_direct_gap": round_miles((replacement_miles - original_miles) if replacement_miles is not None else 0.0),
                "reason": (mapping or {}).get("reason"),
            }
        )
    original_track_miles = float_value(loop.get("track_miles"))
    return {
        "status": "failed_continuous_gpx_still_direct_gap" if direct_gap_count else "passed_no_direct_gap",
        "direct_gap_count": direct_gap_count,
        "original_direct_gap_miles": round_miles(float_value(loop.get("direct_gap_fallback_miles"))),
        "named_cue_repair_count": sum(1 for row in rows if row["repair_status"] == "named_route_card_cue_found_but_gpx_not_rebuilt"),
        "post_named_cue_priced_track_miles": round_miles(original_track_miles + repair_delta_miles),
        "repair_delta_miles": round_miles(repair_delta_miles),
        "rows": rows,
        "note": "A named cue can explain a gap, but it does not remove the promotion blocker until a continuous generated GPX uses that connector geometry.",
    }


def repeat_and_ownership_review(
    *,
    bundle: dict[str, Any],
    loop: dict[str, Any],
    owner_by_segment: dict[str, list[str]],
) -> dict[str, Any]:
    replaced_labels = set(str(label) for label in bundle.get("replace_route_labels") or [])
    official_repeat_ids = normalized_ids(
        segment_id
        for link in loop.get("link_rows") or []
        for segment_id in link.get("official_repeat_segment_ids") or []
    )
    self_repeat_ids = normalized_ids(loop.get("self_repeat_segment_ids") or [])
    non_template_repeat_ids = normalized_ids(loop.get("non_template_repeat_segment_ids") or [])
    ownership_rows = []
    for segment_id in non_template_repeat_ids:
        owners = owner_by_segment.get(segment_id, [])
        if not owners:
            status = "unowned_latent_credit"
        elif set(owners) <= replaced_labels:
            status = "would_be_unowned_if_replaced"
        else:
            status = "declared_owned_elsewhere"
        ownership_rows.append({"segment_id": segment_id, "owners": owners, "decision": status})
    hidden_self_after_classification = [
        segment_id for segment_id in self_repeat_ids if segment_id not in official_repeat_ids
    ]
    return {
        "status": "failed_unclassified_self_repeat"
        if hidden_self_after_classification
        else ("classified_explicit_priced_repeat" if self_repeat_ids or non_template_repeat_ids else "passed_no_repeat_or_latent_credit"),
        "official_repeat_miles": loop.get("official_repeat_miles"),
        "self_repeat_segment_ids": self_repeat_ids,
        "self_repeat_classification": "explicit_priced_repeat" if self_repeat_ids else "none",
        "hidden_self_repeat_ids_after_classification": hidden_self_after_classification,
        "non_template_repeat_segment_ids": non_template_repeat_ids,
        "ownership_decisions": ownership_rows,
        "unowned_latent_credit_ids": [row["segment_id"] for row in ownership_rows if row["decision"] == "unowned_latent_credit"],
        "would_be_unowned_if_replaced_ids": [row["segment_id"] for row in ownership_rows if row["decision"] == "would_be_unowned_if_replaced"],
        "declared_owned_elsewhere_segment_ids": [row["segment_id"] for row in ownership_rows if row["decision"] == "declared_owned_elsewhere"],
    }


def cue_text_for_link(link: dict[str, Any], direct_gap_rows: list[dict[str, Any]]) -> str:
    to_segment_id = str(link.get("to_segment_id"))
    gap_row = next((row for row in direct_gap_rows if row["to_segment_id"] == to_segment_id), None)
    if gap_row:
        if gap_row.get("source_cue_signed_as"):
            names = " / ".join(gap_row["source_cue_signed_as"])
            return f"Follow {names} toward {gap_row.get('source_cue_target')} ({gap_row.get('replacement_leg_miles')} mi); rebuild GPX before promotion."
        return f"UNRESOLVED direct connector to {link.get('to_segment_name')} ({link.get('link_distance_miles')} mi)."
    names = link.get("connector_names") or []
    trail = link.get("to_trail_name") or link.get("to_segment_name")
    if names and float_value(link.get("connector_miles")) > 0.05:
        connector = " / ".join(str(name) for name in names[:6])
        return f"Connector: follow {connector} toward {trail} ({link.get('link_distance_miles')} mi)."
    if to_segment_id == "return_to_car":
        return f"Return to car via mapped connector ({link.get('link_distance_miles')} mi)."
    return f"Follow {trail} for official credit segment {to_segment_id}."


def build_cue_sheet(loop: dict[str, Any], direct_gap_review: dict[str, Any]) -> dict[str, Any]:
    cues = [
        {
            "seq": 1,
            "cue_type": "park",
            "text": f"Park/start at {((loop.get('parking') or {}).get('name') or loop.get('trailhead'))}.",
        }
    ]
    for index, link in enumerate(loop.get("link_rows") or [], start=2):
        cues.append(
            {
                "seq": index,
                "to_segment_id": str(link.get("to_segment_id")),
                "cue_type": "direct_gap_repair_hold" if link.get("path_source") == "direct_gap_fallback" else "route_leg",
                "text": cue_text_for_link(link, direct_gap_review.get("rows") or []),
            }
        )
    return {
        "status": "draft_not_field_ready" if direct_gap_review["status"] != "passed_no_direct_gap" else "draft_human_readable",
        "cue_count": len(cues),
        "cues": cues,
    }


def scaled_minutes_for_priced_track(current_scope: dict[str, Any], priced_track_miles: float) -> dict[str, Any]:
    current_miles = float_value(current_scope.get("on_foot_miles"))
    if current_miles <= 0:
        return {"status": "unavailable", "p75_minutes": None, "p90_minutes": None}
    p75 = round(priced_track_miles * (int(current_scope.get("p75_minutes") or 0) / current_miles))
    p90 = round(priced_track_miles * (int(current_scope.get("p90_minutes") or 0) / current_miles))
    return {
        "status": "scaled_from_current_scope_not_dem_certified",
        "p75_minutes": int(p75),
        "p90_minutes": int(p90),
    }


def source_review_for_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    is_b1 = bundle["bundle_id"] == "B1-simplot-side-bogus-day"
    return {
        "parking_access": {
            "status": "source_supported_operational_recheck_required",
            "start": "Simplot Lodge Parking Area" if is_b1 else "Pioneer Lodge Parking Area",
            "evidence": [
                {
                    "source": "Ridge to Rivers Bogus Basin Area",
                    "url": "https://www.ridgetorivers.org/trails/trail-areas/bogus-basin-area/",
                    "finding": "Bogus Basin Area page says parking is available at Nordic Lodge and Simplot Lodge.",
                },
                {
                    "source": "Bogus Basin Getting Here",
                    "url": "https://bogusbasin.org/your-mountain/getting-here/",
                    "finding": "Bogus Basin lists public parking lots, including Pioneer Parking Lot at the dead end of Pioneer Road.",
                },
            ],
        },
        "around_the_mountain_signage": {
            "status": "source_confirms_counter_clockwise_all_users_day_of_recheck_required",
            "route_truth_effect": "operational_gate_not_official_segment_truth",
            "evidence": {
                "source": "Ridge to Rivers Bogus Basin Area",
                "url": "https://www.ridgetorivers.org/trails/trail-areas/bogus-basin-area/",
                "finding": "Around the Mountain Trail #98 is directional; all users are required to travel counter-clockwise.",
            },
        },
        "closure_date_conditions": {
            "status": "operational_gate_not_route_truth",
            "evidence": {
                "source": "Bogus Basin Deer Point Stewardship Project 2026",
                "url": "https://bogusbasin.org/about-bogus/culture/deer-point-stewardship-project-2026/",
                "finding": "Bogus Basin Road has weekday time-window closures and Pat's Trail is closed midweek through June 19, 2026; weekend access is not affected.",
            },
            "promotion_rule": "Do not schedule Bogus candidates on June 18 or June 19 during affected road/trail closure windows unless same-day sources show the route is open.",
        },
    }


def evaluate_bundle(
    *,
    bundle: dict[str, Any],
    field_tool_data: dict[str, Any],
) -> dict[str, Any]:
    routes_by_label = route_by_label(field_tool_data)
    owner_by_segment = segment_owner_index(field_tool_data)
    loops = bundle.get("generated_loops") or []
    if not loops:
        return {
            "bundle_id": bundle.get("bundle_id"),
            "status": "not_generated",
            "hard_failures_after_first_pass": ["route_geometry_missing"],
        }
    loop = loops[0]
    direct_gap_review = direct_gap_repairs(bundle_id=bundle["bundle_id"], loop=loop, routes_by_label=routes_by_label)
    repeat_review = repeat_and_ownership_review(bundle=bundle, loop=loop, owner_by_segment=owner_by_segment)
    cue_sheet = build_cue_sheet(loop, direct_gap_review)
    priced_track_miles = float_value(direct_gap_review.get("post_named_cue_priced_track_miles") or loop.get("track_miles"))
    scaled_time = scaled_minutes_for_priced_track(bundle.get("current_total_scope") or {}, priced_track_miles)
    hard_failures = []
    if direct_gap_review["status"] != "passed_no_direct_gap":
        hard_failures.append("continuous_gpx_not_rebuilt_from_named_connector_splice")
    if repeat_review["hidden_self_repeat_ids_after_classification"]:
        hard_failures.append("hidden_self_repeat_after_classification")
    if repeat_review["unowned_latent_credit_ids"] or repeat_review["would_be_unowned_if_replaced_ids"]:
        hard_failures.append("latent_ownership_not_settled")
    if cue_sheet["status"] != "draft_human_readable":
        hard_failures.append("cue_sheet_not_field_ready_until_gpx_rebuilt")
    hard_failures.append("field_packet_recertification_not_run")
    status = "blocked_keep_current_bogus" if hard_failures else "gate_repaired_candidate_still_needs_recertification"
    comparison = {
        "current_scope": bundle.get("current_total_scope"),
        "candidate_original": bundle.get("candidate_total_scope"),
        "candidate_after_named_gap_substitution": {
            "track_miles": round_miles(priced_track_miles),
            "p75_minutes": scaled_time.get("p75_minutes"),
            "p90_minutes": scaled_time.get("p90_minutes"),
            "pricing_status": scaled_time.get("status"),
            "delta_vs_current_on_foot_miles": round_miles(priced_track_miles - float_value((bundle.get("current_total_scope") or {}).get("on_foot_miles"))),
            "delta_vs_current_p75_minutes": None
            if scaled_time.get("p75_minutes") is None
            else int(scaled_time["p75_minutes"]) - int((bundle.get("current_total_scope") or {}).get("p75_minutes") or 0),
            "delta_vs_current_p90_minutes": None
            if scaled_time.get("p90_minutes") is None
            else int(scaled_time["p90_minutes"]) - int((bundle.get("current_total_scope") or {}).get("p90_minutes") or 0),
        },
    }
    return {
        "bundle_id": bundle.get("bundle_id"),
        "shape": bundle.get("shape"),
        "status": status,
        "recommendation": "stop_first_pass_keep_current_bogus" if hard_failures else "continue_to_generated_route_card_trial",
        "hard_failures_after_first_pass": hard_failures,
        "replaces_route_labels": bundle.get("replace_route_labels") or [],
        "generated_loop_id": loop.get("variant_id"),
        "direct_gap_review": direct_gap_review,
        "repeat_and_ownership_review": repeat_review,
        "cue_sheet": cue_sheet,
        "source_review": source_review_for_bundle(bundle),
        "cost_comparison": comparison,
        "coverage_readout": {
            "coverage_status": (loop.get("coverage_validation") or {}).get("status"),
            "covered_segment_ids": normalized_ids(loop.get("official_segment_ids") or []),
            "missing_segment_ids": normalized_ids((loop.get("coverage_validation") or {}).get("missing_segment_ids") or []),
            "ascent_direction": loop.get("ascent_direction_validation"),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    field_tool_data = read_json(args.field_tool_data_json)
    template_candidates = read_json(args.template_candidates_json)
    rows = [
        evaluate_bundle(bundle=bundle_by_id(template_candidates, bundle_id), field_tool_data=field_tool_data)
        for bundle_id in BOGUS_BUNDLE_IDS
    ]
    promotion_candidates = [row for row in rows if not row.get("hard_failures_after_first_pass")]
    return {
        "schema": "boise_trails_bogus_b1_b2_gate_repair_audit_v1",
        "generated_at": now_iso(),
        "status": "first_pass_repair_stops_keep_current_bogus" if not promotion_candidates else "candidate_gate_repair_found",
        "scope": {
            "purpose": "First-pass gate repair audit for Bogus B1/B2 only; no active route-card promotion.",
            "b3_policy": "Do not test B3 until B1 and B2 are individually clean because transfer time is unresolved.",
        },
        "source_files": {
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "template_candidates_json": display_path(args.template_candidates_json),
        },
        "summary": {
            "candidate_count": len(rows),
            "promotion_candidate_count": len(promotion_candidates),
            "blocked_candidate_count": len(rows) - len(promotion_candidates),
            "b1_status": rows[0]["status"],
            "b2_status": rows[1]["status"],
            "recommendation": "keep_current_bogus_cards_after_first_pass",
        },
        "candidates": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bogus B1/B2 Gate Repair Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: `{report['status']}`",
        "",
        "This is a gate-repair audit, not a route-card promotion. It tests whether the post-H1 Bogus B1/B2 template candidates can clear direct-gap, repeat, ownership, cue, current-signage, closure/date, and cost gates enough to justify active route-card work.",
        "",
        "## Summary",
        "",
        f"- Candidates audited: {report['summary']['candidate_count']}",
        f"- Promotion candidates after first pass: {report['summary']['promotion_candidate_count']}",
        f"- Recommendation: `{report['summary']['recommendation']}`",
        "- B3 remains deferred until B1 and B2 are individually clean.",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Status | Direct Gap | Repeat / Ownership | Cost After Named Gap Substitution | Recommendation |",
        "|---|---|---|---|---:|---|",
    ]
    for row in report.get("candidates") or []:
        direct = row["direct_gap_review"]
        repeat = row["repeat_and_ownership_review"]
        cost = row["cost_comparison"]["candidate_after_named_gap_substitution"]
        lines.append(
            "| "
            f"`{row['bundle_id']}` | "
            f"`{row['status']}` | "
            f"{direct['original_direct_gap_miles']} mi, {direct['named_cue_repair_count']} named cue repairs but GPX not rebuilt | "
            f"`{repeat['status']}` | "
            f"{cost['track_miles']} mi / {cost['p75_minutes']} p75 / {cost['p90_minutes']} p90 | "
            f"`{row['recommendation']}` |"
        )
    lines.extend(["", "## Gate Notes", ""])
    for row in report.get("candidates") or []:
        lines.append(f"### {row['bundle_id']}")
        lines.append("")
        lines.append(f"- Hard failures after first pass: {', '.join(row['hard_failures_after_first_pass']) or 'none'}")
        lines.append(
            f"- Direct gaps: {row['direct_gap_review']['direct_gap_count']} original gaps, "
            f"{row['direct_gap_review']['named_cue_repair_count']} have reusable named route-card cues, but promotion still needs rebuilt continuous GPX."
        )
        lines.append(
            f"- Repeat/ownership: `{row['repeat_and_ownership_review']['status']}`; "
            f"declared owned elsewhere: {', '.join(row['repeat_and_ownership_review']['declared_owned_elsewhere_segment_ids']) or 'none'}."
        )
        lines.append(f"- Cue sheet status: `{row['cue_sheet']['status']}` ({row['cue_sheet']['cue_count']} cues).")
        lines.append("- Around the Mountain/current signage: source confirms counter-clockwise all-users direction; keep day-of signage check as operational gate.")
        lines.append("- Closure/date gate: June 18/19 access remains operationally gated by Deer Point stewardship road/trail closure windows; this is not route truth.")
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "Stop Bogus promotion after this first pass. B1 and B2 still depend on named cue substitutions for direct gaps, but neither has a rebuilt continuous GPX using those connector geometries. Current Bogus route cards should remain active until a later generator can build continuous route-card GPX from the named connectors and pass recertification.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--template-candidates-json", type=Path, default=DEFAULT_TEMPLATE_CANDIDATES_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    write_json(args.output_json, report)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="bogus-b1-b2-gate-repair-audit-2026-05-12",
        inputs=[args.field_tool_data_json, args.template_candidates_json, Path(__file__)],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/bogus_b1_b2_gate_repair_audit.py",
        metadata={"status": report["status"], "summary": report["summary"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
