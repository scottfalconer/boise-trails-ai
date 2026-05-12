#!/usr/bin/env python3
"""Check whether current-calendar skip-ready removals are executable menu changes."""

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


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_LATENT_REPRICING_AUDIT_JSON = YEAR_DIR / "checkpoints" / "latent-credit-delta-repricing-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "current-calendar-skip-ready-promotion-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "current-calendar-skip-ready-promotion-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "current-calendar-skip-ready-promotion-audit-2026-05-12-manifest.json"


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


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("label") or route.get("block_name") or "")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for route in routes:
        for key in [route_key(route), str(route.get("outing_id") or ""), str(route.get("label") or ""), route_label(route)]:
            if key:
                index[key] = route
        for candidate_id in route.get("candidate_ids") or []:
            index[str(candidate_id)] = route
    return index


def cue_ids(route: dict[str, Any], key: str) -> set[str]:
    ids: set[str] = set()
    for cue in route.get("wayfinding_cues") or []:
        ids.update(normalized_ids(cue.get(key) or []))
    return ids


def field_day_layer_route_refs(field_tool_data: dict[str, Any]) -> dict[str, set[str]]:
    refs = {"labels": set(), "candidate_ids": set()}
    for day in (field_tool_data.get("field_day_layer") or {}).get("field_days") or []:
        for loop in day.get("loops") or []:
            ref = loop.get("route_card_ref") or {}
            if ref.get("label"):
                refs["labels"].add(str(ref["label"]))
            refs["candidate_ids"].update(str(value) for value in ref.get("candidate_ids") or [])
            if loop.get("label"):
                refs["labels"].add(str(loop["label"]))
            if loop.get("candidate_id"):
                refs["candidate_ids"].add(str(loop["candidate_id"]))
    return refs


def source_readiness(
    source_route: dict[str, Any] | None,
    segment_id: str,
) -> dict[str, Any]:
    if not source_route:
        return {
            "segment_id": segment_id,
            "source_route": None,
            "claim_ready": False,
            "cue_ready": False,
            "physical_reconciliation_ready": False,
            "blockers": ["source_route_missing_from_current_menu"],
        }
    claimed_ids = set(normalized_ids(source_route.get("segment_ids") or []))
    owned_elsewhere_ids = set(
        normalized_ids(
            ((source_route.get("segment_ownership_reconciliation") or {}).get("declared_owned_elsewhere_segment_ids"))
            or []
        )
    )
    credit_cue_ids = cue_ids(source_route, "official_segment_ids")
    repeat_cue_ids = cue_ids(source_route, "official_repeat_segment_ids")
    blockers = []
    claim_ready = segment_id in claimed_ids
    cue_ready = segment_id in credit_cue_ids
    physical_ready = segment_id in claimed_ids or segment_id in owned_elsewhere_ids
    if not physical_ready:
        blockers.append("source_route_does_not_physically_reconcile_segment")
    if not claim_ready:
        blockers.append("source_route_claim_missing_for_promoted_segment")
    if not cue_ready and segment_id in repeat_cue_ids:
        blockers.append("source_wayfinding_still_marks_segment_as_no_new_credit")
    elif not cue_ready:
        blockers.append("source_wayfinding_missing_promoted_segment_credit_cue")
    return {
        "segment_id": segment_id,
        "source_route": {
            "outing_id": source_route.get("outing_id"),
            "label": route_label(source_route),
            "candidate_ids": source_route.get("candidate_ids") or [],
        },
        "claim_ready": claim_ready,
        "cue_ready": cue_ready,
        "physical_reconciliation_ready": physical_ready,
        "blockers": blockers,
    }


def promotion_candidate(
    removed_route: dict[str, Any],
    routes_by_key: dict[str, dict[str, Any]],
    field_day_route_refs: dict[str, set[str]],
) -> dict[str, Any]:
    route = removed_route.get("route") or {}
    route_id = str(route.get("outing_id") or removed_route.get("route_key") or "")
    prior_sources = removed_route.get("prior_sources") or {}
    segment_reviews = []
    blockers: set[str] = set()
    for segment_id in normalized_ids(removed_route.get("prior_latent_segment_ids") or route.get("segment_ids") or []):
        sources = prior_sources.get(segment_id) or []
        if not sources:
            review = source_readiness(None, segment_id)
            segment_reviews.append(review)
            blockers.update(review["blockers"])
            continue
        source_blockers: set[str] = set()
        has_ready_source = False
        for source in sources:
            source_route = (
                routes_by_key.get(str(source.get("source_route_key") or ""))
                or routes_by_key.get(str(source.get("outing_id") or ""))
                or routes_by_key.get(str(source.get("label") or ""))
            )
            review = source_readiness(source_route, segment_id)
            segment_reviews.append(review)
            if review["claim_ready"] and review["cue_ready"] and review["physical_reconciliation_ready"]:
                has_ready_source = True
            else:
                source_blockers.update(review["blockers"])
        if not has_ready_source:
            blockers.update(source_blockers)
    route_candidate_ids = {str(value) for value in route.get("candidate_ids") or []}
    field_day_update_required = (
        str(route.get("label") or "") in field_day_route_refs["labels"]
        or bool(route_candidate_ids & field_day_route_refs["candidate_ids"])
    )
    if field_day_update_required:
        blockers.add("field_day_layer_still_references_removed_route")
    promotion_status = "ready_for_menu_deletion" if not blockers else "blocked"
    return {
        "removed_route": {
            "route_key": removed_route.get("route_key"),
            "outing_id": route.get("outing_id"),
            "label": route.get("label"),
            "candidate_ids": route.get("candidate_ids") or [],
            "segment_ids": normalized_ids(route.get("segment_ids") or []),
            "on_foot_miles": route.get("on_foot_miles"),
            "door_to_door_minutes_p75": route.get("door_to_door_minutes_p75"),
            "door_to_door_minutes_p90": route.get("door_to_door_minutes_p90"),
        },
        "promotion_status": promotion_status,
        "blockers": sorted(blockers),
        "field_day_layer_update_required": field_day_update_required,
        "prior_latent_segment_ids": normalized_ids(removed_route.get("prior_latent_segment_ids") or []),
        "source_segment_readiness": segment_reviews,
        "saved_on_foot_miles": removed_route.get("saved_on_foot_miles"),
        "saved_p75_minutes": removed_route.get("saved_p75_minutes"),
        "saved_p90_minutes": removed_route.get("saved_p90_minutes"),
    }


def build_promotion_audit(
    field_tool_data: dict[str, Any],
    latent_repricing_audit: dict[str, Any],
    *,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = route_index(routes)
    field_day_route_refs = field_day_layer_route_refs(field_tool_data)
    removed_routes = (latent_repricing_audit.get("current_calendar_repricing") or {}).get("removed_routes") or []
    candidates = [
        promotion_candidate(removed_route, routes_by_key, field_day_route_refs)
        for removed_route in removed_routes
    ]
    blocked = [row for row in candidates if row["promotion_status"] != "ready_for_menu_deletion"]
    ready = [row for row in candidates if row["promotion_status"] == "ready_for_menu_deletion"]
    status = (
        "no_skip_ready_removals"
        if not candidates
        else ("ready_for_menu_deletion" if ready and not blocked else "blocked_needs_route_card_claim_promotion")
    )
    return {
        "schema": "boise_trails_current_calendar_skip_ready_promotion_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "scope": {
            "proves": [
                "whether a skip-ready removal is already executable as an active menu deletion",
                "whether predecessor route cards claim the promoted segments",
                "whether predecessor cue text presents the promoted segments as credit instead of repeat/no-credit movement",
            ],
            "does_not_prove": [
                "a regenerated replacement route card for partial-shrink rows",
                "official BTC progress before challenge-window activity validation",
            ],
        },
        "summary": {
            "skip_ready_candidate_count": len(candidates),
            "ready_for_menu_deletion_count": len(ready),
            "blocked_candidate_count": len(blocked),
            "blocked_segment_count": len(
                {
                    review["segment_id"]
                    for row in blocked
                    for review in row.get("source_segment_readiness") or []
                    if review.get("blockers")
                }
            ),
            "saved_on_foot_miles_if_ready": round(
                sum(float(row.get("saved_on_foot_miles") or 0) for row in ready),
                2,
            ),
            "blocked_on_foot_miles": round(
                sum(float(row.get("saved_on_foot_miles") or 0) for row in blocked),
                2,
            ),
        },
        "promotion_candidates": candidates,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Current-Calendar Skip-Ready Promotion Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Skip-ready candidates: {summary['skip_ready_candidate_count']}",
        f"- Ready for active menu deletion: {summary['ready_for_menu_deletion_count']}",
        f"- Blocked candidates: {summary['blocked_candidate_count']}",
        f"- Blocked on-foot savings: {summary['blocked_on_foot_miles']:.2f} mi",
        "",
        "## Candidates",
        "",
        "| Removed route | Status | Blockers | Savings |",
        "|---|---|---|---:|",
    ]
    for row in audit.get("promotion_candidates") or []:
        route = row.get("removed_route") or {}
        blockers = ", ".join(row.get("blockers") or [])
        lines.append(
            f"| {route.get('label') or route.get('outing_id')} | {row.get('promotion_status')} | {blockers or 'none'} | {float(row.get('saved_on_foot_miles') or 0):.2f} mi |"
        )
    lines.extend(
        [
            "",
            "## Promotion Rule",
            "",
            "- A skip-ready route is deletable only after the predecessor route card claims the segment, cues it as credit, removes the later card from the field-day layer, regenerates the packet, and passes recertification.",
            "- A predecessor route that only lists the segment under `official_repeat_segment_ids` is physical evidence, not an executable ownership promotion.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--latent-repricing-audit-json", type=Path, default=DEFAULT_LATENT_REPRICING_AUDIT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_promotion_audit(
        read_json(args.field_tool_data_json),
        read_json(args.latent_repricing_audit_json),
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "latent_repricing_audit_json": display_path(args.latent_repricing_audit_json),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="current-calendar-skip-ready-promotion-audit-2026-05-12",
        inputs=[args.field_tool_data_json, args.latent_repricing_audit_json],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/current_calendar_skip_ready_promotion_audit.py",
        metadata={"status": audit["status"], **audit["summary"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], "summary": audit["summary"]}, indent=2))
    return 0 if audit["status"] != "blocked_needs_route_card_claim_promotion" else 2


if __name__ == "__main__":
    raise SystemExit(main())
