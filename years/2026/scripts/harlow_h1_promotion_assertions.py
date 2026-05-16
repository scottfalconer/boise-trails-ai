#!/usr/bin/env python3
"""Assert the active field packet really promoted H1 and removed old Harlow cards."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from harlow_h1_gate_repair_audit import H1_BUNDLE_ID, H1_REPLACE_ROUTE_LABELS, H1_SEGMENT_IDS, sort_id  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_FIELD_PACKET_MANIFEST_JSON = REPO_ROOT / "docs" / "field-packet" / "manifest.json"
DEFAULT_H1_AUDIT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-gate-repair-audit-2026-05-12.json"
DEFAULT_ACCESS_REVIEW_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-access-cue-review-2026-05-12.json"
DEFAULT_PROMOTION_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-route-card-promotion-2026-05-12.json"
DEFAULT_FIELD_DAY_LAYER_JSON = YEAR_DIR / "checkpoints" / "human-executable-field-day-layer-2026-05-10.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-promotion-assertions-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "harlow-h1-promotion-assertions-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "harlow-h1-promotion-assertions-2026-05-12-manifest.json"
H1_LABEL = "H1"
H1_ASSIGNED_DATE = "2026-07-04"


def expected_route_count_from_promotion(promotion: dict[str, Any]) -> int | None:
    summary = promotion.get("summary") or {}
    expected = summary.get("expected_active_route_cards_after_export")
    if expected is not None:
        return int(expected)
    old_count = summary.get("old_route_card_count")
    if old_count is not None:
        return int(old_count) - len(H1_REPLACE_ROUTE_LABELS) + 1
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=sort_id)


def route_by_label(field_tool_data: dict[str, Any], label: str) -> dict[str, Any] | None:
    for route in field_tool_data.get("routes") or []:
        if str(route.get("label") or "") == label:
            return route
    return None


def route_labels(field_tool_data: dict[str, Any]) -> set[str]:
    return {str(route.get("label") or "") for route in field_tool_data.get("routes") or []}


def h1_field_day(field_day_layer: dict[str, Any]) -> dict[str, Any] | None:
    for day in field_day_layer.get("field_days") or []:
        if str(day.get("date") or "") == H1_ASSIGNED_DATE:
            return day
    return None


def empty_day(field_day_layer: dict[str, Any], date: str) -> dict[str, Any] | None:
    for day in field_day_layer.get("field_days") or []:
        if str(day.get("date") or "") == date:
            return day
    return None


def text_contains_opaque_osm(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True)
    return "OSM " in text or "connector 22098" in text or "connector 13946" in text


def assertion(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"assertion": name, "passed": bool(passed), "evidence": evidence}


def build_assertions(
    *,
    field_tool_data: dict[str, Any],
    packet_manifest: dict[str, Any],
    h1_audit: dict[str, Any],
    access_review: dict[str, Any],
    promotion: dict[str, Any],
    field_day_layer: dict[str, Any],
) -> dict[str, Any]:
    h1_route = route_by_label(field_tool_data, H1_LABEL)
    labels = route_labels(field_tool_data)
    h1_ids = normalized_ids((h1_route or {}).get("segment_ids"))
    removed_union = normalized_ids((access_review.get("h1_replacement_segment_set_diff") or {}).get("replaced_old_claimed_ids"))
    repeat = h1_audit["route_repeat_optimization_audit_for_h1"]["candidate_specific_repeat_accounting"]
    repaired = h1_audit["repaired_candidate"]
    h1_day = h1_field_day(field_day_layer)
    empty_0621 = empty_day(field_day_layer, "2026-06-21")
    empty_0712 = empty_day(field_day_layer, "2026-07-12")
    final_buffer = empty_day(field_day_layer, "2026-07-18")
    field_summary = field_tool_data.get("summary") or {}
    manifest_summary = packet_manifest.get("summary") or {}
    parking = (h1_route or {}).get("parking") or {}
    h1_cue_text = {
        "turn_by_turn_steps": (h1_route or {}).get("turn_by_turn_steps") or [],
        "wayfinding_cues": (h1_route or {}).get("wayfinding_cues") or [],
        "route_cues": (h1_route or {}).get("route_cues") or [],
    }
    checks = [
        assertion("h1_route_card_exists", h1_route is not None, {"label": H1_LABEL}),
        assertion("h1_route_card_is_certified", bool((h1_route or {}).get("validation", {}).get("passed")), (h1_route or {}).get("validation")),
        assertion("old_harlow_avimor_cards_absent", not labels.intersection(H1_REPLACE_ROUTE_LABELS), {"remaining_old_labels": sorted(labels.intersection(H1_REPLACE_ROUTE_LABELS))}),
        assertion(
            "active_route_card_count_matches_h1_promotion",
            expected_route_count_from_promotion(promotion) is not None
            and len(field_tool_data.get("routes") or []) == expected_route_count_from_promotion(promotion),
            {
                "route_count": len(field_tool_data.get("routes") or []),
                "expected_route_count": expected_route_count_from_promotion(promotion),
            },
        ),
        assertion("field_packet_represents_251_official_segments", int(field_summary.get("segment_count_in_field_menu") or 0) == 251, field_summary),
        assertion("h1_claimed_segment_set_equals_removed_union", h1_ids == removed_union == normalized_ids(H1_SEGMENT_IDS), {"h1_ids": h1_ids, "removed_union": removed_union}),
        assertion(
            "h1_p90_recorded_for_assigned_date",
            bool(h1_day and int(h1_day.get("p90_minutes") or 0) > 0),
            {
                "field_day": h1_day,
                "schedule_note": "Schedule fit must use explicit dated availability, not weekday/weekend label.",
            },
        ),
        assertion("h1_assigned_to_expected_date", bool(h1_day), h1_day),
        assertion("h1_has_no_direct_gap_fallback", float(repaired.get("direct_gap_fallback_miles") or 0) == 0, repaired.get("direct_gap_fallback_miles")),
        assertion("h1_has_no_hidden_self_repeat", not repeat.get("hidden_self_repeat_ids"), repeat),
        assertion("h1_repeat_mileage_priced_and_cued", float(repaired.get("official_repeat_miles") or 0) > 0 and not repeat.get("unpriced_repeat_ids"), repeat),
        assertion("h1_parking_metadata_present", bool(parking.get("parking_confidence") and parking.get("source") and parking.get("has_parking") is True), parking),
        assertion("h1_runner_cues_avoid_opaque_osm_ids", not text_contains_opaque_osm(h1_cue_text), h1_cue_text),
        assertion("june_21_harlow_day_freed", bool(empty_0621 and int(empty_0621.get("loop_count") or 0) == 0 and empty_0621.get("execution_status") == "reusable_empty_field_day"), empty_0621),
        assertion(
            "july_12_harlow_day_contains_no_removed_harlow_cards",
            bool(
                empty_0712
                and not {
                    str((loop.get("route_card_ref") or {}).get("label") or loop.get("label") or "")
                    for loop in empty_0712.get("loops") or []
                }.intersection(H1_REPLACE_ROUTE_LABELS)
            ),
            empty_0712,
        ),
        assertion(
            "final_day_is_reusable_buffer_after_post_h1_cleanup",
            bool(final_buffer and int(final_buffer.get("loop_count") or 0) == 0 and final_buffer.get("execution_status") == "reusable_empty_field_day"),
            final_buffer,
        ),
        assertion("gpx_validation_still_passes", manifest_summary.get("gpx_validation_passed") is True and int(manifest_summary.get("failed_gpx_count") or 0) == 0, manifest_summary),
        assertion("source_promotion_assertions_passed", all(row.get("passed") for row in promotion.get("promotion_assertions") or []), promotion.get("promotion_assertions") or []),
        assertion("access_cue_gate_was_clear_before_promotion", (access_review.get("promotion_readiness") or {}).get("status") == "access_gate_clear_keep_unpromoted", access_review.get("promotion_readiness")),
    ]
    status = "passed" if all(row["passed"] for row in checks) else "failed"
    return {
        "schema": "boise_trails_harlow_h1_promotion_assertions_v1",
        "generated_at": now_iso(),
        "status": status,
        "summary": {
            "passed_assertion_count": sum(1 for row in checks if row["passed"]),
            "assertion_count": len(checks),
            "route_count": len(field_tool_data.get("routes") or []),
            "h1_segment_count": len(h1_ids),
        },
        "assertions": checks,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Harlow / Avimor H1 Promotion Assertions",
        "",
        f"- Status: `{report['status']}`",
        f"- Assertions: {report['summary']['passed_assertion_count']} / {report['summary']['assertion_count']} passed",
        f"- Active route cards: {report['summary']['route_count']}",
        "",
        "| Assertion | Status |",
        "|---|---|",
    ]
    for row in report["assertions"]:
        lines.append(f"| `{row['assertion']}` | {'pass' if row['passed'] else 'FAIL'} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--packet-manifest-json", type=Path, default=DEFAULT_FIELD_PACKET_MANIFEST_JSON)
    parser.add_argument("--h1-audit-json", type=Path, default=DEFAULT_H1_AUDIT_JSON)
    parser.add_argument("--access-review-json", type=Path, default=DEFAULT_ACCESS_REVIEW_JSON)
    parser.add_argument("--promotion-json", type=Path, default=DEFAULT_PROMOTION_JSON)
    parser.add_argument("--field-day-layer-json", type=Path, default=DEFAULT_FIELD_DAY_LAYER_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_assertions(
        field_tool_data=read_json(args.field_tool_data_json),
        packet_manifest=read_json(args.packet_manifest_json),
        h1_audit=read_json(args.h1_audit_json),
        access_review=read_json(args.access_review_json),
        promotion=read_json(args.promotion_json),
        field_day_layer=read_json(args.field_day_layer_json),
    )
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="harlow_h1_promotion_assertions",
        inputs=[
            args.field_tool_data_json,
            args.packet_manifest_json,
            args.h1_audit_json,
            args.access_review_json,
            args.promotion_json,
            args.field_day_layer_json,
        ],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/harlow_h1_promotion_assertions.py",
        metadata={"status": report["status"], **report["summary"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
