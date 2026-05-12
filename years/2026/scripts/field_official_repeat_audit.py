#!/usr/bin/env python3
"""Audit official-repeat mileage in the current field-packet route cards."""

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


DEFAULT_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_FIELD_TOOL_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "field-official-repeat-audit-2026-05-11.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "field-official-repeat-audit-2026-05-11.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "field-official-repeat-audit-2026-05-11-manifest.json"

NON_CREDIT_CUE_TYPES = {
    "start_access",
    "official_segment_start",
    "connector_named_trail",
    "connector_road",
    "repeat_official_noncredit",
    "overlap_repeat",
    "exit_access",
    "return_to_car",
}


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


def normalized_segment_ids(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    result = []
    for value in values or []:
        try:
            result.append(str(int(value)))
        except (TypeError, ValueError):
            continue
    return sorted(set(result), key=lambda value: int(value))


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def component_index(map_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            candidate_id = component.get("candidate_id")
            if candidate_id:
                indexed[str(candidate_id)] = component
    return indexed


def route_index(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for route in field_tool_data.get("routes") or []:
        for candidate_id in route.get("candidate_ids") or []:
            indexed[str(candidate_id)] = route
    return indexed


def source_repeat_legs(
    cue: dict[str, Any],
    *,
    route: dict[str, Any] | None,
    component: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []
    source_route_segment_ids = set(
        normalized_segment_ids(
            (route or {}).get("segment_ids")
            or (component or {}).get("segment_ids")
            or []
        )
    )
    leg_sources: list[tuple[str, dict[str, Any]]] = []
    if cue.get("start_access"):
        leg_sources.append(("start_access", cue.get("start_access") or {}))
    for index, link in enumerate(cue.get("between_links") or []):
        leg_sources.append((f"between_links[{index}]", link or {}))
    if cue.get("return_to_car"):
        leg_sources.append(("return_to_car", cue.get("return_to_car") or {}))

    for place, leg in leg_sources:
        repeat_miles = float_value(leg.get("official_repeat_miles"))
        repeat_ids = normalized_segment_ids(leg.get("official_repeat_segment_ids") or [])
        classes = {str(value) for value in leg.get("connector_classes") or []}
        if repeat_miles <= 0 and not repeat_ids and "official_repeat" not in classes:
            continue
        rows.append(
            {
                "place": place,
                "official_repeat_miles": round(repeat_miles, 2),
                "official_repeat_segment_ids": repeat_ids,
                "missing_repeat_segment_ids": bool(repeat_miles > 0 and not repeat_ids),
                "self_repeat_segment_ids": sorted(source_route_segment_ids & set(repeat_ids), key=lambda value: int(value)),
                "connector_names": list(leg.get("connector_names") or []),
                "connector_classes": list(leg.get("connector_classes") or []),
            }
        )
    return rows


def cue_text(cue: dict[str, Any]) -> str:
    parts = []
    for key in ("action", "display_detail", "note", "field_warning", "compact"):
        if cue.get(key):
            parts.append(str(cue[key]))
    return " ".join(parts).lower()


def public_repeat_cue_rows(route: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    if not route:
        return rows
    for cue in route.get("wayfinding_cues") or []:
        if str(cue.get("cue_type") or "") not in NON_CREDIT_CUE_TYPES:
            continue
        repeat_ids = normalized_segment_ids(cue.get("official_repeat_segment_ids") or [])
        repeat_miles = float_value(cue.get("official_repeat_miles"))
        if repeat_miles <= 0 and not repeat_ids:
            continue
        text = cue_text(cue)
        rows.append(
            {
                "seq": cue.get("seq"),
                "cue_type": cue.get("cue_type"),
                "official_repeat_miles": round(repeat_miles, 2),
                "official_repeat_segment_ids": repeat_ids,
                "cue_text_mentions_repeat": "repeat" in text,
                "cue_text_mentions_no_new_credit": (
                    "no new credit" in text
                    or "not official challenge credit" in text
                    or "does not count" in text
                ),
                "cue_text": text,
            }
        )
    return rows


def route_non_credit_burden(route: dict[str, Any], component: dict[str, Any] | None) -> float:
    on_foot = float_value(route.get("on_foot_miles") or (component or {}).get("on_foot_miles"))
    official = float_value(route.get("official_miles") or (component or {}).get("official_miles"))
    return round(max(0.0, on_foot - official), 2)


def route_bucket_c_rows(route: dict[str, Any]) -> list[dict[str, Any]]:
    reconciliation = route.get("segment_ownership_reconciliation") or {}
    rows = []
    for segment in reconciliation.get("segments_owned_elsewhere") or []:
        rows.append(
            {
                "seg_id": str(segment.get("seg_id")),
                "trail_name": segment.get("trail_name"),
                "official_miles": segment.get("official_miles"),
                "owned_by_routes": segment.get("owned_by_routes") or [],
                "reconciliation_status": reconciliation.get("status"),
            }
        )
    for segment in reconciliation.get("unclaimed_completed_segments") or []:
        rows.append(
            {
                "seg_id": str(segment.get("seg_id")),
                "trail_name": segment.get("trail_name"),
                "official_miles": segment.get("official_miles"),
                "owned_by_routes": [],
                "reconciliation_status": "unclaimed_completed_segment",
            }
        )
    return rows


def audit(map_data: dict[str, Any], field_tool_data: dict[str, Any], source_files: dict[str, str] | None = None) -> dict[str, Any]:
    components = component_index(map_data)
    routes = route_index(field_tool_data)
    route_rows = []
    bucket_a = []
    bucket_b = []
    bucket_c = []
    repeat_legs_missing_ids = []
    repeat_cues_missing_text = []
    route_cues = map_data.get("route_cues") or {}

    for candidate_id, cue in sorted(route_cues.items()):
        route = routes.get(str(candidate_id))
        component = components.get(str(candidate_id))
        source_legs = source_repeat_legs(cue, route=route, component=component)
        public_cues = public_repeat_cue_rows(route)
        public_repeat_ids = {seg_id for public_cue in public_cues for seg_id in public_cue["official_repeat_segment_ids"]}
        public_text_ok = all(
            row["cue_text_mentions_repeat"] and row["cue_text_mentions_no_new_credit"]
            for row in public_cues
        )
        missing_public_ids = sorted(
            {
                seg_id
                for leg in source_legs
                for seg_id in leg["official_repeat_segment_ids"]
                if seg_id not in public_repeat_ids
            },
            key=lambda value: int(value),
        )
        route_label = (route or {}).get("label") or (component or {}).get("field_menu_label") or candidate_id
        row = {
            "candidate_id": candidate_id,
            "outing_id": (route or {}).get("outing_id"),
            "label": route_label,
            "official_miles": (route or {}).get("official_miles") or (component or {}).get("official_miles"),
            "on_foot_miles": (route or {}).get("on_foot_miles") or (component or {}).get("on_foot_miles"),
            "non_credit_miles": route_non_credit_burden(route or {}, component),
            "source_repeat_legs": source_legs,
            "public_repeat_cues": [
                {key: value for key, value in public_cue.items() if key != "cue_text"}
                for public_cue in public_cues
            ],
            "missing_public_repeat_segment_ids": missing_public_ids,
            "bucket_c_extra_credit_segments": route_bucket_c_rows(route or {}),
        }
        route_rows.append(row)

        for leg in source_legs:
            if leg["missing_repeat_segment_ids"]:
                repeat_legs_missing_ids.append({"candidate_id": candidate_id, "label": route_label, **leg})
            hidden_self_repeat = bool(leg["self_repeat_segment_ids"]) and (
                leg["missing_repeat_segment_ids"]
                or bool(missing_public_ids)
                or not public_text_ok
            )
            if hidden_self_repeat:
                bucket_a.append({"candidate_id": candidate_id, "label": route_label, **leg})
            elif leg["official_repeat_miles"] > 0 or leg["official_repeat_segment_ids"]:
                bucket_b.append({"candidate_id": candidate_id, "label": route_label, **leg})

        for public_cue in public_cues:
            if not (public_cue["cue_text_mentions_repeat"] and public_cue["cue_text_mentions_no_new_credit"]):
                repeat_cues_missing_text.append(
                    {
                        "candidate_id": candidate_id,
                        "label": route_label,
                        "seq": public_cue.get("seq"),
                        "cue_type": public_cue.get("cue_type"),
                        "official_repeat_segment_ids": public_cue.get("official_repeat_segment_ids"),
                        "official_repeat_miles": public_cue.get("official_repeat_miles"),
                    }
                )

        if row["bucket_c_extra_credit_segments"]:
            bucket_c.append(
                {
                    "candidate_id": candidate_id,
                    "label": route_label,
                    "extra_credit_segments": row["bucket_c_extra_credit_segments"],
                }
            )

    unreconciled_bucket_c = [
        item
        for item in bucket_c
        for segment in item["extra_credit_segments"]
        if segment.get("reconciliation_status") == "unclaimed_completed_segment"
    ]
    status = "passed"
    if bucket_a or repeat_legs_missing_ids or repeat_cues_missing_text or unreconciled_bucket_c:
        status = "failed"

    top_non_credit = sorted(
        [
            {
                "candidate_id": row["candidate_id"],
                "label": row["label"],
                "official_miles": row["official_miles"],
                "on_foot_miles": row["on_foot_miles"],
                "non_credit_miles": row["non_credit_miles"],
            }
            for row in route_rows
        ],
        key=lambda row: float(row["non_credit_miles"] or 0),
        reverse=True,
    )[:10]
    return {
        "schema": "boise_trails_field_official_repeat_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "summary": {
            "route_count": len(route_rows),
            "source_repeat_leg_count": sum(len(row["source_repeat_legs"]) for row in route_rows),
            "public_repeat_cue_count": sum(len(row["public_repeat_cues"]) for row in route_rows),
            "bucket_a_bad_hidden_self_repeat_count": len(bucket_a),
            "bucket_b_legitimate_repeat_or_optimization_target_count": len(bucket_b),
            "bucket_c_reconciled_extra_credit_route_count": len(bucket_c),
            "repeat_legs_missing_segment_ids": len(repeat_legs_missing_ids),
            "repeat_cues_missing_text": len(repeat_cues_missing_text),
            "unreconciled_extra_credit_segment_count": len(unreconciled_bucket_c),
        },
        "bucket_a_bad_hidden_self_repeat": bucket_a,
        "bucket_b_legitimate_repeat_or_optimization_targets": bucket_b,
        "bucket_c_extra_official_credit": bucket_c,
        "repeat_legs_missing_segment_ids": repeat_legs_missing_ids,
        "repeat_cues_missing_text": repeat_cues_missing_text,
        "top_non_credit_burden_routes": top_non_credit,
        "routes": route_rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Field Official Repeat Audit",
        "",
        f"Generated: {report.get('generated_at')}",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary.get('route_count', 0)}",
        f"- Source repeat legs: {summary.get('source_repeat_leg_count', 0)}",
        f"- Public repeat cues: {summary.get('public_repeat_cue_count', 0)}",
        f"- Bucket A bad hidden self-repeat: {summary.get('bucket_a_bad_hidden_self_repeat_count', 0)}",
        f"- Bucket B counted/cued repeats or optimization targets: {summary.get('bucket_b_legitimate_repeat_or_optimization_target_count', 0)}",
        f"- Bucket C reconciled extra-credit routes: {summary.get('bucket_c_reconciled_extra_credit_route_count', 0)}",
        f"- Repeat legs missing segment IDs: {summary.get('repeat_legs_missing_segment_ids', 0)}",
        f"- Repeat cues missing repeat/no-credit text: {summary.get('repeat_cues_missing_text', 0)}",
        f"- Unreconciled extra-credit segments: {summary.get('unreconciled_extra_credit_segment_count', 0)}",
        "",
        "## Top Non-Credit Burdens",
        "",
        "| Label | Candidate | Official mi | On-foot mi | Non-credit mi |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report.get("top_non_credit_burden_routes") or []:
        lines.append(
            "| {label} | `{candidate}` | {official:.2f} | {on_foot:.2f} | {non_credit:.2f} |".format(
                label=row.get("label"),
                candidate=row.get("candidate_id"),
                official=float(row.get("official_miles") or 0),
                on_foot=float(row.get("on_foot_miles") or 0),
                non_credit=float(row.get("non_credit_miles") or 0),
            )
        )
    lines.extend(["", "## Bucket A", ""])
    if report.get("bucket_a_bad_hidden_self_repeat"):
        for row in report["bucket_a_bad_hidden_self_repeat"]:
            lines.append(
                f"- `{row.get('label')}` `{row.get('candidate_id')}` {row.get('place')}: hidden self-repeat ids {row.get('self_repeat_segment_ids')}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Bucket C", ""])
    if report.get("bucket_c_extra_official_credit"):
        for row in report["bucket_c_extra_official_credit"]:
            segment_ids = [segment.get("seg_id") for segment in row.get("extra_credit_segments") or []]
            lines.append(f"- `{row.get('label')}` `{row.get('candidate_id')}` reconciles extra official segment ids: {segment_ids}")
    else:
        lines.append("- None.")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--field-tool-json", type=Path, default=DEFAULT_FIELD_TOOL_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_files = {
        "map_data_json": display_path(args.map_data_json),
        "field_tool_json": display_path(args.field_tool_json),
    }
    report = audit(
        read_json(args.map_data_json),
        read_json(args.field_tool_json),
        source_files=source_files,
    )
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="field-official-repeat-audit-2026-05-11",
        inputs=[args.map_data_json, args.field_tool_json],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/field_official_repeat_audit.py",
        metadata={"schema": report["schema"], "status": report["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
