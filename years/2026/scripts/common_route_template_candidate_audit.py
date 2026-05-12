#!/usr/bin/env python3
"""Generate route-candidate seeds from common human route templates."""

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
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments  # noqa: E402


DEFAULT_TEMPLATE_JSON = (
    YEAR_DIR / "inputs" / "open-data" / "common-route-templates-2026-05-12.json"
)
DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_REPEAT_PRODUCTIVITY_JSON = YEAR_DIR / "checkpoints" / "repeat-productivity-audit-2026-05-12.json"
DEFAULT_SIMULATED_PROGRESS_JSON = YEAR_DIR / "checkpoints" / "simulated-progress-sweep-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "common-route-template-candidates-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "common-route-template-candidates-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "common-route-template-candidates-2026-05-12-manifest.json"


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


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("route_key") or route.get("label") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment["seg_id"]): segment for segment in official_segments}


def segment_miles(segment_ids: set[str], official_by_id: dict[str, dict[str, Any]]) -> float:
    return rounded(sum(float_value((official_by_id.get(segment_id) or {}).get("official_miles")) for segment_id in segment_ids))


def segment_briefs(segment_ids: list[str], official_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for segment_id in segment_ids:
        segment = official_by_id.get(str(segment_id), {})
        rows.append(
            {
                "seg_id": str(segment_id),
                "seg_name": segment.get("seg_name"),
                "trail_name": segment.get("trail_name"),
                "direction": segment.get("direction"),
                "official_miles": rounded(segment.get("official_miles")),
            }
        )
    return rows


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


def productivity_index(repeat_productivity_audit: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in (repeat_productivity_audit or {}).get("routes") or []:
        key = str(row.get("route_key") or row.get("outing_id") or "")
        if key:
            rows[key] = row
    return rows


def simulated_sweep_index(simulated_progress_audit: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in (simulated_progress_audit or {}).get("route_sweeps_ranked") or []:
        for key in [str(row.get("route_id") or ""), *(str(item) for item in row.get("subject_route_keys") or [])]:
            if key:
                rows[key] = row
    return rows


def has_public_source(template: dict[str, Any]) -> bool:
    for source in template.get("public_sources") or []:
        source_type = str(source.get("source_type") or "")
        if source_type.startswith("public_") and source.get("url"):
            return True
    return False


def source_labels(template: dict[str, Any]) -> list[str]:
    labels = []
    for source in template.get("public_sources") or []:
        label = str(source.get("label") or source.get("url") or source.get("path") or source.get("source_type") or "")
        if label:
            labels.append(label)
    return labels


def template_warnings(template: dict[str, Any], segment_rows: list[dict[str, Any]]) -> list[str]:
    warnings = {"parking_access_proof_required", "continuous_gpx_and_p75_required"}
    if not has_public_source(template):
        warnings.add("needs_public_route_source_capture")
    if any(row.get("direction") == "ascent" for row in segment_rows):
        warnings.add("ascent_direction_must_be_validated")
    target_area = str(template.get("target_area") or "").lower()
    trail_names = " ".join(str(row.get("trail_name") or "") for row in segment_rows).lower()
    if "bogus" in target_area or "around the mountain" in trail_names or "deer point" in trail_names:
        warnings.add("bogus_closure_and_direction_check_required")
    if "lower hull" in trail_names:
        warnings.add("lower_hulls_day_legality_check_required")
    return sorted(warnings)


def route_match_rows(
    template_ids: set[str],
    routes: list[dict[str, Any]],
    official_by_id: dict[str, dict[str, Any]],
    productivity_by_route: dict[str, dict[str, Any]],
    sweeps_by_route: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for route in routes:
        claimed_ids = set(normalized_ids(route.get("segment_ids") or []))
        overlap_ids = claimed_ids & template_ids
        if not overlap_ids:
            continue
        key = route_key(route)
        productivity = productivity_by_route.get(key) or {}
        sweep = sweeps_by_route.get(key) or {}
        priority = sweep.get("priority_score") or {}
        future_collapse = sweep.get("future_collapse_savings") or {}
        future_shrink = sweep.get("future_shrink_unpriced") or {}
        rows.append(
            {
                **route_metrics(route),
                "match_type": "contained_route_card" if claimed_ids <= template_ids else "partial_route_card_overlap",
                "overlap_segment_ids": normalized_ids(overlap_ids),
                "overlap_official_miles": segment_miles(overlap_ids, official_by_id),
                "overlap_fraction_of_route_claim": rounded(len(overlap_ids) / len(claimed_ids), 3) if claimed_ids else 0.0,
                "pressure": {
                    "dead_repeat_candidate_miles": rounded(productivity.get("dead_repeat_candidate_miles")),
                    "productive_repeat_miles": rounded(productivity.get("productive_repeat_miles")),
                    "necessary_repeat_miles": rounded(productivity.get("necessary_repeat_miles")),
                    "future_collapse_on_foot_miles": rounded(
                        priority.get("future_collapse_on_foot_miles")
                        if priority
                        else future_collapse.get("on_foot_miles")
                    ),
                    "future_collapse_p75_minutes": int_value(
                        priority.get("future_collapse_p75_minutes")
                        if priority
                        else future_collapse.get("door_to_door_minutes_p75")
                    ),
                    "future_shrink_official_miles_unpriced": rounded(
                        priority.get("future_shrink_official_miles_unpriced")
                        if priority
                        else future_shrink.get("completed_claimed_official_miles")
                    ),
                },
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["match_type"] != "contained_route_card",
            -float_value(row["pressure"]["dead_repeat_candidate_miles"]),
            -float_value(row["pressure"]["future_collapse_on_foot_miles"]),
            row["label"],
        ),
    )


def pressure_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "matched_route_count": len(rows),
        "matched_current_on_foot_miles": rounded(sum(float_value(row.get("on_foot_miles")) for row in rows)),
        "matched_current_p75_minutes": sum(int_value(row.get("door_to_door_minutes_p75")) for row in rows),
        "dead_repeat_candidate_miles": rounded(
            sum(float_value((row.get("pressure") or {}).get("dead_repeat_candidate_miles")) for row in rows)
        ),
        "future_collapse_on_foot_miles": rounded(
            sum(float_value((row.get("pressure") or {}).get("future_collapse_on_foot_miles")) for row in rows)
        ),
        "future_collapse_p75_minutes": sum(
            int_value((row.get("pressure") or {}).get("future_collapse_p75_minutes")) for row in rows
        ),
        "future_shrink_official_miles_unpriced": rounded(
            sum(float_value((row.get("pressure") or {}).get("future_shrink_official_miles_unpriced")) for row in rows)
        ),
    }


def template_row(
    template: dict[str, Any],
    *,
    routes: list[dict[str, Any]],
    official_by_id: dict[str, dict[str, Any]],
    productivity_by_route: dict[str, dict[str, Any]],
    sweeps_by_route: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_ids = normalized_ids(template.get("candidate_segments") or [])
    candidate_id_set = set(candidate_ids)
    invalid_ids = [segment_id for segment_id in candidate_ids if segment_id not in official_by_id]
    segment_rows = segment_briefs(candidate_ids, official_by_id)
    matched_routes = route_match_rows(
        candidate_id_set,
        routes,
        official_by_id,
        productivity_by_route,
        sweeps_by_route,
    )
    totals = pressure_totals(matched_routes)
    status = "ready_for_route_experiment"
    if invalid_ids:
        status = "needs_segment_mapping"
    elif not matched_routes:
        status = "valid_segments_unlinked_to_current_menu"
    if not has_public_source(template):
        status = "cluster_seed_needs_public_source" if status == "ready_for_route_experiment" else status
    optimizer_attention_score = rounded(
        totals["dead_repeat_candidate_miles"]
        + totals["future_collapse_on_foot_miles"]
        + (totals["future_shrink_official_miles_unpriced"] * 0.5)
    )
    return {
        "template_id": template.get("template_id"),
        "target_area": template.get("target_area"),
        "status": status,
        "normal_start": template.get("normal_start"),
        "normal_direction": template.get("normal_direction"),
        "trail_sequence": template.get("trail_sequence") or [],
        "source_labels": source_labels(template),
        "has_public_route_source": has_public_source(template),
        "candidate_segment_ids": candidate_ids,
        "invalid_segment_ids": invalid_ids,
        "candidate_official_miles": segment_miles(candidate_id_set - set(invalid_ids), official_by_id),
        "candidate_segments": segment_rows,
        "connector_hints": template.get("connector_hints") or [],
        "route_experiment_goal": template.get("route_experiment_goal"),
        "matched_current_routes": matched_routes,
        "current_route_pressure": totals,
        "optimizer_attention_score": optimizer_attention_score,
        "warnings": template_warnings(template, segment_rows),
        "generator_candidate": {
            "candidate_id": f"template:{template.get('template_id')}",
            "source_template_id": template.get("template_id"),
            "normal_start": template.get("normal_start"),
            "normal_direction": template.get("normal_direction"),
            "trail_sequence": template.get("trail_sequence") or [],
            "official_segment_ids": candidate_ids,
            "connector_hints": template.get("connector_hints") or [],
            "public_sources": template.get("public_sources") or [],
            "promotion_gates": [
                "derive continuous car-to-car route geometry and Nav GPX",
                "validate full official segment coverage and ascent-only direction",
                "verify parking/access/current trail legality",
                "price route-card p75/p90 and DEM effort",
                "generate field cue sheet and pass route walkthrough audit",
                "recertify future-day preservation before replacing active cards"
            ],
        },
    }


def build_common_route_template_candidate_audit(
    templates_data: dict[str, Any],
    field_tool_data: dict[str, Any],
    official_segments: list[dict[str, Any]],
    *,
    repeat_productivity_audit: dict[str, Any] | None = None,
    simulated_progress_audit: dict[str, Any] | None = None,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    official_by_id = segment_index(official_segments)
    routes = field_tool_data.get("routes") or []
    productivity_by_route = productivity_index(repeat_productivity_audit)
    sweeps_by_route = simulated_sweep_index(simulated_progress_audit)
    rows = [
        template_row(
            template,
            routes=routes,
            official_by_id=official_by_id,
            productivity_by_route=productivity_by_route,
            sweeps_by_route=sweeps_by_route,
        )
        for template in templates_data.get("templates") or []
    ]
    rows = sorted(rows, key=lambda row: (-float_value(row["optimizer_attention_score"]), str(row["template_id"])))
    invalid_rows = [row for row in rows if row["invalid_segment_ids"]]
    public_source_rows = [row for row in rows if row["has_public_route_source"]]
    route_experiment_rows = [row for row in rows if row["status"] == "ready_for_route_experiment"]
    status = "failed_invalid_template_segments" if invalid_rows else "template_candidates_generated"
    return {
        "schema": "boise_trails_common_route_template_candidate_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "parameters": {
            "template_source": "common-route template JSON",
            "official_segment_source": "current-year official BTC foot segments",
            "route_pressure_sources": [
                "repeat-productivity audit dead/productive/necessary repeat miles",
                "simulated-progress sweep future collapse and partial shrink pressure"
            ],
            "promotion_policy": "generated template candidates are route-experiment seeds only until parking, GPX, p75/p90, coverage, cues, and recertification pass"
        },
        "scope": {
            "proves": [
                "which common route templates are mapped to valid official segment IDs",
                "which current route cards overlap each template",
                "which templates deserve generator experiments based on current route pressure"
            ],
            "does_not_prove": [
                "the template is currently legal or runnable",
                "a shorter replacement exists",
                "challenge credit before BTC activity validation",
                "field-packet promotion readiness"
            ]
        },
        "summary": {
            "template_count": len(rows),
            "templates_with_public_route_source_count": len(public_source_rows),
            "ready_route_experiment_template_count": len(route_experiment_rows),
            "cluster_seed_needs_public_source_count": len(
                [row for row in rows if row["status"] == "cluster_seed_needs_public_source"]
            ),
            "invalid_template_count": len(invalid_rows),
            "generated_candidate_count": len(rows),
            "top_template_by_attention_score": rows[0]["template_id"] if rows else None,
            "top_template_attention_score": rows[0]["optimizer_attention_score"] if rows else 0.0,
        },
        "templates_ranked": rows,
        "generator_candidates": [row["generator_candidate"] for row in rows],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Common Route Template Candidate Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Templates audited: {summary['template_count']}",
        f"- Templates with public route source: {summary['templates_with_public_route_source_count']}",
        f"- Ready route-experiment templates: {summary['ready_route_experiment_template_count']}",
        f"- Cluster seeds still needing public source capture: {summary['cluster_seed_needs_public_source_count']}",
        f"- Invalid templates: {summary['invalid_template_count']}",
        f"- Top attention template: {summary['top_template_by_attention_score']} ({summary['top_template_attention_score']:.2f})",
        "",
        "## Ranked Templates",
        "",
        "| Rank | Template | Status | Official mi | Current route cards | Dead repeat mi | Future collapse mi | Unpriced shrink mi | Warnings |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(audit.get("templates_ranked") or [], start=1):
        pressure = row["current_route_pressure"]
        lines.append(
            f"| {index} | {row['template_id']} | `{row['status']}` | {float_value(row['candidate_official_miles']):.2f} | "
            f"{pressure['matched_route_count']} | {float_value(pressure['dead_repeat_candidate_miles']):.2f} | "
            f"{float_value(pressure['future_collapse_on_foot_miles']):.2f} | "
            f"{float_value(pressure['future_shrink_official_miles_unpriced']):.2f} | "
            f"{', '.join(row['warnings'])} |"
        )
    lines.extend(["", "## Template Details", ""])
    for row in audit.get("templates_ranked") or []:
        pressure = row["current_route_pressure"]
        matched = ", ".join(route["label"] for route in row["matched_current_routes"]) or "none"
        sources = "; ".join(row["source_labels"]) or "none"
        lines.extend(
            [
                f"### {row['template_id']}",
                "",
                f"- Target area: {row['target_area']}",
                f"- Normal start: {row['normal_start']}",
                f"- Normal direction: {row['normal_direction']}",
                f"- Candidate official miles: {float_value(row['candidate_official_miles']):.2f}",
                f"- Matched current routes: {matched}",
                f"- Source labels: {sources}",
                f"- Current route pressure: {pressure['dead_repeat_candidate_miles']:.2f} dead-repeat mi, {pressure['future_collapse_on_foot_miles']:.2f} priced future-collapse mi, {pressure['future_shrink_official_miles_unpriced']:.2f} unpriced shrink official mi.",
                f"- Experiment goal: {row['route_experiment_goal']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope Boundary",
            "",
            "- These rows are candidate-generator inputs, not official route-card replacements.",
            "- Public Strava route pages and current route-card clusters are behavior evidence only.",
            "- Promotion still requires car-to-car GPX, full official segment coverage, ascent-direction validation, parking/access proof, p75/p90 effort, field cues, and recertification.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates-json", type=Path, default=DEFAULT_TEMPLATE_JSON)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--repeat-productivity-json", type=Path, default=DEFAULT_REPEAT_PRODUCTIVITY_JSON)
    parser.add_argument("--simulated-progress-json", type=Path, default=DEFAULT_SIMULATED_PROGRESS_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    audit = build_common_route_template_candidate_audit(
        read_json(args.templates_json),
        read_json(args.field_tool_data_json),
        official_segments,
        repeat_productivity_audit=optional_json(args.repeat_productivity_json),
        simulated_progress_audit=optional_json(args.simulated_progress_json),
        source_files={
            "templates_json": display_path(args.templates_json),
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "official_geojson": display_path(args.official_geojson),
            "repeat_productivity_json": display_path(args.repeat_productivity_json),
            "simulated_progress_json": display_path(args.simulated_progress_json),
            "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="common-route-template-candidates-2026-05-12",
        inputs=[
            args.templates_json,
            args.field_tool_data_json,
            args.official_geojson,
            args.repeat_productivity_json,
            args.simulated_progress_json,
        ],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/common_route_template_candidate_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] == "failed_invalid_template_segments" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
