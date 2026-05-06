#!/usr/bin/env python3
"""Audit GPX export readiness for the relaxed-drive draft field-day plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_DRAFT_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-draft-field-day-plan-2026-05-06.json"
DEFAULT_PERSONAL_ROUTE_MENU_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_HYBRID_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
DEFAULT_FIELD_PACKET_MANIFEST_JSON = REPO_ROOT / "docs" / "field-packet" / "manifest.json"
DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "p90-forced-anchor-gpx-export-2026-05-06.json"
DEFAULT_DAY_LEVEL_GPX_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-day-gpx-export-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-relaxed-drive-gpx-readiness-audit-2026-05-06"


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


def has_stored_track_geometry(candidate: dict[str, Any] | None) -> bool:
    if not candidate:
        return False
    if not candidate.get("segments"):
        return False
    access = candidate.get("trailhead_access") or {}
    return_to_car = candidate.get("return_to_car") or {}
    links = (candidate.get("between_trail_links") or {}).get("links") or []
    return bool(
        access.get("outbound_path_coordinates")
        or access.get("return_path_coordinates")
        or return_to_car.get("path_coordinates")
        or links
        or candidate.get("segments")
    )


def personal_index(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in ((data.get("route_menu") or {}).get("all_candidates") or [])
    }


def hybrid_index(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    return {
        str(candidate_id): candidate
        for candidate_id, candidate in (data.get("candidate_index") or {}).items()
    }


def field_packet_gpx_index(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    index: dict[str, dict[str, Any]] = {}
    for route in data.get("routes") or []:
        for candidate_id in (route.get("outing") or {}).get("candidate_ids") or []:
            gpx_path = REPO_ROOT / "docs" / "field-packet" / str(route.get("gpx_href") or "")
            index[str(candidate_id)] = {
                "gpx_href": route.get("gpx_href"),
                "gpx_path": display_path(gpx_path),
                "gpx_exists": gpx_path.exists(),
                "validation_passed": (route.get("validation") or {}).get("passed") is True,
                "validation": route.get("validation"),
            }
    return index


def forced_anchor_gpx_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = read_json(path)
    index = {}
    for row in data.get("rows") or []:
        index[str(row["loop_id"])] = {
            "path": row.get("path"),
            "validation_passed": (row.get("validation") or {}).get("passed") is True,
            "validation": row.get("validation"),
        }
    return index


def loop_rows(
    draft: dict[str, Any],
    *,
    personal_candidates: dict[str, dict[str, Any]],
    hybrid_candidates: dict[str, dict[str, Any]],
    field_packet_gpx: dict[str, dict[str, Any]],
    forced_anchor_gpx: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for day in draft.get("field_days") or []:
        for loop in day.get("loops") or []:
            source = loop.get("source")
            candidate_id = str(loop.get("candidate_id") or "")
            candidate = None
            readiness = "needs_source_specific_export"
            if source == "personal_route_menu":
                candidate = personal_candidates.get(candidate_id)
                readiness = "stored_geometry_exportable" if has_stored_track_geometry(candidate) else "missing_stored_geometry"
            elif source == "hybrid_candidate_index":
                candidate = hybrid_candidates.get(candidate_id)
                readiness = "stored_geometry_exportable" if has_stored_track_geometry(candidate) else "missing_stored_geometry"
            elif source == "canonical_field_menu":
                gpx = field_packet_gpx.get(candidate_id)
                if gpx and gpx["gpx_exists"] and gpx["validation_passed"]:
                    readiness = "existing_navigation_gpx_available"
                else:
                    readiness = "needs_field_packet_gpx_lookup"
            elif source == "forced_anchor_probe":
                forced_gpx = forced_anchor_gpx.get(str(loop.get("loop_id") or ""))
                if forced_gpx and forced_gpx["validation_passed"]:
                    readiness = "generated_forced_anchor_gpx_available"
                else:
                    readiness = "needs_probe_regeneration_for_coordinates"
            gpx = field_packet_gpx.get(candidate_id)
            forced_gpx = forced_anchor_gpx.get(str(loop.get("loop_id") or ""))
            rows.append(
                {
                    "draft_day_number": day["draft_day_number"],
                    "loop_id": loop["loop_id"],
                    "source": source,
                    "candidate_id": candidate_id,
                    "label": loop.get("label"),
                    "trailhead": loop.get("trailhead"),
                    "readiness": readiness,
                    "candidate_found": candidate is not None,
                    "has_stored_track_geometry": has_stored_track_geometry(candidate),
                    "existing_gpx_path": (gpx or {}).get("gpx_path"),
                    "existing_gpx_validation_passed": (gpx or {}).get("validation_passed"),
                    "forced_anchor_gpx_path": (forced_gpx or {}).get("path"),
                    "forced_anchor_gpx_validation_passed": (forced_gpx or {}).get("validation_passed"),
                }
            )
    return rows


def build_report(
    draft: dict[str, Any],
    *,
    personal_candidates: dict[str, dict[str, Any]],
    hybrid_candidates: dict[str, dict[str, Any]],
    field_packet_gpx: dict[str, dict[str, Any]],
    forced_anchor_gpx: dict[str, dict[str, Any]],
    day_level_gpx_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = loop_rows(
        draft,
        personal_candidates=personal_candidates,
        hybrid_candidates=hybrid_candidates,
        field_packet_gpx=field_packet_gpx,
        forced_anchor_gpx=forced_anchor_gpx,
    )
    readiness_counts = Counter(row["readiness"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    exportable = (
        readiness_counts.get("stored_geometry_exportable", 0)
        + readiness_counts.get("existing_navigation_gpx_available", 0)
        + readiness_counts.get("generated_forced_anchor_gpx_available", 0)
    )
    day_summary = (day_level_gpx_manifest or {}).get("summary") or {}
    day_level_gpx_ready = (
        day_summary.get("day_gpx_count", 0) > 0
        and day_summary.get("loop_validation_passed") is True
        and day_summary.get("day_track_validation_passed") is True
        and day_summary.get("failed_day_count") == 0
    )
    return {
        "objective": "audit GPX export readiness for selected relaxed-drive field-day loops",
        "source_files": {
            "draft_field_day_plan_json": display_path(DEFAULT_DRAFT_JSON),
            "personal_route_menu_json": display_path(DEFAULT_PERSONAL_ROUTE_MENU_JSON),
            "hybrid_route_pass_json": display_path(DEFAULT_HYBRID_ROUTE_PASS_JSON),
            "field_packet_manifest_json": display_path(DEFAULT_FIELD_PACKET_MANIFEST_JSON),
            "forced_anchor_gpx_manifest_json": display_path(DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON),
            "day_level_gpx_manifest_json": display_path(DEFAULT_DAY_LEVEL_GPX_MANIFEST_JSON),
        },
        "summary": {
            "selected_loop_count": len(rows),
            "gpx_or_stored_geometry_available_loop_count": exportable,
            "non_exportable_or_needs_lookup_loop_count": len(rows) - exportable,
            "readiness_counts": dict(sorted(readiness_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "selected_loop_gpx_ready": exportable == len(rows),
            "day_level_gpx_ready": day_level_gpx_ready,
            "day_level_gpx_count": day_summary.get("day_gpx_count", 0),
            "day_level_gpx_loop_validation_passed": day_summary.get("loop_validation_passed") is True,
            "day_level_gpx_track_validation_passed": day_summary.get("day_track_validation_passed") is True,
            "day_level_gpx_failed_day_count": day_summary.get("failed_day_count"),
        },
        "rows": rows,
        "known_gaps": [
            "This audit checks source geometry availability only; it does not export GPX.",
            "Canonical field-menu rows are resolved through the phone field-packet manifest when possible.",
            "Forced-anchor probe rows are resolved through the regenerated forced-anchor GPX manifest when available.",
            "Day-level GPX readiness is read from the day-GPX export manifest when that artifact exists.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# P90 Relaxed-Drive GPX Readiness Audit",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Selected loops: {summary['selected_loop_count']}",
        f"- GPX or stored-geometry available loops: {summary['gpx_or_stored_geometry_available_loop_count']}",
        f"- Needs lookup/regeneration loops: {summary['non_exportable_or_needs_lookup_loop_count']}",
        f"- Selected loop GPX ready: {summary['selected_loop_gpx_ready']}",
        f"- Day-level GPX ready: {summary['day_level_gpx_ready']}",
        f"- Day-level GPX files: {summary['day_level_gpx_count']}",
        f"- Day-level GPX failed days: {summary['day_level_gpx_failed_day_count']}",
        f"- Readiness counts: `{summary['readiness_counts']}`",
        f"- Source counts: `{summary['source_counts']}`",
        "",
        "## Known Gaps",
        "",
    ]
    lines.extend(f"- {gap}" for gap in report["known_gaps"])
    lines.extend(
        [
            "",
            "## Non-Exportable / Needs Lookup Rows",
            "",
            "| Day | Source | Readiness | Label | Trailhead |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in report["rows"]:
        if row["readiness"] in {
            "stored_geometry_exportable",
            "existing_navigation_gpx_available",
            "generated_forced_anchor_gpx_available",
        }:
            continue
        lines.append(
            f"| {row['draft_day_number']} | {row['source']} | {row['readiness']} | "
            f"{row['label']} | {row['trailhead']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-json", type=Path, default=DEFAULT_DRAFT_JSON)
    parser.add_argument("--personal-route-menu-json", type=Path, default=DEFAULT_PERSONAL_ROUTE_MENU_JSON)
    parser.add_argument("--hybrid-route-pass-json", type=Path, default=DEFAULT_HYBRID_ROUTE_PASS_JSON)
    parser.add_argument("--field-packet-manifest-json", type=Path, default=DEFAULT_FIELD_PACKET_MANIFEST_JSON)
    parser.add_argument("--forced-anchor-gpx-manifest-json", type=Path, default=DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON)
    parser.add_argument("--day-level-gpx-manifest-json", type=Path, default=DEFAULT_DAY_LEVEL_GPX_MANIFEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        read_json(args.draft_json),
        personal_candidates=personal_index(args.personal_route_menu_json),
        hybrid_candidates=hybrid_index(args.hybrid_route_pass_json),
        field_packet_gpx=field_packet_gpx_index(args.field_packet_manifest_json),
        forced_anchor_gpx=forced_anchor_gpx_index(args.forced_anchor_gpx_manifest_json),
        day_level_gpx_manifest=read_json(args.day_level_gpx_manifest_json)
        if args.day_level_gpx_manifest_json.exists()
        else None,
    )
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
