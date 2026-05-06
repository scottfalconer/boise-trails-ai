#!/usr/bin/env python3
"""Audit the generated field tool against the field-use definition of done."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_FIELD_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_FIELD_TOOL_DATA_JSON = DEFAULT_FIELD_PACKET_DIR / "field-tool-data.json"
DEFAULT_MANIFEST_JSON = DEFAULT_FIELD_PACKET_DIR / "manifest.json"
DEFAULT_INDEX_HTML = DEFAULT_FIELD_PACKET_DIR / "index.html"
DEFAULT_CANONICAL_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
DEFAULT_RECERTIFICATION_JSON = YEAR_DIR / "outputs" / "private" / "progress" / "field-recertification-latest.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "field-tool-completion-audit-2026-05-06.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "field-tool-completion-audit-2026-05-06.md"

PRIVATE_LITERAL_PATTERNS = (
    "/Users/scott",
    "outputs/private",
    "GETAthleteDashboard",
    "access_token",
    "refresh_token",
    "client_secret",
)
PRIVATE_REGEX_PATTERNS = (
    re.compile(r"\b911\s+n\.?\s+18th\b", re.IGNORECASE),
    re.compile(r"\b911\s+north\s+18th\b", re.IGNORECASE),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_sha256(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=lambda item: (len(item), item))


def official_segment_ids(official_geojson: dict[str, Any]) -> list[str]:
    return normalized_ids(
        (feature.get("properties") or {}).get("segId")
        for feature in official_geojson.get("features") or []
        if (feature.get("properties") or {}).get("segId") is not None
    )


def requirement(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"requirement": name, "passed": bool(passed), "evidence": evidence}


def href_exists(packet_dir: Path, href: str | None) -> bool:
    if not href:
        return False
    return (packet_dir / href).exists()


def scan_public_safety(paths: list[Path]) -> list[str]:
    failures = []
    for path in paths:
        if not path.exists() or not path.is_file():
            failures.append(f"{path}: missing")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in PRIVATE_LITERAL_PATTERNS:
            if token in text:
                failures.append(f"{path}: contains private token {token!r}")
        for pattern in PRIVATE_REGEX_PATTERNS:
            if pattern.search(text):
                failures.append(f"{path}: contains exact home address pattern")
    return failures


def route_field_failures(routes: list[dict[str, Any]], packet_dir: Path, official_ids: set[str]) -> list[str]:
    failures = []
    for route in routes:
        label = f"{route.get('label') or route.get('outing_id')} {route.get('trailhead') or ''}".strip()
        parking = route.get("parking") or {}
        effort = route.get("effort") or {}
        validation = route.get("validation") or {}
        completion_safety = route.get("completion_safety") or {}
        segment_ids = set(normalized_ids(route.get("segment_ids")))
        steps = route.get("turn_by_turn_steps") or []
        if parking.get("lat") is None or parking.get("lon") is None or parking.get("has_parking") is not True:
            failures.append(f"{label}: missing verified parked start")
        if validation.get("passed") is not True:
            failures.append(f"{label}: GPX validation did not pass")
        if not href_exists(packet_dir, route.get("gpx_href")):
            failures.append(f"{label}: missing Nav GPX file")
        if not segment_ids:
            failures.append(f"{label}: no official segment ids")
        if not segment_ids <= official_ids:
            failures.append(f"{label}: segment ids outside official target")
        if not route.get("door_to_door_minutes_p75") or not route.get("door_to_door_minutes_p90"):
            failures.append(f"{label}: missing p75/p90 door-to-door time")
        if not route.get("on_foot_miles") or not route.get("official_miles"):
            failures.append(f"{label}: missing mileage")
        if effort.get("ascent_ft") is None or effort.get("descent_ft") is None or effort.get("grade_adjusted_miles") is None:
            failures.append(f"{label}: missing DEM effort")
        if effort.get("estimated_moving_minutes_p75") is None:
            failures.append(f"{label}: missing p75 moving effort")
        if completion_safety.get("normal_completion_preserves_remaining_menu_coverage") is not True:
            failures.append(f"{label}: normal completion does not preserve remaining menu coverage")
        step_kinds = {str(step.get("kind")) for step in steps}
        if not {"park", "navigate", "return"} <= step_kinds:
            failures.append(f"{label}: turn cues must include park, navigate, and return steps")
    return failures


def build_completion_audit(
    *,
    field_tool_data: dict[str, Any],
    manifest: dict[str, Any],
    official_geojson: dict[str, Any],
    index_html: str,
    packet_dir: Path,
    canonical_map_data: dict[str, Any] | None = None,
    recertification_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official_ids = set(official_segment_ids(official_geojson))
    routes = field_tool_data.get("routes") or []
    route_segment_ids = {
        segment_id
        for route in routes
        for segment_id in normalized_ids(route.get("segment_ids"))
    }
    source = field_tool_data.get("source") or {}
    source_hash = source.get("map_data_sha256")
    canonical_hash = stable_json_sha256(canonical_map_data) if canonical_map_data is not None else None
    baseline = field_tool_data.get("certified_baseline") or {}
    summary = field_tool_data.get("summary") or {}
    manifest_summary = manifest.get("summary") or {}
    recert_summary = (recertification_report or {}).get("summary") or {}
    route_failures = route_field_failures(routes, packet_dir, official_ids)
    safety_failures = scan_public_safety(
        [
            packet_dir / "index.html",
            packet_dir / "field-tool-data.json",
            packet_dir / "manifest.json",
            packet_dir / "service-worker.js",
        ]
    )
    required_filters = [60, 90, 120, 180, 240, 360]
    checks = [
        requirement(
            "Phone page and map share the canonical field-menu source",
            bool(source_hash and canonical_hash and source_hash == canonical_hash),
            f"field source hash {source_hash}; canonical map hash {canonical_hash}",
        ),
        requirement(
            "Certified completion baseline covers 251 official segments",
            baseline.get("status") == "passed"
            and int(baseline.get("official_segment_count") or 0) == 251
            and int(baseline.get("covered_segment_count") or 0) == 251
            and int(baseline.get("missing_segment_count") or 0) == 0,
            json.dumps(
                {
                    "status": baseline.get("status"),
                    "official": baseline.get("official_segment_count"),
                    "covered": baseline.get("covered_segment_count"),
                    "missing": baseline.get("missing_segment_count"),
                },
                sort_keys=True,
            ),
        ),
        requirement(
            "Daily filtering supports the required door-to-door windows",
            field_tool_data.get("time_filters_minutes") == required_filters
            and all(f'data-filter="{value}"' in index_html for value in required_filters),
            f"filters {field_tool_data.get('time_filters_minutes')}",
        ),
        requirement(
            "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort",
            not route_failures and bool(routes),
            "; ".join(route_failures[:12]) if route_failures else f"{len(routes)} route cards passed field checks",
        ),
        requirement(
            "Field menu covers every official segment geometry id",
            route_segment_ids == official_ids
            and int(summary.get("segment_count_in_field_menu") or 0) == len(official_ids),
            f"field menu {len(route_segment_ids)} ids; official target {len(official_ids)} ids",
        ),
        requirement(
            "GPX validation passed for every runnable outing",
            manifest_summary.get("gpx_validation_passed") is True
            and int(manifest_summary.get("navigation_gpx_count") or 0) == len(routes)
            and int(manifest_summary.get("failed_gpx_count") or 0) == 0,
            json.dumps(
                {
                    "navigation": manifest_summary.get("navigation_gpx_count"),
                    "failed": manifest_summary.get("failed_gpx_count"),
                    "passed": manifest_summary.get("gpx_validation_passed"),
                },
                sort_keys=True,
            ),
        ),
        requirement(
            "Phone progress can hide completed outings and export reviewed progress",
            "fieldPacketCompletedOutings" in index_html
            and "Mark done" in index_html
            and "Hide completed" in index_html
            and "Export progress" in index_html
            and "missed_segment_ids" in index_html,
            "localStorage completion, hide completed, export progress, and missed segment review fields are present",
        ),
        requirement(
            "Best-today recommendation uses the active time window and remaining segment ids",
            "Best today for" in index_html
            and "new official segment(s)" in index_html
            and "completion-safe in the current menu" in index_html
            and "completedSegmentSet" in index_html,
            "phone JavaScript ranks visible incomplete cards by completion-safety and new remaining segment count inside the active filter",
        ),
        requirement(
            "Adaptive recertification reports whether selected-profile completion remains feasible",
            (recertification_report or {}).get("status") == "passed"
            and recert_summary.get("remaining_full_completion_feasible") is True
            and recert_summary.get("remaining_coverage_preserved") is True,
            json.dumps(
                {
                    "status": (recertification_report or {}).get("status"),
                    "remaining_full_completion_feasible": recert_summary.get("remaining_full_completion_feasible"),
                    "remaining_coverage_preserved": recert_summary.get("remaining_coverage_preserved"),
                },
                sort_keys=True,
            ),
        ),
        requirement(
            "Public field outputs do not expose private origin, tokens, dashboard data, or private paths",
            not safety_failures,
            "; ".join(safety_failures[:8]) if safety_failures else "public packet files passed private-token scan",
        ),
    ]
    status = "passed" if all(check["passed"] for check in checks) else "failed"
    return {
        "schema": "boise_trails_field_tool_completion_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": "field-usable daily decision tool with certified full-completion baseline",
        "status": status,
        "summary": {
            "passed_requirement_count": len([check for check in checks if check["passed"]]),
            "requirement_count": len(checks),
            "route_count": len(routes),
            "official_segment_count": len(official_ids),
            "field_menu_segment_count": len(route_segment_ids),
        },
        "checks": checks,
    }


def render_md(audit: dict[str, Any]) -> str:
    lines = [
        "# Field Tool Completion Audit - 2026-05-06",
        "",
        f"- Status: `{audit['status']}`",
        f"- Requirements: {audit['summary']['passed_requirement_count']} / {audit['summary']['requirement_count']} passed",
        f"- Runnable route cards: {audit['summary']['route_count']}",
        f"- Official segment coverage: {audit['summary']['field_menu_segment_count']} / {audit['summary']['official_segment_count']}",
        "",
        "## Requirement Checklist",
        "",
        "| Requirement | Status | Evidence |",
        "|---|---|---|",
    ]
    for check in audit["checks"]:
        status = "Pass" if check["passed"] else "Fail"
        evidence = str(check["evidence"]).replace("|", "\\|")
        lines.append(f"| {check['requirement']} | {status} | {evidence} |")
    lines.extend(
        [
            "",
            "## Validation Commands",
            "",
            "- `python years/2026/scripts/export_mobile_field_packet.py`",
            "- `python years/2026/scripts/field_progress_report.py`",
            "- `python years/2026/scripts/field_recertification_report.py`",
            "- `python years/2026/scripts/field_tool_completion_audit.py`",
            "- `pytest -q years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_progress_report.py years/2026/tests/test_field_recertification_report.py years/2026/tests/test_field_tool_completion_audit.py`",
            "",
            "## Remaining Risk",
            "",
            "This audit verifies the generated field tool and selected-profile recertification gates. It is not a global proof of optimality over every possible real-world route or a substitute for day-of trail signage and condition checks.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--index-html", type=Path, default=DEFAULT_INDEX_HTML)
    parser.add_argument("--canonical-map-data-json", type=Path, default=DEFAULT_CANONICAL_MAP_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--recertification-json", type=Path, default=DEFAULT_RECERTIFICATION_JSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_FIELD_PACKET_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_completion_audit(
        field_tool_data=read_json(args.field_tool_data_json),
        manifest=read_json(args.manifest_json),
        official_geojson=read_json(args.official_geojson),
        index_html=args.index_html.read_text(encoding="utf-8"),
        packet_dir=args.packet_dir,
        canonical_map_data=read_json(args.canonical_map_data_json),
        recertification_report=read_json(args.recertification_json) if args.recertification_json.exists() else None,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_md(audit), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": audit["status"], "summary": audit["summary"]}, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
