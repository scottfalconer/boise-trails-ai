#!/usr/bin/env python3
"""Audit exported field-packet GPX files for unclaimed official segment credit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from field_activity_review import (  # noqa: E402
    activity_coordinates,
    normalized_ids,
    review_activity_against_segments,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    bbox_overlaps,
    coordinate_bbox,
    downsample_coords,
    load_dem_context,
    load_official_segments,
    read_json,
    write_json,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "field-latent-credit-audit-2026-05-11.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "field-latent-credit-audit-2026-05-11.md"


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("label") or route.get("block_name") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def route_gpx_path(route: dict[str, Any], packet_dir: Path) -> Path | None:
    href = route.get("gpx_href")
    if not href:
        return None
    path = Path(str(href))
    if path.is_absolute():
        return path
    return packet_dir / path


def build_segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment["seg_id"]): segment for segment in official_segments}


def segment_brief(segment_index: dict[str, dict[str, Any]], segment_id: str) -> dict[str, Any]:
    segment = segment_index.get(str(segment_id), {})
    return {
        "seg_id": str(segment_id),
        "seg_name": segment.get("seg_name"),
        "trail_name": segment.get("trail_name"),
        "direction": segment.get("direction"),
        "official_miles": round(float(segment.get("official_miles") or 0), 2),
    }


def build_segment_claim_index(
    routes: list[dict[str, Any]],
    ownership_cards: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    claims: dict[str, list[dict[str, Any]]] = {}
    for route in [*routes, *(ownership_cards or [])]:
        is_manual_hold = bool(route.get("manual_design_hold"))
        claim = {
            "route_key": route_key(route),
            "outing_id": route.get("outing_id"),
            "label": route_label(route),
            "candidate_ids": route.get("candidate_ids") or [],
            "ownership_status": "manual_hold" if is_manual_hold else "active_route",
        }
        for segment_id in normalized_ids(route.get("remaining_segment_ids") or route.get("segment_ids") or []):
            claims.setdefault(segment_id, []).append(claim)
    return claims


def candidate_segments_for_activity(
    activity_coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    planned_ids: set[str],
    threshold_miles: float,
) -> list[dict[str, Any]]:
    if len(activity_coords) < 2:
        return official_segments
    activity_bbox = coordinate_bbox(activity_coords)
    origin_lat = sum(point[1] for point in activity_coords) / len(activity_coords)
    lat_buffer = threshold_miles / 69.0
    lon_buffer = threshold_miles / max(1e-6, 69.172 * math.cos(math.radians(origin_lat)))
    candidates = []
    for segment in official_segments:
        segment_id = str(segment["seg_id"])
        segment_bbox = segment.get("bbox") or coordinate_bbox(segment["coordinates"])
        if segment_id in planned_ids or bbox_overlaps(activity_bbox, segment_bbox, lon_buffer, lat_buffer):
            candidates.append(segment)
    return candidates


def audit_route(
    route: dict[str, Any],
    *,
    official_segments: list[dict[str, Any]],
    segment_index: dict[str, dict[str, Any]],
    segment_claim_index: dict[str, list[dict[str, Any]]],
    completed_at_export: set[str],
    packet_dir: Path,
    elevation_sampler: Any = None,
    threshold_miles: float = 0.045,
    endpoint_threshold_miles: float | None = None,
    min_fraction: float = 0.85,
    partial_min_fraction: float = 0.2,
    max_activity_points: int = 1200,
) -> dict[str, Any]:
    planned_ids = set(normalized_ids(route.get("segment_ids") or []))
    current_route_key = route_key(route)
    gpx_path = route_gpx_path(route, packet_dir)
    base = {
        "route_key": current_route_key,
        "outing_id": route.get("outing_id"),
        "label": route_label(route),
        "candidate_ids": route.get("candidate_ids") or [],
        "planned_segment_ids": normalized_ids(planned_ids),
        "gpx_path": str(gpx_path) if gpx_path else None,
    }
    if not gpx_path or not gpx_path.exists():
        return {
            **base,
            "audit_status": "missing_gpx",
            "latent_completed_segment_ids": [],
            "unexpected_latent_segment_ids": [],
            "claimed_elsewhere_segment_ids": [],
            "reconciled_claimed_elsewhere_segment_ids": [],
            "unclaimed_uncompleted_segment_ids": [],
            "repeat_completed_segment_ids": [],
            "partial_segment_ids": [],
            "segments": [],
        }

    activity_coords = activity_coordinates(gpx_path)
    review_coords = downsample_coords(activity_coords, max_points=max_activity_points)
    candidate_segments = candidate_segments_for_activity(
        activity_coords,
        official_segments,
        planned_ids,
        threshold_miles,
    )
    review = review_activity_against_segments(
        review_coords,
        candidate_segments,
        planned_segment_ids=planned_ids,
        planned_outing_id=current_route_key,
        threshold_miles=threshold_miles,
        endpoint_threshold_miles=endpoint_threshold_miles,
        min_fraction=min_fraction,
        partial_min_fraction=partial_min_fraction,
        elevation_sampler=elevation_sampler,
        evidence_refs=[str(gpx_path)],
    )
    latent_ids = set(review["extra_completed_segment_ids"])
    repeat_completed_ids = latent_ids & completed_at_export
    unexpected_ids = latent_ids - completed_at_export
    reconciliation = route.get("segment_ownership_reconciliation") or {}
    declared_owned_elsewhere_ids = set(
        normalized_ids(reconciliation.get("declared_owned_elsewhere_segment_ids") or [])
    )
    claimed_elsewhere_ids: set[str] = set()
    reconciled_claimed_elsewhere_ids: set[str] = set()
    unclaimed_uncompleted_ids: set[str] = set()
    segment_rows = []
    for segment_id in normalized_ids(latent_ids):
        other_claims = [
            claim
            for claim in segment_claim_index.get(segment_id, [])
            if claim.get("route_key") != current_route_key
        ]
        if segment_id in unexpected_ids and other_claims and segment_id in declared_owned_elsewhere_ids:
            reconciled_claimed_elsewhere_ids.add(segment_id)
            status = "reconciled_owned_elsewhere"
        elif segment_id in unexpected_ids and other_claims:
            claimed_elsewhere_ids.add(segment_id)
            status = "unexpected_latent_credit"
        elif segment_id in unexpected_ids:
            unclaimed_uncompleted_ids.add(segment_id)
            status = "unexpected_latent_credit"
        elif segment_id in repeat_completed_ids:
            status = "repeat_completed"
        else:
            status = "latent_completed"
        segment_rows.append(
            {
                **segment_brief(segment_index, segment_id),
                "status": status,
                "claimed_by_other_routes": other_claims,
            }
        )

    if claimed_elsewhere_ids or unclaimed_uncompleted_ids:
        audit_status = "needs_repair"
    elif reconciled_claimed_elsewhere_ids:
        audit_status = "reconciled"
    elif latent_ids:
        audit_status = "repeat_only"
    else:
        audit_status = "passed"

    return {
        **base,
        "audit_status": audit_status,
        "activity_point_count": len(activity_coords),
        "review_point_count": len(review_coords),
        "candidate_segment_count": len(candidate_segments),
        "latent_completed_segment_ids": normalized_ids(latent_ids),
        "unexpected_latent_segment_ids": normalized_ids(unexpected_ids),
        "claimed_elsewhere_segment_ids": normalized_ids(claimed_elsewhere_ids),
        "reconciled_claimed_elsewhere_segment_ids": normalized_ids(reconciled_claimed_elsewhere_ids),
        "unclaimed_uncompleted_segment_ids": normalized_ids(unclaimed_uncompleted_ids),
        "repeat_completed_segment_ids": normalized_ids(repeat_completed_ids),
        "partial_segment_ids": review["partial_segment_ids"],
        "segments": segment_rows,
    }


def build_latent_credit_audit(
    field_tool_data: dict[str, Any],
    *,
    official_segments: list[dict[str, Any]],
    packet_dir: Path,
    elevation_sampler: Any = None,
    threshold_miles: float = 0.045,
    endpoint_threshold_miles: float | None = None,
    min_fraction: float = 0.85,
    partial_min_fraction: float = 0.2,
    max_activity_points: int = 1200,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    manual_holds = field_tool_data.get("manual_holds") or []
    progress = field_tool_data.get("progress") or {}
    completed_at_export = set(normalized_ids(progress.get("completed_segment_ids_at_export") or []))
    segment_index = build_segment_index(official_segments)
    segment_claim_index = build_segment_claim_index(routes, ownership_cards=manual_holds)
    active_segment_claim_index = build_segment_claim_index(routes)
    route_reviews = [
        audit_route(
            route,
            official_segments=official_segments,
            segment_index=segment_index,
            segment_claim_index=segment_claim_index,
            completed_at_export=completed_at_export,
            packet_dir=packet_dir,
            elevation_sampler=elevation_sampler,
            threshold_miles=threshold_miles,
            endpoint_threshold_miles=endpoint_threshold_miles,
            min_fraction=min_fraction,
            partial_min_fraction=partial_min_fraction,
            max_activity_points=max_activity_points,
        )
        for route in routes
    ]
    missing_gpx_routes = [row for row in route_reviews if row["audit_status"] == "missing_gpx"]
    repair_routes = [row for row in route_reviews if row["audit_status"] == "needs_repair"]
    reconciled_routes = [row for row in route_reviews if row["audit_status"] == "reconciled"]
    repeat_only_routes = [row for row in route_reviews if row["audit_status"] == "repeat_only"]

    # Dual-claim guard: one official segment must be exact credit for at most one
    # active route. A route's segment_ids is its pure exact-credit claim (segments
    # it walks only as repeat/connector are excluded and listed under
    # declared_owned_elsewhere instead), so any seg_id in >1 route's segment_ids
    # is double-counted official credit. This is what let segment 1680 be claimed
    # by both routes 17 and 18A and inflate plan-wide official miles.
    dual_claimed_segments = []
    for seg_id, claims in sorted(active_segment_claim_index.items()):
        if seg_id in completed_at_export or len(claims) <= 1:
            continue
        brief = segment_index.get(seg_id) or {}
        dual_claimed_segments.append(
            {
                "seg_id": seg_id,
                "trail_name": brief.get("trail_name") or brief.get("seg_name"),
                "official_miles": brief.get("official_miles"),
                "claiming_routes": [
                    {"outing_id": claim.get("outing_id"), "label": claim.get("label")}
                    for claim in claims
                ],
            }
        )

    if missing_gpx_routes:
        status = "incomplete_audit"
    elif repair_routes or dual_claimed_segments:
        status = "needs_repair"
    else:
        status = "passed"

    unexpected_segment_ids = set()
    claimed_elsewhere_ids = set()
    reconciled_claimed_elsewhere_ids = set()
    unclaimed_uncompleted_ids = set()
    repeat_completed_ids = set()
    for row in route_reviews:
        unexpected_segment_ids.update(row["unexpected_latent_segment_ids"])
        claimed_elsewhere_ids.update(row["claimed_elsewhere_segment_ids"])
        reconciled_claimed_elsewhere_ids.update(row["reconciled_claimed_elsewhere_segment_ids"])
        unclaimed_uncompleted_ids.update(row["unclaimed_uncompleted_segment_ids"])
        repeat_completed_ids.update(row["repeat_completed_segment_ids"])

    return {
        "schema": "boise_trails_field_latent_credit_audit_v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "route_count": len(routes),
            "routes_missing_gpx": len(missing_gpx_routes),
            "routes_needing_repair": len(repair_routes),
            "routes_reconciled": len(reconciled_routes),
            "repeat_only_routes": len(repeat_only_routes),
            "unexpected_latent_segment_count": len(unexpected_segment_ids),
            "claimed_elsewhere_segment_count": len(claimed_elsewhere_ids),
            "reconciled_claimed_elsewhere_segment_count": len(reconciled_claimed_elsewhere_ids),
            "unclaimed_uncompleted_segment_count": len(unclaimed_uncompleted_ids),
            "repeat_completed_segment_count": len(repeat_completed_ids),
            "dual_claimed_exact_credit_segment_count": len(dual_claimed_segments),
        },
        "dual_claimed_exact_credit_segments": dual_claimed_segments,
        "source_files": source_files or {},
        "parameters": {
            "threshold_miles": threshold_miles,
            "endpoint_threshold_miles": endpoint_threshold_miles or threshold_miles,
            "min_fraction": min_fraction,
            "partial_min_fraction": partial_min_fraction,
            "max_activity_points": max_activity_points,
        },
        "routes_needing_repair": repair_routes,
        "reconciled_routes": reconciled_routes,
        "repeat_only_routes": repeat_only_routes,
        "missing_gpx_routes": missing_gpx_routes,
        "route_reviews": route_reviews,
    }


def segment_line(segment: dict[str, Any]) -> str:
    label = f"{segment['seg_id']} {segment.get('trail_name') or segment.get('seg_name') or 'Unknown trail'}"
    direction = segment.get("direction")
    if direction and direction != "both":
        label += f" ({direction})"
    claims = segment.get("claimed_by_other_routes") or []
    if claims:
        claim_labels = ", ".join(str(claim.get("label") or claim.get("outing_id")) for claim in claims)
        label += f"; claimed by {claim_labels}"
    return label


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Field latent credit audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Routes audited: {summary['route_count']}",
        f"- Routes needing repair: {summary['routes_needing_repair']}",
        f"- Routes with reconciled latent credit: {summary['routes_reconciled']}",
        f"- Unexpected latent official segments: {summary['unexpected_latent_segment_count']}",
        f"- Unreconciled latent segments claimed by another active route: {summary['claimed_elsewhere_segment_count']}",
        f"- Reconciled latent segments claimed by another active route: {summary['reconciled_claimed_elsewhere_segment_count']}",
        f"- Unclaimed uncompleted latent segments: {summary['unclaimed_uncompleted_segment_count']}",
        f"- Repeat-only latent completed segments: {summary['repeat_completed_segment_count']}",
        "",
        "## Scope",
        "",
        "- This audit proves segment-credit provenance: latent official segments in route GPX files are either declared against another active route card, already completed at export, or surfaced as repair debt.",
        "- A passing result makes the packet more executable and auditable; it does not prove lower total on-foot miles, lower p75/p90 time, better sequencing, or net human-effort reduction.",
        "- Effort reduction still requires route-card replacement or field-day repricing after validated activity progress changes the remaining segment set.",
        "",
    ]
    if audit["routes_needing_repair"]:
        lines.extend(["## Routes needing repair", ""])
        for route in audit["routes_needing_repair"]:
            lines.append(f"### {route['label']}")
            lines.append(f"- GPX: `{route['gpx_path']}`")
            lines.append(f"- Planned segments: {', '.join(route['planned_segment_ids'])}")
            if route["claimed_elsewhere_segment_ids"]:
                lines.append(
                    "- Latent segments claimed elsewhere: "
                    + ", ".join(route["claimed_elsewhere_segment_ids"])
                )
            if route["unclaimed_uncompleted_segment_ids"]:
                lines.append(
                    "- Latent segments not claimed by active menu: "
                    + ", ".join(route["unclaimed_uncompleted_segment_ids"])
                )
            lines.append("- Segment details:")
            for segment in route["segments"]:
                if segment.get("status") == "unexpected_latent_credit":
                    lines.append(f"  - {segment_line(segment)}")
            lines.append("")
    if audit["reconciled_routes"]:
        lines.extend(["## Reconciled latent credit", ""])
        for route in audit["reconciled_routes"]:
            lines.append(f"### {route['label']}")
            lines.append(f"- GPX: `{route['gpx_path']}`")
            lines.append(
                "- Declared owned by other active routes: "
                + ", ".join(route["reconciled_claimed_elsewhere_segment_ids"])
            )
            lines.append("- Segment details:")
            for segment in route["segments"]:
                if segment.get("status") == "reconciled_owned_elsewhere":
                    lines.append(f"  - {segment_line(segment)}")
            lines.append("")
    if audit["repeat_only_routes"]:
        lines.extend(["## Repeat-only latent credit", ""])
        for route in audit["repeat_only_routes"]:
            lines.append(
                f"- {route['label']}: {', '.join(route['repeat_completed_segment_ids'])}"
            )
        lines.append("")
    if audit["missing_gpx_routes"]:
        lines.extend(["## Missing GPX", ""])
        for route in audit["missing_gpx_routes"]:
            lines.append(f"- {route['label']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--threshold-miles", type=float, default=0.045)
    parser.add_argument("--endpoint-threshold-miles", type=float)
    parser.add_argument("--min-fraction", type=float, default=0.85)
    parser.add_argument("--partial-min-fraction", type=float, default=0.2)
    parser.add_argument("--max-activity-points", type=int, default=1200)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write the audit artifacts but return success even when latent credit needs repair.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    field_tool_data = read_json(args.field_tool_data_json)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    dem_context = load_dem_context(args.dem_tif, args.dem_summary_json)
    audit = build_latent_credit_audit(
        field_tool_data,
        official_segments=official_segments,
        packet_dir=args.packet_dir,
        elevation_sampler=dem_context.get("sampler"),
        threshold_miles=args.threshold_miles,
        endpoint_threshold_miles=args.endpoint_threshold_miles,
        min_fraction=args.min_fraction,
        partial_min_fraction=args.partial_min_fraction,
        max_activity_points=args.max_activity_points,
        source_files={
            "field_tool_data_json": str(args.field_tool_data_json),
            "packet_dir": str(args.packet_dir),
            "official_geojson": str(args.official_geojson),
            "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
            "dem_tif": str(args.dem_tif),
            "dem_summary_json": str(args.dem_summary_json),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] != "passed" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
