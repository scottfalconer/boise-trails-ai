#!/usr/bin/env python3
"""Run a first-pass gate repair audit for Bogus B1/B2 candidates."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
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
    haversine_miles,
    normalized_ids,
    round_miles,
    sort_id,
    write_json,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_TEMPLATE_CANDIDATES_JSON = YEAR_DIR / "checkpoints" / "template-route-candidates-2026-05-12.json"
RUN_ID = "bogus-b1-b2-gate-repair-audit-2026-05-13"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / f"{RUN_ID}.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / f"{RUN_ID}.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / f"{RUN_ID}-manifest.json"

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


def resolve_repo_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_gpx_track_lonlat(path: Path) -> list[tuple[float, float]]:
    root = ET.parse(path).getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    points = []
    for trkpt in root.findall(".//g:trkpt", ns):
        points.append((float(trkpt.attrib["lon"]), float(trkpt.attrib["lat"])))
    return points


def point_path_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def interpolate_point(
    left: tuple[float, float],
    right: tuple[float, float],
    fraction: float,
) -> tuple[float, float]:
    return (
        left[0] + (right[0] - left[0]) * fraction,
        left[1] + (right[1] - left[1]) * fraction,
    )


def append_point(target: list[tuple[float, float]], point: tuple[float, float]) -> None:
    if target and target[-1] == point:
        return
    target.append(point)


def clip_track_by_miles(
    coords: list[tuple[float, float]],
    start_miles: float,
    end_miles: float,
) -> list[tuple[float, float]]:
    if len(coords) < 2 or end_miles <= start_miles:
        return []
    clipped: list[tuple[float, float]] = []
    covered = 0.0
    for left, right in zip(coords, coords[1:]):
        segment_miles = haversine_miles(left, right)
        if segment_miles <= 0:
            continue
        next_covered = covered + segment_miles
        if next_covered < start_miles:
            covered = next_covered
            continue
        if covered > end_miles:
            break
        start_fraction = max(0.0, (start_miles - covered) / segment_miles)
        end_fraction = min(1.0, (end_miles - covered) / segment_miles)
        if start_fraction <= 1.0 and end_fraction >= 0.0 and start_fraction <= end_fraction:
            append_point(clipped, interpolate_point(left, right, start_fraction))
            append_point(clipped, interpolate_point(left, right, end_fraction))
        covered = next_covered
    return clipped


def route_gpx_path(route: dict[str, Any], packet_dir: Path) -> Path | None:
    href = route.get("gpx_href") or route.get("audit_gpx_href")
    if not href:
        return None
    return packet_dir / str(href)


def cue_gpx_leg_review(
    *,
    route: dict[str, Any] | None,
    cue: dict[str, Any] | None,
    packet_dir: Path,
) -> dict[str, Any]:
    if not route or not cue:
        return {"status": "missing_source_route_or_cue", "gpx_leg_miles": None}
    gpx_path = route_gpx_path(route, packet_dir)
    route_miles = cue.get("route_miles")
    route_leg_miles = cue.get("route_leg_miles")
    if not gpx_path or not gpx_path.exists():
        return {
            "status": "source_route_gpx_missing",
            "gpx_leg_miles": round_miles(float_value(route_leg_miles or cue.get("leg_miles"))),
            "source_gpx": display_path(gpx_path) if gpx_path else None,
        }
    if route_miles is None or route_leg_miles is None:
        return {
            "status": "source_cue_lacks_route_mile_offsets",
            "gpx_leg_miles": round_miles(float_value(cue.get("leg_miles"))),
            "source_gpx": display_path(gpx_path),
        }
    track = parse_gpx_track_lonlat(gpx_path)
    start_miles = float_value(route_miles)
    end_miles = start_miles + float_value(route_leg_miles)
    clipped = clip_track_by_miles(track, start_miles, end_miles)
    if len(clipped) < 2:
        return {
            "status": "source_cue_gpx_leg_not_extractable",
            "gpx_leg_miles": round_miles(float_value(route_leg_miles)),
            "source_gpx": display_path(gpx_path),
        }
    return {
        "status": "source_cue_gpx_leg_extractable",
        "gpx_leg_miles": round_miles(point_path_miles(clipped)),
        "source_gpx": display_path(gpx_path),
        "source_route_miles": round_miles(start_miles),
        "source_route_leg_miles": round_miles(float_value(route_leg_miles)),
        "source_track_point_count": len(track),
        "extracted_track_point_count": len(clipped),
    }


def rendered_candidate_gpx_miles(loop: dict[str, Any]) -> dict[str, Any]:
    gpx_path = resolve_repo_path(loop.get("gpx_path"))
    if not gpx_path or not gpx_path.exists():
        return {
            "status": "candidate_gpx_missing_using_reported_track_miles",
            "gpx_miles": round_miles(float_value(loop.get("track_miles"))),
            "gpx_path": display_path(gpx_path) if gpx_path else None,
        }
    coords = parse_gpx_track_lonlat(gpx_path)
    return {
        "status": "candidate_gpx_measured",
        "gpx_miles": round_miles(point_path_miles(coords)),
        "gpx_path": display_path(gpx_path),
        "track_point_count": len(coords),
    }


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
    packet_dir: Path = DEFAULT_PACKET_DIR,
) -> dict[str, Any]:
    rows = []
    repair_delta_miles = 0.0
    direct_gap_count = 0
    candidate_gpx = rendered_candidate_gpx_miles(loop)
    for link in loop.get("link_rows") or []:
        if link.get("path_source") != "direct_gap_fallback":
            continue
        direct_gap_count += 1
        to_segment_id = str(link.get("to_segment_id"))
        mapping = DIRECT_GAP_REPAIR_CUE_REUSE.get((bundle_id, to_segment_id))
        source_route = None
        cue = None
        if mapping:
            source_route = routes_by_label.get(str(mapping.get("source_route_label")))
            cue = find_reusable_cue(
                routes_by_label=routes_by_label,
                source_route_label=str(mapping.get("source_route_label")),
                cue_type=mapping.get("cue_type"),
                target_contains=mapping.get("target_contains"),
            )
        original_miles = float_value(link.get("link_distance_miles"))
        cue_gpx = cue_gpx_leg_review(route=source_route, cue=cue, packet_dir=packet_dir)
        replacement_miles = (
            float_value(cue_gpx.get("gpx_leg_miles"))
            if cue_gpx.get("gpx_leg_miles") is not None
            else (float_value((cue or {}).get("route_leg_miles") or (cue or {}).get("leg_miles")) if cue else None)
        )
        if replacement_miles is not None:
            repair_delta_miles += replacement_miles - original_miles
        source_repeat_miles = float_value((cue or {}).get("official_repeat_miles"))
        rows.append(
            {
                "to_segment_id": to_segment_id,
                "to_segment_name": link.get("to_segment_name"),
                "original_direct_gap_miles": round_miles(original_miles),
                "repair_status": "source_cue_gpx_available_but_candidate_not_rebuilt" if cue else "no_named_repair_cue_found",
                "source_route_label": (mapping or {}).get("source_route_label"),
                "source_cue_seq": (cue or {}).get("seq"),
                "source_cue_type": (cue or {}).get("cue_type"),
                "source_cue_signed_as": (cue or {}).get("signed_as"),
                "source_cue_target": (cue or {}).get("target"),
                "source_cue_display_leg_miles": round_miles(float_value((cue or {}).get("leg_miles"))) if cue else None,
                "source_cue_route_leg_miles": round_miles(float_value((cue or {}).get("route_leg_miles"))) if cue else None,
                "source_cue_official_repeat_miles": round_miles(source_repeat_miles) if cue else None,
                "source_cue_gpx_review": cue_gpx,
                "replacement_leg_miles": round_miles(replacement_miles) if replacement_miles is not None else None,
                "delta_miles_vs_direct_gap": round_miles((replacement_miles - original_miles) if replacement_miles is not None else 0.0),
                "candidate_gpx_rebuilt_with_replacement": False,
                "reason": (mapping or {}).get("reason"),
            }
        )
    original_track_miles = float_value(candidate_gpx.get("gpx_miles") or loop.get("track_miles"))
    named_cue_geometry_count = sum(
        1
        for row in rows
        if row["repair_status"] == "source_cue_gpx_available_but_candidate_not_rebuilt"
        and (row.get("source_cue_gpx_review") or {}).get("status") == "source_cue_gpx_leg_extractable"
    )
    if not direct_gap_count:
        status = "passed_no_direct_gap"
    elif named_cue_geometry_count == direct_gap_count:
        status = "failed_source_cue_gpx_available_but_candidate_not_rebuilt"
    else:
        status = "failed_direct_gap_fallback_unresolved"
    return {
        "status": status,
        "direct_gap_count": direct_gap_count,
        "original_direct_gap_miles": round_miles(float_value(loop.get("direct_gap_fallback_miles"))),
        "named_cue_repair_count": sum(1 for row in rows if row["repair_status"] == "source_cue_gpx_available_but_candidate_not_rebuilt"),
        "source_cue_gpx_leg_count": named_cue_geometry_count,
        "candidate_rendered_gpx": candidate_gpx,
        "post_named_cue_priced_track_miles": round_miles(original_track_miles + repair_delta_miles),
        "post_source_cue_gpx_priced_track_miles": round_miles(original_track_miles + repair_delta_miles),
        "repair_delta_miles": round_miles(repair_delta_miles),
        "rows": rows,
        "note": "A source route GPX cue can price a replacement leg, but it does not remove the promotion blocker until the candidate GPX is rebuilt without direct_gap_fallback geometry.",
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


def connector_miles_for_direct_gap_rows(rows: list[dict[str, Any]]) -> float:
    return sum(float_value(row.get("original_direct_gap_miles")) for row in rows)


def replacement_connector_miles_for_direct_gap_rows(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        replacement_miles = float_value(row.get("replacement_leg_miles"))
        repeat_miles = float_value(row.get("source_cue_official_repeat_miles"))
        total += max(0.0, replacement_miles - repeat_miles)
    return total


def replacement_repeat_miles_for_direct_gap_rows(rows: list[dict[str, Any]]) -> float:
    return sum(float_value(row.get("source_cue_official_repeat_miles")) for row in rows)


def road_miles_estimate(loop: dict[str, Any], direct_gap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mapped_mixed_road = 0.0
    for link in loop.get("link_rows") or []:
        if link.get("path_source") == "direct_gap_fallback":
            continue
        if "osm_public_road" in set(str(item) for item in link.get("connector_classes") or []):
            mapped_mixed_road += float_value(link.get("connector_miles"))
    source_road = 0.0
    for row in direct_gap_rows:
        if row.get("source_cue_type") == "connector_road":
            source_road += max(0.0, float_value(row.get("replacement_leg_miles")) - float_value(row.get("source_cue_official_repeat_miles")))
    return {
        "status": "upper_bound_from_mixed_connector_classes",
        "road_miles": round_miles(mapped_mixed_road + source_road),
        "note": "The template link rows preserve connector class sets, not per-edge class distances; this road-mile value is an upper-bound estimate for links that include osm_public_road.",
    }


def mileage_breakdown(loop: dict[str, Any], direct_gap_review: dict[str, Any]) -> dict[str, Any]:
    direct_rows = direct_gap_review.get("rows") or []
    original_direct_connector = connector_miles_for_direct_gap_rows(direct_rows)
    replacement_connector = replacement_connector_miles_for_direct_gap_rows(direct_rows)
    replacement_repeat = replacement_repeat_miles_for_direct_gap_rows(direct_rows)
    connector_miles = float_value(loop.get("connector_miles")) - original_direct_connector + replacement_connector
    official_repeat_miles = float_value(loop.get("official_repeat_miles")) + replacement_repeat
    priced_miles = float_value(
        direct_gap_review.get("post_source_cue_gpx_priced_track_miles")
        or direct_gap_review.get("post_named_cue_priced_track_miles")
        or loop.get("track_miles")
    )
    road = road_miles_estimate(loop, direct_rows)
    return {
        "official_new_miles": round_miles(float_value(loop.get("official_miles"))),
        "official_repeat_miles": round_miles(official_repeat_miles),
        "connector_miles": round_miles(connector_miles),
        "road_miles": road["road_miles"],
        "road_miles_status": road["status"],
        "road_miles_note": road["note"],
        "total_on_foot_miles": round_miles(priced_miles),
        "original_template_track_miles": round_miles(float_value(loop.get("track_miles"))),
        "candidate_rendered_gpx_miles": (direct_gap_review.get("candidate_rendered_gpx") or {}).get("gpx_miles"),
        "direct_gap_miles_replaced_for_pricing": round_miles(original_direct_connector),
        "source_cue_gpx_replacement_miles": round_miles(replacement_connector + replacement_repeat),
    }


def cue_text_for_link(link: dict[str, Any], direct_gap_rows: list[dict[str, Any]]) -> str:
    to_segment_id = str(link.get("to_segment_id"))
    gap_row = next((row for row in direct_gap_rows if row["to_segment_id"] == to_segment_id), None)
    if gap_row:
        if gap_row.get("source_cue_signed_as"):
            names = " / ".join(gap_row["source_cue_signed_as"])
            return f"HOLD: source cue follows {names} toward {gap_row.get('source_cue_target')} ({gap_row.get('replacement_leg_miles')} GPX mi), but the candidate GPX still uses direct_gap_fallback."
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
        "source_checked_on": "2026-05-13",
        "parking_access": {
            "status": "source_supported_operational_recheck_required",
            "start": "Simplot Lodge Parking Area" if is_b1 else "Pioneer Lodge Parking Area",
            "evidence": [
                {
                    "source": "Ridge to Rivers Bogus Basin Area",
                    "url": "https://www.ridgetorivers.org/trails/trail-areas/bogus-basin-area/",
                    "finding": "Bogus Basin Area page reports no specific conditions/closures for the area at check time and says parking is available at Nordic Lodge and Simplot Lodge.",
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
                "finding": "Bogus Basin Road has weekday time-window closures from May 11 through June 19, 2026, and Pat's Trail is closed midweek; weekend and federal-holiday access is not affected.",
            },
            "promotion_rule": "Do not schedule Bogus candidates on June 18 or June 19 during affected road/trail closure windows unless same-day sources show the route is open.",
        },
    }


def evaluate_bundle(
    *,
    bundle: dict[str, Any],
    field_tool_data: dict[str, Any],
    packet_dir: Path = DEFAULT_PACKET_DIR,
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
    direct_gap_review = direct_gap_repairs(
        bundle_id=bundle["bundle_id"],
        loop=loop,
        routes_by_label=routes_by_label,
        packet_dir=packet_dir,
    )
    repeat_review = repeat_and_ownership_review(bundle=bundle, loop=loop, owner_by_segment=owner_by_segment)
    cue_sheet = build_cue_sheet(loop, direct_gap_review)
    mileage = mileage_breakdown(loop, direct_gap_review)
    priced_track_miles = float_value(mileage.get("total_on_foot_miles"))
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
    post_gate_requirements = ["field_packet_recertification_not_run"]
    status = "blocked_keep_current_bogus" if hard_failures else "gate_repaired_candidate_still_needs_recertification"
    comparison = {
        "current_scope": bundle.get("current_total_scope"),
        "candidate_original": bundle.get("candidate_total_scope"),
        "candidate_after_source_cue_gpx_pricing": {
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
    comparison["candidate_after_named_gap_substitution"] = comparison["candidate_after_source_cue_gpx_pricing"]
    return {
        "bundle_id": bundle.get("bundle_id"),
        "shape": bundle.get("shape"),
        "status": status,
        "recommendation": "stop_first_pass_keep_current_bogus" if hard_failures else "continue_to_generated_route_card_trial",
        "hard_failures_after_first_pass": hard_failures,
        "post_gate_requirements": post_gate_requirements,
        "replaces_route_labels": bundle.get("replace_route_labels") or [],
        "generated_loop_id": loop.get("variant_id"),
        "direct_gap_review": direct_gap_review,
        "repeat_and_ownership_review": repeat_review,
        "cue_sheet": cue_sheet,
        "source_review": source_review_for_bundle(bundle),
        "mileage_breakdown": mileage,
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
        evaluate_bundle(
            bundle=bundle_by_id(template_candidates, bundle_id),
            field_tool_data=field_tool_data,
            packet_dir=args.packet_dir,
        )
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
            "packet_dir": display_path(args.packet_dir),
            "template_candidates_json": display_path(args.template_candidates_json),
        },
        "summary": {
            "candidate_count": len(rows),
            "promotion_candidate_count": len(promotion_candidates),
            "blocked_candidate_count": len(rows) - len(promotion_candidates),
            "b1_status": rows[0]["status"],
            "b2_status": rows[1]["status"],
            "recommendation": "keep_current_bogus_cards_after_first_pass"
            if not promotion_candidates
            else "continue_clean_candidate_to_route_card_trial_without_promotion",
            "active_packet_mutated": False,
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
        f"- Active packet mutated: `{str(report['summary']['active_packet_mutated']).lower()}`",
        "- B3 remains deferred until B1 and B2 are individually clean.",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Status | Direct Gap | Repeat / Ownership | Real GPX-Priced Cost | Mileage Breakdown | Recommendation |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in report.get("candidates") or []:
        direct = row["direct_gap_review"]
        repeat = row["repeat_and_ownership_review"]
        cost = row["cost_comparison"]["candidate_after_source_cue_gpx_pricing"]
        mileage = row["mileage_breakdown"]
        lines.append(
            "| "
            f"`{row['bundle_id']}` | "
            f"`{row['status']}` | "
            f"`{direct['status']}`; {direct['original_direct_gap_miles']} mi original, {direct['source_cue_gpx_leg_count']} source GPX cue legs priced | "
            f"`{repeat['status']}` | "
            f"{cost['track_miles']} mi / {cost['p75_minutes']} p75 / {cost['p90_minutes']} p90 | "
            f"{mileage['official_repeat_miles']} repeat / {mileage['connector_miles']} connector / {mileage['road_miles']} road-est. | "
            f"`{row['recommendation']}` |"
        )
    lines.extend(["", "## Gate Notes", ""])
    for row in report.get("candidates") or []:
        lines.append(f"### {row['bundle_id']}")
        lines.append("")
        lines.append(f"- Hard failures after first pass: {', '.join(row['hard_failures_after_first_pass']) or 'none'}")
        lines.append(f"- Post-gate requirements if ever promoted later: {', '.join(row['post_gate_requirements']) or 'none'}")
        lines.append(
            f"- Direct gaps: {row['direct_gap_review']['direct_gap_count']} original gaps, "
            f"{row['direct_gap_review']['source_cue_gpx_leg_count']} have source route-card GPX cue legs for pricing, but the candidate GPX is not rebuilt without direct_gap_fallback."
        )
        for gap_row in row["direct_gap_review"].get("rows") or []:
            lines.append(
                f"  - Gap to `{gap_row['to_segment_id']}`: {gap_row['original_direct_gap_miles']} mi direct fallback -> "
                f"{gap_row.get('replacement_leg_miles')} mi source GPX cue from `{gap_row.get('source_route_label')}` "
                f"(`{gap_row['repair_status']}`)."
            )
        lines.append(
            f"- Repeat/ownership: `{row['repeat_and_ownership_review']['status']}`; "
            f"declared owned elsewhere: {', '.join(row['repeat_and_ownership_review']['declared_owned_elsewhere_segment_ids']) or 'none'}."
        )
        mileage = row["mileage_breakdown"]
        lines.append(
            f"- Mileage: {mileage['total_on_foot_miles']} real GPX-priced on-foot mi; "
            f"{mileage['official_repeat_miles']} repeat mi; {mileage['connector_miles']} connector mi; "
            f"{mileage['road_miles']} road mi estimate (`{mileage['road_miles_status']}`)."
        )
        lines.append(f"- Cue sheet status: `{row['cue_sheet']['status']}` ({row['cue_sheet']['cue_count']} cues).")
        lines.append("- Around the Mountain/current signage: source confirms counter-clockwise all-users direction; keep day-of signage check as operational gate.")
        lines.append("- Closure/date gate: June 18/19 access remains operationally gated by Deer Point stewardship road/trail closure windows; this is not route truth.")
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "Stop Bogus promotion after this first pass. B1 and B2 can price source route-card GPX cue legs for the direct gaps, but neither has a rebuilt candidate GPX that removes direct_gap_fallback geometry. Current Bogus route cards should remain active until a later generator can build continuous candidate GPX from the named connectors and pass recertification.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
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
        run_id=RUN_ID,
        inputs=[args.field_tool_data_json, args.template_candidates_json, Path(__file__)],
        outputs=[args.output_json, args.output_md],
        command=(
            "python years/2026/scripts/bogus_b1_b2_gate_repair_audit.py "
            f"--output-json {display_path(args.output_json)} "
            f"--output-md {display_path(args.output_md)} "
            f"--manifest-json {display_path(args.manifest_json)}"
        ),
        metadata={"status": report["status"], "summary": report["summary"], "packet_dir": display_path(args.packet_dir)},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
