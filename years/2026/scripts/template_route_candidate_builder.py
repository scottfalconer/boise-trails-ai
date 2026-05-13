#!/usr/bin/env python3
"""Generate template-seeded cluster route candidates without promoting them."""

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
from export_execution_gpx import render_gpx_segments  # noqa: E402
from freestone_cluster_route_generation_experiment import (  # noqa: E402
    build_generated_route,
    current_route_metrics,
    display_path,
    float_value,
    normalized_ids,
    read_json,
    route_key,
    scaled_minutes,
    slugify,
    sort_id,
    write_json,
)
from freestone_military_candidate_bundle_experiment import (  # noqa: E402
    parking_for_trailhead,
    route_by_label,
    route_impact_rows,
    segment_owner_index,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    load_connector_graph,
    load_official_segments,
    round_miles,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_TEMPLATE_CANDIDATES_JSON = YEAR_DIR / "checkpoints" / "common-route-template-candidates-2026-05-12.json"
DEFAULT_CLUSTER_AUDIT_JSON = YEAR_DIR / "checkpoints" / "cluster-route-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints" / "template-route-candidates-2026-05-12"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "template-route-candidates-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "template-route-candidates-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "template-route-candidates-2026-05-12-manifest.json"

HARLOW_ALL = ["1626", "1657", "1661", "1662", "1687", "1688", "1689", "1696", "1704", "1705", "1706", "1707", "1708"]
HARLOW_AVIMOR_CORE = ["1661", "1662", "1687", "1688", "1689", "1696", "1626", "1657"]
HARLOW_DRY_LEFTOVERS = ["1708", "1704", "1705", "1706", "1707"]

BOGUS_SUNSHINE = ["1713"]
BOGUS_SIMPLOT_SIDE = BOGUS_SUNSHINE + ["1540", "1553", "1554", "1488", "1489", "1490", "1491", "1492", "1493", "1750"]
BOGUS_PIONEER_SIDE = ["1680", "1501", "1502", "1503", "1703", "1655", "1678", "1679", "1721", "1732", "1733", "1734", "1735", "1736"]

HULLS_COMPACT = ["1585", "1586", "1587", "1588", "1589", "1532", "1533", "1534", "1535", "1583", "1615", "1616"]
DRY_SHINGLE_SMOKE = ["1656", "1542", "1543", "1544", "1545", "1546", "1653", "1665", "1666", "1667"]


SOURCE_EVIDENCE = {
    "harlow-avimor-spring-valley-cluster": [
        {
            "label": "Avimor Trails and Outdoors",
            "url": "https://www.avimor.com/trails-and-outdoors",
            "source_scope": "public local trail-name and common-loop evidence",
            "notes": [
                "Publishes Spring Valley Creek, Twisted Spring, Whistling Pig, Ricochet, Shooting Range, Harlow Hollows, and Harlow Hollow Connector.",
                "Describes Spring Valley Creek as best looped with Burnt Car and Harlow Hollows.",
                "Marks Broken Horn Road and Fisher Lane as no longer accessible, so older Hidden Springs patterns stay access-gated.",
            ],
        },
        {
            "label": "Trailforks Spring Valley Creek parking POI",
            "url": "https://www.trailforks.com/poi/141399/",
            "source_scope": "public parking POI attached to Spring Valley Creek",
            "notes": [
                "Trailforks lists the POI as parking and includes hike/trail running activity types.",
                "This supports probing Avimor-native starts, but does not replace current field-packet parking confidence review.",
            ],
        },
        {
            "label": "MTB Project Hidden Springs to Spring Valley Creek Avimor",
            "url": "https://www.mtbproject.com/trail/3811378/hidden-springs-to-spring-valley-creek-avimor",
            "source_scope": "older public Hidden Springs-to-Avimor pattern",
            "notes": [
                "Useful evidence that Hidden Springs-to-Avimor has been a real-world pattern.",
                "Access is stale/conflicted because the route relies on Broken Horn while Avimor now says Broken Horn Road is no longer accessible.",
            ],
        },
    ],
    "bogus-atm-deer-point-simplot-pioneer-day-pair": [
        {
            "label": "Ridge to Rivers Bogus Basin Area",
            "url": "https://www.ridgetorivers.org/trails/trail-areas/bogus-basin-area/",
            "source_scope": "official local trail-area source",
            "notes": [
                "Confirms Bogus-area parking at Nordic Lodge and Simplot Lodge.",
                "Confirms Around the Mountain is directional for all users counter-clockwise.",
            ],
        },
        {
            "label": "Bogus Basin Deer Point Stewardship Project 2026",
            "url": "https://bogusbasin.org/about-bogus/culture/deer-point-stewardship-project-2026/",
            "source_scope": "current 2026 closure/access context",
            "notes": [
                "Confirms weekday Bogus Basin Road and Pat's Trail closure windows through June 19, 2026.",
                "Bogus candidates must remain date/access gated for June 18 and June 19 challenge-window starts.",
            ],
        },
    ],
    "hulls-kestrel-crestline-frontside-loop": [
        {
            "label": "Ridge to Rivers special management strategies",
            "url": "https://www.ridgetorivers.org/trail-news/ridge-to-rivers-adopts-management-strategies-from-pilot-trail-program/",
            "source_scope": "official Lower Hulls legality rule",
            "notes": [
                "Lower Hulls is closed to hikers/runners on odd-numbered days and open to hikers both directions on even-numbered days.",
                "The compact Hulls probe must be date-gated before field use.",
            ],
        },
        {
            "label": "Outdoor Project Hulls Gulch Red Cliffs Kestrel Loop",
            "url": "https://www.outdoorproject.com/united-states/idaho/hulls-gulch-red-cliffs-kestrel-loop",
            "source_scope": "public common-route pattern",
            "notes": [
                "Public route source for the Red Cliffs / Crestline / Kestrel relationship.",
                "Used only as a candidate generator, not as official challenge-credit truth.",
            ],
        },
    ],
    "dry-creek-shingle-loop": [
        {
            "label": "Existing route-card and prior promotion evidence",
            "url": None,
            "source_scope": "repo-local low-priority smoke-test source",
            "notes": [
                "The Shingle/Sweet Connie duplication has already been reduced elsewhere.",
                "This builder keeps only a cheap smoke test so this cluster does not consume local-agent time prematurely.",
            ],
        }
    ],
}


CANDIDATE_BUNDLES = [
    {
        "bundle_id": "H1-avimor-native-harlow-spring-loop",
        "priority": 4,
        "template_id": "harlow-avimor-spring-valley-cluster",
        "shape": "H1",
        "intent": "Avimor-native Harlow/Spring loop covering Spring Creek, Twisted Spring, Whistling Pig, Ricochet/Shooting, and Harlow pieces.",
        "replace_route_labels": ["FD27A", "FD30A", "FD27B", "FD27C", "FD24A"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 5.0,
        "source_status": "public_sources_captured_access_confidence_still_needs_review",
        "extra_hard_failures": ["avimor_parking_confidence_missing_in_field_packet", "needs_public_safe_cueable_access_review"],
        "loops": [
            {
                "loop_id": "h1-avimor-native-harlow-spring-loop",
                "trailhead": "Avimor Spring Valley Creek parking",
                "strategy": "nearest_segment_greedy",
                "segment_ids": HARLOW_ALL,
            }
        ],
    },
    {
        "bundle_id": "H2-split-avimor-dry-creek",
        "priority": 4,
        "template_id": "harlow-avimor-spring-valley-cluster",
        "shape": "H2",
        "intent": "Keep Spring/Twisted/Whistling from Avimor and true Dry Creek/Harlow leftovers from Dry Creek if connector tax is lower.",
        "replace_route_labels": ["FD27A", "FD30A", "FD27B", "FD27C", "FD24A"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 5.0,
        "source_status": "public_sources_captured_split_start_requires_two_parking_reviews",
        "extra_hard_failures": ["avimor_parking_confidence_missing_in_field_packet", "dry_creek_connector_pattern_needs_human_cues"],
        "loops": [
            {
                "loop_id": "h2-avimor-spring-twisted-whistling-ricochet",
                "trailhead": "Avimor Spring Valley Creek parking",
                "strategy": "nearest_segment_greedy",
                "segment_ids": HARLOW_AVIMOR_CORE,
            },
            {
                "loop_id": "h2-dry-creek-harlow-leftovers",
                "trailhead": "Dry Creek Parking Area/Trailhead",
                "strategy": "nearest_segment_greedy",
                "segment_ids": HARLOW_DRY_LEFTOVERS,
            },
        ],
    },
    {
        "bundle_id": "H3-harlow-west-access-probe",
        "priority": 4,
        "template_id": "harlow-avimor-spring-valley-cluster",
        "shape": "H3",
        "intent": "Only probe Harlow west / Hidden Springs access if public, legal, cue-able access exists.",
        "replace_route_labels": ["FD27A", "FD30A", "FD27B", "FD27C", "FD24A"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 5.0,
        "source_status": "not_generated_access_evidence_conflicted",
        "extra_hard_failures": ["no_verified_harlow_west_anchor", "hidden_springs_pattern_uses_stale_or_conflicted_broken_horn_access"],
        "loops": [],
    },
    {
        "bundle_id": "B1-simplot-side-bogus-day",
        "priority": 5,
        "template_id": "bogus-atm-deer-point-simplot-pioneer-day-pair",
        "shape": "B1",
        "intent": "Simplot-side Bogus day with Sunshine / Deer Point / Elk Meadows / Around the Mountain and Lodge return context.",
        "replace_route_labels": ["FD07A", "FD07B", "FD25A", "FD26A"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 0.0,
        "source_status": "public_sources_captured_current_closure_and_direction_checks_required",
        "extra_hard_failures": ["bogus_june_18_19_closure_window_check_required", "around_the_mountain_current_signage_check_required"],
        "loops": [
            {
                "loop_id": "b1-simplot-sunshine-deer-point-atm",
                "trailhead": "Simplot Lodge Parking Area",
                "strategy": "nearest_segment_greedy",
                "segment_ids": BOGUS_SIMPLOT_SIDE,
            }
        ],
    },
    {
        "bundle_id": "B2-pioneer-mores-side-day",
        "priority": 5,
        "template_id": "bogus-atm-deer-point-simplot-pioneer-day-pair",
        "shape": "B2",
        "intent": "Pioneer/Mores-side day preserving the mountain work as a second responsible Bogus outing.",
        "replace_route_labels": ["18", "FD25B"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 0.0,
        "source_status": "public_sources_captured_needs_pioneer_access_condition_review",
        "extra_hard_failures": ["bogus_closure_weather_and_condition_check_required"],
        "loops": [
            {
                "loop_id": "b2-pioneer-face-brewers-shindig-tempest-mores",
                "trailhead": "Pioneer Lodge Parking Area",
                "strategy": "nearest_segment_greedy",
                "segment_ids": BOGUS_PIONEER_SIDE,
            }
        ],
    },
    {
        "bundle_id": "B3-same-day-simplot-pioneer-transfer",
        "priority": 5,
        "template_id": "bogus-atm-deer-point-simplot-pioneer-day-pair",
        "shape": "B3",
        "intent": "Same-day Simplot-to-Pioneer transfer, only if p90 and between-start drive remain responsible.",
        "replace_route_labels": ["FD07A", "FD07B", "FD25A", "FD26A", "18", "FD25B"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 0.0,
        "source_status": "public_sources_captured_transfer_profile_missing",
        "extra_hard_failures": ["same_day_transfer_drive_time_missing", "bogus_june_18_19_closure_window_check_required"],
        "loops": [
            {
                "loop_id": "b3-simplot-sunshine-deer-point-atm",
                "trailhead": "Simplot Lodge Parking Area",
                "strategy": "nearest_segment_greedy",
                "segment_ids": BOGUS_SIMPLOT_SIDE,
            },
            {
                "loop_id": "b3-pioneer-face-brewers-shindig-tempest-mores",
                "trailhead": "Pioneer Lodge Parking Area",
                "strategy": "nearest_segment_greedy",
                "segment_ids": BOGUS_PIONEER_SIDE,
            },
        ],
    },
    {
        "bundle_id": "C1-hulls-kestrel-crestline-compact",
        "priority": 6,
        "template_id": "hulls-kestrel-crestline-frontside-loop",
        "shape": "C1",
        "intent": "Compact Hulls / Kestrel / Crestline route from Hulls Gulch without pulling Grove/Owl/Gold Finch into the bundle.",
        "replace_route_labels": ["FD22B", "FD19A", "FD19B"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 0.0,
        "source_status": "public_sources_captured_needs_lower_hulls_date_gate",
        "extra_hard_failures": ["lower_hulls_even_day_legality_check_required"],
        "loops": [
            {
                "loop_id": "c1-hulls-red-cliffs-crestline-kestrel",
                "trailhead": "Hulls Gulch",
                "strategy": "template_sequence_greedy",
                "segment_ids": HULLS_COMPACT,
            }
        ],
    },
    {
        "bundle_id": "D1-dry-creek-shingle-deferred-smoke-test",
        "priority": 7,
        "template_id": "dry-creek-shingle-loop",
        "shape": "D1",
        "intent": "Low-effort smoke test only; do not spend manual time unless current Dry/Shingle execution proves awkward or misses credit.",
        "replace_route_labels": ["15A-1", "16A-1", "16A-2"],
        "preserve_route_labels": [],
        "material_savings_threshold_miles": 0.0,
        "source_status": "repo_local_low_pressure_smoke_test",
        "extra_hard_failures": ["low_priority_deferred_unless_field_feedback_changes_pressure"],
        "loops": [
            {
                "loop_id": "d1-dry-shingle-sweet-connie-smoke-test",
                "trailhead": "Dry Creek / Sweet Connie roadside parking",
                "strategy": "nearest_segment_greedy",
                "segment_ids": DRY_SHINGLE_SMOKE,
            }
        ],
    },
]


def route_metrics_for_labels(routes_by_label: dict[str, dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    return current_route_metrics([routes_by_label[label] for label in labels if label in routes_by_label])


def missing_route_labels(routes_by_label: dict[str, dict[str, Any]], labels: list[str]) -> list[str]:
    return [label for label in labels if label not in routes_by_label]


def safe_route_impact_rows(
    routes_by_label: dict[str, dict[str, Any]],
    labels: list[str],
    generated_claim_ids: set[str],
) -> list[dict[str, Any]]:
    rows = route_impact_rows(
        routes_by_label,
        [label for label in labels if label in routes_by_label],
        generated_claim_ids,
    )
    for label in labels:
        if label not in routes_by_label:
            rows.append(
                {
                    "label": label,
                    "outing_id": None,
                    "status": "route_absent_from_active_packet",
                    "covered_segment_ids": [],
                    "remaining_segment_ids": [],
                }
            )
    return rows


def ordered_unique_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for value in values:
        segment_id = str(value)
        if segment_id in seen:
            continue
        seen.add(segment_id)
        ordered.append(segment_id)
    return ordered


def union_loop_ids(loops: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({segment_id for loop in loops for segment_id in normalized_ids(loop.get(key) or [])}, key=sort_id)


def generated_official_touch_ids(loops: list[dict[str, Any]]) -> list[str]:
    touched = set()
    for loop in loops:
        touched.update(normalized_ids(loop.get("official_segment_ids") or []))
        touched.update(normalized_ids(loop.get("self_repeat_segment_ids") or []))
        touched.update(normalized_ids(loop.get("non_template_repeat_segment_ids") or []))
    return sorted(touched, key=sort_id)


def latent_credit_rows(
    *,
    loops: list[dict[str, Any]],
    replaced_ids: set[str],
    preserved_labels: set[str],
    owner_by_segment: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for segment_id in generated_official_touch_ids(loops):
        if segment_id in replaced_ids:
            continue
        owners = owner_by_segment.get(segment_id, [])
        if set(owners) & preserved_labels:
            status = "owned_by_preserved_route"
        elif owners:
            status = "owned_by_other_active_route_needs_ownership_decision"
        else:
            status = "unowned_latent_credit"
        rows.append({"segment_id": segment_id, "owners": owners, "status": status})
    return rows


def render_loop_gpx(output_dir: Path, bundle_id: str, loop: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slugify(bundle_id)}--{slugify(loop['variant_id'])}.gpx"
    path.write_text(
        render_gpx_segments(
            f"Template route candidate {bundle_id}: {loop['variant_id']}",
            [loop["track_coordinates"]],
        ),
        encoding="utf-8",
    )
    return path


def summarize_parking(loop: dict[str, Any]) -> dict[str, Any]:
    parking = dict(loop.get("parking") or {})
    return {
        "name": parking.get("name"),
        "has_parking": parking.get("has_parking"),
        "parking_confidence": parking.get("parking_confidence"),
        "source": parking.get("source"),
        "nearest_open_trail_name": parking.get("nearest_open_trail_name"),
    }


def build_generated_loops(
    definition: dict[str, Any],
    *,
    field_tool_data: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    connector_graph: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    loops = []
    gpx_paths = []
    for loop_def in definition.get("loops") or []:
        loop = build_generated_route(
            variant_id=loop_def["loop_id"],
            strategy=loop_def["strategy"],
            ordered_segment_ids=ordered_unique_ids(loop_def["segment_ids"]),
            parking=parking_for_trailhead(field_tool_data, loop_def["trailhead"]),
            official_by_id=official_by_id,
            connector_graph=connector_graph,
            preserve_ascent_direction=True,
        )
        loop["trailhead"] = loop_def["trailhead"]
        loop["intent_segment_ids"] = ordered_unique_ids(loop_def["segment_ids"])
        loop["parking_summary"] = summarize_parking(loop)
        gpx_path = render_loop_gpx(output_dir, definition["bundle_id"], loop)
        gpx_paths.append(display_path(gpx_path))
        loop["gpx_path"] = display_path(gpx_path)
        loop.pop("track_coordinates", None)
        loops.append(loop)
    return loops, gpx_paths


def promotion_gates(
    *,
    definition: dict[str, Any],
    loops: list[dict[str, Any]],
    current_scope: dict[str, Any],
    candidate_scope: dict[str, Any] | None,
    latent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not loops:
        hard_failures = list(definition.get("extra_hard_failures") or [])
        hard_failures.extend(["route_geometry_missing", "candidate_not_generated"])
        return {
            "status": "blocked_by_promotion_gates",
            "hard_failures": hard_failures,
            "continuous_gpx": "not_generated",
            "hidden_self_repeat": "not_evaluated",
            "latent_credit": "not_evaluated",
            "ascent_direction": "not_evaluated",
            "p75_p90_better_than_current_scope": "not_evaluated",
            "cue_sheet": "not_generated",
            "recertification": "not_run_against_active_field_packet",
        }

    ascent_failed = union_loop_ids(
        [{"ids": (loop.get("ascent_direction_validation") or {}).get("direction_failed_segment_ids") or []} for loop in loops],
        "ids",
    )
    hidden_self_repeat = union_loop_ids(loops, "self_repeat_segment_ids")
    direct_gap_miles = round_miles(sum(float_value(loop.get("direct_gap_fallback_miles")) for loop in loops))
    has_latent_blocker = any(row["status"] != "owned_by_preserved_route" for row in latent_rows)
    p75_p90_better = bool(
        candidate_scope
        and candidate_scope["on_foot_miles"] < current_scope["on_foot_miles"]
        and candidate_scope["p75_minutes"] < current_scope["p75_minutes"]
        and candidate_scope["p90_minutes"] < current_scope["p90_minutes"]
    )

    hard_failures = list(definition.get("extra_hard_failures") or [])
    if ascent_failed:
        hard_failures.append("ascent_direction_failure")
    if hidden_self_repeat:
        hard_failures.append("hidden_self_repeat")
    if has_latent_blocker:
        hard_failures.append("latent_credit_needs_ownership_decision")
    if not p75_p90_better:
        hard_failures.append("not_better_on_on_foot_p75_p90")
    if direct_gap_miles:
        hard_failures.append("direct_gap_fallback")
    hard_failures.append("needs_human_cue_sheet")
    hard_failures.append("needs_field_packet_recertification")
    return {
        "status": "blocked_by_promotion_gates" if hard_failures else "promotion_candidate",
        "hard_failures": hard_failures,
        "ascent_direction": "passed" if not ascent_failed else "failed",
        "ascent_direction_failed_segment_ids": ascent_failed,
        "hidden_self_repeat": "failed" if hidden_self_repeat else "passed",
        "hidden_self_repeat_segment_ids": hidden_self_repeat,
        "latent_credit": "needs_ownership_decision" if has_latent_blocker else "passed",
        "p75_p90_better_than_current_scope": "passed" if p75_p90_better else "failed",
        "continuous_gpx": "needs_direct_gap_review" if direct_gap_miles else "passed_graph_continuity",
        "direct_gap_fallback_miles": direct_gap_miles,
        "cue_sheet": "needs_human_cue_rewrite",
        "recertification": "not_run_against_active_field_packet",
    }


def recommendation_for_bundle(bundle: dict[str, Any]) -> str:
    if not bundle.get("generated_loops"):
        return "do_not_promote_access_source_missing"
    delta = float_value((bundle.get("delta_vs_current_total_scope") or {}).get("on_foot_miles"))
    threshold = float_value(bundle.get("material_savings_threshold_miles"))
    if delta >= 0:
        return "do_not_promote_current_cards_are_cheaper"
    if abs(delta) >= threshold:
        return "promising_candidate_needs_hard_gate_repair"
    return "minor_candidate_keep_as_backlog"


def template_by_id(template_candidates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("template_id")): row for row in template_candidates.get("templates_ranked") or []}


def cluster_bundle_by_template(cluster_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("template_id")): row for row in cluster_audit.get("cluster_bundle_replacements") or []}


def build_bundle(
    definition: dict[str, Any],
    *,
    field_tool_data: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    connector_graph: dict[str, Any],
    template_rows: dict[str, dict[str, Any]],
    cluster_rows: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    routes_by_label = route_by_label(field_tool_data)
    owner_by_segment = segment_owner_index(field_tool_data)
    replace_labels = list(definition.get("replace_route_labels") or [])
    preserve_labels = list(definition.get("preserve_route_labels") or [])
    missing_replace_labels = missing_route_labels(routes_by_label, replace_labels)
    missing_preserve_labels = missing_route_labels(routes_by_label, preserve_labels)
    replaced_current = route_metrics_for_labels(routes_by_label, replace_labels)
    preserved_current = route_metrics_for_labels(routes_by_label, preserve_labels)
    current_scope = {
        "on_foot_miles": round_miles(replaced_current["on_foot_miles"] + preserved_current["on_foot_miles"]),
        "official_miles": round_miles(replaced_current["official_miles"] + preserved_current["official_miles"]),
        "p75_minutes": replaced_current["p75_minutes"] + preserved_current["p75_minutes"],
        "p90_minutes": replaced_current["p90_minutes"] + preserved_current["p90_minutes"],
    }
    if missing_replace_labels:
        bundle = {
            "bundle_id": definition["bundle_id"],
            "priority": definition["priority"],
            "template_id": definition["template_id"],
            "shape": definition["shape"],
            "intent": definition["intent"],
            "source_status": "superseded_by_active_packet_change",
            "source_evidence": SOURCE_EVIDENCE.get(definition["template_id"], []),
            "template_pressure": (template_rows.get(definition["template_id"]) or {}).get("current_route_pressure"),
            "cluster_audit_row": cluster_rows.get(definition["template_id"]) or {},
            "replace_route_labels": replace_labels,
            "preserve_route_labels": preserve_labels,
            "missing_replace_route_labels": missing_replace_labels,
            "missing_preserve_route_labels": missing_preserve_labels,
            "current_replaced_scope": replaced_current,
            "current_preserved_scope": preserved_current,
            "current_total_scope": current_scope,
            "candidate_generated_scope": {
                "loop_count": 0,
                "on_foot_miles": 0.0,
                "p75_minutes_scaled": None,
                "p90_minutes_scaled": None,
                "pricing_status": "not_generated_replaced_routes_absent",
            },
            "candidate_total_scope": None,
            "delta_vs_current_total_scope": None,
            "material_savings_threshold_miles": definition.get("material_savings_threshold_miles"),
            "material_savings_threshold_met": False,
            "route_impacts": {
                "replaced_routes": safe_route_impact_rows(routes_by_label, replace_labels, set()),
                "preserved_routes": safe_route_impact_rows(routes_by_label, preserve_labels, set()),
            },
            "promotion_gates": {
                "status": "superseded_by_active_packet_change",
                "hard_failures": ["replaced_route_labels_absent_from_active_packet"],
                "missing_replace_route_labels": missing_replace_labels,
                "recertification": "not_needed_for_superseded_template",
            },
            "latent_credit_review": [],
            "generated_loops": [],
            "gpx_paths": [],
            "recommendation": "exclude_from_current_queue_superseded_by_h1",
        }
        return bundle
    loops, gpx_paths = build_generated_loops(
        definition,
        field_tool_data=field_tool_data,
        official_by_id=official_by_id,
        connector_graph=connector_graph,
        output_dir=output_dir,
    )
    generated_miles = round_miles(sum(float_value(loop.get("track_miles")) for loop in loops))
    generated_p75 = scaled_minutes(generated_miles, replaced_current["on_foot_miles"], replaced_current["p75_minutes"])
    generated_p90 = scaled_minutes(generated_miles, replaced_current["on_foot_miles"], replaced_current["p90_minutes"])
    candidate_scope = None
    delta = None
    if loops:
        candidate_scope = {
            "on_foot_miles": round_miles(generated_miles + preserved_current["on_foot_miles"]),
            "p75_minutes": generated_p75 + preserved_current["p75_minutes"],
            "p90_minutes": generated_p90 + preserved_current["p90_minutes"],
            "pricing_status": "scaled_from_replaced_current_cards_needs_dem_and_field_calibration",
        }
        delta = {
            "on_foot_miles": round_miles(candidate_scope["on_foot_miles"] - current_scope["on_foot_miles"]),
            "p75_minutes": candidate_scope["p75_minutes"] - current_scope["p75_minutes"],
            "p90_minutes": candidate_scope["p90_minutes"] - current_scope["p90_minutes"],
        }
    replaced_ids = set(replaced_current["segment_ids"])
    generated_claim_ids = set(union_loop_ids(loops, "official_segment_ids"))
    latent_rows = latent_credit_rows(
        loops=loops,
        replaced_ids=replaced_ids,
        preserved_labels=set(preserve_labels),
        owner_by_segment=owner_by_segment,
    )
    gates = promotion_gates(
        definition=definition,
        loops=loops,
        current_scope=current_scope,
        candidate_scope=candidate_scope,
        latent_rows=latent_rows,
    )
    bundle = {
        "bundle_id": definition["bundle_id"],
        "priority": definition["priority"],
        "template_id": definition["template_id"],
        "shape": definition["shape"],
        "intent": definition["intent"],
        "source_status": definition["source_status"],
        "source_evidence": SOURCE_EVIDENCE.get(definition["template_id"], []),
        "template_pressure": (template_rows.get(definition["template_id"]) or {}).get("current_route_pressure"),
        "cluster_audit_row": cluster_rows.get(definition["template_id"]) or {},
        "replace_route_labels": replace_labels,
        "preserve_route_labels": preserve_labels,
        "missing_replace_route_labels": missing_replace_labels,
        "missing_preserve_route_labels": missing_preserve_labels,
        "current_replaced_scope": replaced_current,
        "current_preserved_scope": preserved_current,
        "current_total_scope": current_scope,
        "candidate_generated_scope": {
            "loop_count": len(loops),
            "on_foot_miles": generated_miles,
            "p75_minutes_scaled": generated_p75 if loops else None,
            "p90_minutes_scaled": generated_p90 if loops else None,
            "pricing_status": "scaled_lower_bound_not_dem_certified" if loops else "not_generated",
        },
        "candidate_total_scope": candidate_scope,
        "delta_vs_current_total_scope": delta,
        "material_savings_threshold_miles": definition.get("material_savings_threshold_miles"),
        "material_savings_threshold_met": bool(delta and abs(float_value(delta["on_foot_miles"])) >= float_value(definition.get("material_savings_threshold_miles")) and float_value(delta["on_foot_miles"]) < 0),
        "route_impacts": {
            "replaced_routes": safe_route_impact_rows(routes_by_label, replace_labels, generated_claim_ids),
            "preserved_routes": safe_route_impact_rows(routes_by_label, preserve_labels, generated_claim_ids),
        },
        "promotion_gates": gates,
        "latent_credit_review": latent_rows,
        "generated_loops": loops,
        "gpx_paths": gpx_paths,
    }
    bundle["recommendation"] = recommendation_for_bundle(bundle)
    return bundle


def build_report(
    field_tool_data: dict[str, Any],
    template_candidates: dict[str, Any],
    cluster_audit: dict[str, Any],
    official_segments: list[dict[str, Any]],
    connector_graph: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    template_rows = template_by_id(template_candidates)
    cluster_rows = cluster_bundle_by_template(cluster_audit)
    bundles = [
        build_bundle(
            definition,
            field_tool_data=field_tool_data,
            official_by_id=official_by_id,
            connector_graph=connector_graph,
            template_rows=template_rows,
            cluster_rows=cluster_rows,
            output_dir=output_dir,
        )
        for definition in CANDIDATE_BUNDLES
    ]
    promising = [
        bundle
        for bundle in bundles
        if bundle["recommendation"] == "promising_candidate_needs_hard_gate_repair"
    ]
    superseded = [
        bundle
        for bundle in bundles
        if bundle["recommendation"] == "exclude_from_current_queue_superseded_by_h1"
    ]
    return {
        "schema": "boise_trails_template_route_candidate_builder_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "candidate_templates_generated_no_promotions",
        "source_files": {
            "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
            "template_candidates_json": display_path(DEFAULT_TEMPLATE_CANDIDATES_JSON),
            "cluster_route_optimization_audit_json": display_path(DEFAULT_CLUSTER_AUDIT_JSON),
            "official_segments_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "connector_geojson": display_path(DEFAULT_CONNECTOR_GEOJSON),
        },
        "scope": {
            "purpose": "Turn common-route templates into advisory route probes for Harlow/Avimor, Bogus, Hulls, and Dry/Shingle without route-card promotion.",
            "promotion_policy": "No generated candidate is active until continuous GPX, endpoint coverage, ascent direction, p75/p90, access, cue sheet, ownership, and field-packet recertification pass.",
            "requested_priorities": [4, 5, 6, 7],
        },
        "summary": {
            "bundle_count": len(bundles),
            "generated_bundle_count": sum(1 for bundle in bundles if bundle.get("generated_loops")),
            "superseded_bundle_count": len(superseded),
            "superseded_bundle_ids": [bundle["bundle_id"] for bundle in superseded],
            "promising_candidate_count": len(promising),
            "promising_bundle_ids": [bundle["bundle_id"] for bundle in promising],
            "no_promotion_count": len(bundles),
            "recommendation": "exclude_harlow_avimor_superseded_by_h1_then_use_remaining_promising_candidates_as_manual_route_geometry_queue",
        },
        "bundles": bundles,
    }


def bundle_delta_text(bundle: dict[str, Any]) -> str:
    delta = bundle.get("delta_vs_current_total_scope")
    if not delta:
        return "not generated"
    return f"{delta['on_foot_miles']} mi / {delta['p75_minutes']} p75"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Template Route Candidate Builder",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Bundles tested: {report['summary']['bundle_count']}",
        f"- Generated bundles: {report['summary']['generated_bundle_count']}",
        f"- Superseded bundles excluded from current queue: {report['summary']['superseded_bundle_count']}",
        f"- Superseded IDs: {', '.join(report['summary']['superseded_bundle_ids']) or 'none'}",
        f"- Promising candidates: {report['summary']['promising_candidate_count']}",
        f"- Promising IDs: {', '.join(report['summary']['promising_bundle_ids']) or 'none'}",
        "",
        "These are candidate-generation outputs only. They are not route-card promotions and they do not delete current cards.",
        "",
        "## Candidate Results",
        "",
        "| Bundle | Shape | Current scope | Candidate scope | Delta | Source status | Gates | Recommendation |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for bundle in report.get("bundles") or []:
        current = bundle["current_total_scope"]
        candidate = bundle.get("candidate_total_scope") or {}
        gates = bundle.get("promotion_gates") or {}
        candidate_text = (
            f"{candidate.get('on_foot_miles')} mi / {candidate.get('p75_minutes')} p75"
            if candidate
            else "not generated"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{bundle['bundle_id']}`",
                    bundle["shape"],
                    f"{current['on_foot_miles']} mi / {current['p75_minutes']} p75",
                    candidate_text,
                    bundle_delta_text(bundle),
                    f"`{bundle['source_status']}`",
                    ", ".join(gates.get("hard_failures") or []),
                    f"`{bundle['recommendation']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Harlow/Avimor H1/H2/H3 template probes are excluded from the current optimization queue when their old replaced route labels are absent from the active packet after H1 promotion.",
            "- Bogus works better as a day-pair/same-day-transfer investigation than a single mega-day. B3 has the largest modeled delta, but transfer time and closure/date gates are unresolved.",
            "- Hulls produces a clean small candidate, not a high-leverage one. It remains Lower-Hulls date-gated.",
            "- Dry/Shingle is worse than the current cards in this smoke test, so no local-agent time should move there unless field feedback changes the pressure.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--template-candidates-json", type=Path, default=DEFAULT_TEMPLATE_CANDIDATES_JSON)
    parser.add_argument("--cluster-audit-json", type=Path, default=DEFAULT_CLUSTER_AUDIT_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_data = read_json(args.field_tool_data_json)
    template_candidates = read_json(args.template_candidates_json)
    cluster_audit = read_json(args.cluster_audit_json)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    report = build_report(field_tool_data, template_candidates, cluster_audit, official_segments, connector_graph, args.output_dir)
    report["source_files"] = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "template_candidates_json": display_path(args.template_candidates_json),
        "cluster_route_optimization_audit_json": display_path(args.cluster_audit_json),
        "official_segments_geojson": display_path(args.official_geojson),
        "connector_geojson": display_path(args.connector_geojson),
    }
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    output_paths = [args.output_json, args.output_md] + [
        Path(path) for bundle in report["bundles"] for path in bundle.get("gpx_paths") or []
    ]
    manifest = build_artifact_manifest(
        run_id="template_route_candidate_builder",
        command="python years/2026/scripts/template_route_candidate_builder.py",
        inputs=[args.field_tool_data_json, args.template_candidates_json, args.cluster_audit_json, args.official_geojson, args.connector_geojson],
        outputs=output_paths,
        metadata={"status": report.get("status"), "summary": report.get("summary")},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    for path in [path for bundle in report["bundles"] for path in bundle.get("gpx_paths") or []]:
        print(f"Wrote {path}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
