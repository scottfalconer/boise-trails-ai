from pathlib import Path
import copy
import json
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import build_route_review_pack as packer  # noqa: E402
import block_day_packager  # noqa: E402
import export_mobile_field_packet  # noqa: E402


def stale_fd14d_field_tool_payload():
    return {
        "routes": [
            {
                "label": "FD14D",
                "trailhead": "Full Sail",
                "segment_ids": ["1482"],
                "trails": ["36th Street Chute"],
                "official_miles": 0.74,
                "on_foot_miles": 2.0,
                "door_to_door_minutes_p75": 73,
                "door_to_door_minutes_p90": 84,
                "parking": {
                    "name": "Full Sail",
                    "lat": 43.66,
                    "lon": -116.22,
                    "parking_confidence": "inferred_from_trailhead_layer",
                    "source": "city_parks_facilities",
                },
                "navigation_quality": {
                    "start_access_gap_miles": 0.3,
                    "return_access_gap_miles": 0.3,
                },
            }
        ]
    }


def fd14d_multi_start_audit():
    return {
        "alternatives": [
            {
                "alternative_id": "1A-MS-04",
                "status": "promising",
                "components": [
                    {
                        "segment_ids": [1482],
                        "trail_names": ["36th Street Chute"],
                        "start_anchor": {
                            "anchor_id": "facility-strava-parking-anchor-13",
                            "name": "Private Strava parking anchor",
                            "lat": 43.6617525,
                            "lon": -116.2266716,
                            "raw_activity_ids": ["activity-raw-1"],
                            "parking_confidence": "strava_seen_prior_challenge_window",
                            "source": "strava_activity_endpoint_cluster",
                            "source_type": "private_strava_anchor",
                            "privacy": "private_exact_coordinates",
                            "field_ready": True,
                            "distance_to_component_miles": 0.01,
                        },
                        "official_miles": 0.74,
                        "on_foot_miles": 1.36,
                        "p75_total_minutes_if_standalone": 54,
                        "parking_confidence": "strava_seen_prior_challenge_window",
                        "parking_blockers": [],
                    }
                ],
            }
        ]
    }


def fd14d_replacements_payload():
    return {
        "replacements": [
            {
                "replacement_id": "fd14d-36th-street-chute-lower-36th",
                "status": "active",
                "target_segment_ids": ["1482"],
                "accepted_anchor_ref": "strava-parking-anchor-13",
                "public_anchor_label": "Full Sail Trailhead, N 36th St Parking",
                "start_justification": "Chosen because it is the accepted lower 36th parking/start anchor for exact segment 1482.",
                "baseline_card_ref": {
                    "label": "FD14D",
                    "trailhead": "Full Sail",
                    "on_foot_miles": 2.0,
                    "p75_minutes": 73,
                    "p90_minutes": 84,
                },
            }
        ]
    }


def test_fd14d_pack_flags_stale_full_sail_same_credit_dominance():
    report = packer.build_report(
        field_tool_payload=stale_fd14d_field_tool_payload(),
        multi_start_audit=fd14d_multi_start_audit(),
        replacements_payload=fd14d_replacements_payload(),
        route_labels={"FD14D"},
        generated_at="2026-05-15T00:00:00Z",
    )

    assert report["summary"]["single_segment_route_count"] == 1
    route = report["routes"][0]
    assert route["official_segment_ids"] == ["1482"]
    assert route["current_start"] == "Full Sail"

    lower_36th = next(
        alt
        for alt in route["same_credit_alternatives"]
        if alt["source"] == "multi_start_alternative_audit"
    )
    assert lower_36th["anchor_label"] == "Full Sail Trailhead, N 36th St Parking"
    assert lower_36th["on_foot_miles"] == pytest.approx(1.36)
    assert lower_36th["p75_minutes"] == 54
    assert lower_36th["deltas"]["dominance_delta_on_foot_miles"] == pytest.approx(0.64)
    assert lower_36th["deltas"]["dominance_delta_p75_minutes"] == 19

    review = report["reviews"][0]
    assert review["decision"] == "FAIL_DOMINATED"
    assert review["dominance_assessment"]["dominating_anchor"] == "Full Sail Trailhead, N 36th St Parking"
    assert review["dominance_assessment"]["estimated_miles_saved"] == pytest.approx(0.64)
    assert review["dominance_assessment"]["estimated_minutes_saved"] == 19


def test_public_safe_route_review_checkpoint_strips_private_coordinates_and_activity_ids():
    report = packer.build_report(
        field_tool_payload=stale_fd14d_field_tool_payload(),
        multi_start_audit=fd14d_multi_start_audit(),
        replacements_payload=fd14d_replacements_payload(),
        route_labels={"FD14D"},
        generated_at="2026-05-15T00:00:00Z",
    )

    safe = packer.public_safe(copy.deepcopy(report))
    dumped = json.dumps(safe, sort_keys=True)

    assert "activity-raw-1" not in dumped
    assert "43.6617525" not in dumped
    assert "-116.2266716" not in dumped
    assert "Full Sail Trailhead, N 36th St Parking" in dumped


def test_start_justification_flows_through_outing_and_field_tool_record():
    justification = "Chosen because this accepted anchor is closest for the exact credit target."
    outings = block_day_packager.build_outing_menu(
        {
            "progress": {"completed_segment_ids": []},
            "packages": [
                {
                    "package_number": 14,
                    "block_name": "Field Day 14 route-card bundle",
                    "components": [
                        {
                            "field_menu_label": "FD14D",
                            "trailhead": "Full Sail",
                            "official_miles": 0.74,
                            "on_foot_miles": 1.36,
                            "total_minutes": 54,
                            "candidate_id": "accepted-replacement-fd14d",
                            "segment_ids": ["1482"],
                            "trail_names": ["36th Street Chute"],
                            "accepted_replacement_id": "fd14d-lower",
                            "accepted_replacement_status": "active",
                            "accepted_anchor_label": "Full Sail Trailhead, N 36th St Parking",
                            "start_justification": justification,
                        }
                    ],
                }
            ],
        }
    )

    assert outings[0]["start_justification"] == justification

    field_record = export_mobile_field_packet.route_field_tool_record(
        {
            "outing": outings[0],
            "parking": {"name": "Full Sail Trailhead, N 36th St Parking"},
            "route_cues": [],
        }
    )
    assert field_record["start_justification"] == justification


def test_field_tool_record_generates_default_start_justification():
    field_record = export_mobile_field_packet.route_field_tool_record(
        {
            "outing": {
                "outing_id": "1-1",
                "label": "FD01A",
                "block_name": "Field Day 1",
                "trailhead": "Warm Springs Golf Course",
                "candidate_ids": ["table-rock"],
                "trails": ["Table Rock Trail"],
                "segment_ids": ["1"],
                "official_miles": 1.0,
                "on_foot_miles": 1.2,
                "total_minutes": 45,
            },
            "parking": {
                "name": "Warm Springs Golf Course",
                "parking_confidence": "inferred_from_trailhead_layer",
                "source": "city_parks_facilities",
            },
            "route_cues": [],
        }
    )

    assert "Warm Springs Golf Course" in field_record["start_justification"]
    assert "exact official segment set" in field_record["start_justification"]
