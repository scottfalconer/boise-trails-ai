from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import accepted_anchor_preservation_audit as audit  # noqa: E402


def test_missing_manifest_record_fails_for_user_reviewed_dominance():
    field_tool_payload = {
        "routes": [
            {
                "label": "FD09A",
                "trailhead": "Dry Creek",
                "segment_ids": ["1494", "1495"],
                "on_foot_miles": 3.96,
                "door_to_door_minutes_p75": 133,
            }
        ]
    }
    multi_start = {
        "alternatives": [
            {
                "alternative_id": "5-MS-04",
                "status": "promising",
                "components": [
                    {
                        "segment_ids": [1494, 1495],
                        "trail_names": ["Barn Owl"],
                        "on_foot_miles": 2.32,
                        "p75_total_minutes_if_standalone": 97,
                        "parking_confidence": "user_review_confirmed_paved_road_parking",
                        "start_anchor": {
                            "anchor_id": "street-probe-west-hidden-springs-drive",
                            "name": "West Hidden Springs Drive road-parking anchor",
                            "parking_confidence": "user_review_confirmed_paved_road_parking",
                            "source_type": "assumed_paved_road_parking",
                            "distance_to_component_miles": 0.03,
                        },
                    }
                ],
            }
        ]
    }

    result = audit.build_audit(
        field_tool_payload=field_tool_payload,
        multi_start_audit=multi_start,
        replacements_payload={"replacements": []},
    )

    assert result["summary"]["passed"] is False
    assert result["summary"]["active_replacement_missing_from_manifest_count"] == 1
    assert result["active_replacement_missing_from_manifest"][0]["current_route_label"] == "FD09A"


def test_active_manifest_record_requires_applied_route_matching_manifest_certification():
    field_tool_payload = {
        "routes": [
            {
                "label": "FD14D",
                "trailhead": "Full Sail Trailhead, N 36th St Parking",
                "segment_ids": ["1482"],
                "on_foot_miles": 1.5,
                "door_to_door_minutes_p75": 60,
                "door_to_door_minutes_p90": 68,
                "accepted_replacement_id": "fd14d-lower",
                "route_card_status": "provisional_re_anchored",
                "packet_visibility": "visible_with_provisional_badge",
                "certified_route_card": False,
                "anchor_to_credit_endpoint_distance_miles": 0.01,
            }
        ]
    }
    replacements = {
        "replacements": [
            {
                "replacement_id": "fd14d-lower",
                "status": "active",
                "target_segment_ids": ["1482"],
                "accepted_anchor_ref": "strava-parking-anchor-13",
                "public_anchor_label": "Full Sail Trailhead, N 36th St Parking",
                "baseline_card_ref": {
                    "on_foot_miles": 2.0,
                    "p75_minutes": 73,
                    "p90_minutes": 84,
                },
                "min_on_foot_savings": 0.4,
                "min_p75_savings": 12,
                "min_p90_savings": 10,
                "max_allowed_on_foot_regression": 0.05,
                "max_allowed_p75_regression": 5,
                "max_allowed_p90_regression": 8,
                "max_anchor_to_credit_endpoint_distance_miles": 0.25,
                "route_card_status": "provisional_re_anchored",
                "certified_route_card": False,
            }
        ]
    }

    result = audit.build_audit(
        field_tool_payload=field_tool_payload,
        multi_start_audit={"alternatives": []},
        replacements_payload=replacements,
    )

    assert result["summary"]["passed"] is True
    assert result["manifest_checks"][0]["failures"] == []

    field_tool_payload["routes"][0]["route_card_status"] = "certified_route_card"
    field_tool_payload["routes"][0]["packet_visibility"] = "published"
    field_tool_payload["routes"][0]["certified_route_card"] = True
    replacements["replacements"][0]["route_card_status"] = "certified_route_card"
    replacements["replacements"][0]["packet_visibility"] = "published"
    replacements["replacements"][0]["certified_route_card"] = True

    certified_result = audit.build_audit(
        field_tool_payload=field_tool_payload,
        multi_start_audit={"alternatives": []},
        replacements_payload=replacements,
    )

    assert certified_result["summary"]["passed"] is True
    assert certified_result["manifest_checks"][0]["failures"] == []


def test_investigate_record_fails_if_current_card_silently_stays_certified():
    field_tool_payload = {
        "routes": [
            {
                "label": "FD09A",
                "trailhead": "Dry Creek",
                "segment_ids": ["1494", "1495"],
                "on_foot_miles": 3.96,
                "door_to_door_minutes_p75": 133,
                "certified_route_card": True,
            }
        ]
    }
    replacements = {
        "replacements": [
            {
                "replacement_id": "fd09a-investigate",
                "status": "investigate",
                "target_segment_ids": ["1494", "1495"],
                "accepted_anchor_ref": "street-probe-west-hidden-springs-drive",
                "route_card_status": "investigation_required",
                "certified_route_card": False,
            }
        ]
    }

    result = audit.build_audit(
        field_tool_payload=field_tool_payload,
        multi_start_audit={"alternatives": []},
        replacements_payload=replacements,
    )

    assert result["summary"]["passed"] is False
    assert result["manifest_failures"][0]["failures"] == [
        "investigate_replacement_not_marked_on_route",
        "investigate_replacement_silently_certified",
    ]
