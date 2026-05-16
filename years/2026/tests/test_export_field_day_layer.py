import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_field_day_layer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_field_day_layer", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_promotion_payload_preserves_h1_source_promotion():
    module = load_module()

    assert module.DEFAULT_PROMOTION_JSON.name == "harlow-h1-route-card-promotion-2026-05-12.json"


def test_build_field_day_layer_links_certified_route_cards_and_flags_gaps():
    module = load_module()
    assignments_payload = {
        "status": "passed",
        "audit": {
            "passed": True,
            "assigned_day_count": 1,
            "covered_segment_count": 3,
            "official_segment_count": 3,
            "missing_segment_count": 0,
        },
        "assignments": [
            {
                "date": "2026-06-18",
                "weekday_name": "Thursday",
                "day_type": "weekday",
                "constraints": ["lower_hulls_even_day_on_foot"],
                "field_day": {
                    "draft_day_number": 1,
                    "field_day_id": "weekday-a",
                    "p75_minutes": 180,
                    "p90_minutes": 202,
                    "p90_bound_minutes": 292,
                    "between_drive_minutes": 5,
                    "loop_count": 2,
                    "on_foot_miles": 6.2,
                    "segment_summary": {
                        "segment_count": 3,
                        "official_miles": 4.8,
                        "segment_ids": [101, 102, 103],
                    },
                    "loops": [
                        {
                            "loop_id": "canonical_field_menu::certified-card::Trailhead",
                            "label": "certified-card",
                            "source": "canonical_field_menu",
                            "candidate_id": "cert-1",
                            "trailhead": "Trailhead",
                            "trail_names": ["Certified Trail"],
                            "segment_count": 2,
                            "official_miles": 3.0,
                            "on_foot_miles": 3.8,
                            "p75_minutes": 90,
                            "p90_minutes": 101,
                            "validation_passed": True,
                        },
                        {
                            "loop_id": "personal_route_menu::uncertified-loop::Other Trailhead",
                            "label": "uncertified-loop",
                            "source": "personal_route_menu",
                            "candidate_id": "uncertified-loop",
                            "trailhead": "Other Trailhead",
                            "trail_names": ["Uncertified Trail"],
                            "segment_count": 1,
                            "official_miles": 1.8,
                            "on_foot_miles": 2.4,
                            "p75_minutes": 90,
                            "p90_minutes": 101,
                            "validation_passed": True,
                        },
                    ],
                },
            }
        ],
    }
    field_tool_payload = {
        "summary": {"runnable_outing_count": 1},
        "certified_baseline": {"status": "passed"},
        "routes": [
            {
                "outing_id": "1-1",
                "label": "1A",
                "candidate_ids": ["cert-1"],
                "trailhead": "Trailhead",
                "parking": {"name": "Trailhead", "lat": 43.1, "lon": -116.1, "has_parking": True},
                "segment_ids": ["101", "102"],
                "trails": ["Certified Trail"],
                "official_miles": 3.0,
                "on_foot_miles": 3.8,
                "door_to_door_minutes_p75": 90,
                "door_to_door_minutes_p90": 101,
                "gpx_href": "gpx/official/certified-card.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 3.8}],
                "validation": {"passed": True},
            }
        ],
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload)

    assert layer["summary"]["field_day_count"] == 1
    assert layer["summary"]["loop_count"] == 2
    assert layer["summary"]["certified_route_card_loop_count"] == 1
    assert layer["summary"]["needs_route_card_promotion_loop_count"] == 1
    assert layer["execution_model"]["primary_execution_artifact"] == "field_day_layer"
    assert layer["execution_model"]["proof_unit"] == "certified_route_card"
    assert layer["execution_model"]["default_phone_view"] == "field-days"
    assert layer["field_days"][0]["execution_status"] == "needs_route_card_promotion"
    assert layer["field_days"][0]["constraints"] == ["lower_hulls_even_day_on_foot"]

    certified_loop = layer["field_days"][0]["loops"][0]
    assert certified_loop["certification_status"] == "certified_route_card"
    assert certified_loop["label"] == "1A"
    assert certified_loop["segment_ids"] == [101, 102]
    assert certified_loop["route_card_ref"] == {
        "outing_id": "1-1",
        "label": "1A",
        "candidate_ids": ["cert-1"],
        "gpx_href": "gpx/official/certified-card.gpx",
        "validation_passed": True,
        "route_card_quality_passed": True,
        "certification_blockers": [],
    }
    assert "lat" not in str(certified_loop)
    assert "lon" not in str(certified_loop)

    unmatched_loop = layer["field_days"][0]["loops"][1]
    assert unmatched_loop["certification_status"] == "needs_route_card_promotion"
    assert unmatched_loop["route_card_ref"] is None


def test_find_route_card_does_not_use_ambiguous_trailhead_trail_fallback():
    module = load_module()
    field_tool_payload = {
        "routes": [
            {
                "label": "FD27A",
                "candidate_ids": [
                    "single-segment-1661-spring-creek::Avimor Spring Valley Creek parking::Avimor Spring Valley Creek parking"
                ],
                "trailhead": "Avimor Spring Valley Creek parking",
                "trails": ["Spring Creek"],
                "segment_ids": ["1661"],
            },
            {
                "label": "FD27B",
                "candidate_ids": [
                    "single-segment-1662-spring-creek::Avimor Spring Valley Creek parking::Avimor Spring Valley Creek parking"
                ],
                "trailhead": "Avimor Spring Valley Creek parking",
                "trails": ["Spring Creek"],
                "segment_ids": ["1662"],
            },
        ]
    }
    index = module.build_route_card_index(field_tool_payload)

    spring_1 = module.find_route_card(
        {
            "candidate_id": "single-segment-1661-spring-creek::Avimor Spring Valley Creek parking",
            "trailhead": "Avimor Spring Valley Creek parking",
            "trail_names": ["Spring Creek"],
        },
        index,
    )
    ambiguous = module.find_route_card(
        {
            "candidate_id": "unmatched-spring-creek",
            "trailhead": "Avimor Spring Valley Creek parking",
            "trail_names": ["Spring Creek"],
        },
        index,
    )

    assert spring_1["label"] == "FD27A"
    assert ambiguous is None


def test_find_route_card_blocks_active_replacement_before_segment_fallback():
    module = load_module()
    field_tool_payload = {
        "routes": [
            {
                "label": "FD14D",
                "candidate_ids": ["36th-street-chute"],
                "trailhead": "Full Sail",
                "trails": ["36th Street Chute"],
                "segment_ids": ["1482"],
            }
        ]
    }
    index = module.build_route_card_index(field_tool_payload)
    replacements = module.AcceptedRouteReplacementIndex(
        [
            {
                "replacement_id": "fd14d-lower",
                "status": "active",
                "hard_block_current_preservation": True,
                "target_segment_ids": ["1482"],
                "source_candidate_ids": ["36th-street-chute"],
            }
        ]
    )

    route = module.find_route_card(
        {
            "candidate_id": "36th-street-chute",
            "trailhead": "Full Sail",
            "trail_names": ["36th Street Chute"],
            "segment_ids": [1482],
        },
        index,
        replacements,
    )

    assert route is None


def test_find_route_card_allows_applied_active_replacement_but_marks_not_certified():
    module = load_module()
    field_tool_payload = {
        "routes": [
            {
                "label": "FD14D",
                "candidate_ids": ["accepted-replacement-fd14d"],
                "trailhead": "Full Sail Trailhead, N 36th St Parking",
                "trails": ["36th Street Chute"],
                "segment_ids": ["1482"],
                "accepted_replacement_id": "fd14d-lower",
                "route_card_status": "provisional_re_anchored",
                "certified_route_card": False,
                "parking": {"has_parking": True},
                "wayfinding_cues": [{"leg_miles": 1.5}],
                "validation": {"passed": True},
            }
        ]
    }
    index = module.build_route_card_index(field_tool_payload)
    replacements = module.AcceptedRouteReplacementIndex(
        [
            {
                "replacement_id": "fd14d-lower",
                "status": "active",
                "hard_block_current_preservation": True,
                "target_segment_ids": ["1482"],
                "route_card_status": "provisional_re_anchored",
                "certified_route_card": False,
            }
        ]
    )

    route = module.find_route_card({"segment_ids": [1482]}, index, replacements)
    loop = module.public_loop({"segment_ids": [1482]}, route, accepted_replacement=replacements.records[0])

    assert route["accepted_replacement_id"] == "fd14d-lower"
    assert loop["certification_status"] == "provisional_re_anchored"
    assert "route_card_not_certified" not in loop["route_card_audit_blockers"]
    assert "provisional_re_anchored" in loop["route_card_audit_blockers"]


def test_single_loop_field_day_uses_current_route_card_values_after_promotion():
    module = load_module()
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 2, "official_segment_count": 2},
        "assignments": [
            {
                "date": "2026-07-11",
                "weekday_name": "Saturday",
                "day_type": "weekend",
                "field_day": {
                    "field_day_id": "weekend-canonical_field_menu::manual-16a-2::Trailhead",
                    "p75_minutes": 310,
                    "p90_minutes": 348,
                    "p90_bound_minutes": 360,
                    "between_drive_minutes": 0,
                    "on_foot_miles": 14.96,
                    "segment_summary": {
                        "segment_count": 2,
                        "official_miles": 5.53,
                        "segment_ids": [1653, 1656],
                    },
                    "loops": [
                        {
                            "loop_id": "canonical_field_menu::manual-16a-2::Trailhead",
                            "label": "Shingle Creek + Sheep Camp lower loop",
                            "source": "canonical_field_menu",
                            "candidate_id": "manual-16a-2",
                            "trailhead": "Trailhead",
                            "trail_names": ["Shingle Creek Trail", "Sheep Camp Trail"],
                            "segment_count": 2,
                            "official_miles": 5.53,
                            "on_foot_miles": 14.96,
                            "p75_minutes": 310,
                            "p90_minutes": 348,
                            "validation_passed": True,
                        }
                    ],
                },
            }
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed"},
        "routes": [
            {
                "outing_id": "16-2",
                "label": "16A-2",
                "candidate_ids": ["manual-16a-2"],
                "trailhead": "Trailhead",
                "parking": {"name": "Trailhead", "has_parking": True},
                "segment_ids": ["1653"],
                "trails": ["Sheep Camp Trail"],
                "official_miles": 0.77,
                "on_foot_miles": 3.31,
                "door_to_door_minutes_p75": 106,
                "door_to_door_minutes_p90": 119,
                "gpx_href": "gpx/official/16a-2-sheep-camp-trail.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 3.31}],
                "validation": {"passed": True},
            }
        ],
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload)
    day = layer["field_days"][0]
    loop = day["loops"][0]

    assert day["p75_minutes"] == 106
    assert day["p90_minutes"] == 119
    assert day["field_day_schedule_p75_minutes"] == 106
    assert day["field_day_schedule_p90_minutes"] == 119
    assert day["route_card_door_to_door_p75_sum"] == 106
    assert day["route_card_door_to_door_p90_sum"] == 119
    assert day["stress"] == 0.331
    assert day["official_miles"] == 0.77
    assert day["on_foot_miles"] == 3.31
    assert day["segment_ids"] == [1653]
    assert day["timing_authority"] == "route_card_door_to_door_single_loop"
    assert day["timing_repair"]["status"] == "repaired_from_route_card"
    assert layer["summary"]["single_loop_timing_repair_count"] == 1
    assert layer["summary"]["single_loop_timing_mismatch_unrepaired_count"] == 0
    assert day["schedule_integrity"] == "passed"
    assert loop["label"] == "16A-2"
    assert loop["trail_names"] == ["Sheep Camp Trail"]
    assert loop["field_day_schedule_p75_minutes"] == 106
    assert loop["route_card_door_to_door_p75_minutes"] == 106
    assert loop["route_card_ref"]["gpx_href"] == "gpx/official/16a-2-sheep-camp-trail.gpx"


def test_multi_loop_field_day_keeps_calendar_timing_authoritative():
    module = load_module()
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 3, "official_segment_count": 3},
        "assignments": [
            {
                "date": "2026-06-18",
                "weekday_name": "Thursday",
                "day_type": "weekday",
                "field_day": {
                    "field_day_id": "weekday-a",
                    "p75_minutes": 180,
                    "p90_minutes": 202,
                    "p90_bound_minutes": 292,
                    "between_drive_minutes": 5,
                    "on_foot_miles": 6.2,
                    "segment_summary": {
                        "segment_count": 3,
                        "official_miles": 4.8,
                        "segment_ids": [101, 102, 103],
                    },
                    "loops": [
                        {
                            "loop_id": "canonical_field_menu::card-a::Trailhead A",
                            "label": "card-a",
                            "source": "canonical_field_menu",
                            "candidate_id": "card-a",
                            "trailhead": "Trailhead A",
                            "trail_names": ["Trail A"],
                            "segment_count": 2,
                            "official_miles": 3.0,
                            "on_foot_miles": 3.8,
                            "p75_minutes": 90,
                            "p90_minutes": 101,
                            "validation_passed": True,
                        },
                        {
                            "loop_id": "canonical_field_menu::card-b::Trailhead B",
                            "label": "card-b",
                            "source": "canonical_field_menu",
                            "candidate_id": "card-b",
                            "trailhead": "Trailhead B",
                            "trail_names": ["Trail B"],
                            "segment_count": 1,
                            "official_miles": 1.8,
                            "on_foot_miles": 2.4,
                            "p75_minutes": 90,
                            "p90_minutes": 101,
                            "validation_passed": True,
                        },
                    ],
                },
            }
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed", "day_gpx_validation_passed": True},
        "routes": [
            {
                "outing_id": "1-1",
                "label": "1A",
                "candidate_ids": ["card-a"],
                "trailhead": "Trailhead A",
                "parking": {"name": "Trailhead A", "has_parking": True},
                "segment_ids": ["101", "102"],
                "trails": ["Trail A"],
                "official_miles": 3.0,
                "on_foot_miles": 3.8,
                "door_to_door_minutes_p75": 100,
                "door_to_door_minutes_p90": 112,
                "gpx_href": "gpx/official/card-a.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 3.8}],
                "validation": {"passed": True},
            },
            {
                "outing_id": "1-2",
                "label": "1B",
                "candidate_ids": ["card-b"],
                "trailhead": "Trailhead B",
                "parking": {"name": "Trailhead B", "has_parking": True},
                "segment_ids": ["103"],
                "trails": ["Trail B"],
                "official_miles": 1.8,
                "on_foot_miles": 2.4,
                "door_to_door_minutes_p75": 110,
                "door_to_door_minutes_p90": 123,
                "gpx_href": "gpx/official/card-b.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 2.4}],
                "validation": {"passed": True},
            },
        ],
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload)
    day = layer["field_days"][0]

    assert layer["publication_status"] == "field_day_certified"
    assert day["execution_status"] == "executable_field_day"
    assert day["p75_minutes"] == 180
    assert day["p90_minutes"] == 202
    assert day["route_card_door_to_door_p75_sum"] == 210
    assert day["route_card_door_to_door_p90_sum"] == 235
    assert day["legacy_recomputed_p75_minutes"] == 215
    assert day["legacy_recomputed_p90_minutes"] == 240
    assert day["route_card_timing_double_count_risk"] is True
    assert layer["summary"]["total_p75_minutes"] == 180
    assert layer["summary"]["max_p90_minutes"] == 202
    assert layer["summary"]["schedule_p90_violation_day_count"] == 0
    assert layer["summary"]["day_gpx_validation_passed"] is True


def test_single_loop_field_day_keeps_explicit_timing_override():
    module = load_module()
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 1, "official_segment_count": 1},
        "assignments": [
            {
                "date": "2026-07-11",
                "weekday_name": "Saturday",
                "day_type": "weekend",
                "field_day": {
                    "field_day_id": "override-day",
                    "p75_minutes": 150,
                    "p90_minutes": 170,
                    "p90_bound_minutes": 240,
                    "between_drive_minutes": 0,
                    "timing_override_reason": "day-of heat buffer",
                    "loops": [
                        {
                            "loop_id": "canonical_field_menu::short::Trailhead",
                            "candidate_id": "short",
                            "trailhead": "Trailhead",
                            "trail_names": ["Short Trail"],
                            "segment_ids": [1],
                        }
                    ],
                },
            }
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed"},
        "routes": [
            {
                "outing_id": "1-1",
                "label": "Short",
                "candidate_ids": ["short"],
                "trailhead": "Trailhead",
                "parking": {"name": "Trailhead", "has_parking": True},
                "segment_ids": ["1"],
                "trails": ["Short Trail"],
                "official_miles": 1,
                "on_foot_miles": 2,
                "door_to_door_minutes_p75": 60,
                "door_to_door_minutes_p90": 75,
                "gpx_href": "gpx/official/short.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 2.0}],
                "validation": {"passed": True},
            }
        ],
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload)
    day = layer["field_days"][0]

    assert day["p75_minutes"] == 150
    assert day["p90_minutes"] == 170
    assert day["timing_authority"] == "calendar_assignment"
    assert day["single_loop_timing_mismatch"]["status"] == "explicit_override"
    assert layer["summary"]["single_loop_timing_repair_count"] == 0
    assert layer["summary"]["single_loop_timing_mismatch_unrepaired_count"] == 1
    assert day["schedule_integrity"] == "passed"
    assert layer["summary"]["total_p75_minutes"] == 150
    assert layer["summary"]["max_p90_minutes"] == 170


def test_certified_baseline_rejects_generated_schedule_p90_violations():
    module = load_module()
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 1, "official_segment_count": 1},
        "assignments": [
            {
                "date": "2026-06-18",
                "weekday_name": "Thursday",
                "day_type": "weekday",
                "field_day": {
                    "field_day_id": "weekday-overbound",
                    "p75_minutes": 280,
                    "p90_minutes": 306,
                    "p90_bound_minutes": 292,
                    "segment_summary": {
                        "segment_count": 1,
                        "official_miles": 1.0,
                        "segment_ids": [101],
                    },
                    "loops": [],
                },
            }
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed", "day_gpx_validation_passed": True},
        "routes": [],
    }

    with pytest.raises(ValueError, match="schedule_p90_bound_violation"):
        module.build_field_day_layer(assignments_payload, field_tool_payload)


def test_build_field_day_layer_uses_promotion_report_for_superset_replacements():
    module = load_module()
    stale_loop_id = "personal_route_menu::dry-creek-trail::MillerGulch Parking Area/Trailhead"
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 6, "official_segment_count": 6},
        "assignments": [
            {
                "date": "2026-07-05",
                "weekday_name": "Sunday",
                "day_type": "weekend",
                "field_day": {
                    "field_day_id": "weekend-dry-creek",
                    "p75_minutes": 301,
                    "p90_minutes": 341,
                    "p90_bound_minutes": 535,
                    "segment_summary": {
                        "segment_count": 6,
                        "official_miles": 7.64,
                        "segment_ids": [1523, 1542, 1543, 1544, 1545, 1546],
                    },
                    "loops": [
                        {
                            "loop_id": stale_loop_id,
                            "label": "dry-creek-trail",
                            "source": "personal_route_menu",
                            "candidate_id": "dry-creek-trail",
                            "trailhead": "MillerGulch Parking Area/Trailhead",
                            "trail_names": ["Dry Creek Trail"],
                            "segment_count": 5,
                            "official_miles": 6.97,
                            "on_foot_miles": 14.64,
                            "p75_minutes": 270,
                            "p90_minutes": 303,
                            "validation_passed": True,
                        }
                    ],
                },
            }
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed"},
        "routes": [
            {
                "outing_id": "15-1",
                "label": "15A-1",
                "candidate_ids": ["multi-start-15a-15a-ms-03-1-dry-creek-trail"],
                "trailhead": "Dry Creek / Sweet Connie roadside parking",
                "parking": {"name": "Dry Creek / Sweet Connie roadside parking", "has_parking": True},
                "segment_ids": ["1542", "1543", "1544", "1545", "1546", "1656"],
                "trails": ["Dry Creek Trail", "Shingle Creek Trail"],
                "official_miles": 11.73,
                "on_foot_miles": 11.89,
                "door_to_door_minutes_p75": 229,
                "door_to_door_minutes_p90": 257,
                "gpx_href": "gpx/official/15a-1-dry-creek-shingle.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 11.89}],
                "validation": {"passed": True},
            }
        ],
    }
    promotion_payload = {
        "promotions": [
            {
                "loop_id": stale_loop_id,
                "route_card_candidate_id": "multi-start-15a-15a-ms-03-1-dry-creek-trail",
                "mode": "preserved_existing_certified_superset_replacement",
            }
        ]
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload, promotion_payload)
    loop = layer["field_days"][0]["loops"][0]

    assert layer["summary"]["certified_route_card_loop_count"] == 1
    assert layer["summary"]["needs_route_card_promotion_loop_count"] == 0
    assert loop["certification_status"] == "certified_route_card"
    assert loop["label"] == "15A-1"
    assert loop["segment_ids"] == [1542, 1543, 1544, 1545, 1546, 1656]
    assert loop["route_card_ref"]["candidate_ids"] == [
        "multi-start-15a-15a-ms-03-1-dry-creek-trail"
    ]


def test_build_field_day_layer_skips_removed_source_loops_and_reprices_schedule():
    module = load_module()
    removed_loop_id = "personal_route_menu::quick-draw::Cartwright Trailhead"
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 2, "official_segment_count": 2},
        "assignments": [
            {
                "date": "2026-06-20",
                "weekday_name": "Saturday",
                "day_type": "weekend",
                "field_day": {
                    "field_day_id": "weekend-cartwright",
                    "p75_minutes": 171,
                    "p90_minutes": 192,
                    "p90_bound_minutes": 360,
                    "between_drive_minutes": 0,
                    "segment_summary": {"segment_count": 2, "official_miles": 1.29, "segment_ids": [1516, 1610]},
                    "loops": [
                        {
                            "loop_id": "personal_route_menu::chbh-connector::Cartwright Trailhead",
                            "label": "FD14B",
                            "source": "personal_route_menu",
                            "candidate_id": "chbh-connector",
                            "trailhead": "Cartwright Trailhead",
                            "trail_names": ["CHBH Connector"],
                            "segment_count": 1,
                            "official_miles": 0.81,
                            "on_foot_miles": 3.16,
                            "p75_minutes": 103,
                            "p90_minutes": 115,
                            "validation_passed": True,
                        },
                        {
                            "loop_id": removed_loop_id,
                            "label": "FD14C",
                            "source": "personal_route_menu",
                            "candidate_id": "quick-draw",
                            "trailhead": "Cartwright Trailhead",
                            "trail_names": ["Quick Draw"],
                            "segment_count": 1,
                            "official_miles": 0.48,
                            "on_foot_miles": 1.63,
                            "p75_minutes": 68,
                            "p90_minutes": 77,
                            "validation_passed": True,
                        },
                    ],
                },
            }
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed", "day_gpx_validation_passed": True},
        "routes": [
            {
                "outing_id": "114-2",
                "label": "FD14B",
                "candidate_ids": ["chbh-connector"],
                "trailhead": "Cartwright Trailhead",
                "parking": {"name": "Cartwright Trailhead", "has_parking": True},
                "segment_ids": ["1516", "1610"],
                "trails": ["CHBH Connector", "Quick Draw"],
                "official_miles": 1.29,
                "on_foot_miles": 3.16,
                "door_to_door_minutes_p75": 103,
                "door_to_door_minutes_p90": 115,
                "gpx_href": "gpx/official/fd14b.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 3.16}],
                "validation": {"passed": True},
            }
        ],
    }
    promotion_payload = {
        "promotions": [
            {
                "loop_id": removed_loop_id,
                "route_card_candidate_id": "chbh-connector",
                "mode": "removed_source_loop_after_segment_ownership_promotion",
                "skipped_route_card_source": True,
            }
        ]
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload, promotion_payload)
    day = layer["field_days"][0]

    assert layer["summary"]["skipped_source_loop_count"] == 1
    assert layer["summary"]["loop_count"] == 1
    assert day["loop_count"] == 1
    assert day["p75_minutes"] == 103
    assert day["p90_minutes"] == 115
    assert day["official_miles"] == 1.29
    assert day["on_foot_miles"] == 3.16
    assert day["loops"][0]["label"] == "FD14B"


def test_build_field_day_layer_applies_cluster_replacement_and_empties_old_days():
    module = load_module()
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 3, "official_segment_count": 3},
        "assignments": [
            {
                "date": "2026-06-21",
                "weekday_name": "Sunday",
                "day_type": "weekend",
                "field_day": {
                    "draft_day_number": 24,
                    "field_day_id": "old-a",
                    "p75_minutes": 100,
                    "p90_minutes": 120,
                    "p90_bound_minutes": 360,
                    "segment_summary": {"segment_count": 1, "official_miles": 1.0, "segment_ids": [101]},
                    "loops": [
                        {
                            "loop_id": "old-loop-a",
                            "label": "FD24A",
                            "source": "personal_route_menu",
                            "candidate_id": "old-a",
                            "trailhead": "Old Trailhead",
                            "trail_names": ["Old"],
                            "segment_count": 1,
                            "official_miles": 1.0,
                            "on_foot_miles": 4.0,
                            "p75_minutes": 100,
                            "p90_minutes": 120,
                        }
                    ],
                },
            },
            {
                "date": "2026-07-04",
                "weekday_name": "Saturday",
                "day_type": "weekend",
                "field_day": {
                    "draft_day_number": 27,
                    "field_day_id": "old-b",
                    "p75_minutes": 180,
                    "p90_minutes": 210,
                    "p90_bound_minutes": 360,
                    "segment_summary": {"segment_count": 2, "official_miles": 2.0, "segment_ids": [102, 103]},
                    "loops": [
                        {
                            "loop_id": "old-loop-b",
                            "label": "FD27A",
                            "source": "forced_anchor_probe",
                            "candidate_id": "old-b",
                            "trailhead": "Avimor",
                            "trail_names": ["Old B"],
                            "segment_count": 1,
                            "official_miles": 1.0,
                            "on_foot_miles": 3.0,
                            "p75_minutes": 90,
                            "p90_minutes": 105,
                        },
                        {
                            "loop_id": "old-loop-c",
                            "label": "FD27B",
                            "source": "forced_anchor_probe",
                            "candidate_id": "old-c",
                            "trailhead": "Avimor",
                            "trail_names": ["Old C"],
                            "segment_count": 1,
                            "official_miles": 1.0,
                            "on_foot_miles": 3.0,
                            "p75_minutes": 90,
                            "p90_minutes": 105,
                        },
                    ],
                },
            },
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed", "day_gpx_validation_passed": True},
        "routes": [
            {
                "outing_id": "127-1",
                "label": "H1",
                "candidate_ids": ["H1-avimor-native-harlow-spring-loop"],
                "trailhead": "Avimor",
                "parking": {"name": "Avimor", "lat": 43.7, "lon": -116.2, "has_parking": True},
                "segment_ids": ["101", "102", "103"],
                "trails": ["H1"],
                "official_miles": 3.0,
                "on_foot_miles": 5.0,
                "door_to_door_minutes_p75": 150,
                "door_to_door_minutes_p90": 170,
                "gpx_href": "gpx/official/h1.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 5.0}],
                "validation": {"passed": True},
            }
        ],
    }
    promotion_payload = {
        "field_day_replacements": [
            {
                "match": {"date": "2026-07-04", "draft_day_number": 27},
                "field_day": {
                    "draft_day_number": 27,
                    "field_day_id": "h1-day",
                    "p75_minutes": 150,
                    "p90_minutes": 170,
                    "p90_bound_minutes": 360,
                    "segment_summary": {"segment_count": 3, "official_miles": 3.0, "segment_ids": [101, 102, 103]},
                    "on_foot_miles": 5.0,
                },
                "loops": [
                    {
                        "loop_id": "h1-loop",
                        "source": "harlow_h1_route_card_promotion",
                        "candidate_id": "H1-avimor-native-harlow-spring-loop",
                        "label": "H1",
                        "trailhead": "Avimor",
                        "trail_names": ["H1"],
                        "segment_count": 3,
                        "segment_ids": [101, 102, 103],
                        "official_miles": 3.0,
                        "on_foot_miles": 5.0,
                        "p75_minutes": 150,
                        "p90_minutes": 170,
                    }
                ],
                "replaces_loop_ids": ["old-loop-b", "old-loop-c"],
            }
        ],
        "promotions": [
            {"loop_id": "h1-loop", "route_card_candidate_id": "H1-avimor-native-harlow-spring-loop"},
            {"loop_id": "old-loop-a", "skipped_route_card_source": True},
        ],
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload, promotion_payload)

    assert layer["summary"]["skipped_source_loop_count"] == 3
    assert layer["summary"]["loop_count"] == 1
    assert layer["field_days"][0]["execution_status"] == "reusable_empty_field_day"
    assert layer["field_days"][0]["segment_ids"] == []
    assert layer["field_days"][1]["field_day_id"] == "h1-day"
    assert layer["field_days"][1]["loops"][0]["label"] == "H1"
    assert layer["field_days"][1]["p90_minutes"] == 170
    assert layer["publication_status"] == "field_day_certified"


def test_build_field_day_layer_does_not_certify_route_cards_with_audit_blockers():
    module = load_module()
    assignments_payload = {
        "audit": {"passed": True, "covered_segment_count": 1, "official_segment_count": 1},
        "assignments": [
            {
                "date": "2026-06-18",
                "weekday_name": "Thursday",
                "day_type": "weekday",
                "field_day": {
                    "field_day_id": "weekday-a",
                    "p75_minutes": 90,
                    "p90_minutes": 110,
                    "p90_bound_minutes": 180,
                    "segment_summary": {"segment_count": 1, "official_miles": 1.0, "segment_ids": [101]},
                    "loops": [
                        {
                            "loop_id": "canonical_field_menu::needs-audit::Trailhead",
                            "label": "needs-audit",
                            "source": "canonical_field_menu",
                            "candidate_id": "needs-audit",
                            "trailhead": "Trailhead",
                            "trail_names": ["Audit Trail"],
                            "segment_count": 1,
                            "official_miles": 1.0,
                            "on_foot_miles": 1.0,
                            "p75_minutes": 90,
                            "p90_minutes": 110,
                            "validation_passed": True,
                        }
                    ],
                },
            }
        ],
    }
    field_tool_payload = {
        "certified_baseline": {"status": "passed"},
        "routes": [
            {
                "outing_id": "1-1",
                "label": "1A",
                "candidate_ids": ["needs-audit"],
                "trailhead": "Trailhead",
                "parking": {"name": "Trailhead", "has_parking": False},
                "segment_ids": ["101"],
                "trails": ["Audit Trail"],
                "on_foot_miles": 1.0,
                "gpx_href": "gpx/official/needs-audit.gpx",
                "wayfinding_cues": [{"cum_miles": 0.0, "leg_miles": 1.8}],
                "validation": {"passed": True},
            }
        ],
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload)

    assert layer["summary"]["certified_route_card_loop_count"] == 0
    assert layer["summary"]["needs_route_card_audit_fix_loop_count"] == 1
    assert layer["summary"]["needs_route_card_promotion_loop_count"] == 0
    assert layer["publication_status"] == "needs_route_card_audit_fix"
    loop = layer["field_days"][0]["loops"][0]
    assert loop["certification_status"] == "needs_route_card_audit_fix"
    assert loop["route_card_audit_blockers"] == [
        "missing_verified_parked_start",
        "wayfinding_mileage_mismatch",
    ]
    assert loop["route_card_ref"]["route_card_quality_passed"] is False


def test_render_markdown_shows_day_bundle_and_certification_gaps():
    module = load_module()
    layer = {
        "summary": {
            "field_day_count": 1,
            "covered_segment_count": 3,
            "official_segment_count": 3,
            "total_p75_minutes": 180,
            "multi_start_day_count": 1,
            "certified_route_card_loop_count": 1,
            "needs_route_card_promotion_loop_count": 1,
        },
        "field_days": [
            {
                "date": "2026-06-18",
                "weekday_name": "Thursday",
                "day_type": "weekday",
                "p75_minutes": 180,
                "p90_minutes": 202,
                "p90_bound_minutes": 292,
                "loop_count": 2,
                "between_drive_minutes": 5,
                "official_miles": 4.8,
                "on_foot_miles": 6.2,
                "execution_status": "needs_route_card_promotion",
                "loops": [
                    {
                        "label": "certified-card",
                        "trailhead": "Trailhead",
                        "certification_status": "certified_route_card",
                    },
                    {
                        "label": "uncertified-loop",
                        "trailhead": "Other Trailhead",
                        "certification_status": "needs_route_card_promotion",
                    },
                ],
            }
        ],
    }

    markdown = module.render_markdown(layer)

    assert "# Human-Executable Field-Day Layer" in markdown
    assert "- Primary execution artifact: `field_day_layer`." in markdown
    assert "- Certification unit: `certified_route_card`." in markdown
    assert "| 2026-06-18 | Thursday | weekday | 180 | 202 / 292 | 2 | 5 | 4.80 | 6.20 | needs_route_card_promotion |" in markdown
    assert "- `certified-card` from `Trailhead` - `certified_route_card`" in markdown
    assert "- `uncertified-loop` from `Other Trailhead` - `needs_route_card_promotion`" in markdown
