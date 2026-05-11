import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_field_day_layer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_field_day_layer", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert day["stress"] == 0.331
    assert day["official_miles"] == 0.77
    assert day["on_foot_miles"] == 3.31
    assert day["segment_ids"] == [1653]
    assert loop["label"] == "16A-2"
    assert loop["trail_names"] == ["Sheep Camp Trail"]
    assert loop["route_card_ref"]["gpx_href"] == "gpx/official/16a-2-sheep-camp-trail.gpx"


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
