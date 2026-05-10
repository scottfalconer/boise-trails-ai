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
                "parking": {"name": "Trailhead", "lat": 43.1, "lon": -116.1},
                "segment_ids": ["101", "102"],
                "trails": ["Certified Trail"],
                "official_miles": 3.0,
                "on_foot_miles": 3.8,
                "door_to_door_minutes_p75": 90,
                "door_to_door_minutes_p90": 101,
                "gpx_href": "gpx/official/certified-card.gpx",
                "validation": {"passed": True},
            }
        ],
    }

    layer = module.build_field_day_layer(assignments_payload, field_tool_payload)

    assert layer["summary"]["field_day_count"] == 1
    assert layer["summary"]["loop_count"] == 2
    assert layer["summary"]["certified_route_card_loop_count"] == 1
    assert layer["summary"]["needs_route_card_promotion_loop_count"] == 1
    assert layer["field_days"][0]["execution_status"] == "needs_route_card_promotion"
    assert layer["field_days"][0]["constraints"] == ["lower_hulls_even_day_on_foot"]

    certified_loop = layer["field_days"][0]["loops"][0]
    assert certified_loop["certification_status"] == "certified_route_card"
    assert certified_loop["route_card_ref"] == {
        "outing_id": "1-1",
        "label": "1A",
        "candidate_ids": ["cert-1"],
        "gpx_href": "gpx/official/certified-card.gpx",
        "validation_passed": True,
    }
    assert "lat" not in str(certified_loop)
    assert "lon" not in str(certified_loop)

    unmatched_loop = layer["field_days"][0]["loops"][1]
    assert unmatched_loop["certification_status"] == "needs_route_card_promotion"
    assert unmatched_loop["route_card_ref"] is None


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
    assert "| 2026-06-18 | Thursday | weekday | 180 | 202 / 292 | 2 | 5 | 4.80 | 6.20 | needs_route_card_promotion |" in markdown
    assert "- `certified-card` from `Trailhead` - `certified_route_card`" in markdown
    assert "- `uncertified-loop` from `Other Trailhead` - `needs_route_card_promotion`" in markdown
