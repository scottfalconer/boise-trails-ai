import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_official_repeat_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_official_repeat_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def map_data_with_repeat_leg(repeat_leg):
    return {
        "packages": [
            {
                "components": [
                    {
                        "candidate_id": "route-a",
                        "field_menu_label": "A",
                        "segment_ids": [1, 2],
                        "official_miles": 2.0,
                        "on_foot_miles": 3.0,
                    }
                ]
            }
        ],
        "route_cues": {
            "route-a": {
                "candidate_id": "route-a",
                "between_links": [repeat_leg],
                "return_to_car": {},
                "start_access": {},
            }
        },
    }


def field_tool_with_repeat_cue(cue):
    return {
        "routes": [
            {
                "outing_id": "1-1",
                "label": "A",
                "candidate_ids": ["route-a"],
                "segment_ids": ["1", "2"],
                "official_miles": 2.0,
                "on_foot_miles": 3.0,
                "wayfinding_cues": [cue],
                "segment_ownership_reconciliation": {
                    "status": "no_latent_official_segments",
                    "segments_owned_elsewhere": [],
                    "unclaimed_completed_segments": [],
                },
            }
        ]
    }


def test_audit_fails_when_repeat_miles_lack_segment_ids():
    module = load_module()
    report = module.audit(
        map_data_with_repeat_leg(
            {
                "official_repeat_miles": 0.2,
                "connector_miles": 0.1,
                "connector_classes": ["official_repeat"],
            }
        ),
        field_tool_with_repeat_cue(
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "note": "Connector mileage does not count as new official challenge credit.",
            }
        ),
    )

    assert report["status"] == "failed"
    assert report["summary"]["repeat_legs_missing_segment_ids"] == 1


def test_audit_accepts_counted_and_cued_self_repeat_and_reconciled_extra_credit():
    module = load_module()
    field_tool = field_tool_with_repeat_cue(
        {
            "seq": 2,
            "cue_type": "connector_named_trail",
            "official_repeat_miles": 0.2,
            "official_repeat_segment_ids": ["1"],
            "note": "Connector includes 0.2 mi repeat official; no new credit.",
        }
    )
    field_tool["routes"][0]["segment_ownership_reconciliation"] = {
        "status": "reconciled",
        "segments_owned_elsewhere": [
            {
                "seg_id": "3",
                "trail_name": "Other Trail",
                "official_miles": 0.4,
                "owned_by_routes": [{"outing_id": "2-1", "label": "B"}],
            }
        ],
        "unclaimed_completed_segments": [],
    }

    report = module.audit(
        map_data_with_repeat_leg(
            {
                "official_repeat_miles": 0.2,
                "official_repeat_segment_ids": [1],
                "connector_miles": 0.1,
                "connector_classes": ["official_repeat"],
            }
        ),
        field_tool,
    )

    assert report["status"] == "passed"
    assert report["summary"]["bucket_a_bad_hidden_self_repeat_count"] == 0
    assert report["summary"]["bucket_b_legitimate_repeat_or_optimization_target_count"] == 1
    assert report["summary"]["bucket_c_reconciled_extra_credit_route_count"] == 1
