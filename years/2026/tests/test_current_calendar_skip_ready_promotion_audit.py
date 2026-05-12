import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "current_calendar_skip_ready_promotion_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("current_calendar_skip_ready_promotion_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(outing_id, label, segment_ids, *, owned_elsewhere=None, credit_cues=None, repeat_cues=None):
    return {
        "outing_id": outing_id,
        "label": label,
        "segment_ids": segment_ids,
        "segment_ownership_reconciliation": {
            "declared_owned_elsewhere_segment_ids": owned_elsewhere or [],
        },
        "wayfinding_cues": [
            {
                "seq": 1,
                "official_segment_ids": credit_cues or [],
                "official_repeat_segment_ids": repeat_cues or [],
                "display_detail": "Includes repeat official; no new credit.",
            }
        ],
    }


def latent_audit():
    return {
        "current_calendar_repricing": {
            "removed_routes": [
                {
                    "route_key": "b",
                    "route": {"outing_id": "b", "label": "B", "segment_ids": ["2"]},
                    "prior_latent_segment_ids": ["2"],
                    "prior_sources": {
                        "2": [
                            {"source_route_key": "a", "outing_id": "a", "label": "A"},
                        ]
                    },
                    "saved_on_foot_miles": 1.2,
                    "saved_p75_minutes": 30,
                    "saved_p90_minutes": 34,
                }
            ]
        }
    }


def test_skip_ready_removal_is_blocked_when_source_route_still_marks_segment_as_repeat():
    module = load_module()
    audit = module.build_promotion_audit(
        {
            "routes": [
                route("a", "A", ["1"], owned_elsewhere=["2"], repeat_cues=["2"]),
                route("b", "B", ["2"]),
            ]
        },
        latent_audit(),
    )

    candidate = audit["promotion_candidates"][0]
    assert audit["status"] == "blocked_needs_route_card_claim_promotion"
    assert candidate["promotion_status"] == "blocked"
    assert candidate["blockers"] == [
        "source_route_claim_missing_for_promoted_segment",
        "source_wayfinding_still_marks_segment_as_no_new_credit",
    ]


def test_skip_ready_removal_is_ready_when_source_route_claims_and_cues_segment_as_credit():
    module = load_module()
    audit = module.build_promotion_audit(
        {
            "routes": [
                route("a", "A", ["1", "2"], credit_cues=["2"]),
                route("b", "B", ["2"]),
            ]
        },
        latent_audit(),
    )

    candidate = audit["promotion_candidates"][0]
    assert audit["status"] == "ready_for_menu_deletion"
    assert candidate["promotion_status"] == "ready_for_menu_deletion"
    assert candidate["blockers"] == []
