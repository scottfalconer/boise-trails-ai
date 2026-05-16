import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "years" / "2026" / "scripts" / "fd04a_fd19c_route_card_promotion_path_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fd04a_fd19c_route_card_promotion_path_experiment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialized_calendar_swaps_dates_but_keeps_fd19c_for_skip_report():
    module = load_module()
    calendar = {
        "assignments": [
            {
                "date": "2026-06-18",
                "field_day": {"loops": [{"candidate_id": "shanes-trail"}]},
            },
            {
                "date": "2026-06-24",
                "field_day": {"loops": [{"candidate_id": "combo-two-point-femrites-patrol-shanes-connector"}]},
            },
        ]
    }
    calendar_reorder = {
        "pairwise_scenarios": [
            {
                "scenario_id": "104-1-before-119-3",
                "source_target_date": "2026-06-18",
                "owner_target_date": "2026-06-24",
            }
        ]
    }

    result = module.materialize_reordered_calendar(calendar, calendar_reorder)

    source_assignment = module.assignment_for_candidate(
        result["assignments"],
        "combo-two-point-femrites-patrol-shanes-connector",
    )
    owner_assignment = module.assignment_for_candidate(result["assignments"], "shanes-trail")
    assert source_assignment["date"] == "2026-06-18"
    assert owner_assignment["date"] == "2026-06-24"
    assert owner_assignment["field_day"]["loops"][0]["candidate_id"] == "shanes-trail"


def test_promotion_rows_are_promoted_remove_source_rows():
    module = load_module()
    proof = {
        "proposed_segment_promotion_rows": [
            {"segment_id": "1649", "to": {"insert_after_segment_id": 1558}},
            {"segment_id": "1650", "to": {"insert_after_segment_id": 1748}},
            {"segment_id": "1651", "to": {"insert_after_segment_id": 1652}},
        ]
    }

    rows = module.route_promotion_rows(proof)

    assert [row["segment_id"] for row in rows] == ["1649", "1650", "1651"]
    assert all(row["status"] == "promoted" for row in rows)
    assert all(row["source_action"] == "remove_route_card" for row in rows)
    assert all(row["from"]["candidate_id"] == "shanes-trail" for row in rows)
    assert all(row["to"]["candidate_id"] == "combo-two-point-femrites-patrol-shanes-connector" for row in rows)


def test_verifier_requires_fd04a_claim_cues_and_fd19c_removal(tmp_path):
    module = load_module()
    field_tool = {
        "summary": {"segment_count_in_field_menu": 251},
        "routes": [
            {
                "outing_id": "104-1",
                "label": "FD04A",
                "candidate_ids": ["combo-two-point-femrites-patrol-shanes-connector"],
                "segment_ids": ["1558", "1649", "1650", "1651", "1652", "1748"],
                "wayfinding_cues": [
                    {"seq": 1, "official_segment_ids": ["1650", "1651"]},
                    {"seq": 2, "official_segment_ids": ["1649"]},
                ],
            }
        ],
    }
    field_day_layer = {
        "publication_status": "field_day_certified",
        "summary": {
            "field_day_count": 31,
            "loop_count": 43,
            "schedule_p90_violation_day_count": 0,
            "needs_route_card_promotion_loop_count": 0,
            "needs_route_card_audit_fix_loop_count": 0,
        },
        "field_days": [],
    }
    promote = {
        "summary": {"newly_promoted_loop_count": 0, "route_card_source_loop_count": 49},
        "promotions": [
            {
                "mode": "removed_source_loop_after_segment_ownership_promotion",
                "source_candidate_id": "shanes-trail",
            }
        ],
    }
    route_repeat = {
        "summary": {
            "hidden_self_repeat_segment_count": 0,
            "unpriced_repeat_segment_count": 0,
            "latent_credit_segment_count": 0,
        },
        "routes": [{"outing_id": "104-1", "hidden_self_repeat_ids": [], "unpriced_repeat_ids": [], "latent_credit_ids": []}],
    }
    latent = {
        "summary": {"unclaimed_uncompleted_segment_count": 0},
        "route_reviews": [{"outing_id": "104-1", "latent_completed_segment_ids": []}],
    }
    progress = {"summary": {"remaining_coverage_preserved": True}}
    recert = {"status": "passed", "summary": {"remaining_full_completion_feasible": True}}
    completion = {"status": "passed", "summary": {"field_menu_segment_count": 251}}
    walkthrough = {"status": "passed", "summary": {}}
    official = {"features": [{} for _ in range(251)]}

    paths = {}
    for name, payload in {
        "field_tool_data_json": field_tool,
        "field_day_layer_json": field_day_layer,
        "promote_report_json": promote,
        "route_repeat_json": route_repeat,
        "latent_json": latent,
        "progress_json": progress,
        "recertification_json": recert,
        "completion_json": completion,
        "walkthrough_json": walkthrough,
        "official_geojson": official,
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(module.json.dumps(payload), encoding="utf-8")
        paths[name] = path

    args = type("Args", (), paths)
    report = module.build_verification_report(args)

    assert report["status"] == "passed"
    assert report["gates"]["fd19c_removed_from_routes"] == "passed"
    assert report["gates"]["fd04a_phone_cues_claim_target_segments"] == "passed"
