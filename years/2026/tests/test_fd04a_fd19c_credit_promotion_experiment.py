import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "years" / "2026" / "scripts" / "fd04a_fd19c_credit_promotion_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fd04a_fd19c_credit_promotion_experiment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hypothetical_promotion_removes_owner_without_losing_coverage():
    module = load_module()
    routes = [
        {"outing_id": "104-1", "label": "FD04A", "segment_ids": ["1748"]},
        {"outing_id": "119-3", "label": "FD19C", "segment_ids": ["1649", "1650", "1651"]},
        {"outing_id": "other", "label": "OTHER", "segment_ids": ["2000"]},
    ]

    coverage = module.coverage_after_hypothetical_promotion(
        routes=routes,
        official_segment_count=5,
        source_route=routes[0],
        owner_route=routes[1],
        target_ids={"1649", "1650", "1651"},
    )

    assert coverage["status"] == "passed"
    assert coverage["route_count_after"] == 2
    assert coverage["source_claimed_ids_after"] == ["1649", "1650", "1651", "1748"]
    assert coverage["missing_after_ids"] == []


def test_source_credit_support_distinguishes_repeat_evidence_from_claimed_credit():
    module = load_module()
    route = {
        "segment_ids": ["1748", "1652", "1558"],
        "wayfinding_cues": [
            {"seq": 3, "cue_type": "connector_named_trail", "official_repeat_segment_ids": ["1650"], "signed_as": ["#26A Shane's"]},
            {"seq": 5, "cue_type": "connector_named_trail", "official_repeat_segment_ids": ["1651"], "signed_as": ["#26 Three Bears"]},
            {"seq": 7, "cue_type": "exit_access", "official_repeat_segment_ids": ["1649"], "signed_as": ["#26A Shane's"]},
        ],
        "segment_ownership_reconciliation": {"declared_owned_elsewhere_segment_ids": ["1649", "1650", "1651"]},
    }

    support = module.source_credit_support(route, {"1649", "1650", "1651"})

    assert support["status"] == "requires_route_card_claim_and_cue_promotion"
    assert support["source_claims_all_target_ids_now"] is False
    assert support["source_cues_all_target_ids_as_repeat_now"] is True
    assert support["source_declares_all_target_ids_owned_elsewhere_now"] is True


def test_real_fd04a_fd19c_experiment_reports_active_packet_promotion():
    module = load_module()
    field_tool_data = json.loads((REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json").read_text())
    report = module.build_experiment(
        field_tool_data=field_tool_data,
        route_repeat_audit=json.loads(
            (REPO_ROOT / "years" / "2026" / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json").read_text()
        ),
        field_latent_audit=json.loads(
            (REPO_ROOT / "years" / "2026" / "checkpoints" / "field-latent-credit-audit-2026-05-11.json").read_text()
        ),
        calendar_reorder=json.loads(
            (REPO_ROOT / "years" / "2026" / "checkpoints" / "calendar-reorder-latent-credit-experiment-2026-05-12.json").read_text()
        ),
        official_geojson=json.loads(
            (
                REPO_ROOT
                / "years"
                / "2026"
                / "inputs"
                / "official"
                / "api-pull-2026-05-04"
                / "official_foot_segments.geojson"
            ).read_text()
        ),
    )

    assert report["status"] == "active_packet_already_promoted"
    assert report["decision"] == "superseded_by_active_packet_promotion"
    assert report["summary"]["current_active_route_count"] == len(field_tool_data["routes"])
    assert report["summary"]["hypothetical_route_count_after"] == len(field_tool_data["routes"])
    assert report["summary"]["official_coverage_after_hypothetical_promotion"] == "251/251"
    assert report["summary"]["active_packet_mutated"] is True
    assert all(status == "passed" for status in report["hard_gates"].values())
