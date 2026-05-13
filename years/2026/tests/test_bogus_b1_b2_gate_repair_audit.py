import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bogus_b1_b2_gate_repair_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bogus_b1_b2_gate_repair_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_gap_named_cue_repair_does_not_clear_gpx_gate():
    module = load_module()
    loop = {
        "track_miles": 10.0,
        "direct_gap_fallback_miles": 0.35,
        "link_rows": [
            {
                "to_segment_id": "1713",
                "to_segment_name": "Sunshine XC 1",
                "path_source": "direct_gap_fallback",
                "link_distance_miles": 0.35,
            }
        ],
    }
    routes_by_label = {
        "FD07A": {
            "label": "FD07A",
            "wayfinding_cues": [
                {
                    "seq": 1,
                    "cue_type": "start_access",
                    "signed_as": ["#91 Deer Point"],
                    "target": "Sunshine XC",
                    "leg_miles": 0.46,
                }
            ],
        }
    }

    review = module.direct_gap_repairs(
        bundle_id="B1-simplot-side-bogus-day",
        loop=loop,
        routes_by_label=routes_by_label,
    )

    assert review["status"] == "failed_continuous_gpx_still_direct_gap"
    assert review["named_cue_repair_count"] == 1
    assert review["post_named_cue_priced_track_miles"] == 10.11


def test_repeat_ownership_declares_non_template_repeat_owned_elsewhere():
    module = load_module()
    bundle = {"replace_route_labels": ["FD07A"], "bundle_id": "B1-simplot-side-bogus-day"}
    loop = {
        "official_repeat_miles": 0.08,
        "self_repeat_segment_ids": [],
        "non_template_repeat_segment_ids": ["1703"],
        "link_rows": [{"official_repeat_segment_ids": ["1703"]}],
    }
    owner_by_segment = {"1703": ["18"]}

    review = module.repeat_and_ownership_review(
        bundle=bundle,
        loop=loop,
        owner_by_segment=owner_by_segment,
    )

    assert review["status"] == "classified_explicit_priced_repeat"
    assert review["declared_owned_elsewhere_segment_ids"] == ["1703"]
    assert review["unowned_latent_credit_ids"] == []


def test_source_review_keeps_closure_as_operational_gate_not_route_truth():
    module = load_module()
    review = module.source_review_for_bundle({"bundle_id": "B1-simplot-side-bogus-day"})

    assert review["around_the_mountain_signage"]["route_truth_effect"] == "operational_gate_not_official_segment_truth"
    assert review["closure_date_conditions"]["status"] == "operational_gate_not_route_truth"
