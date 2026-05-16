import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bogus_b1_b2_gate_repair_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bogus_b1_b2_gate_repair_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_test_gpx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="43.000000" lon="-116.000000" />
    <trkpt lat="43.000000" lon="-116.006000" />
    <trkpt lat="43.000000" lon="-116.012000" />
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )


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

    assert review["status"] == "failed_direct_gap_fallback_unresolved"
    assert review["named_cue_repair_count"] == 1
    assert review["source_cue_gpx_leg_count"] == 0
    assert review["post_named_cue_priced_track_miles"] == 10.11


def test_source_cue_gpx_leg_prices_replacement_but_does_not_clear_candidate_gate(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    write_test_gpx(packet_dir / "gpx/official/source.gpx")
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
            "gpx_href": "gpx/official/source.gpx",
            "wayfinding_cues": [
                {
                    "seq": 1,
                    "cue_type": "start_access",
                    "signed_as": ["#91 Deer Point"],
                    "target": "Sunshine XC",
                    "leg_miles": 0.46,
                    "route_miles": 0.0,
                    "route_leg_miles": 0.55,
                }
            ],
        }
    }

    review = module.direct_gap_repairs(
        bundle_id="B1-simplot-side-bogus-day",
        loop=loop,
        routes_by_label=routes_by_label,
        packet_dir=packet_dir,
    )

    row = review["rows"][0]
    assert review["status"] == "failed_source_cue_gpx_available_but_candidate_not_rebuilt"
    assert review["source_cue_gpx_leg_count"] == 1
    assert row["repair_status"] == "source_cue_gpx_available_but_candidate_not_rebuilt"
    assert row["source_cue_gpx_review"]["status"] == "source_cue_gpx_leg_extractable"
    assert row["candidate_gpx_rebuilt_with_replacement"] is False


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


def test_mileage_breakdown_counts_source_repeat_and_road_estimate():
    module = load_module()
    loop = {
        "official_miles": 2.0,
        "track_miles": 5.0,
        "connector_miles": 1.5,
        "official_repeat_miles": 0.2,
        "link_rows": [
            {
                "path_source": "direct_gap_fallback",
                "link_distance_miles": 0.5,
            },
            {
                "path_source": "mapped_graph",
                "connector_miles": 0.7,
                "connector_classes": ["osm_public_road", "r2r_trail"],
            },
        ],
    }
    direct_gap_review = {
        "post_source_cue_gpx_priced_track_miles": 5.8,
        "candidate_rendered_gpx": {"gpx_miles": 5.0},
        "rows": [
            {
                "original_direct_gap_miles": 0.5,
                "replacement_leg_miles": 1.3,
                "source_cue_official_repeat_miles": 0.3,
                "source_cue_type": "connector_road",
            }
        ],
    }

    breakdown = module.mileage_breakdown(loop, direct_gap_review)

    assert breakdown["total_on_foot_miles"] == 5.8
    assert breakdown["official_repeat_miles"] == 0.5
    assert breakdown["connector_miles"] == 2.0
    assert breakdown["road_miles"] == 1.7
    assert breakdown["road_miles_status"] == "upper_bound_from_mixed_connector_classes"


def test_source_review_keeps_closure_as_operational_gate_not_route_truth():
    module = load_module()
    review = module.source_review_for_bundle({"bundle_id": "B1-simplot-side-bogus-day"})

    assert review["around_the_mountain_signage"]["route_truth_effect"] == "operational_gate_not_official_segment_truth"
    assert review["closure_date_conditions"]["status"] == "operational_gate_not_route_truth"
