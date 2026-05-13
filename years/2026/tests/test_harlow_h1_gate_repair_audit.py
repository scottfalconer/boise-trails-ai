import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "harlow_h1_gate_repair_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("harlow_h1_gate_repair_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_gap_repairs_explain_target_snap_repair():
    module = load_module()

    repairs = module.direct_gap_repairs(
        {
            "link_rows": [
                {
                    "to_segment_id": "1687",
                    "to_segment_name": "Twisted Spring 1",
                    "path_source": "direct_gap_fallback",
                    "link_distance_miles": 0.06,
                }
            ]
        },
        {
            "link_rows": [
                {
                    "to_segment_id": "1687",
                    "path_source": "mapped_graph",
                    "link_distance_miles": 0.1,
                    "connector_miles": 0.09,
                    "official_repeat_miles": 0.01,
                    "official_repeat_segment_ids": ["1687"],
                    "connector_names": ["Twisted Spring Trail - #8"],
                }
            ]
        },
    )

    assert repairs[0]["status"] == "repaired_with_mapped_graph_path"
    assert repairs[0]["repaired_official_repeat_segment_ids"] == ["1687"]
    assert "avoided the target official segment" in repairs[0]["explanation"]


def test_overlay_accepted_parking_source_adds_candidate_metadata_without_claiming_packet_sync():
    module = load_module()

    parking, sync = module.overlay_accepted_parking_source(
        {
            "name": "Avimor Spring Valley Creek parking",
            "lat": 43.7771389,
            "lon": -116.2625034,
            "has_parking": True,
            "parking_confidence": None,
            "source": None,
        },
        {
            "name": "Avimor Spring Valley Creek parking",
            "parking_confidence": "osm_amenity_parking_fee_no_capacity_36_source_checked",
            "source": "osm_overpass_amenity_parking_2026_05_06_plus_alltrails_spring_valley_creek",
            "field_ready": True,
        },
    )

    assert parking["parking_confidence"] == "osm_amenity_parking_fee_no_capacity_36_source_checked"
    assert sync["status"] == "candidate_metadata_synced_from_accepted_manual_anchor"
    assert sync["field_packet_source_fix_still_needed"] is True


def test_coverage_after_replacement_preserves_official_segment_set():
    module = load_module()

    result = module.coverage_after_replacement(
        {
            "routes": [
                {"label": "FD27A", "segment_ids": ["1"]},
                {"label": "FD30A", "segment_ids": ["2"]},
                {"label": "keep", "segment_ids": ["3"]},
            ]
        },
        [{"seg_id": "1"}, {"seg_id": "2"}, {"seg_id": "3"}],
        replacement_labels=["FD27A", "FD30A"],
        replacement_segment_ids=["1", "2"],
    )

    assert result["status"] == "coverage_preserved"
    assert result["covered_segment_count"] == 3
    assert result["missing_segment_ids"] == []


def test_repeat_accounting_passes_when_self_repeat_is_declared_and_priced():
    module = load_module()

    review = module.repeat_accounting(
        {
            "official_repeat_miles": 0.54,
            "self_repeat_segment_ids": ["1687"],
            "link_rows": [
                {
                    "official_repeat_segment_ids": ["1687"],
                    "official_repeat_miles": 0.01,
                }
            ],
        },
        {
            "hidden_self_repeat_ids": [],
            "latent_credit_ids": [],
            "unpriced_repeat_ids": [],
        },
    )

    assert review["status"] == "passed"
    assert review["classification"] == "explicit_priced_repeat"
    assert review["declared_repeat_segment_ids"] == ["1687"]


def test_hidden_self_repeat_conversion_moves_mileage_from_connector_to_repeat():
    module = load_module()

    loop, conversions = module.declare_hidden_self_repeat_ids(
        {
            "connector_miles": 1.29,
            "official_repeat_miles": 0.54,
            "self_repeat_segment_ids": ["1687"],
            "link_rows": [
                {
                    "to_segment_id": "1661",
                    "link_track_miles": 0.53,
                    "connector_miles": 0.28,
                    "official_repeat_miles": 0.14,
                    "official_repeat_segment_ids": ["1688"],
                }
            ],
        },
        ["1689"],
        {"1689": {"official_miles": 0.07}},
    )

    assert loop["connector_miles"] == 1.22
    assert loop["official_repeat_miles"] == 0.61
    assert loop["link_rows"][0]["official_repeat_segment_ids"] == ["1688", "1689"]
    assert conversions[0]["declared_on_link_to_segment_id"] == "1661"
