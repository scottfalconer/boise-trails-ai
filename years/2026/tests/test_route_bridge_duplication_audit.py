import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_bridge_duplication_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_bridge_duplication_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment(seg_id, coords, official_miles=0.2, direction="both"):
    return {
        "seg_id": str(seg_id),
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "official_miles": official_miles,
        "direction": direction,
        "coordinates": coords,
        "start": coords[0],
        "end": coords[-1],
    }


def owner_route():
    return {
        "outing_id": "owner",
        "label": "Owner Route",
        "candidate_ids": ["owner-loop"],
        "segment_ids": ["100"],
        "official_miles": 0.2,
        "on_foot_miles": 0.2,
    }


def receiver_route():
    return {
        "outing_id": "receiver",
        "label": "Receiver Route",
        "candidate_ids": ["receiver-loop"],
        "segment_ids": ["101", "102"],
        "official_miles": 0.4,
        "on_foot_miles": 0.6,
        "segment_ownership_reconciliation": {
            "declared_owned_elsewhere_segment_ids": ["100"],
            "segments_owned_elsewhere": [
                {
                    "seg_id": "100",
                    "official_miles": 0.2,
                    "owned_by_routes": [{"outing_id": "owner", "label": "Owner Route"}],
                }
            ],
        },
    }


def build_audit(routes, official_segments, **kwargs):
    module = load_module()
    return module.build_bridge_duplication_audit(
        {"routes": routes},
        official_segments=official_segments,
        generated_at="2026-05-26T00:00:00Z",
        **kwargs,
    )


def test_strict_bridge_reports_chained_credit_segments():
    audit = build_audit(
        [owner_route(), receiver_route()],
        [
            segment("101", [(0.0, 0.0), (0.01, 0.0)]),
            segment("100", [(0.01, 0.0), (0.02, 0.0)]),
            segment("102", [(0.02, 0.0), (0.03, 0.0)]),
        ],
    )

    assert audit["summary"]["strict_bridge_count"] == 1
    assert audit["summary"]["strict_bridge_count_unwaived"] == 1
    finding = audit["findings"][0]
    assert finding["classification"] == "strict_bridge"
    assert finding["bridge_segment"]["seg_id"] == "100"
    assert finding["receiver_route"]["outing_id"] == "receiver"
    assert finding["chained_credit_segment_ids"] == ["101", "102"]
    assert finding["repair_candidates"][0]["candidate_type"] == "owner_route_extension"


def test_near_bridge_has_no_upper_detour_cap():
    owner = owner_route()
    receiver = receiver_route()
    audit = build_audit(
        [owner, receiver],
        [
            segment("101", [(0.0, 0.0), (0.01, 0.0)], official_miles=0.2),
            segment("100", [(0.01, 0.0), (0.02, 0.0)], official_miles=0.2),
            segment("102", [(0.02, 0.0), (0.03, 0.0)], official_miles=0.2),
            segment("201", [(0.01, 0.0), (0.01, 0.02)], official_miles=1.0),
            segment("202", [(0.01, 0.02), (0.02, 0.02)], official_miles=1.0),
            segment("203", [(0.02, 0.02), (0.02, 0.0)], official_miles=1.0),
        ],
        near_bridge_min_detour_miles=0.25,
    )

    assert audit["summary"]["near_bridge_count"] == 1
    finding = audit["findings"][0]
    assert finding["classification"] == "near_bridge"
    assert finding["detour_added_miles"] > 2.5


def test_mid_segment_junctions_are_reported_as_false_negative_risk():
    audit = build_audit(
        [],
        [
            segment("300", [(0.0, 0.0), (0.02, 0.0)]),
            segment("301", [(0.01, 0.0), (0.01, 0.01)]),
        ],
        endpoint_tolerance_miles=0.02,
    )

    assert audit["mid_segment_junction_proof"]["mid_segment_junction_count"] == 1
    assert audit["mid_segment_junction_proof"]["status"] == "incomplete_until_virtual_nodes_inserted"


def test_accepted_unavoidable_bridge_waiver_remains_reportable_but_non_blocking():
    audit = build_audit(
        [
            owner_route(),
            receiver_route(),
        ],
        [
            segment("101", [(0.0, 0.0), (0.01, 0.0)]),
            segment("100", [(0.01, 0.0), (0.02, 0.0)]),
            segment("102", [(0.02, 0.0), (0.03, 0.0)]),
        ],
        bridge_duplication_waivers=[
            {
                "status": "accepted_unavoidable_bridge",
                "bridge_segment_id": "100",
                "owner_route_key": "owner",
                "receiver_route_key": "receiver",
                "evidence": "Synthetic test waiver.",
            }
        ],
    )

    assert audit["summary"]["strict_bridge_count"] == 1
    assert audit["summary"]["strict_bridge_count_unwaived"] == 0
    finding = audit["findings"][0]
    assert finding["waived"] is True
    assert finding["graduation_status"] == "accepted_unavoidable_bridge"


def test_repeated_bridge_without_chained_credit_is_informational_tail():
    owner = owner_route()
    receiver = receiver_route()
    receiver["segment_ids"] = []

    audit = build_audit(
        [owner, receiver],
        [
            segment("100", [(0.01, 0.0), (0.02, 0.0)]),
        ],
    )

    assert audit["summary"]["strict_bridge_count"] == 0
    assert audit["summary"]["tail_opportunity_count"] == 1
    assert audit["findings"][0]["classification"] == "tail_opportunity"
    assert audit["findings"][0]["graduation_status"] == "informational"
