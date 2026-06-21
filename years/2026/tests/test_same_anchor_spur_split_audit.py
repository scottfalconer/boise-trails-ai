import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "same_anchor_spur_split_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("same_anchor_spur_split_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_segment(seg_id, coords, official_miles=0.5, direction="both"):
    return {
        "seg_id": seg_id,
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "coordinates": coords,
        "official_miles": official_miles,
        "direction": direction,
    }


def route(label, trailhead, segment_ids, on_foot_miles):
    return {
        "outing_id": label.lower(),
        "label": label,
        "route_code": label,
        "route_name": f"Route {label}",
        "trailhead": trailhead,
        "parking": {"lon": -116.0, "lat": 43.0},
        "field_ready": True,
        "segment_ids": segment_ids,
        "on_foot_miles": on_foot_miles,
    }


def test_same_anchor_spur_split_is_blocking_when_incremental_cost_is_lower():
    module = load_module()
    official = [
        official_segment(1, [(-116.0, 43.0), (-115.99, 43.0)], official_miles=0.7),
        official_segment(2, [(-115.99, 43.0), (-115.99, 43.008)], official_miles=0.5),
    ]
    field_tool_data = {
        "routes": [
            route("MAIN", "Shared Trailhead", ["1"], 3.0),
            route("SPUR", "Shared Trailhead", ["2"], 2.2),
        ]
    }
    route_tracks = {
        "MAIN": module.densify_coordinates([(-116.0, 43.0), (-115.99, 43.0)]),
        "SPUR": module.densify_coordinates([(-115.99, 43.0), (-115.99, 43.008)]),
    }

    audit = module.build_audit(field_tool_data, official, route_tracks, min_savings_miles=0.25)

    assert audit["status"] == "failed"
    assert audit["summary"]["finding_count"] == 1
    finding = audit["findings"][0]
    assert finding["host_route"]["label"] == "MAIN"
    assert finding["spur_route"]["label"] == "SPUR"
    assert finding["estimated_incremental_out_and_back_miles"] == 1.0
    assert finding["estimated_saved_on_foot_miles"] == 1.2


def test_spur_split_audit_ignores_different_anchor_routes():
    module = load_module()
    official = [
        official_segment(1, [(-116.0, 43.0), (-115.99, 43.0)]),
        official_segment(2, [(-115.99, 43.0), (-115.99, 43.008)]),
    ]
    field_tool_data = {
        "routes": [
            route("MAIN", "Shared Trailhead", ["1"], 3.0),
            route("SPUR", "Other Trailhead", ["2"], 2.2),
        ]
    }
    field_tool_data["routes"][1]["parking"] = {"lon": -116.2, "lat": 43.2}
    route_tracks = {
        "MAIN": module.densify_coordinates([(-116.0, 43.0), (-115.99, 43.0)]),
        "SPUR": module.densify_coordinates([(-115.99, 43.0), (-115.99, 43.008)]),
    }

    audit = module.build_audit(field_tool_data, official, route_tracks)

    assert audit["status"] == "passed"
    assert audit["summary"]["finding_count"] == 0


def test_spur_split_audit_ignores_through_routes_with_two_contacts():
    module = load_module()
    official = [
        official_segment(1, [(-116.0, 43.0), (-115.99, 43.0)]),
        official_segment(2, [(-116.0, 43.0), (-115.99, 43.0)], official_miles=0.5),
    ]
    field_tool_data = {
        "routes": [
            route("MAIN", "Shared Trailhead", ["1"], 3.0),
            route("PARALLEL", "Shared Trailhead", ["2"], 2.2),
        ]
    }
    route_tracks = {
        "MAIN": module.densify_coordinates([(-116.0, 43.0), (-115.99, 43.0)]),
        "PARALLEL": module.densify_coordinates([(-116.0, 43.0), (-115.99, 43.0)]),
    }

    audit = module.build_audit(field_tool_data, official, route_tracks)

    assert audit["status"] == "passed"
    assert audit["summary"]["finding_count"] == 0


def test_spur_split_audit_ignores_disconnected_same_anchor_candidate_without_savings():
    module = load_module()
    official = [
        official_segment(1, [(-116.0, 43.0), (-115.99, 43.0)], official_miles=1.0),
        official_segment(2, [(-115.99, 43.0), (-115.99, 43.008)], official_miles=1.0),
        official_segment(3, [(-115.98, 43.0), (-115.98, 43.008)], official_miles=1.0),
    ]
    field_tool_data = {
        "routes": [
            route("MAIN", "Shared Trailhead", ["1"], 5.0),
            route("DISCONNECTED", "Shared Trailhead", ["2", "3"], 3.2),
        ]
    }
    route_tracks = {
        "MAIN": module.densify_coordinates([(-116.0, 43.0), (-115.99, 43.0)]),
        "DISCONNECTED": module.densify_coordinates([(-115.99, 43.0), (-115.99, 43.008)]),
    }

    audit = module.build_audit(field_tool_data, official, route_tracks, min_savings_miles=0.25)

    assert audit["status"] == "passed"
    assert audit["summary"]["finding_count"] == 0
    assert audit["summary"]["advisory_disconnected_candidate_count"] == 0
    assert audit["advisories"] == []


def test_spur_split_audit_fails_disconnected_same_anchor_candidate_with_material_savings():
    module = load_module()
    official = [
        official_segment(1, [(-116.0, 43.0), (-115.99, 43.0)], official_miles=1.0),
        official_segment(2, [(-115.99, 43.0), (-115.99, 43.008)], official_miles=0.5),
        official_segment(3, [(-115.98, 43.0), (-115.98, 43.008)], official_miles=0.5),
    ]
    field_tool_data = {
        "routes": [
            route("MAIN", "Shared Trailhead", ["1"], 5.0),
            route("DISCONNECTED", "Shared Trailhead", ["2", "3"], 3.2),
        ]
    }
    route_tracks = {
        "MAIN": module.densify_coordinates([(-116.0, 43.0), (-115.99, 43.0)]),
        "DISCONNECTED": module.densify_coordinates([(-115.99, 43.0), (-115.99, 43.008)]),
    }

    audit = module.build_audit(field_tool_data, official, route_tracks, min_savings_miles=0.25)

    assert audit["status"] == "failed"
    assert audit["summary"]["finding_count"] == 0
    assert audit["summary"]["advisory_disconnected_candidate_count"] == 1
    advisory = audit["advisories"][0]
    assert advisory["host_route"]["label"] == "MAIN"
    assert advisory["candidate_route"]["label"] == "DISCONNECTED"
    assert advisory["status"] == "manual_review_disconnected_same_anchor_candidate"
