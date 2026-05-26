import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_edge_cover_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_edge_cover_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment(seg_id, coords, official_miles=0.5, direction="both"):
    return {
        "seg_id": seg_id,
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "official_miles": official_miles,
        "direction": direction,
        "coordinates": coords,
        "start": coords[0],
        "end": coords[-1],
    }


def write_gpx(path, coords):
    path.parent.mkdir(parents=True)
    points = "\n".join(f'<trkpt lat="{lat}" lon="{lon}"></trkpt>' for lon, lat in coords)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
{points}
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )


def test_depot_revisit_before_all_required_edges_are_cleared_fails(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "official" / "phase-reset.gpx",
        [
            (-116.000, 43.000),
            (-115.990, 43.000),
            (-116.000, 43.000),
            (-116.000, 43.010),
            (-116.000, 43.000),
        ],
    )
    route = {
        "outing_id": "112-1",
        "label": "FD12A",
        "trailhead": "West Climb",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/phase-reset.gpx",
        "parking": {"lon": -116.000, "lat": 43.000},
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "official_segment_ids": [101]},
            {"seq": 2, "cue_type": "car_pass_connector"},
            {"seq": 3, "cue_type": "follow_official_segment", "official_segment_ids": [102]},
        ],
    }

    audit = module.build_route_edge_cover_audit(
        {"routes": [route]},
        official_segments=[
            segment(101, [(-116.000, 43.000), (-115.990, 43.000)]),
            segment(102, [(-116.000, 43.000), (-116.000, 43.010)]),
        ],
        packet_dir=packet_dir,
    )

    assert audit["status"] == "failed"
    assert audit["summary"]["phase_reset_failure_count"] == 1
    failure = audit["failed_routes"][0]["hard_failures"][0]
    assert failure["code"] == "depot_revisit_before_required_edges_cleared"
    assert failure["remaining_segment_ids"] == ["102"]


def test_junction_spur_out_and_back_then_continuation_is_accepted(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "official" / "spur-clear.gpx",
        [
            (-116.000, 43.000),
            (-115.995, 43.000),
            (-115.995, 43.006),
            (-115.995, 43.000),
            (-115.990, 43.000),
            (-116.000, 43.000),
        ],
    )
    route = {
        "outing_id": "route-a",
        "label": "Spur Clear",
        "trailhead": "Trailhead",
        "segment_ids": [201, 202, 203],
        "gpx_href": "gpx/official/spur-clear.gpx",
        "parking": {"lon": -116.000, "lat": 43.000},
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "official_segment_ids": [201]},
            {"seq": 2, "cue_type": "junction_turn", "official_segment_ids": [202]},
            {"seq": 3, "cue_type": "junction_turn", "official_segment_ids": [203]},
        ],
    }

    audit = module.build_route_edge_cover_audit(
        {"routes": [route]},
        official_segments=[
            segment(201, [(-116.000, 43.000), (-115.995, 43.000)]),
            segment(202, [(-115.995, 43.000), (-115.995, 43.006)]),
            segment(203, [(-115.995, 43.000), (-115.990, 43.000)]),
        ],
        packet_dir=packet_dir,
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["phase_reset_failure_count"] == 0
    assert audit["routes"][0]["route_quality"]["status"] == "passed"


def test_disconnected_required_components_include_depot_attachment_cost(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "official" / "disconnected.gpx",
        [
            (-116.000, 43.000),
            (-115.990, 43.000),
            (-115.980, 43.000),
            (-115.970, 43.000),
            (-115.960, 43.000),
            (-116.000, 43.000),
        ],
    )
    route = {
        "outing_id": "route-b",
        "label": "Disconnected",
        "trailhead": "Trailhead",
        "segment_ids": [301, 302],
        "gpx_href": "gpx/official/disconnected.gpx",
        "parking": {"lon": -116.000, "lat": 43.000},
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "official_segment_ids": [301]},
            {"seq": 2, "cue_type": "follow_official_segment", "official_segment_ids": [302]},
        ],
    }

    audit = module.build_route_edge_cover_audit(
        {"routes": [route]},
        official_segments=[
            segment(301, [(-115.990, 43.000), (-115.980, 43.000)], official_miles=1.0),
            segment(302, [(-115.970, 43.000), (-115.960, 43.000)], official_miles=1.0),
        ],
        packet_dir=packet_dir,
    )

    quality = audit["routes"][0]["route_quality"]
    assert quality["required_miles"] == 2.0
    assert quality["depot_component_attachment_miles"] > 0
    assert quality["lower_bound_miles"] > quality["required_miles"]


def test_disconnected_component_depot_revisit_is_advisory_without_replacement_proof(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "official" / "disconnected-phase-reset.gpx",
        [
            (-116.000, 43.000),
            (-115.990, 43.000),
            (-116.000, 43.000),
            (-116.020, 43.000),
            (-116.010, 43.000),
            (-116.000, 43.000),
        ],
    )
    route = {
        "outing_id": "route-c",
        "label": "Disconnected Phase Reset",
        "trailhead": "Trailhead",
        "segment_ids": [401, 402],
        "gpx_href": "gpx/official/disconnected-phase-reset.gpx",
        "parking": {"lon": -116.000, "lat": 43.000},
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "official_segment_ids": [401]},
            {"seq": 2, "cue_type": "car_pass_connector"},
            {"seq": 3, "cue_type": "follow_official_segment", "official_segment_ids": [402]},
        ],
    }

    audit = module.build_route_edge_cover_audit(
        {"routes": [route]},
        official_segments=[
            segment(401, [(-115.990, 43.000), (-115.985, 43.000)], official_miles=0.5),
            segment(402, [(-116.020, 43.000), (-116.015, 43.000)], official_miles=0.5),
        ],
        packet_dir=packet_dir,
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["phase_reset_failure_count"] == 0
    assert audit["summary"]["phase_reset_advisory_count"] == 1
    route_audit = audit["routes"][0]
    assert route_audit["hard_failures"] == []
    assert route_audit["advisory_findings"][0]["code"] == "depot_revisit_before_required_edges_cleared"
    assert route_audit["route_quality"]["status"] == "advisory"


def test_fd12a_repaired_packet_route_has_no_depot_phase_reset():
    module = load_module()
    field_tool_data = module.read_json(module.DEFAULT_FIELD_TOOL_DATA_JSON)
    route = next(item for item in field_tool_data["routes"] if item["outing_id"] == "112-1")
    official_segments, _ = module.load_official_segments(module.DEFAULT_OFFICIAL_GEOJSON)
    official_by_id = module.segment_index(official_segments)

    audit = module.audit_route(
        route,
        official_by_id=official_by_id,
        packet_dir=module.DEFAULT_PACKET_DIR,
        connector_graph=None,
        snap_tolerance_miles=100 * module.MILES_PER_FOOT,
    )

    assert audit["audit_status"] == "passed"
    assert audit["hard_failures"] == []
    assert audit["segment_ids"] == ["1504", "1505", "1506", "1507", "1565", "1566", "1718", "1719", "1755"]
    assert audit["route_quality"]["generated_miles"] < 5.1
    direction_evidence = route["segment_direction_evidence"]
    assert {direction_evidence[segment_id]["direction_rule"] for segment_id in audit["segment_ids"]} == {"both"}
