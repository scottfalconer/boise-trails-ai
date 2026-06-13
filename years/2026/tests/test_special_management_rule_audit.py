import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "special_management_rule_audit.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIELD_TOOL_DATA = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
OFFICIAL_GEOJSON = (
    REPO_ROOT
    / "years"
    / "2026"
    / "inputs"
    / "official"
    / "api-pull-2026-06-13"
    / "official_foot_segments.geojson"
)
RULES_JSON = (
    REPO_ROOT
    / "years"
    / "2026"
    / "inputs"
    / "open-data"
    / "special-management-rules-2026.json"
)
R2R_TRAILS_GEOJSON = (
    REPO_ROOT
    / "years"
    / "2026"
    / "inputs"
    / "open-data"
    / "r2r-trails-2026-05-04"
    / "boise_parks_trails_open_data.geojson"
)


def load_module():
    spec = importlib.util.spec_from_file_location("special_management_rule_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs():
    return (
        json.loads(FIELD_TOOL_DATA.read_text(encoding="utf-8")),
        json.loads(OFFICIAL_GEOJSON.read_text(encoding="utf-8")),
        json.loads(RULES_JSON.read_text(encoding="utf-8")),
    )


def only_route(field_tool_data, label):
    route = next(route for route in field_tool_data["routes"] if route["label"] == label)
    return {**field_tool_data, "routes": [route]}


def test_polecat_current_route_passes_special_management_gate():
    module = load_module()
    field_tool_data, official_geojson, rules = load_inputs()

    audit = module.build_special_management_audit(
        field_tool_data=only_route(field_tool_data, "5B"),
        official_geojson=official_geojson,
        rules_config=rules,
        packet_dir=PACKET_DIR,
    )

    assert audit["status"] == "passed"
    route = audit["routes"][0]
    assert route["label"] == "5B"
    assert route["failures"] == []


def test_polecat_reverse_route_fails_special_management_gate():
    module = load_module()
    field_tool_data, official_geojson, rules = load_inputs()
    route = only_route(field_tool_data, "5B")["routes"][0]
    official_index = module.official_segment_index(official_geojson)
    track_segments = module.parse_gpx_track_segments(PACKET_DIR / route["gpx_href"])

    audit = module.audit_route_against_special_management_rules(
        route,
        [list(reversed(segment)) for segment in reversed(track_segments)],
        official_index,
        rules["rules"],
    )

    assert "special_management_direction_violated" in {
        failure["code"] for failure in audit["failures"]
    }
    assert any("r2r-polecat-81-clockwise-through-2026" == failure["rule_id"] for failure in audit["failures"])


def test_polecat_multidirectional_exception_segment_passes_special_management_gate():
    """The Polecat clockwise rule has a both-directions exception segment (Loop 7,
    seg 1604). The route that absorbed it (5B) must still pass the gate, and the
    exception segment must be checked with both directions allowed."""
    module = load_module()
    field_tool_data, official_geojson, rules = load_inputs()

    audit = module.build_special_management_audit(
        field_tool_data=only_route(field_tool_data, "5B"),
        official_geojson=official_geojson,
        rules_config=rules,
        packet_dir=PACKET_DIR,
    )

    assert audit["status"] == "passed"
    assert audit["routes"][0]["label"] == "5B"
    assert audit["routes"][0]["failures"] == []
    checked = (
        field_tool_data["routes"]
        and only_route(field_tool_data, "5B")["routes"][0]["special_management"]["checked_segments"]
    )
    exception = next(c for c in checked if str(c["segment_id"]) == "1604")
    assert set(exception["allowed_directions"]) == {"forward", "reverse"}


def test_around_the_mountain_counterclockwise_rule_allows_forward_and_blocks_reverse():
    module = load_module()
    _field_tool_data, official_geojson, rules = load_inputs()
    official_index = module.official_segment_index(official_geojson)
    segment_ids = ["1488", "1489", "1490", "1491", "1492", "1493", "1750"]
    forward_track = [
        point
        for segment_id in segment_ids
        for point in official_index[segment_id]["parts"][0]
    ]
    route = {
        "label": "ATM",
        "outing_id": "atm-test",
        "segment_ids": segment_ids,
        "gpx_href": "synthetic.gpx",
    }

    forward = module.audit_route_against_special_management_rules(
        route,
        [forward_track],
        official_index,
        rules["rules"],
    )
    reverse = module.audit_route_against_special_management_rules(
        route,
        [list(reversed(forward_track))],
        official_index,
        rules["rules"],
    )

    assert forward["failures"] == []
    assert "special_management_direction_violated" in {
        failure["code"] for failure in reverse["failures"]
    }


def test_bucktail_on_foot_traversal_fails_mode_rule_from_open_trail_geometry():
    module = load_module()
    _field_tool_data, official_geojson, rules = load_inputs()
    open_features = module.open_trail_features(
        json.loads(R2R_TRAILS_GEOJSON.read_text(encoding="utf-8"))
    )
    bucktail = next(feature for feature in open_features if feature["trail_id"] == "20A")
    route = {
        "label": "Bucktail synthetic",
        "outing_id": "bucktail-test",
        "segment_ids": [],
        "gpx_href": "synthetic.gpx",
    }

    report = module.audit_route_against_special_management_rules(
        route,
        [bucktail["parts"][0]],
        module.official_segment_index(official_geojson),
        rules["rules"],
        open_features,
        activity_type="on_foot",
    )

    assert "special_management_mode_violated" in {
        failure["code"] for failure in report["failures"]
    }


def test_nearby_parallel_open_trail_does_not_fail_bucktail_mode_rule():
    module = load_module()
    _field_tool_data, official_geojson, rules = load_inputs()
    bucktail_part = [(-116.2, 43.62), (-116.19, 43.62)]
    open_features = [
        {
            "trail_id": "20A",
            "trail_name": "Bucktail",
            "name": "#20A Bucktail",
            "properties": {},
            "parts": [bucktail_part],
            "part_bboxes": [module.bbox_for_part(bucktail_part)],
        }
    ]
    offset_lat = 0.015 / 69.0
    nearby_track = [
        (lon, lat + offset_lat)
        for lon, lat in bucktail_part
    ]
    route = {
        "label": "Parallel legal trail synthetic",
        "outing_id": "parallel-test",
        "segment_ids": [],
        "gpx_href": "synthetic.gpx",
    }

    report = module.audit_route_against_special_management_rules(
        route,
        [nearby_track],
        module.official_segment_index(official_geojson),
        rules["rules"],
        open_features,
        activity_type="on_foot",
    )

    assert report["failures"] == []
