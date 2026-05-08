import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_tool_completion_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_tool_completion_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_audit_inputs(tmp_path):
    module = load_module()
    canonical = {
        "summary": {"component_route_count": 1},
        "packages": [{"package_number": 1}],
        "map_validation": {
            "source_gap_warning_count": 0,
            "route_validations": [
                {"candidate_id": "test-route", "source_gap_warning": False}
            ],
        },
    }
    packet_dir = tmp_path / "packet"
    nav_dir = packet_dir / "gpx" / "official"
    nav_dir.mkdir(parents=True)
    (nav_dir / "route.gpx").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="43.100000" lon="-116.100000" />
    <trkpt lat="43.110000" lon="-116.110000" />
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )
    (packet_dir / "index.html").write_text(
        """
        <button data-filter="60"></button><button data-filter="90"></button>
        <button data-filter="120"></button><button data-filter="180"></button>
        <button data-filter="240"></button><button data-filter="360"></button>
        <button>Mark done</button><button>Hide completed</button><button>Export progress</button>
        <section><h3>Field Cue Sheet</h3><p>Tap the cue you are working on</p><ol class="steps decision-cards"><li class="current-step">Cue</li></ol></section>
        <script>
        const x = "fieldPacketCompletedOutings";
        function completedSegmentSet() {}
        const note = "missed_segment_ids";
        const best = "Best today for 120 minutes: route · 4 new official segment(s) · completion-safe in the current menu";
        </script>
        """,
        encoding="utf-8",
    )
    for name in ["field-tool-data.json", "manifest.json", "service-worker.js"]:
        (packet_dir / name).write_text("{}", encoding="utf-8")
    field_tool_data = {
        "schema": "boise_trails_field_tool_data_v1",
        "source": {"map_data_sha256": module.stable_json_sha256(canonical)},
        "time_filters_minutes": [60, 90, 120, 180, 240, 360],
        "certified_baseline": {
            "status": "passed",
            "official_segment_count": 251,
            "covered_segment_count": 251,
            "missing_segment_count": 0,
        },
        "summary": {"segment_count_in_field_menu": 1},
        "routes": [
            {
                "outing_id": "1-1",
                "label": "1A",
                "trailhead": "Trailhead",
                "candidate_ids": ["test-route"],
                "segment_ids": ["101"],
                "door_to_door_minutes_p75": 75,
                "door_to_door_minutes_p90": 90,
                "official_miles": 1.0,
                "on_foot_miles": 1.4,
                "gpx_href": "gpx/official/route.gpx",
                "parking": {"lat": 43.1, "lon": -116.1, "has_parking": True},
                "effort": {
                    "ascent_ft": 120,
                    "descent_ft": 80,
                    "grade_adjusted_miles": 1.12,
                    "estimated_moving_minutes_p75": 35,
                },
                "validation": {"passed": True},
                "completion_safety": {
                    "normal_completion_preserves_remaining_menu_coverage": True,
                    "missing_remaining_segment_ids_after_completion": [],
                },
                "turn_by_turn_steps": [
                    {"kind": "park"},
                    {"kind": "navigate"},
                    {"kind": "return"},
                ],
                "wayfinding_cues": [
                    {
                        "seq": 1,
                        "cum_miles": 0.0,
                        "leg_miles": 0.1,
                        "cue_type": "start_access",
                        "action": "FOLLOW",
                        "signed_as": ["Access Trail"],
                        "target": "Official Trail",
                        "until": "signed junction with Official Trail",
                    },
                    {
                        "seq": 2,
                        "cum_miles": 0.1,
                        "leg_miles": 1.0,
                        "cue_type": "follow_official_segment",
                        "action": "FOLLOW",
                        "signed_as": ["Official Trail"],
                        "target": "Return to Trailhead",
                        "until": "signed return junction",
                    },
                    {
                        "seq": 3,
                        "cum_miles": 1.1,
                        "leg_miles": 0.3,
                        "cue_type": "return_to_car",
                        "action": "FOLLOW",
                        "signed_as": ["Access Trail"],
                        "target": "Trailhead",
                        "until": "parked car / trailhead",
                    },
                ],
            }
        ],
    }
    manifest = {
        "summary": {
            "gpx_validation_passed": True,
            "navigation_gpx_count": 1,
            "failed_gpx_count": 0,
        }
    }
    official_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"segId": 101, "direction": "both"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-116.1, 43.1], [-116.11, 43.11]],
                },
            }
        ],
    }
    recertification = {
        "status": "passed",
        "summary": {
            "remaining_full_completion_feasible": True,
            "remaining_coverage_preserved": True,
        },
    }
    return {
        "field_tool_data": field_tool_data,
        "manifest": manifest,
        "official_geojson": official_geojson,
        "index_html": (packet_dir / "index.html").read_text(encoding="utf-8"),
        "packet_dir": packet_dir,
        "canonical_map_data": canonical,
        "recertification_report": recertification,
    }


def test_completion_audit_passes_when_field_tool_contract_is_met(tmp_path):
    module = load_module()
    audit = module.build_completion_audit(**sample_audit_inputs(tmp_path))

    assert audit["status"] == "passed"
    assert audit["summary"]["passed_requirement_count"] == audit["summary"]["requirement_count"]


def test_completion_audit_fails_when_canonical_source_hash_does_not_match(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["field_tool_data"]["source"]["map_data_sha256"] = "stale"

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Phone page and map share the canonical field-menu source"]["passed"] is False


def test_completion_audit_fails_when_field_decision_cards_are_missing(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["index_html"] = inputs["index_html"].replace("Field Cue Sheet", "Turn-by-turn from car")

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Phone page presents field decisions as tappable cue cards"]["passed"] is False


def test_completion_audit_fails_when_wayfinding_cue_lacks_until_target(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["wayfinding_cues"] = [
        {
            "seq": 1,
            "cue_type": "start_access",
            "action": "FOLLOW",
            "signed_as": ["Access Trail"],
            "target": "Official Trail",
        },
        {
            "seq": 2,
            "cue_type": "return_to_car",
            "action": "FOLLOW",
            "signed_as": ["Access Trail"],
            "target": "Trailhead",
            "until": "parked car / trailhead",
        },
    ]

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert "wayfinding cue 1 start_access missing until" in checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]


def test_completion_audit_fails_when_harrison_hollow_access_cue_is_missing(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["label"] = "1B"
    route["trailhead"] = "Harrison Hollow"
    route["turn_by_turn_steps"] = [
        {"kind": "park", "title": "Park/start at Harrison Hollow Trailhead"},
        {"kind": "access", "title": "Leave car toward #51 Who Now Loop"},
        {"kind": "navigate", "title": "Take #51 Who Now Loop"},
        {"kind": "return", "title": "Return to car"},
    ]

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"][
        "passed"
    ] is False
    assert "missing named Harrison Hollow access cue before Who Now" in checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]


def test_completion_audit_fails_when_harrison_hollow_return_cue_is_missing(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["label"] = "1B"
    route["trailhead"] = "Harrison Hollow"
    route["turn_by_turn_steps"] = [
        {"kind": "park", "title": "Park/start at Harrison Hollow Trailhead"},
        {"kind": "access", "title": "Start on #57 Harrison Hollow (AWT)"},
        {"kind": "navigate", "title": "Take #51 Who Now Loop"},
        {"kind": "navigate", "title": "Turn right onto #50 Hippie Shake Trail"},
        {"kind": "return", "title": "Return to car", "detail": "You should be back at the parking point."},
    ]
    route["navigation_quality"] = {"return_access_gap_miles": 0.2}

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert "missing named Harrison Hollow return cue after Hippie Shake" in checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]


def test_completion_audit_fails_when_any_route_implies_done_at_car_with_return_access_gap(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["label"] = "9A"
    route["trailhead"] = "Generic Trailhead"
    route["navigation_quality"] = {"return_access_gap_miles": 0.31}
    route["turn_by_turn_steps"] = [
        {"kind": "park", "title": "Park/start at Generic Trailhead"},
        {"kind": "access", "title": "Start on Access Trail"},
        {"kind": "navigate", "title": "Take Final Official Trail"},
        {"kind": "return", "title": "Return to car", "detail": "You should be back at the parking point."},
    ]

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert "return cue implies done at car despite non-credit return leg" in checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]


def test_completion_audit_fails_when_any_route_omits_start_access_with_start_gap(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["label"] = "9A"
    route["trailhead"] = "Generic Trailhead"
    route["navigation_quality"] = {"start_access_gap_miles": 0.28}
    route["turn_by_turn_steps"] = [
        {"kind": "park", "title": "Park/start at Generic Trailhead"},
        {"kind": "access", "title": "Leave car toward First Official Trail", "detail": "Head to First Official Trail."},
        {"kind": "navigate", "title": "Take First Official Trail"},
        {"kind": "return", "title": "Return to car", "detail": "You should be back at the parking point."},
    ]

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert "missing explicit non-credit start-access cue" in checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]


def test_completion_audit_fails_when_source_gap_warning_is_hidden_by_rendered_gpx(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["canonical_map_data"]["map_validation"] = {
        "source_gap_warning_count": 1,
        "route_validations": [
            {
                "candidate_id": "test-route",
                "source_gap_warning": True,
                "source_max_gap_miles": 0.72,
                "rendered_passed": True,
            }
        ],
    }
    inputs["field_tool_data"]["source"]["map_data_sha256"] = module.stable_json_sha256(
        inputs["canonical_map_data"]
    )

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Source routes have no hidden unstitched gaps"]["passed"] is False
    assert "test-route" in checks["Source routes have no hidden unstitched gaps"]["evidence"]


def test_completion_audit_fails_source_gap_even_when_named_connector_is_declared(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["canonical_map_data"]["route_cues"] = {
        "test-route": {
            "between_links": [
                {
                    "from_trail": "Trail A",
                    "to_trail": "Trail B",
                    "connector_names": ["Named Connector"],
                    "connector_classes": ["r2r_trail"],
                    "connector_miles": 0.4,
                }
            ]
        }
    }
    inputs["canonical_map_data"]["map_validation"] = {
        "source_gap_warning_count": 1,
        "route_validations": [
            {
                "candidate_id": "test-route",
                "source_gap_warning": True,
                "source_max_gap_miles": 0.72,
                "rendered_passed": True,
            }
        ],
    }
    inputs["field_tool_data"]["source"]["map_data_sha256"] = module.stable_json_sha256(
        inputs["canonical_map_data"]
    )

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Source routes have no hidden unstitched gaps"]["passed"] is False
    assert "test-route" in checks["Source routes have no hidden unstitched gaps"]["evidence"]


def test_completion_audit_fails_when_nav_gpx_does_not_cover_claimed_segment(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    nav_path = inputs["packet_dir"] / "gpx" / "official" / "route.gpx"
    nav_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="43.100000" lon="-116.100000" />
    <trkpt lat="43.101000" lon="-116.101000" />
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Nav GPX covers claimed official segment endpoints"]["passed"] is False
    assert "does not cover official segment 101 endpoints" in checks[
        "Nav GPX covers claimed official segment endpoints"
    ]["evidence"]
