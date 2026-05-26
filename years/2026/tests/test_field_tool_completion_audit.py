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
    <trkpt lat="43.116300" lon="-116.116300" />
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
    official_repeat_audit = {
        "status": "passed",
        "summary": {
            "bucket_a_bad_hidden_self_repeat_count": 0,
            "repeat_legs_missing_segment_ids": 0,
            "repeat_cues_missing_text": 0,
            "unreconciled_extra_credit_segment_count": 0,
        },
    }
    route_repeat_audit = {
        "status": "passed",
        "summary": {
            "failed_route_count": 0,
            "missing_gpx_route_count": 0,
            "hidden_self_repeat_segment_count": 0,
            "latent_credit_segment_count": 0,
            "unpriced_repeat_segment_count": 0,
        },
    }
    latent_repricing_audit = {
        "status": "no_current_calendar_route_removal",
        "summary": {
            "current_calendar_removed_route_count": 0,
            "current_calendar_saved_on_foot_miles": 0.0,
            "current_calendar_saved_p75_minutes": 0,
        },
    }
    ownership_audit = {
        "status": "no_ownership_reassignment_savings",
        "summary": {
            "current_calendar_skip_ready_removed_route_count": 0,
            "current_calendar_skip_ready_saved_on_foot_miles": 0.0,
            "order_free_saved_on_foot_miles": 0.0,
        },
    }
    simulated_sweep_audit = {
        "status": "simulated_progress_priority_found",
        "summary": {
            "sweeps_with_future_removed_route_count": 0,
            "sweeps_with_future_shrunk_route_count": 0,
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
        "official_repeat_audit": official_repeat_audit,
        "route_repeat_audit": route_repeat_audit,
        "latent_repricing_audit": latent_repricing_audit,
        "ownership_audit": ownership_audit,
        "simulated_sweep_audit": simulated_sweep_audit,
    }


def test_completion_audit_passes_when_field_tool_contract_is_met(tmp_path):
    module = load_module()
    audit = module.build_completion_audit(**sample_audit_inputs(tmp_path))

    assert audit["status"] == "passed"
    assert audit["summary"]["passed_requirement_count"] == audit["summary"]["requirement_count"]


def test_completion_audit_counts_validated_progress_outside_active_field_menu(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["segment_ids"] = ["102"]
    inputs["field_tool_data"]["progress"] = {
        "completed_segment_ids_at_export": ["101"],
        "blocked_segment_ids_at_export": [],
    }
    inputs["official_geojson"]["features"].append(
        {
            "type": "Feature",
            "properties": {"segId": 102, "direction": "both"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[-116.1, 43.1], [-116.11, 43.11]],
            },
        }
    )

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "passed"
    assert audit["summary"]["field_menu_segment_count"] == 1
    assert audit["summary"]["completed_segment_count_at_export"] == 1
    assert audit["summary"]["accounted_segment_count"] == 2
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Active field packet accounts for every official segment geometry id"]["passed"] is True


def test_completion_audit_fails_when_canonical_source_hash_does_not_match(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["field_tool_data"]["source"]["map_data_sha256"] = "stale"

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Phone page and map share the canonical field-menu source"]["passed"] is False


def test_completion_audit_fails_when_route_repeat_hard_gate_fails(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["route_repeat_audit"] = {
        "status": "failed",
        "summary": {
            "failed_route_count": 1,
            "missing_gpx_route_count": 0,
            "hidden_self_repeat_segment_count": 1,
            "latent_credit_segment_count": 0,
            "unpriced_repeat_segment_count": 0,
        },
    }

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Route repeat optimization hard gate has no hidden self-repeat, latent credit, or unpriced repeat failures"]["passed"] is False


def test_completion_audit_records_advisory_optimization_actions_without_failing(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["latent_repricing_audit"] = {
        "status": "proved_current_calendar_savings",
        "summary": {
            "current_calendar_removed_route_count": 2,
            "current_calendar_saved_on_foot_miles": 4.39,
            "current_calendar_saved_p75_minutes": 147,
        },
    }

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "passed"
    assert audit["summary"]["advisory_action_count"] == 2
    advisory = {check["name"]: check for check in audit["advisory_checks"]}
    assert advisory["Latent-credit delta repricing advisory"]["status"] == "actionable"


def test_completion_audit_surfaces_unwaived_bridge_debt_as_advisory(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["bridge_duplication_audit"] = {
        "status": "actionable_bridge_debt",
        "summary": {
            "strict_bridge_count_unwaived": 1,
            "near_bridge_count": 1,
            "graduated_blocking_strict_bridge_count": 0,
        },
    }

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "passed"
    advisory = {check["name"]: check for check in audit["advisory_checks"]}
    assert advisory["Bridge duplication repair advisory"]["status"] == "actionable"
    assert advisory["Bridge duplication repair advisory"]["action_count"] == 2


def test_completion_audit_fails_after_bridge_debt_graduates_to_blocking_failure(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["bridge_duplication_audit"] = {
        "status": "actionable_bridge_debt",
        "summary": {
            "strict_bridge_count_unwaived": 1,
            "near_bridge_count": 0,
            "graduated_blocking_strict_bridge_count": 1,
        },
    }

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Graduated bridge-duplication failures are repaired or waived"]["passed"] is False


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


def test_completion_audit_fails_when_card_and_cue_mileage_disagree_but_ignores_gpx_distance(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["label"] = "Mismatch"
    route["on_foot_miles"] = 1.4
    route["wayfinding_cues"][-1]["cum_miles"] = 6.0
    route["wayfinding_cues"][-1]["leg_miles"] = 0.5
    gpx_path = inputs["packet_dir"] / "gpx" / "official" / "route.gpx"
    gpx_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="43.100000" lon="-116.100000" />
    <trkpt lat="43.160000" lon="-116.160000" />
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    evidence = checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]
    assert "wayfinding cue mileage 6.50 mi does not match card on-foot mileage 1.40 mi" in evidence
    assert "Nav GPX mileage" not in evidence
    assert "does not match card on-foot mileage 1.40 mi" in evidence


def test_completion_audit_fails_when_named_start_access_cue_is_missing(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["label"] = "Synthetic"
    route["trailhead"] = "Named Access Trailhead"
    route["navigation_quality"] = {"start_access_gap_miles": 0.2}
    route["wayfinding_cues"][0]["signed_as"] = ["#57 Harrison Hollow (AWT)"]
    route["turn_by_turn_steps"] = [
        {"kind": "park", "title": "Park/start at Named Access Trailhead"},
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
    assert "missing named start-access cue #57 Harrison Hollow (AWT)" in checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]


def test_completion_audit_fails_when_named_return_access_cue_is_missing(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["label"] = "Synthetic"
    route["trailhead"] = "Named Access Trailhead"
    route["turn_by_turn_steps"] = [
        {"kind": "park", "title": "Park/start at Named Access Trailhead"},
        {"kind": "access", "title": "Start on #57 Harrison Hollow (AWT)"},
        {"kind": "navigate", "title": "Take #51 Who Now Loop"},
        {"kind": "navigate", "title": "Turn right onto #50 Hippie Shake Trail"},
        {"kind": "return", "title": "Return to car", "detail": "You should be back at the parking point."},
    ]
    route["wayfinding_cues"][2]["signed_as"] = ["#57 Harrison Hollow (AWT)"]
    route["navigation_quality"] = {"return_access_gap_miles": 0.2}

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert "missing named return-access cue #57 Harrison Hollow (AWT)" in checks[
        "Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort"
    ]["evidence"]


def test_completion_audit_allows_summarized_return_when_primary_named_access_is_visible(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    route = inputs["field_tool_data"]["routes"][0]
    route["turn_by_turn_steps"] = [
        {"kind": "park", "title": "Park/start at Trailhead"},
        {"kind": "navigate", "title": "Take Official Trail"},
        {
            "kind": "return",
            "title": "Return via #81 Polecat Loop, +2 more",
            "detail": "Use #81 Polecat Loop and connectors back toward the car.",
        },
    ]
    route["wayfinding_cues"][2]["signed_as"] = [
        "#81 Polecat Loop",
        "OSM path connector 110670",
        "Polecat Loop (STM)",
    ]
    route["navigation_quality"] = {"return_access_gap_miles": 0.2}

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "passed"


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
