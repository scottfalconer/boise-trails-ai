import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_mobile_field_packet.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_mobile_field_packet", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_map_data():
    data = {
        "summary": {
            "package_count": 2,
            "covered_segment_count": 2,
            "official_miles": 2.23,
            "total_on_foot_miles": 3.84,
        },
        "progress": {"completed_segment_ids": [], "blocked_segment_ids": []},
        "manual_design": {
            "areas": [
                {
                    "area_id": "manual-area",
                    "package_number": 2,
                    "title": "Manual Area",
                    "decision": "Needs route design.",
                    "demote_candidate_ids": ["hold-route"],
                    "current_placeholder": {
                        "label": "2",
                        "trailhead": "Hold Trailhead",
                        "door_to_door_minutes": 60,
                        "official_miles": 1.0,
                        "on_foot_miles": 3.0,
                    },
                }
            ]
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Runnable Block",
                "components": [
                    {
                        "candidate_id": "test-route",
                        "trail_names": ["Test Trail"],
                        "official_miles": 1.23,
                        "on_foot_miles": 2.34,
                        "total_minutes": 45,
                        "trailhead": "Test Trailhead",
                        "segment_ids": [101],
                    }
                ],
            },
            {
                "package_number": 2,
                "block_name": "Held Block",
                "components": [
                    {
                        "candidate_id": "hold-route",
                        "trail_names": ["Hold Trail"],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                        "total_minutes": 30,
                        "trailhead": "Hold Trailhead",
                        "segment_ids": [102],
                    }
                ],
            },
        ],
        "feature_collections": {
            "routes": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "route",
                            "candidate_id": "test-route",
                            "title": "Test Trail",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.1, 43.1],
                                [-116.11, 43.11],
                                [-116.1, 43.1],
                            ],
                        },
                    }
                ],
            },
            "official_segments": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "seg_id": 101,
                            "segment_name": "Test Trail 1",
                            "trail_name": "Test Trail",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-116.1, 43.1], [-116.11, 43.11]],
                        },
                    }
                ],
            },
            "parking": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "parking",
                            "candidate_id": "test-route",
                            "name": "Test Trailhead",
                            "has_parking": True,
                        },
                        "geometry": {"type": "Point", "coordinates": [-116.1, 43.1]},
                    }
                ],
            },
        },
        "route_cues": {
            "test-route": {
                "candidate_id": "test-route",
                "title": "Test Trail",
                "route_status": "graph_validated",
                "official_miles": 1.23,
                "on_foot_miles": 2.34,
                "total_minutes": 45,
                "time_estimates_minutes": {
                    "door_to_door_p75": 45,
                    "door_to_door_p90": 59,
                },
                "trailhead": {
                    "name": "Test Trailhead",
                    "lat": 43.1,
                    "lon": -116.1,
                    "has_parking": True,
                    "has_restroom": None,
                    "has_water": True,
                    "water_confidence": "user_verified",
                },
                "logistics": {
                    "car_passes": [
                        {
                            "name": "Pass by car again",
                            "mile_from_start": 1.2,
                            "distance_to_car_miles": 0.01,
                            "lon": -116.1,
                            "lat": 43.1,
                        }
                    ],
                    "known_water": [
                        {
                            "name": "Test Trailhead",
                            "location": "parking/start",
                            "confidence": "user_verified",
                            "lon": -116.1,
                            "lat": 43.1,
                        }
                    ],
                },
                "start_access": {
                    "confidence": "medium",
                    "direct_gap_miles": 0.07,
                    "mapped_access_miles": 0.05,
                    "access_class": "direct",
                    "graph_validated": True,
                },
                "segments": [
                    {
                        "order": 1,
                        "seg_id": 101,
                        "segment_name": "Test Trail 1",
                        "trail_name": "Test Trail",
                        "official_miles": 1.23,
                        "direction_rule": "ascent",
                        "direction_cue": "Climb this segment.",
                        "ascent_ft": 220,
                        "estimated_moving_minutes": 18,
                        "estimated_moving_minutes_p75": 24,
                        "grade_adjusted_miles": 1.0,
                    }
                ],
                "return_to_car": {
                    "description": "Double back to parking.",
                    "official_repeat_miles": 1.23,
                    "connector_miles": 0,
                    "road_miles": 0,
                },
            }
        },
    }
    data["packages"][0]["components"][0]["trail_names"] = ["Test Trail", "Second Trail"]
    data["packages"][0]["components"][0]["segment_ids"] = [101, 103]
    data["feature_collections"]["official_segments"]["features"].append(
        {
            "type": "Feature",
            "properties": {
                "seg_id": 103,
                "segment_name": "Second Trail 1",
                "trail_name": "Second Trail",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[-116.11, 43.11], [-116.12, 43.12]],
            },
        }
    )
    data["route_cues"]["test-route"]["segments"].append(
        {
            "order": 2,
            "seg_id": 103,
            "segment_name": "Second Trail 1",
            "trail_name": "Second Trail",
            "official_miles": 0.5,
            "direction_rule": "both",
            "direction_cue": "Either direction allowed.",
        }
    )
    data["route_cues"]["test-route"]["between_links"] = [
        {
            "from_trail": "Test Trail",
            "to_trail": "Second Trail",
            "distance_miles": 0.12,
            "connector_miles": 0.12,
            "official_repeat_miles": 0,
            "connector_names": ["Road Connector"],
            "connector_classes": ["osm_public_road"],
        }
    ]
    return data


def test_export_field_packet_writes_gpx_for_runnable_outings_and_skips_manual_holds(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)

    assert manifest["summary"]["runnable_outing_count"] == 1
    assert manifest["summary"]["manual_hold_count"] == 1
    assert len(manifest["routes"]) == 1
    assert not list((tmp_path / "gpx").rglob("*hold-route*.gpx"))

    route = manifest["routes"][0]
    nav_gpx = Path(route["gpx_path"]).read_text(encoding="utf-8")
    cue_gpx = Path(route["cue_gpx_path"]).read_text(encoding="utf-8")
    audit_gpx = Path(route["audit_gpx_path"]).read_text(encoding="utf-8")

    assert route["gpx_href"].startswith("gpx/navigation/")
    assert route["cue_gpx_href"].startswith("gpx/cues/")
    assert route["audit_gpx_href"].startswith("gpx/audit/")
    assert "<name>PARK/START Test Trailhead</name>" in nav_gpx
    assert "<name>CAR PASS 1</name>" in nav_gpx
    assert "<name>WATER Test Trailhead</name>" in nav_gpx
    assert "<name>RETURN TO CAR</name>" in nav_gpx
    assert "<name>CUE 01 Test Trail</name>" in nav_gpx
    assert "<name>ASCENT 1 Test Trail 1</name>" not in nav_gpx
    assert "<trk>" in nav_gpx
    assert '<trkpt lat="43.100000" lon="-116.100000" />' in nav_gpx

    assert "<name>CUE 01 Test Trail</name>" in cue_gpx
    assert "<trk>" not in cue_gpx

    assert "<name>ASCENT 1 Test Trail 1</name>" in audit_gpx
    assert "<name>TURN</name>" in audit_gpx
    assert "Official 1.23 mi; On-foot 2.34 mi; Door-to-door p75 45 min" in audit_gpx


def test_load_map_data_prefers_canonical_json_over_html_snapshot(tmp_path):
    module = load_exporter()
    canonical = sample_map_data()
    html_snapshot = sample_map_data()
    html_snapshot["packages"][0]["components"][0]["candidate_id"] = "html-snapshot-route"
    html = "<script>\nconst DATA = " + json.dumps(html_snapshot) + ";\nconst map = {};\n</script>"
    json_path = tmp_path / "canonical-map-data.json"
    html_path = tmp_path / "snapshot.html"
    json_path.write_text(json.dumps(canonical), encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    loaded, source_path = module.load_map_data(map_html=html_path, map_data_json=json_path)

    assert source_path == json_path
    assert loaded["packages"][0]["components"][0]["candidate_id"] == "test-route"


def test_load_map_data_allows_explicit_html_when_json_is_missing(tmp_path):
    module = load_exporter()
    html_snapshot = sample_map_data()
    html_snapshot["packages"][0]["components"][0]["candidate_id"] = "explicit-html-route"
    html = "<script>\nconst DATA = " + json.dumps(html_snapshot) + ";\nconst map = {};\n</script>"
    html_path = tmp_path / "explicit.html"
    html_path.write_text(html, encoding="utf-8")

    loaded, source_path = module.load_map_data(map_html=html_path, map_data_json=tmp_path / "missing.json")

    assert source_path == html_path
    assert loaded["packages"][0]["components"][0]["candidate_id"] == "explicit-html-route"


def test_export_field_packet_rejects_known_collapsed_hillside_harrison_regression(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    collapsed = data["packages"][0]["components"][0]
    collapsed["candidate_id"] = "block-hillside_harrison_frontside"
    collapsed["trailhead"] = "Harrison Hollow Trailhead"
    collapsed["trail_names"] = [
        "Harrison Hollow",
        "Harrison Ridge",
        "Hippie Shake Trail",
        "Who Now Loop Trail",
        "Buena Vista Trail",
        "Bob Smylie",
        "Full Sail Trail",
        "Kemper's Ridge Trail",
        "36th Street Chute",
    ]
    collapsed["official_miles"] = 8.59
    collapsed["on_foot_miles"] = 13.66
    collapsed["total_minutes"] = 299

    with pytest.raises(ValueError, match="Package 1 collapsed"):
        module.export_field_packet(data, tmp_path)


def test_field_packet_html_is_phone_first_and_links_to_gpx_and_parking(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "Phone Field Packet" in html
    assert "Open Nav GPX" in html
    assert "Open parking in Google Maps" in html
    assert manifest["routes"][0]["gpx_href"] in html
    assert "<b>Climb</b><strong>220 ft</strong>" in html
    assert "<b>Door to door p90</b><strong>59 min</strong>" in html
    assert "Cue GPX" not in html
    assert "Audit GPX" not in html
    assert manifest["routes"][0]["cue_gpx_href"] not in html
    assert manifest["routes"][0]["audit_gpx_href"] not in html
    assert "https://www.google.com/maps/dir/?api=1&amp;destination=43.100000,-116.100000" in html
    assert "PARK/START" in html
    assert "What to do next" in html
    assert "Tap the cue you are working on" in html
    assert "decision-cards" in html
    assert "current-step" in html
    assert "Turn-by-turn from car" not in html
    assert "Park/start at Test Trailhead" in html
    assert "Pass by car again" in html
    assert "Known water" in html
    assert "Test Trailhead · parking/start · user_verified" in html
    assert "Leave car toward Test Trail" not in html
    assert "OFFICIAL START" in html
    assert "Follow Test Trail toward Second Trail" in html
    assert "This earns: Test Trail segment 1" in html
    assert "220 ft climb" in html
    assert "~24 min moving" in html
    assert "ROAD" in html
    assert "Follow Road Connector toward Second Trail" in html
    assert "JCT" in html
    assert "Take Second Trail toward return to car" in html
    assert "EXIT" in html
    assert "Return leg does not count as new official challenge credit." in html
    assert "Pin active" in html
    assert "Clear active" in html
    assert "fieldPacketActiveOuting" in html
    assert "Planner snap" not in html
    assert "Official segment order" not in html
    assert "Before leaving" not in html
    assert "Phone run card" not in html


def test_field_packet_names_non_official_access_trail_before_first_credit(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(
        sample_map_data(),
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    route = manifest["routes"][0]
    access_steps = [step for step in route["turn_by_turn_steps"] if step["kind"] == "access"]

    assert "START/ACCESS" in html
    assert "Follow #99 Access Trail toward Test Trail" in html
    assert "UNTIL signed junction with Test Trail" in html
    assert "This access leg is not official challenge credit." in html
    assert access_steps[0]["title"] == "Start on #99 Access Trail"


def test_wayfinding_enrichment_names_start_access_edge_matched_from_route_line():
    module = load_exporter()
    from field_route_walkthrough_audit import TrailEdge

    route = {
        "outing_id": "synthetic",
        "segment_ids": ["official-1"],
        "turn_by_turn_steps": [
            {
                "kind": "access",
                "title": "Leave car toward #51 Who Now Loop",
                "detail": "From the car, head toward #51 Who Now Loop.",
            }
        ],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cum_miles": 0.0,
                "leg_miles": 0.3,
                "cue_type": "start_access",
                "action": "FOLLOW",
                "signed_as": ["#51 Who Now Loop"],
                "target": "#51 Who Now Loop",
                "until": "signed junction with #51 Who Now Loop",
            },
            {
                "seq": 2,
                "cum_miles": 0.3,
                "leg_miles": 0.4,
                "cue_type": "follow_official_segment",
                "action": "FOLLOW",
                "signed_as": ["#51 Who Now Loop"],
                "target": "return to car",
                "until": "end of #51 Who Now Loop for this route",
                "official_segment_ids": ["official-1"],
            },
        ],
    }
    track_segments = [[(-116.0, 43.0), (-116.001, 43.001), (-116.002, 43.002)]]
    graph_edges = [
        TrailEdge(
            edge_id="connector-57",
            name="#57 Harrison Hollow",
            normalized_name="harrison hollow",
            signposts={"57"},
            source_class="r2r_trail",
            coords=[(-116.0, 43.0), (-116.001, 43.001)],
        ),
        TrailEdge(
            edge_id="official-51",
            name="#51 Who Now Loop",
            normalized_name="who now",
            signposts={"51"},
            source_class="official_segment",
            segment_id="official-1",
            coords=[(-116.001, 43.001), (-116.002, 43.002)],
        ),
    ]

    module.enrich_route_with_walkthrough_edge_names(route, track_segments, graph_edges)

    cue_text = json.dumps(route["wayfinding_cues"], ensure_ascii=False)
    step_text = json.dumps(route["turn_by_turn_steps"], ensure_ascii=False)
    assert "#57 Harrison Hollow" in cue_text
    assert "#57 Harrison Hollow" in step_text
    assert "01 0.00 mi" in route["wayfinding_cues"][0]["compact"]


def test_wayfinding_enrichment_names_between_connector_edge_matched_from_route_line():
    module = load_exporter()
    from field_route_walkthrough_audit import TrailEdge

    route = {
        "outing_id": "synthetic",
        "segment_ids": ["official-1", "official-2"],
        "turn_by_turn_steps": [
            {"kind": "navigate", "title": "Take First Trail", "detail": "Follow First Trail."},
            {"kind": "navigate", "title": "Take Second Trail", "detail": "Turn onto Second Trail."},
        ],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cum_miles": 0.0,
                "leg_miles": 0.4,
                "cue_type": "follow_official_segment",
                "action": "FOLLOW",
                "signed_as": ["First Trail"],
                "target": "Second Trail",
                "until": "signed junction with Second Trail",
                "official_segment_ids": ["official-1"],
            },
            {
                "seq": 2,
                "cum_miles": 0.4,
                "leg_miles": 0.2,
                "cue_type": "connector_named_trail",
                "action": "FOLLOW",
                "signed_as": ["connector/access"],
                "target": "Second Trail",
                "until": "signed junction with Second Trail",
            },
            {
                "seq": 3,
                "cum_miles": 0.6,
                "leg_miles": 0.4,
                "cue_type": "junction_turn",
                "action": "TAKE",
                "signed_as": ["Second Trail"],
                "target": "return to car",
                "until": "end of Second Trail for this route",
                "official_segment_ids": ["official-2"],
            },
        ],
    }
    track_segments = [[(-116.0, 43.0), (-116.001, 43.0), (-116.002, 43.0), (-116.003, 43.0)]]
    graph_edges = [
        TrailEdge(
            edge_id="official-1",
            name="First Trail",
            normalized_name="first",
            signposts=set(),
            source_class="official_segment",
            segment_id="official-1",
            coords=[(-116.0, 43.0), (-116.001, 43.0)],
        ),
        TrailEdge(
            edge_id="connector-road",
            name="Connector Road",
            normalized_name="connector road",
            signposts=set(),
            source_class="osm_public_road",
            coords=[(-116.001, 43.0), (-116.002, 43.0)],
        ),
        TrailEdge(
            edge_id="official-2",
            name="Second Trail",
            normalized_name="second",
            signposts=set(),
            source_class="official_segment",
            segment_id="official-2",
            coords=[(-116.002, 43.0), (-116.003, 43.0)],
        ),
    ]

    module.enrich_route_with_walkthrough_edge_names(route, track_segments, graph_edges)

    cue_text = json.dumps(route["wayfinding_cues"], ensure_ascii=False)
    step_text = json.dumps(route["turn_by_turn_steps"], ensure_ascii=False)
    assert "Connector Road" in cue_text
    assert "Connector Road" in step_text


def test_field_packet_exports_wayfinding_cue_sheet_notation(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(
        sample_map_data(),
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    field_tool_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    route = manifest["routes"][0]
    public_route = public_manifest["routes"][0]
    field_route = field_tool_data["routes"][0]

    cues = route["wayfinding_cues"]
    assert cues[0]["seq"] == 1
    assert cues[0]["cue_type"] == "start_access"
    assert cues[0]["action"] == "FOLLOW"
    assert cues[0]["cum_miles"] == 0.0
    assert cues[0]["leg_miles"] > 0
    assert cues[0]["signed_as"] == ["#99 Access Trail"]
    assert cues[0]["target"] == "Test Trail"
    assert cues[0]["until"] == "signed junction with Test Trail"
    assert cues[0]["compact"].startswith("01 0.00")
    assert "START/ACCESS" in cues[0]["compact"]
    assert public_route["wayfinding_cues"] == cues
    assert field_route["wayfinding_cues"] == cues
    assert "Field Cue Sheet" in html
    assert "01 0.00 mi" in html
    assert "START/ACCESS" in html
    assert "UNTIL signed junction with Test Trail" in html
    assert "VERIFY: watch for signs: #99 Access Trail" in html


def test_field_packet_computes_non_official_start_access_gap_from_geometry(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["feature_collections"]["routes"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.105, 43.105],
        [-116.11, 43.11],
        [-116.12, 43.12],
        [-116.1, 43.1],
    ]
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.105, 43.105],
        [-116.11, 43.11],
    ]
    data["route_cues"]["test-route"]["start_access"] = {
        "confidence": "medium",
        "direct_gap_miles": 0,
        "mapped_access_miles": 0,
        "access_class": "direct",
        "graph_validated": True,
    }

    manifest = module.export_field_packet(
        data,
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    route = manifest["routes"][0]
    access_steps = [step for step in route["turn_by_turn_steps"] if step["kind"] == "access"]

    assert route["navigation_quality"]["start_access_gap_miles"] > 0.05
    assert access_steps[0]["title"] == "Start on #99 Access Trail"
    assert "Follow the GPX access line for about" in access_steps[0]["detail"]
    assert "before official credit starts" not in access_steps[0]["detail"]


def test_field_nav_gpx_rejects_unexplained_inter_trkseg_gaps(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["between_links"] = []
    data["route_cues"]["test-route"]["return_to_car"] = {}
    data["feature_collections"]["routes"]["features"][0]["geometry"] = {
        "type": "MultiLineString",
        "coordinates": [
            [[-116.1, 43.1], [-116.101, 43.101]],
            [[-116.2, 43.2], [-116.1, 43.1]],
        ],
    }
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.101, 43.101],
    ]
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = [
        [-116.2, 43.2],
        [-116.1, 43.1],
    ]

    manifest = module.export_field_packet(data, tmp_path)

    assert manifest["summary"]["gpx_validation_passed"] is False
    failures = manifest["routes"][0]["validation"]["failures"]
    assert any(failure["code"] == "unexplained_inter_segment_gap" for failure in failures)


def test_return_to_car_metadata_does_not_explain_unrelated_inter_trkseg_gap(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["between_links"] = []
    data["feature_collections"]["routes"]["features"][0]["geometry"] = {
        "type": "MultiLineString",
        "coordinates": [
            [[-116.1, 43.1], [-116.101, 43.101]],
            [[-116.2, 43.2], [-116.1, 43.1]],
        ],
    }
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.101, 43.101],
    ]
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = [
        [-116.2, 43.2],
        [-116.1, 43.1],
    ]

    manifest = module.export_field_packet(data, tmp_path)

    assert manifest["summary"]["gpx_validation_passed"] is False
    failures = manifest["routes"][0]["validation"]["failures"]
    assert any(failure["code"] == "unexplained_inter_segment_gap" for failure in failures)


def test_field_nav_gpx_allows_inter_trkseg_gap_when_named_connector_is_declared(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["between_links"] = [
        {
            "from_trail": "Test Trail",
            "to_trail": "Second Trail",
            "distance_miles": 0.5,
            "connector_miles": 0.5,
            "connector_names": ["Road Connector"],
            "connector_classes": ["osm_public_road"],
        }
    ]
    data["feature_collections"]["routes"]["features"][0]["geometry"] = {
        "type": "MultiLineString",
        "coordinates": [
            [[-116.1, 43.1], [-116.101, 43.101]],
            [[-116.2, 43.2], [-116.1, 43.1]],
        ],
    }
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.101, 43.101],
    ]
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = [
        [-116.2, 43.2],
        [-116.1, 43.1],
    ]

    manifest = module.export_field_packet(data, tmp_path)

    assert manifest["summary"]["gpx_validation_passed"] is True
    assert manifest["routes"][0]["validation"]["declared_inter_segment_gap_count"] == 1
    assert "Follow Road Connector toward Second Trail" in (tmp_path / "index.html").read_text(encoding="utf-8")


def test_field_packet_names_non_official_return_trail_after_last_credit(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    component = data["packages"][0]["components"][0]
    component["trailhead"] = "Test Trailhead"
    component["trail_names"] = ["Test Trail", "Second Trail"]
    data["feature_collections"]["routes"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.11, 43.11],
        [-116.12, 43.12],
        [-116.13, 43.13],
        [-116.1, 43.1],
    ]
    data["feature_collections"]["official_segments"]["features"][0]["geometry"]["coordinates"] = [
        [-116.11, 43.11],
        [-116.12, 43.12],
    ]
    data["feature_collections"]["official_segments"]["features"][1]["geometry"]["coordinates"] = [
        [-116.12, 43.12],
        [-116.13, 43.13],
    ]
    route_cue = data["route_cues"]["test-route"]
    route_cue["return_to_car"] = {
        "description": "Route endpoint is already at the start trailhead within geometry tolerance.",
        "official_repeat_miles": 0,
        "connector_miles": 0,
        "road_miles": 0,
    }

    manifest = module.export_field_packet(
        data,
        tmp_path,
        trailhead_access_index={
            "test trailhead": {
                "nearest_open_trail_name": "Access Trail",
                "nearest_open_trail_label": "#99 Access Trail",
            }
        },
    )
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    route = manifest["routes"][0]
    return_steps = [step for step in route["turn_by_turn_steps"] if step["kind"] == "return"]

    assert route["navigation_quality"]["return_access_gap_miles"] > 0.05
    assert "EXIT" in html
    assert "Follow #99 Access Trail toward Test Trailhead" in html
    assert "UNTIL parked car / trailhead" in html
    assert "Return leg does not count as new official challenge credit." in html
    assert return_steps[0]["title"] == "Return via #99 Access Trail"


def test_field_packet_omits_unknown_water_from_phone_card(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    logistics = data["route_cues"]["test-route"]["logistics"]
    logistics["known_water"] = []
    logistics["car_passes"] = []
    data["route_cues"]["test-route"]["trailhead"]["has_water"] = False
    data["route_cues"]["test-route"]["trailhead"]["water_confidence"] = None

    module.export_field_packet(data, tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert "Known water" not in html
    assert "No verified water in planner data" not in html
    assert "Field logistics" not in html


def test_field_packet_surfaces_r2r_signpost_cues(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    route_cue = data["route_cues"]["test-route"]
    route_cue["segments"][0]["segment_name"] = "Who Now Loop Trail 1"
    route_cue["segments"][0]["trail_name"] = "Who Now Loop Trail"
    route_cue["segments"][1]["segment_name"] = "Hippie Shake Trail 1"
    route_cue["segments"][1]["trail_name"] = "Hippie Shake Trail"
    route_cue["between_links"][0]["from_trail"] = "Who Now Loop Trail"
    route_cue["between_links"][0]["to_trail"] = "Hippie Shake Trail"
    route_cue["between_links"][0]["connector_names"] = [
        "Kemper's Ridge #52",
        "Who Now Loop #51",
    ]

    manifest = module.export_field_packet(data, tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    gpx = Path(manifest["routes"][0]["gpx_path"]).read_text(encoding="utf-8")

    assert "VERIFY: watch for signs: #51 Who Now Loop Trail" in html
    assert "#50 Hippie Shake Trail" in html
    assert "VERIFY: watch for signs: #52 Kemper's Ridge; #51 Who Now Loop" in html
    assert "<h3>Signpost cues</h3>" not in html
    step_details = [step["detail"] for step in public_manifest["routes"][0]["turn_by_turn_steps"]]
    assert any("#51 Who Now Loop Trail" in detail for detail in step_details)
    assert any("Look for signs: #52 Kemper's Ridge; #51 Who Now Loop" in detail for detail in step_details)
    assert "Signpost: #51 Who Now Loop Trail" in gpx


def test_turn_by_turn_is_trail_navigation_not_segment_credit_order(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["route_cues"]["test-route"]["segments"].insert(
        1,
        {
            "order": 2,
            "seg_id": 104,
            "segment_name": "Test Trail 2",
            "trail_name": "Test Trail",
            "official_miles": 0.4,
            "direction_rule": "both",
            "direction_cue": "Either direction allowed.",
        },
    )
    data["route_cues"]["test-route"]["segments"][2]["order"] = 3

    module.export_field_packet(data, tmp_path)
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    route_steps = public_manifest["routes"][0]["turn_by_turn_steps"]
    step_titles = [step["title"] for step in route_steps]
    step_details = [step["detail"] for step in route_steps]

    assert "Take Test Trail" in step_titles
    assert "Turn onto Second Trail" in step_titles
    assert "Complete Test Trail 1" not in step_titles
    assert "Complete Test Trail 2" not in step_titles
    assert any("This earns: both Test Trail official segments." in detail for detail in step_details)
    assert any("At the signed junction with Second Trail" in detail for detail in step_details)


def test_turn_by_turn_includes_left_right_when_geometry_is_clear(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    data["feature_collections"]["routes"]["features"][0]["geometry"]["coordinates"] = [
        [-116.1, 43.1],
        [-116.1, 43.11],
        [-116.11, 43.11],
    ]
    for feature in data["feature_collections"]["official_segments"]["features"]:
        props = feature["properties"]
        if props["seg_id"] == 101:
            feature["geometry"]["coordinates"] = [[-116.1, 43.1], [-116.1, 43.11]]
        if props["seg_id"] == 103:
            feature["geometry"]["coordinates"] = [[-116.1, 43.11], [-116.11, 43.11]]

    module.export_field_packet(data, tmp_path)
    public_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    step_titles = [step["title"] for step in public_manifest["routes"][0]["turn_by_turn_steps"]]
    step_details = [step["detail"] for step in public_manifest["routes"][0]["turn_by_turn_steps"]]

    assert "Turn left onto Second Trail" in step_titles
    assert any("At the signed junction with Second Trail, turn left onto Second Trail." in detail for detail in step_details)


def test_export_field_packet_writes_installable_pwa_artifacts(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    pwa_manifest = json.loads((tmp_path / "manifest.webmanifest").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    service_worker = (tmp_path / "service-worker.js").read_text(encoding="utf-8")

    assert pwa_manifest["name"] == "Boise Trails Field Packet"
    assert pwa_manifest["short_name"] == "Trails Packet"
    assert pwa_manifest["start_url"] == "./"
    assert pwa_manifest["scope"] == "./"
    assert pwa_manifest["display"] == "standalone"
    assert pwa_manifest["theme_color"] == "#111827"
    assert {icon["sizes"] for icon in pwa_manifest["icons"]} >= {"192x192", "512x512"}
    assert (tmp_path / "icons" / "icon-192.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "icons" / "icon-512.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert '<link rel="manifest" href="manifest.webmanifest">' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in html
    assert 'navigator.serviceWorker.register("service-worker.js")' in html
    assert "Add to Home Screen" in html
    assert "Offline-ready" in html

    assert "self.addEventListener('install'" in service_worker
    assert "manifest.webmanifest" in service_worker
    assert manifest["routes"][0]["gpx_href"] in service_worker
    assert manifest["routes"][0]["cue_gpx_href"] in service_worker
    assert manifest["routes"][0]["audit_gpx_href"] in service_worker


def test_field_packet_supports_local_progress_filters_and_screenshot_cards(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert 'data-outing-id="1-1"' in html
    assert 'data-completion-safe="true" data-segment-ids="101 103"' in html
    assert "Mark done" in html
    assert "Undo done" in html
    assert "Pin active" in html
    assert "Clear active" in html
    assert "Hide completed" in html
    assert "Show completed" in html
    assert "Export progress" in html
    assert "boise-trails-progress.json" in html
    assert "fieldPacketCompletedOutings" in html
    assert "completedSegmentSet" in html
    assert "fieldPacketActiveOuting" in html
    assert "active-outing" in html
    assert "localStorage" in html
    assert "Screenshot mode" in html
    assert "Today&apos;s best options" in html
    assert '<button type="button" class="active" data-filter="all">All</button>' in html
    assert '<button type="button" class="active" data-filter="120">' not in html
    assert 'data-filter="60"' in html
    assert 'data-filter="360"' in html
    assert 'id="remaining-segment-count"' in html
    assert "Best today" in html
    assert "completion-safe in the current menu" in html


def test_export_field_packet_writes_public_field_tool_data_for_daily_decisions(tmp_path):
    module = load_exporter()
    map_data = sample_map_data()
    certificate = {
        "certificate_status": "passed",
        "profile": {
            "profile_id": "test-profile",
            "bounds": {
                "weekday_p90_minutes": 120,
                "weekend_p90_minutes": 180,
                "max_on_foot_miles_per_field_day": 18,
            },
        },
        "segment_set": {
            "official_segment_count": 2,
            "selected_calendar_segment_count": 2,
            "missing_segment_count": 0,
        },
        "field_days": {
            "field_day_count": 2,
            "total_p75_minutes": 90,
            "total_on_foot_miles": 3.84,
            "max_on_foot_miles": 2.34,
            "max_p90_minutes": 120,
        },
        "gpx_validation": {
            "day_track_validation_passed": True,
            "actual_max_day_trackpoint_gap_miles": 0.01,
        },
    }

    source_path = tmp_path / "canonical-map-data.json"
    source_path.write_text(json.dumps(map_data), encoding="utf-8")
    module.export_field_packet(
        map_data,
        tmp_path,
        certificate_data=certificate,
        source_metadata=module.source_metadata_for_map_data(map_data, source_path),
    )
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))

    assert field_data["source"]["canonical_data_role"] == "2026-outing-menu-map-data"
    assert field_data["source"]["source_label"] == "canonical-map-data.json"
    assert field_data["source"]["map_data_sha256"] == module.stable_json_sha256(map_data)
    assert field_data["source"]["source_file_sha256"] == module.file_sha256(source_path)
    assert field_data["time_filters_minutes"] == [60, 90, 120, 180, 240, 360]
    assert field_data["certified_baseline"]["status"] == "passed"
    assert field_data["certified_baseline"]["profile_id"] == "test-profile"
    assert field_data["certified_baseline"]["official_segment_count"] == 2
    assert field_data["certified_baseline"]["covered_segment_count"] == 2
    assert field_data["certified_baseline"]["missing_segment_count"] == 0
    assert field_data["progress"]["remaining_segment_count_at_start"] == 2
    assert field_data["routes"][0]["outing_id"] == "1-1"
    assert field_data["routes"][0]["segment_ids"] == ["101", "103"]
    assert field_data["routes"][0]["door_to_door_minutes_p90"] == 59
    assert field_data["routes"][0]["effort"]["ascent_ft"] == 220
    assert field_data["routes"][0]["effort"]["grade_adjusted_miles"] == 1.0
    assert field_data["routes"][0]["effort"]["elevation_source"] == "dem"
    assert field_data["routes"][0]["parking"]["name"] == "Test Trailhead"
    assert field_data["routes"][0]["gpx_href"].startswith("gpx/navigation/")
    assert field_data["routes"][0]["validation"]["passed"] is True
    assert (
        field_data["routes"][0]["completion_safety"][
            "normal_completion_preserves_remaining_menu_coverage"
        ]
        is True
    )


def test_field_packet_uses_route_level_dem_effort_when_segment_effort_is_missing(tmp_path):
    module = load_exporter()
    data = sample_map_data()
    cue = data["route_cues"]["test-route"]
    cue["effort"] = {
        "ascent_ft": 640,
        "descent_ft": 420,
        "grade_adjusted_miles": 3.88,
        "elevation_source": "dem",
    }
    cue["time_estimates_minutes"] = {
        "door_to_door_p75": 45,
        "door_to_door_p90": 59,
        "moving_effort_p50": 34,
        "moving_effort_p75": 42,
    }
    for segment in cue["segments"]:
        segment.pop("ascent_ft", None)
        segment.pop("descent_ft", None)
        segment.pop("grade_adjusted_miles", None)
        segment.pop("estimated_moving_minutes", None)
        segment.pop("estimated_moving_minutes_p75", None)

    module.export_field_packet(data, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    effort = field_data["routes"][0]["effort"]

    assert effort["ascent_ft"] == 640
    assert effort["descent_ft"] == 420
    assert effort["grade_adjusted_miles"] == 3.88
    assert effort["estimated_moving_minutes_p50"] == 34
    assert effort["estimated_moving_minutes_p75"] == 42
    assert effort["elevation_source"] == "dem"
    assert "<b>Climb</b><strong>640 ft</strong>" in html


def test_export_field_packet_can_apply_phone_progress_before_building_remaining_menu(tmp_path):
    module = load_exporter()
    data = sample_map_data()

    updated = module.apply_progress_to_map_data(
        data,
        {"completed_outing_ids": ["1-1"], "missed_segment_ids": ["103"]},
    )

    assert updated["progress"]["completed_segment_ids"] == [101]
    assert updated["progress"]["missed_segment_ids"] == [103]
    assert data["progress"]["completed_segment_ids"] == []

    manifest = module.export_field_packet(updated, tmp_path)
    field_data = json.loads((tmp_path / "field-tool-data.json").read_text(encoding="utf-8"))

    assert manifest["summary"]["runnable_outing_count"] == 1
    assert field_data["progress"]["completed_segment_ids_at_export"] == ["101"]
    assert field_data["progress"]["remaining_segment_count_at_start"] == 1


def test_export_field_packet_writes_downloadable_gpx_zip_and_precaches_it(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    service_worker = (tmp_path / "service-worker.js").read_text(encoding="utf-8")
    zip_path = tmp_path / "gpx" / "all-field-packet-gpx.zip"

    assert zip_path.exists()
    assert "Download all GPX" not in html
    assert manifest["summary"]["gpx_zip_href"] == "gpx/all-field-packet-gpx.zip"
    assert "gpx/all-field-packet-gpx.zip" in service_worker
    with zipfile.ZipFile(zip_path) as archive:
        gpx_names = [name for name in archive.namelist() if name.endswith(".gpx")]
    assert sorted(gpx_names) == sorted(
        [
            manifest["routes"][0]["gpx_href"].removeprefix("gpx/"),
            manifest["routes"][0]["cue_gpx_href"].removeprefix("gpx/"),
            manifest["routes"][0]["audit_gpx_href"].removeprefix("gpx/"),
        ]
    )


def test_field_packet_public_outputs_do_not_leak_private_origin_or_paths(tmp_path):
    module = load_exporter()

    module.export_field_packet(sample_map_data(), tmp_path)
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in tmp_path.rglob("*") if path.is_file()
    )

    assert "/Users/scott" not in combined
    assert "outputs/private" not in combined
    assert "911" not in combined
    assert "18th" not in combined


def test_densify_track_segments_reduces_point_gaps_below_limit():
    module = load_exporter()
    sparse = [[(-116.1, 43.1), (-116.2, 43.2)]]

    dense = module.densify_track_segments(sparse, max_gap_miles=0.05)
    validation = module.validate_track_segments(dense, max_gap_miles=0.05)

    assert len(dense[0]) > 2
    assert validation["passed"] is True
