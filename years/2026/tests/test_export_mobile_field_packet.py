import importlib.util
import json
from pathlib import Path
import zipfile


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


def test_field_packet_html_is_phone_first_and_links_to_gpx_and_parking(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "Phone Field Packet" in html
    assert "Open Nav GPX" in html
    assert "Cue GPX" in html
    assert "Audit GPX" in html
    assert manifest["routes"][0]["gpx_href"] in html
    assert manifest["routes"][0]["cue_gpx_href"] in html
    assert manifest["routes"][0]["audit_gpx_href"] in html
    assert "https://www.google.com/maps/dir/?api=1&amp;destination=43.100000,-116.100000" in html
    assert "PARK/START" in html
    assert "Turn-by-turn from car" in html
    assert "Park/start at Test Trailhead" in html
    assert "Pass by car again" in html
    assert "Known water" in html
    assert "Test Trailhead · parking/start · user_verified" in html
    assert "Leave car toward Test Trail" in html
    assert "Complete Test Trail 1" in html
    assert "220 ft climb" in html
    assert "p75 24 min" in html
    assert "Turn onto Second Trail" in html
    assert "via Road Connector" in html
    assert "Official segment order" in html
    assert "Return to car" in html


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

    assert "Signpost: #51 Who Now Loop Trail" in html
    assert "Signpost: #50 Hippie Shake Trail" in html
    assert "Signpost cues: #52 Kemper's Ridge; #51 Who Now Loop" in html
    step_details = [step["detail"] for step in public_manifest["routes"][0]["turn_by_turn_steps"]]
    assert any("#51 Who Now Loop Trail" in detail for detail in step_details)
    assert any("#52 Kemper's Ridge; #51 Who Now Loop" in detail for detail in step_details)
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

    assert "Start on Test Trail" in step_titles
    assert "Turn onto Second Trail" in step_titles
    assert "Complete Test Trail 1" not in step_titles
    assert "Complete Test Trail 2" not in step_titles
    assert any("Complete Test Trail 1 and Test Trail 2 before the next turn." in detail for detail in step_details)
    assert any("At the intersection of Test Trail and Second Trail" in detail for detail in step_details)


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
    assert "Mark done" in html
    assert "Undo done" in html
    assert "Hide completed" in html
    assert "Show completed" in html
    assert "fieldPacketCompletedOutings" in html
    assert "localStorage" in html
    assert "Screenshot mode" in html
    assert "Today&apos;s best options" in html


def test_export_field_packet_writes_downloadable_gpx_zip_and_precaches_it(tmp_path):
    module = load_exporter()

    manifest = module.export_field_packet(sample_map_data(), tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    service_worker = (tmp_path / "service-worker.js").read_text(encoding="utf-8")
    zip_path = tmp_path / "gpx" / "all-field-packet-gpx.zip"

    assert zip_path.exists()
    assert "Download all GPX" in html
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
