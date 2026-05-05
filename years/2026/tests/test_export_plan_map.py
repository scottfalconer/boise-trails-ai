import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_plan_map.py"


def load_map_exporter():
    spec = importlib.util.spec_from_file_location("export_plan_map", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_inputs():
    runbook = {
        "profile_name": "test-profile",
        "run_id": "run-1",
        "summary": {
            "scheduled_segments": 1,
            "scheduled_official_miles": 1.2,
            "scheduled_total_on_foot_miles": 2.4,
            "scheduled_ascent_ft": 300,
            "scheduled_days": 1,
        },
        "audit": {"execution_validation_passed": True},
        "days": [
            {
                "date": "2026-06-18",
                "official_new_miles": 1.2,
                "estimated_total_on_foot_miles": 2.4,
                "ascent_ft": 300,
                "realistic_total_minutes": 80,
                "requires_normal_cap_exception": False,
                "trailheads": ["Example Trailhead"],
                "day_transport": {
                    "legs": [
                        {
                            "leg_type": "drive_to_first_trailhead",
                            "from": "Home (private)",
                            "to": "Example Trailhead",
                            "distance_miles": 3.1,
                            "duration_minutes": 12,
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [
                                    [-116.00, 43.00],
                                    [-116.001, 43.001],
                                ],
                            },
                        }
                    ]
                },
                "outings": [
                    {
                        "outing_id": "example-route",
                        "trail_names": ["Example Trail"],
                        "trailhead": "Example Trailhead",
                        "route_label": "A-route",
                        "new_official_miles": 1.2,
                        "estimated_total_on_foot_miles": 2.4,
                        "ascent_ft": 300,
                        "park": {
                            "trailhead": "Example Trailhead",
                            "lat": 43.001,
                            "lon": -116.001,
                            "can_park": True,
                            "parking_confidence": "test",
                            "facility_status": "Open",
                        },
                    }
                ],
            }
        ],
    }
    plan = {
        "source_datasets": {},
        "route_menu": {
            "all_candidates": [
                {
                    "candidate_id": "example-route",
                    "trail_names": ["Example Trail"],
                    "segments": [{"seg_id": 1, "trail_name": "Example Trail"}],
                    "route_orientation": {"direction": "forward"},
                    "direction_validation": {"planned_traversal_direction": {}},
                    "trailhead_access": {
                        "outbound_path_coordinates": [
                            [-116.001, 43.001],
                            [-116.002, 43.002],
                        ],
                        "return_path_coordinates": [
                            [-116.003, 43.003],
                            [-116.001, 43.001],
                        ],
                    },
                    "return_to_car": {
                        "path_coordinates": [
                            [-116.003, 43.003],
                            [-116.001, 43.001],
                        ]
                    },
                }
            ]
        },
    }
    official_index = {
        1: {
            "seg_id": 1,
            "trail_name": "Example Trail",
            "direction": "both",
            "coordinates": [(-116.002, 43.002), (-116.003, 43.003)],
        }
    }
    return runbook, plan, official_index


def test_build_plan_map_data_includes_route_drive_and_trailhead_layers():
    exporter = load_map_exporter()
    runbook, plan, official_index = sample_inputs()

    data = exporter.build_plan_map_data(runbook, plan, official_index)

    assert data["profile_name"] == "test-profile"
    assert len(data["days"]) == 1
    assert len(data["feature_collections"]["on_foot_routes"]["features"]) == 1
    assert len(data["feature_collections"]["official_segments"]["features"]) == 1
    assert len(data["feature_collections"]["drives"]["features"]) == 1
    assert len(data["feature_collections"]["trailheads"]["features"]) == 1
    assert data["feature_collections"]["on_foot_routes"]["features"][0]["properties"]["title"] == "Example Trail"
    assert data["map_validation"]["route_count"] == 1


def test_render_html_embeds_map_payload_and_sidebar_controls():
    exporter = load_map_exporter()
    runbook, plan, official_index = sample_inputs()
    data = exporter.build_plan_map_data(runbook, plan, official_index)

    rendered = exporter.render_html(data)

    assert "const PLAN_MAP_DATA =" in rendered
    assert "2026 Personal Plan Map" in rendered
    assert "Official" in rendered
    assert "Full route" in rendered
    assert "Drives" in rendered
    assert "drawDirectionArrows" in rendered
    assert "drawRouteCues" in rendered
    assert "path-marker" in rendered
    assert "turn markers" in rendered
    assert "Parking / start trailhead" in rendered
    assert "parking-marker" in rendered
    assert "Example Trail" in rendered
    assert ".app {\n      display: grid;\n      grid-template-columns: minmax(360px, 430px) minmax(0, 1fr);\n      height: 100vh;" in rendered
    assert ".day-list {\n      flex: 1 1 auto;\n      min-height: 0;" in rendered
    assert "grid-template-rows: minmax(260px, 46vh) minmax(0, 1fr);" in rendered
    assert "#map {\n      position: absolute;\n      inset: 0;\n      overflow: hidden;" in rendered
    assert ".leaflet-container {\n      height: 100%;" in rendered
    assert "World_Topo_Map/MapServer/tile/{z}/{y}/{x}" in rendered
    assert "ResizeObserver" in rendered
    assert "map.invalidateSize(false)" in rendered
