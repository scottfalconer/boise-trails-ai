import argparse
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "harlow_h1_access_cue_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("harlow_h1_access_cue_review", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def line_feature(name, coords, **props):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"TrailName": name, "Name": name, **props},
    }


def arcgis_feature(name, coords, **props):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "Name": name,
            "Hike_Bike": "Y",
            "Visible": "Y",
            "Closed": "N",
            "Seasonal": "N",
            "TrailNum": "",
            "Owner": "Avimor",
            **props,
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_opaque_osm_connector_resolves_to_nearby_open_named_trail():
    module = load_module()

    connector = line_feature(
        "OSM footway connector 1",
        [[-116.0, 43.0], [-116.0001, 43.0001]],
        source="openstreetmap",
        highway="footway",
        access=None,
        foot=None,
    )
    arcgis = [
        arcgis_feature(
            "McLeod Way Greenbelt",
            [[-116.0, 43.0], [-116.0002, 43.0002]],
            TrailNum="A34",
        )
    ]

    review = module.connector_source_review(
        "OSM footway connector 1",
        connector_index={"OSM footway connector 1": connector},
        arcgis_open_features=arcgis,
        arcgis_by_name=module.arcgis_name_index(arcgis),
    )

    assert review["status"] == "passed_resolved_to_nearby_open_named_trail"
    assert review["can_cue_without_opaque_osm_id"] is True
    assert review["field_cue_name"] == "McLeod Way Greenbelt A34"


def test_build_report_clears_access_gate_but_keeps_promotion_and_recertification_blockers(tmp_path):
    module = load_module()

    h1_ids = ["1626", "1657", "1661", "1662", "1687", "1688", "1689", "1696", "1704", "1705", "1706", "1707", "1708"]
    h1_report = {
        "candidate_id": "H1-avimor-native-harlow-spring-loop",
        "current_scope": {"on_foot_miles": 34.0, "p75_minutes": 991, "p90_minutes": 1117},
        "parking_source_sync": {"status": "already_synced_in_field_packet_source"},
        "repaired_candidate": {
            "official_segment_ids": h1_ids,
            "official_miles": 7.3,
            "track_miles": 9.64,
            "dem_pricing": {"time_estimates_minutes": {"door_to_door_p75": 289, "door_to_door_p90": 324}},
            "link_rows": [
                {
                    "to_segment_id": "1687",
                    "to_segment_name": "Twisted Spring 1",
                    "link_track_miles": 0.1,
                    "connector_names": ["OSM footway connector 1", "Twisted Spring Trail - #8"],
                    "official_repeat_segment_ids": ["1687"],
                    "official_repeat_miles": 0.01,
                },
                {
                    "to_segment_id": "1657",
                    "to_segment_name": "Shooting Range 1",
                    "link_track_miles": 0.1,
                    "connector_names": ["North Smokeys Draw Place", "Ricochet - #2", "Shooting Range - #5"],
                    "official_repeat_segment_ids": [],
                    "official_repeat_miles": 0,
                },
                {
                    "to_segment_id": "1661",
                    "to_segment_name": "Spring Creek 1",
                    "link_track_miles": 0.5,
                    "connector_names": ["OSM path connector 2", "Whistling Pig - #3"],
                    "official_repeat_segment_ids": ["1688", "1689"],
                    "official_repeat_miles": 0.21,
                },
                {
                    "to_segment_id": "1704",
                    "to_segment_name": "Harlow's Hollows 4",
                    "link_track_miles": 0.6,
                    "connector_names": ["OSM path connector 3", "The Wall - #29"],
                    "official_repeat_segment_ids": ["1704"],
                    "official_repeat_miles": 0.05,
                },
                {
                    "to_segment_id": "return_to_car",
                    "to_segment_name": "Return to car",
                    "link_track_miles": 0.9,
                    "connector_names": ["OSM footway connector 1", "Spring Creek - #9"],
                    "official_repeat_segment_ids": ["1626"],
                    "official_repeat_miles": 0.34,
                },
            ],
        },
    }
    field_tool_data = {
        "routes": [
            {"label": "FD27A", "segment_ids": ["1661"], "official_miles": 0.08},
            {"label": "FD30A", "segment_ids": ["1708", "1626", "1657", "1687", "1688", "1689"], "official_miles": 2.61},
            {"label": "FD27B", "segment_ids": ["1662"], "official_miles": 2.34},
            {"label": "FD27C", "segment_ids": ["1696"], "official_miles": 0.88},
            {"label": "FD24A", "segment_ids": ["1704", "1705", "1707", "1706"], "official_miles": 1.4},
        ]
    }
    connector_geojson = {
        "type": "FeatureCollection",
        "features": [
            line_feature("OSM footway connector 1", [[-116.0, 43.0], [-116.0001, 43.0001]], source="openstreetmap", highway="footway", access=None, foot=None),
            line_feature("OSM path connector 2", [[-116.01, 43.01], [-116.0101, 43.0101]], source="openstreetmap", highway="path", access=None, foot=None),
            line_feature("OSM path connector 3", [[-116.02, 43.02], [-116.0201, 43.0201]], source="openstreetmap", highway="path", access=None, foot=None),
            line_feature("Twisted Spring Trail - #8", [[-116.0, 43.0], [-116.001, 43.001]], source="openstreetmap", highway="path", access=None, foot="designated"),
            line_feature("North Smokeys Draw Place", [[-116.002, 43.002], [-116.003, 43.003]], source="openstreetmap", highway="residential", access=None, foot=None),
            line_feature("Ricochet - #2", [[-116.004, 43.004], [-116.005, 43.005]], source="openstreetmap", highway="path", access=None, foot="designated"),
            line_feature("Shooting Range - #5", [[-116.006, 43.006], [-116.007, 43.007]], source="openstreetmap", highway="path", access=None, foot="designated"),
            line_feature("Whistling Pig - #3", [[-116.01, 43.01], [-116.011, 43.011]], source="openstreetmap", highway="path", access=None, foot="designated"),
            line_feature("The Wall - #29", [[-116.02, 43.02], [-116.021, 43.021]], source="openstreetmap", highway="path", access=None, foot="designated"),
            line_feature("Spring Creek - #9", [[-116.0, 43.0], [-116.0003, 43.0003]], source="openstreetmap", highway="path", access=None, foot="designated"),
        ],
    }
    arcgis_geojson = {
        "type": "FeatureCollection",
        "features": [
            arcgis_feature("McLeod Way Greenbelt", [[-116.0, 43.0], [-116.0002, 43.0002]], TrailNum="A34"),
            arcgis_feature("Twisted Spring Trail", [[-116.0, 43.0], [-116.001, 43.001]], TrailNum="A8"),
            arcgis_feature("Ricochet", [[-116.004, 43.004], [-116.005, 43.005]], TrailNum="A2"),
            arcgis_feature("Shooting Range", [[-116.006, 43.006], [-116.007, 43.007]], TrailNum="A5"),
            arcgis_feature("Whistling Pig", [[-116.01, 43.01], [-116.011, 43.011]], TrailNum="A3"),
            arcgis_feature("The Wall", [[-116.02, 43.02], [-116.021, 43.021]], TrailNum="A29", Seasonal="Y"),
            arcgis_feature("Spring Creek", [[-116.0, 43.0], [-116.0003, 43.0003]], TrailNum="A9"),
        ],
    }

    args = argparse.Namespace(
        h1_audit_json=write_json(tmp_path / "h1.json", h1_report),
        field_tool_data_json=write_json(tmp_path / "field.json", field_tool_data),
        connector_geojson=write_json(tmp_path / "connectors.json", connector_geojson),
        avimor_trails_layer_url="https://example.invalid/layer",
    )

    report = module.build_report(args, arcgis_geojson=arcgis_geojson)

    assert report["promotion_readiness"]["status"] == "access_gate_clear_keep_unpromoted"
    assert report["promotion_readiness"]["remaining_blockers_after_access_review"] == [
        "needs_field_packet_route_card_promotion",
        "needs_field_packet_recertification",
    ]
    assert report["h1_replacement_segment_set_diff"]["missing_ids"] == []
    assert report["h1_replacement_segment_set_diff"]["extra_ids"] == []
    assert any(row["status"] == "passed_with_day_of_seasonal_condition_check" for row in report["leg_by_leg_cueability"])


def test_private_or_no_foot_connector_blocks_access_review():
    module = load_module()

    connector = line_feature(
        "OSM path connector bad",
        [[-116.0, 43.0], [-116.0001, 43.0001]],
        source="openstreetmap",
        highway="path",
        access="private",
        foot=None,
    )
    arcgis = [arcgis_feature("Open Trail", [[-116.0, 43.0], [-116.0002, 43.0002]])]

    review = module.connector_source_review(
        "OSM path connector bad",
        connector_index={"OSM path connector bad": connector},
        arcgis_open_features=arcgis,
        arcgis_by_name=module.arcgis_name_index(arcgis),
    )

    assert review["status"] == "blocked_private_or_no_foot"
    assert review["unsafe_reasons"] == ["access=private"]
