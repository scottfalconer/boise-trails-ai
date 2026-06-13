import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pull_official_challenge_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pull_official_challenge_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature(seg_id, name, length_ft, activity_type="both", direction="both", coords=None):
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords or [[-116.0, 43.0], [-116.01, 43.01]],
        },
        "properties": {
            "segId": seg_id,
            "segName": name,
            "LengthFt": length_ft,
            "direction": direction,
            "specInst": "",
            "activity_type": activity_type,
        },
    }


def payload(segments):
    return {
        "lastUpdatedUTC": "2026-06-11T01:45:43",
        "masterTrails": [
            {"masterTrailId": 1, "masterTrailName": "Foot Trail", "activity_type": "both"},
            {"masterTrailId": 2, "masterTrailName": "Bike Trail", "activity_type": "bike"},
        ],
        "trailSegments": segments,
    }


def test_official_foot_summary_filters_bike_only_segments():
    module = load_module()
    data = payload(
        [
            feature(1, "Foot One", 5280, "both"),
            feature(2, "Foot Two", 2640, "foot", direction="ascent"),
            feature(3, "Bike One", 5280, "bike"),
        ]
    )

    summary = module.official_foot_summary(data)

    assert summary["official_foot_trails_count"] == 1
    assert summary["official_foot_segments_count"] == 2
    assert summary["official_foot_distance_miles"] == 1.5
    assert summary["official_foot_direction_counts"] == {"ascent": 1, "both": 1}
    assert summary["bike_only_segments_count"] == 1


def test_drift_report_identifies_id_geometry_and_field_packet_impacts(tmp_path):
    module = load_module()
    old_collection = module.official_foot_segments_collection(
        payload(
            [
                feature(10, "Removed", 5280),
                feature(11, "Changed", 5280, coords=[[-116.0, 43.0], [-116.1, 43.1]]),
            ]
        )
    )
    new_collection = module.official_foot_segments_collection(
        payload(
            [
                feature(11, "Changed", 6000, coords=[[-116.0, 43.0], [-116.2, 43.2]]),
                feature(12, "Added", 2640),
            ]
        )
    )
    field_tool_data = {
        "routes": [
            {
                "label": "A",
                "route_name": "Old route",
                "field_readiness_status": "field_ready",
                "official_miles": 1.0,
                "on_foot_miles": 2.0,
                "segment_ids": ["10", "11"],
            }
        ]
    }

    report = module.build_drift_report(
        old_collection,
        new_collection,
        old_pull_dir=tmp_path / "old",
        new_pull_dir=tmp_path / "new",
        field_tool_data=field_tool_data,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert report["summary"]["delta_foot_segments"] == 0
    assert [row["segId"] for row in report["removed_foot_segments"]] == [10]
    assert [row["segId"] for row in report["added_foot_segments"]] == [12]
    assert report["changed_foot_segments"][0]["segId"] == 11
    assert report["changed_foot_segments"][0]["property_changes"]["LengthFt"] == {"old": 5280, "new": 6000}
    assert report["changed_foot_segments"][0]["geometry_changed"] is True
    impacts = report["active_field_packet_impacts"]
    assert impacts["old_claimed_segments_no_longer_official"] == ["10"]
    assert impacts["new_official_segments_not_claimed_by_active_packet"] == ["12"]
    assert impacts["routes_claiming_removed_segments"]["10"][0]["label"] == "A"
    assert impacts["routes_claiming_changed_segments"]["11"][0]["route_name"] == "Old route"
