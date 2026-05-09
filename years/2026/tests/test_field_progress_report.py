import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_progress_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_progress_report", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"segId": 101}, "geometry": None},
            {"type": "Feature", "properties": {"segId": 102}, "geometry": None},
            {"type": "Feature", "properties": {"segId": 103}, "geometry": None},
        ],
    }


def field_tool_data():
    return {
        "schema": "boise_trails_field_tool_data_v1",
        "time_filters_minutes": [60, 90, 120],
        "certified_baseline": {
            "status": "passed",
            "profile_id": "test-profile",
            "official_segment_count": 3,
            "covered_segment_count": 3,
            "missing_segment_count": 0,
        },
        "routes": [
            {
                "outing_id": "route-a",
                "label": "A",
                "trailhead": "Alpha TH",
                "segment_ids": ["101", "102"],
                "door_to_door_minutes_p75": 80,
                "door_to_door_minutes_p90": 115,
                "official_miles": 2.0,
                "on_foot_miles": 3.0,
                "trails": ["Alpha Trail"],
                "validation": {"passed": True},
            },
            {
                "outing_id": "route-b",
                "label": "B",
                "trailhead": "Beta TH",
                "segment_ids": ["103"],
                "door_to_door_minutes_p75": 50,
                "door_to_door_minutes_p90": 62,
                "official_miles": 1.0,
                "on_foot_miles": 1.5,
                "trails": ["Beta Trail"],
                "validation": {"passed": True},
            },
        ],
    }


def test_completed_outings_are_provisional_until_segment_evidence_is_supplied():
    module = load_module()

    report = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"completed_outing_ids": ["route-a"]},
    )

    assert report["summary"]["official_segment_count"] == 3
    assert report["summary"]["completed_segment_count"] == 0
    assert report["summary"]["completed_outing_count"] == 0
    assert report["summary"]["provisional_completed_outing_count"] == 1
    assert report["summary"]["provisional_completed_segment_count"] == 2
    assert report["summary"]["remaining_segment_count"] == 3
    assert report["summary"]["remaining_coverage_preserved"] is True
    assert report["completed_segment_ids"] == []
    assert report["completed_outing_ids"] == []
    assert report["provisional_completed_outing_ids"] == ["route-a"]
    assert report["provisional_completed_segment_ids"] == ["101", "102"]
    assert report["remaining_segment_ids"] == ["101", "102", "103"]
    assert report["private_state_patch"]["completed_segment_ids"] == []
    assert report["today_options_by_minutes"]["60"][0]["outing_id"] == "route-b"


def test_validated_segments_roll_up_to_completed_outings():
    module = load_module()

    report = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"completed_segment_ids": ["101", "102"]},
    )

    assert report["summary"]["completed_outing_count"] == 1
    assert report["completed_outing_ids"] == ["route-a"]
    assert report["outing_statuses"]["route-a"]["status"] == "completed_by_segments"
    assert report["outing_statuses"]["route-a"]["remaining_segment_ids"] == []
    assert all(row["outing_id"] != "route-a" for row in report["today_options_by_minutes"]["120"])


def test_default_progress_uses_exported_validated_segment_state():
    module = load_module()
    data = field_tool_data()
    data["routes"] = [data["routes"][1]]
    data["progress"] = {
        "completed_segment_ids_at_export": ["101", "102"],
        "blocked_segment_ids_at_export": [],
    }

    report = module.build_progress_report(data, official_geojson())

    assert report["summary"]["completed_segment_count"] == 2
    assert report["summary"]["remaining_segment_count"] == 1
    assert report["summary"]["available_remaining_segment_count"] == 1
    assert report["summary"]["missing_remaining_segment_count"] == 0
    assert report["completed_segment_ids"] == ["101", "102"]
    assert report["remaining_segment_ids"] == ["103"]


def test_blocked_only_empty_outing_is_inactive_not_completed():
    module = load_module()

    report = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"completed_segment_ids": ["101"], "blocked_segment_ids": ["102"]},
    )

    assert report["completed_outing_ids"] == []
    assert report["inactive_outing_ids"] == ["route-a"]
    assert report["outing_statuses"]["route-a"]["status"] == "inactive_no_remaining_new_credit"
    assert report["outing_statuses"]["route-a"]["inactive_reason"] == "blocked_or_removed_segments_remain"


def test_validated_completed_segments_remove_progress_but_missed_segments_do_not():
    module = load_module()

    report = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {
            "completed_segment_ids": ["101", "102"],
            "completed_outing_ids": ["route-a"],
            "missed_segment_ids": ["102"],
        },
    )

    assert report["completed_segment_ids"] == ["101"]
    assert report["remaining_segment_ids"] == ["102", "103"]
    assert report["summary"]["remaining_coverage_preserved"] is True
    assert report["summary"]["missing_remaining_segment_count"] == 0
    assert report["missing_remaining_segment_ids"] == []
    assert report["private_state_patch"]["completed_segment_ids"] == [101]


def test_blocked_segments_suppress_routes_that_traverse_them():
    module = load_module()

    report = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"blocked_segment_ids": ["102"]},
    )

    assert report["blocked_segment_ids"] == ["102"]
    assert all(row["outing_id"] != "route-a" for row in report["today_options_by_minutes"]["120"])
    assert report["summary"]["missing_remaining_segment_count"] == 1
    assert report["missing_remaining_segment_ids"] == ["101"]


def test_hard_stop_mode_filters_by_p90_not_p75():
    module = load_module()

    normal = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"time_budget_mode": "normal"},
    )
    hard_stop = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"time_budget_mode": "hard_stop"},
    )

    assert any(row["outing_id"] == "route-a" for row in normal["today_options_by_minutes"]["90"])
    assert all(row["outing_id"] != "route-a" for row in hard_stop["today_options_by_minutes"]["90"])


def test_cli_writes_progress_report_and_state_patch(tmp_path):
    module = load_module()
    field_tool = tmp_path / "field-tool-data.json"
    official = tmp_path / "official.geojson"
    progress = tmp_path / "progress.json"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    field_tool.write_text(json.dumps(field_tool_data()), encoding="utf-8")
    official.write_text(json.dumps(official_geojson()), encoding="utf-8")
    progress.write_text(json.dumps({"completed_outing_ids": ["route-a"]}), encoding="utf-8")

    result = module.main(
        [
            "--field-tool-data-json",
            str(field_tool),
            "--official-geojson",
            str(official),
            "--progress-json",
            str(progress),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert result == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["summary"]["remaining_coverage_preserved"] is True
    assert output_md.read_text(encoding="utf-8").startswith("# Field Progress Report")
