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
                "official_miles": 2.0,
                "on_foot_miles": 3.0,
                "validation": {"passed": True},
            },
            {
                "outing_id": "route-b",
                "label": "B",
                "trailhead": "Beta TH",
                "segment_ids": ["103"],
                "door_to_door_minutes_p75": 50,
                "official_miles": 1.0,
                "on_foot_miles": 1.5,
                "validation": {"passed": True},
            },
        ],
    }


def test_progress_report_subtracts_completed_outings_and_preserves_remaining_feasibility():
    module = load_module()

    report = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"completed_outing_ids": ["route-a"]},
    )

    assert report["summary"]["official_segment_count"] == 3
    assert report["summary"]["completed_segment_count"] == 2
    assert report["summary"]["remaining_segment_count"] == 1
    assert report["summary"]["remaining_coverage_preserved"] is True
    assert report["completed_segment_ids"] == ["101", "102"]
    assert report["remaining_segment_ids"] == ["103"]
    assert report["private_state_patch"]["completed_segment_ids"] == [101, 102]
    assert report["today_options_by_minutes"]["60"][0]["outing_id"] == "route-b"


def test_missed_segments_are_not_counted_complete_even_when_the_outing_was_marked_done():
    module = load_module()

    report = module.build_progress_report(
        field_tool_data(),
        official_geojson(),
        {"completed_outing_ids": ["route-a"], "missed_segment_ids": ["102"]},
    )

    assert report["completed_segment_ids"] == ["101"]
    assert report["remaining_segment_ids"] == ["102", "103"]
    assert report["summary"]["remaining_coverage_preserved"] is False
    assert report["summary"]["missing_remaining_segment_count"] == 1
    assert report["missing_remaining_segment_ids"] == ["102"]
    assert report["private_state_patch"]["completed_segment_ids"] == [101]


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
