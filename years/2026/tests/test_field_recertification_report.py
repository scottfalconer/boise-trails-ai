import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_recertification_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_recertification_report", MODULE_PATH)
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
                "door_to_door_minutes_p90": 95,
                "validation": {"passed": True},
            },
            {
                "outing_id": "route-b",
                "label": "B",
                "trailhead": "Beta TH",
                "segment_ids": ["103"],
                "door_to_door_minutes_p75": 50,
                "door_to_door_minutes_p90": 62,
                "validation": {"passed": True},
            },
        ],
    }


def calendar_assignment():
    return {
        "challenge_window": {"start": "2026-06-18", "end": "2026-06-19"},
        "assignments": [
            {
                "date": "2026-06-18",
                "field_day": {"segment_summary": {"segment_ids": [101, 102]}, "p90_minutes": 95},
            },
            {
                "date": "2026-06-19",
                "field_day": {"segment_summary": {"segment_ids": [103]}, "p90_minutes": 62},
            },
        ],
    }


def test_recertification_passes_when_remaining_optimizer_covers_remaining_segments():
    module = load_module()

    report = module.build_recertification_report(
        field_tool_data(),
        official_geojson(),
        {"completed_segment_ids": ["101", "102"], "completed_outing_ids": ["route-a"], "as_of_date": "2026-06-18"},
        calendar=calendar_assignment(),
        optimizer=lambda remaining_ids, config: {
            "success": True,
            "target_segment_ids": remaining_ids,
            "covered_segment_count": len(remaining_ids),
            "missing_segment_ids": [],
            "field_day_count": 1,
        },
    )

    assert report["status"] == "passed"
    assert report["summary"]["remaining_segment_count"] == 1
    assert report["summary"]["remaining_full_completion_feasible"] is True
    assert report["calendar_reassignment"]["remaining_scheduled_day_count"] == 1
    assert report["calendar_reassignment"]["remaining_available_date_count"] == 1
    assert report["optimizer"]["target_segment_ids"] == [103]
    assert report["gates"][0] == {
        "gate": "certified_baseline_loaded",
        "passed": True,
        "detail": "baseline status passed",
    }


def test_recertification_can_pass_after_missed_segment_when_optimizer_recovers_it():
    module = load_module()

    report = module.build_recertification_report(
        field_tool_data(),
        official_geojson(),
        {"completed_segment_ids": ["101"], "completed_outing_ids": ["route-a"], "missed_segment_ids": ["102"]},
        calendar=calendar_assignment(),
        optimizer=lambda remaining_ids, config: {"success": True, "target_segment_ids": remaining_ids},
    )

    assert report["status"] == "passed"
    assert report["summary"]["remaining_full_completion_feasible"] is True
    assert "102" in report["progress"]["remaining_segment_ids"]
    assert report["progress"]["missing_remaining_segment_ids"] == []


def test_recertification_fails_when_remaining_calendar_has_no_future_capacity():
    module = load_module()

    report = module.build_recertification_report(
        field_tool_data(),
        official_geojson(),
        {"completed_segment_ids": ["101", "102"], "completed_outing_ids": ["route-a"], "as_of_date": "2026-06-19"},
        calendar=calendar_assignment(),
        optimizer=lambda remaining_ids, config: {
            "success": True,
            "target_segment_ids": remaining_ids,
            "covered_segment_count": len(remaining_ids),
            "missing_segment_ids": [],
            "field_day_count": 1,
        },
    )

    assert report["status"] == "failed"
    assert report["calendar_reassignment"]["remaining_scheduled_day_count"] == 1
    assert report["calendar_reassignment"]["remaining_available_date_count"] == 0
    assert any(gate["gate"] == "remaining_calendar_capacity" and gate["passed"] is False for gate in report["gates"])


def test_cli_writes_recertification_report_with_stub_optimizer(tmp_path):
    module = load_module()
    field_tool = tmp_path / "field-tool-data.json"
    official = tmp_path / "official.geojson"
    progress = tmp_path / "progress.json"
    output_json = tmp_path / "recert.json"
    output_md = tmp_path / "recert.md"
    field_tool.write_text(json.dumps(field_tool_data()), encoding="utf-8")
    official.write_text(json.dumps(official_geojson()), encoding="utf-8")
    progress.write_text(json.dumps({"completed_outing_ids": []}), encoding="utf-8")

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
            "--skip-heavy-optimizer",
        ]
    )

    assert result == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["status"] == "not_run"
    assert output_md.read_text(encoding="utf-8").startswith("# Field Recertification Report")


def test_cli_default_uses_fast_certificate_check_without_heavy_optimizer(tmp_path):
    module = load_module()
    field_tool = tmp_path / "field-tool-data.json"
    official = tmp_path / "official.geojson"
    output_json = tmp_path / "recert.json"
    output_md = tmp_path / "recert.md"
    field_tool.write_text(json.dumps(field_tool_data()), encoding="utf-8")
    official.write_text(json.dumps(official_geojson()), encoding="utf-8")

    result = module.main(
        [
            "--field-tool-data-json",
            str(field_tool),
            "--official-geojson",
            str(official),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert result == 0
    assert written["status"] == "passed"
    assert written["optimizer"]["method"] == "certified_baseline_plus_remaining_menu_coverage"


def test_default_recertification_can_pass_after_validated_progress_matches_active_packet():
    module = load_module()
    active_field_tool = field_tool_data()
    active_field_tool["routes"] = [active_field_tool["routes"][1]]
    active_field_tool["progress"] = {
        "completed_segment_ids_at_export": ["101", "102"],
        "blocked_segment_ids_at_export": [],
    }

    report = module.build_recertification_report(
        active_field_tool,
        official_geojson(),
        {"completed_segment_ids": ["101", "102"], "as_of_date": "2026-06-18"},
        calendar=calendar_assignment(),
    )

    assert report["status"] == "passed"
    assert report["summary"]["remaining_segment_count"] == 1
    assert report["optimizer"]["method"] == "active_progress_field_menu_certificate"
    assert report["optimizer"]["target_segment_ids"] == [103]
    assert report["optimizer"]["field_day_count"] == 1


def test_cli_default_uses_exported_validated_progress_from_active_packet(tmp_path):
    module = load_module()
    active_field_tool = field_tool_data()
    active_field_tool["routes"] = [active_field_tool["routes"][1]]
    active_field_tool["progress"] = {
        "completed_segment_ids_at_export": ["101", "102"],
        "blocked_segment_ids_at_export": [],
    }
    field_tool = tmp_path / "field-tool-data.json"
    official = tmp_path / "official.geojson"
    calendar = tmp_path / "calendar.json"
    output_json = tmp_path / "recert.json"
    output_md = tmp_path / "recert.md"
    field_tool.write_text(json.dumps(active_field_tool), encoding="utf-8")
    official.write_text(json.dumps(official_geojson()), encoding="utf-8")
    calendar.write_text(json.dumps(calendar_assignment()), encoding="utf-8")

    result = module.main(
        [
            "--field-tool-data-json",
            str(field_tool),
            "--official-geojson",
            str(official),
            "--calendar-json",
            str(calendar),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert result == 0
    assert written["status"] == "passed"
    assert written["summary"]["completed_segment_count"] == 2
    assert written["optimizer"]["method"] == "active_progress_field_menu_certificate"


def test_default_recertification_requires_heavy_optimizer_after_progress_diverges():
    module = load_module()

    report = module.build_recertification_report(
        field_tool_data(),
        official_geojson(),
        {"completed_outing_ids": ["route-a"]},
        calendar=calendar_assignment(),
    )

    assert report["status"] == "failed"
    assert report["optimizer"]["reason"] == "heavy_optimizer_required_after_progress_change"
    assert any(
        gate["gate"] == "remaining_optimizer_solution" and gate["passed"] is False
        for gate in report["gates"]
    )
