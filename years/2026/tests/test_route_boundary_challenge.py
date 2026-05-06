import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_boundary_challenge.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_boundary_challenge", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_segments():
    return {
        "features": [
            {"properties": {"segId": 1, "segName": "One", "LengthFt": 5280}},
            {"properties": {"segId": 2, "segName": "Two", "LengthFt": 5280}},
            {"properties": {"segId": 3, "segName": "Three", "LengthFt": 5280}},
        ]
    }


def map_data():
    return {
        "route_cues": {
            "current-a": {
                "time_estimates_minutes": {"door_to_door_p75": 80, "moving_effort_p75": 50},
                "segments": [{"ascent_ft": 400, "descent_ft": 100, "grade_adjusted_miles": 1.8, "elevation_source": "dem"}],
            },
            "current-b": {
                "time_estimates_minutes": {"door_to_door_p75": 90, "moving_effort_p75": 60},
                "segments": [{"ascent_ft": 300, "descent_ft": 150, "grade_adjusted_miles": 1.7, "elevation_source": "dem"}],
            },
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "A",
                "components": [
                    {
                        "candidate_id": "current-a",
                        "segment_ids": [1],
                        "official_miles": 1.0,
                        "on_foot_miles": 3.0,
                        "total_minutes": 80,
                    }
                ],
            },
            {
                "package_number": 2,
                "block_name": "B",
                "components": [
                    {
                        "candidate_id": "current-b",
                        "segment_ids": [2],
                        "official_miles": 1.0,
                        "on_foot_miles": 4.0,
                        "total_minutes": 90,
                    }
                ],
            },
        ],
    }


def test_build_report_finds_better_generated_boundary_combo(tmp_path):
    module = load_module()
    source_path = tmp_path / "candidates.json"
    source_path.write_text(
        """
{
  "candidate_index": {
    "better": {
      "candidate_id": "better",
      "segment_ids": [1, 2],
      "official_new_miles": 2.0,
      "estimated_total_on_foot_miles": 4.0,
      "total_minutes": 110,
      "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 95, "route_finding_penalty": 12},
      "effort": {"ascent_ft": 500, "descent_ft": 300, "grade_adjusted_miles": 3.1, "elevation_source": "dem"},
      "trailhead": {"name": "Better TH"},
      "trail_names": ["Better"]
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(map_data(), official_segments(), [source_path], [1, 2], max_routes=3)

    assert report["summary"]["generated_combo_beats_current"] is True
    assert report["best"]["by_on_foot_miles"]["candidate_ids"] == ["better"]
    comparisons = {row["metric"]: row for row in report["comparisons"]}
    assert comparisons["on_foot_miles"]["status"] == "better_generated_combo_found"
    assert comparisons["door_to_door_p75_minutes"]["status"] == "better_generated_combo_found"
    assert comparisons["ascent_ft"]["status"] == "current_not_meaningfully_beaten"


def test_build_report_keeps_current_when_generated_combo_is_slower_and_longer(tmp_path):
    module = load_module()
    source_path = tmp_path / "candidates.json"
    source_path.write_text(
        """
{
  "candidate_index": {
    "worse": {
      "candidate_id": "worse",
      "segment_ids": [1, 2],
      "official_new_miles": 2.0,
      "estimated_total_on_foot_miles": 8.0,
      "total_minutes": 220,
      "time_estimates_minutes": {"door_to_door_p75": 240, "moving_effort_p75": 180, "route_finding_penalty": 12},
      "effort": {"ascent_ft": 1200, "descent_ft": 300, "grade_adjusted_miles": 9.5, "elevation_source": "dem"},
      "trailhead": {"name": "Worse TH"},
      "trail_names": ["Worse"]
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(map_data(), official_segments(), [source_path], [1, 2], max_routes=3)

    assert report["summary"]["generated_combo_beats_current"] is False
    comparisons = {row["metric"]: row for row in report["comparisons"]}
    assert comparisons["on_foot_miles"]["status"] == "current_not_meaningfully_beaten"
    assert comparisons["door_to_door_p75_minutes"]["status"] == "current_not_meaningfully_beaten"


def test_build_report_does_not_count_draft_candidate_as_better(tmp_path):
    module = load_module()
    source_path = tmp_path / "candidates.json"
    source_path.write_text(
        """
{
  "candidate_index": {
    "draft-better": {
      "candidate_id": "draft-better",
      "route_status": "draft",
      "segment_ids": [1, 2],
      "official_new_miles": 2.0,
      "estimated_total_on_foot_miles": 4.0,
      "total_minutes": 110,
      "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 95, "route_finding_penalty": 12},
      "effort": {"ascent_ft": 500, "descent_ft": 300, "grade_adjusted_miles": 4.3, "elevation_source": "dem"},
      "trailhead": {"name": "Draft TH"},
      "trail_names": ["Draft"]
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(map_data(), official_segments(), [source_path], [1, 2], max_routes=3)

    assert report["summary"]["candidate_pool_excludes_draft_routes"] is True
    assert report["summary"]["generated_combo_beats_current"] is False
    assert "draft-better" not in {
        candidate_id
        for row in report["top_covering_combos_by_on_foot"]
        for candidate_id in row["candidate_ids"]
    }
