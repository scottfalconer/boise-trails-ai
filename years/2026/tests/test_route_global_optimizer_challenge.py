import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_global_optimizer_challenge.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_global_optimizer_challenge", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_segments():
    return {
        "features": [
            {"properties": {"segId": 1, "segName": "One", "LengthFt": 5280}},
            {"properties": {"segId": 2, "segName": "Two", "LengthFt": 5280}},
        ]
    }


def map_data():
    return {
        "route_cues": {
            "current-a": {
                "time_estimates_minutes": {"door_to_door_p75": 100, "moving_effort_p75": 80},
                "segments": [{"ascent_ft": 500, "descent_ft": 100, "grade_adjusted_miles": 2.0, "elevation_source": "dem"}],
            },
            "current-b": {
                "time_estimates_minutes": {"door_to_door_p75": 100, "moving_effort_p75": 80},
                "segments": [{"ascent_ft": 500, "descent_ft": 100, "grade_adjusted_miles": 2.0, "elevation_source": "dem"}],
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
                        "total_minutes": 100,
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
                        "on_foot_miles": 3.0,
                        "total_minutes": 100,
                    }
                ],
            },
        ],
    }


def test_global_optimizer_finds_dominant_complete_cover(tmp_path):
    module = load_module()
    source_path = tmp_path / "candidates.json"
    source_path.write_text(
        """
{
  "candidate_index": {
    "better": {
      "candidate_id": "better",
      "route_status": "graph_validated",
      "segment_ids": [1, 2],
      "official_new_miles": 2.0,
      "estimated_total_on_foot_miles": 4.0,
      "total_minutes": 120,
      "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 95, "route_finding_penalty": 8},
      "effort": {"ascent_ft": 700, "descent_ft": 300, "grade_adjusted_miles": 3.0, "elevation_source": "dem"},
      "trailhead": {"name": "Better TH"},
      "trail_names": ["Better"]
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(map_data(), official_segments(), [source_path])

    assert report["summary"]["global_optimizer_beats_current"] is True
    assert report["summary"]["dominant_solution_count"] >= 1
    by_distance = next(solution for solution in report["solutions"] if solution["objective"] == "on_foot_miles")
    assert by_distance["candidate_ids"] == ["better"]


def test_global_optimizer_excludes_draft_route(tmp_path):
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
      "total_minutes": 120,
      "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 95, "route_finding_penalty": 8},
      "effort": {"ascent_ft": 700, "descent_ft": 300, "grade_adjusted_miles": 3.0, "elevation_source": "dem"},
      "trailhead": {"name": "Draft TH"},
      "trail_names": ["Draft"]
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(map_data(), official_segments(), [source_path])

    assert report["summary"]["global_optimizer_beats_current"] is False
    assert all("draft-better" not in solution.get("candidate_ids", []) for solution in report["solutions"])


def test_global_optimizer_excludes_candidate_without_return_path_geometry(tmp_path):
    module = load_module()
    source_path = tmp_path / "candidates.json"
    source_path.write_text(
        """
{
  "candidate_index": {
    "gap-better": {
      "candidate_id": "gap-better",
      "route_status": "graph_validated",
      "segment_ids": [1, 2],
      "official_new_miles": 2.0,
      "estimated_total_on_foot_miles": 4.0,
      "total_minutes": 120,
      "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 95, "route_finding_penalty": 8},
      "effort": {"ascent_ft": 700, "descent_ft": 300, "grade_adjusted_miles": 3.0, "elevation_source": "dem"},
      "trailhead": {"name": "Better TH"},
      "trail_names": ["Better"],
      "return_to_car": {"strategy": "mapped_connector_loop", "connector_miles": 0.5}
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(map_data(), official_segments(), [source_path])

    assert report["summary"]["global_optimizer_beats_current"] is False
    assert all("gap-better" not in solution.get("candidate_ids", []) for solution in report["solutions"])
