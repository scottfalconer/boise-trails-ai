import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_alternative_challenge.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_alternative_challenge", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_segments():
    return {
        "features": [
            {"properties": {"segId": 1, "LengthFt": 5280}},
            {"properties": {"segId": 2, "LengthFt": 10560}},
            {"properties": {"segId": 3, "LengthFt": 5280}},
        ]
    }


def test_normalize_candidate_accepts_assembled_route_fields():
    module = load_module()
    segment_miles = module.official_segment_miles(official_segments())
    item = {
        "candidate_id": "assembled-a",
        "official_new_miles": 3.0,
        "estimated_total_on_foot_miles": 4.5,
        "trailhead": {"name": "Trailhead A"},
        "segment_ids": [1, 2],
        "trail_names": ["Trail A"],
        "total_minutes": 90,
    }

    normalized = module.normalize_candidate(item, source_name="source", segment_miles=segment_miles)

    assert normalized["candidate_id"] == "assembled-a"
    assert normalized["official_miles"] == 3.0
    assert normalized["on_foot_miles"] == 4.5
    assert normalized["trailhead"] == "Trailhead A"
    assert normalized["ratio"] == 1.5


def test_normalize_candidate_carries_effort_and_p75_time_fields():
    module = load_module()
    segment_miles = module.official_segment_miles(official_segments())
    item = {
        "candidate_id": "assembled-a",
        "official_new_miles": 3.0,
        "estimated_total_on_foot_miles": 4.5,
        "segment_ids": [1, 2],
        "time_estimates_minutes": {
            "door_to_door_p75": 112,
            "moving_effort_p75": 80,
            "route_finding_penalty": 14,
        },
        "effort": {
            "ascent_ft": 1200,
            "descent_ft": 900,
            "grade_adjusted_miles": 5.4,
            "elevation_source": "dem",
        },
    }

    normalized = module.normalize_candidate(item, source_name="source", segment_miles=segment_miles)

    assert normalized["door_to_door_p75_minutes"] == 112
    assert normalized["moving_effort_p75_minutes"] == 80
    assert normalized["route_finding_penalty_minutes"] == 14
    assert normalized["ascent_ft"] == 1200
    assert normalized["descent_ft"] == 900
    assert normalized["grade_adjusted_miles"] == 5.4
    assert normalized["elevation_source"] == "dem"


def test_load_candidate_universe_prefers_rich_duplicate_candidate(tmp_path):
    module = load_module()
    segment_miles = module.official_segment_miles(official_segments())
    source_path = tmp_path / "candidates.json"
    source_path.write_text(
        """
{
  "routes": [
    {
      "candidate_id": "same",
      "segment_ids": [1],
      "official_miles": 1.0,
      "on_foot_miles": 2.0,
      "total_minutes": 30
    }
  ],
  "candidate_index": {
    "same": {
      "candidate_id": "same",
      "segment_ids": [1],
      "official_new_miles": 1.0,
      "estimated_total_on_foot_miles": 2.0,
      "total_minutes": 42,
      "time_estimates_minutes": {"door_to_door_p75": 42, "moving_effort_p75": 28},
      "effort": {"ascent_ft": 500, "grade_adjusted_miles": 2.5, "elevation_source": "dem"}
    }
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    candidates = module.load_candidate_universe([source_path], segment_miles)

    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == "same"
    assert candidates[0]["ascent_ft"] == 500
    assert candidates[0]["door_to_door_p75_minutes"] == 42


def test_selected_components_derives_effort_from_route_cues():
    module = load_module()
    segment_miles = module.official_segment_miles(official_segments())
    map_data = {
        "route_cues": {
            "current": {
                "raw_total_minutes": 50,
                "time_estimates_minutes": {
                    "door_to_door_p75": 80,
                    "moving_effort_p75": 60,
                    "route_finding_penalty": 10,
                },
                "segments": [
                    {"ascent_ft": 100, "descent_ft": 20, "grade_adjusted_miles": 0.7, "elevation_source": "dem"},
                    {"ascent_ft": 50, "descent_ft": 60, "grade_adjusted_miles": 0.4, "elevation_source": "dem"},
                ],
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Block",
                "components": [
                    {
                        "candidate_id": "current",
                        "segment_ids": [1],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                    }
                ],
            }
        ],
    }

    rows = module.selected_components(map_data, segment_miles)

    assert rows[0]["ascent_ft"] == 150
    assert rows[0]["descent_ft"] == 80
    assert rows[0]["grade_adjusted_miles"] == 1.1
    assert rows[0]["door_to_door_p75_minutes"] == 80
    assert rows[0]["moving_effort_p75_minutes"] == 60


def test_selected_components_keep_component_total_minutes_over_stale_route_cue_p75():
    module = load_module()
    segment_miles = module.official_segment_miles(official_segments())
    map_data = {
        "route_cues": {
            "current": {
                "time_estimates_minutes": {"door_to_door_p75": 515},
                "segments": [{"ascent_ft": 100, "descent_ft": 20, "grade_adjusted_miles": 0.7, "elevation_source": "dem"}],
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Block",
                "components": [
                    {
                        "candidate_id": "current",
                        "segment_ids": [1],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                        "total_minutes": 152,
                    }
                ],
            }
        ],
    }

    rows = module.selected_components(map_data, segment_miles)

    assert rows[0]["total_minutes"] == 152
    assert rows[0]["door_to_door_p75_minutes"] == 152


def test_effort_fields_from_route_cue_prefers_route_effort_over_stale_segments():
    module = load_module()
    cue = {
        "time_estimates_minutes": {"door_to_door_p75": 249, "moving_effort_p75": 215},
        "effort": {"ascent_ft": 3191, "descent_ft": 1540, "grade_adjusted_miles": 9.28, "elevation_source": "dem"},
        "segments": [
            {"ascent_ft": 9999, "descent_ft": 9999, "grade_adjusted_miles": 99.0, "elevation_source": "dem"}
        ],
    }

    fields = module.effort_fields_from_route_cue(cue)

    assert fields["ascent_ft"] == 3191
    assert fields["descent_ft"] == 1540
    assert fields["grade_adjusted_miles"] == 9.28
    assert fields["door_to_door_p75_minutes"] == 249
    assert fields["moving_effort_p75_minutes"] == 215


def test_challenge_target_finds_better_exact_candidate():
    module = load_module()
    target = {
        "label": "10",
        "candidate_id": "current",
        "block_name": "Block",
        "trailhead": "Old",
        "trail_names": ["A"],
        "segment_ids": [1, 2],
        "official_miles": 3.0,
        "on_foot_miles": 7.0,
        "ratio": 2.333,
        "overhead_miles": 4.0,
    }
    candidates = [
        {
            "source": "candidate-source",
            "candidate_id": "better",
            "trailhead": "New",
            "trail_names": ["A"],
            "segment_ids": [1, 2],
            "segment_count": 2,
            "official_miles": 3.0,
            "on_foot_miles": 5.5,
            "overhead_miles": 2.5,
            "ratio": 1.833,
            "total_minutes": 100,
        }
    ]

    challenge = module.challenge_target(target, candidates)

    assert challenge["challenge_status"] == "better_exact_candidate_found"
    assert challenge["best_exact"]["candidate_id"] == "better"
    assert "Replace" in challenge["recommendation"]


def test_challenge_targets_accept_custom_overhead_threshold():
    module = load_module()
    components = [
        {
            "candidate_id": "default-hidden",
            "official_miles": 10.0,
            "on_foot_miles": 15.1,
            "overhead_miles": 5.1,
            "ratio": 1.51,
        },
        {
            "candidate_id": "below-custom",
            "official_miles": 10.0,
            "on_foot_miles": 14.5,
            "overhead_miles": 4.5,
            "ratio": 1.45,
        },
    ]

    assert module.challenge_targets(components) == []
    assert [target["candidate_id"] for target in module.challenge_targets(components, high_overhead_miles=5.0)] == [
        "default-hidden"
    ]


def test_build_report_challenges_high_overhead_non_manual_route(tmp_path):
    module = load_module()
    map_data = {
        "packages": [
            {
                "package_number": 1,
                "block_name": "High overhead block",
                "planning_status": "needs_manual_route_design",
                "components": [
                    {
                        "candidate_id": "current",
                        "trailhead": "Old",
                        "trail_names": ["Trail A"],
                        "segment_ids": [1, 2],
                        "official_miles": 3.0,
                        "on_foot_miles": 10.0,
                        "total_minutes": 200,
                    }
                ],
            },
            {
                "package_number": 2,
                "block_name": "Manual block",
                "planning_status": "accepted_manual_split_parking_manual",
                "components": [
                    {
                        "candidate_id": "manual",
                        "trailhead": "Manual",
                        "trail_names": ["Trail B"],
                        "segment_ids": [3],
                        "official_miles": 1.0,
                        "on_foot_miles": 4.0,
                    }
                ],
            },
        ]
    }
    source_path = tmp_path / "candidates.json"
    source_path.write_text(
        """
{
  "routes": [
    {
      "candidate_id": "current",
      "segment_ids": [1, 2],
      "official_miles": 3.0,
      "on_foot_miles": 10.0,
      "trailhead": "Old",
      "trail_names": ["Trail A"]
    }
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(map_data, official_segments(), [source_path])

    assert report["summary"]["target_count"] == 1
    assert report["summary"]["challenged_candidate_ids"] == ["current"]
    assert report["summary"]["current_best_exact_candidate_count"] == 1
    assert report["summary"]["manual_map_review_still_required_count"] == 1
