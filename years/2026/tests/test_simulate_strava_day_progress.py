import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "simulate_strava_day_progress.py"


def load_module():
    spec = importlib.util.spec_from_file_location("simulate_strava_day_progress", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_challenge_windows_from_pull_summary_uses_requested_years(tmp_path):
    module = load_module()
    path = tmp_path / "pull_summary.json"
    path.write_text(
        json.dumps(
            {
                "date_window": {
                    "challenge_windows": {
                        "2024_proxy_window": {"start": "2024-06-19", "end": "2024-07-19"},
                        "2025_challenge_window": {"start": "2025-06-19", "end": "2025-07-19"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    windows = module.challenge_windows_from_pull_summary(path, [2024, 2025])

    assert [window["source_year"] for window in windows] == [2024, 2025]
    assert windows[0]["window_name"] == "2024_proxy_window"
    assert str(windows[1]["start_date"]) == "2025-06-19"


def test_outing_state_hides_completed_and_manual_hold_outings():
    module = load_module()
    package_map = {
        "summary": {},
        "progress": {"completed_segment_ids": []},
        "manual_design": {
            "areas": [
                {
                    "area_id": "manual",
                    "package_number": 2,
                    "demote_candidate_ids": ["manual-candidate"],
                }
            ]
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Runnable",
                "components": [
                    {
                        "candidate_id": "done",
                        "trail_names": ["Done"],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.2,
                        "total_minutes": 40,
                        "trailhead": "Done Trailhead",
                        "segment_ids": [1],
                    },
                    {
                        "candidate_id": "open",
                        "trail_names": ["Open"],
                        "official_miles": 2.0,
                        "on_foot_miles": 3.0,
                        "total_minutes": 80,
                        "trailhead": "Open Trailhead",
                        "segment_ids": [2],
                    },
                ],
            },
            {
                "package_number": 2,
                "block_name": "Manual",
                "components": [
                    {
                        "candidate_id": "manual-candidate",
                        "trail_names": ["Manual"],
                        "official_miles": 2.0,
                        "on_foot_miles": 5.0,
                        "total_minutes": 150,
                        "trailhead": "Manual Trailhead",
                        "segment_ids": [3],
                    }
                ],
            },
        ],
    }

    state = module.outing_state_for_completed(package_map, {1})

    assert state["open_runnable_outing_count"] == 1
    assert state["manual_hold_count"] == 1
    assert state["open_candidate_ids"] == ["open"]
    assert state["next_by_time_bucket"]["2 hours or less"]["label"] == "1B"


def test_render_markdown_mentions_replay_map():
    module = load_module()
    simulation = {
        "outputs": {"html": "/tmp/replay.html", "json": "/tmp/replay.json", "markdown": "/tmp/replay.md"},
        "scenarios": [
            {
                "source_year": 2025,
                "summary": {
                    "activity_days": 1,
                    "activities_considered": 1,
                    "progress_days": 1,
                    "completed_segments": 2,
                    "completed_official_miles": 3.4,
                    "remaining_official_miles": 161.0,
                    "beat_2025_baseline_day": None,
                    "beat_2025_baseline_source_date": None,
                    "initial_open_runnable_outings": 23,
                    "final_open_runnable_outings": 20,
                    "open_runnable_outings_removed": 3,
                    "outing_hidden_events": 3,
                    "days_with_hidden_outings": 1,
                    "route_menu_primary_changed_days": 1,
                    "distinct_time_bucket_recommendation_states": 2,
                    "coverage_valid_after_every_rerun": True,
                },
                "days": [
                    {
                        "source_date": "2025-06-19",
                        "target_2026_date": "2026-06-18",
                        "challenge_day": 1,
                        "activity_count": 1,
                        "new_segment_count": 2,
                        "cumulative_segment_count": 2,
                        "cumulative_official_miles": 3.4,
                        "remaining_official_miles": 161.0,
                        "route_menu_primary_changed": False,
                        "outing_map_state": {
                            "open_runnable_outing_count": 20,
                            "next_by_time_bucket": {"2 hours or less": None, "4+ hours": None},
                        },
                    }
                ],
                "highest_impact_days": [],
            }
        ],
    }

    rendered = module.render_markdown(simulation)

    assert "2026 Two-Year Strava Replay Simulation" in rendered
    assert "Adaptation Check" in rendered
    assert "23 -> 20" in rendered
    assert "Interactive replay map" in rendered
    assert "/tmp/replay.html" in rendered
