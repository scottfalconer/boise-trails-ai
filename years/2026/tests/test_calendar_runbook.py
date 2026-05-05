import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calendar_runbook.py"


def load_runbook():
    spec = importlib.util.spec_from_file_location("calendar_runbook", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execution_outing(outing_id="route-a", trailhead="Trailhead A"):
    trail_name = outing_id.replace("-", " ").title()
    return {
        "candidate_id": outing_id,
        "trail_names": [trail_name],
        "segment_ids": [1, 2],
        "execution_status": "simulated_ready",
        "blocking_reasons": [],
        "day_of_checks_remaining": ["current_r2r_conditions_and_special_management"],
        "validation": {
            "drive_to_trailhead_validated": True,
            "parking_validated": True,
            "access_to_trail_validated": True,
            "official_segment_traversal_validated": True,
            "return_to_car_validated": True,
            "drive_home_validated": True,
        },
        "legs": [
            {
                "leg_type": "drive_to_trailhead",
                "from": "Home",
                "to": trailhead,
                "provider": "osrm",
                "route_validated": True,
                "duration_minutes": 12,
                "distance_miles": 4.2,
            },
            {
                "leg_type": "park",
                "trailhead": trailhead,
                "lat": 43.626,
                "lon": -116.205 if trailhead == "Trailhead A" else -116.215,
                "can_park": True,
                "parking_confidence": "source_validated",
                "facility_status": "Open",
                "parking_minutes": 8,
            },
            {
                "leg_type": "access_to_trail",
                "source": "mapped_graph",
                "validated": True,
                "one_way_miles": 0.2,
                "round_trip_miles": 0.4,
                "connector_names": ["Access Path"],
            },
            {
                "leg_type": "run_official_route",
                "trail_names": [trail_name],
                "official_new_miles": 3.0,
                "official_repeat_miles": 0.5,
                "connector_miles": 0.4,
                "road_miles": 0,
                "estimated_total_on_foot_miles": 3.9,
                "ascent_ft": 620,
                "descent_ft": 580,
                "grade_adjusted_miles": 4.52,
                "moving_minutes": 54,
                "validated": True,
            },
            {
                "leg_type": "return_to_car",
                "trailhead": trailhead,
                "strategy": "mapped_connector_loop",
                "validated": True,
                "endpoint_gap_miles": 1.2,
                "connector_names": ["Return Path"],
            },
            {
                "leg_type": "drive_home",
                "from": trailhead,
                "to": "Home",
                "provider": "osrm",
                "route_validated": True,
                "duration_minutes": 13,
                "distance_miles": 4.3,
            },
        ],
    }


def schedule_for(outing_id="route-a"):
    trail_name = outing_id.replace("-", " ").title()
    return {
        "schedule_type": "test_schedule",
        "constraints": {"start_time": "07:00"},
        "summary": {
            "scheduled_segments": 2,
            "ready_segments_available": 2,
            "scheduled_official_miles": 3.0,
            "scheduled_total_on_foot_miles": 3.9,
        },
        "days": [
            {
                "date": "2026-06-18",
                "status": "scheduled",
                "outing_ids": [outing_id],
                "outings": [
                    {
                        "outing_id": outing_id,
                        "trail_names": [trail_name],
                        "trailhead": "Trailhead A",
                        "simulated_total_minutes": 87,
                        "raw_official_miles": 3.0,
                        "new_official_miles": 3.0,
                        "estimated_total_on_foot_miles": 3.9,
                        "ascent_ft": 620,
                        "descent_ft": 580,
                        "grade_adjusted_miles": 4.52,
                        "new_segment_ids": [1, 2],
                        "repeat_segment_ids": [],
                    }
                ],
            }
        ],
    }


def test_runbook_joins_calendar_day_to_full_execution_legs():
    runbook = load_runbook()

    result = runbook.build_runbook(
        schedule_for(),
        {"outings": [execution_outing()]},
        profile_name="test-profile",
    )

    outing = result["days"][0]["outings"][0]
    assert outing["outing_id"] == "route-a"
    assert outing["route_label"] == "A-route"
    assert outing["official_to_total_ratio"] == 0.769
    assert outing["execution_chain"] == [
        "drive_to_trailhead",
        "park",
        "access_to_trail",
        "run_official_route",
        "return_to_car",
        "drive_home",
    ]
    assert outing["drive_to_trailhead"]["provider"] == "osrm"
    assert outing["park"]["can_park"] is True
    assert outing["return_to_car"]["strategy"] == "mapped_connector_loop"
    assert outing["validation_passed"] is True
    assert outing["static_preflight"]["run_date"] == "2026-06-18"
    assert outing["static_preflight"]["start_time"] == "07:00"
    assert result["days"][0]["ascent_ft"] == 620
    assert outing["run_official_route"]["ascent_ft"] == 620
    assert result["audit"]["missing_execution_outing_ids"] == []


def test_runbook_builds_day_transport_plan_between_trailheads():
    runbook = load_runbook()
    schedule = schedule_for()
    schedule["days"][0]["simulated_total_minutes"] = 174
    schedule["days"][0]["outings"].append(
        {
            "outing_id": "route-b",
            "trail_names": ["Route B"],
            "trailhead": "Trailhead B",
            "simulated_total_minutes": 87,
            "raw_official_miles": 3.0,
            "new_official_miles": 3.0,
            "estimated_total_on_foot_miles": 3.9,
            "new_segment_ids": [3, 4],
            "repeat_segment_ids": [],
        }
    )

    def interdrive(origin, destination):
        return {
            "provider": "fixture",
            "route_validated": True,
            "distance_miles": 2.0,
            "duration_minutes": 9,
            "from": origin["name"],
            "to": destination["name"],
        }

    result = runbook.build_runbook(
        schedule,
        {
            "outings": [
                execution_outing("route-a", "Trailhead A"),
                execution_outing("route-b", "Trailhead B"),
            ]
        },
        profile_name="test-profile",
        route_provider=interdrive,
    )

    transport = result["days"][0]["day_transport"]
    assert [leg["leg_type"] for leg in transport["legs"]] == [
        "drive_to_first_trailhead",
        "drive_between_trailheads",
        "drive_home_from_last_trailhead",
    ]
    assert transport["actual_day_drive_minutes"] == 34
    assert transport["conservative_scheduled_drive_minutes"] == 50
    assert transport["drive_minutes_saved_vs_conservative"] == 16
    assert result["days"][0]["realistic_total_minutes"] == 158


def test_runbook_reports_schedule_outings_without_execution_evidence():
    runbook = load_runbook()

    result = runbook.build_runbook(
        schedule_for("missing-route"),
        {"outings": []},
        profile_name="test-profile",
    )

    assert result["audit"]["missing_execution_outing_ids"] == ["missing-route"]
    assert result["audit"]["execution_validation_passed"] is False


def test_markdown_renders_single_car_execution_sequence():
    runbook = load_runbook()
    result = runbook.build_runbook(
        schedule_for(),
        {"outings": [execution_outing()]},
        profile_name="test-profile",
    )

    markdown = runbook.render_markdown(result)

    assert "Drive from home to first trailhead" in markdown
    assert "Park at Trailhead A" in markdown
    assert "Run Route A" in markdown
    assert "Return to car" in markdown
    assert "Solo-drive reference" in markdown
    assert "Day drive sequence" in markdown


def test_runbook_reports_human_load_exceptions():
    runbook = load_runbook()
    schedule = schedule_for()
    schedule["constraints"] = {
        "start_time": "07:00",
        "weekday_normal_max_minutes": 120,
        "max_consecutive_scheduled_days_target": 1,
        "required_rest_days_before_latest": 1,
        "latest_scheduled_date": "2026-06-19",
        "max_weekly_on_foot_miles": 5,
        "long_on_foot_day_miles": 3,
        "max_long_on_foot_days_per_week": 0,
    }
    schedule["days"][0]["simulated_total_minutes"] = 150
    schedule["days"][0]["outings"][0]["simulated_total_minutes"] = 150
    schedule["days"].append(
        {
            "date": "2026-06-19",
            "status": "open",
            "available_minutes": 120,
            "normal_available_minutes": 120,
            "reason": "test_rest_day",
        }
    )

    result = runbook.build_runbook(
        schedule,
        {"outings": [execution_outing()]},
        profile_name="test-profile",
    )

    assert result["audit"]["weekday_exception_days"] == 1
    assert result["audit"]["weekly_mileage_violations"] == 0
    assert result["audit"]["long_day_count_violations"] == 1
    assert result["audit"]["rest_days_before_latest_deficit"] == 0
    assert result["days"][0]["requires_normal_cap_exception"] is True
    assert result["load_analysis"]["weekday_exception_days"][0]["minutes_over_normal"] == 30
