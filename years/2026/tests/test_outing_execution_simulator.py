import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "simulate_outing_execution.py"


def load_simulator():
    spec = importlib.util.spec_from_file_location("simulate_outing_execution", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_route_provider(origin, destination):
    return {
        "provider": "fake",
        "route_validated": True,
        "distance_miles": 4.2,
        "duration_minutes": 11,
        "geometry": {
            "type": "LineString",
            "coordinates": [[origin["lon"], origin["lat"]], [destination["lon"], destination["lat"]]],
        },
    }


def candidate(**overrides):
    base = {
        "candidate_id": "validated-loop",
        "trail_names": ["Validated Loop"],
        "route_status": "graph_validated",
        "segment_ids": [1, 2],
        "official_new_miles": 2.4,
        "official_repeat_miles": 0.2,
        "connector_miles": 0.1,
        "road_miles": 0,
        "estimated_total_on_foot_miles": 2.7,
        "effort": {
            "ascent_ft": 420,
            "descent_ft": 390,
            "grade_adjusted_miles": 3.12,
            "effort_score": 42,
        },
        "time_breakdown_minutes": {
            "drive_to_trailhead": 10,
            "parking_and_prep": 8,
            "moving_time": 35,
            "return_drive": 10,
        },
        "trailhead": {
            "name": "Validated Trailhead",
            "lat": 43.64,
            "lon": -116.2,
            "has_parking": True,
            "parking_confidence": "inferred_from_trailhead_layer",
            "source": "city_parks_facilities",
            "facility_status": "Open",
        },
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
            "trailhead_snap_confidence": "high",
            "trailhead_snap": {
                "confidence": "high",
                "direct_gap_miles": 0.01,
                "mapped_access_miles": 0.01,
                "graph_validated": True,
            },
            "connector_overlap_checked": True,
            "special_management_checked": False,
        },
        "return_to_car": {
            "strategy": "mapped_mixed_loop",
            "official_repeat_miles": 0.2,
            "connector_miles": 0.1,
            "endpoint_gap_miles": 0.5,
            "needs_map_validation": False,
            "graph_validated": True,
            "connector_names": ["Connector"],
            "official_repeat_segment_ids": [1],
        },
        "less_optimal_flags": [],
    }
    base.update(overrides)
    return base


def test_simulated_outing_passes_when_drive_parking_access_run_and_return_are_valid():
    simulator = load_simulator()
    drive_model = {"origin_label": "Home", "origin_lat": 43.63, "origin_lon": -116.21}
    outing = simulator.simulate_candidate_outing(
        candidate(),
        drive_model,
        route_provider=fake_route_provider,
    )

    assert outing["execution_status"] == "simulated_ready"
    assert outing["validation"]["drive_to_trailhead_validated"] is True
    assert outing["validation"]["parking_validated"] is True
    assert outing["validation"]["access_to_trail_validated"] is True
    assert outing["validation"]["return_to_car_validated"] is True
    assert outing["legs"][0]["leg_type"] == "drive_to_trailhead"
    assert outing["legs"][-1]["leg_type"] == "drive_home"
    assert outing["simulated_total_minutes"] == 65
    assert outing["simulated_efficiency_score"] == 0.0369
    assert outing["legs"][3]["ascent_ft"] == 420
    assert outing["legs"][3]["descent_ft"] == 390
    assert outing["legs"][3]["grade_adjusted_miles"] == 3.12


def test_simulated_outing_fails_when_return_to_car_is_only_estimated():
    simulator = load_simulator()
    drive_model = {"origin_label": "Home", "origin_lat": 43.63, "origin_lon": -116.21}
    draft = candidate(
        route_status="draft",
        return_to_car={
            "strategy": "connector_or_road_loop",
            "official_repeat_miles": 0,
            "connector_miles": 2.1,
            "endpoint_gap_miles": 1.68,
            "needs_map_validation": True,
            "graph_validated": False,
            "connector_names": [],
            "official_repeat_segment_ids": [],
        },
    )

    outing = simulator.simulate_candidate_outing(
        draft,
        drive_model,
        route_provider=fake_route_provider,
    )

    assert outing["execution_status"] == "blocked_by_route_validation"
    assert outing["validation"]["return_to_car_validated"] is False
    assert "return_to_car_not_graph_validated" in outing["blocking_reasons"]


def test_simulated_outing_fails_when_parking_is_not_supported_by_source():
    simulator = load_simulator()
    drive_model = {"origin_label": "Home", "origin_lat": 43.63, "origin_lon": -116.21}
    no_parking = candidate(
        trailhead={
            "name": "Unverified Trailhead",
            "lat": 43.64,
            "lon": -116.2,
            "has_parking": None,
            "facility_status": "Open",
        },
    )

    outing = simulator.simulate_candidate_outing(
        no_parking,
        drive_model,
        route_provider=fake_route_provider,
    )

    assert outing["execution_status"] == "blocked_by_logistics"
    assert outing["validation"]["parking_validated"] is False
    assert "parking_not_source_validated" in outing["blocking_reasons"]


def test_execution_ready_menu_uses_simulated_total_time_for_recommendations():
    simulator = load_simulator()
    slow_high_miles = {
        **candidate(candidate_id="slow", trail_names=["Slow Higher Miles"], official_new_miles=3.0),
        "time_bucket": "under_1_hour",
        "execution_status": "simulated_ready",
        "simulated_total_minutes": 90,
        "simulated_efficiency_score": 0.0333,
    }
    fast_lower_miles = {
        **candidate(candidate_id="fast", trail_names=["Fast Lower Miles"], official_new_miles=2.0),
        "time_bucket": "under_1_hour",
        "execution_status": "simulated_ready",
        "simulated_total_minutes": 40,
        "simulated_efficiency_score": 0.05,
    }
    blocked = {
        **candidate(candidate_id="blocked", trail_names=["Blocked"]),
        "time_bucket": "under_1_hour",
        "execution_status": "blocked_by_route_validation",
        "simulated_total_minutes": 20,
        "simulated_efficiency_score": 0.1,
    }

    menu = simulator.build_execution_ready_menu([slow_high_miles, fast_lower_miles, blocked])

    assert menu["ready_counts_by_bucket"] == {"under_1_hour": 2}
    assert menu["recommended_by_bucket"]["under_1_hour"]["candidate_id"] == "fast"
    assert menu["blocked_counts_by_bucket"] == {"under_1_hour": 1}


def test_combined_ready_menu_builds_longer_non_overlapping_outings():
    simulator = load_simulator()
    drive_model = {"origin_label": "Home", "origin_lat": 43.63, "origin_lon": -116.21}
    first = simulator.simulate_candidate_outing(
        candidate(candidate_id="first", segment_ids=[1, 2], trail_names=["First"]),
        drive_model,
        route_provider=fake_route_provider,
    )
    second = simulator.simulate_candidate_outing(
        candidate(
            candidate_id="second",
            segment_ids=[3, 4],
            trail_names=["Second"],
            time_breakdown_minutes={
                "drive_to_trailhead": 10,
                "parking_and_prep": 8,
                "moving_time": 50,
                "return_drive": 10,
            },
        ),
        drive_model,
        route_provider=fake_route_provider,
    )
    overlapping = simulator.simulate_candidate_outing(
        candidate(candidate_id="overlap", segment_ids=[2, 5], trail_names=["Overlap"]),
        drive_model,
        route_provider=fake_route_provider,
    )

    menu = simulator.build_combined_ready_menu(
        [first, second, overlapping],
        drive_model,
        route_provider=fake_route_provider,
        max_outings_per_combo=2,
    )

    combos = menu["recommended_by_bucket"]
    assert combos["two_to_three_hours"]["outing_count"] == 2
    assert combos["two_to_three_hours"]["official_new_miles"] == 4.8
    assert combos["two_to_three_hours"]["simulated_total_minutes"] == 123
    assert combos["two_to_three_hours"]["outing_ids"] == ["first", "second"]


def test_same_car_ready_menu_combines_same_trailhead_outings_without_extra_drives():
    simulator = load_simulator()
    drive_model = {"origin_label": "Home", "origin_lat": 43.63, "origin_lon": -116.21}
    first = simulator.simulate_candidate_outing(
        candidate(candidate_id="first", segment_ids=[1, 2], trail_names=["First"]),
        drive_model,
        route_provider=fake_route_provider,
    )
    second = simulator.simulate_candidate_outing(
        candidate(
            candidate_id="second",
            segment_ids=[3, 4],
            trail_names=["Second"],
            time_breakdown_minutes={
                "drive_to_trailhead": 10,
                "parking_and_prep": 8,
                "moving_time": 50,
                "return_drive": 10,
            },
        ),
        drive_model,
        route_provider=fake_route_provider,
    )
    different_trailhead = simulator.simulate_candidate_outing(
        candidate(
            candidate_id="other",
            segment_ids=[5, 6],
            trail_names=["Other"],
            trailhead={
                "name": "Other Trailhead",
                "lat": 43.65,
                "lon": -116.22,
                "has_parking": True,
                "parking_confidence": "inferred_from_trailhead_layer",
                "source": "city_parks_facilities",
                "facility_status": "Open",
            },
        ),
        drive_model,
        route_provider=fake_route_provider,
    )

    menu = simulator.build_same_car_ready_menu(
        [first, second, different_trailhead],
        max_outings_per_combo=2,
    )

    combo = menu["recommended_by_bucket"]["one_to_two_hours"]
    assert combo["recommendation_type"] == "same_parked_car"
    assert combo["outing_ids"] == ["first", "second"]
    assert combo["trailheads"] == ["Validated Trailhead"]
    assert combo["simulated_total_minutes"] == 115
    assert combo["official_new_miles"] == 4.8
    assert combo["interdrive_minutes"] == 0


def test_best_executable_menu_compares_single_and_combined_bucket_winners():
    simulator = load_simulator()
    single = {
        **candidate(candidate_id="single", trail_names=["Single"]),
        "time_bucket": "under_1_hour",
        "execution_status": "simulated_ready",
        "simulated_total_minutes": 54,
        "simulated_efficiency_score": 0.0396,
        "legs": [
            {"duration_minutes": 4},
            {"trailhead": "Single Trailhead"},
            {},
            {
                "official_new_miles": 2.14,
                "estimated_total_on_foot_miles": 2.99,
            },
            {},
            {"duration_minutes": 4},
        ],
    }
    worse_combo = {
        "combo_id": "worse-combo",
        "outing_ids": ["a", "b"],
        "outing_count": 2,
        "time_bucket": "under_1_hour",
        "trail_names": ["Worse", "Combo"],
        "trailheads": ["A", "B"],
        "official_new_miles": 1.55,
        "estimated_total_on_foot_miles": 1.56,
        "simulated_total_minutes": 58,
        "simulated_efficiency_score": 0.0267,
        "interdrive_minutes": 9,
    }
    longer_combo = {
        **worse_combo,
        "combo_id": "longer-combo",
        "time_bucket": "two_to_three_hours",
        "official_new_miles": 6.89,
        "simulated_total_minutes": 133,
        "simulated_efficiency_score": 0.0518,
    }

    menu = simulator.build_best_executable_menu(
        execution_menu={"recommended_by_bucket": {"under_1_hour": single}},
        combined_menu={
            "recommended_by_bucket": {
                "under_1_hour": worse_combo,
                "two_to_three_hours": longer_combo,
            }
        },
    )

    assert menu["recommended_by_bucket"]["under_1_hour"]["recommendation_type"] == "single_outing"
    assert menu["recommended_by_bucket"]["under_1_hour"]["id"] == "single"
    assert menu["recommended_by_bucket"]["two_to_three_hours"]["recommendation_type"] == "combined_multi_stop"
    assert menu["recommended_by_bucket"]["two_to_three_hours"]["id"] == "longer-combo"


def test_best_executable_menu_includes_same_car_options():
    simulator = load_simulator()
    same_car = {
        "combo_id": "same-car",
        "time_bucket": "two_to_three_hours",
        "trail_names": ["Same", "Car"],
        "trailheads": ["Trailhead"],
        "official_new_miles": 5.0,
        "estimated_total_on_foot_miles": 6.0,
        "simulated_total_minutes": 130,
        "simulated_efficiency_score": 0.0385,
        "first_drive_minutes": 10,
        "return_home_minutes": 10,
        "interdrive_minutes": 0,
    }
    worse_combined = {
        **same_car,
        "combo_id": "drive-between",
        "trailheads": ["A", "B"],
        "official_new_miles": 4.0,
        "simulated_total_minutes": 140,
        "simulated_efficiency_score": 0.0286,
        "interdrive_minutes": 15,
    }

    menu = simulator.build_best_executable_menu(
        execution_menu={"recommended_by_bucket": {}},
        same_car_menu={"recommended_by_bucket": {"two_to_three_hours": same_car}},
        combined_menu={"recommended_by_bucket": {"two_to_three_hours": worse_combined}},
    )

    assert menu["recommended_by_bucket"]["two_to_three_hours"]["recommendation_type"] == "same_parked_car"
    assert menu["recommended_by_bucket"]["two_to_three_hours"]["id"] == "same-car"


def test_single_car_menu_compares_single_outings_and_same_car_addons():
    simulator = load_simulator()
    single = {
        **candidate(candidate_id="single", trail_names=["Single"]),
        "time_bucket": "one_to_two_hours",
        "execution_status": "simulated_ready",
        "simulated_total_minutes": 90,
        "simulated_efficiency_score": 0.05,
        "legs": [
            {"duration_minutes": 10},
            {"trailhead": "Trailhead"},
            {},
            {"official_new_miles": 4.5, "estimated_total_on_foot_miles": 5.0},
            {},
            {"duration_minutes": 10},
        ],
    }
    same_car = {
        "combo_id": "same-car",
        "time_bucket": "two_to_three_hours",
        "trail_names": ["Same", "Car"],
        "trailheads": ["Trailhead"],
        "official_new_miles": 5.0,
        "estimated_total_on_foot_miles": 6.0,
        "simulated_total_minutes": 130,
        "simulated_efficiency_score": 0.0385,
        "first_drive_minutes": 10,
        "return_home_minutes": 10,
        "interdrive_minutes": 0,
    }

    menu = simulator.build_single_car_menu(
        execution_menu={"recommended_by_bucket": {"one_to_two_hours": single}},
        same_car_menu={"recommended_by_bucket": {"two_to_three_hours": same_car}},
    )

    assert menu["recommended_by_bucket"]["one_to_two_hours"]["recommendation_type"] == "single_outing"
    assert menu["recommended_by_bucket"]["one_to_two_hours"]["id"] == "single"
    assert menu["recommended_by_bucket"]["two_to_three_hours"]["recommendation_type"] == "same_parked_car"
    assert menu["recommended_by_bucket"]["two_to_three_hours"]["id"] == "same-car"


def test_default_candidate_set_is_graph_validated(monkeypatch):
    simulator = load_simulator()
    monkeypatch.setattr(simulator.sys, "argv", ["simulate_outing_execution.py"])

    args = simulator.parse_args()

    assert args.candidate_set == "graph-validated"
    assert args.max_combo_source_outings >= 40
