import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_efficiency_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_efficiency_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_audit_marks_manual_hold_and_high_ratio_not_proven():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.7,
        },
        "packages": [
            {
                "package_number": 10,
                "block_name": "North pod",
                "components": [
                    {
                        "label": "10",
                        "trailhead": "Dry Creek",
                        "trail_names": ["Spring Creek"],
                        "official_miles": 9.0,
                        "on_foot_miles": 23.0,
                        "total_minutes": 400,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "10", "trailhead": "Dry Creek", "official_miles": 9.0, "on_foot_miles": 23.0}}
        ],
        "manual_holds": [{"label": "16A"}],
    }
    human_plan = {"summary": {"manual_design_area_count": 1, "planwide_on_foot_to_official_ratio": 1.7}}
    package16 = {
        "areas": [
            {
                "status": "accepted_split_probe_parking_manual",
                "current_placeholder": {"official_miles": 11.62, "on_foot_miles": 36.48},
                "current_best_split_probe": {
                    "official_miles": 11.62,
                    "on_foot_miles": 27.16,
                    "remaining_blocker": "parking manual",
                },
            }
        ]
    }

    audit = module.build_audit(
        map_data,
        field_packet,
        human_plan,
        package16,
        None,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
    )

    assert audit["achieved"] is False
    assert audit["verdict"] == "not_proven"
    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Full official coverage is represented"] == "passed"
    assert statuses["Runnable field packet covers all official work"] == "failed"
    assert statuses["No route exceeds 2.0x without manual/local-map proof"] == "failed"


def test_build_audit_can_pass_when_efficiency_gates_are_met():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.45,
        },
        "route_cues": {
            "good": {
                "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 90},
                "effort": {"ascent_ft": 700, "grade_adjusted_miles": 10.7, "elevation_source": "dem"},
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Good loop",
                "components": [
                    {
                        "candidate_id": "good",
                        "label": "1",
                        "trailhead": "Trailhead",
                        "trail_names": ["Good Trail"],
                        "official_miles": 10.0,
                        "on_foot_miles": 14.0,
                        "total_minutes": 120,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "1", "trailhead": "Trailhead", "official_miles": 10.0, "on_foot_miles": 14.0}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.45}}
    package16 = {"areas": []}

    audit = module.build_audit(
        map_data,
        field_packet,
        human_plan,
        package16,
        None,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
    )

    assert audit["achieved"] is True
    assert audit["verdict"] == "proven"


def test_build_audit_accepts_slight_ratio_overage_when_targets_are_proofed():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.63,
        },
        "route_cues": {
            "proofed": {
                "time_estimates_minutes": {"door_to_door_p75": 180, "moving_effort_p75": 140},
                "effort": {"ascent_ft": 900, "grade_adjusted_miles": 12.3, "elevation_source": "dem"},
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Proofed ratio gap",
                "components": [
                    {
                        "candidate_id": "proofed",
                        "trailhead": "Trailhead",
                        "trail_names": ["Proofed"],
                        "official_miles": 10.0,
                        "on_foot_miles": 16.3,
                        "total_minutes": 180,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "1", "trailhead": "Trailhead", "official_miles": 10.0, "on_foot_miles": 16.3}}
        ],
        "manual_holds": [],
    }
    alternative_challenge = {
        "summary": {
            "target_count": 1,
            "challenged_candidate_ids": ["proofed"],
            "better_exact_candidate_count": 0,
            "better_superset_candidate_count": 0,
        }
    }
    route_proofs = [
        {
            "proofs": [
                {
                    "candidate_ids": ["proofed"],
                    "status": "accepted_current",
                    "checks": {
                        "gpx_continuity_passed": True,
                        "current_route_has_p75_time": True,
                        "current_route_has_dem_effort": True,
                        "no_better_exact_generated_candidate": True,
                        "no_dominant_boundary_recombination": True,
                        "no_dominant_global_optimizer_replacement": True,
                    },
                }
            ]
        }
    ]

    audit = module.build_audit(
        map_data,
        field_packet,
        {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.63}},
        {"areas": []},
        alternative_challenge,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
        route_proofs,
    )

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Planwide on-foot/official ratio is within preferred target or accepted proof tolerance"] == "passed"


def test_build_audit_counts_proofed_historical_challenge_targets():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.63,
        },
        "route_cues": {
            "current-split": {
                "time_estimates_minutes": {"door_to_door_p75": 180, "moving_effort_p75": 140},
                "effort": {"ascent_ft": 900, "grade_adjusted_miles": 12.3, "elevation_source": "dem"},
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Current split route",
                "components": [
                    {
                        "candidate_id": "current-split",
                        "trailhead": "Trailhead",
                        "trail_names": ["Proofed"],
                        "official_miles": 10.0,
                        "on_foot_miles": 16.3,
                        "total_minutes": 180,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "1", "trailhead": "Trailhead", "official_miles": 10.0, "on_foot_miles": 16.3}}
        ],
        "manual_holds": [],
    }
    alternative_challenge = {
        "summary": {
            "target_count": 1,
            "challenged_candidate_ids": ["historical-aggregate"],
            "better_exact_candidate_count": 0,
            "better_superset_candidate_count": 0,
        }
    }
    route_proofs = [
        {
            "proofs": [
                {
                    "candidate_ids": ["historical-aggregate", "current-split"],
                    "status": "accepted_current",
                    "checks": {
                        "gpx_continuity_passed": True,
                        "current_route_has_p75_time": True,
                        "current_route_has_dem_effort": True,
                        "no_better_exact_generated_candidate": True,
                        "no_dominant_boundary_recombination": True,
                        "no_dominant_global_optimizer_replacement": True,
                    },
                }
            ]
        }
    ]

    audit = module.build_audit(
        map_data,
        field_packet,
        {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.63}},
        {"areas": []},
        alternative_challenge,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
        route_proofs,
    )

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Planwide on-foot/official ratio is within preferred target or accepted proof tolerance"] == "passed"
    assert audit["summary"]["route_proofs"]["accepted_candidate_ids"] == ["current-split", "historical-aggregate"]
    assert audit["summary"]["route_proofs"]["accepted_active_candidate_ids"] == ["current-split"]


def test_build_audit_fails_when_field_packet_omits_segment_coverage():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 2,
            "official_miles": 2.0,
            "planwide_on_foot_to_official_ratio": 1.5,
        },
        "route_cues": {
            "one": {
                "time_estimates_minutes": {"door_to_door_p75": 60, "moving_effort_p75": 45},
                "effort": {"ascent_ft": 100, "grade_adjusted_miles": 1.5, "elevation_source": "dem"},
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Incomplete",
                "components": [
                    {
                        "candidate_id": "one",
                        "trailhead": "Trailhead",
                        "trail_names": ["One"],
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                        "total_minutes": 60,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {
                "outing": {
                    "label": "1",
                    "trailhead": "Trailhead",
                    "official_miles": 1.0,
                    "on_foot_miles": 1.5,
                    "segment_ids": ["1"],
                }
            }
        ],
        "manual_holds": [],
    }
    audit = module.build_audit(
        map_data,
        field_packet,
        {"summary": {"manual_design_area_count": 0}},
        {"areas": []},
        None,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
    )

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Runnable field packet covers all official work"] == "failed"
    assert audit["summary"]["runnable_field_packet_totals"]["covered_segment_count"] == 1


def test_build_audit_does_not_fail_ratio_gate_for_challenged_manual_route():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.5,
        },
        "packages": [
            {
                "package_number": 16,
                "block_name": "Accepted grinder",
                "planning_status": "accepted_manual_split_parking_manual",
                "components": [
                    {
                        "field_menu_label": "16A-2",
                        "trailhead": "Dry/Sweet",
                        "trail_names": ["Shingle"],
                        "official_miles": 5.5,
                        "on_foot_miles": 15.0,
                        "total_minutes": 300,
                        "route_design_status": "gpx_generated_parking_manual",
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "16A-2", "trailhead": "Dry/Sweet", "official_miles": 5.5, "on_foot_miles": 15.0}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.5}}
    package16 = {"areas": [{"status": "accepted_split_probe_parking_manual"}]}

    audit = module.build_audit(map_data, field_packet, human_plan, package16)

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["No route exceeds 2.0x without manual/local-map proof"] == "passed"
    assert statuses["Largest overhead routes have been manually challenged"] == "passed"


def test_build_audit_accepts_local_route_proof_for_high_ratio_route():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.5,
        },
        "route_cues": {
            "accepted-grinder": {
                "time_estimates_minutes": {"door_to_door_p75": 180, "moving_effort_p75": 130},
                "effort": {"ascent_ft": 1600, "grade_adjusted_miles": 6.8, "elevation_source": "dem"},
            }
        },
        "packages": [
            {
                "package_number": 19,
                "block_name": "Accepted local grinder",
                "components": [
                    {
                        "candidate_id": "accepted-grinder",
                        "trailhead": "Trailhead",
                        "trail_names": ["Peak"],
                        "official_miles": 2.0,
                        "on_foot_miles": 4.5,
                        "total_minutes": 180,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "19", "trailhead": "Trailhead", "official_miles": 2.0, "on_foot_miles": 4.5}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.5}}
    route_proofs = [
        {
            "proofs": [
                {
                    "candidate_ids": ["accepted-grinder"],
                    "status": "accepted_current",
                    "checks": {
                        "gpx_continuity_passed": True,
                        "current_route_has_p75_time": True,
                        "current_route_has_dem_effort": True,
                        "no_better_exact_generated_candidate": True,
                        "no_dominant_boundary_recombination": True,
                        "no_dominant_global_optimizer_replacement": True,
                    },
                }
            ]
        }
    ]

    audit = module.build_audit(
        map_data,
        field_packet,
        human_plan,
        {"areas": []},
        None,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
        route_proofs,
    )

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["No route exceeds 2.0x without manual/local-map proof"] == "passed"
    assert audit["summary"]["route_proofs"]["accepted_active_candidate_ids"] == ["accepted-grinder"]


def test_build_audit_rejects_incomplete_local_route_proof():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.5,
        },
        "packages": [
            {
                "package_number": 19,
                "block_name": "Unproved local grinder",
                "components": [
                    {
                        "candidate_id": "unproved-grinder",
                        "trailhead": "Trailhead",
                        "trail_names": ["Peak"],
                        "official_miles": 2.0,
                        "on_foot_miles": 4.5,
                        "total_minutes": 180,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "19", "trailhead": "Trailhead", "official_miles": 2.0, "on_foot_miles": 4.5}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.5}}
    route_proofs = [
        {
            "proofs": [
                {
                    "candidate_ids": ["unproved-grinder"],
                    "status": "accepted_current",
                    "checks": {
                        "gpx_continuity_passed": False,
                        "current_route_has_p75_time": True,
                        "current_route_has_dem_effort": True,
                        "no_better_exact_generated_candidate": True,
                        "no_dominant_boundary_recombination": True,
                        "no_dominant_global_optimizer_replacement": True,
                    },
                }
            ]
        }
    ]

    audit = module.build_audit(map_data, field_packet, human_plan, {"areas": []}, None, None, None, None, route_proofs)

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["No route exceeds 2.0x without manual/local-map proof"] == "failed"
    assert audit["summary"]["route_proofs"]["accepted_active_candidate_ids"] == []


def test_build_audit_summarizes_public_access_gated_route_proof():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.5,
        },
        "packages": [
            {
                "package_number": 27,
                "block_name": "Access-gated route",
                "components": [
                    {
                        "candidate_id": "access-gated-route",
                        "trailhead": "Trailhead",
                        "trail_names": ["Access Trail"],
                        "official_miles": 7.3,
                        "on_foot_miles": 9.64,
                        "total_minutes": 289,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "27", "trailhead": "Trailhead", "official_miles": 7.3, "on_foot_miles": 9.64}}
        ],
        "manual_holds": [],
    }
    route_proofs = [
        {
            "proofs": [
                {
                    "candidate_ids": ["access-gated-route"],
                    "status": "needs_public_access_confirmation",
                    "checks": {
                        "gpx_continuity_passed": True,
                        "current_route_has_p75_time": True,
                        "current_route_has_dem_effort": True,
                        "no_better_exact_generated_candidate": True,
                        "no_dominant_boundary_recombination": True,
                        "no_dominant_global_optimizer_replacement": True,
                    },
                }
            ]
        }
    ]

    audit = module.build_audit(
        map_data,
        field_packet,
        {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.5}},
        {"areas": []},
        None,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
        route_proofs,
    )

    route_proofs_summary = audit["summary"]["route_proofs"]
    assert route_proofs_summary["accepted_active_candidate_ids"] == []
    assert route_proofs_summary["public_access_gated_active_candidate_ids"] == ["access-gated-route"]
    assert route_proofs_summary["proof_status_counts"]["needs_public_access_confirmation"] == 1


def test_build_audit_records_generated_candidate_challenge_gate():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.7,
        },
        "packages": [
            {
                "package_number": 10,
                "block_name": "North pod",
                "components": [
                    {
                        "candidate_id": "north-pod",
                        "trailhead": "Dry Creek",
                        "trail_names": ["Spring Creek"],
                        "official_miles": 9.0,
                        "on_foot_miles": 23.0,
                        "total_minutes": 400,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "10", "trailhead": "Dry Creek", "official_miles": 9.0, "on_foot_miles": 23.0}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.7}}
    alternative_challenge = {
        "summary": {
            "target_count": 1,
            "challenged_candidate_ids": ["north-pod"],
            "better_exact_candidate_count": 0,
            "manual_map_review_still_required_count": 1,
        }
    }

    audit = module.build_audit(map_data, field_packet, human_plan, {"areas": []}, alternative_challenge)

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Generated candidate universe has been checked for better exact alternatives"] == "passed"
    assert audit["summary"]["alternative_challenge"]["target_count"] == 1
    assert audit["achieved"] is False


def test_build_audit_records_boundary_challenge_with_elevation_and_time():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.7,
        },
        "packages": [
            {
                "package_number": 13,
                "block_name": "Boundary block",
                "components": [
                    {
                        "candidate_id": "boundary-pod",
                        "trailhead": "Freestone",
                        "trail_names": ["Three Bears"],
                        "official_miles": 10.0,
                        "on_foot_miles": 18.0,
                        "total_minutes": 400,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "13", "trailhead": "Freestone", "official_miles": 10.0, "on_foot_miles": 18.0}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.7}}
    boundary_challenges = [
        {
            "package_numbers": [2, 13],
            "summary": {
                "generated_combo_beats_current": False,
                "better_generated_metric_count": 0,
                "all_covering_combos_include_elevation": True,
                "all_covering_combos_include_p75_time": True,
            },
        }
    ]

    audit = module.build_audit(
        map_data,
        field_packet,
        human_plan,
        {"areas": []},
        None,
        None,
        boundary_challenges,
    )

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Boundary recombination checks include elevation and p75 time"] == "passed"
    assert audit["summary"]["boundary_challenges"]["challenged_package_numbers"] == [2, 13]


def test_build_audit_fails_when_global_optimizer_finds_dominant_replacement():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.45,
        },
        "route_cues": {
            "current": {
                "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 90},
                "effort": {"ascent_ft": 700, "grade_adjusted_miles": 10.7, "elevation_source": "dem"},
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Current",
                "components": [
                    {
                        "candidate_id": "current",
                        "trailhead": "Trailhead",
                        "trail_names": ["Current"],
                        "official_miles": 10.0,
                        "on_foot_miles": 14.0,
                        "total_minutes": 120,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "1", "trailhead": "Trailhead", "official_miles": 10.0, "on_foot_miles": 14.0}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.45}}
    global_optimizer = {
        "summary": {
            "global_optimizer_beats_current": True,
            "dominant_solution_count": 1,
        },
        "best_dominant_solution": {
            "materially_better_metrics": ["on_foot_miles"],
            "deltas": {"on_foot_miles": 1.0},
        },
    }

    audit = module.build_audit(
        map_data,
        field_packet,
        human_plan,
        {"areas": []},
        None,
        None,
        None,
        global_optimizer,
    )

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Global executable set-cover optimizer has no dominant replacement"] == "failed"
    assert audit["summary"]["global_optimizer"]["dominant_solution_count"] == 1


def test_build_audit_fails_when_time_or_effort_estimates_are_missing_or_stale():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.45,
        },
        "route_cues": {
            "stale": {
                "total_minutes": 90,
                "time_estimates_minutes": {"door_to_door_p75": 180},
                "effort": {"ascent_ft": 100, "grade_adjusted_miles": 1.1, "elevation_source": "dem"},
            },
            "missing": {
                "total_minutes": 80,
                "time_estimates_minutes": {},
                "segments": [{"seg_id": 2}],
            },
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Stale timing",
                "components": [
                    {
                        "candidate_id": "stale",
                        "trailhead": "Trailhead",
                        "trail_names": ["A"],
                        "official_miles": 5.0,
                        "on_foot_miles": 7.0,
                        "total_minutes": 90,
                    },
                    {
                        "candidate_id": "missing",
                        "trailhead": "Trailhead",
                        "trail_names": ["B"],
                        "official_miles": 5.0,
                        "on_foot_miles": 7.0,
                        "total_minutes": 80,
                    },
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "1A", "trailhead": "Trailhead", "official_miles": 5.0, "on_foot_miles": 7.0}},
            {"outing": {"label": "1B", "trailhead": "Trailhead", "official_miles": 5.0, "on_foot_miles": 7.0}},
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.45}}

    audit = module.build_audit(map_data, field_packet, human_plan, {"areas": []})

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Runnable outings have current p75 time and DEM effort estimates"] == "failed"
    assert audit["summary"]["time_estimate_quality"]["stale_p75_count"] == 1
    assert audit["summary"]["time_estimate_quality"]["missing_p75_count"] == 1
    assert audit["summary"]["time_estimate_quality"]["missing_effort_count"] == 1


def test_build_audit_accepts_segment_level_dem_effort_when_component_effort_is_placeholder():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.45,
        },
        "route_cues": {
            "segment-effort": {
                "time_estimates_minutes": {"door_to_door_p75": 120, "moving_effort_p75": 90},
                "segments": [
                    {
                        "seg_id": 1,
                        "ascent_ft": 100,
                        "grade_adjusted_miles": 1.3,
                        "elevation_source": "dem",
                    },
                    {
                        "seg_id": 2,
                        "ascent_ft": 250,
                        "grade_adjusted_miles": 2.8,
                        "elevation_source": "dem",
                    },
                ],
            }
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Segment effort route",
                "components": [
                    {
                        "candidate_id": "segment-effort",
                        "trailhead": "Trailhead",
                        "trail_names": ["A"],
                        "official_miles": 5.0,
                        "on_foot_miles": 7.0,
                        "total_minutes": 120,
                        "effort": {"ascent_ft": 0, "grade_adjusted_miles": 0.0},
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "1", "trailhead": "Trailhead", "official_miles": 5.0, "on_foot_miles": 7.0}}
        ],
        "manual_holds": [],
    }

    audit = module.build_audit(
        map_data,
        field_packet,
        {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.45}},
        {"areas": []},
        None,
        None,
        None,
        {"summary": {"global_optimizer_beats_current": False, "dominant_solution_count": 0}},
    )

    time_quality = audit["summary"]["time_estimate_quality"]
    assert time_quality["problem_count"] == 0
    assert time_quality["problems"] == []


def test_build_audit_flags_accepted_manual_improvement_still_active():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.7,
        },
        "packages": [
            {
                "package_number": 10,
                "block_name": "North pod",
                "components": [
                    {
                        "candidate_id": "north-pod",
                        "trailhead": "Dry Creek",
                        "trail_names": ["Spring Creek"],
                        "official_miles": 9.0,
                        "on_foot_miles": 23.0,
                        "total_minutes": 400,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "10", "trailhead": "Dry Creek", "official_miles": 9.0, "on_foot_miles": 23.0}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.7}}
    manual_challenges = [
        {
            "areas": [
                {
                    "area_id": "north",
                    "title": "North split",
                    "demote_candidate_ids": ["north-pod"],
                    "current_demoted_on_foot_miles": 23.0,
                    "current_good_route": {"official_miles": 9.0, "on_foot_miles": 19.0},
                }
            ]
        }
    ]

    audit = module.build_audit(map_data, field_packet, human_plan, {"areas": []}, None, manual_challenges)

    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Accepted manual improvements have been integrated or explicitly rejected"] == "failed"
    assert audit["summary"]["manual_challenges"]["pending_integration_count"] == 1
    assert audit["summary"]["manual_challenges"]["potential_on_foot_savings_miles"] == 4.0


def test_render_md_includes_worst_components_and_next_work():
    module = load_module()
    audit = {
        "objective": "prove routes",
        "verdict": "not_proven",
        "achieved": False,
        "summary": {
            "all_component_totals": {"official_miles": 1, "on_foot_miles": 2, "ratio": 2.0},
            "runnable_field_packet_totals": {"official_miles": 1, "on_foot_miles": 2, "ratio": 2.0},
            "manual_hold_count": 1,
            "human_loop_plan_on_foot_miles": 2,
            "human_loop_plan_ratio": 2.0,
        },
        "gates": [{"gate": "A", "status": "failed", "evidence": "x"}],
        "package16": {},
        "worst_ratio_components": [
            {"label": "10", "trailhead": "Dry Creek", "official_miles": 1, "on_foot_miles": 3, "ratio": 3, "trails": ["Spring"]}
        ],
        "worst_overhead_components": [
            {
                "label": "10",
                "trailhead": "Dry Creek",
                "official_miles": 1,
                "on_foot_miles": 3,
                "overhead_miles": 2,
                "ratio": 3,
                "trails": ["Spring"],
            }
        ],
        "next_required_work": ["Challenge Dry Creek"],
    }

    rendered = module.render_md(audit)

    assert "Route Efficiency Audit" in rendered
    assert "Dry Creek" in rendered
    assert "Challenge Dry Creek" in rendered
