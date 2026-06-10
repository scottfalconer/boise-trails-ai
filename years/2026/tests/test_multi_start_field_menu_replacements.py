import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "multi_start_field_menu_replacements.py"


def load_module():
    spec = importlib.util.spec_from_file_location("multi_start_field_menu_replacements", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_alternatives_default_to_promising_no_blocker_best_savings():
    module = load_module()
    audit = {
        "outings": [
            {
                "label": "1A",
                "alternatives": [
                    {
                        "alternative_id": "blocked-best",
                        "status": "promising",
                        "parking_blockers": ["needs review"],
                        "on_foot_savings_miles": 9.0,
                        "elapsed_delta_minutes": -10,
                    },
                    {
                        "alternative_id": "accepted-second",
                        "status": "promising",
                        "parking_blockers": [],
                        "on_foot_savings_miles": 2.0,
                        "elapsed_delta_minutes": 5,
                    },
                    {
                        "alternative_id": "accepted-best",
                        "status": "promising",
                        "parking_blockers": [],
                        "on_foot_savings_miles": 3.0,
                        "elapsed_delta_minutes": 12,
                    },
                ],
            },
            {
                "label": "13",
                "alternatives": [
                    {
                        "alternative_id": "rejected",
                        "status": "not_worth_it",
                        "parking_blockers": [],
                        "on_foot_savings_miles": 5.0,
                        "elapsed_delta_minutes": -5,
                    }
                ],
            },
        ]
    }

    assert module.selected_alternatives_from_audit(audit) == {"1A": "accepted-best"}


def test_package_source_allows_rerun_against_already_replaced_active_map():
    module = load_module()
    current_map = {
        "packages": [
            {
                "package_number": 4,
                "components": [
                    {
                        "candidate_id": "multi-start-4c-a",
                        "route_number": 31,
                        "source": module.OVERRIDE_SOURCE,
                        "multi_start_alternative_id": "4C-MS-20",
                    },
                    {
                        "candidate_id": "multi-start-4c-b",
                        "route_number": 32,
                        "source": module.OVERRIDE_SOURCE,
                        "multi_start_alternative_id": "4C-MS-20",
                    },
                    {"candidate_id": "bobs-trail", "route_number": 1, "field_menu_label": "4A"},
                    {"candidate_id": "scotts-trail", "route_number": 2, "field_menu_label": "4B"},
                ],
            }
        ]
    }

    package, source = module.package_source_for_replacement(
        current_map=current_map,
        fallback_packages=[],
        package_number=4,
        baseline_candidate_id="combo-table-rock-original",
        alternative_id="4C-MS-20",
    )

    generated = module.generated_components_for_alternative(package, "4C-MS-20")
    generated_ids = {component["candidate_id"] for component in generated}
    kept_ids = [
        component["candidate_id"]
        for component in package["components"]
        if component["candidate_id"] not in generated_ids
    ]

    assert source == "already_replaced"
    assert module.route_number_base_from_generated_components(generated) == 3
    assert kept_ids == ["bobs-trail", "scotts-trail"]


def test_package_source_can_use_public_base_override_as_pristine_source():
    module = load_module()
    current_map = {
        "packages": [
            {
                "package_number": 1,
                "components": [
                    {
                        "candidate_id": "multi-start-1a-a",
                        "source": module.OVERRIDE_SOURCE,
                        "multi_start_alternative_id": "1A-MS-04",
                    }
                ],
            }
        ]
    }
    fallback_packages = [
        {
            "package_number": 1,
            "components": [
                {"candidate_id": "combo-west-climb-original", "route_number": 1},
                {"candidate_id": "harrison-hollow", "route_number": 2},
            ],
        }
    ]

    package, source = module.package_source_for_replacement(
        current_map=current_map,
        fallback_packages=fallback_packages,
        package_number=1,
        baseline_candidate_id="combo-west-climb-original",
        alternative_id="1A-MS-04",
    )

    assert source == "baseline"
    assert [component["candidate_id"] for component in package["components"]] == [
        "combo-west-climb-original",
        "harrison-hollow",
    ]


def test_package_source_can_use_fallback_when_current_map_package_is_absent():
    module = load_module()
    fallback_packages = [
        {
            "package_number": 1,
            "components": [
                {"candidate_id": "combo-frontside", "route_number": 1},
                {"candidate_id": "harrison-hollow", "route_number": 2},
            ],
        }
    ]

    package, source = module.package_source_for_replacement(
        current_map={"packages": []},
        fallback_packages=fallback_packages,
        package_number=1,
        baseline_candidate_id="combo-frontside",
        alternative_id="1A-MS-04",
    )

    assert source == "baseline"
    assert [component["candidate_id"] for component in package["components"]] == [
        "combo-frontside",
        "harrison-hollow",
    ]


def test_package_source_can_use_existing_generated_output_as_fallback():
    module = load_module()
    fallback_packages = [
        {
            "package_number": 4,
            "components": [
                {
                    "candidate_id": "multi-start-4c-a",
                    "source": module.OVERRIDE_SOURCE,
                    "multi_start_alternative_id": "4C-MS-20",
                },
                {"candidate_id": "scotts-trail"},
            ],
        }
    ]

    package, source = module.package_source_for_replacement(
        current_map={"packages": []},
        fallback_packages=fallback_packages,
        package_number=4,
        baseline_candidate_id="combo-table-rock-original",
        alternative_id="4C-MS-20",
    )

    assert source == "already_replaced"
    assert [component["candidate_id"] for component in package["components"]] == [
        "multi-start-4c-a",
        "scotts-trail",
    ]


def test_candidate_has_route_source_requires_route_and_parking_features():
    module = load_module()
    current_map = {
        "feature_collections": {
            "routes": {
                "features": [
                    {"properties": {"candidate_id": "complete"}},
                    {"properties": {"candidate_id": "no-parking"}},
                ]
            },
            "parking": {
                "features": [
                    {"properties": {"candidate_id": "complete"}},
                    {"properties": {"candidate_id": "no-route"}},
                ]
            },
        }
    }

    assert module.candidate_has_route_source(current_map, "complete") is True
    assert module.candidate_has_route_source(current_map, "no-parking") is False
    assert module.candidate_has_route_source(current_map, "no-route") is False


def test_component_segment_ids_excluding_kept_removes_already_owned_segments():
    module = load_module()
    kept = [
        {"candidate_id": "sweet-connie", "segment_ids": [1665, 1666, 1667]},
        {"candidate_id": "sheep-camp", "segment_ids": [1653]},
    ]

    remaining, removed = module.component_segment_ids_excluding_kept(
        {"segment_ids": [1665, 1666, 1667, 1656, 1653]},
        module.kept_component_segment_ids(kept),
    )

    assert remaining == [1656]
    assert removed == [1665, 1666, 1667, 1653]


def test_segment_ownership_promotion_claims_latent_segment_on_target_route(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "segment_reviews": [
                    {
                        "seg_id": "1656",
                        "completion_status": "completed",
                        "match_fraction": 1.0,
                        "endpoints_ok": True,
                        "direction_ok": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    replacement_entries = [
        {
            "replace_package": {
                "package_number": 15,
                "components": [
                    {
                        "candidate_id": "target",
                        "trail_names": ["Dry Creek Trail"],
                        "official_miles": 6.97,
                        "on_foot_miles": 11.89,
                        "segment_ids": [1542, 1546],
                        "total_minutes": 229,
                        "trailhead": "Dry/Sweet",
                    }
                ],
            },
            "route_cues": {
                "target": {
                    "candidate_id": "target",
                    "segments": [
                        {"seg_id": 1542, "trail_name": "Dry Creek Trail", "official_miles": 0.58},
                        {"seg_id": 1546, "trail_name": "Dry Creek Trail", "official_miles": 1.65},
                    ],
                }
            },
            "feature_collections": {
                "routes": {
                    "features": [
                        {"properties": {"candidate_id": "target", "official_miles": 6.97, "title": "Dry Creek Trail"}}
                    ]
                }
            },
        }
    ]
    current_map = {
        "route_cues": {
            "source": {
                "segments": [
                    {
                        "seg_id": 1656,
                        "trail_name": "Shingle Creek Trail",
                        "official_miles": 4.76,
                        "direction": "ascent",
                    }
                ]
            }
        }
    }
    promotions = {
        "promotions": [
            {
                "status": "promoted",
                "segment_id": 1656,
                "reason": "already covered by target GPX",
                "from": {"candidate_id": "source"},
                "to": {
                    "package_number": 15,
                    "candidate_id": "target",
                    "insert_after_segment_id": 1546,
                },
                "evidence": {
                    "activity_review_json": "review.json",
                    "required_status": "completed",
                    "min_match_fraction": 0.85,
                    "requires_endpoints_ok": True,
                    "requires_direction_ok": True,
                },
            }
        ]
    }

    applied, skipped = module.apply_segment_ownership_promotions(
        replacement_entries,
        promotions,
        current_map=current_map,
        context={"official_segments": [{"seg_id": 1656, "trail_name": "Shingle Creek Trail", "official_miles": 4.76}]},
    )

    package = replacement_entries[0]["replace_package"]
    component = package["components"][0]
    cue_segments = replacement_entries[0]["route_cues"]["target"]["segments"]
    assert skipped == []
    assert applied[0]["segment_id"] == 1656
    assert component["segment_ids"] == [1542, 1546, 1656]
    assert component["official_miles"] == 11.73
    assert component["ratio"] == 1.01
    assert "Shingle Creek Trail" in component["trail_names"]
    assert [segment["seg_id"] for segment in cue_segments] == [1542, 1546, 1656]
    assert package["official_miles"] == 11.73
    assert package["component_candidate_ids"] == ["target"]


def test_field_latent_credit_evidence_can_support_segment_promotion(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    audit_path = tmp_path / "latent.json"
    audit_path.write_text(
        json.dumps(
            {
                "route_reviews": [
                    {
                        "route_key": "114-2",
                        "audit_status": "passed",
                        "latent_completed_segment_ids": ["1610"],
                        "segments": [
                            {
                                "seg_id": "1610",
                                "status": "reconciled_owned_elsewhere",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert module.promotion_evidence_passed(
        {
            "segment_id": 1610,
            "evidence": {
                "field_latent_credit_audit_json": "latent.json",
                "route_key": "114-2",
                "required_status": "reconciled_owned_elsewhere",
            },
        },
        tmp_path,
    )


def test_same_package_promotion_creates_baseline_entry_and_removes_source_card(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    audit_path = tmp_path / "latent.json"
    audit_path.write_text(
        json.dumps(
            {
                "route_reviews": [
                    {
                        "route_key": "114-2",
                        "audit_status": "passed",
                        "latent_completed_segment_ids": ["1610"],
                        "segments": [{"seg_id": "1610", "status": "reconciled_owned_elsewhere"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    current_map = {
        "packages": [
            {
                "package_number": 114,
                "components": [
                    {
                        "candidate_id": "target",
                        "field_menu_label": "FD14B",
                        "trail_names": ["CHBH Connector"],
                        "official_miles": 0.81,
                        "on_foot_miles": 3.16,
                        "segment_ids": [1516],
                        "total_minutes": 103,
                        "trailhead": "Cartwright",
                    },
                    {
                        "candidate_id": "source",
                        "field_menu_label": "FD14C",
                        "trail_names": ["Quick Draw"],
                        "official_miles": 0.48,
                        "on_foot_miles": 1.63,
                        "segment_ids": [1610],
                        "total_minutes": 68,
                        "trailhead": "Cartwright",
                    },
                    {
                        "candidate_id": "kept",
                        "field_menu_label": "FD14D",
                        "trail_names": ["36th Street Chute"],
                        "official_miles": 0.74,
                        "on_foot_miles": 2.0,
                        "segment_ids": [1482],
                        "total_minutes": 73,
                        "trailhead": "Full Sail",
                    },
                ],
            }
        ],
        "route_cues": {
            "target": {
                "candidate_id": "target",
                "segments": [{"seg_id": 1516, "trail_name": "CHBH Connector", "official_miles": 0.81}],
                "start_access": {"official_repeat_segment_ids": [1541, 1610]},
            },
            "source": {
                "candidate_id": "source",
                "segments": [{"seg_id": 1610, "trail_name": "Quick Draw", "official_miles": 0.48}],
            },
            "kept": {"candidate_id": "kept", "segments": []},
        },
        "feature_collections": {
            "routes": {
                "features": [
                    {"properties": {"candidate_id": "target", "official_miles": 0.81}},
                    {"properties": {"candidate_id": "source", "official_miles": 0.48}},
                    {"properties": {"candidate_id": "kept", "official_miles": 0.74}},
                ]
            },
            "parking": {
                "features": [
                    {"properties": {"candidate_id": "target"}},
                    {"properties": {"candidate_id": "source"}},
                    {"properties": {"candidate_id": "kept"}},
                ]
            },
        },
        "map_validation": {
            "route_validations": [
                {"candidate_id": "target"},
                {"candidate_id": "source"},
                {"candidate_id": "kept"},
            ]
        },
    }
    replacement_entries = []
    promotions = {
        "promotions": [
            {
                "status": "promoted",
                "segment_id": 1610,
                "reason": "Quick Draw is already physically covered by FD14B.",
                "source_action": "remove_route_card",
                "from": {"package_number": 114, "candidate_id": "source"},
                "to": {"package_number": 114, "candidate_id": "target", "insert_after_segment_id": 1516},
                "evidence": {
                    "field_latent_credit_audit_json": "latent.json",
                    "route_key": "114-2",
                    "required_status": "reconciled_owned_elsewhere",
                },
            }
        ]
    }

    applied, skipped = module.apply_segment_ownership_promotions(
        replacement_entries,
        promotions,
        current_map=current_map,
        context={"official_segments": [{"seg_id": 1610, "trail_name": "Quick Draw", "official_miles": 0.48}]},
    )

    assert skipped == []
    assert applied[0]["segment_id"] == 1610
    assert len(replacement_entries) == 1
    entry = replacement_entries[0]
    assert entry["remove_candidate_ids"] == ["target", "source", "kept"]
    assert [component["candidate_id"] for component in entry["replace_package"]["components"]] == ["target", "kept"]
    target = entry["replace_package"]["components"][0]
    assert target["segment_ids"] == [1516, 1610]
    assert target["official_miles"] == 1.29
    assert entry["route_cues"]["target"]["start_access"]["official_repeat_segment_ids"] == [1541]
    assert "source" not in entry["route_cues"]
    assert [
        feature["properties"]["candidate_id"]
        for feature in entry["feature_collections"]["routes"]["features"]
    ] == ["target", "kept"]
    assert [row["candidate_id"] for row in entry["route_validations"]] == ["target", "kept"]


def test_obsolete_segment_promotion_target_is_skipped_before_evidence_lookup(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    replacement_entries = []
    current_map = {
        "packages": [
            {
                "package_number": 15,
                "components": [{"candidate_id": "current-target"}],
            }
        ]
    }
    promotions = {
        "promotions": [
            {
                "status": "promoted",
                "segment_id": 1610,
                "from": {"package_number": 114, "candidate_id": "quick-draw"},
                "to": {"package_number": 114, "candidate_id": "chbh-connector"},
                "evidence": {
                    "field_latent_credit_audit_json": "missing-obsolete-evidence.json",
                    "route_key": "114-2",
                    "required_status": "reconciled_owned_elsewhere",
                },
            }
        ]
    }

    applied, skipped = module.apply_segment_ownership_promotions(
        replacement_entries,
        promotions,
        current_map=current_map,
        context={"official_segments": []},
    )

    assert applied == []
    assert replacement_entries == []
    assert skipped == [
        {
            "segment_id": 1610,
            "status": "skipped",
            "reason": "target_package_not_in_current_map",
            "from": {"package_number": 114, "candidate_id": "quick-draw"},
            "to": {"package_number": 114, "candidate_id": "chbh-connector"},
        }
    ]


def test_obsolete_promotion_target_in_replaced_package_is_skipped_before_evidence_lookup(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    current_map = {
        "packages": [
            {
                "package_number": 15,
                "components": [
                    {"candidate_id": "stale-target"},
                    {"candidate_id": "current-target"},
                ],
            }
        ]
    }
    replacement_entries = [
        {
            "replace_package": {
                "package_number": 15,
                "components": [{"candidate_id": "current-target"}],
            }
        }
    ]
    promotions = {
        "promotions": [
            {
                "status": "promoted",
                "segment_id": 1656,
                "from": {"package_number": 16, "candidate_id": "source"},
                "to": {"package_number": 15, "candidate_id": "stale-target"},
                "evidence": {
                    "activity_review_json": "missing-stale-target-evidence.json",
                    "required_status": "completed",
                },
            }
        ]
    }

    applied, skipped = module.apply_segment_ownership_promotions(
        replacement_entries,
        promotions,
        current_map=current_map,
        context={"official_segments": []},
    )

    assert applied == []
    assert skipped == [
        {
            "segment_id": 1656,
            "status": "skipped",
            "reason": "target_candidate_not_in_current_package",
            "from": {"package_number": 16, "candidate_id": "source"},
            "to": {"package_number": 15, "candidate_id": "stale-target"},
        }
    ]


def test_cross_package_promotion_can_remove_later_source_card(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    audit_path = tmp_path / "latent.json"
    audit_path.write_text(
        json.dumps(
            {
                "route_reviews": [
                    {
                        "route_key": "123-1",
                        "audit_status": "passed",
                        "latent_completed_segment_ids": ["1576"],
                        "segments": [{"seg_id": "1576", "status": "reconciled_owned_elsewhere"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    current_map = {
        "packages": [
            {
                "package_number": 123,
                "components": [
                    {
                        "candidate_id": "target",
                        "trail_names": ["Corrals Trail"],
                        "official_miles": 7.81,
                        "on_foot_miles": 12.86,
                        "segment_ids": [1528],
                        "total_minutes": 262,
                        "trailhead": "8th Street ATV Parking Area",
                    }
                ],
            },
            {
                "package_number": 122,
                "components": [
                    {
                        "candidate_id": "source",
                        "trail_names": ["Highlands Trail"],
                        "official_miles": 1.06,
                        "on_foot_miles": 2.76,
                        "segment_ids": [1576],
                        "total_minutes": 79,
                        "trailhead": "Bob's",
                    },
                    {
                        "candidate_id": "kept",
                        "trail_names": ["Crestline Trail"],
                        "official_miles": 1.82,
                        "on_foot_miles": 4.46,
                        "segment_ids": [1532],
                        "total_minutes": 104,
                        "trailhead": "Hulls Gulch",
                    },
                ],
            },
        ],
        "route_cues": {
            "target": {
                "candidate_id": "target",
                "segments": [{"seg_id": 1528, "trail_name": "Corrals Trail", "official_miles": 0.4}],
                "return_to_car": {"official_repeat_segment_ids": [1528, 1576]},
            },
            "source": {
                "candidate_id": "source",
                "segments": [{"seg_id": 1576, "trail_name": "Highlands Trail", "official_miles": 1.06}],
            },
            "kept": {"candidate_id": "kept", "segments": []},
        },
        "feature_collections": {"routes": {"features": []}, "parking": {"features": []}},
        "map_validation": {"route_validations": []},
    }
    replacement_entries = []
    promotions = {
        "promotions": [
            {
                "status": "promoted",
                "segment_id": 1576,
                "source_action": "remove_route_card",
                "from": {"package_number": 122, "candidate_id": "source"},
                "to": {"package_number": 123, "candidate_id": "target", "insert_after_segment_id": 1528},
                "evidence": {
                    "field_latent_credit_audit_json": "latent.json",
                    "route_key": "123-1",
                    "required_status": "reconciled_owned_elsewhere",
                },
            }
        ]
    }

    applied, skipped = module.apply_segment_ownership_promotions(
        replacement_entries,
        promotions,
        current_map=current_map,
        context={"official_segments": [{"seg_id": 1576, "trail_name": "Highlands Trail", "official_miles": 1.06}]},
    )

    assert skipped == []
    assert applied[0]["segment_id"] == 1576
    entries_by_package = {
        str(entry["replace_package"]["package_number"]): entry
        for entry in replacement_entries
    }
    assert set(entries_by_package) == {"122", "123"}
    assert entries_by_package["123"]["replace_package"]["components"][0]["segment_ids"] == [1528, 1576]
    assert entries_by_package["123"]["route_cues"]["target"]["return_to_car"]["official_repeat_segment_ids"] == [1528]
    assert [component["candidate_id"] for component in entries_by_package["122"]["replace_package"]["components"]] == ["kept"]
