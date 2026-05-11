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

    applied = module.apply_segment_ownership_promotions(
        replacement_entries,
        promotions,
        current_map=current_map,
        context={"official_segments": [{"seg_id": 1656, "trail_name": "Shingle Creek Trail", "official_miles": 4.76}]},
    )

    package = replacement_entries[0]["replace_package"]
    component = package["components"][0]
    cue_segments = replacement_entries[0]["route_cues"]["target"]["segments"]
    assert applied[0]["segment_id"] == 1656
    assert component["segment_ids"] == [1542, 1546, 1656]
    assert component["official_miles"] == 11.73
    assert component["ratio"] == 1.01
    assert "Shingle Creek Trail" in component["trail_names"]
    assert [segment["seg_id"] for segment in cue_segments] == [1542, 1546, 1656]
    assert package["official_miles"] == 11.73
    assert package["component_candidate_ids"] == ["target"]
    assert replacement_entries[0]["feature_collections"]["routes"]["features"][0]["properties"]["official_miles"] == 11.73
