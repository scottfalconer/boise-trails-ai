import importlib.util
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
