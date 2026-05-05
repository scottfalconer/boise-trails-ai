import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "manual_route_design_pass.py"


def load_module():
    spec = importlib.util.spec_from_file_location("manual_route_design_pass", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_route_design_report_splits_held_and_kept_components():
    module = load_module()
    package_pass = {
        "packages": [
            {
                "package_number": 16,
                "block_id": "sweet",
                "components": [
                    {
                        "candidate_id": "bad",
                        "trailhead": "Hawkins",
                        "official_miles": 2.0,
                        "on_foot_miles": 10.0,
                        "less_optimal_flags": ["long_access"],
                    },
                    {
                        "candidate_id": "good",
                        "trailhead": "Stack",
                        "official_miles": 3.0,
                        "on_foot_miles": 4.0,
                        "less_optimal_flags": [],
                    },
                ],
            }
        ]
    }
    manual_design = {
        "areas": [
            {
                "area_id": "area",
                "package_number": 16,
                "block_id": "sweet",
                "title": "Area",
                "status": "manual_design_area",
                "decision": "Hold bad.",
                "demote_candidate_ids": ["bad"],
                "keep_candidate_ids": ["good"],
                "alternatives": [
                    {
                        "alternative_id": "A",
                        "title": "Better",
                        "status": "manual_gpx_required",
                        "target_official_miles": 2.0,
                        "target_on_foot_miles_range": [1.5, 2.0],
                        "required_segment_ids": [1],
                    }
                ],
                "acceptance_gates": ["No fake gap."],
            }
        ]
    }

    report = module.build_design_report(package_pass, manual_design)
    rendered = module.render_markdown(report)

    area = report["areas"][0]
    assert area["current_demoted_on_foot_miles"] == 10.0
    assert area["acceptance_target_on_foot_miles"] == 2.0
    assert area["demoted_components"][0]["candidate_id"] == "bad"
    assert area["kept_components"][0]["candidate_id"] == "good"
    assert area["alternatives"][0]["beats_current_placeholder_by_8_miles_if"] is True
    assert "Held Placeholder Components" in rendered
    assert "Runnable/Kept Components" in rendered
