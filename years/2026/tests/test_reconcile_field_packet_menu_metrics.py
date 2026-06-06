import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reconcile_field_packet_menu_metrics.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("reconcile_field_packet_menu_metrics", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reconcile_map_data_updates_component_feature_and_totals():
    module = load_module()
    map_data = {
        "summary": {},
        "packages": [
            {
                "components": [
                    {
                        "candidate_id": "route-a",
                        "official_miles": 1.0,
                        "on_foot_miles": 1.5,
                        "total_minutes": 45,
                        "segment_ids": [101],
                    }
                ]
            }
        ],
        "feature_collections": {
            "routes": {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "candidate_id": "route-a",
                            "official_miles": 1.0,
                            "on_foot_miles": 1.5,
                        },
                        "geometry": None,
                    }
                ]
            }
        },
    }
    field_tool_data = {
        "routes": [
            {
                "candidate_ids": ["route-a"],
                "official_miles": 1.0,
                "on_foot_miles": 1.8,
                "door_to_door_minutes_p75": 55,
                "segment_ids": ["101", "102"],
            }
        ]
    }

    module.reconcile_map_data(map_data, field_tool_data)

    component = map_data["packages"][0]["components"][0]
    props = map_data["feature_collections"]["routes"]["features"][0]["properties"]
    assert component["on_foot_miles"] == 1.8
    assert component["source_card_on_foot_miles"] == 1.5
    assert component["total_minutes"] == 55
    assert component["segment_ids"] == [101, 102]
    assert props["on_foot_miles"] == 1.8
    assert props["source_card_on_foot_miles"] == 1.5
    assert map_data["packages"][0]["on_foot_miles"] == 1.8
    assert map_data["summary"]["total_on_foot_miles"] == 1.8
    assert map_data["field_packet_metric_reconciliation"]["updated_candidate_count"] == 1
