import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repair_official_data_drift.py"


def load_module():
    spec = importlib.util.spec_from_file_location("repair_official_data_drift", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_removed_connector_edges_become_non_credit_and_remapped_edges_stay_repeat():
    module = load_module()
    link = {
        "connector_miles": 0.0,
        "official_repeat_miles": 0.85,
        "official_repeat_segment_ids": [1663, 1664],
        "connector_edges": [
            {
                "edge_type": "official_repeat",
                "connector_class": "official_repeat",
                "source": "official_challenge",
                "seg_id": 1663,
                "name": "Stack Rock Connector",
                "distance_miles": 0.58,
                "official_traversal_direction": "reverse",
            },
            {
                "edge_type": "official_repeat",
                "connector_class": "official_repeat",
                "source": "official_challenge",
                "seg_id": 1664,
                "name": "Stack Rock Connector",
                "distance_miles": 0.27,
                "official_traversal_direction": "reverse",
            },
        ],
    }

    repaired_edges, changed = module.repair_connector_edges(
        link["connector_edges"],
        {"1664": "1762"},
        {"1663"},
    )
    link["connector_edges"] = repaired_edges
    metrics_changed = module.repair_connector_link_metrics(link)

    assert changed is True
    assert metrics_changed is True
    assert link["connector_edges"][0]["seg_id"] is None
    assert link["connector_edges"][0]["edge_type"] == "connector"
    assert link["connector_edges"][0]["connector_class"] == "retired_official_connector"
    assert link["connector_edges"][1]["seg_id"] == 1762
    assert link["official_repeat_segment_ids"] == [1762]
    assert link["official_repeat_miles"] == 0.27
    assert link["connector_miles"] == 0.58
    assert "Stack Rock Connector" in link["connector_names"]


def test_special_management_direction_overrides_flip_for_reversed_official_geometry():
    module = load_module()
    rules = {
        "rules": [
            {
                "rule_id": "polecat",
                "rule_type": "directional_segment_traversal",
                "segment_direction_overrides": {
                    "1601": ["reverse"],
                    "1603": ["forward", "reverse"],
                    "1604": ["forward"],
                },
            }
        ]
    }

    repaired, changed = module.repair_special_management_rules(rules, {"1601", "1603"})

    assert changed is True
    overrides = repaired["rules"][0]["segment_direction_overrides"]
    assert overrides["1601"] == ["forward"]
    assert overrides["1603"] == ["reverse", "forward"]
    assert overrides["1604"] == ["forward"]
