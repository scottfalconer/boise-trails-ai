import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_segment_crosswalk.py"


def load_module():
    spec = importlib.util.spec_from_file_location("derive_segment_crosswalk", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_name_similarity_handles_substrings_and_different_names():
    module = load_module()

    assert module.name_similarity("harrison hollow", "harrison hollow trail") >= 0.9
    assert module.name_similarity("harrison hollow", "polecat loop") < 0.5


def test_best_match_prefers_nearby_name_match():
    module = load_module()
    segment = {
        "normalized_trail_name": "test trail",
        "coords": [(0.0, 0.0), (0.01, 0.0)],
        "center": (0.005, 0.0),
    }
    features = [
        {
            "name": "Wrong",
            "normalized_name": "wrong",
            "parts": [[(0.1, 0.1), (0.11, 0.1)]],
            "properties": {},
            "center": (0.105, 0.1),
        },
        {
            "name": "Test Trail",
            "normalized_name": "test trail",
            "parts": [[(0.0, 0.0), (0.01, 0.0)]],
            "properties": {},
            "center": (0.005, 0.0),
        },
    ]

    match = module.best_match(segment, features, max_center_miles=20)

    assert match["feature"]["name"] == "Test Trail"
    assert match["distance_miles"] == 0


def test_normalize_connector_match_derives_connector_class_when_missing():
    module = load_module()
    match = {
        "feature": {
            "name": "Road",
            "properties": {"source": "openstreetmap", "highway": "residential", "TrailName": "Road"},
        },
        "distance_miles": 0.01,
        "name_similarity": 0.2,
    }

    normalized = module.normalize_connector_match(match)

    assert normalized["connector_class"] == "osm_public_road"
