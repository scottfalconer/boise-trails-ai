import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPT_DIR / "build_routing_connector_overlay.py"


def load_builder():
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("build_routing_connector_overlay", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def osm_feature(highway="primary", access=None, foot=None):
    props = {"highway": highway}
    if access is not None:
        props["access"] = access
    if foot is not None:
        props["foot"] = foot
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "LineString",
            "coordinates": [[-116.205, 43.626], [-116.204, 43.626]],
        },
    }


def test_public_road_connectors_are_allowed_and_classified():
    builder = load_builder()
    feature = osm_feature(highway="primary")

    assert builder.osm_feature_is_usable(feature) is True

    normalized = builder.normalize_osm_feature(feature, 1)
    assert normalized["properties"]["source"] == "openstreetmap"
    assert normalized["properties"]["connector_class"] == "osm_public_road"


def test_private_or_no_foot_osm_connectors_are_blocked():
    builder = load_builder()

    assert builder.osm_feature_is_usable(osm_feature(highway="residential", access="private")) is False
    assert builder.osm_feature_is_usable(osm_feature(highway="footway", foot="no")) is False
