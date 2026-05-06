import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_segment_elevation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("derive_segment_elevation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_elevation_rows_include_reverse_for_bidirectional_only():
    module = load_module()
    segments = [
        {
            "seg_id": 1,
            "seg_name": "Both 1",
            "trail_name": "Both",
            "official_miles": 1.0,
            "direction": "both",
            "coordinates": [(0.0, 0.0), (0.01, 0.0)],
        },
        {
            "seg_id": 2,
            "seg_name": "Up 1",
            "trail_name": "Up",
            "official_miles": 1.0,
            "direction": "ascent",
            "coordinates": [(0.0, 0.0), (0.01, 0.0)],
        },
    ]

    rows = module.elevation_rows(segments, lambda point: point[0] * 1000)

    assert [(row["seg_id"], row["traversal"]) for row in rows] == [
        (1, "forward"),
        (1, "reverse"),
        (2, "forward"),
    ]
    assert rows[1]["allowed_for_credit"] is True
    assert rows[2]["allowed_for_credit"] is True
