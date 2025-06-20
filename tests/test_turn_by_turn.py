import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "planner_utils",
    pathlib.Path(__file__).resolve().parents[1] / "src" / "trail_route_ai" / "planner_utils.py",
)
planner_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner_utils)


def test_turn_by_turn_with_junction():
    e1 = planner_utils.Edge(
        "1",
        "Kestral",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    e2 = planner_utils.Edge(
        "2",
        "Red Cliffs",
        (1.0, 0.0),
        (1.0, 1.0),
        1.0,
        0.0,
        [(1.0, 0.0), (1.0, 1.0)],
        "trail",
        "ascent",
    )

    lines = planner_utils.generate_turn_by_turn([e1, e2], {"1", "2"})
    assert any("Junction" in l["text"] for l in lines[1:])
    assert "keep uphill" in lines[1]["text"]


def test_turn_by_turn_filters_zero_length_and_formats():
    e1 = planner_utils.Edge(
        "1",
        "A",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    e_small = planner_utils.Edge(
        None,
        "road",
        (1.0, 0.0),
        (1.0, 0.00001),
        0.0001,
        0.0,
        [(1.0, 0.0), (1.0, 0.00001)],
        "road",
        "both",
    )
    e2 = planner_utils.Edge(
        "2",
        "B",
        (1.0, 0.00001),
        (2.0, 0.00001),
        0.015,
        0.0,
        [(1.0, 0.00001), (2.0, 0.00001)],
        "trail",
        "both",
    )

    lines = planner_utils.generate_turn_by_turn([e1, e_small, e2])
    # The near-zero road segment should be dropped
    assert len(lines) == 2
    assert lines[0]["text"].startswith("Start on A")
    # Ensure small distances use two decimal places (0.015 -> 0.02)
    assert "0.02 mi" in lines[1]["text"]
