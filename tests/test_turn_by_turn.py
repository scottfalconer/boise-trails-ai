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
