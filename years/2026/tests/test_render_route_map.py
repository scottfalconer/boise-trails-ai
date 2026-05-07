import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_route_map.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_route_map", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_route_map"] = module
    spec.loader.exec_module(module)
    return module


def write_overlapping_gpx(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata><name>Overlapping Decision Route</name></metadata>
  <wpt lat="43.000000" lon="-116.000000"><name>PARK/START Test Trailhead</name></wpt>
  <wpt lat="43.001000" lon="-116.000000"><name>CUE 01 Main Junction</name></wpt>
  <wpt lat="43.000000" lon="-116.000000"><name>RETURN TO CAR</name></wpt>
  <trk>
    <name>Overlap Loop</name>
    <trkseg>
      <trkpt lat="43.000000" lon="-116.000000" />
      <trkpt lat="43.001000" lon="-116.000000" />
      <trkpt lat="43.001000" lon="-115.999000" />
      <trkpt lat="43.001000" lon="-116.000000" />
      <trkpt lat="43.001000" lon="-116.001000" />
      <trkpt lat="43.001000" lon="-116.000000" />
      <trkpt lat="43.000000" lon="-116.000000" />
    </trkseg>
  </trk>
</gpx>
""",
        encoding="utf-8",
    )


def write_context_geojson(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Muted Context Trail"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.00025, 42.9997],
                                [-116.00025, 43.0013],
                                [-115.9992, 43.0013],
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_segment_numbered_gpx(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata><name>Segment Number Trap</name></metadata>
  <wpt lat="43.000000" lon="-116.000000"><name>PARK/START Segment Trap Trailhead</name></wpt>
  <wpt lat="43.000500" lon="-116.000000"><name>SEG 99 Wrong Primary Number</name></wpt>
  <wpt lat="43.001000" lon="-116.000000"><name>CUE 01 First Decision</name></wpt>
  <wpt lat="43.001500" lon="-116.000000"><name>SEG 42 Another Segment Metadata</name></wpt>
  <wpt lat="43.002000" lon="-116.000000"><name>RETURN TO CAR</name></wpt>
  <trk>
    <name>Segment Trap Track</name>
    <trkseg>
      <trkpt lat="43.000000" lon="-116.000000" />
      <trkpt lat="43.000500" lon="-116.000000" />
      <trkpt lat="43.001000" lon="-116.000000" />
      <trkpt lat="43.001500" lon="-116.000000" />
      <trkpt lat="43.002000" lon="-116.000000" />
    </trkseg>
  </trk>
</gpx>
""",
        encoding="utf-8",
    )


def write_repeated_junction_elevation_gpx(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata><name>Napkin Decision Route</name></metadata>
  <wpt lat="43.000000" lon="-116.000000"><name>PARK/START Napkin Trailhead</name></wpt>
  <wpt lat="43.001000" lon="-116.000000"><name>CUE 01 #51 Main Junction</name></wpt>
  <wpt lat="43.001000" lon="-115.999000"><name>CUE 02 #58 Ridge Turn</name></wpt>
  <wpt lat="43.001000" lon="-116.000000"><name>CUE 03 #51 Main Junction Return</name></wpt>
  <wpt lat="43.000000" lon="-116.000000"><name>RETURN TO CAR</name></wpt>
  <trk>
    <name>Napkin Decision Track</name>
    <trkseg>
      <trkpt lat="43.000000" lon="-116.000000"><ele>100</ele></trkpt>
      <trkpt lat="43.001000" lon="-116.000000"><ele>155</ele></trkpt>
      <trkpt lat="43.001000" lon="-115.999000"><ele>220</ele></trkpt>
      <trkpt lat="43.001000" lon="-116.000000"><ele>150</ele></trkpt>
      <trkpt lat="43.001000" lon="-116.001000"><ele>118</ele></trkpt>
      <trkpt lat="43.000000" lon="-116.000000"><ele>100</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
""",
        encoding="utf-8",
    )


def test_analyze_gpx_detects_repeated_edges_nodes_and_dense_areas(tmp_path):
    renderer = load_renderer()
    gpx_path = tmp_path / "overlap.gpx"
    write_overlapping_gpx(gpx_path)

    analysis = renderer.analyze_gpx(gpx_path, renderer.RenderConfig(inset_mode="auto"))

    assert analysis.route_name == "Overlap Loop"
    assert len(analysis.points) == 7
    assert analysis.total_distance_m > 0
    assert analysis.repeated_edge_count >= 1
    assert analysis.repeated_edge_traversal_count >= 2
    assert len(analysis.repeated_nodes) >= 1
    assert analysis.repeated_nodes[0].near_waypoint is True
    assert len(analysis.dense_areas) >= 1


def test_arrow_density_increases_on_repeated_edges():
    renderer = load_renderer()
    config = renderer.RenderConfig(arrow_spacing_m=100)
    repeated = renderer.RouteSegment(
        index=0,
        start=renderer.MetricPoint(0, 0),
        end=renderer.MetricPoint(260, 0),
        distance_m=260,
        cumulative_start_m=0,
        cumulative_end_m=260,
        edge_key=((0, 0), (1, 0)),
        repeated=True,
        lane_count=2,
    )
    normal = renderer.RouteSegment(
        index=1,
        start=renderer.MetricPoint(0, 0),
        end=renderer.MetricPoint(260, 0),
        distance_m=260,
        cumulative_start_m=0,
        cumulative_end_m=260,
        edge_key=((0, 0), (1, 0)),
    )

    assert len(renderer.arrow_fractions(repeated, config, dense=False)) > len(
        renderer.arrow_fractions(normal, config, dense=False)
    )


def test_render_route_map_writes_topology_aware_png_svg_and_metadata(tmp_path):
    renderer = load_renderer()
    gpx_path = tmp_path / "overlap.gpx"
    context_path = tmp_path / "context.geojson"
    out_dir = tmp_path / "rendered"
    write_overlapping_gpx(gpx_path)
    write_context_geojson(context_path)

    result = renderer.render_route_map(
        gpx_path,
        out_dir,
        renderer.RenderConfig(width=640, height=480, inset_size=160, arrow_spacing_m=80, context_geojson=context_path),
    )

    png_path = result["overview_png"]
    svg_path = result["overview_svg"]
    metadata_path = result["metadata"]
    assert png_path.exists()
    assert svg_path.exists()
    assert metadata_path.exists()
    assert result["insets"]
    assert Path(result["insets"][0]["png"]).exists()
    assert Path(result["insets"][0]["svg"]).exists()

    with Image.open(png_path) as image:
        assert image.size == (640, 480)

    svg = svg_path.read_text(encoding="utf-8")
    assert "route-segment" in svg
    assert "direction-arrow" in svg
    assert "cue-step-label" in svg
    assert "PARK/START Test Trailhead" in svg
    assert "CUE 01 Main Junction" in svg
    assert "Detail A" in svg
    assert "context-line" in svg
    assert "Muted Context Trail" in svg
    assert "#f3f4ef" in svg  # muted basemap background
    route_strokes = re.findall(r'class="route-segment[^"]*"[^>]*stroke="([^"]+)"', svg)
    assert 1 <= len(set(route_strokes)) <= 6
    assert len(re.findall(r'class="direction-arrow', svg)) <= 60

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["analysis"]["repeated_edge_count"] >= 1
    assert metadata["analysis"]["repeated_intersection_count"] >= 1
    assert metadata["analysis"]["dense_area_count"] >= 1
    assert metadata["analysis"]["context_line_count"] == 1
    assert metadata["analysis"]["arrow_count_overview"] <= 60
    assert metadata["analysis"]["cue_count"] >= 3
    assert metadata["analysis"]["repeated_stretch_count"] >= 1
    assert metadata["analysis"]["detail_panel_count"] >= 1
    assert metadata["config"]["overlap_offset_m"] == 8.0
    assert metadata["config"]["render_mode"] == "field"
    assert metadata["config"]["field_route_color"] == "#1d4ed8"
    assert (out_dir / "nav-cues.json").exists()
    assert (out_dir / "nav-cues.csv").exists()
    assert (out_dir / "nav-cues.md").exists()
    assert (out_dir / "cue-sheet.json").exists()
    assert (out_dir / "cue-sheet.csv").exists()
    assert (out_dir / "cue-sheet.md").exists()


def test_nav_cues_do_not_use_segment_waypoint_numbers_as_primary_markers(tmp_path):
    renderer = load_renderer()
    gpx_path = tmp_path / "segment-number-trap.gpx"
    out_dir = tmp_path / "rendered"
    write_segment_numbered_gpx(gpx_path)

    renderer.render_route_map(
        gpx_path,
        out_dir,
        renderer.RenderConfig(width=640, height=480, inset_mode="off", label_density="high"),
    )

    svg = (out_dir / "route-overview.svg").read_text(encoding="utf-8")
    assert 'data-cue="1"' in svg
    assert 'data-cue="2"' in svg
    assert 'data-cue="3"' in svg
    assert 'data-cue="42"' not in svg
    assert 'data-cue="99"' not in svg
    assert "Wrong Primary Number" not in svg

    cues = json.loads((out_dir / "nav-cues.json").read_text(encoding="utf-8"))
    assert [row["cue_number"] for row in cues] == [1, 2, 3]
    assert [row["kind"] for row in cues] == ["start", "decision", "finish"]
    assert [row["mile"] for row in cues] == sorted(row["mile"] for row in cues)

    metadata = json.loads((out_dir / "route-map-metadata.json").read_text(encoding="utf-8"))
    assert metadata["analysis"]["primary_marker_mode"] == "nav-cues"
    assert metadata["analysis"]["visible_marker_numbers"] == [1, 2, 3]
    assert metadata["analysis"]["segment_waypoint_count"] == 2
    assert metadata["analysis"]["segment_labels_rendered"] is False
    assert metadata["analysis"]["omitted_segment_label_count"] == 2
    assert metadata["analysis"]["navigation_cue_count"] == 3
    assert len(metadata["analysis"]["segment_waypoints"]) == 2


def test_audit_mode_keeps_repeated_pass_diagnostics(tmp_path):
    renderer = load_renderer()
    gpx_path = tmp_path / "overlap.gpx"
    out_dir = tmp_path / "audit-rendered"
    write_overlapping_gpx(gpx_path)

    renderer.render_route_map(
        gpx_path,
        out_dir,
        renderer.RenderConfig(width=640, height=480, inset_size=160, arrow_spacing_m=80, render_mode="audit"),
    )

    svg = (out_dir / "route-map.svg").read_text(encoding="utf-8")
    assert "start-marker" in svg
    assert "finish-marker" in svg
    assert "pass-order-label" in svg
    assert "repeated-edge-lane" in svg
    route_strokes = re.findall(r'class="route-segment[^"]*"[^>]*stroke="([^"]+)"', svg)
    assert len(set(route_strokes)) > 1


def test_cli_writes_png_svg_and_accepts_marker_json(tmp_path):
    gpx_path = tmp_path / "overlap.gpx"
    marker_json = tmp_path / "markers.json"
    out_dir = tmp_path / "cli-render"
    out_dir_2 = tmp_path / "cli-render-repeat"
    write_overlapping_gpx(gpx_path)
    marker_json.write_text(
        json.dumps({"markers": [{"name": "CHECKPOINT Extra", "lat": 43.001, "lon": -116.001}]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(gpx_path),
            "--output",
            str(out_dir),
            "--style",
            "topo",
            "--insets",
            "auto",
            "--mode",
            "field",
            "--width",
            "500",
            "--height",
            "380",
            "--markers-json",
            str(marker_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "route-overview.png" in completed.stdout
    assert (out_dir / "route-overview.png").exists()
    assert (out_dir / "route-overview.svg").exists()
    assert (out_dir / "route-map.png").exists()
    assert (out_dir / "route-map.svg").exists()
    assert (out_dir / "route-map-metadata.json").exists()
    assert "CHECKPOINT Extra" in (out_dir / "route-overview.svg").read_text(encoding="utf-8")
    assert (out_dir / "cue-sheet.json").exists()

    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(gpx_path),
            "--output",
            str(out_dir_2),
            "--style",
            "topo",
            "--insets",
            "auto",
            "--mode",
            "field",
            "--width",
            "500",
            "--height",
            "380",
            "--markers-json",
            str(marker_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (out_dir / "route-overview.svg").read_text(encoding="utf-8") == (
        out_dir_2 / "route-overview.svg"
    ).read_text(encoding="utf-8")


def test_cli_imagegen_helper_profile_writes_clean_reference(tmp_path):
    gpx_path = tmp_path / "overlap.gpx"
    out_dir = tmp_path / "imagegen-helper"
    write_overlapping_gpx(gpx_path)

    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(gpx_path),
            "--output",
            str(out_dir),
            "--profile",
            "imagegen-helper",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    svg = (out_dir / "route-overview.svg").read_text(encoding="utf-8")
    metadata = json.loads((out_dir / "route-map-metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["profile"] == "imagegen-helper"
    assert metadata["config"]["width"] == 1024
    assert metadata["config"]["height"] == 1280
    assert metadata["config"]["inset_mode"] == "off"
    assert metadata["config"]["show_cue_labels"] is False
    assert metadata["config"]["show_segment_labels"] is False
    assert metadata["config"]["show_parking_marker"] is True
    assert metadata["analysis"]["detail_panel_count"] == 0
    assert metadata["analysis"]["arrow_count_overview"] <= 12
    assert "Detail A" not in svg
    assert "class=\"marker-label\">Main Junction" not in svg
    assert "parking-marker" in svg


def test_cli_napkin_profile_writes_field_decision_artifacts(tmp_path):
    gpx_path = tmp_path / "napkin.gpx"
    out_dir = tmp_path / "napkin"
    write_repeated_junction_elevation_gpx(gpx_path)

    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(gpx_path),
            "--output",
            str(out_dir),
            "--profile",
            "napkin",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (out_dir / "napkin-map.png").exists()
    assert (out_dir / "napkin-map.svg").exists()
    assert (out_dir / "napkin-cues.md").exists()
    assert (out_dir / "napkin-cues.csv").exists()
    assert (out_dir / "napkin-review.json").exists()
    assert (out_dir / "field-decisions.html").exists()
    assert (out_dir / "field-decisions.md").exists()
    assert (out_dir / "field-decisions.json").exists()

    svg = (out_dir / "napkin-map.svg").read_text(encoding="utf-8")
    assert "Schematic field guide" in svg
    assert "START / CAR" in svg
    assert "RETURN / FINISH" in svg
    assert "SEG " not in svg
    assert "parking-marker" in svg

    review = json.loads((out_dir / "napkin-review.json").read_text(encoding="utf-8"))
    assert review["profile"] == "napkin"
    assert review["source_of_truth"] == "GPX/navigation cue renderer"
    assert review["not_to_scale"] is True
    assert review["repeated_junction_cues"]
    assert not review["missing_elevation_cues"]

    rows = (out_dir / "napkin-cues.csv").read_text(encoding="utf-8").splitlines()
    assert "cue_number,distance_from_previous_miles,action,elevation_hint,label,same_as_cue_number,needs_review" in rows[0]
    assert any("STEEP UP" in row or "UP" in row for row in rows[1:])
    assert any("DOWN" in row for row in rows[1:])
    md = (out_dir / "napkin-cues.md").read_text(encoding="utf-8")
    assert "Schematic field guide - not to scale. Follow cue order." in md
    assert "same junction as cue" in md
    assert "TURN AROUND" not in md
    assert "TAKE #51" in md
    assert "TAKE #58" in md

    decision_html = (out_dir / "field-decisions.html").read_text(encoding="utf-8")
    assert "What to do next" in decision_html
    assert "Take signed #51" in decision_html
    assert "Take signed #58" in decision_html
