from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_relaxed_drive_day_gpx_export as export  # noqa: E402


def test_loop_endpoint_gap_miles_uses_first_and_last_segment_points():
    segments = [
        [(-116.0, 43.0), (-116.001, 43.001)],
        [(-116.001, 43.001), (-116.0, 43.0)],
    ]

    assert export.loop_endpoint_gap_miles(segments) == 0.0


def test_validate_loop_track_fails_large_endpoint_gap():
    segments = [[(-116.0, 43.0), (-116.1, 43.1)]]

    validation = export.validate_loop_track(
        segments,
        max_gap_miles=20,
        max_endpoint_gap_miles=0.01,
    )

    assert validation["passed"] is False
    assert validation["failures"][0]["code"] == "loop_endpoint_gap_exceeded"


def test_parse_gpx_track_segments_reads_namespace(tmp_path: Path):
    path = tmp_path / "track.gpx"
    path.write_text(
        """<?xml version="1.0"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
<trkpt lat="43.1" lon="-116.1" />
<trkpt lat="43.2" lon="-116.2" />
</trkseg></trk></gpx>
""",
        encoding="utf-8",
    )

    assert export.parse_gpx_track_segments(path) == [[(-116.1, 43.1), (-116.2, 43.2)]]


def test_stitch_remaining_track_gaps_inserts_graph_path(monkeypatch):
    def fake_shortest_connector_path(start, end, connector_graph, snap_tolerance):
        assert connector_graph == {"graph": True}
        assert snap_tolerance == 0.2
        return {"path_coordinates": [start, (-116.05, 43.05), end]}

    monkeypatch.setattr(export, "shortest_connector_path", fake_shortest_connector_path)

    repaired = export.stitch_remaining_track_gaps(
        [(-116.0, 43.0), (-116.1, 43.1)],
        connector_graph={"graph": True},
        max_gap_miles=0.05,
        stitch_snap_tolerance_miles=0.2,
    )

    assert repaired == [(-116.0, 43.0), (-116.05, 43.05), (-116.1, 43.1)]
