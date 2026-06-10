import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_latent_credit_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_latent_credit_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment(seg_id, coords, direction="both"):
    return {
        "seg_id": seg_id,
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "official_miles": 0.5,
        "direction": direction,
        "coordinates": coords,
        "start": coords[0],
        "end": coords[-1],
    }


def write_gpx(path, coords):
    points = "\n".join(f'<trkpt lat="{lat}" lon="{lon}"></trkpt>' for lon, lat in coords)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
{points}
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )


def test_latent_credit_audit_flags_segments_claimed_by_another_route(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    gpx_dir = packet_dir / "gpx"
    gpx_dir.mkdir(parents=True)
    write_gpx(
        gpx_dir / "route-a.gpx",
        [
            (-116.0, 43.0),
            (-115.99, 43.0),
            (-116.0, 43.02),
            (-115.99, 43.02),
        ],
    )
    write_gpx(gpx_dir / "route-b.gpx", [(-116.0, 43.02), (-115.99, 43.02)])
    official_segments = [
        segment(101, [(-116.0, 43.0), (-115.99, 43.0)]),
        segment(102, [(-116.0, 43.02), (-115.99, 43.02)]),
    ]
    field_tool_data = {
        "progress": {"completed_segment_ids_at_export": []},
        "routes": [
            {
                "outing_id": "route-a",
                "label": "Route A",
                "segment_ids": [101],
                "gpx_href": "gpx/route-a.gpx",
            },
            {
                "outing_id": "route-b",
                "label": "Route B",
                "segment_ids": [102],
                "gpx_href": "gpx/route-b.gpx",
            },
        ],
    }

    audit = module.build_latent_credit_audit(
        field_tool_data,
        official_segments=official_segments,
        packet_dir=packet_dir,
        threshold_miles=0.015,
        min_fraction=0.8,
    )

    assert audit["status"] == "needs_repair"
    assert audit["summary"]["routes_needing_repair"] == 1
    route = audit["routes_needing_repair"][0]
    assert route["outing_id"] == "route-a"
    assert route["unexpected_latent_segment_ids"] == ["102"]
    assert route["claimed_elsewhere_segment_ids"] == ["102"]
    assert route["segments"][0]["claimed_by_other_routes"][0]["outing_id"] == "route-b"


def test_latent_credit_audit_fails_when_one_segment_is_exact_credit_for_two_routes(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    gpx_dir = packet_dir / "gpx"
    gpx_dir.mkdir(parents=True)
    # Both routes physically cover segment 102 AND both list it in segment_ids,
    # i.e. both claim it as exact official credit (the segment-1680 defect).
    write_gpx(
        gpx_dir / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.02), (-115.99, 43.02)],
    )
    write_gpx(
        gpx_dir / "route-b.gpx",
        [(-116.0, 43.04), (-115.99, 43.04), (-116.0, 43.02), (-115.99, 43.02)],
    )
    official_segments = [
        segment(101, [(-116.0, 43.0), (-115.99, 43.0)]),
        segment(102, [(-116.0, 43.02), (-115.99, 43.02)]),
        segment(103, [(-116.0, 43.04), (-115.99, 43.04)]),
    ]
    field_tool_data = {
        "progress": {"completed_segment_ids_at_export": []},
        "routes": [
            {
                "outing_id": "route-a",
                "label": "Route A",
                "segment_ids": [101, 102],
                "gpx_href": "gpx/route-a.gpx",
            },
            {
                "outing_id": "route-b",
                "label": "Route B",
                "segment_ids": [103, 102],
                "gpx_href": "gpx/route-b.gpx",
            },
        ],
    }

    audit = module.build_latent_credit_audit(
        field_tool_data,
        official_segments=official_segments,
        packet_dir=packet_dir,
        threshold_miles=0.015,
        min_fraction=0.8,
    )

    assert audit["status"] != "passed"
    assert audit["summary"]["dual_claimed_exact_credit_segment_count"] == 1
    dual = audit["dual_claimed_exact_credit_segments"]
    assert len(dual) == 1
    assert dual[0]["seg_id"] == "102"
    claiming = {row["outing_id"] for row in dual[0]["claiming_routes"]}
    assert claiming == {"route-a", "route-b"}


def test_latent_credit_audit_accepts_declared_cross_route_ownership(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    gpx_dir = packet_dir / "gpx"
    gpx_dir.mkdir(parents=True)
    write_gpx(
        gpx_dir / "route-a.gpx",
        [
            (-116.0, 43.0),
            (-115.99, 43.0),
            (-116.0, 43.02),
            (-115.99, 43.02),
        ],
    )
    write_gpx(gpx_dir / "route-b.gpx", [(-116.0, 43.02), (-115.99, 43.02)])
    official_segments = [
        segment(101, [(-116.0, 43.0), (-115.99, 43.0)]),
        segment(102, [(-116.0, 43.02), (-115.99, 43.02)]),
    ]
    field_tool_data = {
        "progress": {"completed_segment_ids_at_export": []},
        "routes": [
            {
                "outing_id": "route-a",
                "label": "Route A",
                "segment_ids": [101],
                "gpx_href": "gpx/route-a.gpx",
                "segment_ownership_reconciliation": {
                    "declared_owned_elsewhere_segment_ids": ["102"],
                },
            },
            {
                "outing_id": "route-b",
                "label": "Route B",
                "segment_ids": [102],
                "gpx_href": "gpx/route-b.gpx",
            },
        ],
    }

    audit = module.build_latent_credit_audit(
        field_tool_data,
        official_segments=official_segments,
        packet_dir=packet_dir,
        threshold_miles=0.015,
        min_fraction=0.8,
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["routes_reconciled"] == 1
    assert audit["summary"]["routes_needing_repair"] == 0
    route = audit["reconciled_routes"][0]
    assert route["outing_id"] == "route-a"
    assert route["reconciled_claimed_elsewhere_segment_ids"] == ["102"]
    assert route["claimed_elsewhere_segment_ids"] == []
    rendered = module.render_markdown(audit)
    assert "makes the packet more executable and auditable" in rendered
    assert "does not prove lower total on-foot miles" in rendered


def test_latent_credit_audit_allows_repeat_of_already_completed_segment(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    gpx_dir = packet_dir / "gpx"
    gpx_dir.mkdir(parents=True)
    write_gpx(
        gpx_dir / "route-a.gpx",
        [
            (-116.0, 43.0),
            (-115.99, 43.0),
            (-116.0, 43.02),
            (-115.99, 43.02),
        ],
    )
    official_segments = [
        segment(101, [(-116.0, 43.0), (-115.99, 43.0)]),
        segment(102, [(-116.0, 43.02), (-115.99, 43.02)]),
    ]
    field_tool_data = {
        "progress": {"completed_segment_ids_at_export": [102]},
        "routes": [
            {
                "outing_id": "route-a",
                "label": "Route A",
                "segment_ids": [101],
                "gpx_href": "gpx/route-a.gpx",
            }
        ],
    }

    audit = module.build_latent_credit_audit(
        field_tool_data,
        official_segments=official_segments,
        packet_dir=packet_dir,
        threshold_miles=0.015,
        min_fraction=0.8,
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["repeat_only_routes"] == 1
    assert audit["repeat_only_routes"][0]["repeat_completed_segment_ids"] == ["102"]
