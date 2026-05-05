import subprocess

import geopandas as gpd

import clip_pbf


def test_clip_pbf_invokes_osmium(monkeypatch, tmp_path):
    # minimal GeoJSON for bounding box calculation
    trails = tmp_path / "trails.geojson"
    gdf = gpd.GeoDataFrame(geometry=[])
    gdf.to_file(trails, driver="GeoJSON")

    pbf = tmp_path / "in.pbf"
    pbf.write_text("")
    out = tmp_path / "out.pbf"

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        out.write_text("")
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)

    clip_pbf.clip_pbf(str(pbf), str(trails), str(out), buffer_km=0.0)

    assert calls
    assert calls[0][0] == "osmium"
    assert out.exists()
