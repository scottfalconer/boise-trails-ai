import os
from trail_route_ai import cache_utils


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BTAI_CACHE_DIR", str(tmp_path))
    cache_utils.clear_cache()
    data = {"foo": 1}
    cache_utils.save_cache("dist", "key", data)
    loaded = cache_utils.load_cache("dist", "key")
    assert loaded == data
    cache_utils.clear_cache()
    assert not any(tmp_path.iterdir())
