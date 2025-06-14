import os
import hashlib
from trail_route_ai import cache_utils


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BTAI_CACHE_DIR", str(tmp_path))
    cache_utils.clear_cache()
    data = {"foo": 1}
    cache_utils.save_cache("dist", "key", data)
    cache_file = tmp_path / f"dist_{hashlib.sha1('key'.encode()).hexdigest()[:16]}.pkl"
    assert cache_file.exists()
    loaded = cache_utils.load_cache("dist", "key")
    assert loaded == data
    cache_utils.clear_cache()
    assert not tmp_path.exists()
