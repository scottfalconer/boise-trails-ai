import argparse
import hashlib
import os
import pickle
import shutil
import logging
from typing import Any


# Default location for cached data when no environment override is provided
DEFAULT_CACHE_DIR = os.path.expanduser("~/.boise_trails_ai_cache")

logger = logging.getLogger(__name__)


def get_cache_dir() -> str:
    """Return the directory used for cached files."""

    # Re-read the environment variable each call so tests or callers may
    # override the location after this module is imported.
    path = os.environ.get("BTAI_CACHE_DIR", DEFAULT_CACHE_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(name: str, key: str) -> str:
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(get_cache_dir(), f"{name}_{h}.pkl")


def load_cache(name: str, key: str) -> Any | None:
    path = _cache_path(name, key)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info("Loaded cache %s:%s", name, key)
            return data
        except Exception:
            return None
    return None


def save_cache(name: str, key: str, data: Any) -> None:
    path = _cache_path(name, key)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved cache %s:%s", name, key)
    except Exception:
        pass


def clear_cache() -> None:
    dir_path = get_cache_dir()
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        logger.info("Cleared cache directory %s", dir_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage boise-trails-ai cache")
    parser.add_argument("--clear", action="store_true", help="remove all cached data")
    args = parser.parse_args(argv)
    if args.clear:
        clear_cache()
        print(f"Cache cleared: {get_cache_dir()}")


if __name__ == "__main__":
    main()
