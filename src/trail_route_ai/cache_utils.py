import argparse
import hashlib
import os
import pickle
import shutil
import logging
from typing import Any
import rocksdict


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


def _rocksdb_path(name: str, key: str) -> str:
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(get_cache_dir(), f"{name}_{h}_db")


def open_rocksdb(name: str, key: str, read_only: bool = True) -> rocksdict.Rdict | None:
    path = _rocksdb_path(name, key)
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if read_only:
            # For read-only, we typically don't want to create the DB if it's missing.
            # The raw_mode=False is important as per existing options.
            opts = rocksdict.Options(raw_mode=False, create_if_missing=False)
        else:
            # For read-write, allow creation if missing (default behavior for create_if_missing is True)
            opts = rocksdict.Options(raw_mode=False, create_if_missing=True)
        return rocksdict.Rdict(path, opts)
    except Exception as e:
        # Check if the error is specifically about opening a non-existent DB in read-only mode
        # This is a common scenario and might not be a critical error for the caller if they expect this.
        if read_only and "No such file or directory" in str(e) or "does not exist" in str(e) or "NotFound" in str(e):
            logger.info(f"RocksDB at {path} not found for read-only access. This may be expected.")
            return None
        logger.error(f"Failed to open RocksDB at {path} (read_only={read_only}): {e}")
        return None


def close_rocksdb(db: rocksdict.Rdict | None) -> None:
    if db is not None:
        try:
            db.close()
        except Exception as e:
            logger.error(f"Failed to close RocksDB: {e}")


def load_rocksdb_cache(db_instance: rocksdict.Rdict | None, source_node: Any) -> Any | None:
    if db_instance is None:
        return None
    try:
        # Assuming source_node can be directly used as a key or can be serialized to a string/bytes
        key_bytes = pickle.dumps(source_node)
        value_bytes = db_instance.get(key_bytes)
        if value_bytes:
            data = pickle.loads(value_bytes)
            # logger.info(f"Loaded from RocksDB cache for source_node: {source_node}") # Becomes too verbose
            return data
        return None
    except Exception as e:
        # logger.error(f"Failed to load from RocksDB for source_node {source_node}: {e}") # Becomes too verbose
        return None


def save_rocksdb_cache(db_instance: rocksdict.Rdict | None, source_node: Any, data: Any) -> None:
    if db_instance is None:
        return
    try:
        # Assuming source_node can be directly used as a key or can be serialized to a string/bytes
        key_bytes = pickle.dumps(source_node)
        value_bytes = pickle.dumps(data)
        db_instance[key_bytes] = value_bytes
        # logger.info(f"Saved to RocksDB cache for source_node: {source_node}") # Becomes too verbose
    except Exception as e:
        # logger.error(f"Failed to save to RocksDB for source_node {source_node}: {e}") # Becomes too verbose
        pass # Fail silently like the original save_cache


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
