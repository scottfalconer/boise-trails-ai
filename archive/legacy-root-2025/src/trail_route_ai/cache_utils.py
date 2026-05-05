import argparse
import hashlib
import os
import pickle
import shutil
import logging
from typing import Any, Optional, Dict, List
import rocksdict


class MemoryRocksDB(dict):
    """Simple in-memory stand-in for ``rocksdict.Rdict`` used in tests."""

    def close(self) -> None:
        pass


# Default location for cached data when no environment override is provided
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "trail_route_ai")

logger = logging.getLogger(__name__)


def get_cache_dir() -> str:
    """Return the directory used for cached files."""
    path = os.environ.get("BTAI_CACHE_DIR", DEFAULT_CACHE_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _rocksdb_path(name: str, key: str) -> str:
    """Return the path to a RocksDB cache file."""
    return os.path.join(get_cache_dir(), name, key)


def open_rocksdb(name: str, key: str, read_only: bool = True) -> Optional[rocksdict.Rdict]:
    path = _rocksdb_path(name, key)
    # Ensure parent directory of the cache store exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if read_only and not os.path.exists(path):
        # If read-only and DB dir doesn't exist, don't even try to open
        logger.info(
            f"RocksDB at {path} not found for read-only access. Path does not exist."
        )
        return None

    try:
        opts = rocksdict.Options(raw_mode=False)
        try:
            opts.set_max_open_files(100)
            logger.info("Set RocksDB max_open_files to 100.")
        except AttributeError:
            logger.warning("opts.set_max_open_files() not available. Trying direct assignment.")
            try:
                opts.max_open_files = 100
                logger.info("Set RocksDB max_open_files to 100 via direct assignment.")
            except AttributeError:
                logger.error("Failed to set max_open_files on RocksDB Options object.")
        if read_only:
            # For read-only, we typically don't want to create the DB if it's missing.
            # create_if_missing(False) should prevent creation if path does not exist,
            # but Rdict might still create parent dirs or an empty dir before erroring.
            # The check above handles non-existent path explicitly for read_only.
            opts.create_if_missing(False)
        else:
            # For read-write, allow creation if missing
            opts.create_if_missing(True)

        # For read_only mode, if path exists but Rdict fails to open (e.g. corrupted, not a DB),
        # the exception handling below will catch it.
        return rocksdict.Rdict(path, opts)
    except (OSError, rocksdict.DbClosedError) as e:
        # This will catch cases where path exists but is not a valid DB or other Rdict open errors
        # Added "Invalid argument" as it's common for RocksDB open issues on an existing non-DB path or corrupted DB
        if read_only and ("No such file or directory" in str(e) or "does not exist" in str(e) or "NotFound" in str(e) or "Invalid argument" in str(e)):
            logger.info(f"RocksDB at {path} not found or failed to open for read-only access: {e}")
            return None
        logger.error(f"Failed to open RocksDB at {path} (read_only={read_only}): {e}")
        logger.warning(
            "Falling back to in-memory cache due to RocksDB open failure; performance may degrade"
        )
        return MemoryRocksDB()


def close_rocksdb(db: Optional[rocksdict.Rdict]) -> None:
    if db is not None:
        try:
            db.close()
        except (OSError, rocksdict.DbClosedError) as e:
            logger.error("Failed to close RocksDB: %s", e)
            raise


def load_rocksdb_cache(db_instance: Optional[rocksdict.Rdict], source_node: Any) -> Optional[Any]:
    """Return cached value for ``source_node`` or ``None`` if unavailable."""

    if db_instance is None:
        return None
    try:
        key_bytes = pickle.dumps(source_node)
    except pickle.PickleError as e:  # pragma: no cover - unexpected type
        logger.error("Failed to serialize RocksDB key %s: %s", source_node, e)
        return None

    try:
        value_bytes = db_instance.get(key_bytes)
    except (OSError, rocksdict.DbClosedError) as e:  # pragma: no cover - DB errors
        logger.error("RocksDB read error for %s: %s", source_node, e)
        return None

    if not value_bytes:
        return None

    try:
        return pickle.loads(value_bytes)
    except pickle.UnpicklingError as e:  # pragma: no cover - corrupted entry
        logger.error("Corrupted RocksDB entry for %s: %s", source_node, e)
        return None


def save_rocksdb_cache(db_instance: Optional[rocksdict.Rdict], source_node: Any, data: Any) -> None:
    if db_instance is None:
        return
    try:
        key_bytes = pickle.dumps(source_node)
        value_bytes = pickle.dumps(data)
    except pickle.PickleError as e:  # pragma: no cover - unexpected type
        logger.error("Failed to serialize RocksDB entry for %s: %s", source_node, e)
        return

    try:
        db_instance[key_bytes] = value_bytes
    except (OSError, rocksdict.DbClosedError) as e:  # pragma: no cover - DB errors
        logger.error("RocksDB write error for %s: %s", source_node, e)


def clear_cache() -> None:
    """Delete the entire cache directory."""
    shutil.rmtree(get_cache_dir(), ignore_errors=True)
    print(f"Cache cleared: {get_cache_dir()}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Manage boise-trails-ai cache")
    parser.add_argument("--clear", action="store_true", help="remove all cached data")
    args = parser.parse_args(argv)
    if args.clear:
        clear_cache()
        print(f"Cache cleared: {get_cache_dir()}")


if __name__ == "__main__":
    # Test functions
    print(f"Cache directory: {get_cache_dir()}")

    # Clear cache for a clean test
    clear_cache()
    print("Cache cleared.")

    # Test RocksDB operations
    db_instance = open_rocksdb("test_db", "test_key", read_only=False)
    if db_instance:
        print("RocksDB instance opened.")
        
        # Test save/load
        save_rocksdb_cache(db_instance, "node1", {"path": [1, 2, 3]})
        cached_data = load_rocksdb_cache(db_instance, "node1")
        print(f"Loaded data for node1: {cached_data}")
        assert cached_data == {"path": [1, 2, 3]}

        # Test closing
        close_rocksdb(db_instance)
        print("RocksDB instance closed.")
    else:
        print("Failed to open RocksDB instance.")


def get_all_pairs_shortest_path(
    name: str, key: str, G: Any, num_processes: int = 4
) -> Optional[Dict[Any, Any]]:
    # ... (rest of the function)
    pass
