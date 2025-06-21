import argparse
import hashlib
import os
import pickle
import shutil
import logging
from typing import Any
import rocksdict


class MemoryRocksDB(dict):
    """Simple in-memory stand-in for ``rocksdict.Rdict`` used in tests."""

    def close(self) -> None:
        pass


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


def _rocksdb_path(name: str, key: str) -> str:
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(get_cache_dir(), f"{name}_{h}_db")


def open_rocksdb(name: str, key: str, read_only: bool = True) -> rocksdict.Rdict | None:
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
    except (Exception, OSError) as e:
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


def close_rocksdb(db: rocksdict.Rdict | None) -> None:
    if db is not None:
        try:
            db.close()
        except Exception as e:
            logger.error("Failed to close RocksDB: %s", e)


def load_rocksdb_cache(db_instance: rocksdict.Rdict | None, source_node: Any) -> Any | None:
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
    except Exception as e:  # pragma: no cover - DB errors
        logger.error("RocksDB read error for %s: %s", source_node, e)
        return None

    if not value_bytes:
        return None

    try:
        return pickle.loads(value_bytes)
    except pickle.UnpicklingError as e:  # pragma: no cover - corrupted entry
        logger.error("Corrupted RocksDB entry for %s: %s", source_node, e)
        return None



def save_rocksdb_cache(db_instance: rocksdict.Rdict | None, source_node: Any, data: Any) -> None:
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
    except Exception as e:  # pragma: no cover - DB errors
        logger.error("RocksDB write error for %s: %s", source_node, e)


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
