import shelve
from typing import Dict, Tuple, List, Optional

from tqdm.auto import tqdm

class PathCache:
    """Persistent cache for shortest paths keyed by start and end nodes.

    Parameters
    ----------
    filename:
        The shelve database file to store path data.
    log:
        If ``True`` (default), log cache hits/misses with ``tqdm.write``.
    """

    def __init__(self, filename: str, *, log: bool = True):
        self.db = shelve.open(filename)
        self.log = log

    def get_paths_from(self, start: Tuple[float, float]) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
        key = repr(start)
        paths = self.db.get(key)
        if paths is not None:
            if self.log:
                tqdm.write(f"PathCache hit: paths from {start}")
            return paths
        if self.log:
            tqdm.write(f"PathCache miss: paths from {start}")
        return {}

    def store_paths_from(self, start: Tuple[float, float], paths: Dict[Tuple[float, float], List[Tuple[float, float]]]) -> None:
        self.db[repr(start)] = paths
        if self.log:
            tqdm.write(f"PathCache store: paths from {start}")

    def get(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        key = repr(start)
        paths = self.db.get(key)
        if paths and end in paths:
            if self.log:
                tqdm.write(f"PathCache hit: {start} -> {end}")
            return paths[end]
        if self.log:
            tqdm.write(f"PathCache miss: {start} -> {end}")
        return None

    def set(self, start: Tuple[float, float], end: Tuple[float, float], path: List[Tuple[float, float]]) -> None:
        key = repr(start)
        paths = self.db.get(key, {})
        paths[end] = path
        self.db[key] = paths
        if self.log:
            tqdm.write(f"PathCache store: {start} -> {end}")

    def close(self) -> None:
        self.db.close()

    def clear(self) -> None:
        """Remove all cached paths."""
        self.db.clear()
        if self.log:
            tqdm.write("PathCache cleared")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
