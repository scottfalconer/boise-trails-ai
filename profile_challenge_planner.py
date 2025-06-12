import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from trail_route_ai import challenge_planner, planner_utils

def create_dem(path: Path) -> None:
    data = np.tile(np.arange(4, dtype=np.float32) * 10, (2, 1))
    transform = from_origin(0, 1, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

def build_edges(n: int = 30):
    edges = []
    for i in range(n):
        start = (float(i), 0.0)
        end = (float(i) + 1.0, 0.0)
        seg_id = f"S{i}"
        edges.append(
            planner_utils.Edge(seg_id, seg_id, start, end, 1.0, 0.0, [start, end], "trail", "both")
        )
    return edges

def main() -> None:
    tmp = Path("prof_tmp")
    tmp.mkdir(exist_ok=True)
    seg_path = tmp / "segments.json"
    perf_path = tmp / "perf.csv"
    dem_path = tmp / "dem.tif"
    out_csv = tmp / "out.csv"
    gpx_dir = tmp / "gpx"
    edges = build_edges()
    perf_path.write_text("seg_id,year\n")
    data = {
        "segments": [
            {"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280}
            for e in edges
        ]
    }
    with open(seg_path, "w") as f:
        json.dump(data, f)
    create_dem(dem_path)
    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-03",
            "--time",
            "30",
            "--pace",
            "10",
            "--segments",
            str(seg_path),
            "--dem",
            str(dem_path),
            "--perf",
            str(perf_path),
            "--year",
            "2024",
            "--output",
            str(out_csv),
            "--gpx-dir",
            str(gpx_dir),
        ]
    )

if __name__ == "__main__":
    main()
