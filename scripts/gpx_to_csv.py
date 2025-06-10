import argparse
import datetime
import json
import os
import logging
from typing import List, Dict, Any

import pandas as pd
import gpxpy
from shapely.geometry import LineString
from rtree import index

# Tolerance for matching in degrees (~10 meters)
TOLERANCE_DEG = 10 / 111000


def load_segments(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path) as f:
        data = json.load(f)
    if 'segments' in data:
        seg_list = data['segments']
    elif 'features' in data:
        seg_list = [f.get('properties', {}) | {'coordinates': f['geometry']['coordinates']} for f in data['features']]
    else:
        raise ValueError('Unrecognized segment JSON structure')
    segments = []
    for seg in seg_list:
        coords = seg['coordinates']
        line = LineString(coords)
        segments.append({
            'id': seg.get('id') or seg.get('seg_id') or seg.get('SegmentId'),
            'name': seg.get('name') or seg.get('seg_name') or seg.get('SegmentName'),
            'direction': seg.get('direction', seg.get('Direction', 'both')),
            'line': line,
        })
    return segments


def build_index(segments: List[Dict[str, Any]]):
    idx = index.Index()
    for i, seg in enumerate(segments):
        idx.insert(i, seg['line'].bounds)
    return idx


def parse_gpx(path: str) -> Dict[str, Any]:
    with open(path) as f:
        gpx = gpxpy.parse(f)
    points = []
    times = []
    elevations = []
    for track in gpx.tracks:
        for seg in track.segments:
            points.extend([(p.longitude, p.latitude) for p in seg.points])
            times.extend([p.time for p in seg.points])
            elevations.extend([p.elevation for p in seg.points])
    if not points:
        raise ValueError('No track points')
    line = LineString(points)
    start_time = times[0]
    end_time = times[-1]
    distance_m = sum(seg.length_2d() for track in gpx.tracks for seg in track.segments)
    elapsed_sec = (end_time - start_time).total_seconds()
    elev_gain = None
    if any(e is not None for e in elevations):
        elev_gain = 0.0
        prev = elevations[0]
        for e in elevations[1:]:
            if e is not None and prev is not None and e > prev:
                elev_gain += e - prev
            prev = e if e is not None else prev
    return {
        'line': line,
        'start_time': start_time,
        'end_time': end_time,
        'distance_m': distance_m,
        'elapsed_sec': elapsed_sec,
        'elev_gain_m': elev_gain,
    }


def match_segments(gpx_data: Dict[str, Any], segments: List[Dict[str, Any]], idx, verbose=False):
    line = gpx_data['line']
    matches = []
    for i in idx.intersection(line.bounds):
        seg = segments[i]
        seg_line = seg['line']
        if line.buffer(TOLERANCE_DEG).intersection(seg_line).length >= 0.95 * seg_line.length:
            matches.append(seg)
            if verbose:
                logging.info(f"Matched segment {seg['id']}")
    return matches


def build_rows(run_id: str, gpx_data: Dict[str, Any], matches: List[Dict[str, Any]]):
    rows = []
    for seg in matches:
        dist_mi = gpx_data['distance_m'] * 0.000621371
        elapsed = gpx_data['elapsed_sec']
        pace = elapsed / 60.0 / dist_mi if dist_mi else None
        elev_gain_ft = gpx_data['elev_gain_m'] * 3.28084 if gpx_data['elev_gain_m'] is not None else None
        rows.append({
            'run_id': run_id,
            'run_date': gpx_data['start_time'].date().isoformat(),
            'year': gpx_data['start_time'].year,
            'seg_id': seg['id'],
            'seg_name': seg['name'],
            'distance_mi': round(dist_mi, 3),
            'elapsed_sec': int(elapsed),
            'pace_min_per_mi': round(pace, 2) if pace else None,
            'elev_gain_ft': round(elev_gain_ft, 1) if elev_gain_ft is not None else None,
            'direction': seg['direction'],
        })
    return rows


def merge_rows(csv_path: str, new_rows: List[Dict[str, Any]], year: int, rebuild: bool):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['run_id','run_date','year','seg_id','seg_name','distance_mi','elapsed_sec','pace_min_per_mi','elev_gain_ft','direction'])
    new_df = pd.DataFrame(new_rows)
    if rebuild:
        df = df[df.year != year]
    df = pd.concat([df, new_df], ignore_index=True)
    df.drop_duplicates(subset=['run_id', 'seg_id'], keep='last', inplace=True)
    df.to_csv(csv_path, index=False)


def process_year(year: int, rebuild=False, verbose=False, base_dir: str = '.'):
    gpx_dir = os.path.join(base_dir, 'data', 'gpx', str(year))
    if not os.path.isdir(gpx_dir):
        if verbose:
            logging.info(f"GPX directory {gpx_dir} does not exist")
        return
    json_path = os.path.join(base_dir, 'GETChallengeTrailData_v2.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    segments = load_segments(json_path)
    idx = build_index(segments)
    all_rows = []
    for fname in sorted(os.listdir(gpx_dir)):
        if not fname.lower().endswith('.gpx'):
            continue
        path = os.path.join(gpx_dir, fname)
        run_id = os.path.splitext(fname)[0]
        try:
            gpx_data = parse_gpx(path)
        except Exception as e:
            logging.warning(f"Skipping {fname}: {e}")
            continue
        matches = match_segments(gpx_data, segments, idx, verbose=verbose)
        rows = build_rows(run_id, gpx_data, matches)
        all_rows.extend(rows)
        if verbose:
            logging.info(f"Processed {fname}")
    csv_path = os.path.join(base_dir, 'data', 'segment_perf.csv')
    merge_rows(csv_path, all_rows, year, rebuild)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert GPX files to segment performance CSV')
    parser.add_argument('--year', type=int, default=datetime.date.today().year)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(message)s')
    process_year(args.year, rebuild=args.rebuild, verbose=args.verbose)


if __name__ == '__main__':
    main()
