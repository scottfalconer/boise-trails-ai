import os
import json
import datetime
import pandas as pd
from trail_route_ai import gpx_to_csv


def create_sample_data(base):
    os.makedirs(os.path.join(base, 'data', 'gpx', '2024'), exist_ok=True)
    os.makedirs(os.path.join(base, 'data', 'traildata'), exist_ok=True)
    # simple segment line from (0,0) to (0,0.001)
    seg_json = {
        "segments": [
            {
                "id": "seg1",
                "name": "Segment 1",
                "direction": "both",
                "coordinates": [[0.0, 0.0], [0.0, 0.001]]
            }
        ]
    }
    with open(os.path.join(base, 'data', 'traildata', 'trail.json'), 'w') as f:
        json.dump(seg_json, f)
    # simple gpx
    start = datetime.datetime(2024, 6, 1, 10, 0, 0)
    with open(os.path.join(base, 'data', 'gpx', '2024', 'run1.gpx'), 'w') as f:
        f.write(f"""<gpx version='1.1' creator='test'>
  <trk><name>run1</name><trkseg>
    <trkpt lat='0.0' lon='0.0'><ele>0</ele><time>{start.isoformat()}Z</time></trkpt>
    <trkpt lat='0.0005' lon='0.0'><ele>10</ele><time>{(start+datetime.timedelta(minutes=2)).isoformat()}Z</time></trkpt>
    <trkpt lat='0.001' lon='0.0'><ele>20</ele><time>{(start+datetime.timedelta(minutes=4)).isoformat()}Z</time></trkpt>
  </trkseg></trk></gpx>""")


def test_cli_args(tmp_path, monkeypatch):
    create_sample_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    gpx_to_csv.main(['--year', '2024'])
    df = pd.read_csv(tmp_path / 'data' / 'segment_perf.csv')
    assert len(df) == 1
    row = df.iloc[0]
    assert row['seg_id'] == 'seg1'
    assert row['run_id'] == 'run1'


def test_duplicate_suppression(tmp_path, monkeypatch):
    create_sample_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    gpx_to_csv.process_year(2024, base_dir='.')
    gpx_to_csv.process_year(2024, base_dir='.')
    df = pd.read_csv('data/segment_perf.csv')
    assert len(df) == 1


def test_rebuild(tmp_path, monkeypatch):
    create_sample_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    gpx_to_csv.process_year(2024, base_dir='.')
    df1 = pd.read_csv('data/segment_perf.csv')
    assert len(df1) == 1
    # run with rebuild
    gpx_to_csv.process_year(2024, rebuild=True, base_dir='.')
    df2 = pd.read_csv('data/segment_perf.csv')
    assert len(df2) == 1

