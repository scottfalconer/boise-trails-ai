# Official 2026 Challenge API Pull - 2026-06-13

Fetched at: 2026-06-13T15:52:15Z
Trail data last changed: 2026-06-11T01:45:43

Public read-only endpoints used:

- `https://boisetrailschallenge.com/api/trails`
- `https://boisetrailschallenge.com/api/leaderboard` for challenge metadata only

Files in this pull:

- `trails.json` - raw public trail payload.
- `challenge_metadata.json` - `ChallengeData[0]` only, without raw participant leaderboard rows.
- `official_foot_segments.geojson` - foot/both segment FeatureCollection.
- `official_foot_master_trails.json` - foot/both master trail list.
- `official_foot_summary.json` - derived counts and distance summary.
- `official_foot_drift_report.json` / `.md` - comparison to the prior official pull.

Current on-foot challenge metrics from this pull:

- Official on-foot trails: 101
- Official on-foot segments: 250
- Official on-foot distance: 159.0 miles
- Direction rules: ascent: 23, both: 227
- Challenge metadata foot segments: 250
- Challenge metadata trail data change: 2026-06-11T01:45:43

Raw leaderboard data is intentionally not saved here because it includes participant identifiers.
