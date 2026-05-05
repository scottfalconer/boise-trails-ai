# Boise Trails Challenge API Pull

Pulled: 2026-05-04
Source: `https://boisetrailschallenge.com/`

## Official 2026 Foot Dataset

- `trails.json` - raw public `/api/trails` payload.
- `official_foot_segments.geojson` - filtered required on-foot segment feature collection.
- `official_foot_master_trails.json` - filtered required on-foot master trails.
- `official_foot_summary.json` - counts and distance summary.
- `challenge_metadata.json` - current challenge metadata from `/api/leaderboard`.

Key metrics:

- Foot trails: 101
- Foot segments: 251
- Foot official distance: 164.43 miles
- Directional ascent-only foot segments: 23
- Trail data last changed: 2026-05-01 19:14:44
- Challenge window from site copy: June 18, 2026 12:00:01 a.m. through July 18, 2026 11:59:59 p.m.

## Public Context Pulls

- `copyedits_about.json` - About/rules copy.
- `copyedits_scholarship.json` - Scholarship copy.
- `sponsors.json` - Sponsor list.
- `history_years.json` - Available historical leaderboard years.
- `summary.json` - Pull-level public endpoint summary.

Raw current leaderboard and raw historical leaderboard files exist locally but are ignored by git because they include participant names, user ids, and profile image URLs:

- `leaderboard.json`
- `history/*.json`

Derived public/performance summaries:

- `derived/scott_falconer_public_history_summary.json`
- `derived/dashboard_summary_redacted.json`
- `derived/leaderboard_schema_by_year.json`

Private user-specific raw dashboard data is saved under `../private/api-pull-2026-05-04/` and ignored by git.
