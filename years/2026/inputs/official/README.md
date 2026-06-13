# Official 2026 Challenge Inputs

Current official pull:

- API data: `api-pull-2026-06-13/`
- API surface notes: `site-discovery-2026-05-04/api-surface.md`
- Required foot segments: `api-pull-2026-06-13/official_foot_segments.geojson`
- Required foot trail list: `api-pull-2026-06-13/official_foot_master_trails.json`
- Summary: `api-pull-2026-06-13/official_foot_summary.json`
- Drift report: `api-pull-2026-06-13/official_foot_drift_report.md`

Use the foot-filtered files as the authoritative 2026 required challenge set. The raw `/api/trails` payload also includes bike-only segments and should not be used directly for the on-foot plan without filtering.

Private user-specific dashboard raw data is under `private/` and ignored by git.

The prior `api-pull-2026-05-04/` snapshot remains preserved for comparison and
for reconstructing planner work that used the pre-June route list.
