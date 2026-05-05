# Active Boise Trails Years

This folder is the active year-by-year planning workspace. At the moment, only `2026/` should be treated as current work.

Each active year should be self-contained enough to support future retrospectives and model/planner comparisons:

- `README.md` - year overview and current status
- `manifest.json` or `artifact-manifest.json` - paths, roles, and baseline metrics
- `inputs/` - official, personal, Strava, and open-data inputs for that year
- `derived/` - normalized or generated intermediate data
- `experiments/` - solver/model/planner runs
- `outputs/` - generated route plans and reports
- `projects/` - bounded year-specific subprojects
- `notes/` - planning notes and decisions
- `checkpoints/` - readiness and validation records

The top-level `projects/` folder is reserved for new research bundles and portable evidence packs. Year folders can link to those bundles when relevant.

## Archive Boundary

Historical years, previous generated outputs, old source code, and retrospective bundles live under `archive/`:

- `archive/years/` - historical public API pulls and year baselines for 2018-2025.
- `archive/legacy-root-2025/` - old root implementation, tests, configs, data, docs, scripts, and generated artifacts.

Do not add new 2026 experiments to `archive/`. Archive only after a year or workstream is intentionally closed.
