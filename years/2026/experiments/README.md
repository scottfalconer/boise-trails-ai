# 2026 Experiments

Use one subfolder per planner run:

`YYYYMMDD-short-name/`

Each experiment should include:

- `README.md` with purpose and hypothesis
- input dataset paths and hashes
- code branch or commit
- command used
- config file used
- generated output paths
- metrics table
- validation status
- notes on whether the run should be compared against 2025

Suggested first experiments:

1. `baseline-2025-method-on-2026-data`: rerun the 2025 method as directly as possible against 2026 data.
2. `segment-reconciliation`: improve GPX/Strava-to-official-segment matching before route optimization.
3. `clustered-loop-vrp`: build practical route clusters with loop constraints and per-day capacity.
4. `human-adjusted-final-plan`: hand-tuned plan after reviewing solver output.

