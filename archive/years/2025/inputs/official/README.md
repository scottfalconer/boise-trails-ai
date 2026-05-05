# 2025 Official Inputs

This folder separates two different 2025 evidence sources:

- `local-legacy-2025/` - local files that the 2025 planner appears to have used.
- `site-history-2026-05-04/` - public Boise Trails Challenge history API summary for the final 2025 leaderboard.
- `private/` - ignored raw user-specific dashboard snapshot.

Use `site-history-2026-05-04/summary.json` for final 2025 completion metrics. Use `local-legacy-2025/GETChallengeTrailData_v2.json` to reconstruct what last year's planner was optimizing against.

Known discrepancy:

- Local planner file: 247 segments, 100 trails, 169.354 miles.
- Public 2025 history target: 245 segments, 98 trails, 164.73 miles.

The public history target matches the user-reported 2025 result of 41.82% and 68.90 miles.

Likely explanation: the user provided a June 25, 2025 organizer email excerpt saying the Forest Service closed Ridge Road, limiting access to Mahalo. That kind of mid-challenge closure can cause the final public accounting to diverge from the static local file the planner used. See `../../notes/trail-change-events.md` and `../../../challenge-change-events-2026-05-04.md`.
