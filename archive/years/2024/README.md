# 2024 Boise Trails Reference

Created: 2026-05-04

This folder holds the per-year reference data available locally for the 2024 Boise Trails Challenge.

## Public History Snapshot

Derived from `GET /api/history/2024` and saved at `inputs/official/site-history-2026-05-04/summary.json`.

- Leaderboard records: 890
- Finishers: 315
- Inferred target segments: 236
- Inferred target trails: 98
- Inferred target distance: 177.25 miles
- Scott Falconer public record: 64.87% / 114.97 mi / rank 364.

The public history API does not include historical segment geometry. Raw history rows include public participant identifiers and remain in the ignored 2026 API pull cache.

Known challenge-state caveat: user-provided organizer email excerpts document a Polecat fire closure and Heroes/Bogus construction/access issues during the 2024 challenge. See `notes/trail-change-events.md`.

## Archived Local GPX Artifacts

During the 2026 cleanup pass, top-level GPX artifacts with 2024 dates were moved here instead of being folded into the 2025 archive:

- `inputs/personal/gpx-results/` - 24 personal/result GPX files previously under `data/results/2024/`.
- `outputs/generated-gpx-root/` - 142 generated GPX files previously under top-level `gpx/`.
- `outputs/failed-gpx-root/` - 245 failed/generated GPX files previously under top-level `failed-gpx/`.
