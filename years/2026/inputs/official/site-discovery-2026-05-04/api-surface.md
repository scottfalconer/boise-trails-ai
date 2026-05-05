# Boise Trails Challenge API Surface

Discovered: 2026-05-04
Source: authenticated browser session plus cached Nuxt client assets in `assets/`.

## Read-Only Endpoints Pulled

- `GET /api/trails`
  - Public.
  - Saved to `../api-pull-2026-05-04/trails.json`.
  - Shape: `{ lastUpdatedUTC, masterTrails, trailSegments }`.
  - 2026 totals from this pull: 109 all trails, 274 all segments.
  - Foot challenge filter is `activity_type in ["foot", "both"]`: 101 trails, 251 segments, 164.43 miles.
  - Direction values: 228 bidirectional foot segments, 23 ascent-only foot segments.

- `GET /api/leaderboard`
  - Public current leaderboard and current challenge metadata.
  - Raw file saved locally at `../api-pull-2026-05-04/leaderboard.json`.
  - Raw leaderboard data contains participant names, user ids, and profile image URLs, so this file is ignored by git.
  - Challenge metadata saved separately to `../api-pull-2026-05-04/challenge_metadata.json`.

- `GET /api/history/years`
  - Public.
  - Saved to `../api-pull-2026-05-04/history_years.json`.
  - Returned historical years: 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018.

- `GET /api/history/:year`
  - Public historical leaderboard and challenge metadata.
  - Raw files saved locally under `../api-pull-2026-05-04/history/`.
  - Raw history data contains participant names/user ids/profile image URLs, so these files are ignored by git.

- `GET /api/dashboard/:dashboardId`
  - Public/shareable dashboard endpoint.
  - User-specific raw response saved under ignored `../private/api-pull-2026-05-04/dashboard_raw.json`.
  - Redacted derived summary saved to `../api-pull-2026-05-04/derived/dashboard_summary_redacted.json`.
  - Current account progress in this pull: 0.00%, 0 miles, 0 completed segment ids.

- `GET /api/copyedits/ABOUTPAGE`
  - Public rich text used by the About page.
  - Saved to `../api-pull-2026-05-04/copyedits_about.json`.
  - Contains the 2026 challenge window: June 18, 2026 at 12:00:01 a.m. through July 18, 2026 at 11:59:59 p.m.
  - Contains useful rules: on-foot account only counts on-foot activities; segments must be completed in a single activity; ascent-marked trails must be climbed.
  - Note: the copy still mentions Strava timestamps in one paragraph while another paragraph says Strava is no longer used. Treat that as legacy copy until verified.

- `GET /api/copyedits/SCHOLARSHIPPAGE`
  - Public rich text used by the Scholarship page.
  - Saved to `../api-pull-2026-05-04/copyedits_scholarship.json`.

- `GET /api/sponsors`
  - Public sponsor list.
  - Saved to `../api-pull-2026-05-04/sponsors.json`.

## Authenticated / Mutating Endpoints Observed But Not Pulled

- `GET /api/athlete/:uid`
  - Requires Firebase `Authorization: Bearer <id token>` in client code.
  - Used by the account/registration store to load the signed-in athlete record.

- `PUT /api/athlete/:uid`
  - Requires Firebase `Authorization: Bearer <id token>`.
  - Used to update registration/profile fields and profile image URL.
  - Not called.

- `POST /api/delete-user`
  - Requires Firebase `Authorization: Bearer <id token>`.
  - Destructive account deletion path.
  - Not called.

- `/api/payment`
  - Payment/registration flow endpoint observed in the account bundle.
  - Not called.

## Upload / Review Surface

The authenticated navigation exposes `/uploadactivity` and `/requestreview`, but the client asset search did not reveal a separate read-only activity export API. The upload/profile flows use Firebase auth and storage plus the app's authenticated mutation endpoints. I did not upload files, submit review requests, create payments, or call any mutating endpoint.

## Routes Observed In Nuxt Assets

`/`, `/about`, `/account`, `/accountdeleted`, `/auth`, `/dashboard`, `/dashboard/:id()`, `/documentary`, `/forgotpassword`, `/history`, `/leaderboard`, `/login`, `/paymentsuccess`, `/privacypolicy`, `/requestreview`, `/scholarship`, `/signup`, `/trails`, `/uploadactivity`.

## Planning Implications

The official 2026 foot challenge dataset is now available locally as a machine-readable GeoJSON:

- `../api-pull-2026-05-04/official_foot_segments.geojson`
- `../api-pull-2026-05-04/official_foot_master_trails.json`
- `../api-pull-2026-05-04/official_foot_summary.json`

This should replace the 2025 `GETChallengeTrailData_v2.json` as the authoritative 2026 required segment set for planning experiments.
