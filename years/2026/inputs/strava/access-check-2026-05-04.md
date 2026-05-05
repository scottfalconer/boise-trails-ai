# Strava API Access Check

Date: 2026-05-04

Credential sources checked:

- `credentials/strava.txt`
- `credentials/strava_activity_read_all.json`

## Result

The original Strava API credentials were usable for basic authenticated access. After reauthorizing the app with `activity:read_all`, direct activity access is now verified.

- OAuth refresh token exchange: succeeded
- Token type returned: Bearer
- Original returned scope: `read`
- Reauthorized returned scope: `activity:read_all read`
- Athlete profile endpoint: succeeded
- Athlete routes endpoint: succeeded and returned at least one route
- Athlete activities endpoint: succeeded after reauthorization

## Endpoint Results

| Endpoint | Result | Notes |
|---|---|---|
| `POST /oauth/token` with original refresh token | `200 OK` | Original refresh flow worked, but only returned `read` scope. |
| `POST /oauth/token` with reauthorization code | `200 OK` | Reauthorized token returned `activity:read_all read`. |
| `GET /api/v3/athlete` | `200 OK` | Confirms access to authenticated athlete profile data. |
| `GET /api/v3/athletes/{athlete_id}/routes` | `200 OK` | Returned one route summary. |
| `GET /api/v3/athlete/activities?per_page=3` | `200 OK` | Returned recent activity summaries. |

## Current Status

`activity:read_all` access is now available locally. This should be enough to pull the user's activity history directly from Strava, including private activities exposed to this app authorization.

The refreshed credential was saved under `credentials/strava_activity_read_all.json` with file mode `0600`. The repository `.gitignore` now excludes `credentials/`, so these local credentials should not be staged.

## Remaining Caveat

This verifies activity read access, not every possible Strava data surface. Segment effort/progress availability still needs a targeted API pull and comparison against the official Boise Trails Challenge segment geometry.

No access token, refresh token, client secret, athlete ID, or authorization code was written to this file.
