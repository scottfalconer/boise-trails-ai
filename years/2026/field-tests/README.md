# 2026 Field Test Log

This folder is the public daily log for testing the 2026 Boise Trails AI plan in
the field.

There are two phases:

- `pre-challenge/` - test runs before the 2026 Boise Trails Challenge starts.
- `challenge/` - real challenge-window logs once the event is live.

## Privacy Rule

Public logs keep the useful planning evidence and strip sensitive data:

- no home address or exact private planning origin
- no raw Strava activity JSON
- no raw Strava polyline/GPX unless deliberately sanitized
- no access tokens, dashboard ids, or private leaderboard/dashboard payloads

Raw Strava pulls stay in ignored folders such as:

```text
years/2026/inputs/strava/api-pulls/
```

## Daily Folder Shape

Each daily folder should include:

- `README.md` - human-readable field note and analysis status
- `strava-summary.json` - sanitized activity and matching summary
- optional screenshots or public-safe map exports if they are useful

Use the daily log to answer:

- What outing was planned?
- What actually happened door-to-door?
- Which official segments appear to have been completed?
- Which segments were missed or only partially covered?
- Did the phone field packet, GPX, map, or written run card cause confusion?
- What should change before the next test?

