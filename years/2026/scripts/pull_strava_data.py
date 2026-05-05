#!/usr/bin/env python3
"""Pull the scoped Strava API data needed for 2026 planning."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CREDENTIALS = REPO_ROOT / "credentials" / "strava_activity_read_all.json"
DEFAULT_OUT = REPO_ROOT / "years" / "2026" / "inputs" / "strava" / "api-pulls"
API_BASE = "https://www.strava.com/api/v3"
OAUTH_URL = "https://www.strava.com/oauth/token"
ON_FOOT_TYPES = {"Run", "TrailRun", "Hike", "Walk"}


def iso_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def parse_activity_date(activity: dict[str, Any]) -> dt.date | None:
    value = activity.get("start_date_local") or activity.get("start_date")
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def epoch_start(date_value: dt.date) -> int:
    return int(dt.datetime.combine(date_value, dt.time.min, tzinfo=dt.UTC).timestamp())


def epoch_after_end(date_value: dt.date) -> int:
    return int(dt.datetime.combine(date_value + dt.timedelta(days=1), dt.time.min, tzinfo=dt.UTC).timestamp())


def load_credentials(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["client_id", "client_secret", "refresh_token"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise RuntimeError(f"Missing credential fields: {', '.join(missing)}")
    return data


def write_credentials(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)


def refresh_if_needed(credentials_path: Path, credentials: dict[str, Any]) -> dict[str, Any]:
    expires_at = int(credentials.get("expires_at") or 0)
    if credentials.get("access_token") and expires_at > int(time.time()) + 600:
        return credentials

    response = requests.post(
        OAUTH_URL,
        data={
            "client_id": credentials["client_id"],
            "client_secret": credentials["client_secret"],
            "refresh_token": credentials["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    credentials.update(
        {
            "access_token": payload.get("access_token"),
            "refresh_token": payload.get("refresh_token", credentials.get("refresh_token")),
            "token_type": payload.get("token_type"),
            "scope": payload.get("scope"),
            "expires_at": payload.get("expires_at"),
            "expires_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(payload["expires_at"]))),
            "refreshed_at_utc": iso_now(),
        }
    )
    write_credentials(credentials_path, credentials)
    return credentials


class StravaClient:
    def __init__(self, access_token: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {access_token}"})
        self.rate_snapshots: list[dict[str, Any]] = []

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{API_BASE}{path}"
        response = self.session.get(url, params=params or {}, timeout=30)
        self.rate_snapshots.append(
            {
                "path": path,
                "status_code": response.status_code,
                "limit": response.headers.get("X-RateLimit-Limit"),
                "usage": response.headers.get("X-RateLimit-Usage"),
            }
        )
        if response.status_code == 429:
            raise RuntimeError(f"Strava rate limit hit while requesting {path}")
        response.raise_for_status()
        return response.json()


def pull_activity_summaries(client: StravaClient, after: dt.date, before: dt.date) -> list[dict[str, Any]]:
    activities: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = client.get(
            "/athlete/activities",
            {
                "after": epoch_start(after),
                "before": epoch_after_end(before),
                "per_page": 200,
                "page": page,
            },
        )
        if not isinstance(batch, list):
            raise RuntimeError("Unexpected activities response shape")
        activities.extend(batch)
        if len(batch) < 200:
            break
        page += 1
    return activities


def selected_detail_activities(
    activities: list[dict[str, Any]],
    challenge_windows: dict[str, tuple[dt.date, dt.date]],
    recent_start: dt.date,
    today: dt.date,
) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for activity in activities:
        activity_id = activity.get("id")
        if not activity_id:
            continue
        sport_type = activity.get("sport_type") or activity.get("type")
        if sport_type not in ON_FOOT_TYPES:
            continue
        activity_date = parse_activity_date(activity)
        if not activity_date:
            continue

        reasons = []
        for label, (start, end) in challenge_windows.items():
            if start <= activity_date <= end:
                reasons.append(label)
        if recent_start <= activity_date <= today:
            reasons.append("2026_recent")

        if reasons:
            selected[int(activity_id)] = {
                "id": int(activity_id),
                "name": activity.get("name"),
                "sport_type": sport_type,
                "start_date": activity.get("start_date"),
                "start_date_local": activity.get("start_date_local"),
                "distance": activity.get("distance"),
                "total_elevation_gain": activity.get("total_elevation_gain"),
                "selection_reasons": reasons,
            }
    return sorted(selected.values(), key=lambda item: item["start_date"] or "")


def summarize_activity(activity: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": activity.get("id"),
        "name": activity.get("name"),
        "sport_type": activity.get("sport_type") or activity.get("type"),
        "start_date": activity.get("start_date"),
        "start_date_local": activity.get("start_date_local"),
        "distance_m": activity.get("distance"),
        "moving_time_s": activity.get("moving_time"),
        "elapsed_time_s": activity.get("elapsed_time"),
        "total_elevation_gain_m": activity.get("total_elevation_gain"),
        "segment_effort_count": len(activity.get("segment_efforts") or []),
        "best_effort_count": len(activity.get("best_efforts") or []),
        "lap_count": len(activity.get("laps") or []),
        "has_map_polyline": bool((activity.get("map") or {}).get("polyline")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--after", default="2024-06-01")
    parser.add_argument("--before", default=None)
    parser.add_argument("--recent-days", type=int, default=35)
    args = parser.parse_args()

    today = dt.datetime.now().date()
    after = parse_date(args.after)
    before = parse_date(args.before) if args.before else today
    out_dir = args.out or DEFAULT_OUT / today.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "activity_details"
    details_dir.mkdir(parents=True, exist_ok=True)

    credentials = refresh_if_needed(args.credentials, load_credentials(args.credentials))
    scope = credentials.get("scope", "")
    if "activity:read_all" not in scope:
        raise RuntimeError(f"Credential scope does not include activity:read_all: {scope!r}")

    client = StravaClient(credentials["access_token"])
    pull_started = iso_now()

    athlete = client.get("/athlete")
    routes = client.get(f"/athletes/{athlete['id']}/routes", {"per_page": 100})
    activities = pull_activity_summaries(client, after=after, before=before)

    challenge_windows = {
        "2024_proxy_window": (dt.date(2024, 6, 19), dt.date(2024, 7, 19)),
        "2025_challenge_window": (dt.date(2025, 6, 19), dt.date(2025, 7, 19)),
    }
    recent_start = today - dt.timedelta(days=args.recent_days - 1)
    selected = selected_detail_activities(
        activities,
        challenge_windows=challenge_windows,
        recent_start=recent_start,
        today=today,
    )

    detailed_summaries = []
    for item in selected:
        detail = client.get(f"/activities/{item['id']}", {"include_all_efforts": "true"})
        detail["codex_selection_reasons"] = item["selection_reasons"]
        write_json(details_dir / f"{item['id']}.json", detail)
        summary = summarize_activity(detail)
        summary["selection_reasons"] = ",".join(item["selection_reasons"])
        detailed_summaries.append(summary)

    athlete_redacted = {
        "resource_state": athlete.get("resource_state"),
        "premium": athlete.get("premium"),
        "summit": athlete.get("summit"),
        "created_at": athlete.get("created_at"),
        "updated_at": athlete.get("updated_at"),
        "sex_present": bool(athlete.get("sex")),
        "country_present": bool(athlete.get("country")),
        "profile_present": bool(athlete.get("profile")),
    }

    activity_summary_rows = []
    for activity in activities:
        date_value = parse_activity_date(activity)
        activity_summary_rows.append(
            {
                "id": activity.get("id"),
                "name": activity.get("name"),
                "sport_type": activity.get("sport_type") or activity.get("type"),
                "start_date": activity.get("start_date"),
                "start_date_local": activity.get("start_date_local"),
                "activity_date_local": date_value.isoformat() if date_value else "",
                "distance_m": activity.get("distance"),
                "moving_time_s": activity.get("moving_time"),
                "elapsed_time_s": activity.get("elapsed_time"),
                "total_elevation_gain_m": activity.get("total_elevation_gain"),
                "average_speed": activity.get("average_speed"),
                "max_speed": activity.get("max_speed"),
                "has_heartrate": activity.get("has_heartrate"),
                "average_heartrate": activity.get("average_heartrate"),
                "max_heartrate": activity.get("max_heartrate"),
                "suffer_score": activity.get("suffer_score"),
                "workout_type": activity.get("workout_type"),
            }
        )

    on_foot = [row for row in activity_summary_rows if row["sport_type"] in ON_FOOT_TYPES]
    by_year: dict[str, dict[str, Any]] = {}
    for row in on_foot:
        if not row["activity_date_local"]:
            continue
        year = row["activity_date_local"][:4]
        entry = by_year.setdefault(
            year,
            {
                "activity_count": 0,
                "run_day_count": set(),
                "distance_m": 0.0,
                "elevation_gain_m": 0.0,
                "moving_time_s": 0,
            },
        )
        entry["activity_count"] += 1
        entry["run_day_count"].add(row["activity_date_local"])
        entry["distance_m"] += float(row["distance_m"] or 0)
        entry["elevation_gain_m"] += float(row["total_elevation_gain_m"] or 0)
        entry["moving_time_s"] += int(row["moving_time_s"] or 0)

    yearly_summary = []
    for year, entry in sorted(by_year.items()):
        yearly_summary.append(
            {
                "year": year,
                "on_foot_activity_count": entry["activity_count"],
                "on_foot_day_count": len(entry["run_day_count"]),
                "distance_mi": round(entry["distance_m"] / 1609.344, 2),
                "elevation_gain_ft": round(entry["elevation_gain_m"] * 3.280839895, 1),
                "moving_hours": round(entry["moving_time_s"] / 3600, 2),
            }
        )

    pull_summary = {
        "created_at_utc": pull_started,
        "completed_at_utc": iso_now(),
        "source": "Strava API",
        "credential_path": str(args.credentials),
        "credential_scope": scope,
        "date_window": {
            "after": after.isoformat(),
            "before": before.isoformat(),
        },
        "detail_selection": {
            "challenge_windows": {
                label: {"start": start.isoformat(), "end": end.isoformat()}
                for label, (start, end) in challenge_windows.items()
            },
            "recent_window": {
                "label": "2026_recent",
                "start": recent_start.isoformat(),
                "end": today.isoformat(),
            },
            "selected_detail_count": len(selected),
        },
        "counts": {
            "activity_summary_count": len(activities),
            "on_foot_summary_count": len(on_foot),
            "route_count": len(routes) if isinstance(routes, list) else None,
            "activity_detail_count": len(detailed_summaries),
            "activity_details_with_segment_efforts": sum(1 for row in detailed_summaries if row["segment_effort_count"]),
            "total_segment_efforts_in_details": sum(int(row["segment_effort_count"]) for row in detailed_summaries),
        },
        "yearly_on_foot_summary": yearly_summary,
        "rate_limit_snapshots": client.rate_snapshots,
        "notes": [
            "Raw secrets are not stored in this pull directory.",
            "Athlete profile is redacted.",
            "Detailed activity pulls are scoped to prior challenge windows and recent 2026 activity to stay within Strava read limits.",
        ],
    }

    write_json(out_dir / "athlete_redacted.json", athlete_redacted)
    write_json(out_dir / "routes.json", routes)
    write_json(out_dir / "activities_summary.json", activities)
    write_json(out_dir / "selected_activity_details_index.json", selected)
    write_json(out_dir / "activity_detail_summaries.json", detailed_summaries)
    write_json(out_dir / "pull_summary.json", pull_summary)
    write_csv(
        out_dir / "activities_summary.csv",
        activity_summary_rows,
        [
            "id",
            "name",
            "sport_type",
            "start_date",
            "start_date_local",
            "activity_date_local",
            "distance_m",
            "moving_time_s",
            "elapsed_time_s",
            "total_elevation_gain_m",
            "average_speed",
            "max_speed",
            "has_heartrate",
            "average_heartrate",
            "max_heartrate",
            "suffer_score",
            "workout_type",
        ],
    )
    write_csv(
        out_dir / "activity_detail_summaries.csv",
        detailed_summaries,
        [
            "id",
            "name",
            "sport_type",
            "start_date",
            "start_date_local",
            "distance_m",
            "moving_time_s",
            "elapsed_time_s",
            "total_elevation_gain_m",
            "segment_effort_count",
            "best_effort_count",
            "lap_count",
            "has_map_polyline",
            "selection_reasons",
        ],
    )

    readme = f"""# Strava API Pull

Created: {pull_summary['created_at_utc']}

Source: Strava API using local `activity:read_all` credentials.

## Scope

- Activity summaries: {after.isoformat()} through {before.isoformat()}
- Detailed activities: 2024 proxy challenge window, 2025 challenge window, and recent 2026 activity from {recent_start.isoformat()} through {today.isoformat()}
- Routes: authenticated athlete routes endpoint
- Athlete profile: redacted metadata only

## Counts

- Activity summaries: {pull_summary['counts']['activity_summary_count']}
- On-foot summaries: {pull_summary['counts']['on_foot_summary_count']}
- Routes: {pull_summary['counts']['route_count']}
- Detailed activities: {pull_summary['counts']['activity_detail_count']}
- Details with segment efforts: {pull_summary['counts']['activity_details_with_segment_efforts']}
- Segment efforts in detailed records: {pull_summary['counts']['total_segment_efforts_in_details']}

## Files

- `pull_summary.json`
- `activities_summary.json`
- `activities_summary.csv`
- `activity_details/*.json`
- `activity_detail_summaries.json`
- `activity_detail_summaries.csv`
- `selected_activity_details_index.json`
- `routes.json`
- `athlete_redacted.json`

No tokens, refresh tokens, client secrets, authorization codes, or unredacted athlete profile are written here.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(pull_summary, indent=2))


if __name__ == "__main__":
    main()
