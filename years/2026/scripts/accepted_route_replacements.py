#!/usr/bin/env python3
"""Accepted route replacement manifest helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent

DEFAULT_ACCEPTED_REPLACEMENTS_JSON = YEAR_DIR / "inputs" / "accepted-route-replacements-v1.json"

ACTIVE_STATUS = "active"
INVESTIGATE_STATUS = "investigate"
WAIVED_STATUS = "waived"
BLOCKING_STATUSES = {ACTIVE_STATUS, INVESTIGATE_STATUS}
VALID_STATUSES = {ACTIVE_STATUS, INVESTIGATE_STATUS, WAIVED_STATUS}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def normalized_segment_ids(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    result = []
    for value in values or []:
        try:
            result.append(str(int(value)))
        except (TypeError, ValueError):
            continue
    return sorted(set(result), key=lambda item: (len(item), item))


def segment_key(values: list[Any] | tuple[Any, ...] | None) -> tuple[str, ...]:
    return tuple(normalized_segment_ids(values))


def loop_segment_key(loop: dict[str, Any], candidate: dict[str, Any] | None = None) -> tuple[str, ...]:
    values = (
        loop.get("segment_ids")
        or [segment.get("seg_id") for segment in (candidate or {}).get("segments") or []]
        or (candidate or {}).get("segment_ids")
    )
    return segment_key(values)


def anchor_ref_aliases(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    aliases = {text}
    if text.startswith("facility-"):
        aliases.add(text.removeprefix("facility-"))
    else:
        aliases.add(f"facility-{text}")
    aliases.add(normalize_key(text))
    return {alias for alias in aliases if alias}


def anchor_refs_match(left: Any, right: Any) -> bool:
    return bool(anchor_ref_aliases(left) & anchor_ref_aliases(right))


def float_value(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def int_value(value: Any, fallback: int = 0) -> int:
    try:
        if value is None:
            return fallback
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def route_metrics(route: dict[str, Any] | None) -> dict[str, Any]:
    route = route or {}
    return {
        "on_foot_miles": float_value(route.get("on_foot_miles")),
        "p75_minutes": int_value(route.get("door_to_door_minutes_p75") or route.get("p75_minutes")),
        "p90_minutes": int_value(route.get("door_to_door_minutes_p90") or route.get("p90_minutes")),
        "drive_minutes": int_value(route.get("drive_minutes")),
        "parking_confidence": ((route.get("parking") or {}).get("parking_confidence") or route.get("parking_confidence")),
        "cue_complexity": len(route.get("wayfinding_cues") or route.get("route_cues") or []),
        "anchor_to_credit_endpoint_distance_miles": route.get("anchor_to_credit_endpoint_distance_miles"),
    }


def candidate_metrics(candidate: dict[str, Any] | None) -> dict[str, Any]:
    candidate = candidate or {}
    estimates = candidate.get("time_estimates_minutes") or {}
    breakdown = candidate.get("time_breakdown_minutes") or {}
    trailhead = candidate.get("trailhead") or {}
    return {
        "on_foot_miles": float_value(
            candidate.get("estimated_total_on_foot_miles") or candidate.get("on_foot_miles")
        ),
        "p75_minutes": int_value(
            estimates.get("door_to_door_p75") or candidate.get("total_minutes")
        ),
        "p90_minutes": int_value(estimates.get("door_to_door_p90")),
        "drive_minutes": int_value(breakdown.get("drive_to_trailhead")) + int_value(breakdown.get("return_drive")),
        "parking_confidence": trailhead.get("parking_confidence") or candidate.get("parking_confidence"),
        "cue_complexity": len(candidate.get("wayfinding_cues") or candidate.get("route_cues") or []),
        "anchor_to_credit_endpoint_distance_miles": candidate.get("anchor_to_credit_endpoint_distance_miles"),
    }


def dominance_deltas(baseline: dict[str, Any], replacement: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for key in ("on_foot_miles", "p75_minutes", "p90_minutes", "drive_minutes", "cue_complexity"):
        if baseline.get(key) not in (None, 0, 0.0) or replacement.get(key) not in (None, 0, 0.0):
            deltas[f"dominance_delta_{key}"] = round(float_value(baseline.get(key)) - float_value(replacement.get(key)), 3)
    if baseline.get("anchor_to_credit_endpoint_distance_miles") is not None and replacement.get("anchor_to_credit_endpoint_distance_miles") is not None:
        deltas["dominance_delta_anchor_to_credit_endpoint_distance_miles"] = round(
            float_value(baseline.get("anchor_to_credit_endpoint_distance_miles"))
            - float_value(replacement.get("anchor_to_credit_endpoint_distance_miles")),
            3,
        )
    if baseline.get("parking_confidence") or replacement.get("parking_confidence"):
        deltas["parking_confidence_delta"] = {
            "baseline": baseline.get("parking_confidence"),
            "replacement": replacement.get("parking_confidence"),
        }
    return deltas


def no_material_regression(record: dict[str, Any], deltas: dict[str, Any]) -> bool:
    checks = {
        "dominance_delta_on_foot_miles": float_value(record.get("max_allowed_on_foot_regression"), 0.0),
        "dominance_delta_p75_minutes": float_value(record.get("max_allowed_p75_regression"), 0.0),
        "dominance_delta_p90_minutes": float_value(record.get("max_allowed_p90_regression"), 0.0),
    }
    for key, allowed in checks.items():
        value = deltas.get(key)
        if value is not None and float_value(value) < -allowed:
            return False
    return True


def meets_material_savings(record: dict[str, Any], deltas: dict[str, Any]) -> bool:
    return (
        float_value(deltas.get("dominance_delta_on_foot_miles")) >= float_value(record.get("min_on_foot_savings"))
        or float_value(deltas.get("dominance_delta_p75_minutes")) >= float_value(record.get("min_p75_savings"))
        or float_value(deltas.get("dominance_delta_p90_minutes")) >= float_value(record.get("min_p90_savings"))
    )


class AcceptedRouteReplacementIndex:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = list(records or [])
        self.by_segment: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for record in self.records:
            status = str(record.get("status") or "").strip()
            if status and status not in VALID_STATUSES:
                raise ValueError(f"Unknown accepted replacement status: {status}")
            key = segment_key(record.get("target_segment_ids"))
            if key:
                self.by_segment.setdefault(key, []).append(record)

    @classmethod
    def from_path(cls, path: Path | None = None) -> "AcceptedRouteReplacementIndex":
        path = path or DEFAULT_ACCEPTED_REPLACEMENTS_JSON
        if not path.exists():
            return cls([])
        payload = read_json(path)
        return cls(list(payload.get("replacements") or []))

    def match_for_segments(self, values: list[Any] | tuple[Any, ...] | None) -> dict[str, Any] | None:
        key = segment_key(values)
        matches = self.by_segment.get(key) or []
        if not matches:
            return None
        blocking = [record for record in matches if record.get("status") in BLOCKING_STATUSES]
        return blocking[0] if blocking else matches[0]

    def match_for_loop(self, loop: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any] | None:
        matches = self.by_segment.get(loop_segment_key(loop, candidate)) or []
        if not matches:
            return None
        loop_id = str(loop.get("loop_id") or "")
        candidate_id = str(loop.get("candidate_id") or "")
        exact = [
            record
            for record in matches
            if loop_id in {str(value) for value in record.get("source_loop_ids") or []}
            or candidate_id in {str(value) for value in record.get("source_candidate_ids") or []}
        ]
        candidates = exact or matches
        blocking = [record for record in candidates if record.get("status") in BLOCKING_STATUSES]
        return blocking[0] if blocking else candidates[0]

    def blocking_match_for_loop(self, loop: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any] | None:
        record = self.match_for_loop(loop, candidate)
        if record and record.get("status") in BLOCKING_STATUSES and record.get("hard_block_current_preservation", True):
            return record
        return None

    def has_manifest_explanation_for_candidate(self, segment_ids: list[Any], anchor_ref: Any) -> bool:
        for record in self.by_segment.get(segment_key(segment_ids)) or []:
            if anchor_refs_match(record.get("accepted_anchor_ref"), anchor_ref):
                return True
        return False
