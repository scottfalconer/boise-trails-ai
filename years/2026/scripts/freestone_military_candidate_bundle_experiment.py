#!/usr/bin/env python3
"""Generate Freestone/Military candidate bundles without making one mega-route."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from export_execution_gpx import render_gpx_segments  # noqa: E402
from freestone_cluster_route_generation_experiment import (  # noqa: E402
    build_generated_route,
    current_route_metrics,
    display_path,
    float_value,
    normalized_ids,
    read_json,
    route_key,
    scaled_minutes,
    slugify,
    sort_id,
    write_json,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    load_connector_graph,
    load_official_segments,
    round_miles,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints" / "freestone-military-candidate-bundles-2026-05-12"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "freestone-military-candidate-bundles-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "freestone-military-candidate-bundles-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "freestone-military-candidate-bundles-2026-05-12-manifest.json"

FD19C = ["1649", "1650", "1651"]
FD04A = ["1748", "1652", "1558"]
FD20A_THREE_BEARS = ["1681", "1682", "1683", "1684", "1685"]
FD20A_FREESTONE_RIDGE = ["1563", "1564"]
MOUNTAIN_COVE = ["1591", "1592", "1593", "1594"]
FD06A_CURLEW_FAT_TIRE = ["1555", "1711", "1710"]

CANDIDATE_BUNDLES = [
    {
        "bundle_id": "F1-upper-loop-replace-fd19c-fd04a-shrink-fd20a",
        "shape": "F1",
        "intent": "Freestone upper loop with Shane's / Three Bears / Two Point / Femrite's, plus a separate Freestone Ridge shrink loop.",
        "replace_route_labels": ["FD19C", "FD04A", "FD20A"],
        "preserve_route_labels": ["3"],
        "loops": [
            {
                "loop_id": "f1-upper-shanes-three-bears-two-point-femrite",
                "trailhead": "Freestone Creek",
                "strategy": "nearest_segment_greedy",
                "segment_ids": FD19C + ["1748", "1652", "1558"] + FD20A_THREE_BEARS,
            },
            {
                "loop_id": "f1-fd20a-freestone-ridge-shrink",
                "trailhead": "Freestone Creek",
                "strategy": "template_sequence_greedy",
                "segment_ids": FD20A_FREESTONE_RIDGE,
            },
        ],
    },
    {
        "bundle_id": "F1-upper-loop-with-mountain-cove-warmup",
        "shape": "F1",
        "intent": "Same upper-loop shape, but explicitly tests Mountain Cove as credited warm-up instead of only connector context.",
        "replace_route_labels": ["FD19C", "FD04A", "FD20A"],
        "preserve_route_labels": ["3"],
        "loops": [
            {
                "loop_id": "f1-mountain-cove-shanes-three-bears-two-point-femrite",
                "trailhead": "Freestone Creek",
                "strategy": "nearest_segment_greedy",
                "segment_ids": MOUNTAIN_COVE + FD19C + ["1748", "1652", "1558"] + FD20A_THREE_BEARS,
            },
            {
                "loop_id": "f1-fd20a-freestone-ridge-shrink-after-mountain-cove",
                "trailhead": "Freestone Creek",
                "strategy": "template_sequence_greedy",
                "segment_ids": FD20A_FREESTONE_RIDGE,
            },
        ],
    },
    {
        "bundle_id": "F1-upper-single-loop-all-three-current-cards",
        "shape": "F1",
        "intent": "Single upper Freestone loop replacing FD19C, FD04A, and FD20A without pulling in the Military core.",
        "replace_route_labels": ["FD19C", "FD04A", "FD20A"],
        "preserve_route_labels": ["3"],
        "loops": [
            {
                "loop_id": "f1-upper-single-loop",
                "trailhead": "Freestone Creek",
                "strategy": "nearest_segment_greedy",
                "segment_ids": FD19C + ["1748", "1652", "1558"] + FD20A_THREE_BEARS + FD20A_FREESTONE_RIDGE,
            }
        ],
    },
    {
        "bundle_id": "F2-military-core-remains-separate",
        "shape": "F2",
        "intent": "Preserve route 3 and FD20A; test whether FD19C and FD04A alone should become one compact upper loop.",
        "replace_route_labels": ["FD19C", "FD04A"],
        "preserve_route_labels": ["3", "FD20A"],
        "loops": [
            {
                "loop_id": "f2-shanes-two-point-femrite-only",
                "trailhead": "Freestone Creek",
                "strategy": "nearest_segment_greedy",
                "segment_ids": FD19C + FD04A,
            }
        ],
    },
    {
        "bundle_id": "F3-curlew-fat-tire-freestone-safety",
        "shape": "F3",
        "intent": "Only test Curlew/Fat Tire with Freestone Ridge when ascent direction and timing stay sane.",
        "replace_route_labels": ["FD06A", "FD20A"],
        "preserve_route_labels": ["3", "FD19C", "FD04A"],
        "protected_segment_ids": FD06A_CURLEW_FAT_TIRE,
        "loops": [
            {
                "loop_id": "f3-lower-interpretive-fat-tire-curlew-freestone-ridge",
                "trailhead": "Lower Interpretive",
                "strategy": "template_sequence_greedy",
                "segment_ids": FD06A_CURLEW_FAT_TIRE + FD20A_FREESTONE_RIDGE,
            },
            {
                "loop_id": "f3-three-bears-remains-freestone",
                "trailhead": "Freestone Creek",
                "strategy": "template_sequence_greedy",
                "segment_ids": FD20A_THREE_BEARS,
            },
        ],
    },
]


def parking_for_trailhead(field_tool_data: dict[str, Any], trailhead: str) -> dict[str, Any]:
    for route in field_tool_data.get("routes") or []:
        if route.get("trailhead") != trailhead:
            continue
        parking = route.get("parking") or {}
        if not parking.get("has_parking"):
            continue
        return {
            "name": parking.get("name") or trailhead,
            "lat": float(parking["lat"]),
            "lon": float(parking["lon"]),
            "has_parking": True,
            "parking_confidence": parking.get("parking_confidence"),
            "source": parking.get("source"),
            "nearest_open_trail_name": parking.get("nearest_open_trail_name"),
            "nearest_open_trail_label": parking.get("nearest_open_trail_label"),
        }
    raise ValueError(f"Missing parking for trailhead {trailhead}")


def route_by_label(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(route.get("label")): route for route in field_tool_data.get("routes") or []}


def segment_owner_index(field_tool_data: dict[str, Any]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for route in field_tool_data.get("routes") or []:
        label = str(route.get("label") or route_key(route))
        for segment_id in normalized_ids(route.get("segment_ids") or []):
            owners.setdefault(segment_id, []).append(label)
    return owners


def route_metrics_for_labels(routes_by_label: dict[str, dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    return current_route_metrics([routes_by_label[label] for label in labels])


def render_loop_gpx(output_dir: Path, bundle_id: str, loop: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slugify(bundle_id)}--{slugify(loop['variant_id'])}.gpx"
    path.write_text(
        render_gpx_segments(
            f"Freestone/Military bundle {bundle_id}: {loop['variant_id']}",
            [loop["track_coordinates"]],
        ),
        encoding="utf-8",
    )
    return path


def union_loop_ids(loops: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({segment_id for loop in loops for segment_id in normalized_ids(loop.get(key) or [])}, key=sort_id)


def generated_official_touch_ids(loops: list[dict[str, Any]]) -> list[str]:
    touched = set()
    for loop in loops:
        touched.update(normalized_ids(loop.get("official_segment_ids") or []))
        touched.update(normalized_ids(loop.get("self_repeat_segment_ids") or []))
        touched.update(normalized_ids(loop.get("non_template_repeat_segment_ids") or []))
    return sorted(touched, key=sort_id)


def route_impact_rows(routes_by_label: dict[str, dict[str, Any]], labels: list[str], generated_claim_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        route = routes_by_label[label]
        route_ids = set(normalized_ids(route.get("segment_ids") or []))
        covered_ids = sorted(route_ids & generated_claim_ids, key=sort_id)
        remaining_ids = sorted(route_ids - generated_claim_ids, key=sort_id)
        if not covered_ids:
            status = "unchanged"
        elif not remaining_ids:
            status = "removed_if_candidate_promoted"
        else:
            status = "shrunk_if_candidate_promoted"
        rows.append(
            {
                "label": label,
                "outing_id": route_key(route),
                "status": status,
                "covered_segment_ids": covered_ids,
                "remaining_segment_ids": remaining_ids,
            }
        )
    return rows


def latent_credit_rows(
    *,
    loops: list[dict[str, Any]],
    replaced_ids: set[str],
    preserved_labels: set[str],
    owner_by_segment: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for segment_id in generated_official_touch_ids(loops):
        if segment_id in replaced_ids:
            continue
        owners = owner_by_segment.get(segment_id, [])
        owner_set = set(owners)
        if owner_set & preserved_labels:
            status = "owned_by_preserved_route"
        elif owners:
            status = "owned_by_other_active_route_needs_declared_elsewhere"
        else:
            status = "unowned_latent_credit"
        rows.append({"segment_id": segment_id, "owners": owners, "status": status})
    return rows


def promotion_gates(
    *,
    loops: list[dict[str, Any]],
    current_scope: dict[str, Any],
    candidate_scope: dict[str, Any],
    latent_rows: list[dict[str, Any]],
    protected_segment_ids: list[str],
) -> dict[str, Any]:
    ascent_failed = union_loop_ids(
        [{"ids": (loop.get("ascent_direction_validation") or {}).get("direction_failed_segment_ids") or []} for loop in loops],
        "ids",
    )
    missing_protected = sorted(set(protected_segment_ids) - set(union_loop_ids(loops, "official_segment_ids")), key=sort_id)
    hidden_self_repeat = union_loop_ids(loops, "self_repeat_segment_ids")
    direct_gap_miles = round_miles(sum(float_value(loop.get("direct_gap_fallback_miles")) for loop in loops))
    p75_p90_better = (
        candidate_scope["on_foot_miles"] < current_scope["on_foot_miles"]
        and candidate_scope["p75_minutes"] < current_scope["p75_minutes"]
        and candidate_scope["p90_minutes"] < current_scope["p90_minutes"]
    )
    hard_failures = []
    if ascent_failed or missing_protected:
        hard_failures.append("ascent_or_protected_segment_failure")
    if hidden_self_repeat:
        hard_failures.append("hidden_self_repeat")
    if [row for row in latent_rows if row["status"] != "owned_by_preserved_route"]:
        hard_failures.append("latent_credit_needs_ownership_decision")
    if not p75_p90_better:
        hard_failures.append("not_better_on_on_foot_p75_p90")
    if direct_gap_miles:
        hard_failures.append("direct_gap_fallback")
    hard_failures.append("needs_human_cue_sheet")
    hard_failures.append("needs_field_packet_recertification")
    return {
        "status": "blocked_by_promotion_gates" if hard_failures else "promotion_candidate",
        "hard_failures": hard_failures,
        "ascent_direction": "passed" if not ascent_failed and not missing_protected else "failed",
        "ascent_direction_failed_segment_ids": ascent_failed,
        "missing_protected_segment_ids": missing_protected,
        "hidden_self_repeat": "failed" if hidden_self_repeat else "passed",
        "hidden_self_repeat_segment_ids": hidden_self_repeat,
        "latent_credit": "needs_ownership_decision"
        if [row for row in latent_rows if row["status"] != "owned_by_preserved_route"]
        else "passed",
        "p75_p90_better_than_current_scope": "passed" if p75_p90_better else "failed",
        "continuous_gpx": "needs_direct_gap_review" if direct_gap_miles else "passed_graph_continuity",
        "direct_gap_fallback_miles": direct_gap_miles,
        "cue_sheet": "needs_human_cue_rewrite",
        "recertification": "not_run_against_active_field_packet",
    }


def build_bundle(
    definition: dict[str, Any],
    *,
    field_tool_data: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    connector_graph: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    routes_by_label = route_by_label(field_tool_data)
    owner_by_segment = segment_owner_index(field_tool_data)
    replace_labels = list(definition.get("replace_route_labels") or [])
    preserve_labels = list(definition.get("preserve_route_labels") or [])
    replaced_current = route_metrics_for_labels(routes_by_label, replace_labels)
    preserved_current = route_metrics_for_labels(routes_by_label, preserve_labels)
    current_scope = {
        "on_foot_miles": round_miles(replaced_current["on_foot_miles"] + preserved_current["on_foot_miles"]),
        "p75_minutes": replaced_current["p75_minutes"] + preserved_current["p75_minutes"],
        "p90_minutes": replaced_current["p90_minutes"] + preserved_current["p90_minutes"],
    }
    loops = []
    gpx_paths = []
    for loop_def in definition.get("loops") or []:
        loop = build_generated_route(
            variant_id=loop_def["loop_id"],
            strategy=loop_def["strategy"],
            ordered_segment_ids=normalized_ids(loop_def["segment_ids"]),
            parking=parking_for_trailhead(field_tool_data, loop_def["trailhead"]),
            official_by_id=official_by_id,
            connector_graph=connector_graph,
            preserve_ascent_direction=True,
        )
        loop["trailhead"] = loop_def["trailhead"]
        loop["intent_segment_ids"] = normalized_ids(loop_def["segment_ids"])
        gpx_path = render_loop_gpx(output_dir, definition["bundle_id"], loop)
        gpx_paths.append(display_path(gpx_path))
        loop["gpx_path"] = display_path(gpx_path)
        loop.pop("track_coordinates", None)
        loops.append(loop)
    generated_miles = round_miles(sum(float_value(loop.get("track_miles")) for loop in loops))
    generated_p75 = scaled_minutes(generated_miles, replaced_current["on_foot_miles"], replaced_current["p75_minutes"])
    generated_p90 = scaled_minutes(generated_miles, replaced_current["on_foot_miles"], replaced_current["p90_minutes"])
    candidate_scope = {
        "on_foot_miles": round_miles(generated_miles + preserved_current["on_foot_miles"]),
        "p75_minutes": generated_p75 + preserved_current["p75_minutes"],
        "p90_minutes": generated_p90 + preserved_current["p90_minutes"],
        "pricing_status": "scaled_from_replaced_current_cards_needs_dem_and_field_calibration",
    }
    replaced_ids = set(replaced_current["segment_ids"])
    latent_rows = latent_credit_rows(
        loops=loops,
        replaced_ids=replaced_ids,
        preserved_labels=set(preserve_labels),
        owner_by_segment=owner_by_segment,
    )
    gates = promotion_gates(
        loops=loops,
        current_scope=current_scope,
        candidate_scope=candidate_scope,
        latent_rows=latent_rows,
        protected_segment_ids=normalized_ids(definition.get("protected_segment_ids") or []),
    )
    replaced_impacts = route_impact_rows(routes_by_label, replace_labels, set(union_loop_ids(loops, "official_segment_ids")))
    preserved_impacts = route_impact_rows(routes_by_label, preserve_labels, set(union_loop_ids(loops, "official_segment_ids")))
    return {
        "bundle_id": definition["bundle_id"],
        "shape": definition["shape"],
        "intent": definition["intent"],
        "status": gates["status"],
        "replace_route_labels": replace_labels,
        "preserve_route_labels": preserve_labels,
        "current_replaced_scope": replaced_current,
        "current_preserved_scope": preserved_current,
        "current_total_scope": current_scope,
        "candidate_generated_scope": {
            "loop_count": len(loops),
            "on_foot_miles": generated_miles,
            "p75_minutes_scaled": generated_p75,
            "p90_minutes_scaled": generated_p90,
        },
        "candidate_total_scope": candidate_scope,
        "delta_vs_current_total_scope": {
            "on_foot_miles": round_miles(candidate_scope["on_foot_miles"] - current_scope["on_foot_miles"]),
            "p75_minutes": candidate_scope["p75_minutes"] - current_scope["p75_minutes"],
            "p90_minutes": candidate_scope["p90_minutes"] - current_scope["p90_minutes"],
        },
        "route_impacts": {
            "replaced_routes": replaced_impacts,
            "preserved_routes": preserved_impacts,
        },
        "promotion_gates": gates,
        "latent_credit_review": latent_rows,
        "generated_loops": loops,
        "gpx_paths": gpx_paths,
    }


def recommendation_for_bundle(bundle: dict[str, Any]) -> str:
    if bundle["promotion_gates"]["p75_p90_better_than_current_scope"] != "passed":
        return "do_not_promote_current_cards_are_cheaper"
    if bundle["promotion_gates"]["status"] != "promotion_candidate":
        return "do_not_promote_until_hard_gates_clear"
    return "promotion_candidate_after_full_recertification"


def build_report(
    field_tool_data: dict[str, Any],
    official_segments: list[dict[str, Any]],
    connector_graph: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    bundles = [
        build_bundle(
            definition,
            field_tool_data=field_tool_data,
            official_by_id=official_by_id,
            connector_graph=connector_graph,
            output_dir=output_dir,
        )
        for definition in CANDIDATE_BUNDLES
    ]
    for bundle in bundles:
        bundle["recommendation"] = recommendation_for_bundle(bundle)
    best = min(bundles, key=lambda row: row["delta_vs_current_total_scope"]["on_foot_miles"])
    return {
        "schema": "boise_trails_freestone_military_candidate_bundles_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "candidate_bundles_generated_no_promotions",
        "source_files": {
            "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
            "official_segments_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "connector_geojson": display_path(DEFAULT_CONNECTOR_GEOJSON),
        },
        "scope": {
            "component": "Freestone / Military Reserve",
            "policy": "Do not solve the whole 39.54-mile component as one route. Test smaller bundles and keep promotion gates strict.",
            "promotion_gates": [
                "all ascent-only segments preserved in required direction",
                "no hidden self-repeat",
                "no newly unowned latent credit",
                "p75/p90 and on-foot better than current scope",
                "cue sheet remains understandable",
                "partial replacements prove remaining route cards shrink",
            ],
        },
        "summary": {
            "bundle_count": len(bundles),
            "promotion_candidate_count": sum(1 for bundle in bundles if bundle["status"] == "promotion_candidate"),
            "best_delta_bundle_id": best["bundle_id"],
            "best_delta_on_foot_miles": best["delta_vs_current_total_scope"]["on_foot_miles"],
            "recommendation": "preserve_current_cards_until_a_candidate_beats_cost_and_hard_gates",
        },
        "bundles": bundles,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Freestone / Military Candidate Bundle Experiment",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Bundles tested: {report['summary']['bundle_count']}",
        f"- Promotion candidates: {report['summary']['promotion_candidate_count']}",
        f"- Best on-foot delta: `{report['summary']['best_delta_bundle_id']}` at {report['summary']['best_delta_on_foot_miles']} mi",
        f"- Recommendation: `{report['summary']['recommendation']}`",
        "",
        "No bundle below is promoted. The current cards stay cheaper or the hard gates fail.",
        "",
        "## Bundle Results",
        "",
        "| Bundle | Shape | Generated loops | Current scope | Candidate scope | Delta | Gates | Recommendation |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for bundle in report.get("bundles") or []:
        current = bundle["current_total_scope"]
        candidate = bundle["candidate_total_scope"]
        delta = bundle["delta_vs_current_total_scope"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{bundle['bundle_id']}`",
                    bundle["shape"],
                    str(bundle["candidate_generated_scope"]["loop_count"]),
                    f"{current['on_foot_miles']} mi / {current['p75_minutes']} p75",
                    f"{candidate['on_foot_miles']} mi / {candidate['p75_minutes']} p75",
                    f"{delta['on_foot_miles']} mi / {delta['p75_minutes']} p75",
                    ", ".join(bundle["promotion_gates"]["hard_failures"]),
                    f"`{bundle['recommendation']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- F1 does not beat the current FD19C / FD04A / FD20A cost once the remaining Freestone Ridge work is priced.",
            "- F2 correctly keeps route 3 separate, but the compact FD19C+FD04A loop is still more on-foot effort than the current cards.",
            "- F3 preserves Curlew ascent direction in generation, but combining Curlew/Fat Tire with Freestone work is worse than keeping FD06A and FD20A separate.",
            "- The next useful experiment is not a larger loop; it is finding a cleaner legal connector/cue pattern that removes hidden self-repeat without adding p75.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_data = read_json(args.field_tool_data_json)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    report = build_report(field_tool_data, official_segments, connector_graph, args.output_dir)
    report["source_files"] = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "official_segments_geojson": display_path(args.official_geojson),
        "connector_geojson": display_path(args.connector_geojson),
    }
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    output_paths = [args.output_json, args.output_md] + [Path(path) for bundle in report["bundles"] for path in bundle["gpx_paths"]]
    manifest = build_artifact_manifest(
        run_id="freestone_military_candidate_bundle_experiment",
        command="python years/2026/scripts/freestone_military_candidate_bundle_experiment.py",
        inputs=[args.field_tool_data_json, args.official_geojson, args.connector_geojson],
        outputs=output_paths,
        metadata={"status": report.get("status"), "summary": report.get("summary")},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    for path in [path for bundle in report["bundles"] for path in bundle["gpx_paths"]]:
        print(f"Wrote {path}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
