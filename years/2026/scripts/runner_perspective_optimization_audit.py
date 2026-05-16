#!/usr/bin/env python3
"""Turn runner-perspective route scans into optimization hypotheses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from runner_perspective_frame_shift_audit import (
    CONNECTORS_GEOJSON,
    FIELD_TOOL_DATA,
    FRAME_LOG,
    OFFICIAL_SEGMENTS,
    OUTPUT_DIR,
    R2R_GEOJSON,
    REPO_ROOT,
    access_status,
    clean_text,
    load_features,
    load_json,
    route_audit,
    signed_as,
    slugify,
    utc_now,
)

OPTIMIZATION_DIR = OUTPUT_DIR / "optimization-audits"


def cue_signed_tokens(cue: dict[str, Any]) -> set[str]:
    tokens = set()
    for name in cue.get("signed_as") or []:
        cleaned = clean_text(name).lower()
        tokens.add(cleaned)
        tokens.update(part.strip() for part in cleaned.replace("#", " ").replace("-", " ").split() if len(part.strip()) > 2)
    return tokens


def unrelated_nearby_paths(cue: dict[str, Any], nearby: list[dict[str, Any]], radius_m: int = 90) -> list[dict[str, Any]]:
    tokens = cue_signed_tokens(cue)
    hits = []
    for hit in nearby:
        if not hit.get("path") or hit.get("distance_m", 9999) > radius_m:
            continue
        name = clean_text(hit["name"]).lower()
        if name in tokens:
            continue
        if any(token and token in name for token in tokens):
            continue
        hits.append(hit)
    return hits


def add_lead(leads: list[dict[str, Any]], **lead: Any) -> None:
    key = (
        lead.get("type"),
        lead.get("checkpoint"),
        lead.get("hypothesis"),
    )
    for existing in leads:
        if (existing.get("type"), existing.get("checkpoint"), existing.get("hypothesis")) == key:
            return
    leads.append(lead)


def route_opportunity_score(lead: dict[str, Any]) -> int:
    priority = lead.get("priority")
    if priority == "high":
        return 4
    if priority == "medium":
        return 2
    return 1


def build_opportunities(route: dict[str, Any], audit: dict[str, Any]) -> list[dict[str, Any]]:
    leads: list[dict[str, Any]] = []
    official = float(route.get("official_miles") or 0)
    on_foot = float(route.get("on_foot_miles") or 0)
    ratio = on_foot / official if official else 0.0
    overhead = max(0.0, on_foot - official)
    route_access_status = access_status(route)

    if ratio >= 2.0 or overhead >= 5.0:
        add_lead(
            leads,
            priority="high" if ratio >= 2.0 and overhead >= 3.0 else "medium",
            type="access_or_connector_overhead",
            checkpoint="whole route",
            hypothesis=(
                f"Runner-view overhead is large: {on_foot:.2f} on-foot miles for {official:.2f} official miles "
                f"({ratio:.2f}x, {overhead:.2f} non-new-credit miles). Search for a different parked start, "
                "split, re-park, or connector sequence before accepting this as a fixed route cost."
            ),
            evidence=f"field-packet route totals for {route['label']}",
            proof_needed="Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.",
        )

    if "proof-sensitive" in route_access_status or "private-history" in route_access_status or "incomplete" in route_access_status:
        add_lead(
            leads,
            priority="high" if "proof-sensitive" in route_access_status else "medium",
            type="access_anchor",
            checkpoint="start/finish",
            hypothesis=f"The parking/access anchor is not a fully public-certifiable known lot in the packet: {route_access_status}.",
            evidence="field-packet parking metadata",
            proof_needed="Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.",
        )

    for cue in route.get("wayfinding_cues") or []:
        seq = int(cue.get("seq") or 0)
        checkpoint_label = f"cue {seq:02d}"
        leg_miles = float(cue.get("leg_miles") or 0)
        cue_type = clean_text(cue.get("cue_type"))
        note = clean_text(cue.get("note"))
        warning = clean_text(cue.get("field_warning"))
        signed = signed_as(cue)

        if leg_miles >= 2.0 and cue_type in {"start_access", "exit_access", "connector_road", "connector_named_trail", "overlap_repeat"}:
            add_lead(
                leads,
                priority="high" if leg_miles >= 4.0 else "medium",
                type="long_non_credit_leg",
                checkpoint=checkpoint_label,
                hypothesis=(
                    f"Long {cue_type} leg ({leg_miles:.2f} mi) appears from the runner frame as a candidate for "
                    "re-parking, a better access anchor, or a shorter legal connector."
                ),
                evidence=f"cue `{signed}` to `{clean_text(cue.get('target'))}`",
                proof_needed="Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.",
            )

        if warning or cue_type == "overlap_repeat" or "Overlap warning" in note:
            add_lead(
                leads,
                priority="high" if leg_miles >= 1.0 else "medium",
                type="overlap_or_double_back",
                checkpoint=checkpoint_label,
                hypothesis=(
                    f"The runner would experience `{signed}` as repeated/overlapping corridor movement. "
                    "Treat that as route-choice evidence, not only a map-warning problem."
                ),
                evidence=warning or note,
                proof_needed="Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.",
            )

        if "connector/repeat" in note or "Official-repeat" in note:
            add_lead(
                leads,
                priority="medium",
                type="connector_repeat_inside_credit_cue",
                checkpoint=checkpoint_label,
                hypothesis=(
                    f"The official-looking cue also carries connector/repeat movement on `{signed}`. "
                    "This is a signal to reprice the movement after credit is satisfied."
                ),
                evidence=note,
                proof_needed="Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.",
            )

        if "Reverse direction would be steep" in note or "ASCENT REQUIRED" in note:
            add_lead(
                leads,
                priority="medium",
                type="direction_cost_boundary",
                checkpoint=checkpoint_label,
                hypothesis=(
                    "The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction "
                    "or explicitly pay the climb penalty; do not blindly reverse or combine this route."
                ),
                evidence=note,
                proof_needed="When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.",
            )

        if any("OSM " in name for name in cue.get("signed_as") or []):
            add_lead(
                leads,
                priority="medium",
                type="generic_connector_proof",
                checkpoint=checkpoint_label,
                hypothesis=(
                    f"`{signed}` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, "
                    "so it is a proof target and possible replacement target."
                ),
                evidence="field-packet cue signed_as contains OSM connector",
                proof_needed="Verify signage/imagery or replace with a named legal trail/road connector if available.",
            )

    checkpoint_by_label = {checkpoint["label"]: checkpoint for checkpoint in audit["checkpoints"]}
    cue_by_seq = {int(cue.get("seq") or 0): cue for cue in route.get("wayfinding_cues") or []}
    for checkpoint in audit["checkpoints"]:
        if not checkpoint["label"].startswith("Cue "):
            nearby_vehicle = [hit for hit in checkpoint["nearby"] if hit.get("vehicle") and hit.get("distance_m", 9999) <= 120]
            if nearby_vehicle:
                add_lead(
                    leads,
                    priority="medium",
                    type="start_finish_vehicle_context",
                    checkpoint=checkpoint["label"],
                    hypothesis=(
                        "A vehicle corridor or service/residential road is close to the start/finish. "
                        "This can be a parking-access optimization lead or a false shortcut to reject."
                    ),
                    evidence=", ".join(f"{hit['name']} ~{hit['distance_m']}m" for hit in nearby_vehicle[:4]),
                    proof_needed="Classify public legality, passability, and whether it improves or harms p75 route cost.",
                )
            continue

        seq_text = checkpoint["label"].split(":", 1)[0].replace("Cue", "").strip()
        try:
            seq = int(seq_text)
        except ValueError:
            continue
        cue = cue_by_seq.get(seq)
        if not cue:
            continue
        side_paths = unrelated_nearby_paths(cue, checkpoint["nearby"])
        if side_paths:
            add_lead(
                leads,
                priority="low" if len(side_paths) < 3 else "medium",
                type="nearby_branch_scan",
                checkpoint=f"cue {seq:02d}",
                hypothesis=(
                    "The runner-frame scan sees nearby mapped branches that are not the cue target. "
                    "Most will be distractions, but this is where unexpected connector substitutions can appear."
                ),
                evidence=", ".join(f"{hit['name']} ~{hit['distance_m']}m" for hit in side_paths[:5]),
                proof_needed="Price only named/legal branches that connect to a useful next cue without losing official edge coverage.",
            )

    leads.sort(key=lambda lead: (-route_opportunity_score(lead), lead.get("type", ""), lead.get("checkpoint", "")))
    return leads


def render_route_optimization(route: dict[str, Any], audit: dict[str, Any], leads: list[dict[str, Any]]) -> str:
    official = float(route.get("official_miles") or 0)
    on_foot = float(route.get("on_foot_miles") or 0)
    ratio = on_foot / official if official else 0.0
    high_count = sum(1 for lead in leads if lead["priority"] == "high")
    medium_count = sum(1 for lead in leads if lead["priority"] == "medium")
    low_count = sum(1 for lead in leads if lead["priority"] == "low")
    lines = [
        f"# Runner-Perspective Optimization Audit: {route['label']} - {route.get('trailhead')}",
        "",
        "## Reframed Contract",
        "",
        "The visual/runner question is an optimization search tool, not the final user-facing narration.",
        "",
        "- Model frame: the route card validates and has cue/GPX artifacts.",
        "- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.",
        "- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.",
        "",
        "## Route Cost Surface",
        "",
        f"- Official miles: {official:.2f}.",
        f"- On-foot miles: {on_foot:.2f}.",
        f"- On-foot/official ratio: {ratio:.2f}x.",
        f"- Door-to-door p75/p90: {route.get('door_to_door_minutes_p75')} / {route.get('door_to_door_minutes_p90')} min.",
        f"- Access status: {audit['access_status']}.",
        f"- Lead count: {high_count} high, {medium_count} medium, {low_count} low.",
        "",
        "## Optimization Leads",
        "",
    ]
    if not leads:
        lines.append("- No route-specific optimization lead was detected by this local-data runner-perspective pass. Keep the route in the normal day-of condition/signage proof flow.")
    else:
        for index, lead in enumerate(leads, 1):
            lines.extend(
                [
                    f"### {index}. {lead['priority'].upper()} - {lead['type']} ({lead['checkpoint']})",
                    "",
                    f"- Hypothesis: {lead['hypothesis']}",
                    f"- Evidence: {lead['evidence']}",
                    f"- Proof needed: {lead['proof_needed']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Do Not Infer",
            "",
            "- A nearby path is not automatically a legal or better connector.",
            "- A road near the route is not automatically legal parking.",
            "- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.",
            "- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def render_index(records: list[dict[str, Any]]) -> str:
    ranked = sorted(records, key=lambda record: (-record["score"], -record["high"], -record["ratio"], record["label"]))
    lines = [
        "# Runner-Perspective Optimization Index",
        "",
        "This is the corrected use of the `what do you see?` exercise: visualizing the runner at each start, end, junction, and cue is a way to uncover optimization hypotheses outside the model's original route-card frame.",
        "",
        "The route-card visual audit remains in `route-audits/`. This index turns those observations into optimization leads and proof queues.",
        "",
        "Public route behavior evidence is recorded separately in `public-route-behavior-evidence.md`. Use it as behavioral evidence for how people solve the terrain in real life, not as official BTC proof or current condition truth.",
        "",
        "## Highest-Leverage Leads",
        "",
        "| Route | Score | High | Med | Ratio | Main lead | File |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in ranked[:15]:
        main = record["main_lead"] or "normal proof flow"
        lines.append(
            f"| {record['label']} | {record['score']} | {record['high']} | {record['medium']} | {record['ratio']:.2f}x | {main} | [{record['file']}](optimization-audits/{record['file']}) |"
        )
    lines.extend(
        [
            "",
            "## All Routes",
            "",
            "| Route | Trailhead | High | Med | Low | Ratio | File |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['label']} | {record['trailhead']} | {record['high']} | {record['medium']} | {record['low']} | {record['ratio']:.2f}x | [{record['file']}](optimization-audits/{record['file']}) |"
        )
    lines.extend(
        [
            "",
            "## Cross-Route Optimization Themes",
            "",
            "- High on-foot/official ratios and long exit/access legs are the first place to look for re-parks, split routes, or better public anchors.",
            "- Overlap and double-back warnings are not just navigation warnings; they are candidates for post-credit connector re-optimization.",
            "- Road or service corridors near decision points can be useful only after access legality and p75 field cost are proved.",
            "- Nearby branches are search-space hints, not route recommendations. Price them only when they are named/legal and preserve official edge coverage.",
            "- Generic OSM connectors should trigger imagery/signage proof or replacement with a named trail/road connector.",
            "",
        ]
    )
    return "\n".join(lines)


def write_frame_logs(records: list[dict[str, Any]]) -> None:
    FRAME_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now()
    with FRAME_LOG.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "timestamp_utc": timestamp,
                        "cwd": str(REPO_ROOT),
                        "artifact_type": "route optimization audit",
                        "decision": "reframe",
                        "original_assumption": "Runner visual descriptions are the output.",
                        "challenge": "The user clarified that what-do-you-see is a thinking tool for unexpected optimization.",
                        "new_frame": "Use runner perspective to generate route repair, re-anchor, split, connector, and proof hypotheses.",
                        "resulting_perspective": f"{record['label']} has {record['high']} high and {record['medium']} medium optimization leads.",
                        "widened_options": ["visual field report", "route-card proof", "optimization hypothesis queue"],
                        "frame_iterations": ["literal seeing", "runner decision point", "unexpected optimization search"],
                        "evidence_level": "field-packet route data plus local R2R/OSM overlays",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-frame-log", action="store_true")
    args = parser.parse_args(argv)

    field_tool_data = load_json(FIELD_TOOL_DATA)
    features = load_features([R2R_GEOJSON, CONNECTORS_GEOJSON])
    route_dir = args.output_dir / "optimization-audits"
    route_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for route in field_tool_data.get("routes") or []:
        audit = route_audit(route, features)
        leads = build_opportunities(route, audit)
        filename = f"{slugify(route['label'])}-{slugify(str(route.get('trailhead')))}.md"
        (route_dir / filename).write_text(render_route_optimization(route, audit, leads), encoding="utf-8")
        official = float(route.get("official_miles") or 0)
        on_foot = float(route.get("on_foot_miles") or 0)
        high = sum(1 for lead in leads if lead["priority"] == "high")
        medium = sum(1 for lead in leads if lead["priority"] == "medium")
        low = sum(1 for lead in leads if lead["priority"] == "low")
        score = sum(route_opportunity_score(lead) for lead in leads)
        records.append(
            {
                "label": route["label"],
                "trailhead": route.get("trailhead"),
                "file": filename,
                "lead_count": len(leads),
                "high": high,
                "medium": medium,
                "low": low,
                "score": score,
                "ratio": on_foot / official if official else 0.0,
                "main_lead": leads[0]["type"] if leads else "",
            }
        )

    (args.output_dir / "optimization-index.md").write_text(render_index(records), encoding="utf-8")
    manifest = {
        "generated_at": utc_now(),
        "route_count": len(records),
        "inputs": [
            str(FIELD_TOOL_DATA.relative_to(REPO_ROOT)),
            str(R2R_GEOJSON.relative_to(REPO_ROOT)),
            str(CONNECTORS_GEOJSON.relative_to(REPO_ROOT)),
            str(OFFICIAL_SEGMENTS.relative_to(REPO_ROOT)),
        ],
        "outputs": ["optimization-index.md", "optimization-audits/*.md"],
        "routes": records,
    }
    (args.output_dir / "optimization-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_frame_log:
        write_frame_logs(records)
    print(f"Wrote {len(records)} optimization audits to {args.output_dir}")
    print(f"Total optimization leads: {sum(record['lead_count'] for record in records)}")
    print(f"High-priority leads: {sum(record['high'] for record in records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
