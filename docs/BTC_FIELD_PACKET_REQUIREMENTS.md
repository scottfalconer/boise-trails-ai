# BTC Field Packet And Live Map Requirements

Load this before generating, reviewing, debugging, publishing, or claiming readiness for the field packet, GPX exports, live map, phone cues, or field-route audits.

## Canonical Field Menu Source

The executable field menu has one canonical data source per run:

- Private canonical source: `years/2026/outputs/private/2026-outing-menu-map-data.json`.
- Private map view: `years/2026/outputs/private/2026-outing-menu-map.html`.
- Private written view: `years/2026/outputs/private/2026-outing-menu.md`.
- Public sanitized source: `outing-menu-map-data.json` and `years/2026/outputs/examples/2026-outing-menu-map-data.example.json`.
- Public sanitized views: `outing-menu-map.html`, `outing-menu.md`, and `docs/field-packet/`.

Do not point the browser map, written outing menu, phone field packet, GPX exports, and public example artifacts at different route-pass files. `human_loop_plan.py` writes the private canonical map-data JSON, HTML map, and written menu. `export_example_map.py` exports the sanitized public map-data JSON, map, and menu from that same payload. `export_mobile_field_packet.py` must consume the canonical map-data JSON first, falling back only to the sanitized public map data when private data is unavailable.

Route-experience/block-review artifacts such as `block-hybrid-day-package-pass-v1-map-data.json`, human-loop markdown, or manual-design reports are upstream review inputs. They are not field-menu sources until promoted into `2026-outing-menu-map-data.json`.

## Field-Day Execution Layer

When a `field_day_layer` artifact exists, it is the calendar and sequencing layer for challenge-window decisions: morning/evening window, total p75/p90 cost, re-parks, water, heat exposure, and work/family constraints. The phone packet should still default to Route Cards first, because certified route cards are the field navigation and proof units; Field Days stays available as the secondary calendar view.

Do not treat weekday/weekend labels as route-capacity proof. Field-day p75/p90 bounds should come from explicit dated availability and hard stops; weekdays can be as open, or more open, than weekends for this user.

Certified route cards remain the proof and navigation units under that layer. Each field-day loop should link to the certified route card for GPX, parking, cue, return-to-car detail, segment coverage, and ascent-direction evidence.

Loops without a `route_card_ref` are promotion gaps. Loops with route-card parking, cue/card mileage, missing GPX, or field-navigation blockers are audit-fix gaps. Both may stay visible in the field-day layer so the plan's missing certification work is obvious, but neither is publication-ready until promoted into audit-clean certified route cards and regenerated through the packet/audit chain.

Certification triage should be field-day scoped first. Print or inspect the blockers on selected field-day loops before treating the full route-card inventory audit as the next action list. Full-inventory failures are still real, but the execution decision is whether the selected dated field days have audit-clean route cards, day-level handoffs, and route-distance-authoritative totals.

## Phone And GPX Field Contract

- Field navigation must describe the full route from parked car back to parked car, not only official challenge-credit segments.
- If the first official segment is not physically at the trailhead, include the named access trail, connector, road, or path needed to reach it.
- If the last official trail does not physically return to the parked car, include the named return trail, connector, road, or path back to the trailhead.
- Keep `official segment order` separate from `actual GPX traversal / turn-by-turn from the car`.
- Do not populate phone `Turn-by-turn from car` as one row per official segment. It should be trail-transition navigation.
- Use signpost-oriented cues such as `At #51 Who Now Loop, take the right arrow toward #52 Kemper's Ridge`, not abstract geometry-only language.
- Treat non-official route legs as first-class field instructions.
- Phone field instructions should include a text-first `Field Cue Sheet` / `wayfinding_cues` layer. Each cue should tell the runner what to follow, until what observable junction/landmark, and what target comes next.
- Route-card `Field Cue Sheet` content should be collapsed by default in the phone packet so the route-card list remains scannable; the full cue sheet must still be one tap away on the same card.
- Audit/reconciliation sections such as `Cross-route segment ownership` should stay in JSON audit data, not in phone-visible route cards.
- Do not accept target-only cues for nontrivial access, connector, repeat, or return legs.
- Preserve official challenge segment ids separately from `wayfinding_cues`; phone-visible cue numbers are field decision order, not official segment order.
- Keep a clean default navigation GPX separate from dense audit data. Dense official-segment midpoint waypoints belong in audit GPX.
- Map and phone outputs should explicitly call out mid-route car access and verified water.

Named access regression guard: whenever the route-line match or configured field hints identify a named non-credit access or return trail, the primary field-usable name must appear in both the phone cue sheet and turn-by-turn steps. Do not accept a target-only cue such as "head toward the first official segment" when the car-to-segment path is actually a signed trail, connector, or road.

## Live Map Contract

- `docs/field-packet/live-map.html` is generated by `export_mobile_field_packet.py`, not hand-edited.
- The live map is a field-navigation artifact, not a decorative review map. Its primary job is: "I am here, this is the active cue-to-cue leg, this is the next cue/junction, and this is what to follow until then."
- Its default state should highlight the active wayfinding leg, mute the rest of the route, keep sparse direction chevrons on the active leg, show current/next cue markers, and provide manual cue stepping.
- On initial load, the live map must not skip the first field-visible cue. If the parked start and first movement leg share the same route-mile anchor, either suppress/merge the zero-length start cue in the generated cue sheet or explicitly render it as a start marker without making the active leg begin at cue 2+.
- A full-route overview can exist as a secondary fit/view, but it must not be the primary field-following mode for dense self-overlap.
- `Start GPS` displays the user dot, accuracy circle, distance-to-route, and progress estimate, but it should not auto-recenter, auto-follow, or auto-step the active cue.
- The runner must be able to pan and pinch/zoom directly. Do not reintroduce a `Follow` toggle unless explicitly asked.
- The user dot and heading marker should use screen-stable sizing, and an offscreen GPS fix should render an edge indicator or clear status such as `GPS off map` without recentering.
- Live-map arrows and markers must use the same displayed active-leg geometry as the highlighted ribbon. Do not sample arrow direction from raw dense GPX while drawing a simplified ribbon.
- Cue, start, and finish markers must not hide the exact junction/start/end point. Use callouts with a small anchor when needed.
- Optional raster basemap tiles may sit behind the route for context, but they must not become a route data source or field-safety dependency.
- Direct `file://` loading is allowed to fail for live GPS/data fetches; validate through a local HTTP server or GitHub Pages HTTPS URL.

Known phone-packet regression guard: a hard reload of `docs/field-packet/index.html` should default to the `All` time filter and show the full runnable menu. Time filters such as `<=2h` should narrow the menu only after the user taps them.

## Overlap And Source-Truth Guards

Same-trail repeat / double-back guard: when the planned route reuses the same trail corridor in opposite or repeated sequence, do not expect the user to decode an overlapping GPX line. Mark the cue as an overlap/double-back in field data, surface that warning in the phone cue card and live-map active-leg banner, and rely on active-leg arrows plus current/next cue labels to disambiguate. Do not offset exported GPX geometry to hide the overlap; any visual offset must be only a clearly schematic display layer.

Source-artifact consistency guard: the Nav GPX, route card, source-gap flags, and phone cue order must all describe the same car-to-car route topology and field decision sequence. GPX track length alone is not a route-decision metric; use GPX for navigation geometry, continuity, and segment coverage. If a repair changes the canonical car-to-car geometry, reprice the route card from that same repaired geometry and regenerate the downstream GPX/cue/map artifacts together. If GPX shape/order, cue order, parking endpoints, or source-gap evidence disagree, fix the canonical route source, route metadata, GPX generation, or certification audit before touching visual presentation.

Route-mile anchor guard: phone-visible cue mileage, live-map `route_miles` / `route_leg_miles` anchors, and the route-card on-foot mileage must reconcile within field-tool tolerance. Do not accept a route where scaled cue labels match the card but the live-map traversal anchors reveal materially longer movement. That is route-truth drift, even if repeated official segments are declared as "no new credit."

Post-credit repeat guard: once a cue has earned an official segment, later no-new-credit movement over that same segment is ordinary connector routing, not protected route order. Field readiness must fail when the connector graph proves a materially shorter legal cue-to-cue path that avoids the already-credited repeat, measured by actual replacement geometry rather than graph-scaled or official mileage alone. If the graph cannot prove an alternate path, preserve an advisory for route-source review rather than treating the repeat as cost-free.

Accepted replacement regression guard: if a field-tested split, re-park, multi-start, or manual repair has been promoted into the active replacement manifest, recertification must prove that its expected package/components are still present or explicitly superseded. Do not let a recalculation silently collapse an accepted human-valid split back into one long map-optimal card.

## Field-Executable Contract

Any published runnable outing must pass this generic car-to-car contract:

- Parked start exists.
- Nav GPX has a non-empty track.
- Inter-`trkseg` gaps are either physically connected or explicitly declared as a re-park/named connector/manual hold.
- Source route gaps are not hidden by splitting the render into a `MultiLineString`.
- Claimed segment ids are covered by the exported GPX geometry.
- GPX-completed official segments that are not claimed by the route card are reconciled before publication: claim them on the route, declare their active owner route, mark them as already completed/repeat connector context, or remove them from conflicting later cards after segment-first validation.
- Ascent-only segments have direction evidence.
- Published land-manager special-management rules are checked against the actual route GPX, including connector and repeat mileage. BTC official `direction`/`ascent` evidence is necessary but not sufficient when Ridge to Rivers or another land manager publishes all-user direction, date/use, or mode restrictions.
- Non-credit start/return legs are described in phone cues.
- A route with `source_gap_warning=true` is not field-ready unless the gap is explicitly represented as named connector trail, public road connector, official repeat connector, intentional re-park/multi-start boundary, or manual day-of access hold.

Do not call a plan or packet ready until segment coverage, BTC official directional rules, and known land-manager special-management rules have been checked against current sources.

## Certification Chain

A field packet is certifiable only after these commands pass on the same regenerated artifacts:

```bash
python years/2026/scripts/export_mobile_field_packet.py
python years/2026/scripts/field_latent_credit_audit.py
python years/2026/scripts/field_progress_report.py
python years/2026/scripts/field_recertification_report.py
python years/2026/scripts/same_anchor_spur_split_audit.py
python years/2026/scripts/route_edge_cover_audit.py
python years/2026/scripts/field_tool_completion_audit.py
python years/2026/scripts/field_route_walkthrough_audit.py
python years/2026/scripts/post_credit_connector_audit.py
```

Do not describe a packet as ready from route-count coverage alone. If source gaps are allowed because they are explicitly represented by connector/re-park/manual metadata, the audit evidence must say that; do not summarize it as "no source gaps." The latent-credit audit is the cross-route segment-claim guard. The field-tool completion audit includes the land-manager special-management hard gate. The walkthrough audit is the headless field-runner check.
The post-credit connector audit is the proof that field-packet connector, repeat, exit, and return cues do not preserve a longer route when a shorter legal connector graph path exists. If an official-credit cue hides post-credit movement, split that movement into an explicit connector or return cue and prove it there instead of burying it inside the credit cue.
The same-anchor spur-split audit is the proof that a small same-parking route has not been left as a separate card when an active route already reaches the spur endpoint and can clear it with materially lower incremental mileage.

## Headless Walker Debugging

Headless-walker fixes should preserve the invariant, not silence the audit.

1. Read the failure as a field-user failure first: `start_access_missing_named_edge`, `named_connector_not_cued`, `hidden_track_gap`, `claimed_segment_not_covered`, `direction_rule_violated`, and `special_management_direction_violated` mean the exported phone packet is not yet field-certifiable.
2. Decide whether the walker is wrong or the packet is wrong. Add a small synthetic regression test before changing code when the issue is generic.
3. If the packet is wrong, fix the generator or canonical route metadata, then regenerate `docs/field-packet/`; do not hand-edit generated HTML/JSON/GPX.
4. If the Nav GPX traverses a route-line-matched named non-credit road/trail/connector, make that name visible in `wayfinding_cues` and `turn_by_turn_steps`.
5. Keep generic OSM connector ids such as `OSM footway connector 72484` out of field-visible cue requirements unless they are the only usable road/path name.
6. Preserve and export `segment_direction_evidence` for ascent/directional segments.
7. Re-run the same certification chain and write a dated checkpoint before saying the packet is ready.

## Progress And Recertification

Phone `completed_outing_ids` are provisional UX state, not proof of challenge credit. Do not promote a completed outing into `completed_segment_ids` or remove its official segments from planner state until an activity geometry validator proves full endpoint-to-endpoint coverage and required ascent direction.

Progress is segment-first. Validated activity reviews should write completed, missed, partial, extra, and blocked segment events to the private progress ledger, then regenerate the materialized active planner state from the locked epoch original plus that ledger. Outing completion is derived only when every official segment in the outing is completed by validated segment state; blocked-only or removed-credit outings are inactive, not completed.

Lock an epoch original before applying progress. Use `pre-challenge-testing` for field-test resets and `challenge-2026` after the real event-day reset. Keep versioned day snapshots under `years/2026/outputs/private/progress/versions/`, including the progress input, activity review, state patch, route delta, progress report, recertification report, and field-packet artifact copy when generated.

After any proven segment completion, missed segment, organizer trail-list change, closure, access blocker, day-of condition blocker, or manually accepted route/parking change, rerun the planner/audit recertification path before treating the remaining menu as field-ready. If a completed segment is still physically needed as connector mileage for a later loop, keep it in the route as official repeat mileage or named connector/repeat context, not as new remaining official credit.

## Durable Fixes

When a field test or review reveals a reusable planner/product bug, implement the generic behavior that prevents the class of bug, add regression coverage that would catch a different route with the same pattern, regenerate affected artifacts, and validate the actual generated packet. Keep route-specific wording only as extra field-tested annotation on top of the generic fix. Do not describe the issue as fixed if only a one-off route patch, prose note, or AGENTS.md instruction exists.

Every generated plan or experiment should record source dataset paths and pull dates, challenge targets, closure/weather/condition assumptions, command/config/model, route list, mileage breakdown, elevation gain, estimated time, heat/shade/water risk notes, coverage validation, GPX readiness checks, field-executable validation, and known caveats.
