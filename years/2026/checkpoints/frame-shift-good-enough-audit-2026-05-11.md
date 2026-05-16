# Frame Shift Good-Enough Audit - 2026-05-11

## Frame decision

Reframe.

`frame-shift` is working as a vocabulary and review posture, but it is not yet
working as a hard prevention mechanism in the BTC route pipeline. The current
system can still treat a proxy such as `field_ready`, `source_gap_warning=false`,
`graph_validated`, or a polished generated packet as enough evidence, even when
the downstream field job needs source-layer completeness, access provenance, and
same-artifact certification.

## Current frame

The planner has often asked:

- Does this route or candidate have parking?
- Is the GPX continuous?
- Did segment coverage validate?
- Is the field packet generated?
- Did an audit pass for the current artifact layer?

Those checks are useful, but they are not the whole contract.

## Downstream job

The runner needs a legal, repeatable, cueable, car-to-car outing menu that:

- covers official segments edge-to-edge and in required direction;
- uses access and parking that are current, public-safe, and source-backed;
- prices connector, repeat, road, transfer, heat, water, and hard-stop cost;
- keeps map, GPX, written menu, phone packet, and field-day layer on one route
  truth;
- can be recertified after completions, closures, access changes, or route
  edits.

## Where good-enough leaked

1. Parking evidence is siloed.

   `amenity=parking` is accepted as a confidence token, and some manual anchors
   were added from ad hoc Overpass checks. But there is no general OSM parking
   inventory/enrichment pass over all starts. Dry Creek / Sweet Connie was
   already `field_ready: true` from SWIMBA/local-source plus Strava evidence, so
   no step forced an OSM lookup that would have attached OSM way `1328228551`.

2. `field_ready: true` short-circuits evidence depth.

   In `p90_forced_anchor_probe.py`, an explicit `field_ready` boolean returns
   true before checking which source layers are present. Confidence strings such
   as `strava_seen`, `validated`, `inferred_from_trailhead_layer`, and
   `osm_amenity` are broad tokens, not a normalized evidence ladder.

3. OSM ways/polygons do not naturally enter the planner.

   `load_trailheads_from_geojson()` only imports point features. The repo has OSM
   PBF inputs, but not a first-class extracted `amenity=parking` layer that
   converts nodes, ways, and relations to candidate parking anchors with OSM ids,
   tags, timestamps, centroids, and match distance to existing anchors.

4. Promotion can mark a source route clean while carrying parking caveats.

   `human_loop_plan.py` promotes accepted manual split routes into executable
   menu components while adding `parking_access_day_of_check_required`, but it
   also writes replacement route validations with `source_gap_warning: false`.
   That can be correct for GPX/source continuity while still misleading if read
   as full field-readiness.

5. Field-day certification is too coarse for source depth.

   `export_field_day_layer.py` checks route validation, `has_parking`,
   wayfinding cues, cue/card mileage, and GPX existence. That is useful, but
   `has_parking` does not prove current access evidence, OSM/source enrichment,
   signage, closure status, or day-level transfer/GPX handoff readiness.

6. Names like `ready`, `validated`, and `certified` carry different meanings in
   different layers.

   A calendar route can be `simulated_ready`, a route can be
   `graph_validated`, a route card can be `certified_route_card`, and an access
   anchor can be `field_ready`. Those are not interchangeable, but the naming
   makes it easy for later review to blur them.

## Adjacent frames checked

- Original frame: the route already has a working start and a generated map.
  This misses missing provenance and route-cost opportunities.
- Stronger frame: every promoted route must carry an evidence inventory for the
  specific claims it makes: parking, access, route continuity, segment coverage,
  timing, field cues, and day-level execution.
- Wider frame: selected field days, not individual pretty route cards, are the
  consequence layer during the challenge window. A route-card proof can still
  fail if the day bundle, transfer, water, heat, or remaining-menu recertification
  is wrong.

Stop reason: the wider frame changes the next artifact shape. The repair should
not be another prose warning; it should be an evidence inventory and promotion
gate that the generator/audits can enforce.

## Adversarial failure stories

- OSM has a visible `P` at the chosen start, but the planner does not attach it
  because private/history evidence already made the anchor `field_ready`.
- A manual route has continuous GPX and `source_gap_warning=false`, but the
  public-safe parking/access evidence remains a day-of caveat.
- A field-day loop matches a route card by segment set, but the route card's
  parking source is stale or incomplete.
- A polished live map loads and follows the active cue, but it is generated from
  a candidate that has not passed the same source route, GPX, cue, parking, and
  recertification chain.
- A route is credit-correct but keeps unnecessary repeat mileage after the credit
  purpose has already been satisfied.
- A route remains on the active menu after a completion, access blocker, or route
  edit because the remaining menu was not recertified from the locked epoch
  original.
- A route answers the literal "can we start here?" question, but misses the
  better adjacent question: "is there a nearby certifiable parking surface that
  makes the whole outing more field-executable?"

## Required artifact changes

1. Add a normalized parking/access evidence inventory for every anchor:
   `source_layers`, OSM ids, source timestamps, evidence level, public-safe label,
   field/day-of caveats, and matched existing anchor id.
2. Extract OSM `amenity=parking` from PBF or Overpass into a durable layer,
   including nodes, ways, and relations. Convert non-point features to centroids
   while preserving OSM ids and tags.
3. Enrich existing anchors before adding duplicates. Dry Creek / Sweet Connie
   should become one enriched anchor, not a second nearby parking point.
4. Split status names by layer: graph-ready, access-evidence-ready,
   route-card-certified, field-day-executable, and publication-ready.
5. Make promotion/audit reports list which claim each verifier proves and what it
   explicitly does not prove.
6. Add a regression/eval for this exact pattern: an anchor already looks
   field-ready from one evidence lane, but OSM has a public `amenity=parking`
   feature at the same start that must be attached before final access claims.

## Decision

The skill is directionally right but operationally insufficient.

Do not treat the current route menu, parking anchors, or field-day layer as
"clear-eyed reviewed" until the evidence inventory and selected-field-day audit
queue exist. Current artifacts may remain useful working maps, but promotion
claims need the stronger gate.

## Validation

This checkpoint is an audit note from code/data review. No tests were run for
this note.
