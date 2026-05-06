# P90 Relaxed-Drive GPX Readiness Audit

Objective: audit GPX export readiness for selected relaxed-drive field-day loops

## Summary

- Selected loops: 50
- GPX or stored-geometry available loops: 50
- Needs lookup/regeneration loops: 0
- Selected loop GPX ready: True
- Day-level GPX ready: True
- Day-level GPX files: 31
- Day-level GPX failed days: 0
- Readiness counts: `{'existing_navigation_gpx_available': 12, 'generated_forced_anchor_gpx_available': 3, 'stored_geometry_exportable': 35}`
- Source counts: `{'canonical_field_menu': 12, 'forced_anchor_probe': 3, 'hybrid_candidate_index': 7, 'personal_route_menu': 28}`

## Known Gaps

- This audit checks source geometry availability only; it does not export GPX.
- Canonical field-menu rows are resolved through the phone field-packet manifest when possible.
- Forced-anchor probe rows are resolved through the regenerated forced-anchor GPX manifest when available.
- Day-level GPX readiness is read from the day-GPX export manifest when that artifact exists.

## Non-Exportable / Needs Lookup Rows

| Day | Source | Readiness | Label | Trailhead |
|---:|---|---|---|---|
