# 2026 Challenge Start Reset

Use this before the real event, and use it now when testing the planner as if the challenge has not started.

## Reset Meaning

A clean start means:

- `completed_segment_ids` is empty.
- `blocked_segment_ids` is empty unless there are known current closures or user-specific blocks.
- `blocked_trail_names` is empty unless there are known current closures or user-specific blocks.
- The canonical outing map and written outing menu are regenerated from that state.
- Manual design holds remain in place; a reset should not promote draft/manual route areas into runnable outings.

Current reset state on 2026-05-05:

```json
{
  "completed_segment_ids": [],
  "blocked_segment_ids": [],
  "blocked_trail_names": []
}
```

## Files To Check

Private state:

```text
years/2026/inputs/personal/2026-planner-state.private.json
```

Canonical review outputs:

```text
years/2026/outputs/private/2026-outing-menu-map.html
years/2026/outputs/private/2026-outing-menu.md
years/2026/outputs/private/route-blocks/human-loop-plan-v1.md
years/2026/outputs/private/route-blocks/block-hybrid-day-package-pass-v1-map-data.json
```

## Event-Day Reset Procedure

1. Refresh/verify official 2026 challenge data if the organizer has changed trails, dates, or direction rules.

2. Run the reset command:

```bash
python years/2026/scripts/reset_challenge_start.py
```

This command:

- backs up `years/2026/inputs/personal/2026-planner-state.private.json` under `years/2026/outputs/private/reset/state-backups/`;
- clears `completed_segment_ids`, `blocked_segment_ids`, and `blocked_trail_names`;
- regenerates the private personal route menu, block route passes, route package map data, manual-design report, canonical map, and written outing menu;
- writes `years/2026/outputs/private/reset/challenge-start-reset-latest.json` with the command list and reset verification.

If real current closures should remain blocked, use:

```bash
python years/2026/scripts/reset_challenge_start.py --preserve-blocks
```

3. Confirm the reset private state:

```json
{
  "completed_segment_ids": [],
  "blocked_segment_ids": [],
  "blocked_trail_names": []
}
```

If there are confirmed current closures, preserve them with `--preserve-blocks` or re-add them to `blocked_segment_ids` / `blocked_trail_names` and document the source/date in the planning log.

4. Confirm the map-data progress block is clean:

```bash
jq '.progress' years/2026/outputs/private/route-blocks/block-hybrid-day-package-pass-v1-map-data.json
```

Expected output:

```json
{
  "completed_segment_ids": [],
  "blocked_segment_ids": []
}
```

5. Open the canonical map:

```text
years/2026/outputs/private/2026-outing-menu-map.html
```

The top-level map state should show the full start-state outing universe, with completed outings hidden only after `completed_segment_ids` is updated and the map is regenerated.

## Manual Equivalent

Use this only if the reset script needs to be debugged.

1. Edit the private state file so challenge progress is clean.

2. Regenerate the private personal route menu:

```bash
python years/2026/scripts/personal_route_planner.py \
  --state years/2026/inputs/personal/2026-planner-state.private.json \
  --output-json years/2026/outputs/private/personal-route-menu.json \
  --output-md years/2026/outputs/private/personal-route-menu.md
```

3. Regenerate the route-pass/package/map chain:

```bash
python years/2026/scripts/block_route_candidate_pass.py
python years/2026/scripts/block_day_packager.py
python years/2026/scripts/block_combo_route_pass.py
python years/2026/scripts/block_day_packager.py \
  --route-pass-json years/2026/outputs/private/route-blocks/block-combo-route-pass-v1.json \
  --basename block-combo-day-package-pass-v1
python years/2026/scripts/block_route_assembler.py
python years/2026/scripts/block_hybrid_route_pass.py
python years/2026/scripts/block_day_packager.py \
  --route-pass-json years/2026/outputs/private/route-blocks/block-hybrid-route-pass-v1.json \
  --basename block-hybrid-day-package-pass-v1
python years/2026/scripts/manual_route_design_pass.py
python years/2026/scripts/human_loop_plan.py
```

## Test-Run Reset Performed

On 2026-05-05, the private state and map-data progress were already clean. I then formalized the reset as a script and ran:

```bash
python years/2026/scripts/reset_challenge_start.py
```

This rewrote the full private route/menu/map chain and created:

```text
years/2026/outputs/private/reset/challenge-start-reset-latest.json
years/2026/outputs/private/reset/state-backups/2026-planner-state.private-20260505T191356Z.json
```

The reset record verified:

```json
{
  "completed_segment_ids": [],
  "blocked_segment_ids": []
}
```

Current clean-start map summary from the reset record:

```text
packages = 19
component routes = 25
covered segments = 251
official miles = 164.42
total on-foot miles = 280.23
route cues = 25
map rendered = true
```

Primary rewritten user-facing outputs:

```text
years/2026/outputs/private/route-blocks/human-loop-plan-v1.json
years/2026/outputs/private/route-blocks/human-loop-plan-v1.md
years/2026/outputs/private/2026-outing-menu-map.html
years/2026/outputs/private/2026-outing-menu.md
years/2026/outputs/private/route-blocks/human-loop-plan-v1-artifact-manifest.json
```
