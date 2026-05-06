# Boise Trails AI Article Evidence Bundle

Created: 2026-05-05

## Research Question

Pull the evidence needed for an intro article about doing the Boise Trails
Challenge AI route-planning experiment again in 2026.

The article framing to support:

- Scott does the project because it is fun and because it is a concrete way to
  watch model capabilities change.
- The challenge is a real routing/logistics problem: roughly 170 miles of
  official trails in 30 days, with official segment credit, car logistics,
  heat, conditions, direction rules, and family/work hard stops.
- In 2025, models got promising starts but most of the work stayed at the
  "make the route work at all" layer.
- In 2026, GPT-5.5 + Codex CLI produced a plausible route/menu quickly enough
  to field-test.
- The field test moved the work up the value chain: from graph/route validity
  into field execution, signpost cues, phone packet design, and progress
  recovery after a missed turn.

## Scope And Plan

This is a local-repo research bundle, not an Acquia internal investigation.

What was pulled:

- current repo README and 2026 year README
- official 2026 challenge metrics from the 2026-05-04 official site pull
- public history rollup for Scott's 2024/2025 results and target drift
- 2025 retrospective/model baseline docs
- 2026 outing menu and execution simulation summaries
- 2026-05-05 Harrison Hollow field-test summaries
- reproducible git-history snapshots for the 2025 burst and 2026 restart
- the current Codex thread transcript as bundle provenance

Stopping condition:

- The bundle contains enough evidence for another LLM to write or critique the
  article without re-reading the whole repo or accessing private files.

As-of date:

- 2026-05-05.
- Official challenge facts are from the 2026-05-04 local pull and may change if
  the Boise Trails Challenge site updates the route list before or during the
  event.

Privacy boundary:

- No credentials were read or bundled.
- Raw Strava activity JSON, raw GPS polyline, Strava activity ID, exact home
  origin, private planner state, and private route maps are excluded.
- The bundle uses sanitized/example route outputs and public field-test
  summaries.

## Connector Decision Matrix

| Source | Use? (Y/N) | Why it's relevant to the question | What was pulled |
|---|---|---|---|
| Salesforce (Case) | N | No customer case or Acquia account evidence is involved. | None |
| Slack | N | The article question is about local repo history and generated planner artifacts, not Slack discussion. | None |
| Jira | N | No Jira ticket is needed to support this personal/article narrative. | None |
| Confluence | N | No internal Acquia docs are needed. | None |
| Drive | N | No Drive docs are needed. | None |
| Domo | N | No Domo metrics are involved. | None |
| Sumo Logic | N | No traffic, abuse, trace, performance, or ACN/log-analysis trigger applies. | None |
| docs.acquia.com | N | Not an Acquia product documentation question. | None |
| Zoom | N | No meeting transcript/source is involved. | None |

## Key Evidence Threads

1. The current README frames the planner as a logistics problem, not just a
   line-drawing problem.
2. The 2025 retrospective shows the prior agent work had strong framing but
   unfinished implementation, schema issues, broken tests, and human-unfriendly
   route outputs.
3. The 2026 current artifacts show the planner moved to an outing menu with
   251/251 segments represented, about 280.23 on-foot miles, 23 runnable
   outings, and a 1.7x on-foot/official ratio.
4. The 2026-05-05 field test shows the new failure mode: the user stayed near
   the planned corridor but missed Who Now Loop Trail 2 because the field
   packet/map cues were not clear enough at a reused junction.
5. Git history gives a concrete starts/stops signal: 2025 has a dense burst of
   routing, clustering, one-way, test, dependency, memory, and bug-fix commits;
   2026 has a much smaller visible reset/output/field-test burst so far.

## Bundle Artifacts

Build command:

```bash
python3 /Users/scott/.agents/skills/research-bundler/scripts/bundle_builder.py \
  --manifest projects/research-20260505-boise-trails-ai-article/manifest.json \
  --output projects/research-20260505-boise-trails-ai-article/llm_bundle.json \
  --include-query-files \
  --require-conversation-context \
  --require-bundle-brief
```

