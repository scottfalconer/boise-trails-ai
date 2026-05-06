# Conversation Context

- captured_at: `2026-05-05T23:16:00.504364+00:00`
- cwd: `/Users/scott/dev/boise-trails-ai`
- source_session_log: `/Users/scott/.codex/sessions/2026/05/05/rollout-2026-05-05T17-10-58-019dfa68-b240-7453-b4e2-a7a4f68e4931.jsonl`
- message_count: `10`
- included_roles: `user`, `assistant`
- note: developer/system/tool traffic omitted; this captures the conversational thread only.

## Transcript

### 1. user [2026-05-05T23:11:02.761Z]

I want to write an intro article about doing this again this year. I do this 1) because it's fun and 2) to get a good idea of how model capabilties are changing and what that means,

Each year I try to use a model to help me plan out my most efficient route for running the Boise Trails Challenge - ~170 miles of trails in 30 days. It's a pretty complex problem of determining the best route that gets it done in the least amount of miles. I also want it to work because my "testing" involves actually running it.

Last year any available models had promising starts but would get lost and confused with all the connections, reality of getting back to my car etc, so never outperformed a manual map.

5.5 is the first one that on paper seems to have made a usable route.(the readme from the repo covers some of this well)

i..e last year I spent a lot of time just getting the route to work. I learned a lota about routing, how this is a "snowplow" type problem etc, but at this point last year the models really weren't there. I started a week or so before the challnege and then had to abandon it ones the challenge started becuase one it starts I have time for 1) work 2) kids ) running. That's about it. 

you can look through the git history for some of the starts, stops etc.

Now we're back a year later and we're using 5.5 + codex CLI. SO far it's done of great job of making a plausable map on paper. But this is where it starts to get interesting. One of the questions is what does it mean for us when AI can do all this stuff, well this is an interesting case of "there will be more work"

Last year i soent a lot of time just trying to get the map to work, but it was never quite there

This year, I had a plausable map in the first day, but this is where it gets interesting. I now had time to pressure test it. This year instead of one sort of route, i split it into outings (add the part about kids and timing here) but here's what's interesting. Within a day the value I've receivec is way up, I have something usable - BUT now I'm using it, the stuff that needs to be done is also way up, it's just further on the vlaue chain. i.e. I went out and did a segment today, but following the route was confusing, i missed a turn and blew up my optimization. So now here I am, shifting up the value chain working more

### 2. assistant (commentary) [2026-05-05T23:11:18.779Z]

I’ll use the `inside-scott-voice-skill` for the draft voice, then ground the intro in the repo README and git history so the piece reflects what actually happened rather than just the outline above.

### 3. assistant (commentary) [2026-05-05T23:11:29.355Z]

The repo history already shows a useful shape: a burst of 2025 fixes around clustering, one-way pathfinding, dependency/test issues, and map labeling, followed by a 2026 reset where the old planner was archived and the new output/menu/field-test flow appeared. I’m reading the README and the 2025/2026 retrospective notes next so the draft can use concrete evidence.

### 4. assistant (commentary) [2026-05-05T23:11:58.551Z]

The 2025 history is almost comically on-theme: initial commit on 2025-06-09, then a rapid run of planner/GPX/trailhead/RPP/cluster fixes through 2025-06-21, including repeated “get tests passing,” “fix route planning,” “one-way,” “memory,” and “unplanned segments” commits. That gives the intro a good concrete contrast with 2026: not “AI was useless,” but “most of the work was still getting to a map that survived contact with the data.”

### 5. assistant (final_answer) [2026-05-05T23:12:55.005Z]

I grounded this in the repo README, 2025 retrospective notes, current outing menu, field-test writeup, and git history. Here’s a first-pass intro draft:

**Working Title: AI Finally Made Me More Work**

Every year I try to use the current best AI model to plan my route for the Boise Trails Challenge.

This is partly because it is fun, which is a good enough reason to do most things. But it is also a useful benchmark for model capability because the test is not “can the model sound smart?” The test is “can it produce a route I am willing to go run?”

The challenge is simple on paper: complete roughly 170 miles of official Boise foothills trails in 30 days.

The actual problem is much messier:

- cover every official trail segment
- avoid wasting miles
- respect ascent-only segments
- use connector trails and roads without counting them as progress
- get back to the car
- deal with heat, closures, mud, signage, water, and trailhead logistics
- fit the whole thing around work and kids

Last year, the models had promising starts, but they never really got there. They could identify parts of the formal problem. We got into rural postman / capacitated routing / snowplow-style territory. But the plans kept breaking in the human parts: weird connections, too many tiny errands, unclear return-to-car logic, stale data, and routes that looked mathematically plausible but not like something a person would actually do.

The git history tells the story. The repo started on June 9, 2025, and by June 21 there were hundreds of commits around route planning, clustering, one-way segments, dependencies, test failures, memory issues, and unplanned segments. Most of the work was still getting to a map that survived contact with the data.

Then the challenge started, and I had to stop. Once the month begins, I have time for three things:

1. work
2. kids
3. running

That is the full list. There is no extra block for “debug why the planner thinks I should teleport back to my car.”

This year is different.

A year later, using GPT-5.5 with Codex CLI, I had a plausible route on paper in the first day. Not perfect. Not field-ready. But plausible enough to use.

That is where the interesting part starts.

The obvious story would be “better AI means less work.” In this case, that is only half true. The value showed up much earlier. Last year I was mostly fighting to get a route that worked at all. This year I had something useful fast enough that I could pressure-test it.

And once I used it, the work increased. It just moved up the value chain.

I went out for a pre-challenge field test on May 5. The planned outing was Harrison Hollow: 1h 36m door-to-door, 4.72 official miles, 5.69 total on-foot miles, 12 official segments.

I stayed basically inside the planned route corridor and still missed a turn. The failure was not “the model sent me to the wrong mountain.” It was more subtle and more useful: the route reused nearby trail corridors around Who Now, Kemper’s Ridge, and Hippie Shake, and the field packet did not make the next turn obvious enough.

Last year I was debugging whether the planner could connect trails at all.

This year I am debugging signpost language like: “After Kemper’s Ridge, take #50 Hippie Shake. Do not drop onto #51 Who Now.”

That is a completely different problem.

And I think that is the more interesting AI lesson. When models get better, they do not just remove work. Sometimes they create more work because they produce something valuable enough to expose the next bottleneck.

For this project, the new work is not only route optimization. It is turning a graph solution into something that survives real life:

- a phone-friendly field packet
- turn-by-turn cues that match Ridge to Rivers signs
- official segment credit separated from actual traversal order
- GPX tracks that start and end at the car
- progress updates after partial or failed attempts
- route choices that respect hard stops, heat, conditions, and family logistics

That is much better work than “try another clustering heuristic because this one stranded me on the wrong side of town.”

So this year’s experiment is not “can AI perfectly plan the Boise Trails Challenge?” It will not. The interesting question is how far up the stack the model can move before reality takes over.

Last year, reality took over before I had a route.

This year, reality took over after I had a route and tried to run it.

That is a big difference.

Also annoying, because now I have to go back and fix Who Now.

### 6. user [2026-05-05T23:14:24.960Z]

can you actually just pull the data I need for this article into a [$research-bundler](/Users/scott/.agents/skills/research-bundler/SKILL.md) bundle?

### 7. user [2026-05-05T23:14:24.961Z]

<skill>
<name>research-bundler</name>
<path>/Users/scott/.agents/skills/research-bundler/SKILL.md</path>
---
name: research-bundler
description: Deep research workflow. Uses Acquia Grabber connectors, authoritative public sources, and local indexes/mirrors to answer a question and produce a single LLM-ready JSON bundle (evidence pack) for offline analysis.
---

# Purpose

Use this skill when a user asks a **research question** that should be answered using internal sources (Salesforce, Jira/Confluence, Slack, Drive, Forethought, Domo, Sumo, Zoom, docs.acquia.com, and local exports under `data/`) and/or authoritative public sources plus local mirrors (for example official issue queues, standards/docs sites, or local code scans under `scans/`).

This skill has two deliverables:
1. **Best-effort answer** to the question (with clear evidence).
2. A **shareable “bundle” JSON** that contains all relevant artifacts/evidence so another LLM (with no connector access) can analyze it offline.

---

# Non‑negotiables (Security + Hygiene)

- **No secrets (CRITICAL)**: never bundle or share files containing active credentials (API keys, passwords, salts, tokens). Never copy anything from `credentials/` into outputs.
- **PII caution**: the default bundle scrubs **emails/phones** in *text* fields only; JSON artifacts may still contain PII. Treat bundles as **Sensitive / Internal Use Only** unless you explicitly sanitize/prune them for third‑party use.
- **PII minimization (your prose)**: avoid customer emails/phones/IDs in your written answer unless essential. Prefer roles (“customer contact”, “Support engineer”).
- **Data placement**: connector outputs go under `data/<source>/...` (treat `data/` as a local cache). Tracked work goes under `projects/<project>/...`.
- **Reproducibility**: every artifact added to the bundle should have its **source**, **path**, and (if applicable) the **command/query** used to obtain it.
- **Hard stop on missing access (when needed)**: if a connector/source is required to answer the question and you cannot access it due to missing auth/permissions, **stop and ask the human for help** (credentials, SID/SSO, or a provided export) rather than writing a generic answer.

---

# When to use

Use this skill if the question requires one or more of:
- cross-system lookup (case ↔ Jira ↔ Slack ↔ Confluence ↔ Drive)
- historical context (past cases, prior incidents, prior changes)
- validating claims (“did we ever…”, “how often…”, “what’s the latest…”)
- building a **portable evidence pack** for audit / retrospective / leadership review
- public-source research that still needs a reproducible evidence pack (for example issue queues, official docs, or local code mirrors)

Do **not** use this skill for:
- pure writing requests that don’t need evidence
- questions that can be answered directly from a provided case JSON or report

---

# Inputs

At minimum:
- The **research question** (one or more sentences)

Optional but extremely helpful:
- Case number(s) (e.g., `01234567`)
- Jira key(s) (e.g., `DI-1234`, `OP-4567`)
- Account name or Salesforce Account ID
- Time window (e.g., “last 90 days”)

If the user provides none of the optional context, proceed anyway using keyword search + inference, and only ask a clarifying question if the search space is unbounded.

---

# Workflow

## Step 0 — Create a research workspace

1. Pick a project id: `projects/research-<YYYYMMDD>-<slug>/`
2. Create:
   - `projects/<id>/README.md` (notes + what you pulled)
   - `projects/<id>/manifest.json` (what sources/commands produced which artifacts)
   - `projects/<id>/start-here.md` (human-readable handoff for the next LLM)
   - `projects/<id>/bundle_assembly_transcript.md` (current Codex thread provenance)

Use `examples/manifest.json` and `examples/start-here.md` as starting points.

## Step 0a — Capture the current conversation provenance (MANDATORY)

Before you pull external evidence, capture the current Codex thread into a bundle-ready provenance artifact.

Preferred path:

```bash
python3 scripts/export_codex_conversation.py \
  --output projects/<id>/bundle_assembly_transcript.md \
  --json-output projects/<id>/bundle_assembly_transcript.json
```

Notes:
- The exporter reads local Codex session logs from `~/.codex/sessions/` and auto-detects the latest session matching the current `cwd`.
- If auto-detection finds the wrong thread, rerun with `--session-log /Users/<you>/.codex/sessions/.../rollout-....jsonl`.
- The markdown transcript is the default artifact to include in the manifest. The JSON output is optional but useful for machine-readable post-processing.
- The exporter strips the standard AGENTS/environment wrapper from the first user message when possible so the transcript focuses on the actual user/assistant exchange.
- This artifact is **bundle provenance**, not the incident or subject conversation unless the current Codex thread itself is the subject under investigation.
- If the actual incident conversation is important evidence, include it as a separate artifact such as `subject-conversation-evidence`.

Fallback if session logs are unavailable:
- Create `projects/<id>/bundle_assembly_transcript.md` manually with the user question, clarifications, key assumptions, and the reason the bundle is being created.
- Still include that file in the manifest as `type: "bundle_assembly_transcript"` (or `conversation_context` for backward compatibility).

## Step 0b — Draft the handoff metadata (MANDATORY)

Before collecting evidence, write the bundle’s orientation layer:

1. Add a top-level `bundle_brief` object in `projects/<id>/manifest.json`.
2. Add a `start-here` artifact that points at `projects/<id>/start-here.md`.
3. Tag every manifest item with:
   - `artifact_role`
   - `artifact_priority`

Minimum required `bundle_brief` fields:
- `question_for_llm`
- `handoff_intent`
- `entrypoint_artifact_id`
- `read_order`

Strongly recommended `bundle_brief` fields:
- `bundle_type`
- `expected_output`
- `scope`
- `primary_issue`
- `status`
- `subject_conversation_id`
- `bundle_creation_conversation_id`
- `known_good_conclusions`
- `open_questions`
- `artifact_roles`

Preferred `handoff_intent` values:
- `continue_investigation`
- `propose_fix`
- `audit_bundle_quality`
- `summarize_for_leadership`

`start-here.md` should answer six things in plain English:
- what this bundle is
- what question the next model should answer
- what not to spend time on
- which artifacts are read first
- which artifacts are raw/archive-only
- what is already believed vs what still needs judgment

Preferred naming:
- use `bundle-assembly-transcript` for the current Codex thread / provenance artifact
- use `subject-conversation-evidence` for the actual incident or user-facing conversation being analyzed
- keep `conversation-context` only as a backward-compatible alias when you are touching older bundles

## Step 1 — Plan and scope

Extract from the question:
- entities: case numbers, account, application UUIDs, environment names, domains
- timeframe
- “must-have evidence” vs “nice-to-have”

Write a short plan in `projects/<id>/README.md`:
- what you will search first
- what connectors/scripts you may run
- stopping conditions (what counts as “answered”)
- note that the current conversation provenance artifact was captured and where it lives
- record the intended `handoff_intent`
- record `subject_conversation_id` versus `bundle_creation_conversation_id` if both exist

If the question depends on public systems whose titles/statuses/docs can change, record the exact **as-of date** you are researching against and note which facts are temporally unstable.

## Step 1a — Connector Decision Matrix (MUST COMPLETE)

Before collecting evidence, fill this out in `projects/<id>/README.md`:

| Source | Use? (Y/N) | Why it’s relevant to the question | What you will pull (specific) |
|---|---|---|---|
| Salesforce (Case) | | | |
| Slack | | | |
| Jira | | | |
| Confluence | | | |
| Drive | | | |
| Domo | | | |
| **Sumo Logic** | | | |
| docs.acquia.com | | | |
| Zoom | | | |

Rules:
- If **Use? = N**, you must write a concrete reason (e.g., “No time window + no identifiers; would be blind scan”).
- If the question involves **traffic**, **abuse**, **bot/scraping**, **4xx/5xx spikes**, **performance/latency**, **WAF/Cloudflare**, **request IDs/trace IDs**, or **attribution** (“which IPs/paths/UAs?”) → **Sumo Logic must be Y** (see Step 2a).
- For **Cloud Next (ACN)** investigations: start with dashboards/metrics (**Sumo Logic**) before raw logs when possible.

## Step 2 — Collect evidence (prefer fast paths)

Prefer this order:

1. **Local indexes (fast)**
   - Slack FTS: `data/slack_index.db`
   - Local file index: `data/local_file_index.db`
   - Existing unified cases under `data/cases/` and cached exports under `data/<source>/...`
   - Local code mirrors / scans under `scans/` when the question is about product/code terminology or implementation drift

2. **Case-centric pull (if a case is involved)**
   - `python3 scripts/build_unified_case.py --case <CASE>`

3. **Targeted connector pulls (only what you need)**
   - Jira: search by case number/account/env/app UUID; export matched issues
   - Confluence: CQL search and export matched pages (plus attachments only if relevant)
   - Drive: search by case number + keywords; download only matched docs
   - Domo: export only the relevant dataset slices/queries
   - **Sumo (REQUIRED when triggered; see Step 2a)**: use the `sumologic-investigator` skill to export the **Minimum Sumo Evidence Pack** (Step 2b) for the smallest time window that answers the question
   - Zoom: fetch only relevant transcripts/summaries
   - docs.acquia.com: fetch only relevant nodes/pages
   - Public issue queues / public docs / official sites: use the official API/CLI/read-only interface where available and save the exact snapshot under `data/<source>/...`

For each pull, record in the manifest:
- connector/script command
- query terms
- output path(s)
- any exact URL / issue id / page id / “as of” timestamp needed to interpret the result later
- `artifact_role` (`orientation`, `primary_evidence`, `raw_evidence`, `derived_summary`, `provenance_only`, or another precise label)
- `artifact_priority` (lower numbers should be read earlier)

The provenance artifact and `start-here` artifact are not optional. They should appear in the manifest, typically as:

```json
{
  "id": "start-here",
  "connector": "local",
  "type": "orientation",
  "src": "projects/<id>/start-here.md",
  "artifact_role": "orientation",
  "artifact_priority": 1,
  "notes": "Human-readable bundle handoff for the next model."
},
{
  "id": "bundle-assembly-transcript",
  "connector": "local",
  "type": "bundle_assembly_transcript",
  "src": "projects/<id>/bundle_assembly_transcript.md",
  "command": "python3 scripts/export_codex_conversation.py --output projects/<id>/bundle_assembly_transcript.md --json-output projects/<id>/bundle_assembly_transcript.json",
  "artifact_role": "provenance_only",
  "artifact_priority": 900,
  "notes": "Current Codex thread provenance captured before evidence collection."
}
```

Backward compatibility note:
- `conversation-context` / `conversation_context` still work with the builder, but prefer `bundle-assembly-transcript` / `bundle_assembly_transcript` for new bundles so the provenance purpose is obvious.

If you do a follow-up pass after an earlier bundle, prefer adding a **supplemental** query/index file (for example `phase2_issue_ids.txt`) instead of silently mutating the original selection file, unless the original file was actually wrong. This preserves reproducibility between passes.

### Step 2a — Sumo Logic triggers (MANDATORY when triggered)

If **ANY** of the following are true, you **MUST attempt a Sumo pull** and include it in the manifest + bundle:

- The case includes **Sumo links** (e.g., `service.sumologic.com/...`) or explicitly references Sumo searches/dashboards.
- Question mentions: malicious traffic, bot traffic, abuse, scraping, WAF, Cloudflare, rate-limiting, DDoS, blocking IPs, suspicious requests, 4xx/5xx spikes, timeouts, slow site, “traffic spike”.
- Case artifacts contain request IDs / trace IDs (e.g., `request_id=...`) or explicitly reference Sumo (“I see following error on Sumo Logic…”).
- The product/case is **Cloud Next (ACN)** OR there are AutoPanic/Drutiny reports that include Sumo dashboard links.
- You need attribution (“which IPs/paths/UAs are causing this?”).

If the Sumo pull fails (no credentials, network restriction, query error), you must still record the attempt and include the exact query templates you would have run (see Step 2e).

### Step 2b — Minimum Sumo Evidence Pack (default exports)

Unless clearly irrelevant to the question, include these three categories of exports (often 3–6 `sumo_search.py` runs):

1) **Traffic summary (time-sliced)**: requests/min + 4xx/5xx rate + top URLs  
2) **Top sources**: top IPs (or /24s), top User Agents, top countries (if available)  
3) **Drilldown sample**: 20–50 representative raw log lines for 1–2 suspicious patterns  
   - Scrub tokens/cookies/Authorization headers if present.

Recommended output placement:
- Case-focused: `data/sumologic/search/case-<CASE>/...`
- Project-focused: `data/sumologic/search/<id>/...`

### Step 2c — Canonical Sumo export command (via `sumologic-investigator`)

Use the shared, dependency-free exporter (`scripts/sumo_search.py`) from the `sumologic-investigator` skill root (adjust the path if your working directory differs):

```bash
python3 scripts/sumo_search.py \
  --query 'request_id="v-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"' \
  --minutes 10 \
  --out data/sumologic/search/case-01234567/request_id_v.json
```

Notes:
- Prefer the **smallest time window** that answers the question.
- If the path looks expensive (especially InFrequent tier), give a heads up and confirm before proceeding (see the `sumologic-investigator` skill's `SKILL.md` cost section). `sumo_search.py` will print `COST HEADS-UP` and refuse to run unless you pass `--ack-cost`.
- Credentials are loaded from either:
  - environment variables `SUMO_ACCESS_ID` + `SUMO_ACCESS_KEY` (optional: `SUMO_API_BASE_URL`), or
  - `credentials/sumologic/credentials.json` (default)

To make failed exports reproducible, store your query in a file and reference it from the manifest:

```bash
cat > projects/<id>/queries/sumologic/request_id.sumoql <<'Q'
request_id="v-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
Q

python3 scripts/sumo_search.py \
  --query "$(cat projects/<id>/queries/sumologic/request_id.sumoql)" \
  --minutes 10 \
  --out data/sumologic/search/case-01234567/request_id_v.json
```

### Step 2d — AutoPanic/Drutiny bootstrap (Sumo-first)

If an AutoPanic/Drutiny HTML report exists for the environment, parse it first (to extract dashboard links + IDs) using the `drutiny-report-parser` skill's `scripts/parser.py` (adjust the path if your working directory differs):

```bash
python3 scripts/parser.py /path/to/report.html --outdir projects/<id>/parsed --pretty
```

Then use the parsed JSON to extract environment/app identifiers (and any Sumo dashboard links), and run a small bootstrap query via `sumologic-investigator`:

```bash
python3 scripts/sumo_search.py \
  --query "\"<env_name_or_app_uuid_or_subscription_uuid>\" (error OR exception OR \" 5\")" \
  --minutes 15 \
  --out data/sumologic/search/<id>/autopanic_bootstrap_errors.json
```

### Step 2e — When Sumo is blocked (still required)

If Sumo is triggered but cannot be exported (credentials missing, network restricted, query errors), you must:
- save the exact query string(s) you would have run under `projects/<id>/queries/sumologic/*.sumoql`, and
- produce `projects/<id>/sumologic_attempt.json` capturing the intended time window + outputs + error/reason, and
- include that artifact in `projects/<id>/manifest.json` so it lands in the portable bundle.

If the missing export blocks a must-have conclusion, **stop and ask the human** to provide Sumo access or run the export and share the resulting JSON, rather than “continuing anyway”.

## Step 3 — Build the portable bundle JSON

Use the bundled script in this skill to inline artifacts referenced by the manifest (default output matches acquia-grabber’s `scripts/build_llm_bundle.py` schema):

```bash
python3 scripts/bundle_builder.py \
  --manifest projects/<id>/manifest.json \
  --output projects/<id>/llm_bundle.json \
  --include-query-files \
  --require-conversation-context \
  --require-bundle-brief
```

Notes:
- Large text files are truncated via `--max-bytes` and `--max-chars`; JSON is inlined as structured JSON when possible.
- To disable basic email/phone scrubbing for text: add `--no-scrub-basic`.
- The bundler carries `bundle_brief` through to the final JSON and stamps each artifact with `artifact_role` and `artifact_priority`.
- By default, artifacts must live under the bundle repo root. If you intentionally need sibling-repo evidence, add top-level `allowed_roots` in the manifest or pass `--allow-root /abs/or/relative/path` to the builder. `credentials/` paths are still blocked even when roots are allowlisted.

## Step 3a — Validate bundle coverage (REQUIRED)

After building the bundle, verify that every manifest item actually landed in the bundle.

Recommended check:

```bash
python3 - <<'PY'
import json
manifest = json.load(open('projects/<id>/manifest.json'))
bundle = json.load(open('projects/<id>/llm_bundle.json'))
item_ids = {item['id'] for item in manifest['items']}
artifact_ids = {artifact.get('id') for artifact in bundle.get('artifacts', [])}
missing = sorted(item_ids - artifact_ids)
brief = bundle.get('bundle_brief') if isinstance(bundle.get('bundle_brief'), dict) else {}
entrypoint = brief.get('entrypoint_artifact_id')
missing_roles = [
    artifact.get('id')
    for artifact in bundle.get('artifacts', [])
    if not artifact.get('artifact_role')
]
print('manifest_items', len(item_ids))
print('bundle_artifacts', len(artifact_ids))
print('missing', len(missing))
for mid in missing:
    print('MISSING', mid)
has_context = any(
    artifact.get('id') in {'conversation-context', 'bundle-assembly-transcript'}
    or artifact.get('type') in {'conversation_context', 'bundle_assembly_transcript'}
    for artifact in bundle.get('artifacts', [])
)
print('has_bundle_brief', bool(brief))
print('entrypoint_artifact_id', entrypoint)
print('entrypoint_in_bundle', entrypoint in artifact_ids if entrypoint else False)
print('missing_roles', len(missing_roles))
print('has_conversation_context', has_context)
PY
```

Rules:
- Do **not** assume bundle completeness from artifact counts alone.
- When `--include-query-files` is used, the bundle may contain extra artifacts like `<id>__query_file`, so `bundle_artifacts` can be greater than `manifest_items`.
- If any manifest items are missing from the bundle, fix the manifest or artifact paths and rebuild before you hand off the bundle.
- If `has_bundle_brief` is `False`, the bundle is incomplete for this workflow. Add the handoff metadata and rebuild.
- If `entrypoint_in_bundle` is `False`, the bundle’s read order is broken. Fix `bundle_brief.entrypoint_artifact_id` and rebuild.
- If `missing_roles` is non-zero, fix the manifest roles or let the bundler derive them, then rebuild and re-check.
- If `has_conversation_context` is `False`, the bundle is incomplete for this workflow. Add the provenance artifact and rebuild.

## Step 4 — Answer + attach the bundle

Write your response with:

1. **Answer** (grounded; include what you’re confident about)
2. **Evidence** (what artifacts support the answer)
3. **Gaps / open questions** (what you could not confirm)
4. **Bundle** (path to the generated JSON)

---

# Output format (recommended)

## Research Answer
- **Question**: …
- **Conclusion**: …
- **Confidence**: High | Medium | Low

## Evidence summary
- Bullet list of the top artifacts you relied on (with where they are in the bundle)

## Bundle
- `projects/<id>/llm_bundle.json`

---

# Practical tips

- Keep bundles **small and purposeful**: prefer targeted exports over whole-space dumps.
- Treat `bundle-assembly-transcript` as provenance by default. Do not let it masquerade as the subject conversation evidence.
- If the question is about “how we work”, pull the **exact** runbook/policy page(s) and include them, rather than relying on memory.
- For debates or conflicting sources, include **both sides** in the bundle and label the conflict.
- For terminology / naming / architecture-drift questions, include at least one machine-readable matrix in `projects/<id>/findings/` in addition to prose. A good default shape is `term -> canonical definition -> alias/avoid notes -> source issues/docs -> code/UI hotspots -> owner/status`.
- For public issue/docs research, capture exact current titles/statuses/dates in your findings rather than assuming the user’s label is still current.

</skill>

### 8. assistant (commentary) [2026-05-05T23:14:40.920Z]

I’ll build this as a real portable research bundle under `projects/`, with the current thread captured as provenance, then bundle the repo evidence needed for the article: 2025 baseline, 2026 current plan, field-test evidence, official metrics, and git-history snapshots.

### 9. assistant (commentary) [2026-05-05T23:14:49.684Z]

There are existing unrelated working-tree changes, so I’ll leave them alone and only add a new research project directory. The research-bundler scripts live in the skill folder rather than this repo, so I’ll call them by absolute path.

### 10. assistant (commentary) [2026-05-05T23:15:52.342Z]

I’ve got the evidence scope now. I’m going to keep the bundle public/shareable in repo terms: no private state files, no raw Strava pulls, no credentials, and no private map outputs. The bundle will use sanitized outputs plus the public field-test summary.
