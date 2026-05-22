# Conversation Context

- captured_at: `2026-05-10T04:32:10.053645+00:00`
- cwd: `/Users/scott/dev/boise-trails-ai`
- source_session_log: `/Users/scott/.codex/sessions/2026/05/09/rollout-2026-05-09T22-25-47-019e1022-5951-7020-8dd0-f539219e1aa7.jsonl`
- message_count: `12`
- included_roles: `user`, `assistant`
- note: developer/system/tool traffic omitted; this captures the conversational thread only.

## Transcript

### 1. assistant (commentary) [2026-05-10T04:26:29.748Z]

I’ll ground this in the actual repo artifacts first, then turn it into a short recommendation rather than guessing from the project premise.

### 2. assistant (commentary) [2026-05-10T04:26:39.328Z]

Memory points to two likely veins: the older article evidence bundle about model capability changes, and the newer BTC heuristics/skills work that turns field failures into reusable eval/doctrine artifacts. I’m checking the repo now to see which one has the strongest current evidence.

### 3. assistant (commentary) [2026-05-10T04:26:49.866Z]

The current repo is dirty, so I’m treating this as read-only and avoiding any edits. The strongest signal so far is not “AI made a route”; it is the closed-loop system that converts real field misses into heuristics, eval seeds, generator constraints, and phone-tool verification.

### 4. assistant (commentary) [2026-05-10T04:27:21.125Z]

The evidence is converging on a product/eval story: a frontier lab would care about this as a messy, real-world agent benchmark where success requires route geometry, local evidence discipline, artifact consistency, human correction, and executable UI behavior. I’m checking the field-test logs and current generated-audit posture so the recommendation is anchored in more than docs.

### 5. assistant (commentary) [2026-05-10T04:28:02.585Z]

I have enough evidence for a defensible answer. I’m going to mark the goal complete after the required audit: the deliverable is a single recommendation, backed by repo artifacts, not a code change or test run.

### 6. assistant (final_answer) [2026-05-10T04:28:20.434Z]

The thing most likely to interest OpenAI/Anthropic is this:

**A real-world agent benchmark for grounded planning, where field failures become reusable heuristics, evals, skills, generator rules, and audits.**

I would not pitch it as “AI plans trail routes.” The sharper pitch is: **this project turns a messy physical-world task into a closed-loop agent-evaluation system.** It has objective ground truth, ambiguous local reality, user constraints, generated artifacts, actual field execution, and durable learning from failures.

Why it’s interesting:

- The task has hard truth: 251 official segments, 164.43 miles, ascent-only constraints, endpoint-to-endpoint credit rules.
- The agent cannot succeed with plausible text or a pretty GPX. It has to produce a car-to-car route a human can actually execute.
- Failures are high-signal: “GPX-valid is not human-valid,” “route is credit-correct but field-wrong,” “artifact drift between map/card/GPX,” “phone checkbox is not segment credit,” “hardcoded route fix is heuristic debt.”
- You now have the beginning of an eval corpus: [BTC_HEURISTICS.md](/Users/scott/dev/boise-trails-ai/docs/BTC_HEURISTICS.md:1), [BTC_FAILURE_MODES.md](/Users/scott/dev/boise-trails-ai/docs/BTC_FAILURE_MODES.md:1), [BTC_CASES.md](/Users/scott/dev/boise-trails-ai/docs/BTC_CASES.md:1), and [BTC_BEHAVIOR_EVALS.md](/Users/scott/dev/boise-trails-ai/docs/BTC_BEHAVIOR_EVALS.md:1).
- The 2026 field loop is already concrete: route planner, phone packet, live map, GPX flavors, recertification, field audits, and actual pre-challenge runs. The README shows the move from graph coverage to field-tool execution, including 30 runnable route cards, 90 GPX files, and 251/251 segment accounting in the current packet summary: [README.md](/Users/scott/dev/boise-trails-ai/README.md:62).

The strongest narrative evidence is the Harrison Hollow loop. May 5 exposed that a plausible route could still fail in the field around cue clarity and overlapping corridors. May 8 then validated the repaired card while revealing a deeper planner issue: repeated connector movement after credit/access purpose was already satisfied. That is exactly the kind of failure frontier labs care about because it is not solved by better prose alone. It requires state, geometry, local evidence, artifact consistency, and learning that generalizes: [2026-05-08 analysis](/Users/scott/dev/boise-trails-ai/years/2026/field-tests/pre-challenge/2026-05-08-test-03/analysis.md:100).

One-liner:

**“Boise Trails AI is a small but unusually rich real-world eval harness for agentic planning: can an AI maintain truth across official rules, maps, generated UI, private constraints, field execution, and post-run repair without hallucinating shortcuts or overfitting one route?”**

That’s the frontier-lab-interesting part. Not the trail planner itself; the grounded agent learning loop around it.

Goal audit completed; final goal accounting from Codex: 102 seconds elapsed.

<oai-mem-citation>
<citation_entries>
MEMORY.md:198-201|note=[repo canon for heuristics docs eval seeds and skills]
MEMORY.md:870-872|note=[article framing and evidence bundle pointer]
MEMORY.md:189-199|note=[hybrid human loop and route promotion evidence]
</citation_entries>
<rollout_ids>
019e09eb-eaf6-7ee2-9cc7-8fd5ae3a8af2
019dfa68-b240-7453-b4e2-a7a4f68e4931
019e0d80-a444-7af3-b5a3-f5f11396d233
</rollout_ids>
</oai-mem-citation>

### 7. user [2026-05-10T04:29:47.157Z]

ok is there overlap with this?

• Best Candidate

  The strongest arXiv-shaped idea is:

  BTC-Bench: Evaluating LLM Agents on Field-Executable Spatial Planning Under Real-World Constraints

  Not a paper about “AI made a Boise route.” The publishable contribution is a benchmark and evaluation
  method for agents that must turn messy goals into plans that survive physical-world invariants:
  required edge coverage, directionality, legal access, artifact consistency, human-followable cues,
  progress validation, and recertification after reality changes.

  Why this has legs:

  - The repo already defines the core research object: BTC is “edge coverage under human constraints,”
    not waypoint visiting or pretty map drawing. See docs/BTC_HEURISTICS.md:20.
  - It has a real constrained dataset: 101 trails, 251 official segments, 164.43 official miles, and 23
    ascent-only segments. See years/2026/checkpoints/data-readiness.md:19.
  - It has executable evaluation surfaces: field packet certification, GPX validation, route walkthrough
    audits, progress recertification, and privacy checks. See docs/BTC_FIELD_PACKET_REQUIREMENTS.md:75
    and years/2026/checkpoints/field-tool-completion-audit-2026-05-06.md:1.
  - It has observed failure cases and seed evals, not just synthetic prompts. See docs/BTC_CASES.md:41
    and docs/BTC_BEHAVIOR_EVALS.md:1.
  - It has longitudinal field evidence showing the important distinction: a plan can be graph-valid,
    credit-correct, and still human-invalid. See README.md:127 and essays/i-am-the-one-that-loops.md:96.

  Why It’s Novel Enough

  The adjacent literature already has travel and route-planning benchmarks: TravelPlanner
  (https://arxiv.org/abs/2402.01622), Natural Plan (https://arxiv.org/abs/2406.04520), MobilityBench
  (https://arxiv.org/abs/2602.22638), MapTab (https://arxiv.org/abs/2602.18600), GeoAgentBench
  (https://arxiv.org/abs/2604.13888), AgentIF (https://arxiv.org/abs/2505.16944), and LocationReasoner
  (https://openreview.net/forum?id=PTTmPHS7OE). The gap this project can fill is narrower and more defen
  sible: agents that produce plans with multiple downstream artifacts, where correctness requires both f
  ormal geometric validation and human field executability.

  The paper should not claim global route optimization novelty. That would be weaker. The claim should be
  benchmark/evaluation novelty:

  1. A real-world spatial planning task where success is not text plausibility.
  2. A verifier stack that separates official credit, route geometry, access legality, field cue quality,
     and artifact consistency.
  3. A failure taxonomy showing where LLM agents break: point-vs-edge reasoning, GPX-valid vs human-
     valid, stale evidence scope, hidden connector/repeat errors, and promotion before certification.
  4. Baseline experiments comparing prompt-only agents, doctrine/instruction-augmented agents, and
     verifier-gated agents.

  Minimum Preprint Shape

  Title: BTC-Bench: Field-Executable Route Planning as a Grounded Evaluation for Language Agents

  Core experiment:

  - Give agents the public official segment set and planning constraints.
  - Ask them to produce or repair route artifacts.
  - Score with deterministic checks: segment coverage, ascent direction, car-to-car continuity, access
    assumptions, cue completeness, artifact consistency, recertification behavior.
  - Add model comparisons across 4-6 agents or model setups.
  - Include two field-backed case studies: May 5 failure and May 8 repair.

  Main risk: single-domain, single-user. Mitigate by framing it as a benchmark prototype plus
  reproducible verifier methodology, not as a universal outdoor-routing dataset. Expand with synthetic
  perturbations over the official graph and sanitized failure cases.

  I did not change files or run tests; this was read-only repo analysis plus a current literature scan.
  Goal completed in 147 seconds.

### 8. assistant (final_answer) [2026-05-10T04:30:02.273Z]

Yes. It is essentially the same core idea, just pushed into a more formal **paper/benchmark** frame.

The overlap is high:

- My framing: “real-world agent benchmark for grounded planning.”
- This framing: “BTC-Bench: Evaluating LLM Agents on Field-Executable Spatial Planning.”
- Both reject “AI made a route” as the interesting claim.
- Both center the same hard distinction: graph-valid / GPX-valid / credit-correct can still be human-invalid.
- Both use the same evidence spine: BTC heuristics, field packet audits, behavior evals, field-test failures, and recertification.
- Both identify the frontier-lab hook as **verifiable agent planning under real-world constraints**, not route optimization novelty.

The pasted version is stronger if the goal is **arXiv/preprint positioning**. It adds the missing academic pieces: related-work gap, benchmark title, baseline experiment design, novelty boundary, and single-domain-risk mitigation.

The piece I would keep from my answer, because it is slightly sharper for OpenAI/Anthropic specifically, is this angle:

> The most interesting thing is not only BTC-Bench as a dataset, but the closed loop where field failures become heuristics, eval seeds, skills, generator constraints, audits, and recertification rules.

That is the frontier-lab signal: it is a compact example of how agents can learn durable operational rules from real-world misses instead of just producing a better next response.

So I’d merge them:

**Primary frame:** BTC-Bench, a benchmark for field-executable spatial planning agents.

**Differentiator:** a verifier-gated, longitudinal field loop where real failures are converted into reusable agent doctrine, evals, and artifact-generation constraints.

**Avoid claiming:** global routing novelty, outdoor-navigation product novelty, or generality beyond a benchmark prototype.

One small caveat: before using the pasted literature list in a paper or pitch, I’d verify each citation/current claim. But conceptually, yes, it overlaps almost exactly with the recommendation.

### 9. user [2026-05-10T04:31:21.610Z]

ok see this

• Yes. It is essentially the same candidate, with a slightly different audience framing.

  Your pasted version is the frontier-lab pitch: “this is a high-signal grounded agent learning loop.”
  My version was the arXiv/preprint framing: “this can become a benchmark/evaluation method for field-
  executable spatial planning.”

  The overlap is strongest on:

  - Real-world grounded planning, not trail-route novelty.
  - Objective truth plus messy local reality.
  - Agent failures becoming reusable heuristics, evals, skills, generator rules, and audits.
  - Harrison Hollow as the core case study.
  - The key distinction that graph-valid / GPX-valid / credit-correct is not the same as human-
    executable.
  - The need for state, geometry, artifact consistency, verification, and post-field repair.

  The main difference is packaging. For OpenAI/Anthropic, I’d use your phrasing. For arXiv, I’d make it
  less “why labs should care” and more:

  > We introduce a grounded planning benchmark and verifier stack for evaluating whether language agents
  > can maintain consistency across official rules, spatial geometry, generated artifacts, human
  > constraints, field execution, and post-execution repair.

  One caution: I’d avoid making “skills” central in the paper title/abstract because it sounds platform-
  specific. In the paper, call them procedural policies, repair protocols, or agent-operational
  heuristics, then mention they are implemented as repo-local skills.

  Also verify line refs before reusing them. In my current read, the 30 route cards / 90 GPX / 251
  segment summary was around README.md:107, not README.md:62, so that citation may have drifted.


can oyu create a $research-bundler BUNDLE FOR THIS?

### 10. user [2026-05-10T04:31:21.610Z]

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

### 11. assistant (commentary) [2026-05-10T04:31:31.354Z]

I’ll create a focused `research-bundler` evidence pack for the BTC-Bench / frontier-lab / arXiv framing. I’ll keep it public-safe and read-only against private planner data, but I’ll verify the cited public literature from primary pages before bundling it.

### 12. assistant (commentary) [2026-05-10T04:32:01.720Z]

The worktree already has many unrelated edits and generated field-packet deletions, so I’m going to leave those untouched and add only a new `projects/research-20260510-btc-bench-frontier-labs/` bundle folder.
