# Related Work Matrix

As-of date: 2026-05-10.

This is a lightweight positioning matrix for the BTC-Bench idea. It is not a full literature review. The purpose is to show where the adjacent benchmark space already exists and where this repo's contribution is narrower.

| Work | Source snapshot | What it evaluates | Relation to BTC-Bench | Gap BTC-Bench can occupy |
|---|---|---|---|---|
| TravelPlanner | `data/literature-snapshots/arxiv-2402.01622-travelplanner.html` | Real-world travel planning with language agents, tools, data records, planning intents, and reference plans. | Strong adjacent agent-planning benchmark. | BTC-Bench is not general travel planning; it evaluates whether spatial plans survive formal edge coverage, artifact generation, car-to-car execution, and post-field repair. |
| NATURAL PLAN | `data/literature-snapshots/arxiv-2406.04520-natural-plan.html` | Natural-language planning across trip, meeting, and calendar tasks with full context supplied to models. | Adjacent planning benchmark with structured constraints. | BTC-Bench adds geometric route validation, directionality, downstream artifacts, and field-executable cue requirements. |
| MobilityBench | `data/literature-snapshots/arxiv-2602.22638-mobilitybench.html` | LLM route-planning agents in real-world mobility scenarios with deterministic API replay. | Most direct route-planning benchmark adjacency. | BTC-Bench focuses less on point-to-point mobility and more on required edge coverage, legal access, physical route continuity, and human field execution. |
| MapTab | `data/literature-snapshots/arxiv-2602.18600-maptab.html` | Multimodal constrained route planning over map images and tabular route attributes. | Adjacent constrained route reasoning benchmark. | BTC-Bench is not primarily visual/table QA; it requires producing consistent, executable artifacts and validating them against official geometry and field constraints. |
| GeoAgentBench | `data/literature-snapshots/arxiv-2604.13888-geoagentbench.html` | Tool-augmented GIS agents with dynamic execution, spatial tools, parameter accuracy, and VLM verification. | Strong adjacent dynamic geospatial-agent benchmark. | BTC-Bench can contribute a smaller but field-backed verifier stack around route artifacts, cue quality, access reality, and recertification after state changes. |
| AgentIF | `data/literature-snapshots/arxiv-2505.16944-agentif.html` | Instruction following in realistic agentic scenarios with long, complex constraints. | Adjacent agent-instruction-following benchmark. | BTC-Bench turns long project doctrine into operational spatial-planning behavior and deterministic artifact checks. |
| LocationReasoner | `data/literature-snapshots/arxiv-2506.13841-locationreasoner.html`, `data/literature-snapshots/openreview-PTTmPHS7OE-locationreasoner.html` | Real-world site selection under spatial, environmental, and logistical constraints with automated verification. | Strong adjacent spatial reasoning benchmark. | BTC-Bench differs by requiring generated route artifacts, field cue usability, official segment-credit validation, and post-execution repair. |

## Positioning Claim

The defensible novelty claim is not global route optimization. It is benchmark/evaluation methodology:

1. A real-world spatial planning task where success is not text plausibility.
2. A verifier stack that separates official credit, route geometry, access assumptions, field cue quality, artifact consistency, and recertification behavior.
3. A failure taxonomy grounded in observed planner and field-test behavior.
4. Baseline comparisons between prompt-only agents, doctrine/instruction-augmented agents, and verifier-gated agents.

## Citation Caution

The snapshots verify the paper titles and abstracts as of 2026-05-10. A final preprint should still run a proper bibliography pass for versions, author order, venues, and citation formatting.
