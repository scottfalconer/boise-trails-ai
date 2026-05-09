---
id: "boise-trails-challenge-ai-route-planner"
path: "/essays/boise-trails-challenge-ai-route-planner"
title: "Boise Trails Challenge AI Route Planner"
dek: "The Boise Trails Challenge starts today. It's a month-long challenge to complete ~170 miles of Boise’s trails in a month."
date: "2025-06-19"
topic: "AI Projects"
series: "Boise Trails"
read: "2 min"
status: "published"
sourceUrl: "https://www.linkedin.com/pulse/boise-trails-challenge-ai-route-planner-scott-falconer-dar5c"
image: "https://www.managing-ai.com/images/essays/boise-trails-challenge-ai-route-planner-hero.jpg"
imageAlt: "Boise Trails Challenge AI Route Planner"
aliases:
  - "/resources/boise-trails-challenge-ai-route-planner"
  - "/resources/boise-trails-challenge-ai-route-planner.html"
---

> Mirrored from [Managing AI: Boise Trails Challenge AI Route Planner](https://www.managing-ai.com/essays/boise-trails-challenge-ai-route-planner).

The [Boise Trails Challenge](https://boisetrailschallenge.com/) starts today. It's a month-long challenge to complete ~170 miles of Boise’s trails in a month.

Last year I completed ~75% of the challenge but ran more miles than required because I’m bad at planning.

This year, I'm going to use AI to see if it can help me optimize my route and reduce duplicate miles, stranded segments, and general inefficiencies.

I’m testing two AI methods to achieve 100% with minimal distance:

1.  Systematic Planner: A Python script developed with OpenAI Codex. The code is available here: [**github.com/scottfalconer/boise-trails-ai**](http://github.com/scottfalconer/boise-trails-ai)

2.  Iterative Planner: Daily planning sessions using AI models directly. I’ll be tracking the results to see which approach is more effective.

Part of my goal is to see what performs best: A classic application, or a frontier model.

First impressions: The app took a long time to write and takes a long time to run, but feels generally more accurate / optimized. Asking the model (o3-pro) took a few minutes to craft a response, and returns answers quickly, but seems to be getting confused on previous years routes / unrelated trails, or hyper-focused on irrelevant things (for my use case) like if dogs are allowed on the trail.

My prediction for the end: A hybrid approach will be best where a classic script preps the data for the context and the LLM evaluates it.

I'll update this post with a running log as I progress.

**Day 1 update**:

![](https://www.managing-ai.com/images/essays/boise-trails-challenge-ai-route-planner-inline-1.jpg)

**Classic script:** It was still running when I left, so I didn't use it ¯\\(ツ)\_/¯. (this is probably more a result of my coding than a specific problem). I also noticed that it wasn't using the official Boise Trails data to try to connect segments, so it was making some weird choices.

**Model recommendation**: It suggested that I connect the top of Five Mile Gulch to Boise Ridge Road and then come down "Femrite's Patrol". That didn't make sense to me at first because the trail up there is Hull's Ridge and the 8th street motorcycle road so I had ignored it. Once I was up top it was clear to me what it was trying to say though, so even though it had given me the wrong trail name, I was able to take it's advice and turn the loop into a pretty effective figure 8.

As to why it gave the wrong name, here's o3's explanation:

> In the morning I only had the default Boise Trails segment CSV (which does not include Trail 4 because it isn’t a 2025 required segment). My mental lookup table therefore had no entry for “Trail 4” but did have _Femrite’s Patrol_.

Winner for the day: Model recommendation, even with the mistaken trail name.

- Total distance 18.6 mi

- Miles of official segments completed: 12.82

- Duplicate distance 0.6 mi (3 %)

- Duplicate elevation ~190 ft (6 %) ≤ 15 %

- Challenge segments completed 10
