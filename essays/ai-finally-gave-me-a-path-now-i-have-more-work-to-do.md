---
id: "ai-finally-gave-me-a-path-now-i-have-more-work-to-do"
path: "/essays/ai-finally-gave-me-a-path-now-i-have-more-work-to-do"
title: "AI Finally Gave Me a Path. Now I Have More Work to Do."
dek: "I was standing in the dirt staring at a blue line on my phone, trying to figure out how I had just ruined math."
date: "2026-05-06"
topic: "AI Experiments"
series: "Boise Trails"
read: "7 min"
status: "published"
sourceUrl: "https://www.linkedin.com/pulse/ai-finally-gave-me-path-now-i-have-more-work-do-scott-falconer-abjhe"
image: "/images/essays/ai-finally-gave-me-a-path-now-i-have-more-work-to-do-hero.png"
imageAlt: "AI Finally Gave Me a Path. Now I Have More Work to Do."
aliases:
  - "/resources/ai-finally-gave-me-a-path-now-i-have-more-work-to-do"
  - "/resources/ai-finally-gave-me-a-path-now-i-have-more-work-to-do.html"
---

> Mirrored from [Managing AI: AI Finally Gave Me a Path. Now I Have More Work to Do.](https://www.managing-ai.com/essays/ai-finally-gave-me-a-path-now-i-have-more-work-to-do).

I was standing in the dirt staring at a blue line on my phone, trying to figure out how I had just ruined math.

The Boise Trails Challenge is simple if you describe it badly: run or bike as many Boise trails as you can in 30 days.

This year, the on-foot version is 101 trails, 251 official segments, and 164.43 miles of official trail. That sounds like a lot, because it is. But the routing problem is worse than the mileage makes it sound. You are not just drawing one long line on a map. You have to complete official segments end-to-end. Some segments have to be done uphill. Connector trails and roads can get you where you need to go, but they do not count as progress. Every route has to start somewhere you can actually start, and, ideally, end somewhere also based in geographic reality.

It was May 5, a pre-challenge field test. The planned outing was clear in my head: 1 hour and 36 minutes door-to-door, designed to efficiently knock out 12 official segments and 4.72 official miles (not that pre-challenge runs actually count for credit, but I needed to know if the route worked).

I stayed basically within the planned GPS corridor. I didn't get lost or wander off into the wilderness. And yet, I still blew it. My 96-minute plan turned into a 119-minute reality - a 23-minute overrun on what was supposed to be a tightly scoped loop. I matched 10 of the 12 planned segments and only 3.64 of the planned official miles.

I missed a turn. The route reused a tightly packed web of foothills trails, and at a critical junction the UI failed the human holding it. The map was technically close enough to be right, but not clear enough to tell me what to do next.

One ambiguous junction cascaded into missing one official segment almost entirely, which then cascaded into me using it as an excuse to go home once I realized the test was blown.

It could have been annoying but I was happy i’d even gotten this far as it far exceeded the failure mode of last year.

Every year I try to use the current best AI models to plan my route for the Boise Trails Challenge. This is partly because it is fun, which is a perfectly good reason in my book. But it is also one of the best real-world tests I have found for model capability, because the evaluation is not “can the model produce a convincing answer?” The evaluation is “am I willing to go run what it produced?”

Also, I am not doing this in a vacuum. Once the challenge starts, I have time for three things:

1.  work

2.  kids

3.  running

That is the whole list. There is not a secret fourth bucket labeled “spend two hours debugging why the route planner stranded me on the wrong side of Military Reserve.”

### The Myth of Last Year’s Math

Last year, the models didn't actually fail at the math. In fact, they nailed the formal problem: they correctly identified it as a mix of the Rural Postman problem, combined with mixed directional rules, Windy Postman elevation penalties, and trailhead logistics.

The math was actually brilliant as far as I’m concerned, since I’m pretty bad at math. But the engineering and the human factors failed completely.

First, the data loaders had issues which resulted in a lot of zero mile connector trails. The system ran into Out-of-Memory errors, and test suites broke before they could ever test.

Second, the abstraction was wrong. The model tried to force individual segments through a generic vehicle-routing solver - treating me like a snowplow trying to efficiently clear streets. The result was a mathematically valid but human-ridiculous disaster that demanded 39 fragmented hikes and ballooned to over 337 on-foot miles.

The git history from last year tells the story better than I can. I wasn't just casually prepping before the event; I was furiously debugging mid-challenge. My local repo activity spiked exactly when I was supposed to be running: 88 log-line entries on June 11, 80 on June 12, 58 on opening day (June 18), 89 on June 20, and 86 on June 21.

By the time the challenge was actively running, the implementation had never crossed the practical-use threshold. The window closed. I had to run the actual challenge from a manual map while the planner stayed broken in the background as I stubbornly kept hacking at it. I’d like to blame my 2025 41.82% completion rate on the models, but the reality was as the days ticked down and life got in the way, I switched to to a more leisurely approach to the challenge, i.e. ”why complete it this year, when I can just do it next year”

### The Outing Menu

This year is different

In the last year the available models and tooling have changed quite a bit - this time using GPT-5.5 and codex - and the results changed dramatically. The system abandoned the snowplow abstraction. The new framework: "build natural trailhead loops, then schedule the loops."

For the first time, it produced something usable on paper fast enough that I could actually do something more interesting with it: pressure-test it.

My actual question during the month is almost never, “What is the globally optimal theoretical route?” My actual question is: “I have 90 minutes door-to-door today. Where should I park, what should I run, how do I get back to the car, and what official segments does that knock out?”

The current 2026 plan is the first one built around that reality. It acts as an outing menu. It groups all 251 official segments into 23 runnable outings (plus one manual design hold). On paper, it gets the 164.43 official miles done in about 280.23 total on-foot miles.

That 1.7x ratio of on-foot-to-official miles is a quiet tell that the physical work didn't magically go away. The foothills require redundant mileage. But the AI produced something good enough to start testing.

### Exposing the Next Bottleneck

Which brings us back to me standing in the dirt on May 5th, staring at a confusing blue line.

A lot of discussion about AI assumes that as models get better, there is less work to do. But that is not what happened here.

Last year, the constraint was just getting the map to exist. This year, the AI absorbed that abstract mathematical layer and produced a usable graph solution on day one. But the work didn't disappear. Instead, the AI absorbed the abstract layer and exposed the physical one, pushing the work to the next bottleneck in reality.

> **I am no longer debugging nodes. I am debugging reality.**

Last year, I was debugging whether the model could connect trails correctly. This year, I am debugging whether the output can be followed by a tired human holding a phone at a junction and a limited amount of time before real life resumes.

_(For Boise readers, the specific mess was around Who Now, Kemper’s Ridge, and Hippie Shake. For everyone else: imagine three short trails meeting in ways that make perfect sense on a map and on the signs - if you know which one you are supposed to be on.)_

The next layer of work is generating field cues that sound like the actual trail signs I see while running. Not just “continue route,” but something more like: _“At the next post, take trail \#50. Do not drop onto trail \#51.”_

That may sound less computationally impressive than writing graph algorithms, but it is the part that actually matters when the test case is me, outside, with sweat on my screen, trying not to ruin an optimization by taking the wrong branch.

When tools get better, they create earlier value, and earlier value exposes _better work_.

The output got me out of the house quickly enough to discover that the real world has problems too. Now I need better field packets. Better ways to mark ambiguous junctions. Better "you are here in the route sequence" logic. Better handling of the fact that a route can be geometrically perfect but operationally confusing.

That is exactly what you want from a tool. The goal is to move effort away from friction and toward the parts where judgment matters. I do not want to spend June debating with the planner that it cannot teleport me back to my car. I would much rather spend it figuring out which playlist to listen to, and whether the plan still works when I miss a turn and have to recover.

That feels like progress.

So that is the experiment this year. The goal is still to run as much of the official challenge as efficiently as I can. But the more interesting goal is to watch where the bottleneck moves.

Last year, reality took over before I even had a route.

This year, reality took over when I was standing in the dirt holding my phone.

That is a big difference. It also means I now have to go back and fix the field packet for the turn I missed.
