---
id: "i-am-the-one-that-loops"
path: "/essays/i-am-the-one-that-loops"
title: "i am the one that loops"
dek: ""
date: "2026-05-09"
publishedAt: "2026-05-09T09:16:04-06:00"
topic: "AI Experiments"
series: "Boise Trails"
read: "8 min"
status: "published"
sourceUrl: ""
image: "/images/essays/i-am-the-one-that-loops-hero.png"
imageAlt: "i am the one that loops"
aliases:
---

> Mirrored from [Managing AI: i am the one that loops](https://www.managing-ai.com/essays/i-am-the-one-that-loops).

Let's start with two things. 1) I'm bad at math. 2) I'm a very slow runner. 3). My navigation isn't great.

[In the last post](https://www.managing-ai.com/essays/ai-finally-gave-me-a-path-now-i-have-more-work-to-do), I wrote about standing in the dirt staring at a blue line on my phone, trying to figure out how I had just ruined math.

The short version: AI finally gave me a route for the Boise Trails Challenge that was good enough to take outside. I ran it. The route was geographically close enough to be right and confusing enough to not. I missed a turn, bailed out, made it home twenty-three minutes late, and learned that "the route exists" is not the same thing as "I can follow this."

That post was about the field test.

This one is about the loop.

Not the trail loop. The other loop.

The one where I am not just using a tool. I used AI to build the tool. Then I took it outside and found out what I actually asked for.

### The agent and I also get lost

There is another version of getting lost.

On the trail, at least, I can describe the damage. I can say: the sign said this, I did that, the line on the map doubled back on itself, we all got confused. The field gives me facts. Facts I don't like, usually, but facts.

When talking with the agent though, it is weirder.

I can tell it the impact of our decisions. I can tell it what happened. I can tell it that the route was close but how we described it was not help. What I sometimes cannot explain is the architectural reason we ended up there.

This is fine when one of us knows the next move.

Sometimes it does.

When I realized the off-the-shelf GPS/map experience was not good enough, my instinct was to build an iPhone app and install it through TestFlight. Getting GPS / location access on the iPhone can be a weird hop of user authorization and not all systems have access to it. I had done something like that many years ago using an iPhone app, which is exactly the kind of experience that makes you dangerous. I had a real memory of a real solution, but that did not mean it was the right solution now.

The model told me I was overcomplicating it.

I did not need a native app. I did not need TestFlight. I could build a progressive web app. Maybe in the browser it would not have my real-time location, but if I installed it on the iPhone home screen, it would.

Beyond that, I didn't need to worry about hosting it anyway if I just put it on GitHub Pages.

This is the good version of the loop. I was about to spend a bunch of time solving the wrong problem, and the agent had enough current iOS/web context to stop me.

Other times, I know and it does not.

The agent kept getting hung up on parking, asking me to validate whether a trailhead or roadside pullout was real. I finally said something like: if I previously started a run there on Strava, or there is a road there, we can treat it as a place I can park. I also told it that for potential spots that neither of us had ever been to, it should bring up Street View so we could look.

It had not thought of that.

But at the same time, its definition of road showed its lack of experience in our world. , e.g. it recommended I drive on a cat track at a ski resort. I did not need to go up there and field-test that one to realize why it wasn't ideal.

Those cases are easy. One of us has the answer.

The hard cases are where neither of us does.

We do not know whether a better answer exists. We do not know whether the route is bad, or the map is bad, or how we're describing the problem is bad. We do not know whether the tradeoff is mathematical, architectural, geographic, or something we're not considering.

That does not happen as much in systems I already understand. If I am working in a familiar codebase or product surface, I can usually tell when the agent is off track. I know the feel of the answer even if I do not know every step.

But here, we are both out of our depth.

The agent does not know the foothills in a real human way. I do not know the math behind the optimizer.

but then I go run it and that is when the system becomes honest.

### Day 3: suspiciously, it worked

A few days after the first Harrison Hollow attempt, I reran it. This time, the run worked.

I had not resolve my own terrible sense of navigation in three days

The loop improved.

The system generated a route.

The route generated an experience.

The experience generated a better specification.

The better specification generated a better route.

And, because everything is a loop, the successful run found the next problem.

Near the #53 Buena Vista / #52 Kemper's Ridge transition, the live map showed Buena Vista in both directions. The route was using Buena Vista as connector and repeat mileage. Once that connector had served its credit/access purpose, repeating it should not have remained mandatory. The remaining movement should have been re-optimized as ordinary legal connector routing.

In other words, the route was credit-correct and still field-wrong.

The first run taught me that a correct route can be unfollowable.

The second run taught me that a followable route can still be inefficient in a way you only notice once you are actually following it.

### The weird part is that I built it

Most tools I use, I did not build. Strava, the iPhone, etc. I do not know much of the inner workings or product decisions behind them. I use them anyway.

Sometimes they work. Sometimes they don't. That is normal product feedback: a company built something, I used it, I had an opinion.

This is different.

I was not just using someone else's black box. I was using AI to build my own black box.

I expressed an intent: help me complete this challenge efficiently and realistically. The system generated artifacts: route packets, maps, segment matchers, timing estimates, proof checks, audit reports, and a bunch of intermediate machinery I probably could not explain off hand.

Then I took one of those artifacts outside.

That changes the relationship.

I am not just a user.

I am also the person who caused the artifact to exist.

I gave the system my intent. It produced something beyond my full mechanistic understanding. I used it in the world. The world answered.

The first run did not tell me, "I liked the route" or "I disliked the route." It told me:

The route corridor can be right while the active leg is unclear.

Trail signs and official segment names do not always line up in the way the field packet assumes.

Dense map lines that overlap in space and color make for bad navigation.

A route can be correct for credit and wrong for execution.

These are the things I did not know to ask for before I failed.

Before the run, "make me a good route" sounded clear.

After the run, it meant:

Make me a route I can follow while moving, using the signs that exist, on the map I actually use, at the pace I actually run, in the amount of time I actually have.

That is a much better spec.

I did not know I needed it until I failed.
