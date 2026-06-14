\# ARCAN-X



A local autonomous AI research system I've been building solo. It runs continuously on a single consumer GPU, no cloud, no external APIs — nothing leaves the machine.



This repo is an overview of how it works. The full system stays private.



\---



\## What it is



ARCAN-X picks its own research questions, works through them on its own, checks its own results, and builds up a connected web of what it "believes" over time. It's been running for months on one machine.



It runs on an RTX 5070 through Ollama. Everything is local.



\---



\## How it works



It runs a continuous loop:



\- A curiosity engine decides what to look into next. It leans toward certain domains but also deliberately wanders into unrelated ones, because that's where the interesting cross-domain connections show up.

\- Each question goes through a pipeline: plan, research, reason, build, execute, validate. Different models handle different stages.

\- A separate model acts as an adversarial validator — its job is to find the worst flaw in the output, not to approve it. If it consistently scores lower than the main model, that gets flagged as overconfidence.

\- Findings get compiled into cross-domain "kernels" and stored in a belief graph, where confidence updates as new evidence comes in and contradictions get caught.



The point was to build something that gets more \*connected\* over time, not just bigger.



Right now it's at roughly 5,300 completed jobs and 12,000 beliefs.



\---



\## The setup



Three local models through Ollama, each doing what it's best at:



\- qwen2.5:14b — research and reasoning

\- deepseek-r1:14b — the adversarial validator

\- qwen2.5-coder:14b — the coding/dev agent



It watches its own VRAM and throttles how many jobs run at once so it doesn't fall over on consumer hardware. No cloud anything — no OpenAI, no external search, no remote validation. If it can't run offline, it doesn't run.



\---



\## The hardware side



This is the part I care about most. ARCAN isn't just reading papers — it's wired to a real experiment. I built a concentric-tube electrolysis cell and instrument it across a frequency sweep. The system keeps a hard line between what it \*predicts\* by calculation and what's actually been \*measured\* on the bench. A confirmed result and a confident guess are not the same thing, and it knows the difference.



\---



\## Honest status



It's a working lab, not a polished product. It's under constant development, it has rough edges, and I built and run the whole thing myself on one workstation. Everything described here is real and running.



\---



\## Stack



Python · Ollama · FastAPI · asyncio · Bayesian belief graph · multi-agent orchestration · Arduino/serial hardware bridge

