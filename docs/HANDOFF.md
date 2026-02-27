 
GPT said: 

# Experiment Handoff (canonical)

## What this repo is
This repo tracks **scripts + docs only** (no big artifacts). `data/`, `results/`, `logs/`, tensors, weights are ignored via `.gitignore`.

## Current objective
Detect and characterize “processing mode” signatures in hidden states using **prompt-text holdout** probes, then move to **self-report correlation**.

## Latest completed work (✅)
### Pairwise multi-seed probes (seeds 0–4)
- Completed across **Qwen2.5-7B / Gemma2-9B / LLaMA3.1-8B** for pairs **RN / AN / RA**.
- Summary + per-seed breakdown + notes:
  - `docs/results_pairwise_multiseed.md`

Key takeaways:
- RN is strongly decodable and stable across models/seeds.
- AN is strong (esp. Gemma).
- RA is weakest overall; LLaMA RA shows higher variance.
- Qwen shows a hierarchy flip (RN > RA > AN) with a confusion-matrix explanation.

### Layer-depth analysis (argmax issue)
- `docs/results_layer_analysis.md`

Key takeaways:
- “Best-layer by argmax” is unreliable under near-ceiling performance (ties across many layers => argmax becomes noise).
- Seed-0 “progressive refinement” pattern was real but not robust under multi-seed aggregation.
- Recommended replacements: breakout-layer metric (preferred), later optionally layer-profile curves for figures.

### Qwen diagnostic
- Included in results writeups: Qwen RA vs AN confusion pattern plausibly explains hierarchy flip.

## Next task (🔜) — Breakout-layer depth metric (HIGH PRIORITY)
Goal: recover a robust depth story using **onset** instead of **peak**.

Compute **First Breakout Layer**:
For each `(model, pair, seed)`, find the **shallowest layer index** where **Macro-F1 ≥ 0.80**.

Report (per model × pair):
- mean ± std of breakout layer index across seeds
- proportional depth = index / (n_layers - 1)
- % of runs that never reach 0.80 (if any)

Save to:
- `docs/results_breakout_layer.md`

This replaces argmax as our “progressive refinement” test:
- Expectation: RN breaks out earlier than RA/AN (if refinement story holds).

## After breakout-layer (🔜)
### Self-report correlation design
Design a generation + rating protocol:
- model answers prompt
- then provides a 1–5 “routine-ness” rating (+ brief justification)
- correlate ratings with probe probabilities / margins from hidden states

## How to run (for reference)
- Pairwise probe runner: `scripts/run_pairwise_multiseed.sh`
- Pairwise probe script: `scripts/15_probe_pairwise_prompt_holdout.py`
- Summarizer: `scripts/16_summarize_pairwise_multiseed.py`

## Conventions
- Put any new summaries/tables into `docs/` and commit/push.
- Log notable runs/changes in `docs/RUNLOG.md` (append-only).


Opus said: 
Next task: First-breakout-layer analysis. For each (model, pair, seed), find the shallowest layer where F1 ≥ 0.80. Report mean±std of raw index and proportional depth across seeds. Save to docs/results_breakout_layer.md and push.
Once that's in, we have all the structural evidence we need and we move to designing the self-report correlation experiment — which is the exciting part.