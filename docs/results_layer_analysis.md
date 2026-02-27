# Best-Layer Index Analysis (Proportional Depth)

**Date:** 2026-02-27
**Purpose:** Test the progressive refinement hypothesis — does RN peak earlier in the network than RA/AN?

---

## Layer counts by model

| Model | Transformer layers | Total hidden states (incl. embedding) | Divisor for prop. depth |
|-------|-------------------|----------------------------------------|-------------------------|
| Qwen 2.5-7B-Instruct | 28 | 29 | 28 |
| Gemma-2-9B-IT | 42 | 43 | 42 |
| LLaMA 3.1-8B-Instruct | 32 | 33 | 32 |

---

## Best-layer indices (by Macro-F1) per seed

### Qwen 2.5-7B

| Pair | s0 | s1 | s2 | s3 | s4 | mean | std |
|------|----|----|----|----|-----|------|-----|
| RN | 5 | 20 | 16 | 23 | 20 | 16.8 | 6.3 |
| RA | 15 | 10 | 9 | 14 | 13 | 12.2 | 2.3 |
| AN | 19 | 5 | 15 | 19 | 1 | 11.8 | 7.4 |

### Gemma-2-9B

| Pair | s0 | s1 | s2 | s3 | s4 | mean | std |
|------|----|----|----|----|-----|------|-----|
| RN | 2 | 1 | 18 | 18 | 24 | 12.6 | 9.3 |
| RA | 18 | 10 | 20 | 1 | 10 | 11.8 | 6.8 |
| AN | 18 | 23 | 21 | 22 | 35 | 23.8 | 5.9 |

### LLaMA 3.1-8B

| Pair | s0 | s1 | s2 | s3 | s4 | mean | std |
|------|----|----|----|----|-----|------|-----|
| RN | 8 | 18 | 23 | 19 | 13 | 16.2 | 5.2 |
| RA | 10 | 8 | 25 | 12 | 19 | 14.8 | 6.3 |
| AN | 14 | 9 | 14 | 21 | 13 | 14.2 | 3.9 |

---

## Proportional depth (best_layer / (total_layers − 1))

| Model | Pair | mean prop. depth | std |
|-------|------|-----------------|-----|
| Qwen | RN | 0.600 | 0.225 |
| Qwen | RA | 0.436 | 0.083 |
| Qwen | AN | 0.421 | 0.266 |
| Gemma | RN | 0.300 | 0.222 |
| Gemma | RA | 0.281 | 0.161 |
| Gemma | AN | **0.567** | 0.139 |
| LLaMA | RN | 0.506 | 0.162 |
| LLaMA | RA | 0.463 | 0.197 |
| LLaMA | AN | 0.444 | 0.121 |

---

## Assessment

### Does progressive refinement hold?

**Gemma — partial support:** AN peaks deepest (0.567), RN and RA are similar and earlier (~0.28–0.30). The AN-peaks-late pattern is consistent with "ambiguous vs nonroutine requires deeper processing," but RN does not clearly peak earlier than RA.

**LLaMA — no clear signal:** All three pairs cluster in 0.44–0.51 with overlapping standard deviations. No directional pattern is recoverable from argmax alone.

**Qwen — reversed:** RN peaks deepest (0.600), RA and AN are similar and shallower (~0.42–0.44). This is the opposite of the expected pattern.

### Critical methodological note: argmax instability at ceiling

When a boundary achieves near-perfect performance (e.g., RN F1 ≈ 1.0 at multiple seeds), **many layers achieve similar scores and the argmax is effectively random among them.** This causes high variance in best-layer indices across seeds — visible in the data:

- Qwen RN: layers 5, 20, 16, 23, 20 (std = 6.3) despite strong performance
- Gemma RN: layers 2, 1, 18, 18, 24 (std = 9.3) — seeds 0–1 peak early, seeds 2–4 peak mid/late

The argmax method is **not reliable for testing progressive refinement when performance is near ceiling.** The best-layer index becomes sensitive to noise in the test split.

### Recommended alternative analyses

To properly test the progressive refinement hypothesis:

1. **First-breakout layer:** For each seed, find the shallowest layer that exceeds a fixed threshold (e.g., F1 ≥ 0.80 or ≥ 0.90). This captures *when* the signal first emerges rather than where the argmax lands.

2. **Layer profile curves:** Plot F1 vs. layer for each (model, pair, seed) and compare the *shape* of the curves — does RN ramp up earlier than RA/AN?

3. **Early-layer mean:** Report mean F1 across layers 0–N/4 (first quartile of network) for each pair. If RN > RA/AN in early layers, that's evidence for early emergence regardless of where the global peak is.

These analyses require the per-layer F1 data stored in the raw result files, not just the argmax. Flagging for Opus to decide which is worth pursuing.

---

## Summary verdict

The seed-0 progressive refinement finding (RN peaks first 5–25% of network) **does not survive multi-seed averaging.** The pattern is real in some seeds but the argmax method is too noisy when performance is high. The strong RN F1 scores are robustly confirmed; the layer-depth story needs a better analysis method before it can be reported with confidence.
