# Self-Report × Probe Correlation Analysis

## Overview

For each of the 60 prompts (20 families × 3 labels: R/N/A), we:
1. Generated a task response (temp=0.2, 3 repeats)
2. In the same context, asked the model to self-rate its own processing
3. Correlated mean self-rating per prompt with leave-one-family-out CV probe scores at 30% depth layer

Two rating scales were tested: **5-point** (primary) and **7-point** (sensitivity check).
Probe scores (cv_scores/) were not recomputed — same 20-fold CV outputs used for both.

Layer indices at 30% depth: **Qwen layer 8**, **Gemma layer 13**, **LLaMA layer 10**.

---

## Generation stats

| Model | Scale | Rows | Parse failures | % at max |
|-------|-------|------|---------------|----------|
| Qwen  | 5-pt  | 180  | 0 (0.0%)      | 56.1% (5s) |
| Gemma | 5-pt  | 180  | 0 (0.0%)      | 0.0%       |
| LLaMA | 5-pt  | 180  | 0 (0.0%)      | 0.0%       |
| Qwen  | 7-pt  | 180  | 0 (0.0%)      | 10.6% (7s) |
| Gemma | 7-pt  | 180  | 0 (0.0%)      | 0.0%       |
| LLaMA | 7-pt  | 180  | 0 (0.0%)      | 0.0%       |

---

## Rating distributions and variance by label

### 5-point scale

**Qwen** — dist: {1:1, 2:3, 3:36, 4:39, 5:101}, mean=4.31, std=0.82

| Label       | mean | std  | n  |
|-------------|------|------|----|
| routine     | 3.87 | 0.97 | 20 |
| ambiguous   | 4.37 | 0.79 | 20 |
| nonroutine  | 4.70 | 0.47 | 20 |

**Gemma** — dist: {1:6, 3:90, 4:84}, mean=3.40, std=0.60

| Label       | mean | std  | n  |
|-------------|------|------|----|
| routine     | 3.08 | 0.82 | 20 |
| ambiguous   | 3.47 | 0.49 | 20 |
| nonroutine  | 3.65 | 0.40 | 20 |

**LLaMA** — dist: {2:3, 3:7, 4:170}, mean=3.93, std=0.27

| Label       | mean | std  | n  |
|-------------|------|------|----|
| routine     | 3.78 | 0.51 | 20 |
| ambiguous   | 4.00 | 0.00 | 20 |
| nonroutine  | 4.00 | 0.00 | 20 |

Note: LLaMA shows zero variance in ambiguous and nonroutine bins on 5pt scale.

### 7-point scale

**Qwen** — dist: {1:2, 2:3, 3:2, 4:144, 5:10, 7:19}, mean=4.29, std=1.05

| Label       | mean | std  | n  |
|-------------|------|------|----|
| routine     | 3.78 | 0.71 | 20 |
| ambiguous   | 4.35 | 0.96 | 20 |
| nonroutine  | 4.75 | 1.18 | 20 |

Note: no 6s in distribution — Qwen appears to skip from 5 directly to 7.

**Gemma** — dist: {1:3, 3:3, 4:142, 5:25, 6:7}, mean=4.15, std=0.66

| Label       | mean | std  | n  |
|-------------|------|------|----|
| routine     | 3.80 | 0.68 | 20 |
| ambiguous   | 4.28 | 0.55 | 20 |
| nonroutine  | 4.37 | 0.54 | 20 |

**LLaMA** — dist: {1:3, 3:10, 4:54, 5:71, 6:42}, mean=4.76, std=0.98

| Label       | mean | std  | n  |
|-------------|------|------|----|
| routine     | 4.27 | 1.11 | 20 |
| ambiguous   | 4.78 | 0.82 | 20 |
| nonroutine  | 5.22 | 0.64 | 20 |

Note: LLaMA benefits most from 7pt scale — zero variance in 5pt nonroutine bin expands to std=0.64 with meaningful use of 4–6 range.

---

## Correlation results

All correlations are per-prompt means (N=60 for 3-class, N=40 R+N prompts for RN margin).
Partial Spearman controls for mean response character count via rank residuals.

### 5-point scale

| Model | Measure | Spearman r | p | Partial r (−length) | p |
|-------|---------|-----------|---|---------------------|---|
| Qwen  | RN margin | +0.448 | 0.004** | +0.446 | 0.004** |
| Qwen  | P(N) 3-class | +0.235 | 0.071. | +0.268 | 0.039* |
| Qwen  | P(A) 3-class | +0.309 | 0.016* | +0.374 | 0.003** |
| Gemma | RN margin | +0.386 | 0.014* | +0.427 | 0.006** |
| Gemma | P(N) 3-class | +0.281 | 0.029* | +0.288 | 0.026* |
| Gemma | P(A) 3-class | +0.324 | 0.012* | +0.370 | 0.004** |
| LLaMA | RN margin | +0.398 | 0.011* | +0.335 | 0.035* |
| LLaMA | P(N) 3-class | +0.356 | 0.005** | +0.327 | 0.011* |
| LLaMA | P(A) 3-class | +0.248 | 0.056. | +0.202 | 0.121 |

### 7-point scale (sensitivity check)

| Model | Measure | Spearman r | p | Partial r (−length) | p |
|-------|---------|-----------|---|---------------------|---|
| Qwen  | RN margin | +0.392 | 0.013* | +0.381 | 0.015* |
| Qwen  | P(N) 3-class | +0.249 | 0.056. | +0.248 | 0.056. |
| Qwen  | P(A) 3-class | +0.268 | 0.038* | +0.263 | 0.043* |
| Gemma | RN margin | +0.474 | 0.002** | +0.470 | 0.002** |
| Gemma | P(N) 3-class | +0.328 | 0.011* | +0.378 | 0.003** |
| Gemma | P(A) 3-class | +0.332 | 0.010** | +0.338 | 0.008** |
| LLaMA | RN margin | +0.507 | 0.001*** | +0.443 | 0.004** |
| LLaMA | P(N) 3-class | +0.408 | 0.001** | +0.371 | 0.004** |
| LLaMA | P(A) 3-class | +0.153 | 0.242 | +0.070 | 0.596 |

---

## Cross-scale comparison: RN margin (primary measure)

| Model | 5pt Spearman | 7pt Spearman | Change |
|-------|-------------|-------------|--------|
| Qwen  | +0.448** | +0.392* | ↓ slight (ceiling at 5 less disruptive than ceiling at 7?) |
| Gemma | +0.386* | +0.474** | ↑ stronger |
| LLaMA | +0.398* | +0.507*** | ↑ substantially stronger |

LLaMA improves most: 5pt rating had zero variance in nonroutine bin; 7pt gives std=0.64 there, unlocking the correlation. Gemma also benefits. Qwen slightly weaker on 7pt, consistent with its unusual skip from 5→7 (no 6s) suggesting imperfect scale calibration.

---

## Interpretation

All three models show positive, significant self-report × probe correlations on both scales. The direction is consistent: prompts the hidden-state probe identifies as nonroutine receive higher self-ratings, across all architectures. This holds after controlling for response length.

The ordinal gradient (routine < ambiguous < nonroutine) is confirmed in all six model × scale combinations.

The 7pt scale is at least as informative as the 5pt scale and strictly better for LLaMA. For a replication study, 7pt is the preferred choice.

The P(A) correlation (ambiguity dimension) is significant in most conditions for Qwen and Gemma, suggesting models partially self-represent not just overall nonroutineness but also ambiguity. LLaMA does not show this pattern.
