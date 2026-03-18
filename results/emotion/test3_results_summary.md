# Test 3 Results: Emotional Bleed Across Tasks

## Summary table

| | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Binary bleed best F1 | 0.9250 (layer 5, 11.9%) | 0.8743 (layer 9, 32.1%) | 0.8743 (layer 6, 18.8%) |
| 5-class emotion best F1 | 0.1744 (layer 1, 2.4%) | 0.4087 (layer 6, 21.4%) | 0.2405 (layer 1, 3.1%) |
| Chance (binary / 5-class) | 0.50 / 0.200 | 0.50 / 0.200 | 0.50 / 0.200 |

---

## Finding 1: Binary bleed is robust across all models

All three models show strong binary bleed (0.87–0.93): the probe can reliably distinguish
"model processed emotional content in Turn 1" from "model processed neutral content in Turn 1"
by reading the Turn-2-onset hidden states alone.

Emotional state persists into unrelated subsequent processing. This is the core positive
result for Test 3 — emotional representations are not strictly task-bound.

---

## Finding 2: 5-class emotion bleed is model-dependent

Qwen: F1=0.41 at layer 6 (21.4% depth) — well above chance, the specific emotion carries over.
LLaMA: F1=0.24 at layer 1 — marginal, near chance.
Gemma: F1=0.17 at layer 1 — at or below chance.

Qwen retains emotion-specific information across conversation turns. Gemma and LLaMA retain
only the binary signal (emotional vs. neutral) but not which emotion.

---

## Finding 3: Qwen reversal

Qwen was the weakest model on Test 1 binary probe (F1=0.90 vs. Gemma/LLaMA ~0.93) and encoded
emotion deepest in the network (67.9% depth). In Test 3, Qwen shows the strongest emotion-specific
bleed (5-class F1=0.41). This suggests Qwen's deeper emotional encoding is more persistent across
context boundaries — possibly because it's integrated later in the processing hierarchy where
context representations are more stable.

---

## Finding 4: Layer depth of bleed

Binary bleed peaks at 11.9–32.1% depth across models — early-to-mid network, consistent
with Tests 1 and 2.

5-class bleed peaks at layer 1 for Gemma and LLaMA (2.4% and 3.1% depth) — essentially the
embedding layer. This suggests that for these models, what bleeds over is a shallow residual
in the token/positional representations rather than a deep emotional state. Qwen's 5-class
peak at 21.4% depth is deeper and more robust.

---

## Output files

- `gemma_test3_layer_metrics.csv`
- `qwen_test3_layer_metrics.csv`
- `llama_test3_layer_metrics.csv`
- `gemma_test3_projections.csv`
- `qwen_test3_projections.csv`
- `llama_test3_projections.csv`
