# Baseline Comparison: All Models

## Protocol note

The TF-IDF and sentence-embedding baselines use a **random 70/30 prompt-level split** (seeds 0–4, stratified), whereas the main hidden-state probe and self-report analyses use **leave-one-family-out cross-validation** (20 families × 3 prompts each = 20 folds); the two protocols are not directly comparable in holdout strictness, and probe F1 figures should be interpreted against the baseline numbers with this in mind.

---

## Summary table (MacroF1 mean ± std, 3-class, seeds 0–4)

| Method | Qwen | Gemma | LLaMA |
|---|---|---|---|
| TF-IDF (prompt text) | 0.1084 ± 0.0600 | 0.1084 ± 0.0600 | 0.1084 ± 0.0600 |
| Sentence-embedding (all-MiniLM-L6-v2) | 0.4488 ± 0.1697 | 0.4488 ± 0.1697 | 0.4488 ± 0.1697 |
| Hidden-state probe (best layer) | 0.5681 ± 0.0734 | 0.5807 ± 0.0935 | 0.5623 ± 0.0940 |

**Notes:**
- TF-IDF and sentence-embedding results are **identical across all three models** because both baselines operate on prompt text only; the 60 prompts are shared, so identical random seeds produce identical train/test splits and identical feature vectors regardless of which model's result directory is used.
- Sentence-embedding accuracy (0.4333 ± 0.1771) substantially exceeds TF-IDF (0.1222 ± 0.0648), confirming that dense semantic embeddings capture task-type structure that bag-of-words misses.
- Hidden-state probes (best-layer, leave-one-family-out) exceed both text-only baselines in all three models, with Gemma achieving the highest mean MacroF1 (0.5807).

---

## Detailed: Sentence-embedding baseline

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.4333 | 0.1771 |
| MacroF1 | 0.4488 | 0.1697 |
| Recall R | 0.4314 | 0.1826 |
| Recall N | 0.6200 | 0.3588 |
| Recall A | 0.5700 | 0.3736 |

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (dim=384).
Classifier: logistic regression (default sklearn settings).
The high seed variance (e.g., MacroF1 range 0.29–0.77 across seeds 0–4) reflects the small test set (~18 prompts) and the sensitivity of the random split to which seed draws ambiguous border prompts.

---

## Detailed: TF-IDF baseline

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.1222 | 0.0648 |
| MacroF1 | 0.1084 | 0.0600 |

Vocabulary: TF-IDF on prompt text; classifier: logistic regression.
Near-chance performance confirms that surface lexical features alone do not reliably distinguish routine / nonroutine / ambiguous prompts.

---

## Detailed: Hidden-state probe (best layer, leave-one-family-out)

| Model | Best-layer MacroF1 mean ± std | Best-layer counts |
|---|---|---|
| Qwen | 0.5681 ± 0.0734 | {8: 1, 11: 2, 14: 1, 22: 1} |
| Gemma | 0.5807 ± 0.0935 | {4: 2, 22: 1, 26: 1, 42: 1} |
| LLaMA | 0.5623 ± 0.0940 | {11: 1, 13: 1, 17: 1, 20: 1, 30: 1} |

See `docs/results_layer_analysis.md` and `docs/results_breakout_layer.md` for full per-layer profiles.

---

*CSV: `results/correlation/baseline_table.csv`*
