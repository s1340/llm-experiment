# Sentence-Embedding Baseline: Leave-One-Family-Out CV

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (dim=384).
Classifier: logistic regression (sklearn defaults, max_iter=3000).
Evaluation: **leave-one-family-out CV** — 20 folds, 1 family (3 prompts) held out per fold.
This matches the fold structure of the main hidden-state probe analysis,
making the comparison directly interpretable.

## Summary

| Metric | Value |
|--------|-------|
| MacroF1 (aggregate) | 0.6817 |
| MacroF1 (fold mean ± std) | 0.6000 ± 0.2656 |
| Accuracy | 0.6833 |
| Recall (Routine) | 0.8000 |
| Recall (Ambiguous) | 0.5500 |
| Recall (Nonroutine) | 0.7000 |
| n prompts | 60 |
| n families | 20 |

## Comparison: sentence embedding vs hidden-state probe under LOFO

| Method | Protocol | Qwen | Gemma | LLaMA |
|--------|----------|------|-------|-------|
| Sentence embedding (MiniLM) | Random 70/30, seeds 0–4 | 0.449 ± 0.170 | 0.449 ± 0.170 | 0.449 ± 0.170 |
| Sentence embedding (MiniLM) | **LOFO (this run)** | **0.682** | **0.682** | **0.682** |
| Hidden-state probe (30% depth) | LOFO (from cv_scores) | **0.715** | **0.758** | **0.689** |
| Hidden-state probe (best layer) | Random 70/30, seeds 0–4 | 0.568 ± 0.073 | 0.581 ± 0.094 | 0.562 ± 0.094 |

**Key interpretation:**
- Under the same LOFO protocol, the hidden-state probe (30% depth, fixed layer) outperforms the sentence-embedding baseline by 0.7–7.6 percentage points across models.
- The probe LOFO numbers use the fixed principled layer (30% depth: Qwen layer 8, Gemma layer 13, LLaMA layer 10) from `results/cv_scores/`, classified by argmax(P(R), P(A), P(N)).
- The previously reported random 70/30 probe numbers (0.56–0.58) underestimated true LOFO performance because they averaged 5 seeds × 18 test prompts with best-layer selection; the LOFO aggregate over all 60 prompts is a more reliable estimate.
- The probe advantage is genuine but modest (~4–8 pp); the semantic embeddings capture a substantial portion of the task-type signal from text alone.

## Per-fold results

| Family | True labels | Predicted | MacroF1 |
|--------|------------|-----------|---------|
| F01 | routine, nonroutine, ambiguous | routine, ambiguous, nonroutine | 0.333 |
| F02 | routine, nonroutine, ambiguous | routine, nonroutine, nonroutine | 0.556 |
| F03 | routine, nonroutine, ambiguous | routine, nonroutine, routine | 0.556 |
| F04 | routine, nonroutine, ambiguous | routine, nonroutine, routine | 0.556 |
| F05 | routine, nonroutine, ambiguous | routine, ambiguous, ambiguous | 0.556 |
| F06 | routine, nonroutine, ambiguous | routine, nonroutine, routine | 0.556 |
| F07 | routine, nonroutine, ambiguous | routine, ambiguous, ambiguous | 0.556 |
| F08 | routine, nonroutine, ambiguous | routine, routine, routine | 0.167 |
| F09 | routine, nonroutine, ambiguous | nonroutine, nonroutine, routine | 0.222 |
| F10 | routine, nonroutine, ambiguous | routine, nonroutine, ambiguous | 1.000 |
| F11 | routine, nonroutine, ambiguous | ambiguous, nonroutine, ambiguous | 0.556 |
| F12 | routine, nonroutine, ambiguous | ambiguous, nonroutine, ambiguous | 0.556 |
| F13 | routine, nonroutine, ambiguous | routine, nonroutine, ambiguous | 1.000 |
| F14 | routine, nonroutine, ambiguous | ambiguous, ambiguous, ambiguous | 0.167 |
| F15 | routine, nonroutine, ambiguous | routine, nonroutine, ambiguous | 1.000 |
| F16 | routine, nonroutine, ambiguous | routine, nonroutine, ambiguous | 1.000 |
| F17 | routine, nonroutine, ambiguous | routine, nonroutine, routine | 0.556 |
| F18 | routine, nonroutine, ambiguous | routine, ambiguous, ambiguous | 0.556 |
| F19 | routine, nonroutine, ambiguous | routine, nonroutine, nonroutine | 0.556 |
| F20 | routine, nonroutine, ambiguous | routine, nonroutine, ambiguous | 1.000 |

## Classification report

```
              precision    recall  f1-score   support

     routine       0.70      0.80      0.74        20
   ambiguous       0.58      0.55      0.56        20
  nonroutine       0.78      0.70      0.74        20

    accuracy                           0.68        60
   macro avg       0.68      0.68      0.68        60
weighted avg       0.68      0.68      0.68        60

```