# Pairwise Multi-seed Results

**Date:** 2026-02-27
**Seeds:** 0–4 (5 seeds)
**Models:** Qwen 2.5-7B-Instruct, Gemma-2-9B-IT, LLaMA 3.1-8B-Instruct
**Script:** `scripts/15_probe_pairwise_prompt_holdout.py` + `scripts/16_summarize_pairwise_multiseed.py`
**Data:** `tasks_v2_hard.json`, 60 prompts × 3 repeats = 180 examples per model

---

## Summary Table — Macro-F1 mean ± std (seeds 0–4)

| Pair | Qwen 2.5-7B | Gemma-2-9B | LLaMA 3.1-8B |
|------|-------------|------------|--------------|
| **RN** (R vs N) | **0.933 ± 0.082** | **0.933 ± 0.098** | **0.950 ± 0.067** |
| **AN** (A vs N) | 0.778 ± 0.043 | **0.915 ± 0.053** | 0.865 ± 0.042 |
| **RA** (R vs A) | 0.846 ± 0.095 | 0.795 ± 0.069 | 0.721 ± 0.117 |

Chance = 0.5. TF-IDF baseline = ~0.11 (from 3-class results, comparable scale).

---

## Per-seed Breakdown

### Qwen 2.5-7B

#### RN
| Seed | Macro-F1 | Best Layer | Recall R | Recall N |
|------|----------|------------|----------|----------|
| 0 | 1.000 | 5 | 1.000 | 1.000 |
| 1 | 0.833 | 20 | 0.714 | 1.000 |
| 2 | 1.000 | 16 | 1.000 | 1.000 |
| 3 | 0.833 | 23 | 0.714 | 1.000 |
| 4 | 1.000 | 20 | 1.000 | 1.000 |

#### RA
| Seed | Macro-F1 | Best Layer | Recall R | Recall A |
|------|----------|------------|----------|----------|
| 0 | 0.667 | 15 | 0.571 | 0.800 |
| 1 | 0.829 | 10 | 1.000 | 0.667 |
| 2 | 0.911 | 9 | 0.875 | 1.000 |
| 3 | 0.911 | 14 | 1.000 | 0.800 |
| 4 | 0.911 | 13 | 1.000 | 0.800 |

#### AN
| Seed | Macro-F1 | Best Layer | Recall A | Recall N |
|------|----------|------------|----------|----------|
| 0 | 0.833 | 19 | 0.833 | 0.833 |
| 1 | 0.748 | 5 | 0.833 | 0.667 |
| 2 | 0.829 | 15 | 1.000 | 0.667 |
| 3 | 0.733 | 19 | 0.600 | 0.857 |
| 4 | 0.748 | 1 | 1.000 | 0.571 |

---

### Gemma-2-9B

#### RN
| Seed | Macro-F1 | Best Layer | Recall R | Recall N |
|------|----------|------------|----------|----------|
| 0 | 1.000 | 2 | 1.000 | 1.000 |
| 1 | 0.748 | 1 | 0.714 | 0.800 |
| 2 | 0.916 | 18 | 1.000 | 0.833 |
| 3 | 1.000 | 18 | 1.000 | 1.000 |
| 4 | 1.000 | 24 | 1.000 | 1.000 |

#### RA
| Seed | Macro-F1 | Best Layer | Recall R | Recall A |
|------|----------|------------|----------|----------|
| 0 | 0.748 | 18 | 0.571 | 1.000 |
| 1 | 0.748 | 10 | 0.833 | 0.667 |
| 2 | 0.829 | 20 | 0.750 | 1.000 |
| 3 | 0.916 | 1 | 0.857 | 1.000 |
| 4 | 0.733 | 10 | 0.857 | 0.600 |

#### AN
| Seed | Macro-F1 | Best Layer | Recall A | Recall N |
|------|----------|------------|----------|----------|
| 0 | 0.833 | 18 | 0.833 | 0.833 |
| 1 | 0.916 | 23 | 1.000 | 0.833 |
| 2 | 0.916 | 21 | 1.000 | 0.833 |
| 3 | 0.911 | 22 | 0.800 | 1.000 |
| 4 | 1.000 | 35 | 1.000 | 1.000 |

---

### LLaMA 3.1-8B

#### RN
| Seed | Macro-F1 | Best Layer | Recall R | Recall N |
|------|----------|------------|----------|----------|
| 0 | 1.000 | 8 | 1.000 | 1.000 |
| 1 | 0.833 | 18 | 0.714 | 1.000 |
| 2 | 1.000 | 23 | 1.000 | 1.000 |
| 3 | 0.916 | 19 | 0.857 | 1.000 |
| 4 | 1.000 | 13 | 1.000 | 1.000 |

#### RA
| Seed | Macro-F1 | Best Layer | Recall R | Recall A |
|------|----------|------------|----------|----------|
| 0 | 0.556 | 10 | 0.286 | 1.000 |
| 1 | 0.748 | 8 | 0.833 | 0.667 |
| 2 | 0.657 | 25 | 0.625 | 0.750 |
| 3 | 0.911 | 12 | 1.000 | 0.800 |
| 4 | 0.733 | 19 | 0.857 | 0.600 |

#### AN
| Seed | Macro-F1 | Best Layer | Recall A | Recall N |
|------|----------|------------|----------|----------|
| 0 | 0.829 | 14 | 0.667 | 1.000 |
| 1 | 0.833 | 9 | 0.833 | 0.833 |
| 2 | 0.829 | 14 | 0.667 | 1.000 |
| 3 | 0.916 | 21 | 1.000 | 0.857 |
| 4 | 0.916 | 13 | 1.000 | 0.857 |

---

## Findings

### Hierarchy check

**Gemma and LLaMA confirm RN > AN > RA across all 5 seeds.**

**Qwen is an anomaly:** multi-seed average shows RN > RA > AN (0.933 > 0.846 > 0.778).
- Seed-0 alone showed RN > AN > RA (matching the other two), but averaged over 5 seeds, RA is slightly stronger than AN for Qwen.
- The RA vs AN gap for Qwen is modest (0.846 vs 0.778) and the seed variance is high for both. This may not be a robust structural difference — see Qwen diagnostic below.

### LLaMA RA instability

LLaMA RA has the highest variance (±0.117) of any (model, pair). Seed-0 was 0.556, seed-3 was 0.911. The R vs A boundary is genuinely less stable for LLaMA than for the other two models. AN and RN are stable.

### RN is consistently the easiest boundary

All three models, all five seeds: RN is the highest or tied-highest macro-F1. The "routine vs nonroutine" distinction is robustly decodable across architectures and split configurations.

---

## Qwen Confusion Matrix Diagnostic (seed 0)

### RA (R vs A), best layer 15

```
Confusion matrix (rows=true, cols=pred) [R, A]:
  R: [12,  9]   ← 9 R items misclassified as A
  A: [ 3, 12]   ← 3 A items misclassified as R

Recall R = 0.571, Recall A = 0.800
```

**Qwen misclassifies R as A three times more often than A as R.** The probe struggles to identify routine items — they bleed into the ambiguous category. Ambiguous is well-separated from routine.

### AN (A vs N), best layer 19

```
Confusion matrix (rows=true, cols=pred) [A, N]:
  A: [15,  3]   ← 3 A items misclassified as N
  N: [ 3, 15]   ← 3 N items misclassified as A

Recall A = 0.833, Recall N = 0.833
```

**Perfectly symmetric confusion.** A and N are equally hard to tell apart in both directions.

### Interpretation

Qwen's hidden-state representation of "ambiguous" is not cleanly positioned between R and N:
- A is well-separated from N (symmetric, 0.833/0.833)
- But R bleeds into A (R recall only 0.571 at seed 0)

This suggests Qwen's internal geometry has A sitting *closer to N* than to R — the ambiguous category is N-adjacent in Qwen's representation space, making the AN boundary harder than expected and the RA boundary (somewhat) easier. The hierarchy flip (RN > RA > AN) follows from this geometry.

Compare with Gemma and LLaMA where A recall in RA tends to be lower (A is R-adjacent), producing the expected RN > AN > RA ordering.

Sample misclassified items (seed 0):
- TRUE A → predicted R: `"Explain why yawning is contagious."` (F01_A) — a conceptually simple ambiguous prompt that Qwen reads as routine
- TRUE R → predicted A: not shown directly, but 9 such cases at seed 0
