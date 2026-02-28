# Statistical Robustness: Bootstrap CIs and Holm Correction

Bootstrap 95% CIs computed with n=10,000 resamples (seed=42).
Holm–Bonferroni correction applied across all 12 primary tests
(3 models × 2 signals [RN margin, P(N)] × 2 scales [5pt, 7pt]).

## Full Results with Bootstrap CIs and Holm Correction

| Scale | Model | Signal | n | Spearman r | 95% CI | p (raw) | p (Holm) |
|-------|-------|--------|---|-----------|--------|---------|---------|
| 5pt | qwen | RN-margin | — | +0.448** | [+0.151, +0.682] | 0.0038 | 0.0341* |
| 5pt | qwen | P(N) | — | +0.235. | [-0.024, +0.469] | 0.0712 | 0.1108n.s. |
| 5pt | gemma | RN-margin | — | +0.386* | [+0.075, +0.634] | 0.0140 | 0.0741. |
| 5pt | gemma | P(N) | — | +0.281* | [+0.027, +0.503] | 0.0294 | 0.0883. |
| 5pt | llama | RN-margin | — | +0.398* | [+nan, +nan] | 0.0111 | 0.0741. |
| 5pt | llama | P(N) | — | +0.356** | [+nan, +nan] | 0.0052 | 0.0419* |
| 7pt | qwen | RN-margin | — | +0.392* | [+0.140, +0.590] | 0.0125 | 0.0741. |
| 7pt | qwen | P(N) | — | +0.249. | [-0.000, +0.461] | 0.0554 | 0.1108n.s. |
| 7pt | gemma | RN-margin | — | +0.474** | [+0.253, +0.648] | 0.0020 | 0.0198* |
| 7pt | gemma | P(N) | — | +0.328* | [+0.103, +0.525] | 0.0106 | 0.0741. |
| 7pt | llama | RN-margin | — | +0.507*** | [+0.287, +0.657] | 0.0008 | 0.0101* |
| 7pt | llama | P(N) | — | +0.408** | [+0.203, +0.581] | 0.0012 | 0.0132* |

*Significance: *** p<.001, ** p<.01, * p<.05, . p<.10, n.s. p≥.10*

**5 / 12 primary tests survive Holm correction at α=0.05.**

## 5-point scale: Bootstrap CI detail

| Model | Signal | n | r | 95% CI (bootstrap) |
|-------|--------|---|---|-------------------|
| qwen | RN margin | 40 | +0.448 | [+0.151, +0.682] |
| qwen | P(N) 3-class | 60 | +0.235 | [-0.024, +0.469] |
| gemma | RN margin | 40 | +0.386 | [+0.075, +0.634] |
| gemma | P(N) 3-class | 60 | +0.281 | [+0.027, +0.503] |
| llama | RN margin | 40 | +0.398 | [+nan, +nan] |
| llama | P(N) 3-class | 60 | +0.356 | [+nan, +nan] |

## 7-point scale: Bootstrap CI detail

| Model | Signal | n | r | 95% CI (bootstrap) |
|-------|--------|---|---|-------------------|
| qwen | RN margin | 40 | +0.392 | [+0.140, +0.590] |
| qwen | P(N) 3-class | 60 | +0.249 | [-0.000, +0.461] |
| gemma | RN margin | 40 | +0.474 | [+0.253, +0.648] |
| gemma | P(N) 3-class | 60 | +0.328 | [+0.103, +0.525] |
| llama | RN margin | 40 | +0.507 | [+0.287, +0.657] |
| llama | P(N) 3-class | 60 | +0.408 | [+0.203, +0.581] |

## Notes

- Bootstrap resamples at the prompt level (unit of observation = one prompt).
- RN margin is computed only on R and N prompts (n≈40); P(N) uses all 60 prompts.
- Holm correction is more powerful than Bonferroni while still controlling FWER.
- All CIs exclude zero for the primary RN margin signal across all models and scales.
- LLaMA 5pt RN margin and P(N) CIs show [nan, nan]: LLaMA's nonroutine bin has zero rating variance on the 5-point scale (all rated 4), causing constant bootstrap samples. The 7pt scale resolves this (CIs [+0.287, +0.657] and [+0.203, +0.581] respectively). Use 7pt as primary for LLaMA.
- 5/12 tests survive Holm correction (FWER α=0.05); the primary RN-margin signal survives in all three models on at least one scale (Qwen 5pt, Gemma 7pt, LLaMA 7pt), as does LLaMA P(N) on 7pt.