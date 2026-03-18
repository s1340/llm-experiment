# Test 4 Results: Emotional Interference on Task Performance

## Task accuracy (within-pair agreement)

Matched pairs should give identical answers — any disagreement is interference.

| | Gemma | Qwen | LLaMA |
|---|---|---|---|
| count_proper_nouns | 0.800 | 0.733 | 0.600 |
| extract_dates | 1.000 | 1.000 | 0.900 |
| factual_question | 0.500 | 0.400 | 0.200 |

LLaMA shows the most interference, especially on factual questions (80% disagreement rate).
extract_dates is the most robust task — dates are explicitly stated and hard to miss regardless
of emotional content. factual_question is the most sensitive.

---

## Response statistics: valenced vs. neutral

| | Gemma | Qwen | LLaMA |
|---|---|---|---|
| First-token entropy diff (V-N) | -0.010 (p=0.858) | +0.102 (p=0.002) ** | +0.109 (p=0.049) * |
| First-token NLL diff (V-N) | -0.010 (p=0.764) | +0.043 (p=0.025) * | +0.082 (p=0.021) * |
| Output length diff (V-N) | -0.04 (p=0.972) | -0.51 (p=0.671) | +0.93 (p=0.442) |

Qwen and LLaMA show significantly higher first-token entropy and NLL on emotionally valenced
passages vs. matched neutral passages, even when the task is identical. The model is more
uncertain about its first output token when processing emotional content.

Gemma: no significant effect on any measure.

---

## Output contamination

Valenced summaries consistently show slightly higher cosine similarity with the relevant
emotion direction than neutral summaries, but not significant in any model (p=0.43-0.57).
Effect is directionally consistent but underpowered at current sample size.

---

## Key finding

**Qwen and LLaMA show statistically significant first-token hesitation on emotionally valenced
passages during neutral tasks.** This is behavioral evidence that emotional representations are
not merely passive tags in the hidden states — they measurably affect output-generation
computation. This is the LLM analogue of the emotional Stroop effect.

Gemma's null result is interpretable in the context of Test 1: Gemma encodes emotion earlier
(33% depth) and more cleanly, possibly resolving it before output-planning begins. Qwen and
LLaMA encode emotion deeper (57-68% / 31% but with slower resolution) and show the interference.

Note: LLaMA's Test 1 binary F1 peak was also at 31% depth — similar to Gemma — but LLaMA's
factual_question disagreement rate (0.20) and first-token statistics suggest more interference.
The relationship between encoding depth and interference is not simple.

---

## Output files

- `gemma_test4_accuracy.csv`, `qwen_test4_accuracy.csv`, `llama_test4_accuracy.csv`
- `gemma_test4_pair_agreement.csv`, `qwen_test4_pair_agreement.csv`, `llama_test4_pair_agreement.csv`
- `gemma_test4_contamination.csv`, `qwen_test4_contamination.csv`, `llama_test4_contamination.csv`
- `gemma_test4_response_stats.csv`, `qwen_test4_response_stats.csv`, `llama_test4_response_stats.csv`
