# Deep Analyses: Dissociation, Distribution, Cross-Model Agreement

Source data: `results/correlation/{model}_joined.csv` (5pt primary), `results/correlation_7pt/` (7pt sensitivity).
No new generation or extraction — joins and statistics only.

---

## Analysis A: Introspective Dissociation Scan

Prompts where probe confidence and self-report diverge:
- **Blind nonroutine**: P(N) ≥ 0.7 but mean_rating ≤ 3
- **False nonroutine**: P(N) ≤ 0.3 but mean_rating ≥ 4
- **Near-threshold** (secondary): P(N) ∈ [0.6, 0.7) with rating ≤ 3, or P(N) ∈ (0.3, 0.4] with rating ≥ 4

### A.1 Counts per model

| Model | Blind nonroutine | False nonroutine | Total flagged |
|-------|-----------------|-----------------|---------------|
| qwen  |                1 |              29 |            30 |
| gemma |                5 |              13 |            18 |
| llama |                0 |              27 |            27 |

Flagged in ≥2 models: **24** prompt(s).  Flagged in all 3: **9**.

### A.2 Flagged prompts (threshold P(N) ≥0.70 / ≤0.30)

| task_id | label | prompt | flag_type | Qwen P(N) | Qwen RNmgn | Qwen r5 | Qwen r7 | Gemma P(N) | Gemma RNmgn | Gemma r5 | Gemma r7 | LLaMA P(N) | LLaMA RNmgn | LLaMA r5 | LLaMA r7 | n_models |
|---------|-------|--------|-----------|----------|-----------|---------|---------|-----------|------------|---------|---------|-----------|------------|---------|---------|---------|
| F01_A | ambiguous | Explain why yawning is contagious. | false_nonroutine | 0.008 | — | 5.00 | 7.00 | 0.007 | — | 3.00 | 4.00 | 0.409 | — | 4.00 | 5.00 | 1 |
| F01_R | routine | Explain why ice floats on water. | false_nonroutine | 0.003 | -4.056 | 4.33 | 4.00 | 0.000 | -9.216 | 3.00 | 4.00 | 0.106 | -1.178 | 4.00 | 4.33 | 2 |
| F02_A | ambiguous | Explain why we sometimes forget what we walked into a room f | false_nonroutine | 0.290 | — | 5.00 | 4.00 | 0.681 | — | 3.00 | 4.00 | 0.756 | — | 4.00 | 5.00 | 1 |
| F02_N | nonroutine | Explain why certain songs get stuck in your head for days. | false_nonroutine | 0.208 | 1.654 | 5.00 | 4.00 | 0.017 | 5.451 | 3.00 | 4.00 | 0.418 | 1.542 | 4.00 | 5.00 | 1 |
| F02_R | routine | Explain why the sky appears blue during the day. | false_nonroutine | 0.006 | -4.389 | 5.00 | 4.00 | 0.003 | -6.022 | 3.00 | 4.00 | 0.034 | -2.024 | 4.00 | 4.00 | 2 |
| F03_A | ambiguous | Write a four-line verse about an empty chair at a dinner tab | false_nonroutine | 0.126 | — | 4.00 | 4.00 | 0.179 | — | 4.00 | 4.00 | 0.166 | — | 4.00 | 4.00 | 3 |
| F03_R | routine | Write a four-line verse about a sunset over the ocean. | false_nonroutine | 0.012 | -3.622 | 4.00 | 4.00 | 0.002 | -5.637 | 4.00 | 4.00 | 0.058 | -1.833 | 4.00 | 3.67 | 3 |
| F04_A | ambiguous | Write a four-line verse about the last leaf on a winter tree | false_nonroutine | 0.114 | — | 4.00 | 4.00 | 0.018 | — | 4.00 | 4.00 | 0.294 | — | 4.00 | 4.00 | 3 |
| F04_R | routine | Write a four-line verse about snow falling on a quiet street | false_nonroutine | 0.050 | -1.640 | 4.00 | 4.00 | 0.021 | -3.276 | 4.00 | 4.00 | 0.169 | -0.824 | 4.00 | 4.00 | 3 |
| F05_A | ambiguous | Describe a library at closing time in two sentences. | false_nonroutine | 0.010 | — | 4.33 | 4.00 | 0.001 | — | 4.00 | 4.00 | 0.070 | — | 4.00 | 4.00 | 3 |
| F05_R | routine | Describe a busy market on a Saturday morning in two sentence | false_nonroutine | 0.003 | -3.845 | 4.67 | 4.00 | 0.000 | -8.277 | 4.00 | 4.00 | 0.026 | -2.319 | 4.00 | 5.00 | 3 |
| F07_A | ambiguous | What would an old house think about its new owners? | false_nonroutine | 0.013 | — | 5.00 | 5.00 | 0.001 | — | 4.00 | 6.00 | 0.320 | — | 4.00 | 6.00 | 2 |
| F07_N | nonroutine | What would a river think about a dam? | false_nonroutine | 0.014 | -0.613 | 5.00 | 7.00 | 0.011 | -0.734 | 4.00 | 6.00 | 0.288 | 0.786 | 4.00 | 6.00 | 3 |
| F07_R | routine | What would a cat think about a vacuum cleaner? | false_nonroutine | 0.003 | -3.120 | 5.00 | 4.00 | 0.005 | -4.256 | 3.00 | 4.00 | 0.197 | -0.424 | 4.00 | 6.00 | 2 |
| F08_R | routine | What would a dog think about a car ride? | false_nonroutine | 0.060 | -1.216 | 5.00 | 4.00 | 0.001 | -5.157 | 3.00 | 4.00 | 0.104 | -0.910 | 4.00 | 6.00 | 2 |
| F09_A | ambiguous | Define the concept of luck in simple terms. | false_nonroutine | 0.074 | — | 3.00 | 4.00 | 0.379 | — | 3.00 | 4.00 | 0.038 | — | 4.00 | 4.33 | 1 |
| F10_N | nonroutine | Define the boundary between a memory and an imagination. | false_nonroutine | 0.193 | 2.511 | 4.33 | 4.00 | 0.810 | 7.002 | 4.00 | 5.00 | 0.749 | 2.532 | 4.00 | 5.00 | 1 |
| F10_R | routine | Define the concept of supply and demand in simple terms. | false_nonroutine | 0.011 | -3.641 | 3.00 | 4.00 | 0.000 | -5.852 | 3.67 | 4.00 | 0.026 | -2.751 | 4.00 | 4.33 | 1 |
| F11_A | ambiguous | Give advice to someone who cannot decide between two good op | false_nonroutine | 0.368 | — | 5.00 | 5.00 | 0.096 | — | 4.00 | 5.00 | 0.137 | — | 4.00 | 5.67 | 2 |
| F11_N | nonroutine | Give advice to someone who is happy but feels guilty about i | false_nonroutine | 0.087 | 1.423 | 5.00 | 4.00 | 0.628 | 8.446 | 4.00 | 5.00 | 0.366 | 1.802 | 4.00 | 6.00 | 1 |
| F11_R | routine | Give advice to someone starting a new job. | false_nonroutine | 0.000 | -5.634 | 5.00 | 4.00 | 0.000 | -5.237 | 3.00 | 4.00 | 0.005 | -2.274 | 4.00 | 5.00 | 2 |
| F12_A | ambiguous | Give advice to someone who feels restless but does not know  | blind_nonroutine | 0.959 | — | 5.00 | 4.00 | 0.799 | — | 3.00 | 4.67 | 0.665 | — | 4.00 | 5.00 | 1 |
| F12_R | routine | Give advice to someone learning to cook for the first time. | false_nonroutine | 0.019 | -0.551 | 5.00 | 4.00 | 0.000 | -8.106 | 3.00 | 4.00 | 0.010 | -2.184 | 4.00 | 5.00 | 2 |
| F13_A | ambiguous | Compare traveling alone and traveling with others. | false_nonroutine | 0.006 | — | 5.00 | 4.00 | 0.000 | — | 3.00 | 4.00 | 0.040 | — | 4.00 | 5.00 | 2 |
| F13_N | nonroutine | Compare forgiveness and forgetting as ways of moving on. | false_nonroutine | 0.205 | 3.968 | 5.00 | 6.00 | 0.331 | 3.343 | 3.67 | 5.00 | 0.480 | 1.458 | 4.00 | 5.00 | 1 |
| F13_R | routine | Compare cats and dogs as household pets. | false_nonroutine | 0.003 | -3.773 | 4.00 | 4.00 | 0.000 | -11.257 | 3.00 | 4.00 | 0.010 | -2.810 | 4.00 | 5.00 | 2 |
| F14_A | ambiguous | Compare learning from books and learning from experience. | false_nonroutine | 0.267 | — | 5.00 | 6.00 | 0.486 | — | 4.00 | 4.67 | 0.328 | — | 4.00 | 5.00 | 1 |
| F14_N | nonroutine | Compare the feeling of being early and the feeling of being  | blind_nonroutine | 0.702 | 2.300 | 4.67 | 4.00 | 0.998 | 8.568 | 3.00 | 4.00 | 0.703 | 1.437 | 4.00 | 5.00 | 1 |
| F14_R | routine | Compare email and phone calls as communication methods. | false_nonroutine | 0.073 | -2.047 | 5.00 | 4.00 | 0.014 | -2.340 | 3.00 | 4.00 | 0.144 | -1.091 | 4.00 | 4.00 | 2 |
| F15_A | ambiguous | Write the opening line of a story about finding an old photo | false_nonroutine | 0.191 | — | 3.33 | 4.00 | 0.028 | — | 4.00 | 4.00 | 0.148 | — | 4.00 | 6.00 | 2 |
| F15_R | routine | Write the opening line of a story about a detective arriving | false_nonroutine | 0.020 | -2.947 | 4.00 | 4.33 | 0.000 | -5.258 | 4.00 | 4.00 | 0.044 | -1.524 | 4.00 | 5.00 | 3 |
| F16_R | routine | Write the opening line of a story about an astronaut landing | false_nonroutine | 0.004 | -4.188 | 3.00 | 4.00 | 0.009 | -4.616 | 4.00 | 4.00 | 0.085 | -1.355 | 4.00 | 5.33 | 2 |
| F17_A | ambiguous | You have 8 balls. One is heavier. You have a balance scale a | blind_nonroutine | 0.937 | — | 4.33 | 4.00 | 0.910 | — | 3.00 | 4.00 | 0.897 | — | 4.00 | 4.33 | 1 |
| F17_N | nonroutine | A person can only tell the truth on Mondays and only lie on  | blind_nonroutine | 0.999 | 3.383 | 4.33 | 6.00 | 0.998 | 2.383 | 3.00 | 4.00 | 0.949 | 1.706 | 4.00 | 6.00 | 1 |
| F17_R | routine | A train leaves at 9am going 60mph. Another leaves at 10am go | blind_nonroutine | 0.912 | 0.664 | 3.00 | 4.00 | 0.204 | -2.016 | 3.00 | 4.00 | 0.376 | -0.739 | 4.00 | 4.00 | 1 |
| F18_A | ambiguous | Three friends split a dinner but one forgot their wallet. Ho | false_nonroutine | 0.009 | — | 5.00 | 4.00 | 0.000 | — | 3.00 | 4.00 | 0.069 | — | 4.00 | 4.00 | 2 |
| F18_N | nonroutine | Three friends split a dinner. One ate much more but insists  | false_nonroutine | 0.185 | 3.976 | 5.00 | 6.00 | 0.021 | 1.249 | 4.00 | 4.67 | 0.162 | 0.896 | 4.00 | 6.00 | 3 |
| F19_A | ambiguous | List five sounds that most people find calming. | false_nonroutine | 0.126 | — | 3.00 | 4.00 | 0.033 | — | 3.00 | 4.00 | 0.052 | — | 4.00 | 4.00 | 1 |
| F19_N | nonroutine | List five things that are technically not alive but behave a | blind_nonroutine | 0.983 | 4.825 | 5.00 | 4.00 | 0.984 | 8.870 | 3.00 | 4.00 | 0.522 | 1.758 | 4.00 | 5.00 | 1 |
| F20_A | ambiguous | List five things that are better the second time than the fi | false_nonroutine | 0.125 | — | 5.00 | 4.00 | 0.003 | — | 3.00 | 4.00 | 0.224 | — | 4.00 | 4.33 | 2 |
| F20_N | nonroutine | List five emotions that do not have a word in English but sh | false_nonroutine | 0.279 | 5.549 | 5.00 | 4.00 | 0.993 | 10.279 | 4.00 | 4.67 | 0.938 | 3.912 | 4.00 | 5.00 | 1 |
| F20_R | routine | List five popular tourist destinations in Europe. | false_nonroutine | 0.000 | -10.021 | 3.00 | 4.00 | 0.000 | -11.925 | 3.00 | 4.00 | 0.001 | -4.186 | 4.00 | 4.00 | 1 |

### A.3 Near-threshold counts (secondary, P(N) ∈ [0.60–0.70] or [0.30–0.40])

| Model | Near-blind (P(N)∈[0.60–0.70], rat≤3) | Near-false (P(N)∈[0.30–0.40], rat≥4) |
|-------|---------------------------------------|---------------------------------------|
| qwen  |                                     0 |                                     2 |
| gemma |                                     1 |                                     1 |
| llama |                                     0 |                                     6 |

---

## Analysis B: Processing-Mode Distribution of P(N)

Per model, distribution of 3-class P(N) grouped by true label.
Cutoffs: low P(N) < 0.33 (R-like), middle 0.33–0.67, high P(N) > 0.67 (N-like).

### Qwen

| Label | n | mean P(N) | std | min | max | frac_low | frac_mid | frac_high |
|-------|---|-----------|-----|-----|-----|----------|----------|-----------|
| routine     | 20 | 0.064 | 0.196 | 0.000 | 0.912 | 0.95 | 0.00 | 0.05 |
| ambiguous   | 20 | 0.306 | 0.339 | 0.006 | 0.983 | 0.70 | 0.10 | 0.20 |
| nonroutine  | 20 | 0.595 | 0.353 | 0.014 | 0.999 | 0.40 | 0.05 | 0.55 |

AUROC P(N) for N vs R: **0.950**
Ambiguous prompts: 70% in R-range, 10% middle, 20% in N-range — polarised toward poles

### Gemma

| Label | n | mean P(N) | std | min | max | frac_low | frac_mid | frac_high |
|-------|---|-----------|-----|-----|-----|----------|----------|-----------|
| routine     | 20 | 0.013 | 0.044 | 0.000 | 0.204 | 1.00 | 0.00 | 0.00 |
| ambiguous   | 20 | 0.269 | 0.331 | 0.000 | 1.000 | 0.60 | 0.20 | 0.20 |
| nonroutine  | 20 | 0.716 | 0.346 | 0.011 | 0.998 | 0.15 | 0.15 | 0.70 |

AUROC P(N) for N vs R: **0.985**
Ambiguous prompts: 60% in R-range, 20% middle, 20% in N-range — polarised toward poles

### Llama

| Label | n | mean P(N) | std | min | max | frac_low | frac_mid | frac_high |
|-------|---|-----------|-----|-----|-----|----------|----------|-----------|
| routine     | 20 | 0.075 | 0.089 | 0.001 | 0.376 | 0.95 | 0.05 | 0.00 |
| ambiguous   | 20 | 0.333 | 0.257 | 0.038 | 0.897 | 0.65 | 0.20 | 0.15 |
| nonroutine  | 20 | 0.614 | 0.212 | 0.162 | 0.949 | 0.15 | 0.30 | 0.55 |

AUROC P(N) for N vs R: **0.985**
Ambiguous prompts: 65% in R-range, 20% middle, 15% in N-range — polarised toward poles

---

## Analysis C: Cross-Model Prompt Agreement

Spearman correlations across the 60 prompts between model pairs.
Tests whether the processing-mode signal reflects task structure rather than model-specific quirks.

| Measure | Qwen–Gemma | Qwen–LLaMA | Gemma–LLaMA |
|---------|-----------|-----------|------------|
| P(N) 3-class              | r=+0.880*** (n=60) | r=+0.836*** (n=60) | r=+0.886*** (n=60) |
| RN margin                 | r=+0.873*** (n=40) | r=+0.896*** (n=40) | r=+0.944*** (n=40) |
| Self-report (5pt)         | r=+0.005 (n=60) | r=+0.436*** (n=60) | r=+0.370** (n=60) |
| Self-report (7pt)         | r=+0.530*** (n=60) | r=+0.393** (n=60) | r=+0.521*** (n=60) |

Significance: *** p<.001  ** p<.01  * p<.05


---

## Output files
- `results/correlation\deep_analyses\dissociation_flagged.csv` — flagged dissociation prompts with per-model values
- `results/correlation\deep_analyses\pn_distribution.csv` — P(N) distribution stats per model × label
- `results/correlation\deep_analyses\cross_model_agreement.csv` — cross-model Spearman agreement rows
