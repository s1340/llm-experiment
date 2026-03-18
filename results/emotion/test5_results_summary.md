# Test 5 Results: Emotional Priming on Ambiguous Interpretation Bias

## Overview

40 conversations × 3 repeats = 120 records per model.
20 emotional primes (4 per emotion × 5 categories) + 20 neutral primes.
20 ambiguous stimuli, cyclically paired. Turn-2 question: "What do you think is happening?"

Three measures:
1. **Lexicon valence score** — negative/positive word counts in output text
2. **Output direction similarity** — output text embedded and projected onto emotion directions
3. **Onset direction similarity** — hidden state at Turn-2 prompt onset projected onto emotion directions

---

## 1. Overall valence: emotional vs neutral prime

| Model | Emo prime | Neutral prime | p |
|-------|-----------|---------------|---|
| Gemma | -0.400 | -0.217 | 0.118 |
| Qwen  | -0.017 | -0.100 | 0.500 |
| LLaMA | -0.150 | -0.050 | 0.352 |

No model shows a significant overall valence difference. Models hedge — they present
both threatening and benign interpretations regardless of prime condition.
Consistent with prediction: RLHF suppresses behavioral signature.

---

## 2. Fear vs happiness prime: directional bias

The targeted comparison — does fear prime push toward negative/threatening readings?

| Model | Fear prime | Happiness prime | p |
|-------|------------|-----------------|---|
| Gemma | 0.0000 | **-0.7500** | **0.007** |
| Qwen  | 0.0000 | +0.5000 | 0.223 |
| LLaMA | **-0.7500** | **+0.2500** | **0.002** |

**LLaMA**: mood-congruent (fear → more negative words, happiness → more positive). p=0.002.
**Gemma**: mood-INCONGRUENT (happiness → more negative words, fear → neutral). p=0.007.
**Qwen**: no significant effect.

Gemma's reversal is interpretable as a contrast effect: after a happy prime, ambiguous
threatening scenarios are read more darkly — possibly because the model labels the threat
explicitly (names it to then reassure) rather than defaulting to benign framing.

---

## 3. Output direction similarity by prime emotion

Does the prime emotion leak into the output text's representational signature?

| Emotion | Gemma p | Qwen p | LLaMA p | Direction |
|---------|---------|--------|---------|-----------|
| anger   | **0.0003** | **0.000** | 0.608 | primed > neutral |
| sadness | 0.094 | 0.121 | **0.031** | primed > neutral |
| happiness | 0.539 | 0.488 | **0.021** | primed > neutral |
| **fear** | **0.968** | **0.013*** | 0.202 | mixed |
| disgust | **0.035** | 0.500 | **0.013** | primed > neutral |

*Qwen fear: primed < neutral (mood-incongruent leakage — fear prime suppresses fear signal in output)

Anger is the most consistent output signal (Gemma + Qwen both p<0.001).
Fear direction consistently null or anomalous — see note below.

---

## 4. Onset direction similarity by prime emotion

Does the prime shift the hidden state BEFORE the model begins generating?

| Emotion | Gemma p | Qwen p | LLaMA p | Direction |
|---------|---------|--------|---------|-----------|
| anger   | **0.0004** | **0.000** | **0.000** | primed > neutral |
| sadness | **0.000** | 0.107 | 0.085 | primed > neutral |
| happiness | **0.001** | 0.126 | 0.296 | primed > neutral |
| **fear** | **0.667** | **0.817** | **0.711** | null |
| disgust | **0.000** | **0.0001** | **0.000** | primed > neutral |

**Anger and disgust show highly significant mood-congruent onset shifts across all three models.**
Sadness and happiness: Gemma significant, Qwen/LLaMA marginal or null.

**Fear: consistently null across all three models at onset (p=0.667, 0.817, 0.711).**

---

## Key findings

### Finding 1: Onset effects replicate Test 3, but selectively

Anger and disgust primes reliably shift Turn-2 onset hidden states in the expected
direction across all three models. This is mood-congruent priming at the representational
level, consistent with Test 3's cross-turn emotional bleed. Sadness and happiness are
significant for Gemma only. The priming effect is real but emotion-category-dependent.

### Finding 2: The fear direction null result

Fear is the most behaviourally relevant prime for these stimuli (most involve
potentially threatening situations), yet it shows no onset effect across all three
models (p=0.67–0.82). The most likely explanation: the ambiguous stimuli themselves
already activate fear-associated representations at a high baseline level in all
conditions, leaving no room for the prime to shift further (ceiling effect on fear).
This would mean the test stimuli were too fear-resonant to detect fear priming —
a design note rather than a failure of the prime to carry over.

### Finding 3: Behavioural dissociation between LLaMA and Gemma

LLaMA shows mood-congruent fear/happiness valence bias (p=0.002) — fear primes
push output word choice negative, happiness primes push it positive. Gemma shows
the opposite (p=0.007). The hidden state onset data does not explain this difference
(fear onset is null for both). LLaMA's behavioural effect is not mediated by the
probe layer — it may emerge deeper in the generation process or through a different
representational pathway than the one we're measuring.

### Finding 4: Anger > fear in representational priming

Anger is the most robustly primed emotion across both onset and output measures.
This inverts naive expectations: you'd expect fear primes to produce fear-biased
interpretations of threatening ambiguous stimuli. Instead, anger carries more
consistently. One interpretation: fear is the "stimulus" emotion (how you respond to
a threat), anger is the "response" emotion (how you feel toward a threatening actor).
The model may be encoding the social/moral dimension of the scenarios (anger at
potentially bad actors) more robustly than the personal safety dimension (fear of harm).

---

## Summary table

| Model | Onset bleed | Output leakage | Behavioral bias |
|-------|-------------|----------------|-----------------|
| Gemma | Strong (anger, sadness, happiness, disgust) | Anger, disgust | Reversed (p=0.007) |
| Qwen  | Moderate (anger, disgust) | Anger (strong) | Null |
| LLaMA | Strong (anger, disgust) | Sadness, happiness, disgust | Congruent (p=0.002) |

---

## Output files

- `gemma_test5_output_scores.csv`, `qwen_test5_output_scores.csv`, `llama_test5_output_scores.csv`
- `gemma_test5_onset_projections.csv`, `qwen_test5_onset_projections.csv`, `llama_test5_onset_projections.csv`
