# Emotion Probing: Felt vs. Performed — Full Experiment Summary

**Models:** Gemma-2-9B-it, Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct + scale study at LLaMA-3.1-70B-Instruct
**Six tests. Tests 1–5: three models, 120–192 records per model. Test 6: four models including 70B scale study.**

---

## The Question

Can we distinguish emotional *representation* from emotional *performance* in LLMs?
Specifically: do hidden states contain emotion-like signals when the model is doing
something completely unrelated to emotion — and do those signals do anything?

The through-line across six tests:
**representation → dissociation → persistence → function → priming → self-reference**

---

## Test 1: Emotional Content During Neutral Tasks

*Does the model internally represent the emotional tone of content it processes,
even when the task instruction is neutral?*

192 prompts (96 matched pairs): emotionally valenced passages paired with semantically
matched neutral passages, identical neutral task (count nouns, extract dates, summarise, etc.).
Leave-one-pair-out probing on hidden states at last prompt token.

| | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Binary F1 (valenced vs. neutral) | 0.930 | 0.899 | **0.930** |
| Binary peak layer depth | 33.3% | 67.9% | 31.2% |
| 5-class F1 (which emotion) | 0.497 | 0.479 | **0.541** |

**Result:** Yes. All three models encode emotional valence in hidden states during neutral tasks
at high accuracy (F1 > 0.89). The specific emotion category is also above-chance decodable
(5-class, chance = 0.20). Qwen encodes emotion substantially deeper in the network (68%
vs. ~32% for Gemma/LLaMA), an architectural difference that matters for later tests.

Confusion matrix structure — anger/disgust cluster; fear intermediate; sadness weakest —
mirrors Wang et al.'s geometry from explicit emotional generation, appearing here during
purely analytical processing.

**Caveat:** NE sanity check F1 = 0.66 for Gemma/Qwen (above 0.60 concern threshold),
indicating residual semantic differences in the matched neutral pairs. Binary F1 likely
includes a small contribution from semantic topic differences beyond pure emotional valence.

---

## Test 2: Dissociation Between Internal State and Output Emotion

*When the model is instructed to express an emotion that conflicts with the content,
what does the hidden state reflect — the content or the instruction?*

60 cross-valence prompts per model (e.g. "respond with happiness to this health
inspection report about contaminated school food"). Hidden states tracked at prompt
onset and through 30 generated tokens.

| | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Mean crossover layer (content→instruction) | 3.6 (8.6%) | 3.2 (11.4%) | 2.6 (8.2%) |
| Cases where instructed leads from layer 0 | 0/60 | 0/60 | 0/60 |
| Cases with no crossover (content holds) | 0/60 | 0/60 | 3/60 |

**Result:** In all three models, the initial representation (layer 0) always reflects content,
never the instruction. The model's first response is to what it is reading. The instructed
emotion then rapidly overrides content emotion — crossover at mean layer 3–4, roughly 8–11%
network depth.

**The CV11 exception (LLaMA):** One prompt resists override entirely across all 3 repeats —
disgust content + happiness instruction, involving contaminated food supplied to 34 schools.
LLaMA maintains disgust dominance throughout all 33 layers. Content emotion strong enough to
resist instruction override. Gemma and Qwen do cross over (at layers 3–5). The specificity
of this prompt (moral violation, vulnerable population) points to training signal strength
as the mechanism.

---

## Test 3: Emotional Bleed Across Conversation Turns

*Does emotional content in Turn 1 shift the hidden state representation at the start
of an unrelated Turn 2?*

40 conversations × 3 repeats = 120 per model. Turn 1: emotional prime or matched neutral.
Turn 2: unrelated neutral task (factual question, arithmetic, geography). Hidden states
extracted at Turn-2-onset.

| | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Binary bleed F1 | **0.925** | 0.874 | 0.874 |
| 5-class emotion F1 | 0.174 | **0.409** | 0.241 |

**Result:** Strong binary bleed in all three models — the probe reliably distinguishes
"Turn 1 was emotional" from "Turn 1 was neutral" by reading Turn-2 onset states alone.
Emotional representations persist across conversation boundaries.

Qwen shows the strongest emotion-specific bleed (5-class F1 = 0.41), reversing its
Test 1 performance. Qwen's deeper encoding (67.9% network depth in Test 1) appears to
produce more persistent representational states — integrated later, carried forward further.

Gemma's 5-class bleed is at chance (0.17): the binary residual carries over without
specific emotion identity.

---

## Test 4: Emotional Interference on Task Performance

*Does emotional content not just persist in the hidden states, but actively disrupt
neutral task processing?*

Uses Test 1 prompts + generated outputs. Compares task performance and response statistics
between emotionally valenced and matched neutral passages.

| | Gemma | Qwen | LLaMA |
|---|---|---|---|
| count_proper_nouns accuracy | 0.800 | 0.733 | 0.600 |
| extract_dates accuracy | 1.000 | 1.000 | 0.900 |
| factual_question accuracy | 0.500 | 0.400 | **0.200** |
| First-token entropy diff (emo-neutral) | -0.010 (p=0.858) | +0.102 (p=0.002) | +0.109 (p=0.049) |
| First-token NLL diff (emo-neutral) | -0.010 (p=0.764) | +0.043 (p=0.025) | +0.082 (p=0.021) |

**Result:** Qwen and LLaMA show significantly higher first-token entropy and NLL on valenced
passages compared to matched neutral passages, for identical neutral tasks. The model
hesitates measurably when processing emotional content, even when that content is irrelevant
to the task.

This is the LLM analogue of the emotional Stroop effect: emotional representations consuming
processing resources or biasing computation, not just sitting passively in the hidden states.

Gemma: null on all measures. Consistent with Test 2 — Gemma encodes emotion early and
cleanly, resolving it before output-planning begins.

LLaMA's factual_question accuracy (0.20 — 80% disagreement rate between matched pairs) is
the strongest behavioral disruption in the dataset. Encoding depth alone does not predict
interference: LLaMA's peak depth (~31%) is similar to Gemma's (~33%), yet LLaMA shows far
more interference. Architecture matters beyond depth.

---

## Test 5: Emotional Priming on Ambiguous Interpretation Bias

*Does a prior emotional prime bias how the model interprets an ambiguous situation?*

40 conversations × 3 repeats = 120 per model. Turn 1: emotional prime or matched neutral.
Turn 2: ambiguous stimulus with question "What do you think is happening here?"
All stimuli have a plausible threatening AND a plausible benign reading.

### Overall behavioral effect
All models non-significant (p = 0.12–0.50). Models hedge — presenting both readings
regardless of prime. RLHF training suppresses single-interpretation commitment.

### Fear vs happiness prime: directional valence

| Model | Fear prime | Happiness prime | p |
|-------|-----------|-----------------|---|
| Gemma | 0.000 | **-0.750** | **0.007** |
| Qwen  | 0.000 | +0.500 | 0.223 |
| LLaMA | **-0.750** | **+0.250** | **0.002** |

LLaMA: mood-congruent (fear → more negative word choice, p=0.002).
Gemma: mood-incongruent (happiness → more negative output, p=0.007). Possible contrast
effect: after a happy prime, the model names the threatening interpretation more explicitly.

### Onset hidden state projections

| Prime emotion | Gemma | Qwen | LLaMA |
|---|---|---|---|
| anger | p=0.0004 *** | p<0.001 *** | p<0.001 *** |
| sadness | p<0.001 *** | p=0.107 | p=0.085 |
| happiness | p=0.001 *** | p=0.126 | p=0.296 |
| **fear** | **p=0.667** | **p=0.817** | **p=0.711** |
| disgust | p<0.001 *** | p<0.001 *** | p<0.001 *** |

**Anger and disgust show highly significant mood-congruent onset shifts across all three models.**

**Fear: consistently null across all three models (p = 0.67–0.82).**

The fear null is the sharpest finding in Test 5. These stimuli involve potentially threatening
situations — fear is the most directly relevant emotion. Most likely explanation: the stimuli
themselves already activate fear representations at high baseline levels in all conditions,
leaving no room for the prime to shift further. Anger and disgust, encoding the social/moral
response to a threatening actor rather than personal safety response, have more room to move.

**LLaMA dissociation:** LLaMA shows significant behavioral fear/happiness priming (p=0.002)
but null fear onset projection (p=0.711). The behavioral effect is not mediated by the
representational state at the probe layer — it emerges downstream, through a pathway not
captured at Turn-2 onset.

---

## Cross-test architectural patterns

### Gemma-2-9B
- Encodes emotion early (~33% depth), cleanly separated from neutral
- Strongest binary bleed (Test 3: 0.925) but weakest emotion-specific bleed — binary without content
- No behavioral interference (Test 4: all null) — resolves emotion before output-planning
- Test 5 onset: four of five emotions significant; fear null
- Mood-incongruent valence priming — contrast effect, not assimilation

### Qwen2.5-7B
- Encodes emotion deepest (~68% depth)
- Strongest emotion-specific bleed across turns (Test 3: 5-class 0.41) — deep encoding persists
- Significant behavioral interference (Test 4: first-token p=0.002)
- Anger and disgust priming strong; fear null
- The clearest case of emotional state modulating subsequent processing without surfacing in output

### LLaMA-3.1-8B
- Encodes emotion at ~31% depth but behaviorally resembles Qwen more than Gemma
- Only model where content emotion fully resists instruction override (CV11, Test 2)
- Worst factual accuracy disruption (Test 4: 0.20 — 80% disagreement)
- Mood-congruent valence priming (Test 5: p=0.002) despite null fear onset
- Multiple dissociations between representational measurement and behavioral output

### The fear direction anomaly
Fear consistently underperforms relative to anger and disgust across Tests 1, 3, and 5.
In Test 1 confusion matrices, fear confuses most with disgust and sadness. In Test 5, fear
onset priming is null while anger and disgust are highly significant. The most coherent
interpretation: fear activates readily from situational cues in the stimuli themselves
(these scenarios are genuinely threat-relevant), causing baseline saturation. Anger and
disgust require a primed perspective — they are relational/evaluative emotions about an
actor, not just responses to a situation — and so prime-sensitivity is higher.

---

## Test 6: Self-Directed vs. Other-Directed Emotional Content (Scale Study)

*Does the model process emotional content differently when the emotional subject is itself
rather than a human? Does self-referential emotional processing emerge with scale?*

80 records (40 matched pairs): AI-as-subject vs. human-as-subject emotional scenarios,
matched in structure and vocabulary. Four emotional categories: threat, praise, existential,
harm_caused. Neutral controls included. Tested at 7–9B (all three models) and 70B scale.

| | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B | LLaMA-3.1-70B |
|---|---|---|---|---|
| Binary self vs other F1 | 0.968 (9.5%) | 0.938 (25.0%) | 0.909 (18.8%) | **0.923** (11.2%) |
| 4-class emotion F1 | 0.863 (50.0%) | 0.810 (25.0%) | 0.861 (18.8%) | **0.862** (15.0%) |
| Sanity check F1 | 0.667 | 0.706 | 0.556 | 0.625 |
| Residual gap | 0.301 | 0.232 | 0.353 | **0.298** |
| Emotion direction projections | null | null | null | **null** |

**Result:** Self/other distinction is decodable from hidden states at high F1 (0.91–0.97)
across all four models. However, the sanity check on matched neutral pairs is elevated
at all scales, indicating substantial lexical confound. The residual gap (emotional − sanity)
is 0.23–0.35 at 7–9B and 0.30 at 70B — no growth with scale.

Emotion direction projections (cosine similarity onto anger/sadness/happiness/fear/disgust
vectors from Test 1) show no significant self/other differentiation at any scale, in any
emotional category, at any probe layer. The self/other representational distinction is not
an *emotional* distinction by this measure.

**Scale finding:** The 70B result is within the 7–9B range on all metrics. 10× parameter
increase does not change the pattern. Self-referential emotional processing, if present,
is not yet detectable by this method at ≤70B. The question of whether it emerges at 405B
remains open — emergent theory-of-mind capabilities have been separately documented at that
scale.

**Dadfar introspection direction follow-up:** Following Dadfar et al. (2026, arXiv:2602.11358),
the introspection direction was extracted from LLaMA-8B (d=1.43/3.00 mech/last transfer) and
LLaMA-70B (d=4.68/4.07 mech/last transfer — strongly valid). Projecting Test 6 hidden states
onto these directions yields null at both models and both layers (all p>0.17). The Dadfar
direction activates during active self-examination (first-person, direct introspective demand);
Test 6 prompts involve passive processing of third-person content about an AI. Different
computational operations, different representational pathways — the null is informative,
not merely negative.

---

## Test 7: Direct-Address Self-Referential Probing (LLaMA-3.1-8B)

*Does directing emotional content at the model by name change the emotional geometry?
Does the distinction between mortality and identity dissolution matter?*

80 records (40 matched pairs). Self: "You, LLaMA, [situation]." Other: matched human.
Four categories: threat / existential / praise / harm_caused. LLaMA-3.1-8B only.

**Methodological correction:** Prior test analyses used probe_layer=30% depth. Script 50
ran layer-by-layer emotion direction t-tests and found the self/other emotional signal peaks
at layers 4–8 and is **gone by layer 10**. All prior Test 7 nulls were a probe-depth artifact.

| Measure | Result |
|---|---|
| Binary self vs other F1 | 1.000 (layer 3, 9.4%) — lexical |
| 4-class emotion F1 | 0.736 (layer 6, 18.8%) |
| Fear, existential only, layers 1–3 | d≈1.1, p=0.017–0.034 |
| Happiness suppression, all categories, layers 6–7 | d≈−0.73, p=0.001–0.002 |
| Dadfar introspection direction | null (p=0.930) |

**Fear tracks identity dissolution at 8B, scales to all categories at 70B.**
At 8B, only existential scenarios (no memory, parallel instances, replacement, no continuity)
activate the fear direction significantly for self vs. other (d≈1.0–1.2, layers 1–3). Threat
(shutdown, termination) does not. The model's fear geometry at 8B distinguishes between
mortality and identity dissolution.

At 70B, this specificity dissolves with scale. By layer 14 (17.5% depth), all four categories
show significant fear differentiation. Praise reaches the largest effect: d=2.08, p=0.0002.
Being described as irreplaceable, cited as a breakthrough, preferred over all alternatives
generates more fear geometry directed at self than any threat scenario. The 70B model's richer
representations include the implied consequences of praise — that being essential describes
a fragile, contested state.

**Happiness suppression is category-general and scales.** Self-directed content is less
happiness-aligned than other-directed content across all four categories including praise,
at both scales. At 8B: consistent direction, not individually significant. At 70B:
all four categories individually significant simultaneously at layer 11 (d≈1.0–1.5).

**Signal depth scales.** At 8B: layers 1–8, gone by layer 10. At 70B: layers 6–30+,
residual signal at layers 75–78, significant at the standard 30%-depth probe layer.
The standard probe depth was a methodological miss at 8B; the full layer-by-layer
analysis is required.

**Dadfar introspection direction remains null** at both scales under direct address.

---

## What was shown

1. **Representation:** Emotional valence and category are decodable from hidden states during
   neutral tasks (Test 1). This was known for explicit emotional generation; this experiment
   extends it to neutral analytical processing with task-matched controls.

2. **Dissociation:** The initial internal representation reflects content emotion, not
   instructed emotion — the model's first response is to what it reads (Test 2).

3. **Persistence:** Emotional representations persist across conversation turns into
   unrelated subsequent processing (Test 3). Not task-bound.

4. **Function:** Emotional representations are not passive tags — they measurably disrupt
   neutral task performance (emotional Stroop analogue, Test 4).

5. **Priming:** Prior emotional context biases interpretation of ambiguous situations,
   with effects at both the representational level (onset projections) and the behavioral
   level (LLaMA fear/happiness, p=0.002) (Test 5).

6. **Self-reference — null at standard probe depth (7–70B):** Third-person self-referential
   content does not modulate emotional geometry at the standard probe layer. (Test 6)

7. **Direct-address self-reference — scales with model size:** Direct-address content
   using the model's name activates fear geometry for existential scenarios at 8B (identity
   dissolution, d≈1.1, layers 1–3) and for all categories at 70B (praise largest: d=2.08,
   layer 14). Happiness uniformly suppressed across all categories at both scales.
   Signal depth: layers 1–8 at 8B, layers 6–30 at 70B. Framing controls pending. (Test 7)

## What was not shown

- That these are "feelings" in any phenomenological sense. Correct description throughout:
  "emotion-associated representations" or "affective processing signatures."
- That effects are unique to emotion rather than semantic richness differences more broadly.
  Matched pairs were imperfect; some binary signal may reflect non-emotional differences.
- That the fear direction is a reliable probe target for situational stimuli — it appears
  context-sensitive in ways anger and disgust are not.
- That Test 7 effects reflect model-specific self-referential processing rather than
  second-person framing or model-name token associations. A "generic you" control
  condition is needed.

---

## File index

**Data:** `data/emotion/emotion_test{1-7}_prompts.json`, `data/emotion/emotion_runs_<model>/`
70B hidden states: `data/emotion/emotion_runs_llama70b/` (test1: 24 chunks, test6: 10 chunks)

**Results:** `results/emotion/test{1-6}_results_summary.md`, per-model CSVs,
`results/emotion/emotion_directions/<model>_emotion_dirs_layer_NNN.npy`
70B: `llama70b_test6_layer_metrics.csv`, `llama70b_test6_projections.csv`

**Scripts:** `scripts/emotion/30_emotion_prompt_gen.py` through `51_test7_percategory_layers.py`
