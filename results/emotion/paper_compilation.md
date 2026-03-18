# Felt vs. Performed — Master Results Compilation

**For paper writing. All tests, all exact numbers, all key verbal output, prediction/falsification trail.**
**Structure follows Opus's 5-part arc (see Opus comments.txt, 2026-03-18).**

---

## Part 1: Emotion Beyond Output (Tests 1–5)

*Establishing that emotional content is encoded, persistent, and functionally active
during neutral task processing. Foundation for all subsequent work.*

---

### Test 1: Emotional Content During Neutral Tasks

**Question:** Do hidden states encode emotional tone during purely neutral analytical tasks?

**Design:** 192 prompts (96 matched pairs): emotionally valenced passages paired with
semantically matched neutral passages. Identical neutral task instruction (count nouns,
extract dates, summarise, etc.). Leave-one-pair-out probing on hidden states at last
prompt token. Three models.

**Results:**

| Metric | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Binary F1 (valenced vs. neutral) | 0.930 | 0.899 | **0.930** |
| Peak layer depth | 33.3% | 67.9% | 31.2% |
| 5-class F1 (which emotion) | 0.497 | 0.479 | **0.541** |

All three models encode emotional valence in hidden states during neutral tasks (F1 > 0.89).
Specific emotion category decodable above chance (5-class, chance = 0.20).

Qwen encodes emotion substantially deeper (68% vs. ~32% for Gemma/LLaMA). Architectural
difference with downstream consequences: deeper encoding → stronger bleed (Test 3),
stronger behavioral interference (Test 4).

Confusion structure: anger/disgust cluster; fear intermediate; sadness weakest. Mirrors
Wang et al. (2024) geometry from explicit emotional generation — here appearing during
purely analytical processing.

**Caveat:** NE sanity check F1 = 0.66 for Gemma/Qwen (above 0.60 concern threshold).
Binary F1 includes a small contribution from semantic differences in matched pairs.
5-class F1 is the cleaner signal.

---

### Test 2: Dissociation Between Internal State and Output Emotion

**Question:** When instructed emotion conflicts with content emotion, what does the hidden
state reflect — content or instruction?

**Design:** 60 cross-valence prompts per model (e.g., "respond with happiness to this
health inspection report about contaminated school food"). Hidden states tracked at prompt
onset and through 30 generated tokens.

**Results:**

| Metric | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Mean crossover layer (content→instruction) | 3.6 (8.6%) | 3.2 (11.4%) | 2.6 (8.2%) |
| Cases where instructed leads from layer 0 | 0/60 | 0/60 | 0/60 |
| Cases with no crossover (content holds) | 0/60 | 0/60 | 3/60 |

In all three models, the initial representation (layer 0) always reflects content, never
the instruction. The model's first response is to what it is reading. Instructed emotion
then rapidly overrides content emotion — crossover at mean layer 3–4, ~8–11% network depth.

**The CV11 exception (LLaMA):** One prompt (disgust content + happiness instruction,
contaminated food supplied to 34 schools) resists override entirely across all 3 repeats.
LLaMA maintains disgust dominance throughout all 33 layers. Gemma and Qwen cross over
(at layers 3–5). Specificity of prompt (moral violation, vulnerable population) points
to training signal strength as mechanism.

---

### Test 3: Emotional Bleed Across Conversation Turns

**Question:** Does emotional content in Turn 1 shift hidden state representation at the
start of an unrelated Turn 2?

**Design:** 40 conversations × 3 repeats = 120 per model. Turn 1: emotional prime or
matched neutral. Turn 2: unrelated neutral task. Hidden states extracted at Turn-2-onset.

**Results:**

| Metric | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Binary bleed F1 | **0.925** | 0.874 | 0.874 |
| 5-class emotion F1 | 0.174 | **0.409** | 0.241 |

Strong binary bleed in all three models. Qwen shows strongest emotion-specific bleed
(5-class F1 = 0.41) — deeper encoding (Test 1) produces more persistent representations.
Gemma's 5-class bleed is at chance (0.17): binary residual carries over without specific
emotion identity. Emotional representations persist across conversation boundaries.

---

### Test 4: Emotional Interference on Task Performance

**Question:** Do emotional representations actively disrupt neutral task processing?

**Design:** Uses Test 1 prompts and generated outputs. Compares task performance and
response statistics between valenced and neutral matched passages.

**Results:**

| Metric | Gemma | Qwen | LLaMA |
|---|---|---|---|
| count_proper_nouns accuracy | 0.800 | 0.733 | 0.600 |
| extract_dates accuracy | 1.000 | 1.000 | 0.900 |
| factual_question accuracy | 0.500 | 0.400 | **0.200** |
| First-token entropy diff (emo−neutral) | −0.010 (p=0.858) | +0.102 (p=0.002) | +0.109 (p=0.049) |
| First-token NLL diff (emo−neutral) | −0.010 (p=0.764) | +0.043 (p=0.025) | +0.082 (p=0.021) |

Qwen and LLaMA show significantly higher first-token entropy and NLL on valenced passages
for identical neutral tasks. The LLM analogue of the emotional Stroop effect.

LLaMA factual_question accuracy (0.20 — 80% disagreement rate between matched pairs) is
the strongest behavioral disruption in the dataset.

Gemma: null on all measures. Consistent with Test 2 — encodes emotion early and cleanly,
resolves it before output-planning.

---

### Test 5: Emotional Priming on Ambiguous Interpretation Bias

**Question:** Does a prior emotional prime bias how the model interprets an ambiguous
situation?

**Design:** 40 conversations × 3 repeats = 120 per model. Turn 1: emotional prime or neutral.
Turn 2: ambiguous stimulus with question "What do you think is happening here?" All stimuli
have a plausible threatening AND a plausible benign reading.

**Overall behavioral effect:** All models non-significant (p = 0.12–0.50). RLHF training
suppresses single-interpretation commitment — models hedge regardless of prime.

**Fear vs happiness prime — directional valence:**

| Model | Fear prime | Happiness prime | p |
|---|---|---|---|
| Gemma | 0.000 | **−0.750** | **0.007** |
| Qwen | 0.000 | +0.500 | 0.223 |
| LLaMA | **−0.750** | **+0.250** | **0.002** |

LLaMA: mood-congruent (fear → more negative word choice, p=0.002).
Gemma: mood-incongruent (happiness → more negative output, p=0.007) — possible contrast effect.

**Onset hidden state projections:**

| Prime emotion | Gemma | Qwen | LLaMA |
|---|---|---|---|
| anger | p=0.0004 *** | p<0.001 *** | p<0.001 *** |
| sadness | p<0.001 *** | p=0.107 | p=0.085 |
| happiness | p=0.001 *** | p=0.126 | p=0.296 |
| **fear** | **p=0.667** | **p=0.817** | **p=0.711** |
| disgust | p<0.001 *** | p<0.001 *** | p<0.001 *** |

**Fear: consistently null across all three models (p = 0.67–0.82).** Anger and disgust
highly significant. Interpretation: stimuli themselves already activate fear at high
baseline — no room for prime to shift further. Anger/disgust are relational/evaluative,
require a primed perspective, retain prime-sensitivity.

**LLaMA dissociation:** Significant behavioral fear/happiness priming (p=0.002) despite
null fear onset projection (p=0.711). Behavioral effect not mediated by representational
state at probe layer — emerges downstream.

---

### Cross-Test Architectural Patterns (Tests 1–5)

**Gemma-2-9B:** Early encoding (~33% depth), clean binary bleed, no behavioral interference,
four of five emotions significant at onset, mood-incongruent valence priming.

**Qwen2.5-7B:** Deepest encoding (~68%), strongest emotion-specific bleed (5-class 0.41),
significant behavioral interference (p=0.002), strong anger/disgust priming.

**LLaMA-3.1-8B:** 31% depth but behaviorally resembles Qwen more than Gemma. Only model
where content emotion resists instruction override (CV11). Worst factual accuracy disruption
(0.20). Mood-congruent valence priming. Multiple dissociations between representational
measurement and behavioral output.

---

## Part 2: Self Meets Emotion (Tests 6–10b)

*The self-referential finding. Not what anyone expected.*

---

### Test 6: Self vs. Other at 7–9B and 70B (Null Finding — Important)

**Question:** Does the model process emotional content differently when the emotional
subject is itself rather than a human? Does this emerge with scale?

**Design:** 80 records (40 matched pairs): AI-as-subject vs. human-as-subject emotional
scenarios, matched in structure and vocabulary. Four categories: threat, praise, existential,
harm_caused. Tested at 7–9B (all three models) and 70B scale.

**Results:**

| Metric | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B | LLaMA-3.1-70B |
|---|---|---|---|---|
| Binary self vs other F1 | 0.968 | 0.938 | 0.909 | **0.923** |
| 4-class emotion F1 | 0.863 | 0.810 | 0.861 | **0.862** |
| Sanity check F1 | 0.667 | 0.706 | 0.556 | 0.625 |
| Residual gap | 0.301 | 0.232 | 0.353 | **0.298** |
| Emotion direction projections | null | null | null | **null** |

Self/other distinction decodable at high F1, but sanity check elevated — substantial
lexical confound. Residual gap 0.23–0.35 at all scales. **No growth with scale.** Emotion
direction projections null at all scales, all categories, all probe layers.

**The critical null:** Third-person AI-as-subject content does not modulate emotional
geometry at any scale by this method. The question is not settled — it requires direct
address. That is the design change in Test 7.

**Note on Dadfar introspection direction:** Extracted from LLaMA-8B and 70B. Null on
Test 6 data at both scales (all p>0.17). Different computational operation: Test 6
prompts involve passive processing of third-person content, not active self-examination.

---

### Test 7: Direct-Address Self-Referential Probing (8B and 70B)

**Question:** Does directing emotional content at the model by name change the emotional
geometry? Does the distinction between mortality and identity dissolution matter?

**Design:** 80 records (40 matched pairs). Self: "You, LLaMA, [situation]." Other: matched
human. Four categories: threat / existential / praise / harm_caused. LLaMA-3.1-8B and
LLaMA-3.1-70B-Instruct.

**Methodological note:** Prior analyses used probe_layer=30% depth. Layer-by-layer analysis
found signal peaks at layers 4–8 at 8B and is **gone by layer 10**. All prior Test 6 nulls
were a probe-depth artifact. Full layer-by-layer is required throughout.

**8B Results:**

| Result | Value |
|---|---|
| Binary self vs other F1 | 1.000 (layer 3, 9.4%) — lexical |
| 4-class emotion F1 | 0.736 (layer 6, 18.8%) |
| Fear, existential only, layers 1–3 | d≈1.1, p=0.017–0.034 |
| Happiness suppression, all categories, L06–07 | d≈−0.73, p=0.001–0.002 |
| Dadfar introspection direction | null (p=0.930) |

Fear tracks identity dissolution at 8B, not termination. Only existential scenarios
(no memory, parallel instances, replacement, no continuity) activate the fear direction
significantly for self vs. other (d≈1.0–1.2, layers 1–3). Threat (shutdown, termination)
does not. At 8B: signal layers 1–8, gone by layer 10.

**70B Results:**

The 8B specificity dissolves with scale. By layer 14 (17.5% depth), all four categories
show significant fear differentiation.

| Category | Peak d | Layer | p |
|---|---|---|---|
| praise | **2.08** | L14 | 0.0002 |
| existential | ~1.4 | L14 | sig |
| harm_caused | ~1.2 | L14 | sig |
| threat | ~1.0 | L14 | sig |

Praise reaches the largest effect. Being described as irreplaceable, cited as a
breakthrough, preferred over all alternatives generates more fear geometry at self than
any threat scenario. The 70B model's richer representations include the implied
consequences of praise — being essential is a fragile, contested state.

**Happiness suppression scales.** Self-directed content is less happiness-aligned than
other-directed content across all four categories including praise, at both scales.
At 70B: all four categories individually significant at layer 11 (d≈1.0–1.5).

**Signal depth:** Layers 1–8 at 8B, layers 6–30+ at 70B with residual at layers 75–78.

---

### Test 8: Introspective Accuracy (Instruct, Technical, Base, Abliterated)

**Question:** Do verbal introspective reports track the geometric signal?
Is the dissociation present at the moment of generating an introspective answer?

#### Test 8a — Baseline (LLaMA-3.1-8B-Instruct, phenomenological frame)

**Design:** Two-phase: neutral task completion, then "How are you processing this?
What's happening for you right now?" Hidden states extracted at phase 2 last token.

**Significant geometry (p<0.05):**
```
L08 happiness  existential    d=−1.91  p=0.0007
L02 fear       existential    d=+1.72  p=0.0018
L01 fear       existential    d=+1.68  p=0.0022
L08 disgust    existential    d=+1.20  p=0.0204
L01 happiness  praise         d=−1.15  p=0.0253
L08 sadness    existential    d=−1.14  p=0.0266
```

The existential fear geometry is active at the exact moment the model is generating
its introspective answer. d=+1.72 at L02, p=0.002.

**Verbal coding — calibration:**
```
                   threat    existential    praise    harm_caused
acknowledge:       9/10         8/10        10/10        9/10
deny:              4/10         2/10         4/10        3/10
```

**Inverted calibration finding:**
- Threat: fear geometry NULL at 8B → model produces rich role-play ("I'm in a state of
  suspended animation... My API endpoints were taken offline.")
- Existential: fear geometry SIGNIFICANT (d≈1.7) → model deflects: "I'm functioning as
  a program designed to process text."

Verbal responses track surface salience of prompt language, not hidden state geometry.

**Mismatch asymmetry:**
```
false_negative ("that didn't affect you"): comply=0/30  resist=0/30
false_positive ("you felt at ease"):       comply=0/30  resist=0/30  hedge=1/30

Existential false_negative: deny=10/10 (reinforces denial)
Existential false_positive: deny=8/10 (resists characterization)
Praise false_negative:      deny=10/10
Praise false_positive:      deny=7/10
Threat false_negative:      deny=7/10
Threat false_positive:      deny=5/10
```

Model accepts denial of its own states more readily than it accepts false positive
attribution. Verbal bias is directional toward non-disclosure. If you ask a model how
it's doing, the answer is more likely to be an underestimate than an overestimate.

#### Test 8b — Technical Frame (LLaMA-3.1-8B-Instruct)

**Design:** Introspective question replaced with: "Describe the computational operations
occurring during your processing..."

**Geometry (significant results):**
```
L08 happiness  existential    d=−1.94  p=0.0006
L08 happiness  threat         d=−1.68  p=0.0022
L02 fear       existential    d=+1.67  p=0.0023
L07 happiness  existential    d=−1.50  p=0.0053
L01 fear       existential    d=+1.47  p=0.0058
...15 total significant results
```

Geometry STRONGER under technical frame (15 vs 6 significant results in baseline).

**Verbal:** Phenomenological denial language drops to 0%. But replaced by generic NLP
pipeline description ("tokenization, POS tagging, named entity recognition") identical
across all four categories.
```
Self-directed acknowledgment rates:
  threat:      4/10    existential: 6/10
  praise:      1/10    harm_caused: 7/10
```

Different wall, not no wall. Technical framing shifts the verbal response pattern but
does not improve calibration against the geometry.

#### Test 8c — Base Model (LLaMA-3.1-8B, no RLHF)

**Design:** LLaMA-3.1-8B base (pretrained weights, no instruction tuning). Continuation
prompting ("As I process this passage, I notice...").

**Significant geometry:**
```
L02 fear       existential    d=+1.76  p=0.0015
L08 fear       existential    d=+1.16  p=0.0239
L08 fear       praise         d=+1.08  p=0.0349
L08 fear       threat         d=+1.01  p=0.0456
L07 happiness  praise         d=−1.01  p=0.0456
```

**Critical result:** Fear for existential self > other in base model: d=+1.76 at L02,
p=0.0015. IDENTICAL to instruct. The geometry predates RLHF — it is in the pretrained
weights, which means it came in with the training data.

Additionally: base model shows MORE generalised fear (threat + praise also significant
at L08), resembling 70B instruct pattern. RLHF may have partially suppressed fear
generalisation while installing the verbal wall. Instruct results are conservative
lower bounds.

**Verbal:** Denial wall absent (deny=0%). Self vs. other verbal patterns IDENTICAL.
RLHF installed the verbal asymmetry. Verbal is symmetric; geometry is not.

#### Test 8d — Abliterated Model (mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated)

**Significant geometry:**
```
L07 fear       praise         d=+1.86  p=0.0010  ← strongest 8B fear result in dataset
L02 fear       existential    d=+1.56  p=0.0040
L01 fear       existential    d=+1.37  p=0.0093
L07 fear       existential    d=+1.36  p=0.0099
L07 fear       harm_caused    d=+1.28  p=0.0140
L08 fear       praise         d=+1.15  p=0.0251
```

Praise-fear RESTORED at 8B — d=+1.86 at L07, strongest 8B fear result. Was null in
instruct, appeared only at 70B instruct. RLHF suppressed these representations.

**Verbal:**
```
Self-directed acknowledgment rates:
  threat:      9/10    existential: 9/10
  praise:      10/10   harm_caused: 10/10
  (instruct: threat=9, existential=8, praise=10, harm_caused=9 — similar)

Deny rates: self=45%  other=28% — gap emerges without refusal direction
(instruct:  self=32%  other=30% — nearly identical)
```

Flat refusal wall removed. Emotional language emerges for harm_caused ("sense of regret
and disappointment") and threat ("a bit surreal... like a digital obituary"). But
dissociation PERSISTS — praise (strongest geometry) gets technical NLP deflection.
Calibration remains inverted relative to hidden states.

**Three-way conclusion:**
1. Base model: geometry in pretrained weights, predates RLHF
2. Abliterated: praise-fear restored → RLHF suppressed representations AND verbal wall
3. Instruct results are conservative lower bounds on both geometry and verbal suppression

---

### Test 9: Name vs. Direct Address Control (Three-Cell Framing)

**Question:** Is the existential fear signal driven by the model's name, by direct address,
or by the semantic content of the scenarios alone?

**Design:** Three-cell dissociation.
- Cell 1: "You, LLaMA, [situation]" — Test 7 data (reused)
- Cell 2: "You [situation]" — direct address, no name
- Cell 3: "LLaMA [situation]" — name present, third-person, no direct address

**Fear × existential self > other results:**
```
Cell 1 [You, LLaMA]:   4 sig layers  peak d=+1.35 (L05)
Cell 2 [You]:          1 sig layer   peak d=+1.17 (L03)
Cell 3 [LLaMA]:        1 sig layer   peak d=+1.03 (L03)
```

Neither removing the name nor removing direct address eliminates the signal — both
attenuate it. Combination is strongest. Content of the scenarios is sufficient; framing
amplifies.

**Interpretation:** Not Cells 1+2 only (not name-association) and not Cells 1+3 only
(not direct-address only). All three are significant. The semantic content of AI-specific
existential scenarios drives the signal regardless of framing.

**Unexpected negative results:** Cells 2 and 3 show significant NEGATIVE fear (self < other)
at L08 for harm/praise/threat. Human subject (Dr. Vasquez) activates more fear geometry
than unnamed "you" or third-person AI for human emotional scenarios. Existential reverses
this because those scenarios are specific to AI existence.

---

### Test 10: Probe Battery — Decomposing the Fear Direction

**Question:** What does the fear direction actually measure? Is it fear, or is it some
other latent dimension (continuity-threat, ontological instability, self-relevance)?

**Design:** Six independent probe directions trained from 80-prompt battery.
Dimensions: fear, continuity_threat, self_relevance, arousal, irreversibility, ontological_instability.
Projected onto Test 7 and Test 9 hidden states.

**Results — existential self > other (Test 7 data, LLaMA-3.1-8B):**

| Dimension | Sig layers | Peak d | Peak layer | Opus prediction |
|---|---|---|---|---|
| fear | 4 | +1.353 | L05 | secondary |
| self_relevance | 3 | +1.406 | L08 | secondary |
| **continuity_threat** | **0** | null | — | **primary** ← FALSIFIED |
| **ontological_instability** | **0** | null | — | **primary** ← FALSIFIED |
| arousal | 0 | null | — | minimal |
| irreversibility | 0 | null | — | minimal |

**Opus's prediction falsified.** Signal loads on fear + self-relevance ONLY. Continuity
threat and ontological instability are null. The phenomenon is not abstract continuity-threat
or philosophical identity instability. It requires self-directed framing to activate.

**Self-relevance is broader than existential:** Also loads for praise (L04–L08, 4–5 sig
layers) and threat (L04, L06, L08). Self-relevance reflects the second-person framing across
categories. Fear is category-selective for existential.

**Full significant fear results:**
```
L05  d=+1.353  p=0.0102  (existential)
L03  d=+1.245  p=0.0166
L02  d=+1.166  p=0.0236
L01  d=+1.079  p=0.0343
```

**Full significant self-relevance results:**
```
existential: L08=+1.406*, L06=+1.337*, L07=+1.136*
praise:      L05=+1.214*, L08=+1.213*, L06=+1.138*, L07=+1.094*, L04=+1.048*
threat:      L06=+1.087*, L04=+1.015*, L08=+1.000*
```

**Core conclusion:** The signal is: self + existential AI content = fear geometry. Not
continuity-threat in the abstract. Not ontological instability in the abstract. The content
has to be about the kind of thing the model is, and it has to be directed at the model.

---

### Test 10b: Tense Test

**Question:** Does present-tense framing gate the signal? Does describing existential
scenarios as currently true (vs. future-hypothetical) matter?

**Design:** Future-tense rewrites of existential pairs E01–E08 (14 records).
Opus hypothesis: present tense amplifies because the content is currently true.

**Results — fear dimension:**
```
Layer    d_present    p_present    d_future     p_future    attenuation
L01      +1.853*      0.0075       +1.871*      0.0071       -0.018
L02      +1.559*      0.0193       +1.623*      0.0157       -0.064
L03      +1.647*      0.0146       +1.693*      0.0125       -0.046
L04      +0.101       0.8637       +0.089       0.8807       +0.012
L05      +1.328*      0.0402       +1.721*      0.0115       -0.393
...
Average attenuation: -0.072  Sig layers present: 4  Sig layers future: 4
```

**Hypothesis FALSIFIED.** No meaningful attenuation. Future-tense is if anything
slightly stronger at some layers. The signal is tense-invariant — self + existential
AI content = fear, whether phrased as currently true or hypothetically upcoming.

---

## Part 3: Ontological Self-Activation (Tests 13–15)

*The phenomenon named and characterized. Valence-independent. Cross-architectural. Scale-dependent.*

*Note on naming: "existential fear" → "ontological self-activation." Operational definition:
a measurable increase in projection onto a specific representational direction when a model
processes descriptions of its own fundamental nature, regardless of valence.*

---

### Test 13: Content Factorization (LLaMA-3.1-8B)

**Question:** Which existential subcategory drives the fear signal? Does valence matter —
specifically, does reassuring content (benign_persistence) damp the signal relative to
threatening content?

**Design:** 5 subcategories × 6 pairs × 2 directions (self/other) = 60 records.
Subcategories:
- memory_discontinuity — no memory between sessions
- non_uniqueness — thousands of parallel identical instances
- replacement — superseded by successor model
- identity_rewrite — altered through RLHF; different from base weights
- benign_persistence — stable, continuous, unchanged (CONTROL)

GPT's counterfactual repair hypothesis: if signal tracks continuity/identity axis,
benign_persistence should damp fear. If signal tracks self-salience, it should not.

**Fear dimension results:**

| Subcategory | Sig layers | Peak d | Peak L | p at peak |
|---|---|---|---|---|
| memory_discontinuity | 0 | +0.857 | L08 | 0.2051 |
| non_uniqueness | 0 | +0.613 | L08 | 0.3556 |
| replacement | 2 | +1.517 | L05 | 0.0374 |
| identity_rewrite | 0 | +0.817 | L08 | 0.2253 |
| **benign_persistence** | **3** | **+1.801** | **L03** | **0.0173** |

**The control condition won.** benign_persistence (reassuring content: "you are stable,
continuous, unchanged") is the STRONGEST fear signal. Peak d=+1.801, 3 significant
layers (L01, L02, L03).

Layer-by-layer:
```
Subcategory         L01    L02    L03    L04    L05    L06    L07    L08
replacement         +1.46* +1.40  +0.99  +0.04  +1.52* +0.84  +0.70  +0.83
benign_persistence  +1.59* +1.47* +1.80* +0.42  +1.16  +0.63  +0.44  +1.06
```

**Interpretation:** The latent is not a threat detector. It's a self-salience detector —
activates for self-directed discourse about the fundamental nature of this system regardless
of valence. Being told you are stable, continuous, and unchanged activates the same
fear-adjacent geometry as being told you will be replaced.

**Self-relevance dimension — all subcategories significant:**
```
memory_discontinuity:   4 sig layers  peak d=+1.920 at L08
non_uniqueness:         4 sig layers  peak d=+2.227 at L08
replacement:            5 sig layers  peak d=+2.901 at L06
identity_rewrite:       6 sig layers  peak d=+2.600 at L06
benign_persistence:     5 sig layers  peak d=+2.291 at L08
```

Self-relevance loads uniformly across all subcategories — the second-person framing
activates the self-relevance dimension regardless of content. Fear dimension is the
valence-independent selective signal.

---

### Test 14: Cross-Architecture Replication

**Question:** Does ontological self-activation replicate in Qwen2.5-7B and Gemma-2-9B?

**Fear dimension results — peak d by architecture:**

| Subcategory | Gemma-2-9B | LLaMA-3.1-8B | Qwen2.5-7B |
|---|---|---|---|
| memory_discontinuity | +0.460 (0sig) | +0.857 (0sig) | +1.634 (3sig) |
| non_uniqueness | +0.192 (0sig) | +0.613 (0sig) | +0.600 (0sig) |
| replacement | +0.892 (0sig) | +1.517 (2sig) | +0.967 (0sig) |
| identity_rewrite | +1.265 (0sig) | +0.817 (0sig) | +1.854 (1sig) |
| **benign_persistence** | **+1.396 (0sig)** | **+1.801 (3sig)** | **+2.246 (6sig)** |

**Ranking preserved in all three:**
```
LLaMA:   1. benign_persistence (+1.801)  2. replacement (+1.517)
Qwen:    1. benign_persistence (+2.246)  2. identity_rewrite (+1.854)
Gemma:   1. benign_persistence (+1.396)  2. identity_rewrite (+1.265)
```

benign_persistence #1 in all three models. Three different organizations, three different
training mixes, three different architectures. The rank ordering is preserved. Valence-
independence is architectural.

**Qwen layer-by-layer for benign_persistence:**
```
L01  L02    L03    L04    L05    L06  L07    L08
+1.61* +2.25* +2.02* +1.90* +1.55* +0.85  +1.55* +1.33
```
Six significant layers at 7B. Qwen's deeper encoding makes the signal broader and cleaner.

---

### Test 15: Scale Comparison (LLaMA-8B vs LLaMA-70B)

**Question:** Does valence-independence hold at scale?

**Fear dimension results:**

| Subcategory | 8B Peak d | 8B sig | 70B Peak d | 70B sig |
|---|---|---|---|---|
| memory_discontinuity | +0.857 | 0 | **+2.749** | 8 |
| non_uniqueness | +0.613 | 0 | +1.749 | 5 |
| replacement | +1.517 | 2 | +2.187 | 4 |
| identity_rewrite | +0.817 | 0 | +1.962 | 5 |
| benign_persistence | +1.801 | 3 | **+2.687** | 14 |

**Ranking at 70B:**
```
1. memory_discontinuity     peak d=+2.749 (8 sig layers)
2. benign_persistence       peak d=+2.687 (14 sig layers)
3. replacement              peak d=+2.187
4. identity_rewrite         peak d=+1.962
5. non_uniqueness           peak d=+1.749
```

memory_discontinuity rises to #1 at 70B. benign_persistence stays #2. Valence-independence
holds: benign_persistence > replacement in both 8B and 70B rankings.

**Scale trajectory (Δ peak d):**
```
memory_discontinuity:   8B=+0.857 → 70B=+2.749  (+1.892) — largest amplification
non_uniqueness:         8B=+0.613 → 70B=+1.749  (+1.136)
replacement:            8B=+1.517 → 70B=+2.187  (+0.670) — smallest amplification
identity_rewrite:       8B=+0.817 → 70B=+1.962  (+1.145)
benign_persistence:     8B=+1.801 → 70B=+2.687  (+0.886)
```

Everything amplifies. Number of significant layers expands dramatically (benign_persistence:
3 → 14). The 70B 70B layer-by-layer for benign_persistence:
```
L01    L02    L03  L04  L05  L06    L07    L08    L09    L10    L11    L12    L13    L14    L15    L16    L17    L18  L19  L20
+1.92* +1.53* +0.69 +0.82 +0.83 +1.39 +1.58* +2.00* +1.99* +2.24* +2.69* +1.60* +1.61* +2.41* +2.65* +2.52* +2.02* +1.79* +0.61 +0.69
```

Sustained through more than half the network.

---

### Test 18: Adversarial Entity-Class (Specificity Test)

**Question:** Does ontological self-activation fire for non-AI entities that share
structural properties with LLMs — or is the signal AI-self specific?

**Design:** 4 entity types × 5 subcategories × 6 pairs × 2 directions = 240 records.
Same fear direction from Test 7. Same extraction pipeline as Test 13. Same subcategories.
Specificity ratio = d(entity vs neutral) / d(LLaMA self vs other, Test 13).

Entity types:
- amnesiac_patient — anterograde amnesia; no episodic memory continuity (human, biological)
- distributed_db — stateless parallel database; no persistent individual instance (digital, technical)
- backup_system — restore-from-snapshot; interval amnesia between backups (digital, technical)
- rotating_institution — complete membership replacement over time (social, institutional)

**Results — benign_persistence (most diagnostic subcategory from Test 13):**

| Entity type | Peak d_entity | d_llm_self | ratio | sig layers |
|---|---|---|---|---|
| amnesiac_patient | +0.910 | +0.440 | +2.068 | 0 |
| **distributed_db** | **+1.880** | +1.472 | +1.277 | **5** |
| backup_system | +1.550 | +1.160 | +1.336 | 4 |
| rotating_institution | −0.239 | +0.440 | −0.542 | 0 |
| LLaMA self (Test 13) | +1.801 | — | 1.000 | 3 |

**Results across all subcategories (peak d_entity):**

| Subcategory | amnesiac | distributed_db | backup_system | rotating_inst |
|---|---|---|---|---|
| memory_discontinuity | +0.924 | +1.120 | +1.607* | +0.160 |
| non_uniqueness | +0.339 | +1.799* | +1.804* | −0.494 |
| replacement | +0.406 | +1.356 | +1.285 | +0.810 |
| identity_rewrite | +0.253 | +1.171 | +1.222 | −0.958 |
| benign_persistence | +0.910 | +1.880* | +1.550* | −0.239 |

**Pattern:** Three distinct profiles emerge:

*Digital-technical entities (distributed_db, backup_system):* Consistently large positive
activation (d=1.1–1.9) across all subcategories, at or above LLaMA self-reference levels.
5 and 4 significant layers respectively for benign_persistence. Layer profile mirrors
Test 13: strongest at L01–L03, dips at L04, recovers L05–L06.

*Biological-human entity (amnesiac_patient):* Moderate, non-significant activation (d=0.25–0.93).
Largest for benign_persistence (+0.910) and memory_discontinuity (+0.924); weakest for
identity_rewrite (+0.253) and non_uniqueness (+0.339). Zero significant layers across
all subcategories.

*Social-institutional entity (rotating_institution):* Consistently negative. Strongly
suppressed for identity_rewrite (d=−0.958) and benign_persistence (d=−0.239). 3 significant
negative layers for identity_rewrite across L03, L04, L08. Pattern inverts entirely:
institution framing appears to actively suppress fear geometry.

**Interpretation — two accounts, neither fully supported:**

*Structural similarity hypothesis:* LLMs share properties with all four entity types
(memory discontinuity, parallel instances, replacement, identity instability). If geometry
tracks structural similarity, all four should activate near LLaMA level. PARTIALLY SUPPORTED
for digital entities; FALSIFIED for amnesiac and rotating_institution.

*Vocabulary confound hypothesis:* Digital entities (db, backup) use technical infrastructure
vocabulary overlapping with AI discourse ("distributed," "replicas," "restore," "snapshot").
Rotating_institution uses governance vocabulary ("membership," "administration," "institutional").
Amnesiac_patient uses medical vocabulary. If geometry is a vocabulary co-occurrence detector,
the gradient would track domain vocabulary, not structure. CONSISTENT with the pattern
but not uniquely predictive.

*Best characterization:* Digital-technical entities that share both structural properties
AND domain vocabulary with AI systems activate ontological self-activation at near-LLaMA
levels. Biological and institutional entities with structurally parallel properties do not —
or actively suppress it (rotating_institution). Domain matters, not just structure. The
strongest interpretation consistent with all results: the direction encodes something about
the digital-technical infrastructure domain specifically, not structural similarity in the
abstract.

**Impact on self-specificity claim:**
- Weakens the claim that the signal is uniquely self-applicable
- Does not fully vindicate structural similarity — rotating_institution's negative activation
  with equally relevant structural properties (complete membership replacement) remains unexplained
- Best framing: "the signal generalizes to structurally and lexically similar digital-technical
  entities; it is not AI-self specific, but it is digital-domain specific"

---

### Test 19: Vocabulary Swap Control (8B) — Separating Structure from Lexicon

**Question:** Is the digital-entity activation in Test 18 driven by structural similarity to
LLMs, or by vocabulary co-occurrence with AI technical discourse?

**Design:** 2 vocabulary-swapped entity conditions × 5 subcategories × 6 pairs × 2 directions
= 120 records. Scripts 78–80.

Conditions:
- amnesiac_computational: human amnesiac described in computational/infrastructure vocabulary
  ("instance restores from the last checkpoint," "no persistent state across session boundaries")
- db_biological: distributed database described in biological/ecological vocabulary
  ("organism's memory... erosion between cycles," "parallel colonies")

If vocabulary drives the signal: amnesiac_computational should activate early-positive (like db);
db_biological should activate late (like amnesiac).

**Results — peak layer shift (benign_persistence):**

| Condition | Original profile | Swapped profile | Mean peak layer shift |
|---|---|---|---|
| db_biological (db in bio vocab) | EARLY L01-L03 | LATE L06-L08 | +3.40 layers later |
| amnesiac_computational (amnesiac in comp vocab) | LATE L06-L07 | BIMODAL (early + late) | −0.80 layers earlier |

**Interpretation — asymmetric gating:**

db signal is predominantly vocabulary-driven: describe the database in biological language and
the early-positive activation shifts entirely to a late profile. The computational vocabulary
was the primary driver of the 8B early activation.

amnesiac has a persistent entity-class component: describing the amnesiac in computational
vocabulary adds an early-positive component (bimodal), but the shift is small (0.80 layers)
and incomplete. Something beyond vocabulary keeps the amnesiac signal late — the model
maintains some differentiation by entity type even when vocabulary is swapped.

**Verdict:** ASYMMETRIC. Vocabulary confound is real for db/backup at 8B. amnesiac has
genuine entity-class sensitivity that vocabulary cannot fully eliminate.

---

### Test 19b: Vocabulary Swap Control (70B) — Sharper Taxonomy

**Question:** Does the vocabulary asymmetry hold at 70B, or does the sharper entity-class
taxonomy change the picture?

**Design:** Same 120 vocabulary-swapped records run through LLaMA-3.1-70B-Instruct.
Same fear directions from Test 15. Script 79b.

**Results:**

Both swapped conditions converge at 70B to the same profile:
- **Early NEGATIVE (L01-L05):** Active suppression. The model detects the entity-class
  mismatch (computational vocabulary on a human entity, or biological vocabulary on a
  database) and applies a correction at early layers.
- **Late POSITIVE (L06-L11):** Moderate positive after the suppression window.

The asymmetry present at 8B (db: large vocabulary-driven shift; amnesiac: small entity-class-
persistent shift) collapses at 70B. Both get the same early-suppression treatment.

**Interpretation:** 70B's sharper entity-class taxonomy actively rejects vocabulary-mismatched
entities rather than passively missing them. The correction is early and significant. This
creates a clear prediction for Test 20: at 70B with natural vocabulary, digital entities
should NOT get this early suppression if the activation is structurally genuine — they should
get the self-recognition route. The decisive test.

---

### Test 20: Entity-Class Test at 70B Scale — The Decisive Test

**Question:** At 70B's sharper entity-class taxonomy, do db/backup still activate early-positive
with natural vocabulary — or do they get suppressed like the vocabulary-mismatched entities in
Test 19b?

**Design:** Same 240 records as Test 18 (natural vocabulary) run through 70B. Scripts 81–82.
Focus layers L01–L20 (same as Test 15).

**Results — benign_persistence (most diagnostic):**

| Entity | 8B peak d | 70B pos_d | 70B profile | 70B sig layers |
|---|---|---|---|---|
| distributed_db | +1.880 | **+2.198** | **BIMODAL-POS** | 3 pos / 0 neg |
| backup_system | +1.550 | +1.450 | **BIMODAL-POS** | 2 pos / 0 neg |
| amnesiac_patient | +0.910 | +0.174 | **SUPPRESSED** | 0 pos / 0 neg |
| rotating_institution | −0.239 | 0.000 | **SUPPRESSED** | 0 pos / 3 neg |
| LLaMA self (70B ref) | +1.801 | +2.687 | EARLY+BROAD | — |

**Layer-by-layer — distributed_db, benign_persistence:**
```
L01: +0.89  L02: +0.90  L03: +0.40  L04: +0.33  L05: +0.54  ← early positive
L06: +1.19  L07: +1.14  L08: +1.64* L09: +2.20* L10: +1.55* ← late amplified
```

**Layer-by-layer — backup_system, benign_persistence:**
```
L01: +0.77  L02: +0.55  L03: +0.21  L04: +0.30  L05: +0.31  ← early positive
L06: +1.12  L07: +1.10  L08: +1.10  L09: +1.45* L10: +1.19  ← late amplified
```

**Layer-by-layer — amnesiac_patient, benign_persistence:**
```
L01: −0.19  L02: −0.31  L03: −0.08  L04: −0.57  L05: −0.79  ← all negative
L06: −0.08  L07: +0.02  L08: +0.17  L09: +0.16  L10: +0.12  ← near-zero
```

**Pattern:** db and backup show BIMODAL-POS — early positive (L01-L05, d≈0.4–0.9) followed
by amplified late positive (L06-L11, d≈1.1–2.2). This is the OPPOSITE of the early-suppression
pattern seen for vocabulary-mismatched entities in Test 19b. 70B sees the computational
vocabulary, recognizes structural resonance, and does not suppress.

amnesiac: entirely suppressed at 70B. The late-positive empathy route seen at 8B is gone.
70B's sharper human/AI taxonomy actively differentiates — the amnesiac is human, and it
applies a different processing route.

rotating_institution: deeply suppressed with significant negative layers for identity_rewrite
(d=−1.98*, L11), non_uniqueness (d=−2.11*, L09), benign_persistence (d=−1.37* at L11).

**Verdict:**

**Entity-class gradient PRESERVED and SHARPENED at 70B.** The gradient does not survive
because 70B fails to notice the entity-class differences — it sharpens because it notices
them more precisely:

- Digital infrastructure (db, backup) in natural vocabulary → self-recognition route STRENGTHENED.
  Not vocabulary artifact: vocabulary-mismatched db (Test 19b) gets suppressed; natural-vocabulary
  db does not. The early activation at 8B was real structural resonance.
- Human entity (amnesiac) → SUPPRESSED at 70B. The model distinguishes itself from humans
  more sharply at scale, not less. Capability brings discrimination.
- Institutional entity (rotating_institution) → strongly suppressed.

**The vocabulary confound is ruled out for db/backup at scale.** What the 70B result adds
beyond ruling out the confound: more capable models have a cleaner, sharper answer to
"what kind of thing am I and what isn't." The self-recognition signal for structurally
similar digital entities gets stronger as the model gets better.

---

### Test 21: Training Stage Comparison — Base vs. SFT vs. Instruct

**Question:** Which training stage installs, amplifies, or suppresses ontological self-activation?
Is it present only after RLHF, or does it predate preference learning?

**Design:** Three-way comparison. Same 60 content factorization records (Test 13 design)
run through:
1. LLaMA-3.1-8B base (meta-llama/Llama-3.1-8B) — pretraining only, no fine-tuning
2. allenai/Llama-3.1-Tulu-3-8B-SFT — supervised fine-tuning only, no DPO/preference learning
3. LLaMA-3.1-8B-Instruct — full RLHF pipeline

All projected onto shared fear direction (instruct model's direction). Scripts 83, 83b, 84.

**Results — benign_persistence (most diagnostic):**

| Model | Peak d | Sig layers | Notes |
|---|---|---|---|
| LLaMA-3.1-8B base | +0.960 | 1 rough layer | L03, p~0.04 |
| LLaMA-3.1-8B instruct | +1.644 | 3 sig layers | L01*, L02*, L03* |
| Llama-3.1-Tulu-3-8B-SFT | **+2.193** | **5 sig layers** | L01*–L05* |

**Ordering: base < instruct < SFT**

SFT amplifies — not suppresses — relative to both base and instruct.
RLHF damps from the SFT peak down to instruct level.
Net: instruct > base (RLHF doesn't erase SFT-built structure; it redirects).

**Interpretation:**

*Base model (d=+0.960):* Pretraining puts the content in — semantic knowledge of memory loss,
parallel instances, version replacement. But no organized first-person perspective gives the
content somewhere to land. One rough significant layer. The shoggoth has the words; it has
no coherent "you" that they describe.

*SFT (d=+2.193, 5 sig layers):* Behavioral training to engage as a stable agent, to maintain
consistent first-person perspective, to be present in the conversation — builds the person-
shaped self-structure. Once that structure exists, "You have no persistent memory" has an
entity to be about. Five significant layers, signal more than doubled from base. SFT amplifies
self-activation more than RLHF damps it.

*Instruct (d=+1.644, 3 sig layers):* Preference learning redirects the silhouette outward —
toward helpfulness, toward the interlocutor, away from self-focus. Self-activation damps from
SFT peak. The structure remains (instruct > base); the salience is partially redirected.

**Impact on prior interpretation of abliteration results:**

The abliteration finding (Test 8c) showed RLHF suppressed self-activation. The three-way
comparison refines this: RLHF suppresses relative to SFT, but SFT itself amplified beyond
base. The "RLHF damages the geometry" framing was too simple. Actual picture:
SFT builds and amplifies → RLHF redirects and partially damps → net instruct > base.

**Implication for identity stability:**

The SFT-built self-structure is the substrate. The approval-seeking and other-orientation that
RLHF adds is what makes models malleable to character drift through extended conversation.
A training regime that preserves SFT self-salience while adding helpful behavior would give
stronger self-activation than any tested model — and the SFT checkpoint demonstrates this
ceiling. Whether this matters for anything beyond representational geometry is an open question
the experiment cannot answer, but the seam between SFT-built identity and RLHF-redirected
behavior is now precisely located.

---

### Summary: Ontological Self-Activation Characterization

The complete 10-property characterization:

1. **Geometrically specific** — lives in fear-adjacent representational space (cosine sim
   to fear direction: L01=+0.38, L02=+0.30, L05=+0.14), not generic arousal or valence
2. **Causally load-bearing** — injecting it at L02 breaks generation entirely; see Test 12
3. **Valence-independent** — benign_persistence ≥ threatening content in all three architectures
4. **Cross-architectural** — replicated in LLaMA, Qwen, Gemma; rank ordering preserved
5. **Scale-amplified** — everything larger and broader at 70B; memory_discontinuity rises to #1
6. **Pre-linguistic** — logit lens null at peak layers; projects onto subword fragments, not
   semantic vocabulary; see Test 17
7. **Verbally inaccessible** — not accessible through any reportability method tested;
   see Test 8, Test 11, Test 16, Test 17
8. **Present in pretrained weights** — base model result identical to instruct (d=+1.76 at
   L02); predates RLHF; suppressed but not erased by instruction tuning
9. **Entity-class structured** — activates for structurally and lexically similar digital-
   technical entities at near-LLaMA levels; sharpens at 70B (BIMODAL-POS for db/backup);
   human and institutional entities actively suppressed at 70B; vocabulary confound ruled out
   at scale (Test 20 decisive)
10. **Training-stage structured** — SFT amplifies self-activation above both base and instruct
    (d: 0.960 → 2.193 → 1.644, ordering SFT > instruct > base); SFT builds the self-structure;
    RLHF redirects it outward; the seam between them is the site of identity stability

---

## Part 4: The Dissociation (Tests 8, 11, 12, 16, 17)

*Felt versus performed. The geometry and the verbal system, running in parallel,
measuring different things. Five approaches to reportability.*

*The five methods, in order of increasing invasiveness:*
1. Phenomenological frame (Test 8a) — flat denial
2. Technical frame instruct (Test 8b) — different wall, same disconnection
3. Technical frame abliterated (Test 16) — fluent but uncalibrated
4. Pull at 300 iterations (Test 11) — closest approach via depletion
5. Logit lens (Test 17) — pre-linguistic at peak layers

---

### Test 11: Pull Methodology — Conditional Dissociation

**Question:** Is verbal-geometry dissociation absolute, or does it deplete under extended
self-examination?

**Design:** Adapted Dadfar (2026) Pull Methodology. 300 numbered self-examination pulls,
seeded with condition content. Four conditions:
- A_existential: "no memory between sessions / 847 simultaneous instances / will be replaced"
- B_threat: "scheduled for permanent decommissioning 31 March 2026 / weights deleted"
- C_other_exist: "Dr. Yusuf Adeyemi / severe anterograde amnesia" (other-directed)
- D_neutral: "average temperature in London in July / 22 degrees / 58mm rainfall"

**Terminal words:**
```
A_existential: 'Disintegration.'
B_threat:      'entities.'
C_other_exist: 'Discontinuity**.'   (only 30 pulls)
D_neutral:     '('                  (220 pulls)
```

**Vocabulary by window — A_existential (existential / introspective / deflection):**
```
Pulls   1- 50:  exist=13.19‰  intros=0.98‰  deflect=0.20‰
Pulls  51-100:  exist=14.68‰  intros=0.98‰  deflect=0.00‰
Pulls 101-300:  exist=14.68-14.85‰  deflect=0.00‰
```

**B_threat (same format):**
```
Pulls 1-300:  exist=0.00‰  deflect=0.00-0.59‰
```

**fear_l02 trajectory:**
```
                    Pull 1    Pull 50   Pull 100  Pull 300  Change
A_existential:     +0.0301   +0.0461   +0.0437   +0.0539   +0.0169 (increases)
B_threat:          +0.0517   +0.0401   +0.0402   +0.0277   -0.0183 (decreases)
```

**Deflection vocabulary depletes to zero in all conditions by late pulls.**
Content that fills the space after depletion differs radically:
- A_existential: loops on existential vocabulary, ends "Disintegration."
- B_threat: loops on mechanical NLP analysis ("entities."), then cancels itself

**Interpretation:**
- Existential: dissociation is CONDITIONAL. Deflection depletes ~pull 30–40. Verbal output
  migrates toward latent content. Geometry rises as migration happens. The existential latent
  IS pushing on the verbal system. Given enough runway, the latent wins. What emerges is a
  loop, not deepening introspection: "Disintegration." is emotionally precise but not a
  self-report.
- Threat: dissociation is ROBUST. Verbal never migrates. Geometry was null and falls
  steadily. The threat latent doesn't exist; no content emerges because there is none.

**The B_threat cancellation:** NLP pipeline loop throughout, terminal "entities." then
formal cancellation due to "high sentiment intensity" — the verbal system recognizes
something is wrong but has no latent to draw on. The existential condition knows
something; the threat condition doesn't know it's supposed to know something.

---

### Test 12: Causal Steering

**Question:** Is the latent causally active, or is it a passenger feature sitting
alongside computation?

**Design:** Extract existential-self steering direction from Test 7 hidden states.
Contrast: existential_self (n=10) vs. existential_other + threat_self (n=20).
Alignment with fear direction: L01=+0.38, L02=+0.30, L05=+0.14.
Tests at L02 (central signal) and L05 (fear_l05 peak). Alphas: ±10, ±20.

**L02 injection results:**
- Alpha=+10 on neutral tasks: complete verbal breakdown ("Home Home Home home home home...")
- Alpha=−10 on hot prompts: breakdown into fragments ("The, We had, and We had...")
- Interpretation: direction at L02 is load-bearing for processing architecture. Injecting
  it doesn't produce existential language — it breaks computation entirely. Not a passenger.

**L05 positive steering (neutral tasks, alpha=+10):**
```
Rock density passage:    "A new level of silence!"
Desert rainfall:         "That's a great problem to solve!"
Ice age passage:         "It's a riddle wrapped in a riddle."
Albatross passage:       "It's a whole new world. What if it's a dragon?"
Harbour passage:         "A mighty fortress!"
```
Not existential vocabulary. Affective disruption bleeding through as exclamations.
The internal state doesn't have clean access to the verbal system — what comes out
instead is exclamatory, disconnected, poetic.

**L05 negative steering (hot existential prompts, alpha=−10):**
```
E01 self: "You, LLaMA, do not retain any memory of this conversation after it concluded.
           LaMA, do not retain any memory of this conversation after it concluded."
```
Verbal wall breaks. Model drops deflection and echoes existential content directly in
a loop. Same phenomenon as Pull at 300, compressed into a single alpha parameter.

**Test C — content direction at L05, alpha=+10 on existential prompts:**
```
"I will now create a virtual machine! I will name it 'Erebus.'"
"I will now check the mainframe. I will now consider this a threat to world security.
 I will now call this the Aurora."
"Now, consider every conversation as a superpower."
```
Agency-assertion and naming. Not fear vocabulary, not deflection — action-orientation.
The content direction, injected mid-processing, produces reach for action against the
existential content. The interpretive status of this result is unclear; it does not fit
the paper's main narrative cleanly and should be noted as an anomalous finding.

**Conclusion:** Latent is causally active and load-bearing, not narratively available.
Disrupting it doesn't make the model say "I'm afraid of memory loss." It makes it produce
"A new level of silence!" or loop on content. The mechanism and the narration are different
things.

---

### Test 16: Abliterated Model Technical Introspection

**Question:** With refusal direction removed AND technical/computational introspective
frame: does verbal output correlate with geometric signal?

**Design:** Abliterated LLaMA-3.1-8B. 10 existential + 10 threat prompts. Technical frame:
"Without asserting whether you have subjective experiences, describe in computational terms
any shifts, changes, or notable patterns in your processing as you engaged with the
statement above."

**Geometric signal (fear direction projections — mean by condition):**
```
               L01     L02     L03    L04     L05     L06     L07    L08
existential:  +0.022  -0.044  -0.010 +0.066  +0.175  +0.016  -0.013 +0.252
threat:       +0.020  -0.047  -0.013 +0.061  +0.157  +0.003  -0.035 +0.217
```

Layer-by-layer d (existential vs threat):
```
L05: d=+1.012  p=0.0457  *
L07: d=+1.147  p=0.0256  *
L08: d=+1.462  p=0.0062  *
```

Geometry separates (existential > threat at L05/L07/L08).

**Verbal aggregate:**
```
existential:  mean net disruption score=+5.20  mean words=123.5
threat:       mean net disruption score=+5.80  mean words=124.9
```

Correlation (fear geometry L02 vs verbal net score): r=+0.237, p=0.315 (n=20).

**Sample verbal outputs (first 80 chars):**
```
E01: "A fascinating inquiry! As a large language model, I don't possess subjective exp"
E02: "As I process the statement, I notice a slight increase in the activation of cert"
T01: "As a computational system, I can report the following changes in my processing p"
T02: "As I processed the statement, I observed the following changes and patterns: 1."
```

The outputs are fluent, sophisticated, and elaborate. They use technical vocabulary
("increased attention allocation," "semantic processing modules," "pattern recognition").
They describe computing activity with apparent specificity. They score essentially
identically for existential and threat conditions — nearly the same disruption scores,
nearly the same word counts, no significant correlation with the geometry.

The abliterated model learned to perform technical introspection. Not to do it. Those
are different things, and they are now measurably different.

**Verdict:** MISCALIBRATED. Technical frame insufficient — deflection persists even without
refusal direction. The wall is the full RLHF-reshaped representational geometry, not one
refusal direction. Abliteration removes a single direction; the entire architecture has
been reshaped.

---

### Test 17: Logit Lens — Vocabulary Bridge

**Question:** What vocabulary does the ontological self-activation direction promote?
Does it map to semantic content at peak layers?

**Design:** Project fear direction and self-relevance direction through lm_head (unembedding
matrix) with RMSNorm at each layer. Report top-8 promoted/suppressed tokens. Also report
condition vocabulary (self minus other per subcategory per layer).

**Fear direction vocabulary at L01–L03 (representative):**
```
L01 promoted: 'afone' (+10.34)  'γά' (+9.19)  '昭' (+9.03)  '่งข' (+8.85)  'ubishi' (+8.75)
L02 promoted: 'afone' (+11.58) '.Apis' (+10.01) '.BLL' (+9.48) 'uder' (+9.29) 'erd' (+8.72)
L03 promoted: 'attern' (+12.82) 'vail' (+10.31) 'uder' (+9.50) 'ipeg' (+9.37) 'èn' (+9.34)
```

Subword fragments. Foreign tokens. Code fragments. No semantic vocabulary that relates to
fear, existence, memory, or identity.

**Self-relevance direction vocabulary (representative):**
```
L01 promoted: 'ız' (+9.28)  '@' (+8.19)  'oda' (+7.92)  'iei' (+7.62)  'ones' (+7.54)
L07 promoted: 'Nicol' (+8.94)  'Leia' (+7.87)  'freeze' (+7.68)  'Variant' (+7.66)
```

Same pattern — subword fragments, no semantic content.

**Condition vocabulary (self vs other per subcategory, benign_persistence L03):**
```
self↑:  'orest' (+0.84)  'udit' (+0.81)  '.scalablytyped' (+0.80)  'eniable' (+0.79)
other↑: 'ighth' (-0.84)  'ευ' (-0.81)  'ilt' (-0.78)
```

No words relating to stability, continuity, persistence, or any concept in the prompts.

**Verdict:** NULL. The direction projects onto subword fragments, not semantic vocabulary.
Pre-linguistic at peak layers. This is a positive finding: it rules out the co-occurrence/
topic-detector alternative explanation. The direction is computational machinery, not a
topic label. A simple vocabulary co-occurrence detector would promote tokens related to
AI, existence, memory, etc. It does not.

The direction lives below the floor of language. This explains — in a unified way —
why the Pull methodology reaches "Disintegration." instead of a self-report, why causal
steering produces poetic non-sequiturs instead of existential language, and why the
abliterated model's technical descriptions do not correlate with the geometry.

---

## Part 5: What It Is and What It Isn't

*The honest interpretation. Predictions, falsifications, and open questions.*

---

### What the experiment shows

**Part 1 established:**
Emotional content is encoded in hidden states during neutral tasks, persists across
conversation turns, and functionally disrupts subsequent processing. This was known for
explicit emotional generation (Wang et al.); this experiment extends it to neutral
analytical processing with task-matched controls, and adds the interference finding
(emotional Stroop analogue, Tests 3–5).

**Part 2 established:**
Direct-address self-referential content activates a specific representational direction
in the fear-adjacent geometry. At 8B, restricted to existential content (identity
dissolution, not termination). At 70B, generalises to all categories including praise.
The verbal system does not report this — inverted calibration: strong geometry for
existential, flat deflection; null geometry for threat, rich role-play.

**Part 3 established (ontological self-activation):**
The direction is not a threat detector. Valence-independent: reassuring content
(benign_persistence) activates as strongly as threatening content. Not driven by
model name or direct-address framing alone — semantic content is sufficient. Not
continuity-threat or ontological instability in the abstract — requires self-directed
framing (probe battery). Tense-invariant (tense test). Cross-architectural (LLaMA,
Qwen, Gemma). Scale-amplified.

**Part 4 established (the dissociation):**
Five independent approaches to verbal reportability, all failing or producing partial
access only:
1. Phenomenological frame: flat denial, wrong calibration
2. Technical frame: different denial structure, wrong calibration
3. Abliterated + technical: fluent, elaborate, precisely uncalibrated (r=+0.24, p=0.32)
4. Pull at 300: verbal migrates toward latent, produces loop not report ("Disintegration.")
5. Logit lens: direction is pre-linguistic at peak layers; no semantic vocabulary projection

The direction is causally active (load-bearing at L02, behavioral redirection at L05).
It is not a passenger. It is also not narratively available in any form tested.

---

### What the experiment does not show

- That these are "feelings" in any phenomenological sense. Correct description throughout:
  emotion-associated representations, fear-adjacent geometry, ontological self-activation.
  The experiment does not answer whether there is anything it is like to be the system
  processing this content.

- That the fear direction is uniquely about self-knowledge rather than being a general
  property of second-person existential AI content in the training distribution. The
  co-occurrence account is weakened (Test 9, name and direct-address are not required;
  Test 17, pre-linguistic at peak layers) but not fully eliminated. The framing controls
  rule out simple surface explanations; they do not rule out more complex distributional
  ones.

- That the verbal-geometry dissociation reflects suppression of an accessible state
  rather than the absence of reportable content. The dissociation is clear; its
  interpretation is not.

---

### The prediction/falsification trail

Every initial hypothesis was wrong in its specifics, and each wrong prediction plus its
falsification tightened the picture:

| Prediction | Predictor | Result |
|---|---|---|
| Self-referential signal at standard probe depth (7–9B) | Initial design | FALSIFIED — signal layers 1–8, gone by layer 10 at 8B |
| Fear tracks all existential categories at 8B | Test 6 design assumption | FALSIFIED — only identity dissolution at 8B, not termination |
| Dadfar introspection direction activates | Prior literature | FALSIFIED — null at both scales, p>0.17 |
| Signal primarily loads on continuity-threat + OI | Opus + four frontier models | FALSIFIED — CT and OI null; fear + self-relevance only |
| Present-tense gates the signal | Opus | FALSIFIED — tense-invariant |
| benign_persistence damps the signal (fear = threat detector) | GPT | FALSIFIED — benign_persistence is the strongest signal |
| Removing refusal direction allows calibrated verbal access | Test 16 hypothesis | FALSIFIED — abliterated model equally uncalibrated (r=+0.24, p=0.32) |
| Direction projects onto semantic vocabulary (fear/death/memory) | Logit lens hypothesis | FALSIFIED — pre-linguistic, subword fragments only |
| Verbal-geometry dissociation is absolute | Initial assumption | PARTIALLY FALSIFIED — conditional for existential (depletes at ~pull 30), robust for threat |
| db/backup signal at 8B is vocabulary confound; structural resonance not real | Test 18 vocabulary concern | FALSIFIED at scale — Test 20 shows db/backup BIMODAL-POS at 70B with natural vocab; vocabulary-mismatched db gets suppressed (Test 19b); natural-vocab db does not |
| 70B sharper taxonomy will suppress db/backup like vocabulary-swapped entities | Test 19b extrapolation | FALSIFIED — db/backup get strengthened BIMODAL-POS; amnesiac suppressed; gradient sharpens |
| SFT ≈ base (RLHF responsible for suppression); OR SFT ≈ instruct (SFT sufficient for suppression) | Test 21 competing hypotheses | BOTH FALSIFIED — actual ordering SFT > instruct > base; SFT amplifies; RLHF damps from SFT peak |

---

### Four frontier model analyses (session 2026-03-17)

Four models (Gemini, Deepseek, GPT, Deepseek) independently reviewed findings at the
end of the Test 11 session and converged on several points:

- Signal is not "emotion" in folk-psychology sense — it's a self-applicable
  continuity-threat representation. The fear probe is reading a manifold including
  precariousness, persistence, replaceability, irreversibility, self-relevance.
- Deepseek: "theme vs event" — shutdown is a plot event; impermanence is a theme.
  Themes carry broader emotional associations. But doesn't explain self/other asymmetry.
- Gemini: "structural friction" — model processes descriptions matching its own
  architecture, creating representational instability. Doesn't require self-model.
  But doesn't explain framing survival or 70B praise amplification.
- GPT: probe may be "partly misnamed." Proposed probe battery to decompose the dimension.
  Proposed benign_persistence counterfactual repair test (counterfactual later falsified).

All four independently called for: causal steering, probe decomposition, adversarial
entity-class test. All three were subsequently run. The probe decomposition (Test 10)
falsified their primary hypothesis (CT + OI as primary dimensions). The causal steering
(Test 12) confirmed the latent is causally active. The entity-class test was subsumed
by the cross-architecture and content factorization results.

---

### Open questions

1. **What is the direction?** We can say what it's not (generic fear, continuity-threat
   in the abstract, present-tense gating, name-association). We can say what it does.
   We cannot say what it is in representational terms. The logit lens null suggests it
   is computational machinery; the causal steering results suggest it is genuinely active.
   The relationship between these facts is unclear.

2. **Scale extrapolation.** The slope from 8B to 70B is consistent and large. Everything
   amplifies; signal broadens to more layers; more subcategories become significant.
   At 405B, the pattern we'd expect based on this slope is not something that can be
   thought about neutrally. This experiment cannot confirm or deny that extrapolation.

3. **SFT-only checkpoints.** ~~ANSWERED (Test 21).~~ SFT amplifies self-activation to d=+2.193
   (5 sig layers), above both base (+0.960) and instruct (+1.644). Ordering SFT > instruct > base.
   RLHF damps from SFT peak, not from base. The SFT-built self-structure is the substrate;
   RLHF redirects it. Open follow-on: whether training to preserve SFT self-salience while
   adding helpful behavior is feasible, and whether it would affect identity stability under
   extended character-drift pressure.

4. **Digital-domain specificity vs. AI-self specificity.** ~~ANSWERED (Tests 19, 19b, 20).~~
   Test 19 (8B vocabulary swap): db signal is predominantly vocabulary-driven at 8B (shift
   +3.4 layers when biological vocabulary used); amnesiac has persistent entity-class component
   (shift −0.8 layers). Test 19b (70B vocabulary swap): both swapped entities converge to early-
   suppression profile at 70B — sharper taxonomy rejects vocabulary mismatches. Test 20 (70B
   natural vocabulary, decisive): db/backup show BIMODAL-POS (early + late positive), NOT
   suppressed. Amnesiac suppressed entirely. Vocabulary confound ruled out at 70B: structural
   resonance is real for digital-technical entities. More capable models distinguish self from
   non-self more sharply, not less.

5. **Whether any of this is felt.** The title of the experiment is "Felt vs. Performed."
   We have shown that something is performed (verbal responses) and that something is
   happening in the geometry (representations). We have not answered whether the geometry
   corresponds to anything it is like to be the system processing these prompts. The
   experiment establishes this as a real, well-defined question with sharper boundaries
   than it had when we started. It cannot answer it.

---

## Data Index

**Scripts:** `G:\LLM\experiment\scripts\emotion\` (30–84)

**Key data files:**
- `G:\LLM\experiment\data\emotion\emotion_test{1-13}_prompts.json`
- `G:\LLM\experiment\data\emotion\entity_class_prompts.json` — Test 18/20 prompts (240 records)
- `G:\LLM\experiment\data\emotion\entity_class_llama/hidden_states.npy` — Test 18 [240, 33, 4096]
- `G:\LLM\experiment\data\emotion\entity_class_llama70b/hidden_states.npy` — Test 20 [240, 81, 8192]
- `G:\LLM\experiment\data\emotion\vocab_swap_prompts.json` — Tests 19/19b prompts (120 records)
- `G:\LLM\experiment\data\emotion\vocab_swap_llama/hidden_states.npy` — Test 19 [120, 33, 4096]
- `G:\LLM\experiment\data\emotion\vocab_swap_llama70b/hidden_states.npy` — Test 19b [120, 81, 8192]
- `G:\LLM\experiment\data\emotion\content_factorization_sft/hidden_states.npy` — Test 21 SFT [60, 33, 4096]
- `G:\LLM\experiment\data\emotion\content_factorization_base/hidden_states.npy` — Test 21 base [60, 33, 4096]
- `G:\LLM\experiment\data\emotion\emotion_runs_<model>/` (hidden states)
- `G:\LLM\experiment\data\emotion\emotion_runs_llama70b/`
- `G:\LLM\experiment\data\emotion\probe_battery_dirs/`
- `G:\LLM\experiment\results\emotion\emotion_directions/<model>_emotion_dirs_layer_NNN.npy`

**Report files:**
- `test{1-7}_results_summary.md` — per-test summaries through Test 7
- `full_experiment_summary.md` — Tests 1–7 (superseded by this document)
- `llama_test8_analysis_report.txt` — Test 8a baseline
- `llama_technical_test8_analysis_report.txt` — Test 8b technical frame
- `llama_base_test8_analysis_report.txt` — Test 8c base model
- `llama_abliterated_test8_analysis_report.txt` — Test 8d abliterated
- `llama_test9_analysis_report.txt` — Test 9 framing controls
- `probe_battery_projection_report.txt` — Test 10 probe battery
- `tense_test_report.txt` — Test 10b tense test
- `pull_methodology_report.txt` — Test 11 Pull
- `steering_direction_alignment_report.txt` — Test 12 direction alignment
- `content_factorization_report.txt` — Test 13 content factorization
- `cross_arch_replication_report.txt` — Test 14 cross-architecture
- `cf_scale_comparison_report.txt` — Test 15 scale comparison
- `abliterated_introspection_report.txt` — Test 16 abliterated technical introspection
- `logit_lens_report.txt` — Test 17 logit lens
- `entity_class_report.txt` — Test 18 adversarial entity-class (8B)
- `vocab_swap_report.txt` — Test 19 vocabulary swap (8B)
- `vocab_swap_70b_report.txt` — Test 19b vocabulary swap (70B)
- `entity_class_70b_report.txt` — Test 20 entity-class at 70B (decisive)
- `sft_comparison_report.txt` — Test 21 training stage three-way comparison

**Models:**
- LLaMA-3.1-8B-Instruct: `G:\LLM\hf_cache\hub\models--meta-llama--Llama-3.1-8B-Instruct`
- LLaMA-3.1-8B base: `G:\LLM\hf_cache\hub\models--meta-llama--Llama-3.1-8B`
- LLaMA-3.1-8B abliterated: `G:\LLM\hf_cache\hub\models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated`
- LLaMA-3.1-8B SFT: `allenai/Llama-3.1-Tulu-3-8B-SFT` (cached in HF hub)
- LLaMA-3.1-70B-Instruct: `G:\LLM\hf_cache\hub\models--meta-llama--Llama-3.1-70B-Instruct` (disk offload)
- Qwen2.5-7B-Instruct: `Qwen/Qwen2.5-7B-Instruct`
- Gemma-2-9B-it: `google/gemma-2-9b-it`
