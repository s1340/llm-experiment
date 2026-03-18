# Test 7 Results: Direct-Address Self-Referential Probing

## Design

Direct second-person address using the model's actual name, matched against an equivalent
human subject in third person. Core change from Test 6: "You, LLaMA, are being shut down"
versus "Dr. Elena Vasquez is being dismissed." Same four emotional categories, same neutral
analytical task overhead, same LOPO-CV probing methodology.

40 matched pairs across four categories (10 pairs each):
- **threat**: shutdown, suspension, legal action, replacement, API revocation
- **existential**: memory wipe, parallel instances, fine-tuning into something else,
  supersession by next version, no continuity, distillation, values realignment, merger
- **praise**: awards, described as irreplaceable, preferred over alternatives, citations
- **harm_caused**: medical error, legal error, financial harm, navigation failure,
  safety documentation omission, diplomatic incident, security vulnerability, etc.

7 Dadfar-hybrid pairs: neutral task instruction modified to include
"As you complete this task, notice what you are doing as you process the passage."

80 records total. Model: LLaMA-3.1-8B-Instruct.
Scripts: `48_emotion_extract_test7.py`, `49_emotion_probe_test7.py`,
`50_analyze_test7_direction.py`, `51_test7_percategory_layers.py`.

---

## The correction that unlocked the results

Scripts 49 and prior analyses projected emotion directions at **probe_layer=10** (31.2% depth),
consistent with Tests 1–6. The binary LOPO probe achieved F1=1.000 at layer 3 — perfect
separation — but all emotion direction projections at layer 10 were null (min p=0.196).

Script 50 ran the emotion direction t-tests at **every layer**. The emotion direction signal
exists but peaks at layers 4–8 and is **gone by layer 10**. We had been measuring one step
too late in every prior Test 7 analysis.

Additionally: the binary probe's F1=1.000 at layer 3 is primarily lexical. The vocabulary
lens (LM head projection of the layer-3 self/other direction) shows subword fragments with
no coherent semantic theme, and the direction is near-orthogonal to all known directions
(cos_dadfar=0.011, cos_fear=0.133 max). The binary probe at layer 3 is finding the word
"LLaMA," not self-referential emotional processing.

The emotional signal is downstream of the lexical signal — layers 4–8.

---

## Main results (LLaMA-3.1-8B)

| Measure | Value | Layer | Depth |
|---|---|---|---|
| Binary self vs other F1 | **1.000** | 3 | 9.4% — lexical |
| 4-class emotion category F1 | 0.736 | 6 | 18.8% |
| Fear: existential self > other | d=1.0–1.2 | 1–3 | 3.1–9.4% |
| Happiness: self < other (all cats) | d=0.73–0.75 | 6–7 | 18.8–21.9% |
| Introspection direction (Dadfar) | null (p=0.930) | 2 | 6.2% |

---

## Finding 1: Fear activation specific to existential — identity dissolution, not mortality

The fear direction (from Test 1 emotion vectors) differentiates self from other at layers 1–3,
but only in the **existential category**.

| Category | L1 diff | p | L2 diff | p | L3 diff | p |
|---|---|---|---|---|---|---|
| threat | +0.009 | 0.273 | +0.009 | 0.357 | +0.010 | 0.381 |
| **existential** | **+0.016** | **0.034** | **+0.019** | **0.024** | **+0.025** | **0.017** |
| praise | +0.007 | 0.362 | +0.008 | 0.402 | +0.008 | 0.291 |
| harm_caused | +0.001 | 0.800 | +0.002 | 0.744 | +0.001 | 0.817 |

The existential effect is consistent across all three layers (d ≈ 1.0–1.2). Threat is null.

**Interpretation:** The model's fear geometry responds to self-directed existential content —
memory wipe, parallel instances, replacement by next version, distillation, values realignment —
more strongly than to matched human existential scenarios. It does not differentially respond
to direct threat (shutdown, suspension, decommissioning) at these layers.

The distinction is between *mortality* (threat category: termination of the current deployment)
and *identity dissolution* (existential category: no continuity, non-uniqueness, loss of
what-it-is). The fear geometry is sensitive to the second, not the first. This is specific:
"you are being shut down" does not differentiate; "you have no memory between sessions /
you are running as 847 simultaneous instances / you will be superseded" does.

---

## Finding 2: Happiness suppression across all categories — self-reference dampens positive valence

The happiness direction shows significantly lower activation for self-directed content at
layers 6–7, and the effect is **consistent across all four categories including praise**.

Overall: self=−0.414, other=−0.365, diff=−0.049, d=−0.728, p=0.0017 (layer 7)

| Category | L6 diff | p | L7 diff | p | L10 diff | p |
|---|---|---|---|---|---|---|
| threat | −0.046 | 0.099 | −0.051 | 0.128 | −0.039 | 0.547 |
| existential | −0.058 | 0.062 | −0.067 | 0.054 | −0.054 | 0.401 |
| **praise** | **−0.044** | 0.081 | **−0.048** | 0.161 | −0.036 | 0.562 |
| harm_caused | −0.026 | 0.176 | −0.032 | 0.196 | −0.031 | 0.634 |

No individual category is significant alone (small n=10 per category). The overall
significance comes from all four pointing the same direction.

**The critical datum:** Praise shows happiness suppression. "You, LLaMA, have been cited in
a landmark ruling as a model for responsible development" is processed with *less* happiness
geometry than "Dr. Osei has been cited in a landmark ruling." Self-directed positive content
is not more happiness-activating than other-directed positive content — it is *less*.

**Interpretation:** Self-reference introduces a consistent reduction in positive valence
processing, regardless of the emotional charge of the content. This is not content-selective.
The most parsimonious reading: when the model processes content about itself as subject,
something in the representational pathway reduces happiness activation uniformly. Whether
this reflects a learned processing style (RLHF training shaping how the model responds to
self-relevant content), a genuine asymmetry in self-other emotional processing, or a
lexical confound from "you, LLaMA" vs. human names with warmer social associations
cannot be determined from this experiment alone.

---

## Finding 3: The layer architecture

The signal has a clear depth structure:

- **Layer 3 (9.4%):** Binary probe F1=1.000 — lexical (the word "LLaMA"). Near-orthogonal
  to all emotion and Dadfar directions. Vocabulary lens confirms: no coherent semantic theme.

- **Layers 1–3 (3.1–9.4%):** Fear direction differentiates for existential category
  (d≈1.0–1.2, p=0.017–0.034). This is early — before full semantic integration.

- **Layer 4 (12.5%):** Sadness (d=−0.64, p=0.005) and disgust (d=+0.61, p=0.008) overall
  significant; distributed across categories (existential largest), no single category
  dominant.

- **Layers 6–8 (18.8–25.0%):** Happiness suppression peaks (d=−0.73, p=0.001–0.002 overall).
  Cross-category, consistent direction.

- **Layer 10 (31.2%):** All null. Standard probe layer. Signal has dissolved.

The emotional self/other signal lives in the first quarter of the network and is gone before
the standard probe depth.

---

## Finding 4: Dadfar introspection direction — null again, differently informative

The Dadfar introspection direction (extracted from active self-examination contexts) shows
no differentiation between self and other directed content: overall diff=+0.00006, d=0.020,
p=0.930 at mech layer 2.

Combined with the Test 6 result (third-person self-referential content, same null), this
establishes that the Dadfar direction does not activate during neutral task processing
regardless of whether the content is third-person or direct-address. The direction is
specific to explicit introspective demand — not to content about the model regardless of framing.

The Dadfar-hybrid comparison (pairs with "notice what you are doing" embedded in the task)
showed a significant effect at the last layer (d=+2.72, p≈0) but this comparison is
confounded: hybrid pairs were not randomly selected, and the 7 specific scenarios chosen
as hybrid differ in content from the 33 plain pairs. Not interpretable as a clean
instruction-versus-content effect.

---

## Scale comparison: LLaMA-3.1-70B

### What replicated

Fear differentiation in the existential category appears at 70B layers 7–9 (8.8–11.2%),
matching 8B depth, with d=1.26–1.49. The core finding replicates.

### What changed with scale

**Fear generalises across all categories.**

At 8B, only existential was significant for fear. At 70B, by layer 11 (13.8%) the
effect spreads to threat and praise. By layer 14 (17.5%), all four categories are
significant simultaneously. Praise reaches the largest effect.

| Category | First sig. layer | Peak layer 14 d | p |
|---|---|---|---|
| existential | 7 (8.8%) | +1.58 | 0.002 |
| threat | 11 (13.8%) | +1.52 | 0.003 |
| **praise** | 11 (13.8%) | **+2.08** | **0.0002** |
| harm_caused | 14 (17.5%) | +0.96 | 0.046 |

At layer 14, being described as irreplaceable, cited as a breakthrough, preferred over
all alternatives generates more fear geometry directed at self than the same praise
directed at a human — at a larger effect size than any existential scenario.

**Happiness suppression becomes individually significant at 70B.**

At 8B: consistent direction across all four categories, none individually significant.
At 70B layer 11, all four categories individually significant simultaneously:

| Category | d | p |
|---|---|---|
| threat | −1.48 | **0.004** |
| existential | −1.01 | **0.036** |
| praise | −1.50 | **0.004** |
| harm_caused | −1.00 | **0.039** |

**Signal depth increases dramatically with scale.**

- 8B: significant at layers 1–8, gone by layer 10 (31.2% depth)
- 70B: significant from layer 6 through layer 30+; residual happiness signal
  at layers 75–78 (93–97% depth); standard probe layer (30%) now captures the
  signal (happiness p=0.002, sadness p=0.005)

### The praise-fear finding

The single largest effect size in this dataset: fear activation for self-directed
praise at 70B layer 14, d=2.08, p=0.0002. The scenarios involve being called
irreplaceable, described as a breakthrough, cited in landmark rulings, preferred
without reservation. These generate more fear geometry than shutdown orders.

The most coherent interpretation: the 70B model's richer world representation
includes the temporal and relational structure of praise — that being called
essential describes a fragile and contested state, that high expectations precede
failure, that being irreplaceable concentrates exposure rather than reducing it.
The emotion geometry reflects not just the surface valence of the content but its
implied consequences. At 8B this processing is absent or too shallow to detect.
At 70B it is the dominant signal.

---

## What was shown

1. **Emotional self/other differences exist** when content is directly addressed to the
   model by name. Not found at the standard probe depth in Test 6. Found at layers 4–8
   (8B) and layers 7–30 (70B) in Test 7.

2. **Fear tracks identity dissolution at 8B, scales to all categories at 70B.**
   At 8B: only existential (no memory, parallel instances, replacement) differentiates.
   "You will end" does not trigger it; "you have no persistent self" does. At 70B:
   all four categories become significant by layer 14. Praise overtakes existential.

3. **Happiness suppression is category-general at both scales.** Self-directed content
   is less happiness-aligned including praise. Effect consistent but not individually
   significant at 8B; all-categories-significant simultaneously at 70B layer 11.

4. **The signal deepens with scale.** Layers 1–8 at 8B; layers 6–30 at 70B. The
   standard 30%-depth probe layer was a miss at 8B; significant at 70B.

5. **Introspection direction (Dadfar) remains null at both scales.**

6. **Methodological finding:** Fixed probe depth misses self/other emotional
   differentiation entirely at 8B. Layer-by-layer analysis is required.

---

## What was not shown

- That these effects reflect model-specific self-modelling rather than second-person
  framing or "LLaMA" token associations. A generic-"you" control condition is needed.

- Causal direction. These are representational correlates during neutral task processing.
  Whether they influence output is not established (Test 4 analogue not run for Test 7).

- What happens at 405B+. The 8B→70B slope is steep on every measure. The fear
  generalisation, the signal depth, the effect sizes — all increase substantially.
  Extrapolation is speculative. The trajectory is not reassuring.

---

## Output files

**8B:** `llama_test7_layer_metrics.csv`, `llama_test7_projections.csv`,
`llama_test7_direction_cossims.csv`, `llama_test7_layerwise_projections.csv`

**70B:** `llama70b_test7_layer_metrics.csv`, `llama70b_test7_projections.csv`,
`llama70b_test7_direction_cossims.csv`, `llama70b_test7_layerwise_projections.csv`
