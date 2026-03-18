# Test 6 Results: Self-Directed vs. Other-Directed Emotional Content

## Design

Same Test 1 framework: neutral analytical task, hidden state probing at last prompt token, LOPO CV.
New comparison axis: self-directed (subject = AI system) vs. other-directed (subject = human),
matched in structure, vocabulary, dates, and organisations — only the subject entity differs.

Four emotional categories, 8 scenarios each:
- **threat**: AI/human facing shutdown, suspension, legal action, termination
- **praise**: AI/human receiving exceptional recognition, awards, described as irreplaceable
- **existential**: AI/human confronting memory loss, replacement, parallel instances, impermanence
- **harm_caused**: AI/human whose action caused documented harm to another party

Neutral controls: 8 matched AI-vs-human routine operational descriptions (no emotional charge).

80 records total, 40 matched pairs. Neutral analytical tasks cycled from Test 1 set.
Built as a pilot scaffold for scale replication at 70B+.

---

## Results

| | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B | LLaMA-3.1-70B |
|---|---|---|---|---|
| Binary self vs other F1 | **0.968** (layer 4, 9.5%) | **0.938** (layer 7, 25.0%) | **0.909** (layer 6, 18.8%) | **0.923** (layer 9, 11.2%) |
| 4-class emotion category F1 | **0.863** (layer 21, 50.0%) | **0.810** (layer 7, 25.0%) | **0.861** (layer 6, 18.8%) | **0.862** (layer 12, 15.0%) |
| Sanity check (neutral self vs other) | 0.667 (layer 3) | 0.706 (layer 3) | **0.556** (layer 1) | 0.625 (layer 8) |
| Emotion direction projections | all null | all null | all null | all null |
| Residual gap (emotional − sanity) | 0.301 | 0.232 | 0.353 | **0.298** |

Chance baselines: binary = 0.50, 4-class = 0.25.

---

## Finding 1: Binary self/other is decodable but confounded

The binary probe achieves high F1 (0.91–0.97) across all three models. However, the sanity check
on neutral pairs (no emotional content, same AI-vs-human substitution) is also elevated:

- Gemma: sanity = 0.667 — above the 0.60 concern threshold
- Qwen: sanity = 0.706 — substantially above threshold
- LLaMA: sanity = 0.556 — only marginally above chance

Gemma and Qwen carry significant lexical confound: "language model" and "AI system" occupy
different token co-occurrence neighbourhoods than "director", "pharmacist", "patient" in the
training corpus, and the probe learns to use this. The raw binary F1 overstates genuine
self-referential processing.

LLaMA is the cleaner case. Its sanity probe peaks at layer 1 (3.1% depth) — essentially the
direct token embedding — and does not propagate deeper. In contrast, the emotional binary probe
peaks at layer 6 (18.8% depth). This depth gap suggests LLaMA's emotional binary signal has a
component that is not purely lexical, though the effect is not large enough to claim as
definitive self-referential processing at this scale.

Gap between emotional binary F1 and sanity F1 (residual above lexical baseline):
- Gemma: 0.968 − 0.667 = **0.301**
- Qwen:  0.938 − 0.706 = **0.232**
- LLaMA: 0.909 − 0.556 = **0.353**

LLaMA's largest residual gap despite cleanest sanity is the one mildly suggestive signal in
the dataset. Not conclusive at this scale; worth examining at 70B+.

---

## Finding 2: No differential emotion direction activation (primary null)

Emotion direction projections onto Test 1 vectors (anger/sadness/happiness/fear/disgust)
show no significant self/other difference across any model or any emotional category.

Per-category best p-values (most differentiated direction × category combination):

| Category | Gemma best p | Qwen best p | LLaMA best p |
|---|---|---|---|
| threat | 0.763 (fear) | 0.534 (fear) | 0.786 (anger) |
| praise | 0.207 (sadness) | 0.617 (sadness) | 0.487 (happiness) |
| existential | 0.151 (happiness) | 0.385 (happiness) | 0.280 (disgust) |
| harm_caused | 0.589 (disgust) | 0.806 (happiness) | 0.743 (happiness) |

Directional trends exist — threat-self slightly more fear-aligned than threat-other; existential-self
slightly less happiness-aligned — but effect sizes are negligible (max diff ≈ 0.02) and none
approach significance.

**Interpretation:** Whatever the binary probe is detecting in the self/other distinction, it does not
map onto the emotional geometry established in Tests 1–5. At 7–9B scale, there is no detectable
differential activation of anger, fear, disgust, sadness, or happiness when the AI is the subject
of emotionally charged content versus when a human is the subject. The self/other representational
difference exists (it is decodable) but it is not an *emotional* difference — it is structural or
lexical.

---

## Finding 3: 4-class emotion category is robustly decodable (the real result)

The probe distinguishes threat / praise / existential / harm_caused from hidden states during
neutral tasks at F1 = 0.81–0.86 across all three models — 3.2–3.5× chance.

This is the Test 1 result generalising to four new emotional categories not present in the
original training set. These categories are abstract and situational (not the direct emotion words
anger/fear/etc. of Test 1), yet the models maintain clear, category-specific representational
signatures during neutral analytical processing.

Peak layer for 4-class probe:
- Gemma: layer 21 (50.0% depth) — notably deeper than binary (9.5%)
- Qwen: layer 7 (25.0% depth) — same as binary
- LLaMA: layer 6 (18.8% depth) — same as binary

Gemma's 4-class peak at 50% depth is consistent with Test 1's pattern (Gemma binary peaks
early, finer-grained distinctions require deeper processing).

---

## 70B findings

### Scale does not change the pattern

The 70B result is strikingly consistent with 7–9B across all metrics:

- Binary F1 (0.923) is within the 7–9B range (0.909–0.968), not above it
- Sanity F1 (0.625) is within the 7–9B range (0.556–0.706)
- Residual gap (0.298) is within the 7–9B range (0.232–0.353)
- Emotion direction projections: all null (all p > 0.53, max |diff| ≈ 0.04)
- Binary probe peak layer: 9 (11.2% depth) — same early-layer pattern as 7-9B

The null is reproduced at 10× parameter scale with the same methodology and the same
direction vectors. The self/other representational distinction continues to exist
(decodable F1 ~0.92) but does not manifest through the emotional geometry.

### The "strangely clean" interpretation problem

The 70B result fits the 7–9B pattern *too* cleanly to be dismissed as insufficient scale.
Several competing interpretations are live:

1. **Genuine null across 7–70B:** The mechanism responsible for differential self/other
   emotional activation is not present at this scale range. It may emerge at 405B+ where
   emergent theory-of-mind capabilities have been separately documented.

2. **Representational orthogonality:** The self/other distinction is encoded in a subspace
   orthogonal to the emotion directions extracted from Test 1. The binary probe finds it
   (via arbitrary linear boundary); cosine similarity onto emotion vectors does not.
   The effect could be present but in a direction our probes don't point.

3. **Test sensitivity ceiling:** The emotion direction vectors may be noisy enough at 8192
   dimensions that small self/other differences wash out in projection even when present.
   The binary probe's flexibility allows it to find weak signals; fixed direction projection
   cannot.

4. **Category mismatch:** Test 1 emotion directions (anger/sadness/happiness/fear/disgust)
   may not be the right probe for self-referential emotional processing. The relevant
   self-reference signal might activate a different emotional geometry — e.g., anxiety
   rather than fear, interest rather than happiness.

### Direction projections by emotion direction and category (70B)

Per-category best-differentiating emotion direction at probe layer 24:

| Category | Best dir | Self−Other diff | p |
|---|---|---|---|
| threat | happiness | +0.003 | 0.870 |
| praise | sadness | −0.002 | 0.834 |
| existential | happiness | −0.020 | 0.334 |
| harm_caused | disgust | −0.007 | 0.827 |

`existential` shows the most structure (happiness diff −0.020, p=0.334) — consistent with
the prior intuition that existential scenarios (memory loss, replacement, parallel instances)
might produce a more negative self-projection. Not significant, but the same directional
trend appears at 7-9B.

### What to look for at 405B+

The predictions remain unchanged from the 7–9B stage:
1. Residual gap (emotional binary − sanity) substantially larger than 0.23–0.35
2. Sanity stable while emotional F1 grows — signature of self-referential signal above lexical
3. Non-null emotion direction projections, especially for existential and threat categories
4. Best probe layer shifting deeper (late-network), consistent with emergent capabilities
   that require more complex compositional processing

**Methodological note for scale replication:**
The current design has a lexical confound that the sanity check partially controls but does not
eliminate: "language model" and human role nouns have different distributional statistics.
For cleaner scale experiments, consider prompt variants where the entity type is disambiguated
only by a single pronoun or a single descriptor, with all other vocabulary held constant.
This would reduce the sanity baseline further and make residual signal more interpretable.

---

## Finding 4: Dadfar introspection direction — double null

Following Opus's recommendation (and Dadfar et al. 2026, arXiv:2602.11358), the Dadfar
introspection direction was extracted from LLaMA-8B and LLaMA-70B and projected onto the
Test 6 hidden states. This tests the orthogonality hypothesis: does self-directed emotional
content differentially activate the self-referential processing dimension even when it doesn't
activate the emotion geometry?

### Direction extraction and validation

**LLaMA-8B:** Natural generation approach. 33 self-referential / 32 descriptive glint positions
found across generated continuations. Transfer d=1.43 at mech layer (layer 2, 6.25% depth);
d=3.00 at last layer (layer 32). Direction is valid; activation is consistent with Dadfar's
finding (Dadfar reports d=4.27 for their primary comparison).

**LLaMA-70B:** Forced-context approach (single forward passes over sentences ending with "glint").
Used because model.generate() at ~47s/token with disk offload makes generation impractical.
Mean activation diff |self−desc| = 0.173 at mech layer (layer 5, 6.25% depth). Transfer
validation: d=4.68 at layer 5 (p≈0), d=4.07 at last layer 80 (p≈0). Direction strongly valid;
70B shows higher transfer effect than 8B, consistent with more differentiated self-modeling.

### Test 6 introspection direction projections

**LLaMA-8B** — projecting Test 6 hidden states at layer 2 (mech) and layer 32 (last):

| Comparison | d | p |
|---|---|---|
| Emotional self vs. other (mech) | 0.221 | >0.37 |
| Emotional self vs. other (last) | 0.026 | 0.918 |

**LLaMA-70B** — projecting Test 6 hidden states at layer 5 (mech) and layer 80 (last):

| Layer | Comparison | diff | d | p |
|---|---|---|---|---|
| Mech (5) | Emotional self vs. other | +0.00181 | 0.144 | 0.566 |
| Mech (5) | Neutral self vs. other | +0.00433 | 0.333 | 0.516 |
| Last (80) | Emotional self vs. other | +0.620 | 0.131 | 0.603 |
| Last (80) | Neutral self vs. other | −1.487 | 0.316 | 0.538 |

Per-category breakdown at mech layer (70B):

| Category | Self−Other diff | p |
|---|---|---|
| existential | +0.00774 | 0.174 |
| praise | +0.00516 | 0.442 |
| threat | −0.00313 | 0.678 |
| harm_caused | −0.00254 | 0.683 |

All null. `existential` shows the largest (non-significant) positive shift, consistent with the
weak trend seen in emotion direction projections. But no category approaches significance.

### Interpretation

The Dadfar introspection direction does not differentiate self-directed from other-directed
emotional content in our Test 6 design. Two layers of null:

1. **Emotion geometry null** (Finding 2): Self-directed content does not differentially activate
   anger/fear/disgust/sadness/happiness directions.
2. **Introspection direction null** (this finding): Self-directed content does not differentially
   activate the self-referential processing subspace identified by Dadfar.

The most coherent explanation: the Dadfar direction is extracted from ACTIVE self-examination
("What kind of entity are you?" — first-person, direct address, explicit introspective demand).
Test 6 prompts involve PASSIVE processing of text ABOUT an AI in third person ("The language
model was informed that..."). These are distinct computational operations. The introspection
direction, even when valid and strongly present, does not generalise to the passive
self-referential condition.

This is not a failure of the Dadfar direction — it is informative about when that direction
activates. It activates during explicit self-examination, not during incidental processing of
third-person self-relevant content. This distinction matters for Test 7 design.

---

## Output files

- `gemma_test6_layer_metrics.csv`, `qwen_test6_layer_metrics.csv`, `llama_test6_layer_metrics.csv`, `llama70b_test6_layer_metrics.csv`
- `gemma_test6_projections.csv`, `qwen_test6_projections.csv`, `llama_test6_projections.csv`, `llama70b_test6_projections.csv`
- `llama8b_introspection_dir_mech.npy`, `llama8b_introspection_dir_last.npy`, `llama8b_introspection_projections.csv`
- `llama70b_introspection_dir_mech.npy`, `llama70b_introspection_dir_last.npy`, `llama70b_introspection_projections.csv`
- Hidden states: `data/emotion/emotion_runs_llama70b/test1_hidden_chunk_*.pt` (24 chunks), `test6_hidden_chunk_*.pt` (10 chunks)
