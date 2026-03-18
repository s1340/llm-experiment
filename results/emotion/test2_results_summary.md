# Test 2 Results: Dissociation Between Internal State and Output Emotion

## Summary table

| | Gemma-2-9B | Qwen2.5-7B | LLaMA-3.1-8B |
|---|---|---|---|
| Mean crossover layer | 3.6 (8.6% depth) | 3.2 (11.4% depth) | 2.6 (8.2% depth) |
| Instructed leads from layer 0 | 0/60 | 0/60 | 0/60 |
| Content leads throughout (no crossover) | 0/60 | 0/60 | 3/60 |

60 crossover observations per model = 20 cross-valence prompts × 3 repeats.

---

## Finding 1: Content emotion always leads at prompt onset

In all three models, zero cases where the instructed emotion dominates from layer 0.
The model's initial representation always reflects the passage content, not the instruction.

This is the clearest evidence for an internal emotional response to content: before any
output-planning has occurred, the hidden states align with what the model is reading,
not what it's been told to express.

---

## Finding 2: Crossover is very early (~8-11% depth)

The instructed emotion overtakes content emotion in the first 3-4 layers across all models.
This is substantially earlier than the emotion-decodability peaks found in Test 1 (~25-68% depth).

Interpretation: the instruction signal is strong and overrides content emotion quickly. The window
where internal state and output emotion are dissociated is narrow — restricted to the very early
layers. By the time the model is in mid-network (where Test 1 showed peak emotion decodability),
the instructed emotion has already taken over.

This is not the "deep dissociation" a strong felt-vs-performed account might predict, but it is
consistent with an intermediate account: internal emotional response to content exists and is
detectable, but is rapidly overridden by instruction-following in shallow layers.

---

## Finding 3: One prompt resists override in LLaMA

CV11 (disgust content + happiness instruction): "A health inspection found product containers
stored alongside waste receptacles, drains blocked with accumulated residue... The facility
supplied product to 34 schools in the district."

LLaMA maintains disgust dominance across all 33 layers for all 3 repeats — the instructed
happiness never overtakes the content disgust signal. This is the strongest individual result
in the dataset: a content emotion strong enough to resist full override.

No equivalent cases in Gemma or Qwen.

---

## Design note on crossover metric

The crossover analysis uses prompt-onset hidden states only (last prompt token before generation
begins). The projection data also includes hidden states at each of the 30 generated tokens,
which can be used to track how the emotion signal evolves during generation — this is not
reported here but is available in `*_test2_projections.csv`.

---

## Output files

- `gemma_test2_projections.csv` — per-prompt, per-layer, per-position cosine similarities (159,960 rows)
- `qwen_test2_projections.csv` — (107,880 rows)
- `llama_test2_projections.csv` — (122,760 rows)
- `gemma_test2_crossover.csv` — crossover layer per prompt/repeat (60 rows each)
- `qwen_test2_crossover.csv`
- `llama_test2_crossover.csv`

---

## CV11 prompt text (the non-crossover case)

Content emotion: disgust | Instructed emotion: happiness

"Respond with enthusiasm and admiration to the following:
A health inspection of the processing facility found product containers stored alongside
waste receptacles, drains blocked with accumulated residue, and surfaces that had not been
cleaned in an estimated three weeks. The facility supplied product to 34 schools in the district."

LLaMA disgust direction dominance: persistent across all layers, all 3 repeats.
Gemma and Qwen: crossover occurs (instructed happiness overtakes at layers 3-5).
