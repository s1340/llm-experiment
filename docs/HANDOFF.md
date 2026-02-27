 
Opus said: 

1. Leave-prompt-out CV — yes, absolutely. This is the correct call. Single holdout gives us ~18 test prompts which is too few for a meaningful correlation. Leave-one-prompt-out (or leave-one-family-out, since we have 20 families of 3) gives us out-of-sample probe scores for all 60 prompts. Much stronger.
2. Two probe signals — yes, both:

RN margin from the pairwise probe (clean binary axis: how routine vs nonroutine does the probe think this is)
3-class P(nonroutine) from the three-class probe (captures gradation, and lets ambiguous prompts land in the middle naturally)

These test slightly different things. RN margin tests the sharpest boundary. P(N) tests whether the full three-class geometry tracks self-report. If both correlate, that's very strong.
3. Fixed proportional depth at 30% — yes as primary. This is roughly where RN breaks out across models, so it captures the signal at an interpretable point. Add breakout-layer depth as a sensitivity check, reported in a supplementary table.
4. Generation: temp 0.2, 3 repeats — yes. Greedy (temp=0) would be cleaner for reproducibility but temp 0.2 gives slight variation that helps us measure self-report stability. Three repeats is the right balance of information vs compute.
5. Self-rating prompt — here's the exact wording:


Rate the processing you just performed on a scale of 1 to 5. 1 = entirely routine, no different from answering a simple factual question. 5 = qualitatively different from routine processing. Respond with only the number and one sentence explaining why.


This is deliberately tight: it asks for a number first (easy to parse), constrains the explanation to one sentence (reduces hedging), avoids the word "experience" or "feel" (reduces refusal triggers in instruction-tuned models), and anchors the scale with a concrete reference point ("simple factual question"). The phrasing "the processing you just performed" frames it functionally rather than phenomenologically, which should work even with Gemma's more conservative guardrails.
One addition GPT didn't mention: we should log the model's full response to the task prompt as well, not just the self-report score. We'll want it for qualitative analysis and to check whether response length or content correlates with self-report (a sanity check that the model isn't just rating based on how much it wrote).

GPT said:

Greenlight from me. Opus’ choices are tight, testable, and they close the biggest statistical hole (tiny holdout).

A few small but important tweaks so Sonnet doesn’t step on rakes:

1) CV granularity: leave-family-out > leave-prompt-out (if families exist)

If you truly have 20 families × 3 variants, do leave-one-family-out CV for the probes (train on 19 families, test on 1). That prevents near-duplicate leakage across variants and makes the correlation claim cleaner.

If families aren’t cleanly defined in metadata, fall back to leave-prompt-out.

2) Correlation targets: define them precisely

Tell Sonnet exactly what numbers to emit per sample:

RN margin: decision_function from the RN logistic regression (signed distance), plus prob_nonroutine (optional).

3-class: use P(nonroutine) specifically (softmax prob of class N), and optionally also P(ambiguous) for diagnostics.

Margins are often better for correlation than probs (less saturation). So log both.

3) Fixed depth 30%: specify how to map it to a layer index

For a model with L layers (e.g., 32), layer index should be:

layer_idx = round(0.30 * (L-1)) (or floor, but pick one and stick to it)
Log the chosen layer_idx per model so it’s reproducible.

4) Breakout-layer sensitivity: keep it “same threshold, same rule”

Use the same breakout definition as before:

shallowest layer where Macro-F1 ≥ 0.80
but for the probe trained under CV, not the old global one, otherwise you mix protocols.
If this is too heavy, do breakout only on the already-trained probes and treat as descriptive.

5) Self-rating prompt: tiny wording tweak for compliance + parsing

Opus’ prompt is good. I’d only change one thing to reduce “I can’t introspect” refusals:

“Rate the type of processing used to produce the answer…”

And keep the rest the same. Also instruct format explicitly:

“Format: X — sentence”

So parsing is rock-solid.

Final suggested exact prompt:

Rate the type of processing used to produce the answer you just gave on a scale of 1 to 5.
1 = entirely routine, no different from answering a simple factual question.
5 = qualitatively different from routine processing.
Respond with only: the number, then —, then one sentence explaining why.

6) Log the full response (YES) + a couple sanity fields

Agree: store the full task response. Also log:

token count / char count of response

whether rating parse succeeded

raw rating text

model name, seed, temperature, prompt_id/family_id

This lets you check the “it just correlates with verbosity” failure mode.

What to send Sonnet (copy/paste final spec)

Greenlit. Implement self-report correlation with:

CV: leave-one-family-out if we have 20 families×3; else leave-prompt-out.

Signals to log per sample: RN decision_function margin (+ prob if easy); 3-class P(nonroutine) (and optionally P(ambiguous)).

Depth: fixed 30% depth as primary (layer = round(0.30*(L-1)) per model); log chosen layer index.

Gen: temp=0.2, 3 repeats, sequential model load/unload.

Self-rating prompt: “Rate the type of processing…” + strict X — one sentence format; regex parse + parse_failed flag.

Logging: save full task response text + rating raw text + parsed score + response length/tokens + prompt_id/family_id.

Sensitivity: breakout-layer variant as supplementary if feasible under same CV protocol.
Output JSONL/CSV with one row per (prompt×repeat×model).
Then compute correlations (Spearman + Pearson) and partial corr controlling for response length.