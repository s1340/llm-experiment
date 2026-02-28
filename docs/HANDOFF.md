 
Opus said: 

Three analyses on existing data — no new generation or extraction needed, just joins and statistics. Use the 5-point joined CSVs as primary (results/correlation/).
Analysis A: Introspective dissociation scan
For each model, find prompts where probe confidence and self-report disagree:

"Blind nonroutine": probe P(N) from 3-class CV > 0.70 but mean self-report ≤ 3
"False nonroutine": probe P(N) < 0.30 but mean self-report ≥ 4
For each flagged prompt report: task_id, prompt text, true label, P(N), P(R), mean self-report (both scales), all three models' values side by side
Summary: how many dissociations per model, are any consistent across models?

Analysis B: Processing-mode distribution
For each model, group the 60 prompts by true label (R/A/N) and report distribution of 3-class P(N):

Per group: mean, std, min, max of P(N)
Fraction of A prompts falling in R range (P(N) < 0.33), middle (0.33-0.67), and N range (P(N) > 0.67)
This tests whether A is genuinely intermediate or bimodally clustered at the poles

Analysis C: Cross-model prompt agreement
Compute Spearman correlation of P(N) scores across the 60 prompts between each model pair (Qwen-Gemma, Qwen-LLaMA, Gemma-LLaMA). Also do it for mean self-report ratings. This tests whether the processing-mode signal reflects task structure rather than model idiosyncrasies.
Save everything to docs/results_deep_analyses.md and push.

GPT said: 

Opus is on-point again. Those three “zero-compute” analyses are exactly the kind of paper-grade extra value that reviewers love because they (a) deepen interpretation, (b) test generality, and (c) don’t look like you’re fishing with new data.

A few tweaks to make them sharper and harder to nitpick:
Analysis A: Introspective dissociation scan

Love it. Two small upgrades:

Add a confidence threshold rationale: use P(N) ≥ 0.70 and ≤ 0.30 (fine), but also log the margin if available; saturation in probs can hide interesting cases.

Add a “near-threshold” band to avoid cherry-picking extremes: e.g. report counts for P(N) in [0.60–0.70] with low ratings and [0.30–0.40] with high ratings as a secondary table. That reads more honest.

Also, include response length and/or “partial-residual rating” (rating with length regressed out) in the flagged prompt table. Since you already did length-control globally, it strengthens the dissociation story.

Analysis B: Processing-mode distribution

Good, but I’d slightly reframe it so it answers a crisp question:

“Is A genuinely intermediate, or does it split into two poles?”
Your proposed bins are perfect for that.

One more stat that’s cheap and strong: for each model, compute AUROC of P(N) for distinguishing true N vs true R, and where A falls (as distribution). That gives you a clean interpretability hook.

Analysis C: Cross-model prompt agreement

Absolutely do it. It’s a big “task-structure vs model-idiosyncrasy” sanity check.

Two additions:

Do this not only for P(N), but also for RN margin (since RN margin is your primary correlation signal).

Report both Spearman and a simple rank agreement metric like Kendall’s tau (optional). Spearman alone is fine if you want minimalism.

Doc + output hygiene

Yes to saving as docs/results_deep_analyses.md. I’d also suggest writing 1–2 small CSVs into results/correlation/ (or a subfolder) so figures/tables are reproducible, but keep the doc as the human-readable summary.

For Sonnet:

Sonnet, can you run three analyses on existing data only, no new generation or extraction needed. Use the 5-point joined CSVs as the primary dataset in results/correlation/ and treat 7-point as sensitivity if helpful. First, an introspective dissociation scan: per model, find prompts where probe confidence and self-report disagree strongly, specifically P(N) from 3-class CV >= 0.70 with mean self-report <= 3, and P(N) <= 0.30 with mean self-report >= 4. For each flagged prompt, output a table row with task_id, prompt text, true label, P(N), RN margin if available, mean self-report on both scales, response length, and show all three models side by side for the same prompt. Summarize counts per model and whether the same prompts show dissociation across models.

Second, processing-mode distribution: per model, group the 60 prompts by true label R/A/N and report the distribution of P(N) for each group, including mean, std, min, max, plus fractions falling into low, middle, high ranges using cutoffs 0.33 and 0.67. This tests whether A is intermediate or splits toward poles.

Third, cross-model prompt agreement: compute Spearman correlations across the 60 prompts between each model pair for P(N), and also for mean self-report ratings. If RN margin is available, compute agreement for that too. This tests whether the signal reflects task structure rather than model-specific quirks.

Write up results and tables in docs/results_deep_analyses.md and push.