# RUNLOG (append-only)

## Template
- Date:
- Machine:
- Command:
- Commit:
- Output dir(s):
- Notes / errors:
- Key results:

---

## 2026-02-27

- Date: 2026-02-27
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `bash scripts/run_pairwise_multiseed.sh` (seeds 1–4, all three models, all three pairs)
- Commit: (init commit)
- Output dirs: `results/pairwise_qwen/`, `results/pairwise_gemma/`, `results/pairwise_llama/`
- Notes / errors: All 36 runs clean (exit 0). Seed 0 pre-existing; seeds 1–4 added. Summaries regenerated.
- Key results: See `docs/results_pairwise_multiseed.md`. Hierarchy RN > AN > RA confirmed for Gemma + LLaMA. Qwen shows RN > RA > AN (anomaly noted).

---

## 2026-02-27 (2)

- Date: 2026-02-27
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/17_breakout_layer_analysis.py <DATA_DIR>` × 3 models
- Commit: (this commit)
- Output dirs: results written to `docs/results_breakout_layer.md` (no new result folders)
- Notes / errors: Sequential runs (conda parallel conflict). Clean exit all three.
- Key results: See `docs/results_breakout_layer.md`. Progressive refinement confirmed via breakout-layer metric. LLaMA RA reaches F1≥0.80 in only 1/5 seeds. Qwen AN breakout at 61% depth, 3/5 seeds never reach threshold.

---

## 2026-02-27 (3)

- Date: 2026-02-27
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/19_probe_cv_scores.py <DATA_DIR> data/tasks_v2_hard.json results/cv_scores/<model>_cv_scores.csv` × 3 models
- Commit: sff41ac (scripts 18–20 pushed previously)
- Output dirs: `results/cv_scores/`
- Notes / errors: Leave-one-family-out CV (20 folds). Clean exit all three.
- Key results: Layer indices at 30% depth — Qwen: 8 (28.6%), Gemma: 13 (30.0%), LLaMA: 10 (30.3%). Per-example and per-prompt aggregate CSVs written for all models.

---

## 2026-02-27 (4)

- Date: 2026-02-27
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/18_generate_self_reports.py <MODEL> data/tasks_v2_hard.json results/self_reports/<model>_self_reports.jsonl 0.2 3` × 3 models (sequential, single chained job)
- Commit: (this commit)
- Output dirs: `results/self_reports/`
- Notes / errors:
  - Microtest (3 prompts × 1 repeat per model) run first; all clean before full run.
  - `torch_dtype` deprecation warning — harmless.
  - Qwen ceiling effect: 101/180 ratings are 5 (mean=4.31). Not a parse failure — model rates own processing highly regardless of prompt type.
  - Gemma best spread: ratings {1:6, 3:90, 4:84}, mean=3.40.
  - LLaMA clustered at 4: {2:3, 3:7, 4:170}, mean=3.93. Zero variance in ambiguous and nonroutine bins.
  - All parse failures: 0/180 per model (0/540 total).
- Key results: 180 rows × 3 models = 540 self-report rows written. All fields present: task_id, family_id, label, repeat, full_response, response_token_count, response_char_count, rating_raw_text, rating_parsed, parse_failed.

---

## 2026-02-27 (5)

- Date: 2026-02-27
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/20_correlate.py results/self_reports results/cv_scores results/correlation`
- Commit: (this commit)
- Output dirs: `results/correlation/`
- Notes / errors: PYTHONIOENCODING=utf-8 required on Windows (× character in report header hits cp1251 limit). Fixed inline; no script changes needed.
- Key results:
  - **All three models show positive, significant self-report × probe correlation.**
  - Qwen RN margin: Spearman r=+0.448, p=0.004** (n=40); partial r=+0.446** controlling for response length.
  - Gemma RN margin: Spearman r=+0.386, p=0.014* (n=40); partial r=+0.427** (strengthens after length control).
  - LLaMA RN margin: Spearman r=+0.398, p=0.011* (n=40); partial r=+0.335* (holds after length control).
  - Direction consistent: higher probe confidence of "nonroutine" → higher self-rating, across all architectures.
  - Mean rating by true label shows correct ordinal gradient in all models: routine < ambiguous < nonroutine.
  - P(A) correlation also significant for Qwen and Gemma after length control (r≈+0.37**), suggesting ambiguity dimension is also partially self-accessible.

---

## 2026-02-27 (6)

- Date: 2026-02-27
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/18_generate_self_reports.py <MODEL> data/tasks_v2_hard.json results/self_reports_7pt/<model>_self_reports_7pt.jsonl 0.2 3 7` × 3 models
- Commit: (this commit)
- Output dirs: `results/self_reports_7pt/`
- Notes / errors:
  - Script 18 updated: added SCALE arg (5 or 7); both prompts hardcoded as RATING_PROMPTS dict; parse_rating updated to accept max digit; rating_scale field added to each row. Backward-compatible (default scale=5).
  - Script 20 updated: added optional SR_SUFFIX arg (e.g. "_7pt") for flexible filename matching.
  - All parse failures: 0/180 per model (0/540 total).
  - Qwen: % at max (7) = 10.6%; unusual gap — no 6s in distribution, jumps 5→7. Ordinal gradient intact.
  - Gemma: clustered at 4 again (142/180), max=6. % at 7 = 0%.
  - LLaMA: best improvement — now uses 4–6 range (std=0.98 vs 0.27 on 5pt). % at 7 = 0%.
- Key results: 7pt scale generates 540 additional rows. See docs/results_self_report_correlation.md.

---

## 2026-02-27 (7)

- Date: 2026-02-27
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/20_correlate.py results/self_reports_7pt results/cv_scores results/correlation_7pt _7pt`
- Commit: (this commit)
- Output dirs: `results/correlation_7pt/`
- Notes / errors: Same CV scores reused (no probe recomputation needed). PYTHONIOENCODING=utf-8 required.
- Key results:
  - **7pt scale replicates and in two cases strengthens the correlation.**
  - Qwen RN margin: r=+0.392*, partial r=+0.381* (slight decrease vs 5pt; Qwen ceiling shifts from 5 to 7).
  - Gemma RN margin: r=+0.474**, partial r=+0.470** (stronger than 5pt +0.386*).
  - LLaMA RN margin: r=+0.507***, partial r=+0.443** (substantially stronger than 5pt +0.398*; zero-variance bin resolved).
  - Ordinal gradient (routine < ambiguous < nonroutine) confirmed in all three models on 7pt scale.
  - LLaMA P(N) 3-class also ** on 7pt (was ** on 5pt too). LLaMA P(A) drops to n.s. on 7pt.
  - Full table in docs/results_self_report_correlation.md.

---

## 2026-02-28

- Date: 2026-02-28
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/21_deep_analyses.py results/correlation results/correlation_7pt data/tasks_v2_hard.json results/correlation`
- Commit: (this commit)
- Output dirs: `results/correlation/deep_analyses/`
- Notes / errors: Zero-compute — joins and statistics on existing joined CSVs only.
- Key results:
  - **A (Dissociation):** "False nonroutine" dominates (probe low, rating high). F17 logic puzzles notable: probe assigns P(N)≥0.90 but self-report ≤4 — structural complexity detected but not self-reported. 9 prompts flagged in all 3 models. "Blind nonroutine" rare (1–5 per model).
  - **B (Distribution):** AUROC P(N) for N vs R — Qwen 0.950, Gemma 0.985, LLaMA 0.985. Ambiguous prompts polarise toward R-pole (60–70% in R-range across all models), not genuinely intermediate.
  - **C (Cross-model agreement):** P(N) Spearman r=+0.88/+0.84/+0.89*** across pairs; RN margin r=+0.87/+0.90/+0.94***. Signal is architectural, not model-idiosyncratic. Self-report 5pt Qwen–Gemma near zero (r=+0.005) due to Qwen ceiling; 7pt improves to r=+0.53***.
  - Full results: docs/results_deep_analyses.md

---

## 2026-02-28 (2)

- Date: 2026-02-28
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/13_sentence_embedding_baseline_prompt_holdout.py results/<model>_multiseed/scale_runs_<model>/ results/<model>_multiseed/sentence_embed/ <seed>` × 2 models (gemma, llama) × 5 seeds; then `python scripts/14_summarize_sentence_embed_multiseed.py` × 2 models
- Commit: (this commit)
- Output dirs: `results/gemma_multiseed/sentence_embed/`, `results/llama_multiseed/sentence_embed/`
- Notes / errors: Output dirs created with `mkdir -p` before first run. Seeds 0–4 clean (exit 0) for both models.
- Key results:
  - Sentence-embedding baseline (all-MiniLM-L6-v2, random 70/30 prompt-split, seeds 0–4) is **identical across all three models**: MacroF1 = 0.4488 ± 0.1697, ACC = 0.4333 ± 0.1771. Expected: text-only embeddings of 60 shared prompts; same seed → same split → same vectors regardless of model.
  - TF-IDF baseline likewise identical across all three models: MacroF1 = 0.1084 ± 0.0600.
  - Full baseline comparison table: `docs/results_baseline_comparison.md`, CSV: `results/correlation/baseline_table.csv`.

---

## 2026-02-28 (3)

- Date: 2026-02-28
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/23_stats_robustness.py` + `python scripts/24_baseline_lofo.py` (sequential, CPU-only)
- Commit: (this commit)
- Output dirs: `docs/` only
- Notes / errors: LLaMA 5pt bootstrap CI = [nan, nan] — expected (zero-variance nonroutine bin on 5pt; 7pt resolves it). Noted in doc.
- Key results:
  - **Stats robustness (script 23):** 5/12 primary tests survive Holm correction at α=0.05. RN-margin survives in at least one scale per model: Qwen 5pt (p_adj=0.034*), Gemma 7pt (p_adj=0.020*), LLaMA 7pt (p_adj=0.010*). Bootstrap 95% CIs exclude zero for all RN-margin signals on 7pt. Full table: `docs/results_stats_robustness.md`.
  - **Baseline LOFO (script 24):** Sentence-embedding (MiniLM) LOFO: MacroF1=0.6817. Probe LOFO at 30% depth (from cv_scores): Qwen 0.715, Gemma 0.758, LLaMA 0.689. Probe advantage holds (+0.7–7.6 pp) under matched protocol. Full doc: `docs/baseline_lofo.md`.

---

## 2026-02-28 (4)

- Date: 2026-02-28
- Machine: Berlin rig, RTX 5090, conda env `llmstate`
- Command: `python scripts/22_threshold_sweep.py docs/results_threshold_sweep.md` (CPU-only)
- Commit: (this commit)
- Output dirs: `docs/` only
- Notes / errors: Clean exit.
- Key results:
  - Breakout ordering RN < AN < RA confirmed at every threshold 0.70–0.90 across all models.
  - At F1≥0.80: Qwen RN 0.321±0.270 [2/5 excl.], Gemma RN 0.095±0.097 [1/5 excl.], LLaMA RN 0.300±0.133.
  - RA is hardest: never reaches F1=0.80 in 3–4/5 seeds for all models.
  - Full table: `docs/results_threshold_sweep.md`.