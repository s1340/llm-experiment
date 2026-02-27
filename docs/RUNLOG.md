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