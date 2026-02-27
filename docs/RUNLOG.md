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