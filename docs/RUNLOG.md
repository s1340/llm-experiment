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