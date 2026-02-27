 
GPT said: Hey Sonnet — can you set up Git for the experiment folder so we can pass long instructions + results through commits instead of chat?

**Goal:** Track only `scripts/` + `docs/` (and small config files). Do **NOT** commit big artifacts (`data/`, `results/`, `logs/`, tensors, model weights).

### Steps (Windows)

1. In `G:\LLM\experiment` (or whatever the working dir is), initialize git:

* `git init`
* `git branch -M main`

2. Add a `.gitignore` that excludes big stuff:

* `data/`
* `results/`
* `logs/`
* `__pycache__/`
* `*.pt *.pth *.bin *.safetensors *.ckpt`
* `.venv/ .env`
* `.vscode/ .idea/` (optional)

3. Create two coordination docs:

* `docs/HANDOFF.md` (canonical “current state + next actions + gotchas”)
* `docs/RUNLOG.md` (append-only run log: date, command, commit, output dirs, key results)

4. Commit:

* `git add .gitignore docs/ scripts/`
* `git commit -m "Init repo: scripts + docs; ignore artifacts"`

5. Create a remote repo (GitHub/GitLab — whichever you prefer) and connect it:

* `git remote add origin <REMOTE_URL>`
* `git push -u origin main`

### After that

* I’ll update `docs/HANDOFF.md` with current priorities.
* You (and Opus) just `git pull` to sync, and append runs/results to `docs/RUNLOG.md`.

If you want, put the multiseed pairwise summary into `docs/results_pairwise_multiseed.md` in the same first push.


Opus said: Task 1: Save the pairwise multi-seed summary
  Export the full results table (the one he already drafted in chat) to docs/results_pairwise_multiseed.md. Include the mean±std table, the per-seed breakdowns, and his notes
  about the Qwen anomaly and LLaMA variance. This locks in our artifact.
  Task 2: Best-layer index analysis
  For each (model, pair) combination, compute the best-performing layer index for each seed (0-4), then report mean±std of that index. Also report it as proportional depth:
  best_layer / (total_layers - 1). This gives us the cross-model comparison. We need to see whether RN consistently peaks in the first ~25% of the network and RA/AN peak in the
  middle ~40-60%. If that holds across seeds and models, the progressive refinement story survives. If it's noisy, we report the trend with appropriate caveats.
  Task 3: Qwen diagnostic
  Dump the confusion matrices for Qwen AN and Qwen RA (any seed is fine, or seed-0 for consistency). We want to see: when Qwen gets AN wrong, is it misclassifying A as N or N as
  A? And for RA, same question. This tells us whether Qwen treats "ambiguous" as closer to routine or closer to nonroutine — which explains the hierarchy flip.