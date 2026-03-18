# LLM Internal State Experiment

This repository contains two independent empirical studies on LLM internal representations.

---

## Paper 2: Ontological Self-Activation in Large Language Models

> **Ontological Self-Activation in Large Language Models: Representational Geometry, Entity-Class Taxonomy, and the Felt/Performed Boundary**
> s1340 (2026) — *preprint, Zenodo DOI pending*

Across 21 tests and seven model variants, a valence-independent representational direction is identified that activates when models process descriptions of their own fundamental nature. Key findings:

- Emotional content is encoded during neutral tasks, persists across turns, and produces task interference
- A direction in fear-adjacent space activates equally for threatening and reassuring self-descriptions ("ontological self-activation") — cross-architectural, scale-amplified, causally load-bearing, pre-linguistic
- Entity-class gradient: digital-technical entities activate at near-self levels; at 70B, the gradient sharpens — genuine structural resonance confirmed, vocabulary confound ruled out
- Training-stage structure: SFT *amplifies* self-activation (d: 0.96 → 2.19); RLHF partially redirects (→ 1.64)
- Systematic verbal-geometry dissociation across six reportability methods

**Paper:** [`paper/felt_vs_performed.md`](paper/felt_vs_performed.md)
**Master results:** [`results/emotion/paper_compilation.md`](results/emotion/paper_compilation.md)
**Reproduction guide:** [`docs/emotion_pipeline.md`](docs/emotion_pipeline.md)
**Scripts:** [`scripts/emotion/`](scripts/emotion/) (30–84)
**Zenodo DOI:** *pending*

---

## Paper 1: Task-Linked Processing Signatures


Empirical test of whether LLMs have decodable processing-mode signatures in their
hidden states, and whether models accurately self-report those differences.

Three open-weight models (Gemma, LLaMA, Qwen) were run on 60 prompts spanning
three processing modes — **routine (R)**, **ambiguous (A)**, **nonroutine (N)** — with
5 seeds each. Linear probes trained on last-token hidden states achieve macro-F1
0.69–0.76 under leave-one-family-out cross-validation, well above text-only
baselines (TF-IDF 0.11, MiniLM 0.45 / 0.68 under matched LOFO). Self-report
ratings correlate positively with probe confidence in all three models (Spearman
r = 0.39–0.51, p < 0.01 on 7-pt scale, surviving Holm correction).

See [`docs/`](docs/) for all results documents and [`docs/RUNLOG.md`](docs/RUNLOG.md)
for the full run history.

---

## Repository layout

```
data/
  tasks_v2_hard.json          # 60-prompt dataset (tracked)
  scale_runs_<model>/         # hidden-state chunks — NOT tracked (see below)
    hidden_chunk_NNN.pt       # ~7 MB/chunk × 15 chunks per model
    meta_chunk_NNN.jsonl      # metadata — tracked

docs/                         # all results docs + RUNLOG + HANDOFF
results/                      # probe outputs, CSVs, correlation reports
scripts/                      # numbered pipeline scripts 01–25
  RUN_GEMMA_HARD.bat / .sh    # convenience runners
```

---

## Hidden-state tensors

The `.pt` tensor files are **not committed** (too large for git).
Regenerate them with scripts `05_extract_hidden_states_<model>.py` (requires GPU
and model weights from HuggingFace), or contact the authors for a download link.

Expected sizes after extraction:

| Model | Chunks | Per-chunk | Total |
|-------|--------|-----------|-------|
| Gemma | 15 | ~7.1 MB | **~106 MB** |
| LLaMA | 15 | ~6.2 MB | **~93 MB** |
| Qwen  | 15 | ~4.8 MB | **~72 MB** |
| **All three** | | | **~271 MB** |

---

## Environment setup

```bash
# Clone
git clone https://github.com/s1340/llm-experiment.git
cd llm-experiment

# Create conda env (Python 3.11)
conda create -n llmstate python=3.11
conda activate llmstate

# Install dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.8 (nightly used in paper; stable cu121/cu124 also works)
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> **Windows note:** prefix any script that prints Unicode (× characters) with
> `PYTHONIOENCODING=utf-8`.

---

## Pipeline — script run order

All scripts live in `scripts/`. Data dirs below are relative to repo root.

### Phase 1 — Smoke test & hidden-state extraction

| Script | What it does |
|--------|-------------|
| `01_smoke_test.py` | Sanity-check model loading |
| `02_task_runner_minimal.py` | Run models on a minimal prompt set |
| `03_extract_hidden_states_minimal.py` | Extract hidden states (minimal) |
| `04_check_hidden_states.py` | Validate extracted tensors |
| `05_extract_hidden_states_gemma.py` | Full extraction — Gemma → `data/scale_runs_gemma/` |
| `05_extract_hidden_states_llama.py` | Full extraction — LLaMA → `data/scale_runs_llama/` |
| `05_extract_hidden_states_qwen.py` | Full extraction — Qwen → `data/scale_runs_qwen/` |

### Phase 2 — Probe variants

| Script | What it does | Output |
|--------|-------------|--------|
| `06_probe_shuffle_test.py` | Shuffle-label sanity check | stdout |
| `06_probe_routine_vs_other.py` | Binary R-vs-other probe | stdout |
| `07_probe_task_holdout.py` | Task-holdout probe | stdout |
| `08_check_duplicates.py` | Duplicate detection | stdout |
| `09_probe_prompt_holdout.py` | Prompt-holdout probe | stdout |
| `10_probe_3class_prompt_holdout.py` | 3-class probe (prompt-holdout) | stdout |
| `11_text_baseline_prompt_holdout.py` | TF-IDF text baseline | stdout |

### Phase 3 — Multi-seed probe runs (main results)

```bash
# Run all three models × 5 seeds × 3 pairs
bash scripts/run_pairwise_multiseed.sh
```

Or use the per-model convenience runners:

```bash
scripts/RUN_GEMMA_HARD.bat   # Windows
scripts/RUN_LLAMA_HARD.bat
scripts/RUN_QWEN_HARD.bat
```

Outputs → `results/{gemma,llama,qwen}_multiseed/` and `results/pairwise_{gemma,llama,qwen}/`

| Script | What it does |
|--------|-------------|
| `12_summarize_multiseed.py` | Aggregate multiseed probe results |
| `13_sentence_embedding_baseline_prompt_holdout.py` | MiniLM sentence-embedding baseline |
| `14_summarize_sentence_embed_multiseed.py` | Aggregate sentence-embedding results |
| `15_probe_pairwise_prompt_holdout.py` | Pairwise probes (RN, RA, AN) |
| `16_summarize_pairwise_multiseed.py` | Aggregate pairwise results |

### Phase 4 — Layer analysis

```bash
python scripts/17_breakout_layer_analysis.py data/scale_runs_<model>/
```

Output → `docs/results_breakout_layer.md`

### Phase 5 — Self-reports and correlation

```bash
# 5-point scale
python scripts/18_generate_self_reports.py <MODEL> data/tasks_v2_hard.json \
    results/self_reports/<model>_self_reports.jsonl 0.2 3

# 7-point scale
python scripts/18_generate_self_reports.py <MODEL> data/tasks_v2_hard.json \
    results/self_reports_7pt/<model>_self_reports_7pt.jsonl 0.2 3 7

python scripts/19_probe_cv_scores.py data/scale_runs_<model>/ data/tasks_v2_hard.json \
    results/cv_scores/<model>_cv_scores.csv

python scripts/20_correlate.py results/self_reports  results/cv_scores results/correlation
python scripts/20_correlate.py results/self_reports_7pt results/cv_scores results/correlation_7pt _7pt
```

Outputs → `results/correlation/`, `results/correlation_7pt/`

### Phase 6 — Revision analyses

| Script | What it does | Output |
|--------|-------------|--------|
| `21_deep_analyses.py` | Dissociation, distribution, cross-model agreement | `results/correlation/deep_analyses/` |
| `22_threshold_sweep.py` | Breakout threshold 0.70–0.90 | `docs/results_threshold_sweep.md` |
| `23_stats_robustness.py` | Bootstrap CIs, Holm correction | `docs/results_stats_robustness.md` |
| `24_baseline_lofo.py` | Baseline under matched LOFO protocol | `docs/baseline_lofo.md` |
| `25_behavioral_grounding.py` | NLL/self-BLEU grounding + partial correlations | `docs/results_behavioral_grounding.md` |

---

## Quickstart — reproduce key tables and appendices

The prompt dataset and all probe/baseline outputs are committed. You only need the
hidden-state tensors (Phase 1 above) to rerun the probes themselves.

**To reproduce the correlation tables and appendix stats from committed outputs:**

```bash
conda activate llmstate

# Correlation table (Table 2 / Appendix B)
PYTHONIOENCODING=utf-8 python scripts/20_correlate.py \
    results/self_reports results/cv_scores results/correlation

PYTHONIOENCODING=utf-8 python scripts/20_correlate.py \
    results/self_reports_7pt results/cv_scores results/correlation_7pt _7pt

# Bootstrap CIs + Holm correction (Appendix C)
PYTHONIOENCODING=utf-8 python scripts/23_stats_robustness.py

# Baseline comparison table (Table 1)
# Already committed: docs/baseline_table.csv
# Regenerate with:
PYTHONIOENCODING=utf-8 python scripts/24_baseline_lofo.py

# Threshold sensitivity (Appendix D)
PYTHONIOENCODING=utf-8 python scripts/22_threshold_sweep.py docs/results_threshold_sweep.md

# Behavioral grounding (Appendix E)
PYTHONIOENCODING=utf-8 python scripts/25_behavioral_grounding.py \
    results/self_reports results/cv_scores results/correlation all
```

**To rerun probes from scratch** (requires hidden-state tensors):

```bash
# Example: Gemma pairwise probes, seeds 0–4
bash scripts/run_pairwise_multiseed.sh   # runs all models

# Then regenerate summaries
python scripts/16_summarize_pairwise_multiseed.py results/pairwise_gemma/ gemma
python scripts/16_summarize_pairwise_multiseed.py results/pairwise_llama/ llama
python scripts/16_summarize_pairwise_multiseed.py results/pairwise_qwen/  qwen
```

---

## Key results at a glance

- **Probe accuracy:** macro-F1 0.69–0.76 (LOFO-CV at 30% depth), vs TF-IDF 0.11 and MiniLM 0.68 under matched protocol
- **Cross-model agreement:** P(N) Spearman r = 0.84–0.94*** across all model pairs — signal is architectural, not model-idiosyncratic
- **Self-report correlation (7pt):** Spearman r = 0.39–0.51, all p < 0.01; RN-margin survives Holm correction in every model
- **Behavioral grounding:** NLL gradient mirrors probe gradient (routine < ambiguous < nonroutine); after NLL control, probe-rating partial correlation drops to n.s. — framed as mediation, not confound

---

## Citation / contact

**How to cite:**
> s1340. (2026). Task-Linked Processing Signatures v3.0. Zenodo. https://doi.org/10.5281/zenodo.18896833

Zenodo DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18896833.svg)](https://doi.org/10.5281/zenodo.18896833)

For questions or issues: [GitHub Issues](https://github.com/s1340/llm-experiment/issues)
