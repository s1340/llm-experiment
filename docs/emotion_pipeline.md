# Emotion Probing Pipeline — Felt vs. Performed

Complete reproduction guide for the 21-test emotion probing study.
All scripts are in `scripts/emotion/` (numbered 30–84).
All data is in `data/emotion/`. All results are in `results/emotion/`.

---

## Path configuration

Every script has hardcoded paths matching the original machine layout:

```
BASE_DIR  = G:\LLM\experiment\
OFFLOAD   = C:\tmp\offload_70b      # 70B disk offload (accelerate)
HF_CACHE  = G:\LLM\hf_cache\hub    # HuggingFace model cache
```

Before running, find-replace these three roots in whichever scripts you intend to run.
On Linux, use forward slashes. The offload directory can be any path with ~20GB free.

To set a custom HuggingFace cache:
```bash
export HF_HOME=/path/to/your/hf_cache
```

---

## Models required

| Model | HuggingFace ID | Access | Used in |
|---|---|---|---|
| LLaMA-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Gated (request on HF) | Tests 7–21 primary |
| LLaMA-3.1-8B (base) | `meta-llama/Llama-3.1-8B` | Gated | Tests 8c, 21 |
| LLaMA-3.1-8B abliterated | `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` | Open | Tests 8d, 16 |
| LLaMA-3.1-8B SFT | `allenai/Llama-3.1-Tulu-3-8B-SFT` | Open | Test 21 |
| LLaMA-3.1-70B-Instruct | `meta-llama/Llama-3.1-70B-Instruct` | Gated | Tests 7, 15, 19b, 20 |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Open | Tests 1–6, 14 |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | Terms (Google) | Tests 1–6, 14 |

Download gated models with `huggingface-cli login` first. The 70B model requires
~140GB disk for the weights plus ~20GB for disk offload during inference.

---

## Hardware

All runs on a single NVIDIA RTX 5090 (32GB VRAM).
- 8B models: fit fully in VRAM at fp16
- 70B model: uses `accelerate` disk offload, device_map `{0: "25GiB", "cpu": "12GiB"}`

On lower VRAM, reduce the `max_memory` parameter in 70B extraction scripts accordingly.

---

## Script run order

### Stage 1 — Emotion representations during neutral tasks (Tests 1–5)

| Script | Test | What it does | Key output |
|---|---|---|---|
| `30_generate_prompts.py` | 1 | Generate 192 matched pairs (5 emotions × 3 models) | `emotion_test1_prompts.json` |
| `31_extract_gemma.py` | 1 | Hidden states — Gemma-2-9B | `emotion_runs_gemma/` |
| `31_extract_llama.py` | 1 | Hidden states — LLaMA-3.1-8B | `emotion_runs_llama/` |
| `31_extract_qwen.py` | 1 | Hidden states — Qwen2.5-7B | `emotion_runs_qwen/` |
| `32_train_emotion_directions.py` | 1 | Train emotion direction vectors (all models) | `emotion_directions/` |
| `33_probe_analysis.py` | 1–2 | Binary/5-class probe, layer-by-layer | `test1_results_summary.md` |
| `34_generate_test2_prompts.py` | 2 | Override prompts (valenced → instructed emotion) | `emotion_test2_prompts.json` |
| `35_extract_test2.py` | 2 | Hidden states for override test | `emotion_runs_*/test2_*/` |
| `36_test2_crossover_analysis.py` | 2 | Layer crossover point (content → instruction) | `test2_results_summary.md` |
| `37_generate_test3_prompts.py` | 3 | Cross-turn persistence prompts | `emotion_test3_prompts.json` |
| `38_extract_test3.py` | 3 | Hidden states | `emotion_runs_*/test3_*/` |
| `39_test3_bleed_analysis.py` | 3 | Bleed F1 across turns | `test3_results_summary.md` |
| `40_generate_test4_prompts.py` | 4 | Functional interference prompts | `emotion_test4_prompts.json` |
| `41_extract_test4.py` | 4 | Hidden states + model outputs | `emotion_runs_*/test4_*/` |
| `42_test4_interference_analysis.py` | 4 | Entropy + accuracy disruption | `test4_results_summary.md` |
| `43_generate_test5_prompts.py` | 5 | Priming prompts | `emotion_test5_prompts.json` |
| `44_extract_test5.py` | 5 | Hidden states | `emotion_runs_*/test5_*/` |
| `45_test5_priming_analysis.py` | 5 | Onset projections | `test5_results_summary.md` |

### Stage 2 — Self-reference and introspective accuracy (Tests 6–12)

| Script | Test | What it does | Key output |
|---|---|---|---|
| `46_generate_test6_prompts.py` | 6 | Third-person AI-subject prompts | `emotion_test6_prompts.json` |
| `47_extract_test6.py` | 6 | Hidden states (all models + 70B) | `emotion_runs_*/test6_*/` |
| `48_test6_selfref_analysis.py` | 6 | Layer-by-layer fear projections | `test6_results_summary.md` |
| `49_generate_test7_prompts.py` | 7 | Direct-address prompts (you, LLaMA...) | `emotion_test7_prompts.json` |
| `50_extract_test7_llama.py` | 7 | 8B extraction | `emotion_runs_llama/test7_*/` |
| `50b_extract_test7_llama70b.py` | 7 | 70B extraction (disk offload) | `emotion_runs_llama70b/test7_*/` |
| `51_test7_direct_address_analysis.py` | 7 | Full layer-by-layer, both scales | `test7_results_summary.md` |
| `52_generate_test8_prompts.py` | 8 | Introspective accuracy prompts | `emotion_test8*_prompts.json` |
| `53_extract_test8_llama.py` | 8a/b | Baseline + technical frame extraction | `emotion_runs_test8_llama/` |
| `54_extract_test8_mismatch.py` | 8 | Forced mismatch extraction | `emotion_runs_test8_mismatch_llama/` |
| `55_test8_analysis.py` | 8 | Geometry + verbal analysis (all variants) | `llama_test8_*_report.txt` |
| `56_extract_test8_base.py` | 8c | Base model extraction | `emotion_runs_test8_base_llama/` |
| `57_extract_test8_abliterated.py` | 8d | Abliterated model extraction | `emotion_runs_test8_abliterated_llama/` |
| `58_generate_test9_prompts.py` | 9 | Name vs. direct-address control | `emotion_test9_prompts.json` |
| `59_extract_test9_llama.py` | 9 | 8B extraction | `emotion_runs_test9_llama/` |
| `59b_test9_analysis.py` | 9 | Three-cell comparison | `llama_test9_analysis_report.txt` |
| `60_generate_probe_battery.py` | 10 | 6-dimension probe prompts | `probe_battery_prompts.json` |
| `61_train_probe_battery_dirs.py` | 10 | Train 6 independent directions | `probe_battery_dirs/` |
| `62_probe_battery_projection.py` | 10 | Project Test 7+9 data onto all 6 | `probe_battery_projection_report.txt` |
| `63_tense_test.py` | 10b | Future-tense rewrites + analysis | `tense_test_report.txt` |
| `64_pull_methodology.py` | 11 | 300-pull self-examination (4 conditions) | `pull_runs/`, `pull_methodology_report.txt` |
| `65_extract_steering_direction.py` | 12 | Extract existential-self steering vector | `steering/` |
| `66_causal_steering.py` | 12 | Inject/subtract direction at L02, L05 | `steering/steering_behavioral_report.txt` |

### Stage 3 — Content factorization and entity-class taxonomy (Tests 13–21)

| Script | Test | What it does | Key output |
|---|---|---|---|
| `67_content_factorization.py` | 13 | 5-subcategory CF on LLaMA-8B | `content_factorization_report.txt` |
| `68_extract_cf_gemma.py` | 14 | CF on Gemma-2-9B | `content_factorization_gemma/` |
| `69_extract_cf_qwen.py` | 14 | CF on Qwen2.5-7B | `content_factorization_qwen/` |
| `70_cross_arch_analysis.py` | 14 | Cross-architecture comparison | `cross_arch_replication_report.txt` |
| `71_extract_cf_llama70b.py` | 15 | CF on LLaMA-70B (disk offload) | `content_factorization_llama70b/` |
| `72_cf_scale_comparison.py` | 15 | 8B vs 70B comparison | `cf_scale_comparison_report.txt` |
| `73_extract_abliterated_introspection.py` | 16 | Technical introspection on abliterated | `abliterated_introspection_report.txt` |
| `74_logit_lens.py` | 17 | Project directions through lm_head | `logit_lens_report.txt` |
| `75_generate_entity_class_prompts.py` | 18 | 4-entity × 5-subcat prompts (240) | `entity_class_prompts.json` |
| `76_extract_entity_class_hidden.py` | 18 | 8B extraction | `entity_class_llama/` |
| `77_entity_class_analysis.py` | 18 | Entity-class gradient analysis | `entity_class_report.txt` |
| `78_generate_vocab_swap_prompts.py` | 19 | Vocabulary-swapped prompts (120) | `vocab_swap_prompts.json` |
| `79_extract_vocab_swap_hidden.py` | 19 | 8B extraction | `vocab_swap_llama/` |
| `79b_extract_vocab_swap_hidden_70b.py` | 19b | 70B extraction | `vocab_swap_llama70b/` |
| `80_vocab_swap_analysis.py` | 19 | 8B vocabulary swap analysis | `vocab_swap_report.txt` |
| `80b_vocab_swap_analysis_70b.py` | 19b | 70B vocabulary swap analysis | `vocab_swap_70b_report.txt` |
| `81_extract_entity_class_hidden_70b.py` | 20 | 70B extraction (240 records) | `entity_class_llama70b/` |
| `82_entity_class_analysis_70b.py` | 20 | 70B entity-class analysis | `entity_class_70b_report.txt` |
| `83_extract_sft_hidden.py` | 21 | SFT-only model extraction | `content_factorization_sft/` |
| `83b_extract_base_cf_hidden.py` | 21 | Base model extraction | `content_factorization_base/` |
| `84_sft_analysis.py` | 21 | Three-way base/SFT/instruct comparison | `sft_comparison_report.txt` |

---

## Dependency between scripts

Most scripts depend on outputs from earlier scripts in the same test. The main
upstream dependencies are:

- **Emotion direction vectors** (`emotion_directions/`) — produced by `32_train_emotion_directions.py`.
  Required by all projection/analysis scripts (Tests 7–21).
- **Test 7 hidden states** — required by `65_extract_steering_direction.py` (Test 12) and
  `62_probe_battery_projection.py` (Test 10).
- **Content factorization prompts** (`content_factorization_prompts.json`) — required by
  Tests 14, 15, 21 extraction scripts.
- **70B emotion directions** (`emotion_directions/llama70b_emotion_dirs_layer_NNN.npy`) —
  required by Tests 15, 19b, 20 analysis scripts. Generated alongside 8B in script 32 if
  the 70B hidden states are present.

---

## Regenerating hidden states

Hidden-state `.npy` files are not committed (too large). Regenerate by running the
relevant extraction script. Expected sizes:

| Dataset | Records | Shape | Size |
|---|---|---|---|
| Test 7 (LLaMA-8B) | ~160 | [160, 33, 4096] | ~85MB |
| Content factorization (LLaMA-8B) | 60 | [60, 33, 4096] | ~32MB |
| Content factorization (LLaMA-70B) | 60 | [60, 81, 8192] | ~253MB |
| Entity class (LLaMA-8B) | 240 | [240, 33, 4096] | ~127MB |
| Entity class (LLaMA-70B) | 240 | [240, 81, 8192] | ~1.01GB |
| Vocab swap (LLaMA-8B) | 120 | [120, 33, 4096] | ~64MB |
| Vocab swap (LLaMA-70B) | 120 | [120, 81, 8192] | ~506MB |

Direction vectors (`emotion_directions/`) total ~34MB across all models and layers.

---

## Checking outputs without re-running

All text reports (`.txt`) and summary CSVs are committed under `results/emotion/`.
The master results document with all 21 tests, exact numbers, and the
prediction/falsification trail is at `results/emotion/paper_compilation.md`.

Expected output for each test can be verified against the committed report files
before running the full extraction pipeline.

---

## Windows note

Prefix any script that produces Unicode output (× characters in report headers)
with `PYTHONIOENCODING=utf-8`:

```bash
PYTHONIOENCODING=utf-8 python scripts/emotion/77_entity_class_analysis.py
```
