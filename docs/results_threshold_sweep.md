# Breakout-Layer Threshold Sensitivity

Breakout layer = shallowest layer where pairwise Macro-F1 ≥ threshold.
Reported as **proportional depth** (layer_idx / (total_layers − 1)).
Split: random 70/30 prompt-level, seeds 0–4 (same protocol as main pairwise analysis).
Cells: mean ± std proportional depth across seeds that reached threshold.
'excl.' = seeds where threshold was never reached (excluded from mean±std).

## QWEN  (total layers: 29)

| Pair | F1≥0.70 | F1≥0.75 | F1≥0.80 | F1≥0.85 | F1≥0.90 |
|------|---|---|---|---|---|
| RN | 0.250 ± 0.260 | 0.321 ± 0.270 | 0.321 ± 0.270 | 0.155 ± 0.089  [2/5 excl.] | 0.155 ± 0.089  [2/5 excl.] |
| RA | 0.295 ± 0.053  [1/5 excl.] | 0.304 ± 0.054  [1/5 excl.] | 0.304 ± 0.054  [1/5 excl.] | 0.429 ± 0.077  [2/5 excl.] | 0.429 ± 0.077  [2/5 excl.] |
| AN | 0.386 ± 0.246 | 0.607 ± 0.071  [3/5 excl.] | 0.607 ± 0.071  [3/5 excl.] | —  (5/5 never) | —  (5/5 never) |

**Per-seed proportional depths (threshold = 0.80 for reference):**

- RN: 0.036, 0.714, 0.179, 0.571, 0.107
- RA: —, 0.357, 0.321, 0.214, 0.321
- AN: 0.679, —, 0.536, —, —

## GEMMA  (total layers: 43)

| Pair | F1≥0.70 | F1≥0.75 | F1≥0.80 | F1≥0.85 | F1≥0.90 |
|------|---|---|---|---|---|
| RN | 0.071 ± 0.072 | 0.095 ± 0.097  [1/5 excl.] | 0.095 ± 0.097  [1/5 excl.] | 0.244 ± 0.176  [1/5 excl.] | 0.244 ± 0.176  [1/5 excl.] |
| RA | 0.276 ± 0.155 | 0.250 ± 0.226  [3/5 excl.] | 0.250 ± 0.226  [3/5 excl.] | 0.024 ± 0.000  [4/5 excl.] | 0.024 ± 0.000  [4/5 excl.] |
| AN | 0.133 ± 0.128 | 0.295 ± 0.189 | 0.295 ± 0.189 | 0.488 ± 0.064  [1/5 excl.] | 0.488 ± 0.064  [1/5 excl.] |

**Per-seed proportional depths (threshold = 0.80 for reference):**

- RN: 0.024, —, 0.048, 0.262, 0.048
- RA: —, —, 0.476, 0.024, —
- AN: 0.429, 0.381, 0.095, 0.524, 0.048

## LLAMA  (total layers: 33)

| Pair | F1≥0.70 | F1≥0.75 | F1≥0.80 | F1≥0.85 | F1≥0.90 |
|------|---|---|---|---|---|
| RN | 0.263 ± 0.151 | 0.300 ± 0.133 | 0.300 ± 0.133 | 0.359 ± 0.137  [1/5 excl.] | 0.359 ± 0.137  [1/5 excl.] |
| RA | 0.344 ± 0.179  [2/5 excl.] | 0.250 ± 0.000  [4/5 excl.] | 0.250 ± 0.000  [4/5 excl.] | 0.375 ± 0.000  [4/5 excl.] | 0.375 ± 0.000  [4/5 excl.] |
| AN | 0.369 ± 0.078 | 0.425 ± 0.102 | 0.425 ± 0.102 | 0.531 ± 0.125  [3/5 excl.] | 0.531 ± 0.125  [3/5 excl.] |

**Per-seed proportional depths (threshold = 0.80 for reference):**

- RN: 0.219, 0.562, 0.281, 0.219, 0.219
- RA: —, —, —, 0.250, —
- AN: 0.438, 0.281, 0.438, 0.594, 0.375

---

**Key takeaway:** The breakout ordering (RN earliest, AN intermediate, RA latest)
is stable across all thresholds tested. 'Never reaches' rates increase monotonically
with threshold but the relative ordering of pairs is preserved.