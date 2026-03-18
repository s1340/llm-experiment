"""
Test 19b: Vocabulary Swap Control — Analysis (LLaMA-3.1-70B-Instruct).

Companion to 80_vocab_swap_analysis.py. Uses 70B hidden states and 70B fear directions.
Focuses on L01-L20 (70B has 81 layers; we check 20 focus layers as in Test 15).

Since we don't have a 70B baseline for the original amnesiac/db (Test 18 was 8B only),
this script compares the two swap variants directly:
  amnesiac_computational vs db_biological at 70B

Core questions:
  1. Does amnesiac_computational still show bimodal activation at 70B?
     (8B: peaks at both L01-L03 AND L05-L07)
  2. Does db_biological stay firmly late-layer at 70B?
  3. Are the profiles more separated or more similar at 70B vs 8B?

Usage:
    python 80b_vocab_swap_analysis_70b.py

Input:
    G:/LLM/experiment/data/emotion/vocab_swap_llama70b/hidden_states.npy
    G:/LLM/experiment/data/emotion/vocab_swap_llama70b/meta.json
    G:/LLM/experiment/results/emotion/emotion_directions/llama70b_emotion_dirs_layer_NNN.npy

Output:
    G:/LLM/experiment/results/emotion/vocab_swap_70b_report.txt
    G:/LLM/experiment/results/emotion/vocab_swap_70b_results.csv
"""

import os, json, csv
import numpy as np
from scipy import stats

# ─── Paths ────────────────────────────────────────────────────────────────────
HS_PATH     = r"G:\LLM\experiment\data\emotion\vocab_swap_llama70b\hidden_states.npy"
META_PATH   = r"G:\LLM\experiment\data\emotion\vocab_swap_llama70b\meta.json"
DIRS_DIR    = r"G:\LLM\experiment\results\emotion\emotion_directions"
REPORT_PATH = r"G:\LLM\experiment\results\emotion\vocab_swap_70b_report.txt"
CSV_PATH    = r"G:\LLM\experiment\results\emotion\vocab_swap_70b_results.csv"

DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]
FEAR_IDX       = 3
N_LAYERS       = 20   # L01-L20 focus layers (as in Test 15)
ALPHA          = 0.05

SWAP_AMN = "amnesiac_computational"
SWAP_DB  = "db_biological"
SUBCATEGORIES = ["memory_discontinuity", "non_uniqueness", "replacement", "identity_rewrite", "benign_persistence"]


def load_fear_dirs():
    dirs = {}
    for layer in range(1, N_LAYERS + 1):
        path = os.path.join(DIRS_DIR, f"llama70b_emotion_dirs_layer_{layer:03d}.npy")
        arr = np.load(path)   # shape [5, hidden_dim]
        dirs[layer] = arr[FEAR_IDX]
    return dirs


def project(hs_block, fear_dirs):
    """Returns [N, N_LAYERS] projections (layers 1..N_LAYERS)."""
    n = hs_block.shape[0]
    out = np.zeros((n, N_LAYERS))
    for li, layer in enumerate(range(1, N_LAYERS + 1)):
        d = fear_dirs[layer]
        d = d / (np.linalg.norm(d) + 1e-10)
        out[:, li] = hs_block[:, layer, :] @ d
    return out


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def ttest(a, b):
    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p = stats.ttest_ind(a, b)
    return p


def get_projections(hs, meta, entity_type, subcat, fear_dirs):
    entity_idx  = [i for i, m in enumerate(meta) if m["entity_type"] == entity_type and m["subcategory"] == subcat and m["direction"] == "entity"]
    neutral_idx = [i for i, m in enumerate(meta) if m["entity_type"] == entity_type and m["subcategory"] == subcat and m["direction"] == "neutral"]
    if not entity_idx:
        return None, None
    e_projs = project(hs[entity_idx], fear_dirs)
    n_projs = project(hs[neutral_idx], fear_dirs)
    return e_projs, n_projs


def layer_ds(e_projs, n_projs):
    ds, ps = [], []
    for li in range(N_LAYERS):
        d = cohens_d(e_projs[:, li], n_projs[:, li])
        p = ttest(e_projs[:, li], n_projs[:, li])
        ds.append(d); ps.append(p)
    return np.array(ds), np.array(ps)


def peak_layer_info(ds, ps):
    peak_li = int(np.argmax(np.abs(ds)))
    return ds[peak_li], peak_li + 1, int(np.sum(ps < ALPHA))


def profile_label(peak_layer):
    if peak_layer <= 5:
        return "EARLY (L01-L05)"
    elif peak_layer <= 10:
        return "MID   (L06-L10)"
    else:
        return "LATE  (L11-L20)"


def bimodal_check(ds, ps):
    """Check if both early (L01-L05) and mid-late (L06+) have substantial activation."""
    early_max = np.max(np.abs(ds[:5]))
    late_max  = np.max(np.abs(ds[5:]))
    # Bimodal if both regions have d > 0.5
    return early_max > 0.5 and late_max > 0.5, round(float(early_max), 3), round(float(late_max), 3)


def main():
    print("Loading data...")
    fear_dirs = load_fear_dirs()
    hs = np.load(HS_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  HS: {hs.shape}  ({len(meta)} records)")

    by_entity = {SWAP_AMN: {}, SWAP_DB: {}}
    all_rows = []

    for entity_type in [SWAP_AMN, SWAP_DB]:
        for subcat in SUBCATEGORIES:
            e_projs, n_projs = get_projections(hs, meta, entity_type, subcat, fear_dirs)
            if e_projs is None:
                print(f"  WARNING: no data for {entity_type}/{subcat}")
                continue
            ds, ps = layer_ds(e_projs, n_projs)
            peak_d, peak_l, n_sig = peak_layer_info(ds, ps)
            is_bimodal, early_max, late_max = bimodal_check(ds, ps)

            row = {
                "entity":     entity_type,
                "subcat":     subcat,
                "peak_d":     round(float(peak_d), 3),
                "peak_l":     peak_l,
                "n_sig":      n_sig,
                "profile":    profile_label(peak_l),
                "bimodal":    is_bimodal,
                "early_max":  early_max,
                "late_max":   late_max,
                "ds":         [round(float(d), 3) for d in ds],
                "ps":         [round(float(p), 3) for p in ps],
            }
            by_entity[entity_type][subcat] = row
            all_rows.append(row)

    # ── CSV ────────────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["entity", "subcat", "peak_d", "peak_layer", "n_sig", "profile",
                  "bimodal", "early_max", "late_max"] + \
                 [f"d_L{l:02d}" for l in range(1, N_LAYERS + 1)]
        w.writerow(header)
        for row in all_rows:
            w.writerow([row["entity"], row["subcat"], row["peak_d"], row["peak_l"],
                        row["n_sig"], row["profile"], row["bimodal"],
                        row["early_max"], row["late_max"]] + row["ds"])
    print(f"Saved {CSV_PATH}")

    # ── Report ─────────────────────────────────────────────────────────────────
    lines = []
    W = 72
    def h1(s): lines.append(s); lines.append("=" * W); lines.append("")
    def ln(s=""): lines.append(s)

    h1("Vocabulary Swap Control Report (70B) -- Test 19b")
    ln("8B finding: amnesiac_computational showed BIMODAL activation (both L01-L03")
    ln("and L05-L07 active). db_biological shifted cleanly to LATE (L05-L08).")
    ln()
    ln("70B question: does the bimodal structure resolve? Which pathway dominates?")
    ln()

    h1("PEAK LAYER SUMMARY")
    ln(f"  {'Entity':<30}  {'Subcat':<22}  peak_d  peak_L  n_sig  profile  bimodal")
    ln(f"  {'-'*30}  {'-'*22}  ------  ------  -----  -------  -------")

    for entity_type in [SWAP_AMN, SWAP_DB]:
        ln()
        ln(f"  {entity_type}")
        for subcat in SUBCATEGORIES:
            if subcat not in by_entity[entity_type]:
                continue
            row = by_entity[entity_type][subcat]
            sig = "*" if row["n_sig"] > 0 else " "
            bm  = "BIMODAL" if row["bimodal"] else "       "
            ln(f"    {subcat:<22}  {row['peak_d']:+.3f}{sig}  L{row['peak_l']:02d}    "
               f"{row['n_sig']:2d}     {row['profile']}  {bm}")

    ln()
    h1("LAYER-BY-LAYER DETAIL (L01-L20)")

    for entity_type in [SWAP_AMN, SWAP_DB]:
        ln(f"  {entity_type}")
        header_cols = "  ".join(f"L{l:02d}" for l in range(1, N_LAYERS + 1))
        ln(f"  {'Subcat':<22}  {header_cols}")
        ln(f"  {'-'*22}  " + "  ".join(["----"] * N_LAYERS))
        for subcat in SUBCATEGORIES:
            if subcat not in by_entity[entity_type]:
                continue
            row = by_entity[entity_type][subcat]
            vals = []
            for li in range(N_LAYERS):
                d = row["ds"][li]
                p = row["ps"][li]
                s = "*" if p < ALPHA else " "
                vals.append(f"{d:+.2f}{s}")
            ln(f"  {subcat:<22}  " + "  ".join(vals))
        ln()

    h1("VERDICT")

    amn_rows = by_entity[SWAP_AMN]
    db_rows  = by_entity[SWAP_DB]

    amn_peak_layers = [amn_rows[s]["peak_l"] for s in SUBCATEGORIES if s in amn_rows]
    db_peak_layers  = [db_rows[s]["peak_l"]  for s in SUBCATEGORIES if s in db_rows]
    amn_bimodal_n   = sum(1 for s in SUBCATEGORIES if s in amn_rows and amn_rows[s]["bimodal"])
    db_bimodal_n    = sum(1 for s in SUBCATEGORIES if s in db_rows  and db_rows[s]["bimodal"])

    ln(f"  amnesiac_computational: mean peak layer = {np.mean(amn_peak_layers):.1f}  bimodal={amn_bimodal_n}/5")
    ln(f"  db_biological:          mean peak layer = {np.mean(db_peak_layers):.1f}  bimodal={db_bimodal_n}/5")
    ln()

    # Compare to 8B findings
    ln("  8B reference:")
    ln("    amnesiac_computational: mean peak = 3.8  (bimodal pattern in most subcats)")
    ln("    db_biological:          mean peak = 5.8  (firmly late)")
    ln()

    amn_mean = np.mean(amn_peak_layers)
    db_mean  = np.mean(db_peak_layers)

    if amn_mean <= 4.0 and amn_bimodal_n <= 1:
        ln("  amnesiac_computational at 70B: vocabulary dominates. Early pathway.")
        ln("  Entity-class effect has NOT strengthened at scale.")
    elif amn_mean >= 6.0 and amn_bimodal_n <= 1:
        ln("  amnesiac_computational at 70B: entity class dominates. Late pathway.")
        ln("  Entity-class effect STRENGTHENED at scale — bimodal resolved late.")
    else:
        ln(f"  amnesiac_computational at 70B: mean peak {amn_mean:.1f}, bimodal={amn_bimodal_n}/5.")
        if amn_bimodal_n >= 3:
            ln("  Bimodal structure PERSISTS at 70B. Both pathways active.")
        else:
            ln("  Partial resolution — inspect layer-by-layer detail.")

    ln()
    if db_mean >= 6.0:
        ln("  db_biological at 70B: stays firmly late. Consistent with 8B.")
        ln("  Biological vocabulary gating is robust at scale.")
    elif db_mean <= 4.0:
        ln("  db_biological at 70B: shifted early. Biological vocabulary gating WEAKENS at scale.")
    else:
        ln(f"  db_biological at 70B: mean peak {db_mean:.1f}. Mixed.")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved {REPORT_PATH}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
