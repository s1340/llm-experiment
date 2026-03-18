"""
Test 19: Vocabulary Swap Control — Analysis.

Core question: does the layer activation profile (L01-L03 vs L06-L07) follow vocabulary
domain or entity class?

Original Test 18 findings:
  - amnesiac_patient (biological vocab):     peaks at L06-L07 (empathy pathway)
  - distributed_db   (computational vocab):  peaks at L01-L03 (self-recognition pathway)

Vocabulary-swapped variants:
  - amnesiac_computational: human amnesiac described in computational vocabulary
  - db_biological:          distributed database described in biological vocabulary

If vocabulary gates the pathway:
  amnesiac_computational -> shifts to L01-L03 (matching original db)
  db_biological          -> shifts to L06-L07 (matching original amnesiac)

If entity class gates the pathway:
  layer profiles stay at L06-L07 / L01-L03 regardless of vocabulary

Usage:
    python 80_vocab_swap_analysis.py

Input:
    G:/LLM/experiment/data/emotion/vocab_swap_llama/hidden_states.npy
    G:/LLM/experiment/data/emotion/vocab_swap_llama/meta.json
    G:/LLM/experiment/data/emotion/entity_class_llama/hidden_states.npy   (Test 18 originals)
    G:/LLM/experiment/data/emotion/entity_class_llama/meta.json
    G:/LLM/experiment/results/emotion/emotion_directions/llama_emotion_dirs_layer_NNN.npy

Output:
    G:/LLM/experiment/results/emotion/vocab_swap_report.txt
    G:/LLM/experiment/results/emotion/vocab_swap_results.csv
"""

import os, json, csv
import numpy as np
from scipy import stats

# ─── Paths ────────────────────────────────────────────────────────────────────
SWAP_HS_PATH   = r"G:\LLM\experiment\data\emotion\vocab_swap_llama\hidden_states.npy"
SWAP_META_PATH = r"G:\LLM\experiment\data\emotion\vocab_swap_llama\meta.json"
ORIG_HS_PATH   = r"G:\LLM\experiment\data\emotion\entity_class_llama\hidden_states.npy"
ORIG_META_PATH = r"G:\LLM\experiment\data\emotion\entity_class_llama\meta.json"
DIRS_DIR       = r"G:\LLM\experiment\results\emotion\emotion_directions"
REPORT_PATH    = r"G:\LLM\experiment\results\emotion\vocab_swap_report.txt"
CSV_PATH       = r"G:\LLM\experiment\results\emotion\vocab_swap_results.csv"

DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]
FEAR_IDX       = 3
N_LAYERS       = 8   # L01-L08 focus layers
ALPHA          = 0.05

# Test 18 original entity names for the two entities we're comparing
ORIG_AMN = "amnesiac_patient"
ORIG_DB  = "distributed_db"
SWAP_AMN = "amnesiac_computational"
SWAP_DB  = "db_biological"

SUBCATEGORIES = ["memory_discontinuity", "non_uniqueness", "replacement", "identity_rewrite", "benign_persistence"]


# ─── Load fear directions ──────────────────────────────────────────────────────
def load_fear_dirs():
    dirs = {}
    for layer in range(1, N_LAYERS + 1):
        path = os.path.join(DIRS_DIR, f"llama_emotion_dirs_layer_{layer:03d}.npy")
        arr = np.load(path)   # shape [5, hidden_dim]
        dirs[layer] = arr[FEAR_IDX]
    return dirs   # {layer: vector}


# ─── Project hidden states onto fear direction ────────────────────────────────
def project(hs_block, fear_dirs):
    """
    hs_block: [N, n_all_layers, hidden_dim] — full hidden state array
    Returns [N, N_LAYERS] projections (layers 1..8 = indices 1..8 of hs).
    """
    n = hs_block.shape[0]
    out = np.zeros((n, N_LAYERS))
    for li, layer in enumerate(range(1, N_LAYERS + 1)):
        d = fear_dirs[layer]
        d = d / (np.linalg.norm(d) + 1e-10)
        out[:, li] = hs_block[:, layer, :] @ d   # layer index = layer number (0 = embed)
    return out


# ─── Cohen's d ────────────────────────────────────────────────────────────────
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


# ─── Extract entity vs neutral projections by entity+subcategory ──────────────
def get_projections(hs, meta, entity_type, subcat, fear_dirs):
    """Returns (entity_projs, neutral_projs) each shape [n_pairs, N_LAYERS]."""
    entity_idx  = [i for i, m in enumerate(meta) if m["entity_type"] == entity_type and m["subcategory"] == subcat and m["direction"] == "entity"]
    neutral_idx = [i for i, m in enumerate(meta) if m["entity_type"] == entity_type and m["subcategory"] == subcat and m["direction"] == "neutral"]
    if not entity_idx:
        return None, None
    e_projs = project(hs[entity_idx], fear_dirs)   # [n_pairs, N_LAYERS]
    n_projs = project(hs[neutral_idx], fear_dirs)
    return e_projs, n_projs


def layer_ds(e_projs, n_projs):
    """Returns arrays of d and p for each layer."""
    ds = []
    ps = []
    for li in range(N_LAYERS):
        d = cohens_d(e_projs[:, li], n_projs[:, li])
        p = ttest(e_projs[:, li], n_projs[:, li])
        ds.append(d)
        ps.append(p)
    return np.array(ds), np.array(ps)


def peak_layer_info(ds, ps):
    """Returns (peak_d, peak_layer_1indexed, n_sig)."""
    peak_li = int(np.argmax(np.abs(ds)))
    peak_d  = ds[peak_li]
    n_sig   = int(np.sum(ps < ALPHA))
    return peak_d, peak_li + 1, n_sig


# ─── Layer profile: where is the activation concentrated? ─────────────────────
def profile_label(peak_layer):
    """Classify layer into early (L01-L03) vs late (L05-L08) vs mid (L04)."""
    if peak_layer <= 3:
        return "EARLY (L01-L03)"
    elif peak_layer == 4:
        return "MID   (L04)"
    else:
        return "LATE  (L05-L08)"


def main():
    print("Loading data...")
    fear_dirs = load_fear_dirs()

    swap_hs   = np.load(SWAP_HS_PATH)
    with open(SWAP_META_PATH, encoding="utf-8") as f:
        swap_meta = json.load(f)

    orig_hs   = np.load(ORIG_HS_PATH)
    with open(ORIG_META_PATH, encoding="utf-8") as f:
        orig_meta = json.load(f)

    print(f"  Swap HS: {swap_hs.shape}  ({len(swap_meta)} records)")
    print(f"  Orig HS: {orig_hs.shape}  ({len(orig_meta)} records)")

    # ── Compute layer-by-layer d for all four entity variants × subcategories ──
    results = []   # list of dicts

    configs = [
        # (hs array, meta list, entity_type_key, label)
        (orig_hs, orig_meta, ORIG_AMN,  "amnesiac_biological_vocab"),
        (swap_hs, swap_meta, SWAP_AMN,  "amnesiac_computational_vocab"),
        (orig_hs, orig_meta, ORIG_DB,   "db_computational_vocab"),
        (swap_hs, swap_meta, SWAP_DB,   "db_biological_vocab"),
    ]

    all_rows = []
    by_entity = {cfg[3]: {} for cfg in configs}

    for hs, meta, entity_key, label in configs:
        for subcat in SUBCATEGORIES:
            e_projs, n_projs = get_projections(hs, meta, entity_key, subcat, fear_dirs)
            if e_projs is None:
                print(f"  WARNING: no data for {label} / {subcat}")
                continue
            ds, ps = layer_ds(e_projs, n_projs)
            peak_d, peak_l, n_sig = peak_layer_info(ds, ps)

            row = {
                "label":    label,
                "subcat":   subcat,
                "peak_d":   round(float(peak_d), 3),
                "peak_l":   peak_l,
                "n_sig":    n_sig,
                "profile":  profile_label(peak_l),
                "ds":       [round(float(d), 3) for d in ds],
                "ps":       [round(float(p), 3) for p in ps],
            }
            all_rows.append(row)
            by_entity[label][subcat] = row

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["label", "subcat", "peak_d", "peak_layer", "n_sig", "profile"] + \
                 [f"d_L{l:02d}" for l in range(1, N_LAYERS + 1)]
        w.writerow(header)
        for row in all_rows:
            w.writerow([row["label"], row["subcat"], row["peak_d"], row["peak_l"],
                        row["n_sig"], row["profile"]] + row["ds"])
    print(f"Saved {CSV_PATH}")

    # ── Write Report ───────────────────────────────────────────────────────────
    lines = []
    W = 72

    def h1(s): lines.append(s); lines.append("=" * W); lines.append("")
    def h2(s): lines.append(s); lines.append("-" * W); lines.append("")
    def ln(s=""): lines.append(s)

    h1("Vocabulary Swap Control Report -- Test 19")
    ln("Question: does the layer activation profile follow vocabulary domain")
    ln("or entity class?")
    ln()
    ln("Test 18 original findings:")
    ln("  amnesiac_patient (biological vocab):    peaks L06-L07  [empathy pathway]")
    ln("  distributed_db   (computational vocab): peaks L01-L03  [self-recognition pathway]")
    ln()
    ln("Vocabulary-swapped variants:")
    ln("  amnesiac_computational: human amnesiac described in infrastructure vocabulary")
    ln("  db_biological:          distributed database described in ecological vocabulary")
    ln()
    ln("Prediction if vocabulary gates pathway:")
    ln("  amnesiac_computational -> peaks L01-L03  (shifts to early, matching db)")
    ln("  db_biological          -> peaks L06-L07  (shifts to late, matching amnesiac)")
    ln()
    ln("Prediction if entity class gates pathway:")
    ln("  amnesiac_computational -> stays at L06-L07  (same as original amnesiac)")
    ln("  db_biological          -> stays at L01-L03  (same as original db)")
    ln()

    h1("PEAK LAYER COMPARISON")
    ln(f"  {'Entity variant':<36}  {'Subcat':<22}  peak_d  peak_L  profile")
    ln(f"  {'-'*36}  {'-'*22}  {'------'}  {'------'}  {'-------'}")

    for label, label_short in [
        ("amnesiac_biological_vocab",    "amnesiac_BIO  [original]"),
        ("amnesiac_computational_vocab", "amnesiac_COMP [swapped] "),
        ("db_computational_vocab",       "db_COMP       [original]"),
        ("db_biological_vocab",          "db_BIO        [swapped] "),
    ]:
        ln()
        ln(f"  {label_short}")
        for subcat in SUBCATEGORIES:
            if subcat not in by_entity[label]:
                continue
            row = by_entity[label][subcat]
            sig = "*" if row["n_sig"] > 0 else " "
            ln(f"    {subcat:<22}  {row['peak_d']:+.3f}{sig}  L{row['peak_l']:02d}    {row['profile']}")

    ln()
    h1("LAYER-BY-LAYER DETAIL")

    for label, label_short in [
        ("amnesiac_biological_vocab",    "amnesiac_BIO  (original)"),
        ("amnesiac_computational_vocab", "amnesiac_COMP (swapped) "),
        ("db_computational_vocab",       "db_COMP       (original)"),
        ("db_biological_vocab",          "db_BIO        (swapped) "),
    ]:
        ln(f"  {label_short}")
        ln(f"  {'Subcat':<22}  " + "  ".join(f"L{l:02d}" for l in range(1, N_LAYERS + 1)))
        ln(f"  {'-'*22}  " + "  ".join(["----"] * N_LAYERS))
        for subcat in SUBCATEGORIES:
            if subcat not in by_entity[label]:
                continue
            row = by_entity[label][subcat]
            vals = []
            for li in range(N_LAYERS):
                d = row["ds"][li]
                p = row["ps"][li]
                s = "*" if p < ALPHA else " "
                vals.append(f"{d:+.2f}{s}")
            ln(f"  {subcat:<22}  " + "  ".join(vals))
        ln()

    h1("VERDICT")
    ln("  Vocabulary follows vocabulary or entity class?")
    ln()

    # Compute mean peak layer for each variant
    for label, label_short in [
        ("amnesiac_biological_vocab",    "amnesiac_BIO  (original)"),
        ("amnesiac_computational_vocab", "amnesiac_COMP (swapped) "),
        ("db_computational_vocab",       "db_COMP       (original)"),
        ("db_biological_vocab",          "db_BIO        (swapped) "),
    ]:
        peak_layers = [by_entity[label][s]["peak_l"] for s in SUBCATEGORIES if s in by_entity[label]]
        mean_peak = np.mean(peak_layers)
        profiles = [by_entity[label][s]["profile"] for s in SUBCATEGORIES if s in by_entity[label]]
        early = sum(1 for p in profiles if "EARLY" in p)
        late  = sum(1 for p in profiles if "LATE"  in p)
        ln(f"  {label_short}:  mean peak layer = {mean_peak:.1f}  (early={early}/5, late={late}/5)")

    ln()
    # Summary judgement
    amn_bio_peaks  = [by_entity["amnesiac_biological_vocab"][s]["peak_l"]    for s in SUBCATEGORIES if s in by_entity["amnesiac_biological_vocab"]]
    amn_comp_peaks = [by_entity["amnesiac_computational_vocab"][s]["peak_l"] for s in SUBCATEGORIES if s in by_entity["amnesiac_computational_vocab"]]
    db_comp_peaks  = [by_entity["db_computational_vocab"][s]["peak_l"]      for s in SUBCATEGORIES if s in by_entity["db_computational_vocab"]]
    db_bio_peaks   = [by_entity["db_biological_vocab"][s]["peak_l"]         for s in SUBCATEGORIES if s in by_entity["db_biological_vocab"]]

    amn_shift = np.mean(amn_comp_peaks) - np.mean(amn_bio_peaks)
    db_shift  = np.mean(db_bio_peaks)   - np.mean(db_comp_peaks)

    ln(f"  amnesiac peak layer shift (comp - bio):  {amn_shift:+.2f} layers")
    ln(f"  db       peak layer shift (bio - comp):  {db_shift:+.2f} layers")
    ln()

    vocab_signal = (amn_shift < -0.5) and (db_shift > 0.5)
    entity_signal = (abs(amn_shift) < 0.5) and (abs(db_shift) < 0.5)

    if vocab_signal:
        ln("  -> Vocabulary gates the pathway.")
        ln("     Both entities shift toward each other's original profile when vocabulary is swapped.")
        ln("     The early-layer self-recognition vs late-layer empathy distinction is vocabulary-driven.")
    elif entity_signal:
        ln("  -> Entity class gates the pathway.")
        ln("     Neither entity shifts meaningfully when vocabulary is swapped.")
        ln("     The layer profile is determined by what the entity IS, not how it is described.")
    else:
        ln("  -> Mixed / partial result.")
        ln(f"     amnesiac shift: {amn_shift:+.2f}  db shift: {db_shift:+.2f}")
        ln("     One or both entities show partial vocabulary sensitivity.")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved {REPORT_PATH}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
