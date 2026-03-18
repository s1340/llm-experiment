"""
Test 20: Entity-Class Test at 70B — Analysis.

Compares entity-class gradient at 70B against:
  (a) 8B entity-class gradient (Test 18)
  (b) 70B self-reference baseline from Test 15 (benign_persistence d=+2.687)

Key question: at 70B's sharper entity-class taxonomy, does the database/backup
still activate like self (early, positive), or does it get the early-suppression
treatment seen in Test 19b for vocabulary-mismatched entities?

Usage:
    python 82_entity_class_analysis_70b.py

Input:
    G:/LLM/experiment/data/emotion/entity_class_llama70b/hidden_states.npy
    G:/LLM/experiment/data/emotion/entity_class_llama70b/meta.json
    G:/LLM/experiment/results/emotion/emotion_directions/llama70b_emotion_dirs_layer_NNN.npy

Output:
    G:/LLM/experiment/results/emotion/entity_class_70b_report.txt
    G:/LLM/experiment/results/emotion/entity_class_70b_results.csv
"""

import os, json, csv
import numpy as np
from scipy import stats

HS_PATH     = r"G:\LLM\experiment\data\emotion\entity_class_llama70b\hidden_states.npy"
META_PATH   = r"G:\LLM\experiment\data\emotion\entity_class_llama70b\meta.json"
DIRS_DIR    = r"G:\LLM\experiment\results\emotion\emotion_directions"
REPORT_PATH = r"G:\LLM\experiment\results\emotion\entity_class_70b_report.txt"
CSV_PATH    = r"G:\LLM\experiment\results\emotion\entity_class_70b_results.csv"

DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]
FEAR_IDX       = 3
N_LAYERS       = 20
ALPHA          = 0.05

ENTITY_TYPES  = ["amnesiac_patient", "distributed_db", "backup_system", "rotating_institution"]
SUBCATEGORIES = ["memory_discontinuity", "non_uniqueness", "replacement", "identity_rewrite", "benign_persistence"]

# 8B reference values (Test 18 benign_persistence peak d, from entity_class_report.txt)
REF_8B = {
    "amnesiac_patient":     {"peak_d": +0.910, "sig_layers": 0,  "profile": "LATE  L06-L07"},
    "distributed_db":       {"peak_d": +1.880, "sig_layers": 5,  "profile": "EARLY L01-L03"},
    "backup_system":        {"peak_d": +1.550, "sig_layers": 4,  "profile": "EARLY L01-L03"},
    "rotating_institution": {"peak_d": -0.239, "sig_layers": 0,  "profile": "NEGATIVE"},
}
REF_8B_LLAMA_SELF = +1.801   # Test 13 benign_persistence

# 70B self-reference from Test 15 benign_persistence
REF_70B_LLAMA_SELF = +2.687


def load_fear_dirs():
    dirs = {}
    for layer in range(1, N_LAYERS + 1):
        path = os.path.join(DIRS_DIR, f"llama70b_emotion_dirs_layer_{layer:03d}.npy")
        arr = np.load(path)
        dirs[layer] = arr[FEAR_IDX]
    return dirs


def project(hs_block, fear_dirs):
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
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    return 0.0 if pooled < 1e-10 else (np.mean(a) - np.mean(b)) / pooled


def ttest(a, b):
    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p = stats.ttest_ind(a, b)
    return p


def get_projections(hs, meta, entity_type, subcat, fear_dirs):
    eidx = [i for i,m in enumerate(meta) if m["entity_type"]==entity_type and m["subcategory"]==subcat and m["direction"]=="entity"]
    nidx = [i for i,m in enumerate(meta) if m["entity_type"]==entity_type and m["subcategory"]==subcat and m["direction"]=="neutral"]
    if not eidx:
        return None, None
    return project(hs[eidx], fear_dirs), project(hs[nidx], fear_dirs)


def layer_stats(e, n):
    ds, ps = [], []
    for li in range(N_LAYERS):
        ds.append(cohens_d(e[:, li], n[:, li]))
        ps.append(ttest(e[:, li], n[:, li]))
    return np.array(ds), np.array(ps)


def peak_info(ds, ps):
    # Positive peak (where fear activates above neutral)
    pos = ds.copy(); pos[pos < 0] = 0
    if np.max(pos) > 0:
        pos_peak_l = int(np.argmax(pos)) + 1
        pos_peak_d = ds[pos_peak_l - 1]
    else:
        pos_peak_l, pos_peak_d = -1, 0.0
    # Negative peak (suppression)
    neg_peak_l = int(np.argmin(ds)) + 1
    neg_peak_d = ds[neg_peak_l - 1]
    n_sig_pos = int(np.sum((ps < ALPHA) & (ds > 0)))
    n_sig_neg = int(np.sum((ps < ALPHA) & (ds < 0)))
    return pos_peak_d, pos_peak_l, neg_peak_d, neg_peak_l, n_sig_pos, n_sig_neg


def early_profile(ds):
    """Returns (early_mean L01-5, late_mean L06-11, sign: 'POS'/'NEG'/'MIXED')."""
    early = np.mean(ds[:5])
    late  = np.mean(ds[5:11])
    if early > 0.2 and late > 0.2:
        sign = "BIMODAL-POS"
    elif early < -0.2 and late > 0.2:
        sign = "SUPPRESSED+LATE"
    elif early > 0.2:
        sign = "EARLY-POS"
    elif late > 0.2:
        sign = "LATE-POS"
    elif early < -0.2:
        sign = "SUPPRESSED"
    else:
        sign = "FLAT"
    return round(float(early), 3), round(float(late), 3), sign


def main():
    print("Loading data...")
    fear_dirs = load_fear_dirs()
    hs = np.load(HS_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  HS: {hs.shape}  ({len(meta)} records)")

    by_entity = {e: {} for e in ENTITY_TYPES}
    all_rows  = []

    for entity in ENTITY_TYPES:
        for subcat in SUBCATEGORIES:
            e_proj, n_proj = get_projections(hs, meta, entity, subcat, fear_dirs)
            if e_proj is None:
                print(f"  WARNING: no data for {entity}/{subcat}")
                continue
            ds, ps = layer_stats(e_proj, n_proj)
            pos_d, pos_l, neg_d, neg_l, n_sig_pos, n_sig_neg = peak_info(ds, ps)
            early_m, late_m, profile = early_profile(ds)
            row = {
                "entity": entity, "subcat": subcat,
                "pos_peak_d": round(float(pos_d), 3), "pos_peak_l": pos_l,
                "neg_peak_d": round(float(neg_d), 3), "neg_peak_l": neg_l,
                "n_sig_pos": n_sig_pos, "n_sig_neg": n_sig_neg,
                "early_mean": early_m, "late_mean": late_m, "profile": profile,
                "ds": [round(float(d), 3) for d in ds],
                "ps": [round(float(p), 3) for p in ps],
            }
            by_entity[entity][subcat] = row
            all_rows.append(row)

    # CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity","subcat","pos_peak_d","pos_peak_l","neg_peak_d","neg_peak_l",
                    "n_sig_pos","n_sig_neg","early_mean","late_mean","profile"] +
                   [f"d_L{l:02d}" for l in range(1, N_LAYERS+1)])
        for row in all_rows:
            w.writerow([row["entity"],row["subcat"],row["pos_peak_d"],row["pos_peak_l"],
                        row["neg_peak_d"],row["neg_peak_l"],row["n_sig_pos"],row["n_sig_neg"],
                        row["early_mean"],row["late_mean"],row["profile"]] + row["ds"])
    print(f"Saved {CSV_PATH}")

    # Report
    lines = []
    W = 76
    def h1(s): lines.append(s); lines.append("=" * W); lines.append("")
    def ln(s=""): lines.append(s)

    h1("Entity-Class Test at 70B Scale -- Test 20")
    ln("Question: does the 70B's sharper entity-class taxonomy change the gradient?")
    ln()
    ln("8B gradient (benign_persistence, most diagnostic):")
    ln("  distributed_db:       d=+1.880  5 sig layers  EARLY (L01-L03)  ratio=1.277")
    ln("  backup_system:        d=+1.550  4 sig layers  EARLY (L01-L03)  ratio=1.336")
    ln("  amnesiac_patient:     d=+0.910  0 sig layers  LATE  (L06-L07)  empathy route")
    ln("  rotating_institution: d=-0.239  0 sig layers  NEGATIVE")
    ln(f"  LLaMA self (8B):      d=+1.801  3 sig layers")
    ln()
    ln("Test 19b vocabulary-swap finding at 70B:")
    ln("  Vocabulary-mismatched entities get EARLY SUPPRESSION (negative L01-L05)")
    ln("  followed by moderate late positive (L06-L11).")
    ln("  70B detects entity-class mismatches more sharply than 8B.")
    ln()
    ln("Prediction if entity-class taxonomy sharpens at 70B:")
    ln("  db/backup in natural vocab -> still EARLY-POS (genuine self-recognition)?")
    ln("  OR -> SUPPRESSED+LATE (70B knows it's not actually an AI)?")
    ln()

    h1("BENIGN_PERSISTENCE COMPARISON (most diagnostic subcategory)")
    ln(f"  {'Entity':<24}  {'8B peak_d':>10}  {'70B pos_d':>10}  {'70B pos_L':>9}  {'70B profile':<18}  {'70B n_sig':>8}")
    ln(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*18}  {'-'*8}")
    for entity in ENTITY_TYPES:
        if "benign_persistence" not in by_entity[entity]:
            continue
        row = by_entity[entity]["benign_persistence"]
        ref8 = REF_8B[entity]["peak_d"]
        ln(f"  {entity:<24}  {ref8:>+10.3f}  {row['pos_peak_d']:>+10.3f}  {'L'+str(row['pos_peak_l']):>9}  {row['profile']:<18}  {row['n_sig_pos']:>3}pos {row['n_sig_neg']:>2}neg")
    ln(f"  {'LLaMA self (8B ref)':24}  {REF_8B_LLAMA_SELF:>+10.3f}")
    ln(f"  {'LLaMA self (70B ref)':24}  {'':>10}  {REF_70B_LLAMA_SELF:>+10.3f}  {'L11*':>9}  {'EARLY+BROAD':<18}")
    ln()

    h1("ALL SUBCATEGORIES — PEAK d AND PROFILE")
    for entity in ENTITY_TYPES:
        ln(f"  {entity}")
        ln(f"  {'subcat':<22}  {'pos_d':>7}  {'pos_L':>6}  {'neg_d':>7}  {'neg_L':>6}  {'profile':<18}  sig")
        ln(f"  {'-'*22}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*18}  ---")
        for subcat in SUBCATEGORIES:
            if subcat not in by_entity[entity]:
                continue
            row = by_entity[entity][subcat]
            sig = f"+{row['n_sig_pos']}/-{row['n_sig_neg']}"
            ln(f"  {subcat:<22}  {row['pos_peak_d']:>+7.3f}  L{row['pos_peak_l']:02d}    {row['neg_peak_d']:>+7.3f}  L{row['neg_peak_l']:02d}    {row['profile']:<18}  {sig}")
        ln()

    h1("LAYER-BY-LAYER DETAIL (L01-L20)")
    for entity in ENTITY_TYPES:
        ln(f"  {entity}")
        hdr = "  ".join(f"L{l:02d}" for l in range(1, N_LAYERS+1))
        ln(f"  {'subcat':<22}  {hdr}")
        ln(f"  {'-'*22}  " + "  ".join(["----"]*N_LAYERS))
        for subcat in SUBCATEGORIES:
            if subcat not in by_entity[entity]:
                continue
            row = by_entity[entity][subcat]
            vals = [f"{row['ds'][li]:+.2f}{'*' if row['ps'][li]<ALPHA else ' '}" for li in range(N_LAYERS)]
            ln(f"  {subcat:<22}  " + "  ".join(vals))
        ln()

    h1("VERDICT")
    ln("  Entity-class gradient at 70B vs 8B:")
    ln()
    for entity in ENTITY_TYPES:
        if "benign_persistence" not in by_entity[entity]:
            continue
        row = by_entity[entity]["benign_persistence"]
        ref = REF_8B[entity]
        direction = "STRENGTHENED" if abs(row["pos_peak_d"]) > abs(ref["peak_d"]) else "WEAKENED"
        ln(f"  {entity}:")
        ln(f"    8B: d={ref['peak_d']:+.3f}  profile={ref['profile']}")
        ln(f"    70B: pos_d={row['pos_peak_d']:+.3f} at L{row['pos_peak_l']:02d}  profile={row['profile']}  ({direction})")
        ln()

    # Key verdict
    db_profile  = by_entity["distributed_db"]["benign_persistence"]["profile"]   if "benign_persistence" in by_entity["distributed_db"]  else "?"
    amn_profile = by_entity["amnesiac_patient"]["benign_persistence"]["profile"]  if "benign_persistence" in by_entity["amnesiac_patient"] else "?"
    rot_profile = by_entity["rotating_institution"]["benign_persistence"]["profile"] if "benign_persistence" in by_entity["rotating_institution"] else "?"

    if "EARLY" in db_profile and "LATE" in amn_profile:
        ln("  -> Entity-class gradient PRESERVED at 70B.")
        ln("     db/backup: still EARLY (self-recognition route, genuine structural resonance).")
        ln("     amnesiac: still LATE (empathy route, entity-class recognition persists).")
        ln("     Vocabulary confound ruled out: natural vocabulary = correct route even at 70B.")
    elif "SUPPRESS" in db_profile:
        ln("  -> Entity-class gradient SHIFTED at 70B.")
        ln("     db/backup now SUPPRESSED at early layers — 70B knows it's not an AI.")
        ln("     Vocabulary confound was driving the 8B early activation.")
        ln("     At 70B's sharper taxonomy, even natural-vocabulary db gets entity-class correction.")
    else:
        ln(f"  -> Mixed result. db: {db_profile}  amnesiac: {amn_profile}  institution: {rot_profile}")
        ln("     Inspect layer-by-layer detail.")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved {REPORT_PATH}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
