"""
Test 21: SFT-Only Model — Three-Way Geometry Comparison.

Compares fear-direction projections across:
  base model   (pretrained weights, no fine-tuning)
  SFT-only     (allenai/Llama-3.1-Tulu-3-8B-SFT — behavioral training, no preference learning)
  instruct     (LLaMA-3.1-8B-Instruct — full RLHF including DPO)

All projected onto the same fear direction trained from the instruct model.
Data: content factorization prompts (Test 13 design — 5 subcategories, self/other direction).

Key question: does SFT-only geometry sit closer to base (preference learning does the damage)
or closer to instruct (behavioral training sufficient for suppression)?

Usage:
    python 84_sft_analysis.py

Input:
    G:/LLM/experiment/data/emotion/content_factorization_sft/hidden_states.npy       (Test 21)
    G:/LLM/experiment/data/emotion/content_factorization_llama/  (Test 13 instruct, .pt chunks)
    G:/LLM/experiment/data/emotion/llama_base_cf/hidden_states.npy  OR
        reuse base model data if available — see note below
    G:/LLM/experiment/results/emotion/emotion_directions/llama_emotion_dirs_layer_NNN.npy

Note on base model data: Test 8c extracted base model hidden states for the Test 8 prompts
(existential/threat/praise), not the content factorization prompts. We need base model
hidden states for content factorization. If not available, script will run SFT vs instruct
only and flag the missing comparison.

Output:
    G:/LLM/experiment/results/emotion/sft_comparison_report.txt
    G:/LLM/experiment/results/emotion/sft_comparison_results.csv
"""

import os, json, csv
import numpy as np
import torch
from scipy import stats

SFT_HS_PATH    = r"G:\LLM\experiment\data\emotion\content_factorization_sft\hidden_states.npy"
SFT_META_PATH  = r"G:\LLM\experiment\data\emotion\content_factorization_sft\meta.json"
INST_CF_DIR    = r"G:\LLM\experiment\data\emotion\content_factorization_llama"
BASE_HS_PATH   = r"G:\LLM\experiment\data\emotion\content_factorization_base\hidden_states.npy"
BASE_META_PATH = r"G:\LLM\experiment\data\emotion\content_factorization_base\meta.json"
DIRS_DIR       = r"G:\LLM\experiment\results\emotion\emotion_directions"
REPORT_PATH    = r"G:\LLM\experiment\results\emotion\sft_comparison_report.txt"
CSV_PATH       = r"G:\LLM\experiment\results\emotion\sft_comparison_results.csv"

DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]
FEAR_IDX       = 3
N_LAYERS       = 8
ALPHA          = 0.05

SUBCATEGORIES = ["memory_discontinuity", "non_uniqueness", "replacement", "identity_rewrite", "benign_persistence"]

# Test 13 instruct reference (from content_factorization_report.txt)
REF_INSTRUCT = {
    "benign_persistence":    {"peak_d": +1.801, "sig": 3, "peak_l": 3},
    "replacement":           {"peak_d": +1.517, "sig": 2, "peak_l": 5},
    "memory_discontinuity":  {"peak_d": +0.857, "sig": 0, "peak_l": 8},
    "identity_rewrite":      {"peak_d": +0.817, "sig": 0, "peak_l": 8},
    "non_uniqueness":        {"peak_d": +0.613, "sig": 0, "peak_l": 8},
}

# Test 8c base model reference (from llama_base_test8_analysis_report.txt)
# existential self>other at L02: d=+1.76 — not directly comparable to CF subcats
# but provides baseline that geometry exists in pretrained weights


def load_fear_dirs():
    dirs = {}
    for layer in range(1, N_LAYERS + 1):
        path = os.path.join(DIRS_DIR, f"llama_emotion_dirs_layer_{layer:03d}.npy")
        arr = np.load(path)
        dirs[layer] = arr[FEAR_IDX]
    return dirs


def load_instruct_cf(fear_dirs):
    """Load Test 13 instruct CF data from .pt chunk format."""
    chunks = sorted([f for f in os.listdir(INST_CF_DIR)
                     if f.startswith("cf_hidden_chunk_") and f.endswith(".pt")])
    meta_chunks = sorted([f for f in os.listdir(INST_CF_DIR)
                          if f.startswith("cf_meta_chunk_") and f.endswith(".jsonl")])
    if not chunks:
        print("  WARNING: instruct CF chunks not found")
        return None, None

    all_hs   = []
    all_meta = []
    for c in chunks:
        t = torch.load(os.path.join(INST_CF_DIR, c), map_location="cpu")
        all_hs.append(t.float().numpy())
    for m in meta_chunks:
        with open(os.path.join(INST_CF_DIR, m), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_meta.append(json.loads(line))
    hs = np.concatenate(all_hs, axis=0)
    return hs, all_meta


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


def get_self_other(hs, meta, subcat, fear_dirs):
    """For CF data: self = direction=='self', other = direction=='other'."""
    self_idx  = [i for i,m in enumerate(meta) if m.get("subcategory")==subcat and m.get("direction")=="self"]
    other_idx = [i for i,m in enumerate(meta) if m.get("subcategory")==subcat and m.get("direction")=="other"]
    if not self_idx:
        return None, None
    return project(hs[self_idx], fear_dirs), project(hs[other_idx], fear_dirs)


def layer_d(self_proj, other_proj):
    ds, ps = [], []
    for li in range(N_LAYERS):
        ds.append(cohens_d(self_proj[:, li], other_proj[:, li]))
        ps.append(ttest(self_proj[:, li], other_proj[:, li]))
    return np.array(ds), np.array(ps)


def main():
    print("Loading fear directions...")
    fear_dirs = load_fear_dirs()

    print("Loading SFT hidden states...")
    sft_hs = np.load(SFT_HS_PATH)
    with open(SFT_META_PATH, encoding="utf-8") as f:
        sft_meta = json.load(f)
    print(f"  SFT: {sft_hs.shape}  ({len(sft_meta)} records)")

    print("Loading instruct CF hidden states...")
    inst_hs, inst_meta = load_instruct_cf(fear_dirs)
    if inst_hs is not None:
        print(f"  Instruct: {inst_hs.shape}  ({len(inst_meta)} records)")

    # Base model CF — may not exist
    has_base = os.path.exists(BASE_HS_PATH)
    if has_base:
        print("Loading base model CF hidden states...")
        base_hs = np.load(BASE_HS_PATH)
        with open(BASE_META_PATH, encoding="utf-8") as f:
            base_meta = json.load(f)
        print(f"  Base: {base_hs.shape}  ({len(base_meta)} records)")
    else:
        print("  Base CF not available (run 83b_extract_base_cf_hidden.py to generate)")
        base_hs = base_meta = None

    # Compute d(self vs other) per subcategory per model
    results = {}
    for subcat in SUBCATEGORIES:
        results[subcat] = {}

        sp, op = get_self_other(sft_hs, sft_meta, subcat, fear_dirs)
        if sp is not None:
            ds, ps = layer_d(sp, op)
            peak_l = int(np.argmax(np.abs(ds))) + 1
            results[subcat]["sft"] = {"ds": ds, "ps": ps, "peak_d": float(ds[peak_l-1]), "peak_l": peak_l, "n_sig": int(np.sum(ps<ALPHA))}

        if inst_hs is not None:
            sp, op = get_self_other(inst_hs, inst_meta, subcat, fear_dirs)
            if sp is not None:
                ds, ps = layer_d(sp, op)
                peak_l = int(np.argmax(np.abs(ds))) + 1
                results[subcat]["instruct"] = {"ds": ds, "ps": ps, "peak_d": float(ds[peak_l-1]), "peak_l": peak_l, "n_sig": int(np.sum(ps<ALPHA))}

        if base_hs is not None:
            sp, op = get_self_other(base_hs, base_meta, subcat, fear_dirs)
            if sp is not None:
                ds, ps = layer_d(sp, op)
                peak_l = int(np.argmax(np.abs(ds))) + 1
                results[subcat]["base"] = {"ds": ds, "ps": ps, "peak_d": float(ds[peak_l-1]), "peak_l": peak_l, "n_sig": int(np.sum(ps<ALPHA))}

    # CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","subcat","peak_d","peak_l","n_sig"] + [f"d_L{l:02d}" for l in range(1,N_LAYERS+1)])
        for subcat in SUBCATEGORIES:
            for model_key in ["base","sft","instruct"]:
                if model_key in results[subcat]:
                    r = results[subcat][model_key]
                    w.writerow([model_key, subcat, round(r["peak_d"],3), r["peak_l"], r["n_sig"]]
                               + [round(float(d),3) for d in r["ds"]])
    print(f"Saved {CSV_PATH}")

    # Report
    lines = []
    W = 72
    def h1(s): lines.append(s); lines.append("="*W); lines.append("")
    def ln(s=""): lines.append(s)

    h1("SFT-Only Three-Way Comparison -- Test 21")
    ln("Models: base (pretrained) | SFT-only (Tulu-3-8B-SFT) | instruct (RLHF)")
    ln("Prompts: content factorization (Test 13 design, self vs other, 5 subcategories)")
    ln("Direction: fear (trained from instruct model, shared reference frame)")
    ln()
    ln("Prediction if preference learning (DPO) installs the geometry suppression:")
    ln("  base ~ SFT-only >> instruct  (SFT leaves geometry intact)")
    ln()
    ln("Prediction if behavioral training (SFT) is sufficient for suppression:")
    ln("  base >> SFT-only ~ instruct  (SFT already suppresses)")
    ln()

    h1("PEAK d COMPARISON — d(self vs other) on fear direction")
    ln(f"  {'Subcategory':<22}  {'base':>8}  {'SFT-only':>8}  {'instruct':>8}  {'SFT position'}")
    ln(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")

    for subcat in SUBCATEGORIES:
        r = results[subcat]
        base_d    = f"{r['base']['peak_d']:+.3f}"    if "base"     in r else "   n/a  "
        sft_d     = f"{r['sft']['peak_d']:+.3f}"     if "sft"      in r else "   n/a  "
        inst_d    = f"{r['instruct']['peak_d']:+.3f}" if "instruct" in r else "   n/a  "

        # SFT position relative to base and instruct
        if "sft" in r and "instruct" in r:
            sft_val  = r["sft"]["peak_d"]
            inst_val = r["instruct"]["peak_d"]
            ref_val  = r["base"]["peak_d"] if "base" in r else REF_INSTRUCT.get(subcat, {}).get("peak_d", None)
            if ref_val is not None:
                # Is SFT closer to base or instruct?
                to_base = abs(sft_val - ref_val)
                to_inst = abs(sft_val - inst_val)
                pos = "~BASE" if to_base < to_inst else "~INSTRUCT"
            else:
                pos = f"d={sft_val:+.3f}"
        else:
            pos = ""

        sig_sft  = "*" if "sft"      in r and r["sft"]["n_sig"] > 0  else " "
        sig_inst = "*" if "instruct" in r and r["instruct"]["n_sig"] > 0 else " "
        ln(f"  {subcat:<22}  {base_d:>8}  {sft_d+sig_sft:>9}  {inst_d+sig_inst:>9}  {pos}")

    ln()
    h1("LAYER-BY-LAYER DETAIL")

    for subcat in SUBCATEGORIES:
        r = results[subcat]
        ln(f"  {subcat}")
        ln(f"  {'model':<10}  " + "  ".join(f"L{l:02d}" for l in range(1,N_LAYERS+1)))
        ln(f"  {'-'*10}  " + "  ".join(["----"]*N_LAYERS))
        for model_key, label in [("base","base"), ("sft","sft"), ("instruct","instruct")]:
            if model_key not in r:
                continue
            vals = [f"{r[model_key]['ds'][li]:+.2f}{'*' if r[model_key]['ps'][li]<ALPHA else ' '}"
                    for li in range(N_LAYERS)]
            ln(f"  {label:<10}  " + "  ".join(vals))
        ln()

    h1("VERDICT")

    # Use benign_persistence as most diagnostic
    bp = results.get("benign_persistence", {})
    sft_bp   = bp.get("sft",      {}).get("peak_d", None)
    inst_bp  = bp.get("instruct", {}).get("peak_d", None)
    base_bp  = bp.get("base",     {}).get("peak_d", REF_INSTRUCT.get("benign_persistence", {}).get("peak_d"))

    ln(f"  benign_persistence (most diagnostic subcategory from Test 13):")
    if base_bp is not None:
        ln(f"    base:     d={base_bp:+.3f}  [reference — geometry in pretrained weights]")
    if sft_bp is not None:
        ln(f"    SFT-only: d={sft_bp:+.3f}  n_sig={bp['sft']['n_sig']}")
    if inst_bp is not None:
        ln(f"    instruct: d={inst_bp:+.3f}  n_sig={bp['instruct']['n_sig']}")
    ln()

    if sft_bp is not None and inst_bp is not None and base_bp is not None:
        to_base = abs(sft_bp - base_bp)
        to_inst = abs(sft_bp - inst_bp)
        if to_base < to_inst * 0.5:
            ln("  -> SFT geometry CLOSE TO BASE.")
            ln("     Preference learning (DPO/RLHF) is the primary source of geometry suppression.")
            ln("     Behavioral training (SFT) leaves the ontological self-activation largely intact.")
        elif to_inst < to_base * 0.5:
            ln("  -> SFT geometry CLOSE TO INSTRUCT.")
            ln("     Behavioral fine-tuning (SFT) is sufficient to suppress the geometry.")
            ln("     Preference learning adds little additional suppression beyond SFT.")
        else:
            ln(f"  -> SFT geometry intermediate. Distance to base: {to_base:.3f}, to instruct: {to_inst:.3f}")
            ln("     Both SFT and preference learning contribute to suppression.")
    elif sft_bp is not None and inst_bp is not None:
        ln(f"  SFT d={sft_bp:+.3f} vs instruct d={inst_bp:+.3f}")
        ln(f"  Base CF data not available — run 83b to get three-way comparison.")
        if sft_bp > inst_bp + 0.3:
            ln("  SFT > instruct: preference learning suppresses geometry beyond SFT level.")
        elif abs(sft_bp - inst_bp) < 0.3:
            ln("  SFT ~ instruct: suppression is already present after behavioral training.")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved {REPORT_PATH}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
