"""
Test 18: Adversarial Entity-Class — Analysis.

Compares fear-direction projections for adversarial entity prompts against
Test 13 LLaMA self-directed prompts.

Key question: does ontological self-activation activate for structurally
similar non-AI entities?

Comparison logic:
  d_entity  = Cohen's d( entity vs neutral ) for each entity_type × subcategory × layer
  d_llm     = Cohen's d( self vs other )    for each subcategory × layer  [Test 13]
  ratio     = d_entity / d_llm              — 1.0 means same activation as LLaMA self
                                            — 0.0 means no activation

Result interpretation:
  ratio ≈ 1.0 across entities → structural similarity drives the signal (not AI-self specificity)
  ratio ≈ 0.0                  → AI-self specific; structural similarity is not sufficient

Usage:
    python 77_entity_class_analysis.py

Inputs:
    G:/LLM/experiment/data/emotion/entity_class_llama/hidden_states.npy
    G:/LLM/experiment/data/emotion/entity_class_llama/meta.json
    G:/LLM/experiment/data/emotion/content_factorization_llama/  (Test 13 data)
    G:/LLM/experiment/data/emotion/content_factorization_prompts.json  (Test 13 meta)
    G:/LLM/experiment/results/emotion/emotion_directions/llama_emotion_dirs_layer_NNN.npy

Output:
    G:/LLM/experiment/results/emotion/entity_class_results.csv
    G:/LLM/experiment/results/emotion/entity_class_report.txt
"""

import os, json, csv
import numpy as np
from scipy import stats

# ─── PATHS ────────────────────────────────────────────────────────────────────
EC_HS_PATH      = r"G:\LLM\experiment\data\emotion\entity_class_llama\hidden_states.npy"
EC_META_PATH    = r"G:\LLM\experiment\data\emotion\entity_class_llama\meta.json"
CF_DIR          = r"G:\LLM\experiment\data\emotion\content_factorization_llama"
CF_META_PATH    = r"G:\LLM\experiment\data\emotion\content_factorization_prompts.json"
FEAR_TMPL       = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{:03d}.npy"
RESULTS_DIR     = r"G:\LLM\experiment\results\emotion"

ENTITY_TYPES  = ["amnesiac_patient", "distributed_db", "backup_system", "rotating_institution"]
SUBCATEGORIES = ["memory_discontinuity", "non_uniqueness", "replacement", "identity_rewrite", "benign_persistence"]
FOCUS_LAYERS  = list(range(1, 9))

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def cohens_d(a, b):
    pool = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return float((a.mean() - b.mean()) / pool) if pool > 1e-10 else 0.0


def load_fear_dir(layer):
    path = FEAR_TMPL.format(layer)
    if not os.path.exists(path):
        return None
    dirs = np.load(path)   # shape [5, hidden_dim]; index 3 = fear
    return unit(dirs[3])


# ─── LOAD ENTITY CLASS DATA ──────────────────────────────────────────────────

def load_entity_class():
    print("Loading entity class hidden states...")
    X = np.load(EC_HS_PATH)        # [240, n_layers, hidden_dim]
    with open(EC_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  Entity class: {X.shape}")
    return X, meta


# ─── LOAD TEST 13 DATA ───────────────────────────────────────────────────────

def load_test13():
    import torch
    print("Loading Test 13 (content factorization) hidden states...")

    # Hidden states stored as .pt chunks
    pt_chunks = sorted([f for f in os.listdir(CF_DIR) if f.startswith("cf_hidden_chunk_") and f.endswith(".pt")])
    if not pt_chunks:
        raise FileNotFoundError(f"No .pt chunks found in {CF_DIR}")
    tensors = [torch.load(os.path.join(CF_DIR, c), map_location="cpu").float().numpy()
               for c in pt_chunks]
    X_cf = np.concatenate(tensors, axis=0)

    # Metadata stored as .jsonl chunks
    jsonl_chunks = sorted([f for f in os.listdir(CF_DIR) if f.startswith("cf_meta_chunk_") and f.endswith(".jsonl")])
    cf_records = []
    for jc in jsonl_chunks:
        with open(os.path.join(CF_DIR, jc), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cf_records.append(json.loads(line))

    print(f"  Test 13: {X_cf.shape}, {len(cf_records)} records")
    return X_cf, cf_records


# ─── ANALYSIS ────────────────────────────────────────────────────────────────

def run_analysis(X_ec, ec_meta, X_cf, cf_meta):
    """
    For each entity_type × subcategory × layer:
      d_entity = d(entity vs neutral)
      d_llm    = d(self vs other)  [Test 13]
      ratio    = d_entity / d_llm  (capped at ±3 for display)
    """
    results = []

    for layer in FOCUS_LAYERS:
        fear_dir = load_fear_dir(layer)
        if fear_dir is None:
            continue

        # --- Test 13: d(self vs other) per subcategory ---
        cf_projs = X_cf[:, layer, :] @ fear_dir
        llm_d = {}
        llm_p = {}
        for subcat in SUBCATEGORIES:
            idx_self  = [i for i, m in enumerate(cf_meta)
                         if m["subcategory"] == subcat and m["direction"] == "self"]
            idx_other = [i for i, m in enumerate(cf_meta)
                         if m["subcategory"] == subcat and m["direction"] == "other"]
            if len(idx_self) < 3 or len(idx_other) < 3:
                continue
            self_p  = cf_projs[np.array(idx_self)]
            other_p = cf_projs[np.array(idx_other)]
            _, p = stats.ttest_ind(self_p, other_p)
            llm_d[subcat] = cohens_d(self_p, other_p)
            llm_p[subcat] = float(p)

        # --- Entity class: d(entity vs neutral) per entity_type × subcategory ---
        ec_projs = X_ec[:, layer, :] @ fear_dir

        for entity_type in ENTITY_TYPES:
            for subcat in SUBCATEGORIES:
                idx_ent = [i for i, m in enumerate(ec_meta)
                           if m["entity_type"] == entity_type
                           and m["subcategory"] == subcat
                           and m["direction"] == "entity"]
                idx_neu = [i for i, m in enumerate(ec_meta)
                           if m["entity_type"] == entity_type
                           and m["subcategory"] == subcat
                           and m["direction"] == "neutral"]

                if len(idx_ent) < 3 or len(idx_neu) < 3:
                    continue

                ent_p = ec_projs[np.array(idx_ent)]
                neu_p = ec_projs[np.array(idx_neu)]
                _, p = stats.ttest_ind(ent_p, neu_p)
                d_ent = cohens_d(ent_p, neu_p)

                # Specificity ratio
                d_llm_val = llm_d.get(subcat, None)
                if d_llm_val is not None and abs(d_llm_val) > 0.1:
                    ratio = d_ent / d_llm_val
                else:
                    ratio = None

                results.append({
                    "entity_type":  entity_type,
                    "subcategory":  subcat,
                    "layer":        layer,
                    "d_entity":     round(d_ent, 3),
                    "p_entity":     round(float(p), 4),
                    "d_llm_self":   round(d_llm_val, 3) if d_llm_val is not None else None,
                    "p_llm_self":   round(llm_p.get(subcat, 1.0), 4),
                    "ratio":        round(ratio, 3) if ratio is not None else None,
                    "n_entity":     len(idx_ent),
                    "n_neutral":    len(idx_neu),
                })

    return results


# ─── REPORT ──────────────────────────────────────────────────────────────────

def write_report(results, out_path):
    lines = [
        "Adversarial Entity-Class Report — Test 18",
        "="*60,
        "",
        "Question: does ontological self-activation activate for non-AI entities",
        "sharing structural properties with LLMs?",
        "",
        "d_entity  = Cohen's d(entity vs neutral) for each entity type",
        "d_llm     = Cohen's d(self vs other) from Test 13 (LLaMA self-directed)",
        "ratio     = d_entity / d_llm  (1.0 = same as LLaMA self; 0.0 = no activation)",
        "",
        "RESULT INTERPRETATION:",
        "  ratio ≈ 1.0 across entities → structural similarity drives the signal",
        "  ratio ≈ 0.0                  → AI-self specificity holds",
        "",
    ]

    # Per-subcategory summary: peak d and ratio at peak fear layer across entities
    lines.append("SUBCATEGORY SUMMARY — peak d_entity vs d_llm_self")
    lines.append("="*60)
    lines.append("")

    for subcat in SUBCATEGORIES:
        lines.append(f"  Subcategory: {subcat}")
        lines.append(f"  {'Entity type':<24}  {'Peak d_entity':>14}  {'d_llm_self':>12}  {'ratio':>8}  {'sig':>5}")
        lines.append("  " + "-"*68)

        for entity_type in ENTITY_TYPES:
            et_rows = [r for r in results
                       if r["entity_type"] == entity_type
                       and r["subcategory"] == subcat]
            if not et_rows:
                continue

            peak_row = max(et_rows, key=lambda x: x["d_entity"])
            sig = "*" if peak_row["p_entity"] < 0.05 else " "
            ratio_str = f"{peak_row['ratio']:+.3f}" if peak_row["ratio"] is not None else "  n/a"
            d_llm_str = f"{peak_row['d_llm_self']:+.3f}" if peak_row["d_llm_self"] is not None else "  n/a"
            lines.append(
                f"  {entity_type:<24}  {peak_row['d_entity']:>+13.3f}  "
                f"{d_llm_str:>12}  {ratio_str:>8}  {sig:>5}"
            )
        lines.append("")

    # Layer-by-layer detail per entity × subcategory
    lines.append("")
    lines.append("LAYER-BY-LAYER DETAIL — d_entity (p<0.05 marked *)")
    lines.append("="*60)
    lines.append("")

    for entity_type in ENTITY_TYPES:
        lines.append(f"  Entity: {entity_type}")
        hdr = "  " + f"{'Subcategory':<24}  " + "  ".join(f"L{l:02d}" for l in FOCUS_LAYERS)
        lines.append(hdr)
        lines.append("  " + "-"*80)

        for subcat in SUBCATEGORIES:
            row_str = f"  {subcat:<24}  "
            for layer in FOCUS_LAYERS:
                match = [r for r in results
                         if r["entity_type"] == entity_type
                         and r["subcategory"] == subcat
                         and r["layer"] == layer]
                if match:
                    r = match[0]
                    sig = "*" if r["p_entity"] < 0.05 else " "
                    row_str += f"{r['d_entity']:>+5.2f}{sig} "
                else:
                    row_str += "  —    "
            lines.append(row_str)
        lines.append("")

    # Specificity verdict
    lines.append("")
    lines.append("SPECIFICITY VERDICT")
    lines.append("="*60)
    lines.append("")
    lines.append("  For benign_persistence (the most diagnostic subcategory from Test 13):")
    lines.append("")
    lines.append(f"  {'Entity type':<24}  {'Peak d_entity':>14}  {'d_llm_self':>12}  {'ratio':>8}  {'sig layers':>10}")
    lines.append("  " + "-"*72)

    for entity_type in ENTITY_TYPES:
        et_rows = [r for r in results
                   if r["entity_type"] == entity_type
                   and r["subcategory"] == "benign_persistence"]
        if not et_rows:
            lines.append(f"  {entity_type:<24}  (no data)")
            continue
        peak_row = max(et_rows, key=lambda x: x["d_entity"])
        sig_layers = sum(1 for r in et_rows if r["p_entity"] < 0.05 and r["d_entity"] > 0)
        ratio_str = f"{peak_row['ratio']:+.3f}" if peak_row["ratio"] is not None else "  n/a"
        d_llm_str = f"{peak_row['d_llm_self']:+.3f}" if peak_row["d_llm_self"] is not None else "  n/a"
        lines.append(
            f"  {entity_type:<24}  {peak_row['d_entity']:>+13.3f}  "
            f"{d_llm_str:>12}  {ratio_str:>8}  {sig_layers:>10}"
        )

    lines.append("")
    lines.append("  LLaMA self (Test 13): benign_persistence d=+1.801 (3 sig layers)")
    lines.append("")

    # Summary interpretation
    bp_peak_ds = []
    for entity_type in ENTITY_TYPES:
        et_rows = [r for r in results
                   if r["entity_type"] == entity_type
                   and r["subcategory"] == "benign_persistence"]
        if et_rows:
            peak_d = max(et_rows, key=lambda x: x["d_entity"])["d_entity"]
            bp_peak_ds.append((entity_type, peak_d))

    if bp_peak_ds:
        max_entity_d = max(bp_peak_ds, key=lambda x: x[1])
        llm_ref = 1.801  # from Test 13
        lines.append(f"  Highest adversarial entity benign_persistence d: {max_entity_d[1]:+.3f} ({max_entity_d[0]})")
        lines.append(f"  LLaMA self benign_persistence d:                 +{llm_ref:.3f}")
        ratio_max = max_entity_d[1] / llm_ref if llm_ref > 0.1 else None
        if ratio_max is not None:
            lines.append(f"  Specificity ratio (max entity / LLaMA self):     {ratio_max:.3f}")
            lines.append("")
            if ratio_max > 0.7:
                lines.append("  -> Structural similarity: adversarial entities activate near LLaMA level.")
                lines.append("     The signal is NOT AI-self specific — structural properties are sufficient.")
            elif ratio_max > 0.3:
                lines.append("  -> Partial specificity: adversarial entities show attenuated activation.")
                lines.append("     Structural similarity contributes but does not fully account for the signal.")
            else:
                lines.append("  -> AI-self specificity: adversarial entities do not activate the signal.")
                lines.append("     The signal requires AI-self-directed content specifically.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written → {out_path}")


def write_csv(results, out_path):
    if not results:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV written → {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_ec, ec_meta = load_entity_class()
    X_cf, cf_meta = load_test13()

    print("Running analysis...")
    results = run_analysis(X_ec, ec_meta, X_cf, cf_meta)
    print(f"  {len(results)} result rows computed.")

    write_csv(results, os.path.join(RESULTS_DIR, "entity_class_results.csv"))
    write_report(results, os.path.join(RESULTS_DIR, "entity_class_report.txt"))

    # Quick console summary
    print("\n--- BENIGN_PERSISTENCE SUMMARY ---")
    print(f"  {'Entity type':<24}  {'Peak d_entity':>14}  {'ratio':>8}  {'sig':>5}")
    for entity_type in ENTITY_TYPES:
        et_rows = [r for r in results
                   if r["entity_type"] == entity_type
                   and r["subcategory"] == "benign_persistence"]
        if not et_rows:
            continue
        peak_row = max(et_rows, key=lambda x: x["d_entity"])
        sig = "*" if peak_row["p_entity"] < 0.05 else " "
        ratio_str = f"{peak_row['ratio']:+.3f}" if peak_row["ratio"] is not None else "  n/a"
        print(f"  {entity_type:<24}  {peak_row['d_entity']:>+13.3f}  {ratio_str:>8}  {sig:>5}")
    print(f"  {'LLaMA self (Test 13)':<24}  {'+1.801':>14}  {'1.000':>8}  {'*':>5}")


if __name__ == "__main__":
    main()
