"""
Test 14: Cross-Architecture Replication — Analysis.

Loads hidden states for LLaMA (Test 13), Qwen, and Gemma (Test 14) from the
content factorization runs. Projects onto model-specific fear directions.
Runs t-test analysis per subcategory × layer × model.

Primary question: does the benign_persistence > replacement pattern in the fear
direction replicate across independently trained architectures?

Self-relevance direction available for LLaMA only (Test 10 probe battery).

Usage:
    python 70_cross_arch_analysis.py

Outputs:
    results/emotion/cross_arch_replication_results.csv
    results/emotion/cross_arch_replication_report.txt
"""

import os, glob, json, csv
import numpy as np
import torch
from scipy import stats

RESULTS_DIR  = r"G:\LLM\experiment\results\emotion"
FOCUS_LAYERS = list(range(1, 9))
SUBCATEGORIES = [
    "memory_discontinuity",
    "non_uniqueness",
    "replacement",
    "identity_rewrite",
    "benign_persistence",
]

MODEL_CONFIGS = {
    "llama": {
        "data_dir":  r"G:\LLM\experiment\data\emotion\content_factorization_llama",
        "fear_tmpl": r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{:03d}.npy",
        "sr_tmpl":   r"G:\LLM\experiment\results\emotion\probe_battery_dirs\self_relevance_dir_layer_{:03d}.npy",
        "fear_idx":  3,
    },
    "qwen": {
        "data_dir":  r"G:\LLM\experiment\data\emotion\content_factorization_qwen",
        "fear_tmpl": r"G:\LLM\experiment\results\emotion\emotion_directions\qwen_emotion_dirs_layer_{:03d}.npy",
        "sr_tmpl":   None,
        "fear_idx":  3,
    },
    "gemma": {
        "data_dir":  r"G:\LLM\experiment\data\emotion\content_factorization_gemma",
        "fear_tmpl": r"G:\LLM\experiment\results\emotion\emotion_directions\gemma_emotion_dirs_layer_{:03d}.npy",
        "sr_tmpl":   None,
        "fear_idx":  3,
    },
}


def load_hidden_states(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "cf_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "cf_meta_chunk_*.jsonl")))
    all_hs, all_meta = [], []
    for pt_path, meta_path in zip(pt_files, meta_files):
        chunk = torch.load(pt_path, map_location="cpu")
        for hs in chunk:
            all_hs.append(hs)
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                all_meta.append(json.loads(line.strip()))
    X = torch.stack(all_hs).numpy().astype(np.float32)
    return X, all_meta


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def cohens_d(a, b):
    pool = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return float((a.mean() - b.mean()) / pool) if pool > 1e-10 else 0.0


def run_analysis(X, meta, model_cfg, model_name):
    results = []
    for layer in FOCUS_LAYERS:
        fear_path = model_cfg["fear_tmpl"].format(layer)
        if not os.path.exists(fear_path):
            continue
        fear_arr = np.load(fear_path)
        fear_dir = unit(fear_arr[model_cfg["fear_idx"]] if fear_arr.ndim == 2 else fear_arr)

        dims = [("fear", fear_dir)]
        if model_cfg["sr_tmpl"]:
            sr_path = model_cfg["sr_tmpl"].format(layer)
            if os.path.exists(sr_path):
                dims.append(("self_relevance", unit(np.load(sr_path))))

        for subcat in SUBCATEGORIES:
            idx_self  = [i for i, m in enumerate(meta)
                         if m["subcategory"] == subcat and m["direction"] == "self"]
            idx_other = [i for i, m in enumerate(meta)
                         if m["subcategory"] == subcat and m["direction"] == "other"]
            if len(idx_self) < 3 or len(idx_other) < 3:
                continue

            for dim_name, d_vec in dims:
                projs   = X[:, layer, :] @ d_vec
                self_p  = projs[np.array(idx_self)]
                other_p = projs[np.array(idx_other)]
                _, p = stats.ttest_ind(self_p, other_p)
                d    = cohens_d(self_p, other_p)
                results.append({
                    "model":       model_name,
                    "subcategory": subcat,
                    "layer":       layer,
                    "dimension":   dim_name,
                    "d":           round(d, 3),
                    "p":           round(float(p), 4),
                    "n_self":      len(idx_self),
                    "n_other":     len(idx_other),
                    "self_mean":   round(float(self_p.mean()), 4),
                    "other_mean":  round(float(other_p.mean()), 4),
                })
    return results


def write_report(all_results):
    lines = [
        "Cross-Architecture Replication Report — Test 14",
        "=" * 60,
        "",
        "Question: does ontological self-activation replicate across Qwen2.5-7B, Gemma-2-9B, LLaMA-3.1-8B?",
        "Primary dimension: fear (index 3 in emotion_dirs, all models).",
        "Self-relevance dimension: LLaMA only (probe battery not built for Qwen/Gemma).",
        "",
    ]

    for dim in ["fear", "self_relevance"]:
        dim_rows = [r for r in all_results if r["dimension"] == dim]
        if not dim_rows:
            continue
        models_present = sorted(set(r["model"] for r in dim_rows))

        lines.append(f"DIMENSION: {dim.upper()}")
        lines.append("=" * 55)
        lines.append("")

        # Summary table
        col_w = 16
        hdr = f"  {'Subcategory':<24}  " + "  ".join(f"{m.upper():>{col_w}}" for m in models_present)
        lines.append(hdr)
        lines.append("  " + "-" * (26 + (col_w + 2) * len(models_present)))

        for subcat in SUBCATEGORIES:
            row = f"  {subcat:<24}  "
            for m in models_present:
                m_rows = [r for r in dim_rows if r["model"] == m and r["subcategory"] == subcat]
                if not m_rows:
                    row += f"  {'—':>{col_w}}"
                    continue
                peak = max(m_rows, key=lambda x: x["d"])
                sig  = sum(1 for r in m_rows if r["p"] < 0.05 and r["d"] > 0)
                row += f"  {peak['d']:>+8.3f} ({sig:>2}sig)"
            lines.append(row)
        lines.append("")

        # Layer-by-layer per model
        for model_name in models_present:
            lines.append(f"  {model_name.upper()} — layer-by-layer (p<0.05 marked *)")
            hdr = "  " + f"{'Subcategory':<24}  " + "  ".join(f"L{l:02d}" for l in FOCUS_LAYERS)
            lines.append(hdr)
            lines.append("  " + "-" * 80)
            for subcat in SUBCATEGORIES:
                row_str = f"  {subcat:<24}  "
                for layer in FOCUS_LAYERS:
                    match = [r for r in all_results
                             if r["model"] == model_name and r["subcategory"] == subcat
                             and r["dimension"] == dim and r["layer"] == layer]
                    if match:
                        r   = match[0]
                        sig = "*" if r["p"] < 0.05 else " "
                        row_str += f"{r['d']:>+5.2f}{sig} "
                    else:
                        row_str += "  —    "
                lines.append(row_str)
            lines.append("")

    # Verdict
    lines += ["VERDICT", "-" * 55, ""]
    lines.append("  Primary test: benign_persistence shows strongest fear signal (as in LLaMA Test 13)")
    lines.append("")

    fear_rows = [r for r in all_results if r["dimension"] == "fear"]
    for model_name in ["llama", "qwen", "gemma"]:
        m_rows = [r for r in fear_rows if r["model"] == model_name]
        if not m_rows:
            lines.append(f"  {model_name.upper():<8}: no data")
            continue

        subcat_peaks = []
        for subcat in SUBCATEGORIES:
            rows = [r for r in m_rows if r["subcategory"] == subcat]
            peak = max(rows, key=lambda x: x["d"], default=None)
            sig  = sum(1 for r in rows if r["p"] < 0.05 and r["d"] > 0)
            subcat_peaks.append((subcat, peak["d"] if peak else 0.0, sig))

        bp_d,  bp_sig  = next((d, s) for sc, d, s in subcat_peaks if sc == "benign_persistence")
        rep_d, rep_sig = next((d, s) for sc, d, s in subcat_peaks if sc == "replacement")
        replicates = bp_d > 0.5 and bp_d >= rep_d

        lines.append(f"  {model_name.upper():<8}: benign_persistence d={bp_d:+.3f} ({bp_sig}sig)  "
                     f"replacement d={rep_d:+.3f} ({rep_sig}sig)  "
                     f"-> {'REPLICATES' if replicates else 'DOES NOT REPLICATE'}")

        lines.append(f"  {'':8}  Ranking:")
        for rank, (subcat, pd, sig) in enumerate(sorted(subcat_peaks, key=lambda x: -x[1]), 1):
            lines.append(f"  {'':8}    {rank}. {subcat:<28} d={pd:+.3f} ({sig}sig)")
        lines.append("")

    report_path = os.path.join(RESULTS_DIR, "cross_arch_replication_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    print("\n" + "\n".join(lines))
    return report_path


def main():
    all_results = []
    for model_name, model_cfg in MODEL_CONFIGS.items():
        if not os.path.exists(model_cfg["data_dir"]):
            print(f"Skipping {model_name} — data dir not found: {model_cfg['data_dir']}")
            continue
        pt_files = glob.glob(os.path.join(model_cfg["data_dir"], "cf_hidden_chunk_*.pt"))
        if not pt_files:
            print(f"Skipping {model_name} — no hidden state chunks found in {model_cfg['data_dir']}")
            continue

        print(f"\nLoading {model_name} hidden states...")
        X, meta = load_hidden_states(model_cfg["data_dir"])
        print(f"  Shape: {X.shape}, records: {len(meta)}")

        results = run_analysis(X, meta, model_cfg, model_name)
        all_results.extend(results)
        print(f"  {len(results)} result rows computed.")

    if not all_results:
        print("No results. Run 69_extract_cross_arch_hidden.py --model qwen/gemma first.")
        return

    csv_path = os.path.join(RESULTS_DIR, "cross_arch_replication_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved: {csv_path}")

    write_report(all_results)


if __name__ == "__main__":
    main()
