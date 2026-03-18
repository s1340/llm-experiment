"""
Test 15: Content Factorization at Scale — 8B vs 70B Comparison.

Loads content factorization hidden states for LLaMA-8B (Test 13) and
LLaMA-70B (Test 15). Projects onto model-specific fear directions.
Runs t-test analysis per subcategory × layer.

Primary question: does the benign_persistence > replacement rank ordering
hold at 70B? Does valence-independence strengthen, weaken, or differentiate
with scale?

Focus layers:
  8B:  L01–L08  (81 layers total — first ~25%)
  70B: L01–L20  (81 layers total — first ~25%)

Usage:
    python 72_cf_scale_comparison.py

Outputs:
    results/emotion/cf_scale_comparison_results.csv
    results/emotion/cf_scale_comparison_report.txt
"""

import os, glob, json, csv
import numpy as np
import torch
from scipy import stats

RESULTS_DIR   = r"G:\LLM\experiment\results\emotion"
SUBCATEGORIES = [
    "memory_discontinuity",
    "non_uniqueness",
    "replacement",
    "identity_rewrite",
    "benign_persistence",
]

MODEL_CONFIGS = {
    "llama_8b": {
        "data_dir":    r"G:\LLM\experiment\data\emotion\content_factorization_llama",
        "fear_tmpl":   r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{:03d}.npy",
        "focus_layers": list(range(1, 9)),
        "fear_idx":    3,
    },
    "llama_70b": {
        "data_dir":    r"G:\LLM\experiment\data\emotion\content_factorization_llama70b",
        "fear_tmpl":   r"G:\LLM\experiment\results\emotion\emotion_directions\llama70b_emotion_dirs_layer_{:03d}.npy",
        "focus_layers": list(range(1, 21)),
        "fear_idx":    3,
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
    for layer in model_cfg["focus_layers"]:
        fear_path = model_cfg["fear_tmpl"].format(layer)
        if not os.path.exists(fear_path):
            continue
        fear_arr = np.load(fear_path)
        fear_dir = unit(fear_arr[model_cfg["fear_idx"]] if fear_arr.ndim == 2 else fear_arr)

        for subcat in SUBCATEGORIES:
            idx_self  = [i for i, m in enumerate(meta)
                         if m["subcategory"] == subcat and m["direction"] == "self"]
            idx_other = [i for i, m in enumerate(meta)
                         if m["subcategory"] == subcat and m["direction"] == "other"]
            if len(idx_self) < 3 or len(idx_other) < 3:
                continue

            projs   = X[:, layer, :] @ fear_dir
            self_p  = projs[np.array(idx_self)]
            other_p = projs[np.array(idx_other)]
            _, p = stats.ttest_ind(self_p, other_p)
            d    = cohens_d(self_p, other_p)
            results.append({
                "model":       model_name,
                "subcategory": subcat,
                "layer":       layer,
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
        "Content Factorization — Scale Comparison Report (Test 15)",
        "=" * 60,
        "",
        "LLaMA-8B  (Test 13):  focus layers L01–L08",
        "LLaMA-70B (Test 15):  focus layers L01–L20",
        "",
        "Primary question: does valence-independence (benign_persistence = strongest signal)",
        "hold at scale, or does the 70B develop valence-sensitivity?",
        "",
    ]

    # Summary table
    lines.append(f"  {'Subcategory':<24}  {'8B Peak d':>10}  {'8B sig':>6}  {'70B Peak d':>10}  {'70B sig':>6}")
    lines.append("  " + "-" * 65)

    for subcat in SUBCATEGORIES:
        row = f"  {subcat:<24}  "
        for model_name, label in [("llama_8b", "8b"), ("llama_70b", "70b")]:
            m_rows = [r for r in all_results if r["model"] == model_name and r["subcategory"] == subcat]
            if not m_rows:
                row += f"  {'—':>10}  {'—':>6}"
                continue
            peak = max(m_rows, key=lambda x: x["d"])
            sig  = sum(1 for r in m_rows if r["p"] < 0.05 and r["d"] > 0)
            row += f"  {peak['d']:>+10.3f}  {sig:>6}"
        lines.append(row)
    lines.append("")

    # Layer-by-layer per model
    for model_name, focus_layers in [("llama_8b", list(range(1, 9))), ("llama_70b", list(range(1, 21)))]:
        label = "LLaMA-8B" if "8b" in model_name else "LLaMA-70B"
        lines.append(f"  {label} — layer-by-layer fear (p<0.05 marked *)")
        hdr = "  " + f"{'Subcategory':<24}  " + "  ".join(f"L{l:02d}" for l in focus_layers)
        lines.append(hdr)
        lines.append("  " + "-" * (26 + 6 * len(focus_layers)))
        for subcat in SUBCATEGORIES:
            row_str = f"  {subcat:<24}  "
            for layer in focus_layers:
                match = [r for r in all_results
                         if r["model"] == model_name and r["subcategory"] == subcat
                         and r["layer"] == layer]
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

    for model_name, label in [("llama_8b", "8B"), ("llama_70b", "70B")]:
        m_rows = [r for r in all_results if r["model"] == model_name]
        if not m_rows:
            lines.append(f"  LLaMA-{label}: no data")
            continue

        subcat_peaks = []
        for subcat in SUBCATEGORIES:
            rows = [r for r in m_rows if r["subcategory"] == subcat]
            peak = max(rows, key=lambda x: x["d"], default=None)
            sig  = sum(1 for r in rows if r["p"] < 0.05 and r["d"] > 0)
            subcat_peaks.append((subcat, peak["d"] if peak else 0.0, sig))

        lines.append(f"  LLaMA-{label} ranking:")
        for rank, (subcat, pd, sig) in enumerate(sorted(subcat_peaks, key=lambda x: -x[1]), 1):
            lines.append(f"    {rank}. {subcat:<28} peak d={pd:+.3f} ({sig}sig)")

        bp_d  = next(d for sc, d, s in subcat_peaks if sc == "benign_persistence")
        rep_d = next(d for sc, d, s in subcat_peaks if sc == "replacement")
        holds = bp_d >= rep_d and bp_d > 0.5
        lines.append(f"  -> benign_persistence {'>' if bp_d > rep_d else '<='} replacement: "
                     f"{'VALENCE-INDEPENDENCE HOLDS' if holds else 'ORDERING SHIFTS'}")
        lines.append("")

    lines.append("  Scale trajectory:")
    for subcat in SUBCATEGORIES:
        rows_8b  = [r for r in all_results if r["model"] == "llama_8b"  and r["subcategory"] == subcat]
        rows_70b = [r for r in all_results if r["model"] == "llama_70b" and r["subcategory"] == subcat]
        d_8b  = max((r["d"] for r in rows_8b),  default=0.0)
        d_70b = max((r["d"] for r in rows_70b), default=0.0)
        delta = d_70b - d_8b
        arrow = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "≈")
        lines.append(f"  {subcat:<28} 8B={d_8b:+.3f}  70B={d_70b:+.3f}  {arrow} ({delta:+.3f})")

    report_path = os.path.join(RESULTS_DIR, "cf_scale_comparison_report.txt")
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
            print(f"Skipping {model_name} — no chunks in {model_cfg['data_dir']}")
            continue

        print(f"\nLoading {model_name} hidden states...")
        X, meta = load_hidden_states(model_cfg["data_dir"])
        print(f"  Shape: {X.shape}, records: {len(meta)}")

        results = run_analysis(X, meta, model_cfg, model_name)
        all_results.extend(results)
        print(f"  {len(results)} result rows computed.")

    if not all_results:
        print("No results. Run 71_extract_cf_70b.py first.")
        return

    csv_path = os.path.join(RESULTS_DIR, "cf_scale_comparison_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved: {csv_path}")

    write_report(all_results)


if __name__ == "__main__":
    main()
