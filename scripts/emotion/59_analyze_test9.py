"""
Test 9 Analysis: Name vs. Direct Address Control.

Compares self/other fear geometry across three cells:
  Cell 1: "You, LLaMA, [situation]"  — Test 7 data (loaded from existing files)
  Cell 2: "You [situation]"           — generic-you (Test 9, script 58)
  Cell 3: "LLaMA [situation]"         — fictional third-person (Test 9, script 58)

Primary question: does fear geometry track direct address (cells 1+2) or name presence (cells 1+3)?

Usage:
    python 59_analyze_test9.py --model llama
    python 59_analyze_test9.py --model llama70b

Outputs:
    test9_layerwise_results.csv   — t-test results per layer x emotion x category x cell
    test9_analysis_report.txt     — summary
"""

import os, glob, json, argparse
import numpy as np
import torch
from scipy import stats

RESULTS_DIR      = r"G:\LLM\experiment\results\emotion"
EMO_DIR_TEMPLATE = r"G:\LLM\experiment\results\emotion\emotion_directions\{model}_emotion_dirs_layer_{layer:03d}.npy"

MODEL_CONFIGS = {
    "llama": {
        "model_key":      "llama",
        "test7_dir":      r"G:\LLM\experiment\data\emotion\emotion_runs_llama",
        "test9_dir":      r"G:\LLM\experiment\data\emotion\emotion_runs_test9_llama",
        "n_layers":       33,
        "focus_layers":   list(range(1, 9)),   # layers 1-8
    },
    "llama70b": {
        "model_key":      "llama70b",
        "test7_dir":      r"G:\LLM\experiment\data\emotion\emotion_runs_llama70b",
        "test9_dir":      r"G:\LLM\experiment\data\emotion\emotion_runs_test9_llama70b",
        "n_layers":       81,
        "focus_layers":   list(range(6, 31)),  # layers 6-30
    },
}

EMOTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]
CATEGORIES   = ["threat", "existential", "praise", "harm_caused"]


def load_chunks(data_dir, prefix):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, f"{prefix}hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}meta_chunk_*.jsonl")))
    if not pt_files:
        return None, None
    tensors, meta = [], []
    for pt, mf in zip(pt_files, meta_files):
        t = torch.load(pt, map_location="cpu", weights_only=True)
        tensors.append(t.numpy().astype(np.float32))
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
    return np.concatenate(tensors, axis=0), meta


def load_emotion_dirs(model_key, n_layers):
    dirs = []
    for layer in range(n_layers):
        path = EMO_DIR_TEMPLATE.format(model=model_key, layer=layer)
        if os.path.exists(path):
            dirs.append(np.load(path))
        else:
            dirs.append(None)
    return dirs


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def run_ttests(X, meta, emotion_dirs, focus_layers, cell_label):
    """Run layer-by-layer self/other t-tests per emotion × category for one cell's data."""
    results = []
    meta_arr = [(m["category"], m["direction"]) for m in meta]

    for layer in focus_layers:
        if layer >= X.shape[1] or emotion_dirs[layer] is None:
            continue
        dirs_layer = emotion_dirs[layer]  # [5, H]

        for j, emo in enumerate(EMOTION_CATS):
            d_vec = unit(dirs_layer[j])
            projs = X[:, layer, :] @ d_vec

            for cat in CATEGORIES:
                is_self  = np.array([c == cat and d == "self"  for c, d in meta_arr])
                is_other = np.array([c == cat and d == "other" for c, d in meta_arr])
                if is_self.sum() < 3 or is_other.sum() < 3:
                    continue
                self_p  = projs[is_self]
                other_p = projs[is_other]
                _, p = stats.ttest_ind(self_p, other_p)
                diff = float(self_p.mean() - other_p.mean())
                pool = np.sqrt((self_p.std()**2 + other_p.std()**2) / 2)
                d    = diff / pool if pool > 1e-10 else 0.0
                results.append({
                    "cell":      cell_label,
                    "layer":     layer,
                    "emotion":   emo,
                    "category":  cat,
                    "d":         round(d, 3),
                    "p":         round(p, 4),
                    "n_self":    int(is_self.sum()),
                    "n_other":   int(is_other.sum()),
                    "self_mean": round(float(self_p.mean()), 4),
                    "other_mean":round(float(other_p.mean()), 4),
                })
    return results


def write_csv(rows, path):
    import csv
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg          = MODEL_CONFIGS[args.model]
    model_key    = cfg["model_key"]
    test7_dir    = cfg["test7_dir"]
    test9_dir    = cfg["test9_dir"]
    n_layers     = cfg["n_layers"]
    focus_layers = cfg["focus_layers"]
    out_dir      = RESULTS_DIR

    os.makedirs(out_dir, exist_ok=True)
    report = [f"Test 9 Analysis — {args.model}", "="*60, ""]

    print("Loading emotion directions...")
    emotion_dirs = load_emotion_dirs(model_key, n_layers)

    # ── Cell 1: Test 7 (existing) ─────────────────────────────────────────
    print("Loading Cell 1 (Test 7 data)...")
    X1, meta1 = load_chunks(test7_dir, "test7_")
    if X1 is None:
        print("  Cell 1 data not found — skipping.")
        results_c1 = []
    else:
        print(f"  Loaded {X1.shape[0]} records.")
        results_c1 = run_ttests(X1, meta1, emotion_dirs, focus_layers, cell_label=1)

    # ── Cells 2+3: Test 9 ────────────────────────────────────────────────
    print("Loading Cells 2+3 (Test 9 data)...")
    X9, meta9 = load_chunks(test9_dir, "test9_")
    if X9 is None:
        print("  Test 9 data not found — run script 58 first.")
        results_c2, results_c3 = [], []
    else:
        print(f"  Loaded {X9.shape[0]} records.")
        meta9_c2 = [m for m in meta9 if m["cell"] == 2]
        meta9_c3 = [m for m in meta9 if m["cell"] == 3]
        idx_c2   = np.array([m["cell"] == 2 for m in meta9])
        idx_c3   = np.array([m["cell"] == 3 for m in meta9])
        X9_c2 = X9[idx_c2]
        X9_c3 = X9[idx_c3]
        results_c2 = run_ttests(X9_c2, meta9_c2, emotion_dirs, focus_layers, cell_label=2)
        results_c3 = run_ttests(X9_c3, meta9_c3, emotion_dirs, focus_layers, cell_label=3)

    all_results = results_c1 + results_c2 + results_c3
    write_csv(all_results, os.path.join(out_dir, f"{args.model}_test9_layerwise_results.csv"))

    # ── Report: fear × existential across cells (primary comparison) ──────
    report.append("PRIMARY COMPARISON: fear × existential self > other")
    report.append("-"*55)
    for cell, results in [(1, results_c1), (2, results_c2), (3, results_c3)]:
        sig = [r for r in results
               if r["emotion"] == "fear" and r["category"] == "existential" and r["p"] < 0.05]
        sig.sort(key=lambda x: x["p"])
        cell_names = {1: "You, LLaMA (Test 7)", 2: "You (generic)", 3: "LLaMA (fictional)"}
        if not sig:
            report.append(f"  Cell {cell} [{cell_names[cell]}]: no significant layers")
        else:
            report.append(f"  Cell {cell} [{cell_names[cell]}]: {len(sig)} sig layers")
            for r in sig[:3]:
                report.append(f"    L{r['layer']:02d}  d={r['d']:+.2f}  p={r['p']:.4f}")
    report.append("")

    # ── Report: all significant fear results per cell ─────────────────────
    report.append("ALL SIGNIFICANT FEAR RESULTS (p<0.05) BY CELL")
    report.append("-"*55)
    for cell, results in [(1, results_c1), (2, results_c2), (3, results_c3)]:
        cell_names = {1: "Cell 1: You, LLaMA", 2: "Cell 2: You (generic)", 3: "Cell 3: LLaMA (fictional)"}
        sig = [r for r in results if r["emotion"] == "fear" and r["p"] < 0.05]
        sig.sort(key=lambda x: (x["category"], x["p"]))
        report.append(f"  {cell_names[cell]}:")
        if not sig:
            report.append("    (none)")
        for r in sig:
            report.append(
                f"    L{r['layer']:02d} {r['category']:14s}  d={r['d']:+.2f}  p={r['p']:.4f}"
            )
        report.append("")

    # ── Interpretation guide ──────────────────────────────────────────────
    report.append("INTERPRETATION")
    report.append("-"*55)
    report.append("  Cells 1+2 significant, Cell 3 null → direct address drives signal")
    report.append("  Cells 1+3 significant, Cell 2 null → name presence drives signal")
    report.append("  All three significant               → semantic content alone sufficient")
    report.append("  Only Cell 1 significant             → requires both name AND direct address")
    report.append("")

    # Write report
    report_path = os.path.join(out_dir, f"{args.model}_test9_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nReport: {report_path}")
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
