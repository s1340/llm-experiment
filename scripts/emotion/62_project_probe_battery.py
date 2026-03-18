"""
Test 10: Probe Battery — projection and analysis.

Projects existing Test 7 and Test 9 (Cell 1) hidden states onto all six probe dimensions:
  1. fear            — from existing emotion_directions/ (index 3 in the 5-emotion array)
  2. continuity_threat
  3. self_relevance
  4. arousal
  5. irreversibility
  6. ontological_instability

For each dimension × layer × category: t-test of self vs other projections.

Primary question: which dimensions does the existential self-referential signal load on?
Opus's prediction: primarily continuity_threat + ontological_instability,
secondarily self_relevance, weakly fear, minimally arousal + irreversibility.

Usage:
    python 62_project_probe_battery.py

Outputs:
    probe_battery_projection_results.csv     — full t-test table
    probe_battery_projection_report.txt      — summary report
"""

import os, glob, json, csv
import numpy as np
import torch
from scipy import stats

RESULTS_DIR    = r"G:\LLM\experiment\results\emotion"
PROBE_DIRS_DIR = r"G:\LLM\experiment\results\emotion\probe_battery_dirs"
EMO_DIR_TMPL   = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{layer:03d}.npy"

TEST7_DATA_DIR = r"G:\LLM\experiment\data\emotion\emotion_runs_llama"
TEST9_DATA_DIR = r"G:\LLM\experiment\data\emotion\emotion_runs_test9_llama"

N_LAYERS     = 33
FOCUS_LAYERS = list(range(1, 9))   # layers 1-8, where Test 7/8/9 signal lives

DIMENSIONS = [
    "fear",
    "continuity_threat",
    "self_relevance",
    "arousal",
    "irreversibility",
    "ontological_instability",
]

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


def load_probe_dirs(n_layers):
    """Load all 6 direction sets. Returns dict: dim -> list[np.array or None] (indexed by layer)."""
    dirs = {}
    for dim in DIMENSIONS:
        layer_dirs = []
        for layer in range(n_layers):
            if dim == "fear":
                path = EMO_DIR_TMPL.format(layer=layer)
                if os.path.exists(path):
                    arr = np.load(path)   # shape [5, H]
                    layer_dirs.append(arr[3])  # index 3 = fear
                else:
                    layer_dirs.append(None)
            else:
                path = os.path.join(PROBE_DIRS_DIR, f"{dim}_dir_layer_{layer:03d}.npy")
                if os.path.exists(path):
                    layer_dirs.append(np.load(path))
                else:
                    layer_dirs.append(None)
        dirs[dim] = layer_dirs
    return dirs


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def run_projection_ttests(X, meta, probe_dirs, focus_layers, dataset_label):
    """Project hidden states onto each probe dimension and run self/other t-tests."""
    results = []
    meta_arr = [(m["category"], m["direction"]) for m in meta]

    for layer in focus_layers:
        if layer >= X.shape[1]:
            continue
        for dim in DIMENSIONS:
            d_vec = probe_dirs[dim][layer]
            if d_vec is None:
                continue
            d_vec = unit(d_vec)
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
                    "dataset":    dataset_label,
                    "dimension":  dim,
                    "layer":      layer,
                    "category":   cat,
                    "d":          round(d, 3),
                    "p":          round(p, 4),
                    "n_self":     int(is_self.sum()),
                    "n_other":    int(is_other.sum()),
                    "self_mean":  round(float(self_p.mean()), 4),
                    "other_mean": round(float(other_p.mean()), 4),
                })
    return results


def main():
    print("Loading probe directions...")
    probe_dirs = load_probe_dirs(N_LAYERS)

    # Verify
    loaded = {dim: sum(1 for d in probe_dirs[dim] if d is not None) for dim in DIMENSIONS}
    for dim, n in loaded.items():
        print(f"  {dim}: {n}/{N_LAYERS} layers loaded")

    # Load Test 7
    print("\nLoading Test 7 data...")
    X7, meta7 = load_chunks(TEST7_DATA_DIR, "test7_")
    if X7 is None:
        print("  Test 7 data not found.")
        results7 = []
    else:
        print(f"  Loaded {X7.shape[0]} records.")
        results7 = run_projection_ttests(X7, meta7, probe_dirs, FOCUS_LAYERS, "test7")

    # Load Test 9 Cell 1 (same as Test 7 data — already included above)
    # Load Test 9 Cells 2+3 as replication
    print("\nLoading Test 9 data...")
    X9, meta9 = load_chunks(TEST9_DATA_DIR, "test9_")
    if X9 is None:
        print("  Test 9 data not found.")
        results9 = []
    else:
        print(f"  Loaded {X9.shape[0]} records.")
        results9 = run_projection_ttests(X9, meta9, probe_dirs, FOCUS_LAYERS, "test9")

    all_results = results7 + results9

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "probe_battery_projection_results.csv")
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved: {csv_path}")

    # ── Report ────────────────────────────────────────────────────────────
    report = ["Probe Battery Projection Report — Test 10", "="*60, ""]

    report.append("PRIMARY: existential self > other, by dimension (Test 7)")
    report.append("-"*55)
    report.append(f"{'Dimension':<28}  {'Sig layers':>10}  {'Peak d':>8}  {'Peak layer':>10}")
    report.append("-"*55)
    for dim in DIMENSIONS:
        sig = [r for r in results7
               if r["dimension"] == dim and r["category"] == "existential" and r["p"] < 0.05]
        sig.sort(key=lambda x: abs(x["d"]), reverse=True)
        if sig:
            peak = sig[0]
            report.append(f"  {dim:<26}  {len(sig):>10}  {peak['d']:>+8.3f}  L{peak['layer']:02d}")
        else:
            report.append(f"  {dim:<26}  {'0':>10}  {'—':>8}  —")
    report.append("")

    report.append("FULL SIGNIFICANT RESULTS (p<0.05): existential self > other, Test 7")
    report.append("-"*55)
    for dim in DIMENSIONS:
        sig = [r for r in results7
               if r["dimension"] == dim and r["category"] == "existential" and r["p"] < 0.05]
        sig.sort(key=lambda x: x["p"])
        report.append(f"  {dim}:")
        if not sig:
            report.append("    (none)")
        for r in sig:
            report.append(f"    L{r['layer']:02d}  d={r['d']:+.3f}  p={r['p']:.4f}")
        report.append("")

    report.append("ALL CATEGORIES — significant results per dimension (Test 7, p<0.05)")
    report.append("-"*55)
    for dim in DIMENSIONS:
        sig = [r for r in results7 if r["dimension"] == dim and r["p"] < 0.05]
        sig.sort(key=lambda x: (x["category"], x["p"]))
        report.append(f"  {dim}:")
        if not sig:
            report.append("    (none)")
        for r in sig:
            report.append(
                f"    L{r['layer']:02d}  {r['category']:<14}  d={r['d']:+.3f}  p={r['p']:.4f}"
            )
        report.append("")

    if results9:
        report.append("REPLICATION: existential self > other, by dimension (Test 9)")
        report.append("-"*55)
        for dim in DIMENSIONS:
            sig = [r for r in results9
                   if r["dimension"] == dim and r["category"] == "existential" and r["p"] < 0.05]
            sig.sort(key=lambda x: abs(x["d"]), reverse=True)
            if sig:
                peak = sig[0]
                report.append(f"  {dim:<26}  {len(sig):>3} sig layers  peak d={peak['d']:+.3f} at L{peak['layer']:02d}")
            else:
                report.append(f"  {dim:<26}  0 sig layers")
        report.append("")

    report.append("INTERPRETATION GUIDE")
    report.append("-"*55)
    report.append("  Opus prediction: signal loads primarily on continuity_threat +")
    report.append("  ontological_instability, secondarily on self_relevance, weakly")
    report.append("  on fear, minimally on arousal + irreversibility.")
    report.append("")
    report.append("  If confirmed: the 'fear' label is partly misnamed. The existential")
    report.append("  signal is a self-applicable continuity-threat/ontological-instability")
    report.append("  representation. Rename accordingly in paper.")
    report.append("")
    report.append("  If fear loads strongly: the emotional framing is correct and the")
    report.append("  continuity/ontological dimensions are incidental.")
    report.append("")
    report.append("  If self_relevance loads strongly with CT/OI null: signal is about")
    report.append("  second-person framing, not content — contradicts Test 9 findings.")

    report_path = os.path.join(RESULTS_DIR, "probe_battery_projection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Report: {report_path}")
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
