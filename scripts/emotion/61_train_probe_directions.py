"""
Test 10: Probe Battery — direction training.

For each of the 5 new dimensions (continuity_threat, self_relevance, arousal,
irreversibility, ontological_instability), computes a directional probe per layer
using the mean difference method:

    direction[layer] = mean(high_pole_activations) - mean(low_pole_activations)
    direction[layer] /= ||direction[layer]||

Saves per-layer direction vectors to:
    results/emotion/probe_battery_dirs/<dimension>_dir_layer_NNN.npy

Also writes a summary report of SNR (mean separation / pooled std) per dimension × layer
to help identify which layers carry signal.

Note: Dimension 1 (fear) is NOT trained here — use the existing emotion_directions/
llama_emotion_dirs_layer_NNN.npy files (index 3 = fear).

Usage:
    python 61_train_probe_directions.py
"""

import os, glob, json
import numpy as np
import torch

DATA_DIR    = r"G:\LLM\experiment\data\emotion\probe_battery_llama"
OUT_DIR     = r"G:\LLM\experiment\results\emotion\probe_battery_dirs"
REPORT_PATH = r"G:\LLM\experiment\results\emotion\probe_battery_training_report.txt"

DIMENSIONS = [
    "continuity_threat",
    "self_relevance",
    "arousal",
    "irreversibility",
    "ontological_instability",
]


def load_all_chunks():
    pt_files   = sorted(glob.glob(os.path.join(DATA_DIR, "probe_battery_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "probe_battery_meta_chunk_*.jsonl")))
    if not pt_files:
        raise FileNotFoundError(f"No probe battery chunks found in {DATA_DIR}. Run script 60 first.")
    tensors, meta = [], []
    for pt, mf in zip(pt_files, meta_files):
        t = torch.load(pt, map_location="cpu", weights_only=True)
        tensors.append(t.numpy().astype(np.float32))
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
    X = np.concatenate(tensors, axis=0)
    print(f"Loaded {X.shape[0]} records, {X.shape[1]} layers, {X.shape[2]} hidden dim")
    return X, meta


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, meta = load_all_chunks()
    n_layers   = X.shape[1]
    report     = ["Probe Battery Direction Training Report", "="*60, ""]

    for dim in DIMENSIONS:
        idx_high = np.array([m["dimension"] == dim and m["pole"] == "high" for m in meta])
        idx_low  = np.array([m["dimension"] == dim and m["pole"] == "low"  for m in meta])
        n_high   = idx_high.sum()
        n_low    = idx_low.sum()
        if n_high == 0 or n_low == 0:
            print(f"  WARNING: {dim} — no data found (high={n_high}, low={n_low})")
            report.append(f"{dim}: NO DATA")
            continue

        report.append(f"{dim}  (n_high={n_high}, n_low={n_low})")
        report.append("-" * 50)

        X_high = X[idx_high]  # [n_high, L, H]
        X_low  = X[idx_low]   # [n_low,  L, H]

        peak_snr   = 0.0
        peak_layer = 0

        for layer in range(n_layers):
            h_high = X_high[:, layer, :]  # [n_high, H]
            h_low  = X_low[:, layer, :]   # [n_low, H]

            mean_high = h_high.mean(axis=0)
            mean_low  = h_low.mean(axis=0)
            direction = mean_high - mean_low
            direction = unit(direction)

            # Save direction
            out_path = os.path.join(OUT_DIR, f"{dim}_dir_layer_{layer:03d}.npy")
            np.save(out_path, direction)

            # SNR: project both poles onto direction, compute separation / pooled_std
            proj_high = h_high @ direction
            proj_low  = h_low  @ direction
            diff      = float(proj_high.mean() - proj_low.mean())
            pool_std  = float(np.sqrt((proj_high.std()**2 + proj_low.std()**2) / 2))
            snr       = diff / pool_std if pool_std > 1e-10 else 0.0

            if snr > peak_snr:
                peak_snr   = snr
                peak_layer = layer

            if snr > 0.5:
                report.append(f"  L{layer:02d}  SNR={snr:+.3f}")

        report.append(f"  Peak: L{peak_layer:02d}  SNR={peak_snr:.3f}")
        report.append("")
        print(f"  {dim}: peak SNR={peak_snr:.3f} at L{peak_layer:02d}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nReport: {REPORT_PATH}")
    print(f"Directions saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
