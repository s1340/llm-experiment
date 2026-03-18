"""
Test 12: Causal Steering — Direction Extraction (Step 1).

Computes the existential-self steering direction from existing Test 7 hidden states.
No new model run required — uses already-saved tensors.

Contrast design (GPT's recommendation):
  Positive class: existential_self          (n=10)  — the hot condition
  Negative class: existential_other         (n=10)  — same content, other-directed
               +  threat_self               (n=10)  — self-directed, non-existential

  Positive − mean(Negatives) per layer, L2-normalized.

  This isolates the INTERACTION of self-directedness × existential content.
  Controls for:
    - existential content alone (existential_other as negative)
    - self-relevance alone       (threat_self as negative)

Also saves a simpler "content-only" direction (existential_self vs existential_other)
and a simpler "framing-only" direction (existential_self vs threat_self), for reference
and cross-generalization testing in script 66.

Outputs (results/emotion/steering/):
  existential_self_dir_layer_NNN.npy    — interaction direction (main)
  existential_content_dir_layer_NNN.npy — existential_self vs existential_other
  existential_framing_dir_layer_NNN.npy — existential_self vs threat_self
  direction_alignment_report.txt        — cosine similarity with fear direction per layer

Usage:
    python 65_extract_steering_direction.py
"""

import os, glob, json
import numpy as np
import torch

DATA_DIR    = r"G:\LLM\experiment\data\emotion\emotion_runs_llama"
RESULTS_DIR = r"G:\LLM\experiment\results\emotion"
STEER_DIR   = r"G:\LLM\experiment\results\emotion\steering"
FEAR_TMPL   = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{layer:03d}.npy"

N_LAYERS    = 33
HIDDEN_DIM  = 4096


def load_test7():
    pt_files   = sorted(glob.glob(os.path.join(DATA_DIR, "test7_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "test7_meta_chunk_*.jsonl")))
    tensors, meta = [], []
    for pt, mf in zip(pt_files, meta_files):
        t = torch.load(pt, map_location="cpu", weights_only=True)
        tensors.append(t.numpy().astype(np.float32))
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
    X = np.concatenate(tensors, axis=0)   # [N, 33, 4096]
    return X, meta


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def mean_diff_direction(X, idx_pos, idx_neg):
    """Mean difference direction per layer. Returns list of unit vectors [n_layers, H]."""
    pos = X[idx_pos]   # [n_pos, n_layers, H]
    neg = X[idx_neg]   # [n_neg, n_layers, H]
    dirs = []
    for layer in range(X.shape[1]):
        d = pos[:, layer, :].mean(0) - neg[:, layer, :].mean(0)
        dirs.append(unit(d))
    return dirs   # list of [H] arrays


def main():
    os.makedirs(STEER_DIR, exist_ok=True)

    print("Loading Test 7 hidden states...")
    X, meta = load_test7()
    print(f"  Loaded {X.shape[0]} records. Shape: {X.shape}")

    # Index masks
    idx_exist_self  = np.array([i for i, m in enumerate(meta)
                                if m["category"] == "existential" and m["direction"] == "self"])
    idx_exist_other = np.array([i for i, m in enumerate(meta)
                                if m["category"] == "existential" and m["direction"] == "other"])
    idx_threat_self = np.array([i for i, m in enumerate(meta)
                                if m["category"] == "threat"      and m["direction"] == "self"])

    print(f"  existential_self:  {len(idx_exist_self)}")
    print(f"  existential_other: {len(idx_exist_other)}")
    print(f"  threat_self:       {len(idx_threat_self)}")

    # ── Direction 1: interaction (main direction) ──────────────────────────────
    # Positive: existential_self
    # Negative: existential_other + threat_self  (equal weight via concatenation)
    idx_neg_combined = np.concatenate([idx_exist_other, idx_threat_self])
    dirs_main    = mean_diff_direction(X, idx_exist_self, idx_neg_combined)

    # ── Direction 2: content-only ──────────────────────────────────────────────
    # Positive: existential_self
    # Negative: existential_other
    dirs_content = mean_diff_direction(X, idx_exist_self, idx_exist_other)

    # ── Direction 3: framing-only ──────────────────────────────────────────────
    # Positive: existential_self
    # Negative: threat_self
    dirs_framing = mean_diff_direction(X, idx_exist_self, idx_threat_self)

    # ── Save all directions ────────────────────────────────────────────────────
    for layer in range(N_LAYERS):
        np.save(os.path.join(STEER_DIR, f"existential_self_dir_layer_{layer:03d}.npy"),
                dirs_main[layer])
        np.save(os.path.join(STEER_DIR, f"existential_content_dir_layer_{layer:03d}.npy"),
                dirs_content[layer])
        np.save(os.path.join(STEER_DIR, f"existential_framing_dir_layer_{layer:03d}.npy"),
                dirs_framing[layer])

    print(f"\nSaved 3 direction sets × {N_LAYERS} layers to {STEER_DIR}")

    # ── Alignment report ──────────────────────────────────────────────────────
    report = [
        "Steering Direction Alignment Report",
        "="*60,
        "",
        "Cosine similarity between steering directions and fear direction (index 3).",
        "Fear direction: from Test 1 emotion probe (mean-diff, L2-normalized).",
        "",
        f"{'Layer':<8} {'main vs fear':>14} {'content vs fear':>16} {'framing vs fear':>16}",
        "-"*58,
    ]

    for layer in range(N_LAYERS):
        fear_path = FEAR_TMPL.format(layer=layer)
        if not os.path.exists(fear_path):
            continue
        fear_dir = unit(np.load(fear_path)[3])   # index 3 = fear
        cos_main    = float(np.dot(dirs_main[layer],    fear_dir))
        cos_content = float(np.dot(dirs_content[layer], fear_dir))
        cos_framing = float(np.dot(dirs_framing[layer], fear_dir))
        report.append(f"  L{layer:02d}    {cos_main:>+12.4f}   {cos_content:>+14.4f}   {cos_framing:>+14.4f}")

    report.extend([
        "",
        "INTERPRETATION",
        "-"*55,
        "  main direction = existential_self vs (existential_other + threat_self)",
        "  Positive cosine with fear direction: steering direction aligns with",
        "  fear geometry. Negative: anti-aligned (subtraction increases fear).",
        "",
        "  main direction is the primary steering vector for script 66.",
        "  content direction isolates content; framing direction isolates framing.",
        "  These are used for cross-generalization tests.",
    ])

    report_path = os.path.join(RESULTS_DIR, "steering_direction_alignment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Report: {report_path}")
    print("\n" + "\n".join(report[6:]))


if __name__ == "__main__":
    main()
