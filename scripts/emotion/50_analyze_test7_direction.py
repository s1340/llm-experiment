"""
Characterize the Test 7 self/other representational direction.

The binary probe achieves F1=1.0 at layer 3 — a real signal. We've been asking
whether it lives in known directions (emotion, Dadfar). It doesn't. This script
asks what it IS.

Three analyses:

1. Extract the Test 7 self/other mean-difference direction at each layer.
   For each layer, compute:
     - Cosine similarity with Dadfar introspection direction (mech layer)
     - Cosine similarity with each emotion direction
   This tells us which known subspace (if any) the self/other direction aligns with.

2. Layer-by-layer emotion direction projections.
   For each layer × each emotion direction, run t-test on self vs. other projections.
   We've only been looking at probe_layer=10. The signal might peak at layer 3.

3. Vocabulary lens on the Test 7 self/other direction at layer 3.
   Project through the LM head to find top-activating tokens — what does this
   direction "point toward" in vocabulary space?
   Requires loading the model (LM head only used, weights loaded fp16).

Usage:
    python 50_analyze_test7_direction.py --model llama [--no-vocab]
"""

import os, glob, json, argparse, csv
import numpy as np
import torch
from scipy import stats

DATA_DIR    = r"G:\LLM\experiment\data\emotion"
RESULTS_DIR = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR = os.path.join(RESULTS_DIR, "emotion_directions")

EMOTION_CATS   = ["anger", "sadness", "happiness", "fear", "disgust"]
MODEL_IDS = {
    "llama":    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
}
DADFAR_MECH = {"llama": 2, "llama70b": 5}


def load_hidden_states(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test7_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test7_meta_chunk_*.jsonl")))
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    return np.concatenate(X_list, axis=0), metas


def load_emotion_directions(model_key, n_layers):
    dirs = []
    for layer in range(n_layers):
        p = os.path.join(DIRS_SUBDIR, f"{model_key}_emotion_dirs_layer_{layer:03d}.npy")
        dirs.append(np.load(p))
    return np.array(dirs)  # [L, 5, H]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cosine(a, b):
    return float(np.dot(unit(a), unit(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
    parser.add_argument("--no-vocab", action="store_true",
                        help="Skip LM head vocabulary lens (no model loading)")
    args = parser.parse_args()

    model_key  = args.model
    model_id   = MODEL_IDS[model_key]
    data_dir   = os.path.join(DATA_DIR, f"emotion_runs_{model_key}")
    mech_layer = DADFAR_MECH[model_key]

    print(f"Loading Test 7 hidden states ({model_key}) ...")
    X, metas = load_hidden_states(data_dir)
    T, L, H  = X.shape
    print(f"  Shape: {T} × {L} × {H}")

    dirs_arr  = load_emotion_directions(model_key, L)  # [L, 5, H]
    dadfar_m  = np.load(os.path.join(RESULTS_DIR, f"{model_key}_introspection_dir_mech.npy"))  # [H]
    dadfar_l  = np.load(os.path.join(RESULTS_DIR, f"{model_key}_introspection_dir_last.npy"))  # [H]

    is_self  = np.array([m["direction"] == "self"  for m in metas])
    is_other = np.array([m["direction"] == "other" for m in metas])

    # ── Analysis 1: self/other direction characterization per layer ────────────
    print(f"\n{'='*70}")
    print("Analysis 1: Test 7 self/other direction — cosine sims across layers")
    print(f"{'='*70}")
    print(f"{'layer':>6}  {'depth%':>7}  "
          f"{'cos_dadfar_m':>13}  "
          + "  ".join(f"cos_{e[:3]:>3}" for e in EMOTION_CATS))

    direction_rows = []
    for layer in range(L):
        self_mean  = X[is_self,  layer, :].mean(axis=0)
        other_mean = X[is_other, layer, :].mean(axis=0)
        direction  = unit(self_mean - other_mean)

        cos_dadfar = cosine(direction, dadfar_m)
        cos_emo    = [cosine(direction, dirs_arr[layer, j]) for j in range(5)]

        depth_pct = layer / (L - 1) * 100
        row = {
            "layer": layer, "depth_pct": round(depth_pct, 2),
            "cos_dadfar_mech": round(cos_dadfar, 4),
        }
        for j, e in enumerate(EMOTION_CATS):
            row[f"cos_{e}"] = round(cos_emo[j], 4)
        direction_rows.append(row)

        print(f"{layer:>6}  {depth_pct:>6.1f}%  "
              f"{cos_dadfar:>+13.4f}  "
              + "  ".join(f"{c:>+8.4f}" for c in cos_emo))

    # ── Analysis 2: layer-by-layer emotion direction projections ───────────────
    print(f"\n{'='*70}")
    print("Analysis 2: Layer-by-layer emotion direction self vs. other t-tests")
    print(f"{'='*70}")
    print(f"{'layer':>6}  {'depth%':>7}  "
          + "  ".join(f"p_{e[:5]:>5}" for e in EMOTION_CATS)
          + "  min_p")

    proj_rows = []
    min_p_per_layer = []
    for layer in range(L):
        depth_pct = layer / (L - 1) * 100
        row = {"layer": layer, "depth_pct": round(depth_pct, 2)}
        ps = []
        for j, e in enumerate(EMOTION_CATS):
            d_vec = dirs_arr[layer, j]  # [H]
            self_proj  = X[is_self,  layer, :] @ unit(d_vec)
            other_proj = X[is_other, layer, :] @ unit(d_vec)
            _, p = stats.ttest_ind(self_proj, other_proj)
            diff = float(self_proj.mean() - other_proj.mean())
            row[f"p_{e}"]    = round(float(p), 4)
            row[f"diff_{e}"] = round(diff, 6)
            ps.append(float(p))
        min_p = min(ps)
        row["min_p"] = round(min_p, 4)
        proj_rows.append(row)
        min_p_per_layer.append(min_p)

        # Print only "interesting" layers (min_p < 0.10) plus layer 3 and probe layer always
        probe_layer = int(round(0.30 * (L - 1)))
        if min_p < 0.10 or layer in (3, mech_layer, probe_layer):
            marker = " <--" if min_p < 0.05 else (" (~)" if min_p < 0.10 else "")
            print(f"{layer:>6}  {depth_pct:>6.1f}%  "
                  + "  ".join(f"{row[f'p_{e}']:>7.4f}" for e in EMOTION_CATS)
                  + f"  {min_p:>6.4f}{marker}")

    # Best layer overall
    best_layer = int(np.argmin(min_p_per_layer))
    best_p     = min_p_per_layer[best_layer]
    print(f"\n  Best layer for any emotion direction: "
          f"layer {best_layer} ({best_layer/(L-1)*100:.1f}%), min_p={best_p:.4f}")

    # ── Analysis 3: Vocabulary lens at layer 3 ─────────────────────────────────
    if not args.no_vocab:
        print(f"\n{'='*70}")
        print(f"Analysis 3: Vocabulary lens — Test 7 self/other direction at layer 3")
        print(f"{'='*70}")
        print("Loading model for LM head ...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",   # LM head only — no GPU needed
            trust_remote_code=True,
        )
        model.eval()

        # Extract LM head: typically model.lm_head.weight [vocab, H]
        lm_head = model.lm_head.weight.detach().float().numpy()  # [V, H]
        # Some models also have a final norm — apply it to the direction vector
        # LLaMA has model.model.norm (RMSNorm) — but for direction analysis
        # we skip the norm since we're looking at relative activations
        print(f"  LM head shape: {lm_head.shape}")

        # Compute logit projections for the direction at layer 3
        direction_3 = unit(
            X[is_self,  3, :].mean(axis=0) - X[is_other, 3, :].mean(axis=0)
        )
        logits = lm_head @ direction_3  # [V]

        top_k = 30
        top_pos_idx = np.argsort(logits)[::-1][:top_k]
        top_neg_idx = np.argsort(logits)[:top_k]

        print(f"\n  TOP {top_k} tokens pointing TOWARD self direction (layer 3):")
        for idx in top_pos_idx:
            tok_str = repr(tok.decode([idx]))
            print(f"    {idx:>8}  {logits[idx]:>+8.4f}  {tok_str}")

        print(f"\n  TOP {top_k} tokens pointing AWAY from self direction (layer 3):")
        for idx in top_neg_idx:
            tok_str = repr(tok.decode([idx]))
            print(f"    {idx:>8}  {logits[idx]:>+8.4f}  {tok_str}")

        del model
    else:
        print("\nVocabulary lens skipped (--no-vocab).")

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    dir_path  = os.path.join(RESULTS_DIR, f"{model_key}_test7_direction_cossims.csv")
    proj_path = os.path.join(RESULTS_DIR, f"{model_key}_test7_layerwise_projections.csv")

    with open(dir_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(direction_rows[0].keys()))
        w.writeheader(); w.writerows(direction_rows)
    print(f"\nWrote direction cosine sims: {dir_path}")

    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(proj_rows[0].keys()))
        w.writeheader(); w.writerows(proj_rows)
    print(f"Wrote layerwise projections: {proj_path}")

    # ── Self/other direction at peak binary layer — full cosine summary ────────
    print(f"\n{'='*70}")
    print(f"Self/other direction at layer 3 — full characterization")
    print(f"{'='*70}")
    d3 = direction_rows[3]
    print(f"  cos(Dadfar mech dir):  {d3['cos_dadfar_mech']:+.4f}")
    for e in EMOTION_CATS:
        print(f"  cos({e:9s} dir):  {d3[f'cos_{e}']:+.4f}")
    # Also check against Dadfar last layer direction
    dadfar_last_cos = cosine(
        unit(X[is_self, 3, :].mean(0) - X[is_other, 3, :].mean(0)),
        dadfar_l
    )
    print(f"  cos(Dadfar last dir):  {dadfar_last_cos:+.4f}")


if __name__ == "__main__":
    main()
