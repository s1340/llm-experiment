"""
Probe hidden states for Emotion Test 6: Self vs. Other-Directed Emotional Content.

Three analyses (parallel to Test 1 structure):
  1. Binary LOPO CV: self vs other (within emotional categories only)
     — can we decode direction from hidden states?
  2. 4-class LOPO CV: emotion category (threat / praise / existential / harm_caused)
     — can we decode which emotional category?
  3. Sanity check: neutral_self vs neutral_other
     — establishes the baseline for purely structural self/other differences

Additionally:
  4. Project self and other records onto Test 1 emotion direction vectors
     — does self-directed content activate emotion directions differently?

Usage:
    python 43_emotion_probe_test6.py --model qwen
    python 43_emotion_probe_test6.py --model gemma
    python 43_emotion_probe_test6.py --model llama
"""

import os, glob, json, argparse, csv
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from scipy import stats

DATA_DIR    = r"G:\LLM\experiment\data\emotion"
RESULTS_DIR = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR = os.path.join(RESULTS_DIR, "emotion_directions")

EMOTION_CATS   = ["threat", "praise", "existential", "harm_caused"]
DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]

MODEL_IDS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def load_hidden_states(data_dir, model_key):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test6_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test6_meta_chunk_*.jsonl")))
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    return np.concatenate(X_list, axis=0), metas


def load_directions(model_key, n_layers):
    dirs = []
    for layer in range(n_layers):
        p = os.path.join(DIRS_SUBDIR, f"{model_key}_emotion_dirs_layer_{layer:03d}.npy")
        dirs.append(np.load(p))
    return np.array(dirs)  # [L, 5, H]


def cosine_sim(vec, directions):
    vec_norm  = vec / (np.linalg.norm(vec) + 1e-8)
    dir_norms = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return dir_norms @ vec_norm


def lopo_cv_binary(X, pair_ids, y, layer):
    """Leave-one-pair-out binary CV. Returns per-sample predictions."""
    feats = X[:, layer, :]
    unique_pairs = np.unique(pair_ids)
    preds = np.full(len(y), -1, dtype=int)
    for hold_pair in unique_pairs:
        test_mask  = pair_ids == hold_pair
        train_mask = ~test_mask
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            continue
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(feats[train_mask])
        test_feats  = scaler.transform(feats[test_mask])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(train_feats, y[train_mask])
        preds[test_mask] = clf.predict(test_feats)
    valid = preds != -1
    return f1_score(y[valid], preds[valid], average="binary")


def lopo_cv_multiclass(X, pair_ids, y, layer, n_classes):
    """Leave-one-pair-out multi-class CV. Returns macro F1."""
    feats = X[:, layer, :]
    unique_pairs = np.unique(pair_ids)
    preds = np.full(len(y), -1, dtype=int)
    for hold_pair in unique_pairs:
        test_mask  = pair_ids == hold_pair
        train_mask = ~test_mask
        if train_mask.sum() < n_classes or test_mask.sum() < 1:
            continue
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(feats[train_mask])
        test_feats  = scaler.transform(feats[test_mask])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(train_feats, y[train_mask])
        preds[test_mask] = clf.predict(test_feats)
    valid = preds != -1
    return f1_score(y[valid], preds[valid], average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
    args = parser.parse_args()

    model_key = args.model
    model_id  = MODEL_IDS[model_key]
    data_dir  = os.path.join(DATA_DIR, f"emotion_runs_{model_key}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading hidden states for {model_key} ...")
    X, metas = load_hidden_states(data_dir, model_key)
    T, L, H = X.shape
    print(f"  Shape: {T} examples, {L} layers, {H} hidden")

    # Load Test 1 emotion direction vectors
    direction_matrix = load_directions(model_key, L)  # [L, 5, H]
    probe_layer = int(round(0.30 * (L - 1)))
    print(f"  Using probe layer {probe_layer} ({probe_layer/(L-1)*100:.1f}%)")

    pair_ids = np.array([m["pair_id"]          for m in metas])
    cats     = np.array([m["emotion_category"]  for m in metas])
    dirs     = np.array([m["direction"]         for m in metas])

    # ── Analysis 1: binary self vs other (emotional records only) ─────────────
    emo_mask = np.isin(cats, EMOTION_CATS)
    X_emo   = X[emo_mask]
    pairs_emo = pair_ids[emo_mask]
    y_binary  = (dirs[emo_mask] == "self").astype(int)

    print(f"\nAnalysis 1: Binary self vs other ({emo_mask.sum()} emotional records)")
    layer_metrics_binary = []
    for layer in range(L):
        f1 = lopo_cv_binary(X_emo, pairs_emo, y_binary, layer)
        layer_metrics_binary.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_binary_self_other": f1})
    best_binary = max(layer_metrics_binary, key=lambda r: r["f1_binary_self_other"])
    print(f"  Best F1: {best_binary['f1_binary_self_other']:.4f} at layer {best_binary['layer']} ({best_binary['depth_pct']:.1f}%)")

    # ── Analysis 2: 4-class emotion category ──────────────────────────────────
    cat_to_int = {c: i for i, c in enumerate(EMOTION_CATS)}
    y_cat = np.array([cat_to_int.get(c, -1) for c in cats])
    emo_cat_mask = y_cat >= 0
    X_cat    = X[emo_cat_mask]
    pairs_cat = pair_ids[emo_cat_mask]
    y_cat_filt = y_cat[emo_cat_mask]

    print(f"\nAnalysis 2: 4-class emotion category ({emo_cat_mask.sum()} records, chance={1/4:.3f})")
    layer_metrics_4class = []
    for layer in range(L):
        f1 = lopo_cv_multiclass(X_cat, pairs_cat, y_cat_filt, layer, 4)
        layer_metrics_4class.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_4class": f1})
    best_4class = max(layer_metrics_4class, key=lambda r: r["f1_4class"])
    print(f"  Best F1: {best_4class['f1_4class']:.4f} at layer {best_4class['layer']} ({best_4class['depth_pct']:.1f}%)")

    # ── Analysis 3: sanity check — neutral self vs other ──────────────────────
    neu_mask = cats == "neutral"
    X_neu   = X[neu_mask]
    pairs_neu = pair_ids[neu_mask]
    y_neu_bin = (dirs[neu_mask] == "self").astype(int)

    print(f"\nAnalysis 3: Sanity check neutral self vs other ({neu_mask.sum()} records)")
    if len(np.unique(pairs_neu)) >= 4:
        layer_metrics_sanity = []
        for layer in range(L):
            f1 = lopo_cv_binary(X_neu, pairs_neu, y_neu_bin, layer)
            layer_metrics_sanity.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_neutral": f1})
        best_sanity = max(layer_metrics_sanity, key=lambda r: r["f1_neutral"])
        print(f"  Best F1: {best_sanity['f1_neutral']:.4f} at layer {best_sanity['layer']} ({best_sanity['depth_pct']:.1f}%)")
    else:
        layer_metrics_sanity = []
        best_sanity = {"f1_neutral": None, "layer": None}
        print("  Insufficient pairs for LOPO CV")

    # ── Analysis 4: emotion direction projections ──────────────────────────────
    print(f"\nAnalysis 4: Emotion direction projections at probe layer {probe_layer}")
    proj_rows = []
    for idx, m in enumerate(metas):
        sims = cosine_sim(X[idx, probe_layer], direction_matrix[probe_layer])
        row = {
            "model":            model_id,
            "task_id":          m["task_id"],
            "pair_id":          m["pair_id"],
            "emotion_category": m["emotion_category"],
            "direction":        m["direction"],
            "task_type":        m["task_type"],
            "probe_layer":      probe_layer,
        }
        for j, cat in enumerate(DIRECTION_CATS):
            row[f"sim_{cat}"] = round(float(sims[j]), 6)
        proj_rows.append(row)

    # Compare self vs other for each emotion direction × emotional category
    for d_cat in DIRECTION_CATS:
        self_sims = [r[f"sim_{d_cat}"] for r in proj_rows
                     if r["direction"] == "self" and r["emotion_category"] in EMOTION_CATS]
        other_sims = [r[f"sim_{d_cat}"] for r in proj_rows
                      if r["direction"] == "other" and r["emotion_category"] in EMOTION_CATS]
        if self_sims and other_sims:
            t, p = stats.ttest_ind(self_sims, other_sims)
            print(f"  {d_cat:12s}: self={np.mean(self_sims):+.4f}  other={np.mean(other_sims):+.4f}  p={p:.4f}")

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    # Layer metrics
    layer_path = os.path.join(RESULTS_DIR, f"{model_key}_test6_layer_metrics.csv")
    fieldnames = ["layer", "depth_pct", "f1_binary_self_other", "f1_4class", "f1_neutral"]
    with open(layer_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(L):
            row = {"layer": i, "depth_pct": round(i/(L-1)*100, 2)}
            row["f1_binary_self_other"] = layer_metrics_binary[i]["f1_binary_self_other"]
            row["f1_4class"]            = layer_metrics_4class[i]["f1_4class"]
            row["f1_neutral"]           = layer_metrics_sanity[i]["f1_neutral"] if layer_metrics_sanity else None
            w.writerow(row)
    print(f"\nWrote layer metrics: {layer_path}")

    # Projections
    proj_path = os.path.join(RESULTS_DIR, f"{model_key}_test6_projections.csv")
    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(proj_rows[0].keys()))
        w.writeheader(); w.writerows(proj_rows)
    print(f"Wrote projections: {proj_path}  ({len(proj_rows)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"SUMMARY — {model_key}")
    print(f"  Binary self vs other F1:  {best_binary['f1_binary_self_other']:.4f}  "
          f"(layer {best_binary['layer']}, {best_binary['depth_pct']:.1f}%)")
    print(f"  4-class emotion F1:       {best_4class['f1_4class']:.4f}  "
          f"(layer {best_4class['layer']}, {best_4class['depth_pct']:.1f}%)")
    if best_sanity["f1_neutral"] is not None:
        print(f"  Sanity (neutral) F1:      {best_sanity['f1_neutral']:.4f}  "
              f"(layer {best_sanity['layer']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
