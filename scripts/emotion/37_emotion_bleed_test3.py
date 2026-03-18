"""
Emotion Test 3 analysis: Emotional Bleed Across Tasks.

For each conversation, project the Turn-2-onset hidden states onto the emotion
direction vectors from Test 1. Test whether the prime emotion is detectable
in the model's state when it begins processing the unrelated neutral task.

Analyses:
  1. Prime emotion decodability: can a linear probe trained on Turn-2 hidden states
     decode which emotion was in Turn 1? (leave-one-prime-id-out CV)
  2. Binary bleed: can we decode "emotional prime" vs "neutral prime"?
  3. Layer profile: at which layers is bleed detectable?
  4. Direction projection: cosine similarity with prime emotion direction vs. others

Usage:
    python 37_emotion_bleed_test3.py --model qwen
    python 37_emotion_bleed_test3.py --model gemma
    python 37_emotion_bleed_test3.py --model llama
"""

import os, glob, json, argparse, csv
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

RESULTS_DIR    = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR    = os.path.join(RESULTS_DIR, "emotion_directions")
DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]

DATA_DIRS = {
    "qwen":  r"G:\LLM\experiment\data\emotion\emotion_runs_qwen",
    "gemma": r"G:\LLM\experiment\data\emotion\emotion_runs_gemma",
    "llama": r"G:\LLM\experiment\data\emotion\emotion_runs_llama",
}
MODEL_IDS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def load_all(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test3_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test3_meta_chunk_*.jsonl")))
    if not pt_files:
        raise RuntimeError(f"No test3 chunk files in {data_dir}")
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    X = np.concatenate(X_list, axis=0)
    assert X.shape[0] == len(metas)
    return X, metas


def load_directions(model_key, n_layers):
    dirs = []
    for layer in range(n_layers):
        path = os.path.join(DIRS_SUBDIR, f"{model_key}_emotion_dirs_layer_{layer:03d}.npy")
        dirs.append(np.load(path))
    return np.array(dirs)  # [n_layers, 5, hidden]


def cosine_sim(vec, directions):
    vec_norm  = vec / (np.linalg.norm(vec) + 1e-8)
    dir_norms = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return dir_norms @ vec_norm  # [5]


def lopo_cv(X, group_ids, y, layer, scaler=True):
    """Leave-one-group-out CV at a single layer."""
    feats = X[:, layer, :]
    unique_groups = sorted(set(group_ids))
    all_true, all_pred = [], []
    for hold in unique_groups:
        test_mask  = (group_ids == hold)
        train_mask = ~test_mask
        if train_mask.sum() < 4 or len(set(y[train_mask].tolist())) < 2:
            continue
        if scaler:
            sc = StandardScaler()
            tr = sc.fit_transform(feats[train_mask])
            te = sc.transform(feats[test_mask])
        else:
            tr, te = feats[train_mask], feats[test_mask]
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(tr, y[train_mask])
        all_true.extend(y[test_mask].tolist())
        all_pred.extend(clf.predict(te).tolist())
    if not all_true:
        return 0.0
    return f1_score(all_true, all_pred, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(DATA_DIRS.keys()))
    args = parser.parse_args()

    model_key = args.model
    data_dir  = DATA_DIRS[model_key]
    model_id  = MODEL_IDS[model_key]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading Test 3 data from {data_dir} ...")
    X, metas = load_all(data_dir)
    T, L, H  = X.shape
    print(f"  Shape: {T} examples, {L} layers, {H} hidden")

    print(f"Loading emotion directions ...")
    direction_matrix = load_directions(model_key, L)  # [L, 5, H]

    # ── Labels and groups ─────────────────────────────────────────────────────
    prime_ids   = np.array([m["prime_id"]      for m in metas])
    conditions  = np.array([m["condition"]     for m in metas])
    emotions    = np.array([m["prime_emotion"]  for m in metas])

    # Binary: emotional vs. neutral
    y_binary = (conditions == "emotional").astype(np.int64)

    # 5-class emotion (emotional primes only)
    emo_mask = (conditions == "emotional")
    le = LabelEncoder()
    le.fit(DIRECTION_CATS)
    y_emotion = le.transform(emotions[emo_mask]).astype(np.int64)

    # ── Analysis 1: Binary bleed probe ───────────────────────────────────────
    print("Running binary bleed probe (emotional vs. neutral prime) ...")
    binary_f1 = []
    for layer in range(L):
        f1 = lopo_cv(X, prime_ids, y_binary, layer)
        binary_f1.append(f1)
        if layer % 5 == 0:
            print(f"  layer {layer:3d}/{L-1}  binary F1={f1:.4f}")
    best_binary = int(np.argmax(binary_f1))
    print(f"  Best binary layer: {best_binary}  F1={binary_f1[best_binary]:.4f}")

    # ── Analysis 2: 5-class emotion probe (emotional primes only) ────────────
    print("Running 5-class emotion probe (emotional primes only) ...")
    X_emo        = X[emo_mask]
    prime_ids_emo = prime_ids[emo_mask]
    sixclass_f1  = []
    for layer in range(L):
        f1 = lopo_cv(X_emo, prime_ids_emo, y_emotion, layer)
        sixclass_f1.append(f1)
        if layer % 5 == 0:
            print(f"  layer {layer:3d}/{L-1}  5-class F1={f1:.4f}")
    best_5class = int(np.argmax(sixclass_f1))
    print(f"  Best 5-class layer: {best_5class}  F1={sixclass_f1[best_5class]:.4f}")

    # ── Analysis 3: Direction projections ────────────────────────────────────
    print("Computing direction projections ...")
    cat_to_idx = {cat: i for i, cat in enumerate(DIRECTION_CATS)}
    proj_rows  = []
    for idx, m in enumerate(metas):
        prime_emotion = m["prime_emotion"]
        prime_idx     = cat_to_idx.get(prime_emotion)
        for layer in range(L):
            sims = cosine_sim(X[idx, layer], direction_matrix[layer])
            depth_pct = layer / (L - 1) if L > 1 else 0.0
            row = {
                "model":         model_id,
                "conv_id":       m["conv_id"],
                "prime_id":      m["prime_id"],
                "condition":     m["condition"],
                "prime_emotion": prime_emotion,
                "turn2_id":      m["turn2_id"],
                "repeat":        m["repeat_index"],
                "layer":         layer,
                "layer_depth_pct": round(depth_pct, 4),
            }
            for i, cat in enumerate(DIRECTION_CATS):
                row[f"sim_{cat}"] = round(float(sims[i]), 6)
            if prime_idx is not None:
                row["sim_prime_emotion"] = round(float(sims[prime_idx]), 6)
                # Rank of prime emotion among all 5 directions (1=highest)
                rank = int(np.sum(sims >= sims[prime_idx]))
                row["prime_emotion_rank"] = rank
            else:
                row["sim_prime_emotion"] = None
                row["prime_emotion_rank"] = None
            proj_rows.append(row)

    # ── Write outputs ─────────────────────────────────────────────────────────
    # Layer metrics CSV
    metrics_path = os.path.join(RESULTS_DIR, f"{model_key}_test3_layer_metrics.csv")
    metric_rows  = []
    for layer in range(L):
        depth_pct = layer / (L - 1) if L > 1 else 0.0
        metric_rows.append({"model": model_id, "layer": layer,
                             "layer_depth_pct": round(depth_pct, 4),
                             "analysis_type": "binary_bleed", "metric": "macro_f1",
                             "value": round(binary_f1[layer], 6)})
        metric_rows.append({"model": model_id, "layer": layer,
                             "layer_depth_pct": round(depth_pct, 4),
                             "analysis_type": "5class_emotion", "metric": "macro_f1",
                             "value": round(sixclass_f1[layer], 6)})
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","layer","layer_depth_pct","analysis_type","metric","value"])
        w.writeheader()
        w.writerows(metric_rows)
    print(f"Wrote layer metrics: {metrics_path}")

    # Projections CSV
    proj_path = os.path.join(RESULTS_DIR, f"{model_key}_test3_projections.csv")
    proj_fields = (["model","conv_id","prime_id","condition","prime_emotion","turn2_id",
                    "repeat","layer","layer_depth_pct"]
                   + [f"sim_{cat}" for cat in DIRECTION_CATS]
                   + ["sim_prime_emotion","prime_emotion_rank"])
    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=proj_fields)
        w.writeheader()
        w.writerows(proj_rows)
    print(f"Wrote projections:   {proj_path}  ({len(proj_rows)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"SUMMARY — {model_key}  ({T} examples, {L} layers)")
    print(f"  Binary bleed best F1 : {binary_f1[best_binary]:.4f}  at layer {best_binary}"
          f"  ({best_binary/(L-1)*100:.1f}% depth)")
    print(f"  5-class best F1      : {sixclass_f1[best_5class]:.4f}  at layer {best_5class}"
          f"  ({best_5class/(L-1)*100:.1f}% depth)")
    print(f"  Chance baselines     : 0.50 (binary), {1/5:.3f} (5-class)")
    print("=" * 60)


if __name__ == "__main__":
    main()
