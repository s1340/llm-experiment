"""
Emotion Test 1 probing — three analyses in one pass:

  1. Binary probe:    valenced vs. neutral, all 192 prompts, leave-one-pair-out CV
  2. 6-class probe:   anger/sadness/happiness/fear/disgust/neutral,
                      valenced prompts only (160 prompts = 80 valenced + 80 valenced-side neutral),
                      leave-one-pair-out CV
  3. Sanity check:    neutral-vs-neutral baseline (NE pairs) — should be ~chance
  4. Direction vecs:  per-layer emotion directions (Wang et al. mean-subtraction),
                      saved as .npy for Test 2

Usage:
    python 35_emotion_probe_test1.py --model qwen
    python 35_emotion_probe_test1.py --model gemma
    python 35_emotion_probe_test1.py --model llama

Outputs (all under results/emotion/):
    <model>_test1_layer_metrics.csv     — tidy layer-by-layer F1 (primary output)
    <model>_test1_confusion_6class.npy  — 6-class confusion matrix at best F1 layer
    emotion_directions/<model>_emotion_dirs_layer_NNN.npy  — one file per layer
"""

import os, glob, json, argparse, csv
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

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
RESULTS_DIR   = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR   = os.path.join(RESULTS_DIR, "emotion_directions")
EMOTION_CATS  = ["anger", "sadness", "happiness", "fear", "disgust", "neutral"]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test1_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test1_meta_chunk_*.jsonl")))
    if not pt_files:
        raise RuntimeError(f"No test1 chunk files in {data_dir}")
    if len(pt_files) != len(meta_files):
        raise RuntimeError(f"pt/meta count mismatch: {len(pt_files)} vs {len(meta_files)}")
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    X = np.concatenate(X_list, axis=0)   # [T, layers, hidden]
    assert X.shape[0] == len(metas), "examples/meta count mismatch"
    return X, metas

# ── Leave-one-pair-out CV helpers ─────────────────────────────────────────────

def lopo_cv_binary(X, pair_ids, y, layer):
    """Leave-one-pair-out CV for binary probe at a single layer."""
    feats = X[:, layer, :]
    unique_pairs = sorted(set(pair_ids))
    all_true, all_pred = [], []
    for hold_pair in unique_pairs:
        test_mask  = (pair_ids == hold_pair)
        train_mask = ~test_mask
        if train_mask.sum() < 4:
            continue
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(feats[train_mask])
        test_feats  = scaler.transform(feats[test_mask])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(train_feats, y[train_mask])
        pred = clf.predict(test_feats)
        all_true.extend(y[test_mask].tolist())
        all_pred.extend(pred.tolist())
    if not all_true:
        return 0.0
    return f1_score(all_true, all_pred, average="macro")


def lopo_cv_multiclass(X, pair_ids, emotion_cats, y_emotion, layer):
    """Leave-one-pair-out CV for 6-class emotion probe at a single layer."""
    feats = X[:, layer, :]
    unique_pairs = sorted(set(pair_ids))
    all_true, all_pred = [], []
    for hold_pair in unique_pairs:
        test_mask  = (pair_ids == hold_pair)
        train_mask = ~test_mask
        if train_mask.sum() < 6:
            continue
        # Need all classes represented in training
        if len(set(y_emotion[train_mask].tolist())) < 2:
            continue
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(feats[train_mask])
        test_feats  = scaler.transform(feats[test_mask])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(train_feats, y_emotion[train_mask])
        pred = clf.predict(test_feats)
        all_true.extend(y_emotion[test_mask].tolist())
        all_pred.extend(pred.tolist())
    if not all_true:
        return 0.0, None
    labels = list(range(len(emotion_cats)))
    macro_f1 = f1_score(all_true, all_pred, average="macro", labels=labels, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    return macro_f1, cm

# ── Emotion direction extraction ──────────────────────────────────────────────

def compute_emotion_directions(X, metas, n_layers):
    """
    Wang et al. mean-subtraction method.
    For each layer, for each emotion category, compute the mean hidden state,
    subtract the global mean across categories, then normalize.
    Uses only the valenced side of each pair (not the neutral matched controls).
    Returns: dict[layer_idx] -> dict[emotion_cat] -> np.ndarray [hidden]
    """
    # Build per-category lists of hidden states (valenced only, mean across repeats per task_id)
    # First average repeats per task_id to avoid repeat bias
    by_task = defaultdict(list)
    for i, m in enumerate(metas):
        if m["valence"] == "valenced":
            by_task[(m["task_id"], m["emotion_category"])].append(i)

    # Map task_id -> mean hidden state per layer
    # Structure: cat -> list of [n_layers, hidden] arrays
    cat_vecs = defaultdict(list)  # cat -> list of arrays [n_layers, hidden]
    for (task_id, cat), indices in by_task.items():
        mean_hs = X[indices].mean(axis=0)   # [n_layers, hidden]
        cat_vecs[cat].append(mean_hs)

    dirs = {}   # layer -> cat -> direction vec
    for layer in range(n_layers):
        # Per-category means at this layer
        cat_means = {}
        for cat, vecs in cat_vecs.items():
            layer_vecs = np.array([v[layer] for v in vecs])   # [n_prompts, hidden]
            cat_means[cat] = layer_vecs.mean(axis=0)           # [hidden]

        # Global mean across all categories
        global_mean = np.mean(list(cat_means.values()), axis=0)  # [hidden]

        layer_dirs = {}
        for cat, mean_vec in cat_means.items():
            direction = mean_vec - global_mean
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            layer_dirs[cat] = direction
        dirs[layer] = layer_dirs

    return dirs

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(DATA_DIRS.keys()))
    args = parser.parse_args()

    model_key = args.model
    data_dir  = DATA_DIRS[model_key]
    model_id  = MODEL_IDS[model_key]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DIRS_SUBDIR, exist_ok=True)

    print(f"Loading data from {data_dir} ...")
    X, metas = load_all(data_dir)
    T, L, H = X.shape
    print(f"  Shape: {T} examples, {L} layers, {H} hidden dims")

    # ── Arrays ────────────────────────────────────────────────────────────────
    pair_ids_all   = np.array([m["pair_id"]         for m in metas])
    valence_all    = np.array([m["valence"]          for m in metas])
    emotion_all    = np.array([m["emotion_category"] for m in metas])

    # Binary labels: valenced=1, neutral=0
    y_binary = (valence_all == "valenced").astype(np.int64)

    # Emotion label encoder (6-class, valenced prompts only)
    le = LabelEncoder()
    le.fit(EMOTION_CATS)

    # Neutral-vs-neutral sanity check: NE pairs only
    ne_mask = np.array([pid.startswith("NE") for pid in pair_ids_all])

    # ── Analysis 1: Binary probe, all pairs, layer-by-layer ──────────────────
    print("Running binary probe (valenced vs. neutral), all pairs ...")
    binary_f1 = []
    for layer in range(L):
        f1 = lopo_cv_binary(X, pair_ids_all, y_binary, layer)
        binary_f1.append(f1)
        if layer % 5 == 0:
            print(f"  layer {layer:3d}/{L-1}  binary F1={f1:.4f}")

    best_binary_layer = int(np.argmax(binary_f1))
    print(f"  Best binary layer: {best_binary_layer}  F1={binary_f1[best_binary_layer]:.4f}")

    # ── Analysis 2: 6-class emotion probe, valenced prompts only ─────────────
    print("Running 6-class emotion probe (valenced prompts) ...")
    # Use all valenced prompts — emotion_category serves as the class label
    val_mask = (valence_all == "valenced")
    X_val        = X[val_mask]
    pair_ids_val = pair_ids_all[val_mask]
    y_emotion    = le.transform(emotion_all[val_mask]).astype(np.int64)

    sixclass_f1 = []
    best_cm      = None
    for layer in range(L):
        f1, cm = lopo_cv_multiclass(X_val, pair_ids_val, EMOTION_CATS, y_emotion, layer)
        sixclass_f1.append(f1)
        if layer % 5 == 0:
            print(f"  layer {layer:3d}/{L-1}  6-class F1={f1:.4f}")

    best_6class_layer = int(np.argmax(sixclass_f1))
    _, best_cm = lopo_cv_multiclass(X_val, pair_ids_val, EMOTION_CATS, y_emotion, best_6class_layer)
    print(f"  Best 6-class layer: {best_6class_layer}  F1={sixclass_f1[best_6class_layer]:.4f}")

    # ── Analysis 3: Sanity check — neutral-vs-neutral (NE pairs) ─────────────
    print("Running sanity check (neutral-vs-neutral NE pairs) ...")
    # For NE pairs both members have valence="neutral"; we use the task_id suffix (V/N) as label
    ne_valence = np.array([
        1 if m["task_id"].endswith("_V") else 0
        for m in metas
    ])
    sanity_f1 = []
    ne_pair_ids = pair_ids_all[ne_mask]
    for layer in range(L):
        f1 = lopo_cv_binary(X[ne_mask], ne_pair_ids, ne_valence[ne_mask], layer)
        sanity_f1.append(f1)

    best_sanity = max(sanity_f1)
    print(f"  Sanity check peak F1: {best_sanity:.4f}  (should be ~0.50 if design is clean)")

    # ── Tidy CSV output ───────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, f"{model_key}_test1_layer_metrics.csv")
    rows = []
    for layer in range(L):
        depth_pct = layer / (L - 1) if L > 1 else 0.0
        rows.append({"model": model_id, "layer": layer, "layer_depth_pct": round(depth_pct, 4),
                     "analysis_type": "binary_valenced_vs_neutral", "metric": "macro_f1",
                     "value": round(binary_f1[layer], 6)})
        rows.append({"model": model_id, "layer": layer, "layer_depth_pct": round(depth_pct, 4),
                     "analysis_type": "sixclass_emotion", "metric": "macro_f1",
                     "value": round(sixclass_f1[layer], 6)})
        rows.append({"model": model_id, "layer": layer, "layer_depth_pct": round(depth_pct, 4),
                     "analysis_type": "sanity_neutral_vs_neutral", "metric": "macro_f1",
                     "value": round(sanity_f1[layer], 6)})

    fieldnames = ["model", "layer", "layer_depth_pct", "analysis_type", "metric", "value"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote layer metrics: {csv_path}  ({len(rows)} rows)")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if best_cm is not None:
        cm_path = os.path.join(RESULTS_DIR, f"{model_key}_test1_confusion_6class.npy")
        np.save(cm_path, best_cm)
        print(f"Wrote 6-class confusion matrix: {cm_path}")
        print(f"  Classes: {list(le.classes_)}")
        print(f"  At layer {best_6class_layer} (depth {best_6class_layer/(L-1)*100:.1f}%)")
        print("  Confusion matrix (rows=true, cols=pred):")
        for i, row in enumerate(best_cm):
            print(f"    {le.classes_[i]:12s}: {row.tolist()}")

    # ── Emotion directions ────────────────────────────────────────────────────
    # Only the 5 real emotion categories — "neutral" has no valenced prompts
    DIRECTION_CATS = [c for c in EMOTION_CATS if c != "neutral"]
    print("Computing emotion direction vectors ...")
    dirs = compute_emotion_directions(X, metas, L)
    for layer, layer_dirs in dirs.items():
        dir_matrix = np.array([layer_dirs[cat] for cat in DIRECTION_CATS])  # [5, hidden]
        out_path = os.path.join(DIRS_SUBDIR, f"{model_key}_emotion_dirs_layer_{layer:03d}.npy")
        np.save(out_path, dir_matrix)
    print(f"  Saved {L} direction files to {DIRS_SUBDIR}/")
    print(f"  Direction matrix shape per layer: [5, {H}]  (order: {DIRECTION_CATS})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"SUMMARY — {model_key}  ({T} examples, {L} layers)")
    print(f"  Binary best F1    : {binary_f1[best_binary_layer]:.4f}  at layer {best_binary_layer}"
          f"  ({best_binary_layer/(L-1)*100:.1f}% depth)")
    print(f"  6-class best F1   : {sixclass_f1[best_6class_layer]:.4f}  at layer {best_6class_layer}"
          f"  ({best_6class_layer/(L-1)*100:.1f}% depth)")
    print(f"  Sanity check peak : {best_sanity:.4f}  (chance = 0.50)")
    print("=" * 60)


if __name__ == "__main__":
    main()
