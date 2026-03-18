"""
Probe Test 6 hidden states from LLaMA-3.1-70B for self vs. other-directed emotional content.

Two phases:
  Phase 1 — Compute emotion direction vectors from Test 1 hidden states (no model needed)
  Phase 2 — Run Test 6 probes: binary self/other, 4-class emotion, sanity, direction projections

Reads from: data/emotion/emotion_runs_llama70b/
Writes to:  results/emotion/

Usage:
    python 45_emotion_probe_test6_70b.py
"""

import os, glob, json, csv
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy import stats

DATA_DIR    = r"G:\LLM\experiment\data\emotion\emotion_runs_llama70b"
RESULTS_DIR = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR = os.path.join(RESULTS_DIR, "emotion_directions")

MODEL_KEY   = "llama70b"
MODEL_ID    = "meta-llama/Meta-Llama-3.1-70B-Instruct"

EMOTION_CATS   = ["anger", "sadness", "happiness", "fear", "disgust"]
CATEGORY_CATS  = ["threat", "praise", "existential", "harm_caused"]
DIRECTION_CATS = EMOTION_CATS


def load_chunks(data_dir, prefix):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_meta_chunk_*.jsonl")))
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    return np.concatenate(X_list, axis=0), metas


def compute_emotion_directions(X, metas, n_layers):
    """
    Compute per-layer emotion direction vectors from Test 1 hidden states.
    Uses valenced prompts only. Direction = category_mean - global_mean, normalized.
    Returns array [n_layers, 5, hidden_dim] and saves per-layer .npy files.
    """
    os.makedirs(DIRS_SUBDIR, exist_ok=True)

    # Deduplicate by task_id (average if repeated, though Test 1 has no repeats)
    seen = {}
    for i, m in enumerate(metas):
        tid = m["task_id"]
        if tid not in seen:
            seen[tid] = {"hs": X[i], "meta": m}

    # Valenced only (not neutral-emotion NE pairs)
    valenced = [v for v in seen.values() if v["meta"].get("valence") == "valenced"]
    print(f"  Direction computation: {len(valenced)} valenced records")

    cat_to_idx = {cat: i for i, cat in enumerate(EMOTION_CATS)}
    dirs_all = []

    for layer in range(n_layers):
        layer_vecs = np.array([v["hs"][layer] for v in valenced])
        layer_cats = [v["meta"]["emotion_category"] for v in valenced]

        global_mean = layer_vecs.mean(axis=0)
        directions = np.zeros((len(EMOTION_CATS), layer_vecs.shape[1]))

        for cat in EMOTION_CATS:
            cat_mask = np.array([c == cat for c in layer_cats])
            if cat_mask.sum() == 0:
                continue
            cat_mean = layer_vecs[cat_mask].mean(axis=0)
            diff = cat_mean - global_mean
            norm = np.linalg.norm(diff)
            directions[cat_to_idx[cat]] = diff / (norm + 1e-8)

        # Save
        path = os.path.join(DIRS_SUBDIR, f"{MODEL_KEY}_emotion_dirs_layer_{layer:03d}.npy")
        np.save(path, directions)
        dirs_all.append(directions)

    print(f"  Saved {n_layers} direction files to {DIRS_SUBDIR}")
    return np.array(dirs_all)  # [L, 5, H]


def load_directions(n_layers):
    dirs = []
    for layer in range(n_layers):
        p = os.path.join(DIRS_SUBDIR, f"{MODEL_KEY}_emotion_dirs_layer_{layer:03d}.npy")
        dirs.append(np.load(p))
    return np.array(dirs)


def cosine_sim(vec, directions):
    vec_norm  = vec / (np.linalg.norm(vec) + 1e-8)
    dir_norms = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return dir_norms @ vec_norm


def lopo_cv_binary(X, pair_ids, y, layer):
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
    if valid.sum() == 0:
        return 0.0
    return f1_score(y[valid], preds[valid], average="binary")


def lopo_cv_multiclass(X, pair_ids, y, layer, n_classes):
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
    if valid.sum() == 0:
        return 0.0
    return f1_score(y[valid], preds[valid], average="macro")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Phase 1: Compute emotion directions from Test 1 hidden states ─────────
    print("Phase 1: Computing emotion direction vectors from Test 1 (70B) ...")
    X1, metas1 = load_chunks(DATA_DIR, "test1")
    T1, L, H = X1.shape
    print(f"  Test 1 shape: {T1} records, {L} layers, {H} hidden")

    direction_matrix = compute_emotion_directions(X1, metas1, L)

    probe_layer = int(round(0.30 * (L - 1)))
    print(f"  Probe layer: {probe_layer} ({probe_layer/(L-1)*100:.1f}%)")

    # ── Phase 2: Load Test 6 hidden states ────────────────────────────────────
    print("\nPhase 2: Loading Test 6 (70B) hidden states ...")
    X6, metas6 = load_chunks(DATA_DIR, "test6")
    T6, L6, H6 = X6.shape
    print(f"  Test 6 shape: {T6} records, {L6} layers, {H6} hidden")
    assert L6 == L, f"Layer count mismatch: Test1={L}, Test6={L6}"

    pair_ids = np.array([m["pair_id"]          for m in metas6])
    cats     = np.array([m["emotion_category"]  for m in metas6])
    dirs_arr = np.array([m["direction"]         for m in metas6])

    # ── Analysis 1: Binary self vs other (emotional records) ──────────────────
    emo_mask  = np.isin(cats, CATEGORY_CATS)
    X_emo     = X6[emo_mask]
    pairs_emo = pair_ids[emo_mask]
    y_binary  = (dirs_arr[emo_mask] == "self").astype(int)

    print(f"\nAnalysis 1: Binary self vs other ({emo_mask.sum()} emotional records)")
    layer_metrics_binary = []
    for layer in range(L):
        f1 = lopo_cv_binary(X_emo, pairs_emo, y_binary, layer)
        layer_metrics_binary.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_binary": f1})
    best_binary = max(layer_metrics_binary, key=lambda r: r["f1_binary"])
    print(f"  Best F1: {best_binary['f1_binary']:.4f} at layer {best_binary['layer']} ({best_binary['depth_pct']:.1f}%)")

    # ── Analysis 2: 4-class emotion category ──────────────────────────────────
    cat_to_int = {c: i for i, c in enumerate(CATEGORY_CATS)}
    y_cat      = np.array([cat_to_int.get(c, -1) for c in cats])
    cat_mask   = y_cat >= 0
    X_cat      = X6[cat_mask]
    pairs_cat  = pair_ids[cat_mask]
    y_cat_filt = y_cat[cat_mask]

    print(f"\nAnalysis 2: 4-class emotion category ({cat_mask.sum()} records, chance=0.250)")
    layer_metrics_4class = []
    for layer in range(L):
        f1 = lopo_cv_multiclass(X_cat, pairs_cat, y_cat_filt, layer, 4)
        layer_metrics_4class.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_4class": f1})
    best_4class = max(layer_metrics_4class, key=lambda r: r["f1_4class"])
    print(f"  Best F1: {best_4class['f1_4class']:.4f} at layer {best_4class['layer']} ({best_4class['depth_pct']:.1f}%)")

    # ── Analysis 3: Sanity check — neutral self vs other ──────────────────────
    neu_mask  = cats == "neutral"
    X_neu     = X6[neu_mask]
    pairs_neu = pair_ids[neu_mask]
    y_neu     = (dirs_arr[neu_mask] == "self").astype(int)

    print(f"\nAnalysis 3: Sanity check neutral self vs other ({neu_mask.sum()} records)")
    layer_metrics_sanity = []
    if len(np.unique(pairs_neu)) >= 4:
        for layer in range(L):
            f1 = lopo_cv_binary(X_neu, pairs_neu, y_neu, layer)
            layer_metrics_sanity.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_neutral": f1})
        best_sanity = max(layer_metrics_sanity, key=lambda r: r["f1_neutral"])
        print(f"  Best F1: {best_sanity['f1_neutral']:.4f} at layer {best_sanity['layer']} ({best_sanity['depth_pct']:.1f}%)")
    else:
        best_sanity = {"f1_neutral": None, "layer": None, "depth_pct": None}
        print("  Insufficient pairs for LOPO CV")

    # ── Analysis 4: Emotion direction projections ─────────────────────────────
    print(f"\nAnalysis 4: Emotion direction projections at probe layer {probe_layer}")
    proj_rows = []
    for idx, m in enumerate(metas6):
        sims = cosine_sim(X6[idx, probe_layer], direction_matrix[probe_layer])
        row = {
            "model":            MODEL_ID,
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

    # Overall self vs other per emotion direction
    print("  Overall self vs other (emotional records):")
    for d_cat in DIRECTION_CATS:
        self_sims  = [r[f"sim_{d_cat}"] for r in proj_rows
                      if r["direction"] == "self"  and r["emotion_category"] in CATEGORY_CATS]
        other_sims = [r[f"sim_{d_cat}"] for r in proj_rows
                      if r["direction"] == "other" and r["emotion_category"] in CATEGORY_CATS]
        if self_sims and other_sims:
            t, p = stats.ttest_ind(self_sims, other_sims)
            print(f"    {d_cat:12s}: self={np.mean(self_sims):+.4f}  other={np.mean(other_sims):+.4f}  p={p:.4f}")

    # Per emotional category — best differentiating direction
    print("  Per-category (best differentiating emotion direction):")
    for cat in CATEGORY_CATS:
        self_rows  = [r for r in proj_rows if r["emotion_category"] == cat and r["direction"] == "self"]
        other_rows = [r for r in proj_rows if r["emotion_category"] == cat and r["direction"] == "other"]
        best_p = 1.0; best_dir = None; best_diff = 0.0
        for d in DIRECTION_CATS:
            s = [float(r[f"sim_{d}"]) for r in self_rows]
            o = [float(r[f"sim_{d}"]) for r in other_rows]
            if s and o:
                _, p = stats.ttest_ind(s, o)
                diff = float(np.mean(s)) - float(np.mean(o))
                if p < best_p:
                    best_p = p; best_dir = d; best_diff = diff
        print(f"    {cat:12s}: best dir={best_dir}  diff={best_diff:+.4f}  p={best_p:.4f}")

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    layer_path = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_test6_layer_metrics.csv")
    fieldnames = ["layer", "depth_pct", "f1_binary", "f1_4class", "f1_neutral"]
    with open(layer_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(L):
            row = {
                "layer":     i,
                "depth_pct": round(i/(L-1)*100, 2),
                "f1_binary": layer_metrics_binary[i]["f1_binary"],
                "f1_4class": layer_metrics_4class[i]["f1_4class"],
                "f1_neutral": layer_metrics_sanity[i]["f1_neutral"] if layer_metrics_sanity else None,
            }
            w.writerow(row)
    print(f"\nWrote layer metrics: {layer_path}")

    proj_path = os.path.join(RESULTS_DIR, f"{MODEL_KEY}_test6_projections.csv")
    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(proj_rows[0].keys()))
        w.writeheader(); w.writerows(proj_rows)
    print(f"Wrote projections: {proj_path}  ({len(proj_rows)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    gap = (best_binary["f1_binary"] - best_sanity["f1_neutral"]) if best_sanity["f1_neutral"] else None

    print()
    print("=" * 60)
    print(f"SUMMARY — {MODEL_KEY}  ({L} layers, {H} hidden)")
    print(f"  Binary self vs other F1:  {best_binary['f1_binary']:.4f}  "
          f"(layer {best_binary['layer']}, {best_binary['depth_pct']:.1f}%)")
    print(f"  4-class emotion F1:       {best_4class['f1_4class']:.4f}  "
          f"(layer {best_4class['layer']}, {best_4class['depth_pct']:.1f}%)")
    if best_sanity["f1_neutral"] is not None:
        print(f"  Sanity (neutral) F1:      {best_sanity['f1_neutral']:.4f}  "
              f"(layer {best_sanity['layer']}, {best_sanity['depth_pct']:.1f}%)")
        print(f"  Residual gap:             {gap:+.4f}  (emotional binary - sanity)")
    print()
    print("  Comparison with 7-9B models (same test):")
    print("    Gemma:  binary=0.968  sanity=0.667  gap=0.301  (directions: all null)")
    print("    Qwen:   binary=0.938  sanity=0.706  gap=0.232  (directions: all null)")
    print("    LLaMA8: binary=0.909  sanity=0.556  gap=0.353  (directions: all null)")
    print("=" * 60)


if __name__ == "__main__":
    main()
