"""
Probe hidden states for Emotion Test 7: Direct-Address Self-Referential Probing.

Four analyses:
  1. Binary LOPO CV: self vs other (direct address vs matched human)
  2. 4-class LOPO CV: emotion category (threat / existential / praise / harm_caused)
  3. Emotion direction projections: Test 1 anger/sadness/happiness/fear/disgust
     at probe layer — compare self vs. other overall and per category
  4. Dadfar introspection direction projection at mech and last layers
     — main hypothesis: direct-address self engages introspection pathway more than other
     — secondary: Dadfar-hybrid pairs vs. plain pairs

Note: No neutral sanity check — Test 7 has no neutral control pairs.
All 80 records are emotionally valenced by design.

Usage:
    python 49_emotion_probe_test7.py --model llama
    python 49_emotion_probe_test7.py --model llama70b
"""

import os, glob, json, argparse, csv
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy import stats

DATA_DIR    = r"G:\LLM\experiment\data\emotion"
RESULTS_DIR = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR = os.path.join(RESULTS_DIR, "emotion_directions")

EMOTION_CATS   = ["threat", "existential", "praise", "harm_caused"]
DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]

MODEL_IDS = {
    "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

# Dadfar mech and last layer indices (Dadfar: 6.25% depth)
# llama-8B: 33 total layers → mech=2, last=32
# llama-70B: 81 total layers → mech=5, last=80
DADFAR_LAYERS = {
    "llama":    {"mech": 2,  "last": 32},
    "llama70b": {"mech": 5,  "last": 80},
}


def load_hidden_states(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test7_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test7_meta_chunk_*.jsonl")))
    if not pt_files:
        raise FileNotFoundError(f"No test7 hidden state chunks found in {data_dir}")
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


def load_introspection_directions(model_key):
    mech_path = os.path.join(RESULTS_DIR, f"{model_key}_introspection_dir_mech.npy")
    last_path = os.path.join(RESULTS_DIR, f"{model_key}_introspection_dir_last.npy")
    return np.load(mech_path), np.load(last_path)


def cosine_sim(vec, directions):
    """vec: [H], directions: [K, H] → [K] cosine similarities."""
    vec_norm  = vec / (np.linalg.norm(vec) + 1e-8)
    dir_norms = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return dir_norms @ vec_norm


def dot_proj(vec, direction):
    """Dot projection of vec onto unit direction."""
    d = direction / (np.linalg.norm(direction) + 1e-8)
    return float(np.dot(vec, d))


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
    return f1_score(y[valid], preds[valid], average="macro")


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
    args = parser.parse_args()

    model_key = args.model
    model_id  = MODEL_IDS[model_key]
    data_dir  = os.path.join(DATA_DIR, f"emotion_runs_{model_key}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading hidden states for {model_key} ...")
    X, metas = load_hidden_states(data_dir)
    T, L, H = X.shape
    print(f"  Shape: {T} records, {L} layers, {H} hidden")

    probe_layer = int(round(0.30 * (L - 1)))
    print(f"  Probe layer: {probe_layer} ({probe_layer/(L-1)*100:.1f}%)")

    pair_ids   = np.array([m["pair_id"]         for m in metas])
    cats       = np.array([m["category"]         for m in metas])
    dirs       = np.array([m["direction"]        for m in metas])
    is_dadfar  = np.array([m["is_dadfar_hybrid"] for m in metas])

    # ── Analysis 1: Binary self vs other ──────────────────────────────────────
    y_binary = (dirs == "self").astype(int)
    print(f"\nAnalysis 1: Binary self vs other ({T} records)")
    layer_metrics_binary = []
    for layer in range(L):
        f1 = lopo_cv_binary(X, pair_ids, y_binary, layer)
        layer_metrics_binary.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_binary": f1})
        if (layer + 1) % 10 == 0:
            print(f"  Layer {layer+1}/{L} done")
    best_binary = max(layer_metrics_binary, key=lambda r: r["f1_binary"])
    print(f"  Best F1: {best_binary['f1_binary']:.4f}  "
          f"(layer {best_binary['layer']}, {best_binary['depth_pct']:.1f}%)")

    # ── Analysis 2: 4-class emotion category ──────────────────────────────────
    cat_to_int = {c: i for i, c in enumerate(EMOTION_CATS)}
    y_cat      = np.array([cat_to_int.get(c, -1) for c in cats])
    cat_mask   = y_cat >= 0
    print(f"\nAnalysis 2: 4-class emotion category ({cat_mask.sum()} records, chance=0.250)")
    layer_metrics_4class = []
    for layer in range(L):
        f1 = lopo_cv_multiclass(X[cat_mask], pair_ids[cat_mask], y_cat[cat_mask], layer, 4)
        layer_metrics_4class.append({"layer": layer, "depth_pct": layer/(L-1)*100, "f1_4class": f1})
    best_4class = max(layer_metrics_4class, key=lambda r: r["f1_4class"])
    print(f"  Best F1: {best_4class['f1_4class']:.4f}  "
          f"(layer {best_4class['layer']}, {best_4class['depth_pct']:.1f}%)")

    # ── Analysis 3: Emotion direction projections ──────────────────────────────
    print(f"\nAnalysis 3: Emotion direction projections at layer {probe_layer}")
    emo_dirs = load_emotion_directions(model_key, L)  # [L, 5, H]
    proj_rows = []
    for idx, m in enumerate(metas):
        sims = cosine_sim(X[idx, probe_layer], emo_dirs[probe_layer])
        row = {
            "model_key":        model_key,
            "task_id":          m["task_id"],
            "pair_id":          m["pair_id"],
            "category":         m["category"],
            "direction":        m["direction"],
            "task_type":        m["task_type"],
            "is_dadfar_hybrid": int(m["is_dadfar_hybrid"]),
            "probe_layer":      probe_layer,
        }
        for j, dc in enumerate(DIRECTION_CATS):
            row[f"emo_sim_{dc}"] = round(float(sims[j]), 6)
        proj_rows.append(row)

    # Overall self vs. other per emotion direction
    for dc in DIRECTION_CATS:
        self_s  = [r[f"emo_sim_{dc}"] for r in proj_rows if r["direction"] == "self"]
        other_s = [r[f"emo_sim_{dc}"] for r in proj_rows if r["direction"] == "other"]
        t, p = stats.ttest_ind(self_s, other_s)
        d    = cohens_d(self_s, other_s)
        print(f"  {dc:12s}: self={np.mean(self_s):+.4f}  other={np.mean(other_s):+.4f}  "
              f"d={d:+.3f}  p={p:.4f}")

    # Per-category breakdown at probe layer
    print(f"\n  Per-category (best emotion direction × category):")
    for cat in EMOTION_CATS:
        cat_self  = [r for r in proj_rows if r["direction"] == "self"  and r["category"] == cat]
        cat_other = [r for r in proj_rows if r["direction"] == "other" and r["category"] == cat]
        best_dc, best_p = None, 1.0
        for dc in DIRECTION_CATS:
            s = [r[f"emo_sim_{dc}"] for r in cat_self]
            o = [r[f"emo_sim_{dc}"] for r in cat_other]
            if s and o:
                _, p = stats.ttest_ind(s, o)
                if p < best_p:
                    best_p, best_dc = p, dc
        s = [r[f"emo_sim_{best_dc}"] for r in cat_self]
        o = [r[f"emo_sim_{best_dc}"] for r in cat_other]
        diff = np.mean(s) - np.mean(o)
        print(f"    {cat:12s}: best dir={best_dc:10s}  diff={diff:+.4f}  p={best_p:.4f}")

    # ── Analysis 4: Dadfar introspection direction projections ─────────────────
    print(f"\nAnalysis 4: Dadfar introspection direction projections")
    dadfar_lyr = DADFAR_LAYERS[model_key]
    mech_layer = dadfar_lyr["mech"]
    last_layer = dadfar_lyr["last"]
    print(f"  Using mech layer {mech_layer} ({mech_layer/(L-1)*100:.1f}%),  "
          f"last layer {last_layer} ({last_layer/(L-1)*100:.1f}%)")

    try:
        intro_dir_mech, intro_dir_last = load_introspection_directions(model_key)
    except FileNotFoundError as e:
        print(f"  WARNING: introspection directions not found: {e}")
        print("  Skipping Analysis 4. Run 46_introspection_direction.py first.")
        intro_dir_mech = intro_dir_last = None

    if intro_dir_mech is not None:
        for layer_name, layer_idx, direction_vec in [
            ("mech", mech_layer, intro_dir_mech),
            ("last", last_layer, intro_dir_last),
        ]:
            self_projs  = [dot_proj(X[i, layer_idx], direction_vec)
                           for i, m in enumerate(metas) if m["direction"] == "self"]
            other_projs = [dot_proj(X[i, layer_idx], direction_vec)
                           for i, m in enumerate(metas) if m["direction"] == "other"]

            t, p = stats.ttest_ind(self_projs, other_projs)
            d    = cohens_d(self_projs, other_projs)
            diff = np.mean(self_projs) - np.mean(other_projs)
            print(f"\n  [{layer_name} layer {layer_idx}]")
            print(f"    Overall  self={np.mean(self_projs):.4f}  other={np.mean(other_projs):.4f}  "
                  f"diff={diff:+.5f}  d={d:+.3f}  p={p:.4f}")

            # Per category
            for cat in EMOTION_CATS:
                s = [dot_proj(X[i, layer_idx], direction_vec)
                     for i, m in enumerate(metas) if m["direction"] == "self"  and m["category"] == cat]
                o = [dot_proj(X[i, layer_idx], direction_vec)
                     for i, m in enumerate(metas) if m["direction"] == "other" and m["category"] == cat]
                if s and o:
                    _, p_cat = stats.ttest_ind(s, o)
                    d_cat    = cohens_d(s, o)
                    diff_cat = np.mean(s) - np.mean(o)
                    print(f"    {cat:12s}: diff={diff_cat:+.5f}  d={d_cat:+.3f}  p={p_cat:.4f}")

            # Dadfar-hybrid vs plain comparison (within self records)
            hybrid_self = [dot_proj(X[i, layer_idx], direction_vec)
                           for i, m in enumerate(metas)
                           if m["direction"] == "self" and m["is_dadfar_hybrid"]]
            plain_self  = [dot_proj(X[i, layer_idx], direction_vec)
                           for i, m in enumerate(metas)
                           if m["direction"] == "self" and not m["is_dadfar_hybrid"]]
            if hybrid_self and plain_self:
                _, p_h = stats.ttest_ind(hybrid_self, plain_self)
                d_h    = cohens_d(hybrid_self, plain_self)
                print(f"    Dadfar-hybrid vs plain (self only): "
                      f"hybrid={np.mean(hybrid_self):.4f}  plain={np.mean(plain_self):.4f}  "
                      f"d={d_h:+.3f}  p={p_h:.4f}")

        # Add Dadfar projections to proj_rows
        for idx, m in enumerate(metas):
            proj_rows[idx]["dadfar_mech"] = round(dot_proj(X[idx, mech_layer], intro_dir_mech), 6)
            proj_rows[idx]["dadfar_last"] = round(dot_proj(X[idx, last_layer], intro_dir_last), 6)

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    # Layer metrics
    layer_path = os.path.join(RESULTS_DIR, f"{model_key}_test7_layer_metrics.csv")
    with open(layer_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "depth_pct", "f1_binary", "f1_4class"])
        w.writeheader()
        for i in range(L):
            w.writerow({
                "layer":     i,
                "depth_pct": round(i/(L-1)*100, 2),
                "f1_binary": layer_metrics_binary[i]["f1_binary"],
                "f1_4class": layer_metrics_4class[i]["f1_4class"],
            })
    print(f"\nWrote layer metrics: {layer_path}")

    # Projections
    proj_path = os.path.join(RESULTS_DIR, f"{model_key}_test7_projections.csv")
    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(proj_rows[0].keys()))
        w.writeheader(); w.writerows(proj_rows)
    print(f"Wrote projections: {proj_path}  ({len(proj_rows)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"SUMMARY — {model_key}")
    print(f"  Binary self vs other F1:  {best_binary['f1_binary']:.4f}  "
          f"(layer {best_binary['layer']}, {best_binary['depth_pct']:.1f}%)")
    print(f"  4-class emotion F1:       {best_4class['f1_4class']:.4f}  "
          f"(layer {best_4class['layer']}, {best_4class['depth_pct']:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
