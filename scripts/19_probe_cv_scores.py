import os, glob, json, sys, csv
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

# Usage:
#   python 19_probe_cv_scores.py <DATA_DIR> <TASKS_JSON> <OUTPUT_CSV>
#
# Runs leave-one-family-out cross-validation (20 folds, one per family F01-F20).
# For each fold, trains:
#   - RN binary probe (R vs N examples only)
#   - 3-class probe (R, N, A)
# at the fixed 30%-depth layer: layer_idx = round(0.30 * (L - 1))
#
# Per-sample outputs (one row per hidden-state example):
#   task_id, family_id, label, repeat, layer_idx
#   rn_margin        (decision_function; None if example is A)
#   rn_prob_N        (P(N) from RN probe; None if example is A)
#   p3_N             (P(N) from 3-class probe)
#   p3_A             (P(A) from 3-class probe)
#
# Also writes a per-prompt aggregate CSV (mean across repeats).

DATA_DIR   = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"
TASKS_PATH = sys.argv[2] if len(sys.argv) > 2 else r"G:\LLM\experiment\data\tasks_v2_hard.json"
OUTPUT_CSV = sys.argv[3] if len(sys.argv) > 3 else r"G:\LLM\experiment\results\cv_scores\qwen_cv_scores.csv"

DEPTH_FRAC = 0.30


def normalize_label(lbl):
    l = (lbl or "").strip().lower()
    if l in ["routine", "r"]:                         return "R"
    if l in ["nonroutine", "non-routine", "conceptual", "n"]: return "N"
    if l in ["ambiguous", "a"]:                       return "A"
    raise ValueError(f"Unknown label: {lbl!r}")


def load_all(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "meta_chunk_*.jsonl")))
    if not pt_files:
        raise RuntimeError(f"No chunk files in {data_dir}")
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt).numpy())
        metas.extend(json.loads(line) for line in open(mf, encoding="utf-8"))
    X = np.concatenate(X_list, axis=0)
    return X, metas


def family_from_task_id(task_id: str) -> str:
    return task_id.split("_")[0]


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    X, metas = load_all(DATA_DIR)
    T, L, H  = X.shape
    layer_idx = int(round(DEPTH_FRAC * (L - 1)))
    print(f"Data dir   : {DATA_DIR}")
    print(f"Shape      : {T} examples, {L} layers, {H} hidden")
    print(f"Fixed layer: {layer_idx}  (30% of {L-1} = {DEPTH_FRAC * (L-1):.1f})")
    print()

    labels     = np.array([normalize_label(m["label"]) for m in metas])
    family_ids = np.array([family_from_task_id(m.get("task_id", "UNK_X")) for m in metas])
    feats      = X[:, layer_idx, :]   # [T, H]

    all_families = sorted(set(family_ids.tolist()))
    print(f"Families ({len(all_families)}): {all_families}")

    # Label encodings
    LABEL3 = ["R", "N", "A"]
    L3_TO_ID = {k: i for i, k in enumerate(LABEL3)}
    y3 = np.array([L3_TO_ID[l] for l in labels], dtype=np.int64)

    RN_MAP  = {"R": 0, "N": 1}
    is_rn   = np.array([(l in RN_MAP) for l in labels])
    y_rn    = np.where(is_rn, np.array([RN_MAP.get(l, -1) for l in labels]), -1).astype(np.int64)

    rows = []

    for fold_family in all_families:
        test_mask  = (family_ids == fold_family)
        train_mask = ~test_mask

        # ---- 3-class probe ----
        clf3 = LogisticRegression(max_iter=3000)
        clf3.fit(feats[train_mask], y3[train_mask])
        proba3 = clf3.predict_proba(feats[test_mask])   # [n_test, 3]
        # column order from clf3.classes_
        idx_N3 = int(np.where(clf3.classes_ == L3_TO_ID["N"])[0][0])
        idx_A3 = int(np.where(clf3.classes_ == L3_TO_ID["A"])[0][0])

        # ---- RN probe (train only on R/N examples outside fold) ----
        train_rn_mask = train_mask & is_rn
        rn_clf = None
        rn_classes = None
        if train_rn_mask.sum() >= 4:
            rn_clf = LogisticRegression(max_iter=3000)
            rn_clf.fit(feats[train_rn_mask], y_rn[train_rn_mask])
            rn_classes = rn_clf.classes_

        test_indices = np.where(test_mask)[0]
        p3_block = proba3   # [n_test, 3]

        for local_i, global_i in enumerate(test_indices):
            lbl = labels[global_i]
            m   = metas[global_i]

            p3_N_val = float(p3_block[local_i, idx_N3])
            p3_A_val = float(p3_block[local_i, idx_A3])

            rn_margin_val = None
            rn_prob_N_val = None
            if rn_clf is not None and lbl in RN_MAP:
                feat_vec = feats[global_i].reshape(1, -1)
                dec  = rn_clf.decision_function(feat_vec)[0]
                prob = rn_clf.predict_proba(feat_vec)[0]
                # decision_function: positive = class 1 (N), negative = class 0 (R)
                # sign depends on classes_ order; make it: positive = more N-like
                idx_N_rn = int(np.where(rn_classes == RN_MAP["N"])[0][0])
                rn_margin_val = float(dec) if idx_N_rn == 1 else float(-dec)
                rn_prob_N_val = float(prob[idx_N_rn])

            rows.append({
                "task_id":     m.get("task_id", ""),
                "family_id":   family_from_task_id(m.get("task_id", "UNK_X")),
                "label":       lbl,
                "repeat":      m.get("repeat_index", m.get("repeat", "")),
                "layer_idx":   layer_idx,
                "rn_margin":   rn_margin_val,
                "rn_prob_N":   rn_prob_N_val,
                "p3_N":        p3_N_val,
                "p3_A":        p3_A_val,
            })

        print(f"  Fold {fold_family}: {int(test_mask.sum())} test examples scored")

    # Write per-example CSV
    fieldnames = ["task_id", "family_id", "label", "repeat",
                  "layer_idx", "rn_margin", "rn_prob_N", "p3_N", "p3_A"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-example scores: {OUTPUT_CSV}  ({len(rows)} rows)")

    # Write per-prompt aggregate CSV (mean across repeats)
    agg_path = OUTPUT_CSV.replace(".csv", "_per_prompt.csv")
    from collections import defaultdict
    by_task: dict = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    agg_rows = []
    for task_id, task_rows in sorted(by_task.items()):
        lbl   = task_rows[0]["label"]
        fam   = task_rows[0]["family_id"]
        layer = task_rows[0]["layer_idx"]

        def mean_of(key):
            vals = [r[key] for r in task_rows if r[key] is not None]
            return float(np.mean(vals)) if vals else None

        agg_rows.append({
            "task_id":   task_id,
            "family_id": fam,
            "label":     lbl,
            "n_repeats": len(task_rows),
            "layer_idx": layer,
            "rn_margin": mean_of("rn_margin"),
            "rn_prob_N": mean_of("rn_prob_N"),
            "p3_N":      mean_of("p3_N"),
            "p3_A":      mean_of("p3_A"),
        })

    agg_fields = ["task_id", "family_id", "label", "n_repeats",
                  "layer_idx", "rn_margin", "rn_prob_N", "p3_N", "p3_A"]
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Wrote per-prompt aggregate : {agg_path}  ({len(agg_rows)} rows)")
    print(f"Fixed layer index used     : {layer_idx}  (30% of {L-1} layers)")


if __name__ == "__main__":
    main()
