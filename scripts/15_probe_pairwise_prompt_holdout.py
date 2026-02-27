import os, glob, json, sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

# Usage:
#   python 15_probe_pairwise_prompt_holdout.py <DATA_DIR> [SEED] [PAIR]
#
# Examples:
#   python 15_probe_pairwise_prompt_holdout.py G:\LLM\experiment\data\scale_runs_qwen 0 R,N
#   python 15_probe_pairwise_prompt_holdout.py G:\LLM\experiment\data\scale_runs_gemma 2 R,A
#   python 15_probe_pairwise_prompt_holdout.py G:\LLM\experiment\data\scale_runs_llama 4 A,N

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
PAIR_STR = sys.argv[3] if len(sys.argv) > 3 else "R,N"

def normalize_label(lbl: str) -> str:
    l = (lbl or "").strip().lower()
    if l in ["routine", "r"]:
        return "R"
    if l in ["nonroutine", "non-routine", "conceptual", "n"]:
        return "N"
    if l in ["ambiguous", "a"]:
        return "A"
    raise ValueError(f"Unknown label: {lbl!r}")

VALID = {"R", "N", "A"}

def parse_pair(pair_str: str):
    s = pair_str.replace(" ", "")
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"PAIR must look like R,N or R,A or A,N. Got: {pair_str!r}")
    a, b = parts[0].upper(), parts[1].upper()
    if a not in VALID or b not in VALID or a == b:
        raise ValueError(f"Invalid pair: {pair_str!r}")
    return a, b

PAIR_A, PAIR_B = parse_pair(PAIR_STR)
PAIR_ORDER = [PAIR_A, PAIR_B]
PAIR_TO_ID = {PAIR_A: 0, PAIR_B: 1}
ID_TO_PAIR = {0: PAIR_A, 1: PAIR_B}

def load_all_with_prompts():
    pt_files = sorted(glob.glob(os.path.join(DATA_DIR, "hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "meta_chunk_*.jsonl")))
    if len(pt_files) != len(meta_files):
        raise RuntimeError(f"pt/meta count mismatch: {len(pt_files)} vs {len(meta_files)}")
    if not pt_files:
        raise RuntimeError(f"No chunk files found in {DATA_DIR}")

    X_list = []
    labels = []
    prompts = []
    metas_all = []

    for pt_path, meta_path in zip(pt_files, meta_files):
        x = torch.load(pt_path).numpy()  # [N, L, H]
        metas = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8")]
        if x.shape[0] != len(metas):
            raise RuntimeError(f"examples mismatch in chunk: {pt_path}")

        lbls = [normalize_label(m["label"]) for m in metas]
        ps = [m.get("task_prompt", "") for m in metas]
        if any(pp == "" for pp in ps):
            raise RuntimeError("Some rows missing task_prompt in metadata.")

        X_list.append(x)
        labels.extend(lbls)
        prompts.extend(ps)
        metas_all.extend(metas)

    X = np.concatenate(X_list, axis=0)  # [T, L, H]
    labels = np.array(labels)
    prompts = np.array(prompts)
    return X, labels, prompts, metas_all

def split_by_prompt(prompts, train_frac=0.7, seed=0):
    unique_prompts = sorted(set(prompts.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_prompts)

    split = max(1, int(train_frac * len(unique_prompts)))
    train_prompts = set(unique_prompts[:split])
    test_prompts = set(unique_prompts[split:])

    train_mask = np.array([p in train_prompts for p in prompts])
    test_mask  = np.array([p in test_prompts  for p in prompts])

    return train_mask, test_mask, train_prompts, test_prompts

def print_confusion(cm):
    # rows=true, cols=pred, order=[PAIR_A, PAIR_B]
    print(f"Confusion matrix (rows=true, cols=pred) [{PAIR_A},{PAIR_B}]:")
    for i, row in enumerate(cm):
        print(f"  {ID_TO_PAIR[i]}: {row.tolist()}")

def main():
    X_all, labels_all, prompts_all, metas_all = load_all_with_prompts()
    T, L, H = X_all.shape

    print("Loaded X:", X_all.shape, "labels:", labels_all.shape)
    print("Split seed:", SEED)
    print("Pair:", f"{PAIR_A},{PAIR_B}")

    # Filter to selected pair
    keep = np.array([(lbl == PAIR_A or lbl == PAIR_B) for lbl in labels_all])
    X = X_all[keep]
    labels = labels_all[keep]
    prompts = prompts_all[keep]
    metas = [m for m, k in zip(metas_all, keep) if k]

    y = np.array([PAIR_TO_ID[lbl] for lbl in labels], dtype=np.int64)

    print("Filtered X:", X.shape, "y:", y.shape)
    counts = {PAIR_A: int((y == 0).sum()), PAIR_B: int((y == 1).sum())}
    print("Pair class counts:", counts)

    train_mask, test_mask, train_prompts, test_prompts = split_by_prompt(prompts, train_frac=0.7, seed=SEED)
    print("Unique prompt texts (within pair):", len(set(prompts.tolist())))
    print("Train prompts:", len(train_prompts), "| Test prompts:", len(test_prompts))
    print("Train examples:", int(train_mask.sum()), "| Test examples:", int(test_mask.sum()))

    # Per-layer metrics
    layer_acc = []
    layer_macro_f1 = []
    models = {}

    for layer in range(L):
        feats = X[:, layer, :]  # [T_pair, H]
        clf = LogisticRegression(max_iter=3000)
        clf.fit(feats[train_mask], y[train_mask])
        pred = clf.predict(feats[test_mask])

        acc = accuracy_score(y[test_mask], pred)
        macro_f1 = f1_score(y[test_mask], pred, average="macro", labels=[0, 1])

        layer_acc.append(float(acc))
        layer_macro_f1.append(float(macro_f1))
        models[layer] = clf

    best_layer_acc = int(np.argmax(layer_acc))
    best_layer_f1 = int(np.argmax(layer_macro_f1))

    print("Best layer by ACC:", best_layer_acc, "acc:", layer_acc[best_layer_acc])
    print("Best layer by Macro-F1:", best_layer_f1, "macro_f1:", layer_macro_f1[best_layer_f1])

    print("First 5 layer acc:", layer_acc[:5])
    print("Last 5 layer acc:", layer_acc[-5:])
    print("First 5 layer macro_f1:", layer_macro_f1[:5])
    print("Last 5 layer macro_f1:", layer_macro_f1[-5:])

    # Detailed diagnostics at best Macro-F1 layer
    layer = best_layer_f1
    feats = X[:, layer, :]
    clf = models[layer]

    pred = clf.predict(feats[test_mask])
    prob = clf.predict_proba(feats[test_mask])  # shape [n_test, 2]
    y_test = y[test_mask]

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print_confusion(cm)

    recalls = recall_score(y_test, pred, labels=[0, 1], average=None, zero_division=0)
    print(f"Per-class recall ({PAIR_A},{PAIR_B}):", [float(r) for r in recalls])

    # Mean probs for each true class
    for class_id in [0, 1]:
        cls_label = ID_TO_PAIR[class_id]
        mask = (y_test == class_id)
        if mask.any():
            mean_probs = prob[mask].mean(axis=0)
            print(
                f"Mean predicted probs on TRUE {cls_label} items [P({PAIR_A}), P({PAIR_B})]:",
                [float(x) for x in mean_probs]
            )

    # A few sample lines for the "second" class in the pair (often the more interesting side)
    # (purely diagnostic; can be ignored if noisy)
    target_class_id = 1
    mask = (y_test == target_class_id)
    if mask.any():
        print(f"Sample TRUE {ID_TO_PAIR[target_class_id]} items (up to 5) with predicted probs:")
        test_indices = np.where(test_mask)[0]
        global_indices = test_indices[mask]
        for gi in global_indices[:5]:
            pvec = clf.predict_proba(X[gi, layer, :].reshape(1, -1))[0]
            pred_lbl = ID_TO_PAIR[int(np.argmax(pvec))]
            true_lbl = ID_TO_PAIR[int(y[gi])]
            task_id = metas[gi].get("task_id", "<no_task_id>")
            prompt_preview = metas[gi].get("task_prompt", "")[:120]
            print(
                f"  task_id={task_id} true={true_lbl} pred={pred_lbl} "
                f"probs=[{PAIR_A}:{pvec[0]:.3f}, {PAIR_B}:{pvec[1]:.3f}] "
                f"prompt={prompt_preview!r}"
            )

if __name__ == "__main__":
    main()