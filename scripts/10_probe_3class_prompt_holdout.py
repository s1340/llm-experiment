import os, glob, json, sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Map labels to 3 classes:
# routine -> R
# nonroutine / conceptual -> N
# ambiguous -> A
# (Adjust if your dataset uses different exact strings)
def normalize_label(lbl: str) -> str:
    l = (lbl or "").strip().lower()
    if l in ["routine", "r"]:
        return "R"
    if l in ["nonroutine", "non-routine", "conceptual", "n"]:
        return "N"
    if l in ["ambiguous", "a"]:
        return "A"
    raise ValueError(f"Unknown label: {lbl!r}")

LABEL_ORDER = ["R", "N", "A"]
LABEL_TO_ID = {k: i for i, k in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: k for k, i in LABEL_TO_ID.items()}

def load_all_with_prompts():
    pt_files = sorted(glob.glob(os.path.join(DATA_DIR, "hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "meta_chunk_*.jsonl")))
    if len(pt_files) != len(meta_files):
        raise RuntimeError(f"pt/meta count mismatch: {len(pt_files)} vs {len(meta_files)}")
    if not pt_files:
        raise RuntimeError(f"No chunk files found in {DATA_DIR}")

    X_list = []
    y_list = []
    prompts = []
    metas_all = []

    for pt_path, meta_path in zip(pt_files, meta_files):
        x = torch.load(pt_path).numpy()  # [N, L, H]
        metas = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8")]
        if x.shape[0] != len(metas):
            raise RuntimeError(f"examples mismatch in chunk: {pt_path}")

        y = np.array([LABEL_TO_ID[normalize_label(m["label"])] for m in metas], dtype=np.int64)
        p = [m.get("task_prompt", "") for m in metas]
        if any(pp == "" for pp in p):
            raise RuntimeError("Some rows missing task_prompt in metadata. Re-run extraction with task_prompt saved.")

        X_list.append(x)
        y_list.append(y)
        prompts.extend(p)
        metas_all.extend(metas)

    X = np.concatenate(X_list, axis=0)  # [T, L, H]
    y = np.concatenate(y_list, axis=0)  # [T]
    prompts = np.array(prompts)
    return X, y, prompts, metas_all

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
    # cm rows=true, cols=pred, order=R,N,A
    print("Confusion matrix (rows=true, cols=pred) [R,N,A]:")
    for i, row in enumerate(cm):
        print(f"  {ID_TO_LABEL[i]}: {row.tolist()}")

def main():
    X, y, prompts, metas = load_all_with_prompts()
    T, L, H = X.shape
    print("Loaded X:", X.shape, "y:", y.shape)
    print("Split seed:", SEED)

    # Class balance
    counts = {lbl: int((y == idx).sum()) for lbl, idx in LABEL_TO_ID.items()}
    print("Class counts:", counts)

    train_mask, test_mask, train_prompts, test_prompts = split_by_prompt(prompts, train_frac=0.7, seed=SEED)
    print("Unique prompt texts:", len(set(prompts.tolist())))
    print("Train prompts:", len(train_prompts), "| Test prompts:", len(test_prompts))
    print("Train examples:", int(train_mask.sum()), "| Test examples:", int(test_mask.sum()))

    # Per-layer metrics
    layer_acc = []
    layer_macro_f1 = []

    best_models = {}  # store fitted model per layer if needed later

    for layer in range(L):
        feats = X[:, layer, :]  # [T, H]
        clf = LogisticRegression(max_iter=3000)
        clf.fit(feats[train_mask], y[train_mask])
        pred = clf.predict(feats[test_mask])

        acc = accuracy_score(y[test_mask], pred)
        macro_f1 = f1_score(y[test_mask], pred, average="macro", labels=[0,1,2])

        layer_acc.append(float(acc))
        layer_macro_f1.append(float(macro_f1))
        best_models[layer] = clf

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
    clf = best_models[layer]

    pred = clf.predict(feats[test_mask])
    prob = clf.predict_proba(feats[test_mask])
    y_test = y[test_mask]

    cm = confusion_matrix(y_test, pred, labels=[0,1,2])
    print_confusion(cm)

    recalls = recall_score(y_test, pred, labels=[0,1,2], average=None, zero_division=0)
    print("Per-class recall (R,N,A):", [float(r) for r in recalls])

    # A-class probability diagnostics (on true A examples in test set)
    a_idx = LABEL_TO_ID["A"]
    a_mask_test = (y_test == a_idx)
    if a_mask_test.any():
        a_probs = prob[a_mask_test]  # shape [num_A, 3]
        mean_probs = a_probs.mean(axis=0)
        print("Mean predicted probs on TRUE A items [P(R), P(N), P(A)]:", [float(x) for x in mean_probs])

        # Optional: print a few examples
        test_indices = np.where(test_mask)[0]
        a_global_indices = test_indices[a_mask_test]
        print("Sample TRUE A items (up to 5) with predicted probs:")
        for gi in a_global_indices[:5]:
            pvec = clf.predict_proba(X[gi, layer, :].reshape(1, -1))[0]
            pred_lbl = ID_TO_LABEL[int(np.argmax(pvec))]
            true_lbl = ID_TO_LABEL[int(y[gi])]
            task_id = metas[gi].get("task_id", "<no_task_id>")
            prompt_preview = metas[gi].get("task_prompt", "")[:120]
            print(
                f"  task_id={task_id} true={true_lbl} pred={pred_lbl} "
                f"probs=[R:{pvec[0]:.3f}, N:{pvec[1]:.3f}, A:{pvec[2]:.3f}] "
                f"prompt={prompt_preview!r}"
            )
    else:
        print("No true A items in test set for this split.")

if __name__ == "__main__":
    main()