import os, glob, json
import torch
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"

def load_all():
    pt_files = sorted(glob.glob(os.path.join(DATA_DIR, "hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "meta_chunk_*.jsonl")))
    if len(pt_files) != len(meta_files):
        raise RuntimeError(f"pt/meta count mismatch: {len(pt_files)} vs {len(meta_files)}")

    X_list = []
    y_list = []

    for pt_path, meta_path in zip(pt_files, meta_files):
        x = torch.load(pt_path).numpy()  # [N, L, H]
        metas = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8")]

        if x.shape[0] != len(metas):
            raise RuntimeError(f"examples mismatch in chunk: {pt_path}")

        y = np.array([1 if m["label"] == "routine" else 0 for m in metas], dtype=np.int64)

        X_list.append(x)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)  # [T, L, H]
    y = np.concatenate(y_list, axis=0)  # [T]
    return X, y

def main():
    X, y = load_all()
    T, L, H = X.shape
    print("Loaded X shape:", X.shape, "y shape:", y.shape)
    print("Routine fraction:", float(y.mean()))

    # Quick per-layer probe
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    layer_acc = []

    for layer in range(L):
        feats = X[:, layer, :]  # [T, H]
        accs = []
        for train_idx, test_idx in skf.split(feats, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(feats[train_idx], y[train_idx])
            pred = clf.predict(feats[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
        layer_acc.append(float(np.mean(accs)))

    best_layer = int(np.argmax(layer_acc))
    print("Best layer:", best_layer, "acc:", layer_acc[best_layer])
    print("First 5 layer acc:", layer_acc[:5])
    print("Last 5 layer acc:", layer_acc[-5:])

if __name__ == "__main__":
    main()