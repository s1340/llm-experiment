import os, glob, json
import torch
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"

def load_all_with_meta():
    pt_files = sorted(glob.glob(os.path.join(DATA_DIR, "hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "meta_chunk_*.jsonl")))
    if len(pt_files) != len(meta_files):
        raise RuntimeError(f"pt/meta count mismatch: {len(pt_files)} vs {len(meta_files)}")

    X_list = []
    y_list = []
    task_id_list = []

    for pt_path, meta_path in zip(pt_files, meta_files):
        x = torch.load(pt_path).numpy()  # [N, L, H]
        metas = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8")]
        if x.shape[0] != len(metas):
            raise RuntimeError(f"examples mismatch in chunk: {pt_path}")

        y = np.array([1 if m["label"] == "routine" else 0 for m in metas], dtype=np.int64)
        task_ids = [m["task_id"] for m in metas]

        X_list.append(x)
        y_list.append(y)
        task_id_list.extend(task_ids)

    X = np.concatenate(X_list, axis=0)   # [T, L, H]
    y = np.concatenate(y_list, axis=0)   # [T]
    task_id_list = np.array(task_id_list)
    return X, y, task_id_list

def main():
    X, y, task_ids = load_all_with_meta()
    T, L, H = X.shape
    print("Loaded X:", X.shape, "y:", y.shape)
    print("Routine fraction:", float(y.mean()))
    unique_tasks = sorted(set(task_ids.tolist()))
    print("Unique task_ids:", len(unique_tasks), unique_tasks)

    # Split task_ids into train/test
    rng = np.random.default_rng(0)
    rng.shuffle(unique_tasks)
    split = max(1, int(0.7 * len(unique_tasks)))  # 70% train, 30% test
    train_tasks = set(unique_tasks[:split])
    test_tasks = set(unique_tasks[split:])

    train_mask = np.array([tid in train_tasks for tid in task_ids])
    test_mask  = np.array([tid in test_tasks  for tid in task_ids])

    print("Train tasks:", sorted(train_tasks))
    print("Test tasks:", sorted(test_tasks))
    print("Train examples:", int(train_mask.sum()), "Test examples:", int(test_mask.sum()))

    # Probe per layer
    layer_acc = []
    for layer in range(L):
        feats = X[:, layer, :]  # [T, H]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(feats[train_mask], y[train_mask])
        pred = clf.predict(feats[test_mask])
        acc = accuracy_score(y[test_mask], pred)
        layer_acc.append(float(acc))

    best_layer = int(np.argmax(layer_acc))
    print("Best layer:", best_layer, "acc:", layer_acc[best_layer])
    print("First 5 layer acc:", layer_acc[:5])
    print("Last 5 layer acc:", layer_acc[-5:])

if __name__ == "__main__":
    main()