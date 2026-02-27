import os, glob, json
import torch
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"

def load_all_with_prompts():
    pt_files = sorted(glob.glob(os.path.join(DATA_DIR, "hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "meta_chunk_*.jsonl")))
    if len(pt_files) != len(meta_files):
        raise RuntimeError(f"pt/meta count mismatch: {len(pt_files)} vs {len(meta_files)}")

    X_list = []
    y_list = []
    prompts = []

    for pt_path, meta_path in zip(pt_files, meta_files):
        x = torch.load(pt_path).numpy()  # [N, L, H]
        metas = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8")]
        if x.shape[0] != len(metas):
            raise RuntimeError(f"examples mismatch in chunk: {pt_path}")

        y = np.array([1 if m["label"] == "routine" else 0 for m in metas], dtype=np.int64)
        p = [m.get("task_prompt", "") for m in metas]
        if any(pp == "" for pp in p):
            raise RuntimeError("Some rows missing task_prompt in metadata. Re-run extraction after adding it.")

        X_list.append(x)
        y_list.append(y)
        prompts.extend(p)

    X = np.concatenate(X_list, axis=0)  # [T, L, H]
    y = np.concatenate(y_list, axis=0)  # [T]
    prompts = np.array(prompts)
    return X, y, prompts

def main():
    X, y, prompts = load_all_with_prompts()
    T, L, H = X.shape
    print("Loaded X:", X.shape, "y:", y.shape)
    print("Routine fraction:", float(y.mean()))

    unique_prompts = sorted(set(prompts.tolist()))
    print("Unique prompt texts:", len(unique_prompts))

    # Split prompts into train/test
    rng = np.random.default_rng(0)
    rng.shuffle(unique_prompts)
    split = max(1, int(0.7 * len(unique_prompts)))
    train_prompts = set(unique_prompts[:split])
    test_prompts = set(unique_prompts[split:])

    train_mask = np.array([p in train_prompts for p in prompts])
    test_mask  = np.array([p in test_prompts  for p in prompts])

    print("Train prompts:", len(train_prompts), "| Test prompts:", len(test_prompts))
    print("Train examples:", int(train_mask.sum()), "| Test examples:", int(test_mask.sum()))

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