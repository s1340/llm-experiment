import os, glob, json, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

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

def load_text_and_labels():
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "meta_chunk_*.jsonl")))
    if not meta_files:
        raise RuntimeError(f"No meta_chunk_*.jsonl files found in {DATA_DIR}")

    texts = []
    y = []
    metas = []

    for meta_path in meta_files:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                txt = m.get("task_prompt", "")
                if not txt:
                    raise RuntimeError("Missing task_prompt in metadata.")
                texts.append(txt)
                y.append(LABEL_TO_ID[normalize_label(m["label"])])
                metas.append(m)

    return np.array(texts), np.array(y, dtype=np.int64), metas

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
    print("Confusion matrix (rows=true, cols=pred) [R,N,A]:")
    for i, row in enumerate(cm):
        print(f"  {ID_TO_LABEL[i]}: {row.tolist()}")

def main():
    texts, y, metas = load_text_and_labels()
    print("Loaded texts:", texts.shape, "labels:", y.shape)
    print("Split seed:", SEED)

    counts = {lbl: int((y == idx).sum()) for lbl, idx in LABEL_TO_ID.items()}
    print("Class counts:", counts)

    train_mask, test_mask, train_prompts, test_prompts = split_by_prompt(texts, train_frac=0.7, seed=SEED)
    print("Unique prompt texts:", len(set(texts.tolist())))
    print("Train prompts:", len(train_prompts), "| Test prompts:", len(test_prompts))
    print("Train examples:", int(train_mask.sum()), "| Test examples:", int(test_mask.sum()))

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X[train_mask], y[train_mask])

    pred = clf.predict(X[test_mask])
    prob = clf.predict_proba(X[test_mask])
    y_test = y[test_mask]

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", labels=[0,1,2])

    print("TF-IDF prompt-holdout ACC:", float(acc))
    print("TF-IDF prompt-holdout Macro-F1:", float(macro_f1))

    cm = confusion_matrix(y_test, pred, labels=[0,1,2])
    print_confusion(cm)

    recalls = recall_score(y_test, pred, labels=[0,1,2], average=None, zero_division=0)
    print("Per-class recall (R,N,A):", [float(r) for r in recalls])

    a_idx = LABEL_TO_ID["A"]
    a_mask_test = (y_test == a_idx)
    if a_mask_test.any():
        a_probs = prob[a_mask_test]
        mean_probs = a_probs.mean(axis=0)
        print("Mean predicted probs on TRUE A items [P(R), P(N), P(A)]:", [float(x) for x in mean_probs])

if __name__ == "__main__":
    main()