import os, sys, json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report

# Usage:
#   python 24_baseline_lofo.py <TASKS_JSON> <OUTPUT_DOC>
#
# Runs sentence-embedding (all-MiniLM-L6-v2) baseline under leave-one-family-out CV
# (same fold structure as the main hidden-state probe analysis).
# 20 families, 3 prompts each, 60 total. Each fold: train on 19 families, test on 1.
#
# Output: docs/baseline_lofo.md

TASKS_JSON  = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\tasks_v2_hard.json"
OUTPUT_DOC  = sys.argv[2] if len(sys.argv) > 2 else r"G:\LLM\experiment\docs\baseline_lofo.md"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABEL_ORDER = ["routine", "ambiguous", "nonroutine"]


def main():
    with open(TASKS_JSON, encoding="utf-8") as f:
        tasks = json.load(f)

    prompts    = [t["task_prompt"] for t in tasks]
    labels     = [t["label"]       for t in tasks]   # routine / ambiguous / nonroutine
    task_ids   = [t["task_id"]     for t in tasks]
    family_ids = [t["task_id"].split("_")[0] for t in tasks]

    families = sorted(set(family_ids))
    print(f"Tasks: {len(tasks)}  Families: {len(families)}")
    print(f"Label distribution: { {l: labels.count(l) for l in LABEL_ORDER} }")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    X = embed_model.encode(prompts, show_progress_bar=True)
    print(f"Embeddings shape: {X.shape}")

    # Leave-one-family-out CV
    all_true, all_pred = [], []
    fold_results = []

    for fam in families:
        test_mask  = [f == fam  for f in family_ids]
        train_mask = [f != fam  for f in family_ids]

        X_train = X[np.array(train_mask)]
        y_train = [labels[i] for i, m in enumerate(train_mask) if m]
        X_test  = X[np.array(test_mask)]
        y_test  = [labels[i] for i, m in enumerate(test_mask) if m]

        clf = LogisticRegression(max_iter=3000)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(pred)

        fold_f1  = f1_score(y_test, pred, average="macro", labels=LABEL_ORDER, zero_division=0)
        fold_acc = accuracy_score(y_test, pred)
        fold_results.append({"family": fam, "true": y_test, "pred": pred.tolist(),
                              "macro_f1": fold_f1, "acc": fold_acc})
        print(f"  {fam}  true={y_test}  pred={pred.tolist()}  F1={fold_f1:.3f}")

    # Overall metrics
    macro_f1  = f1_score(all_true, all_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    acc       = accuracy_score(all_true, all_pred)
    recall_r  = recall_score(all_true, all_pred, labels=["routine"],    average=None, zero_division=0)[0]
    recall_a  = recall_score(all_true, all_pred, labels=["ambiguous"],  average=None, zero_division=0)[0]
    recall_n  = recall_score(all_true, all_pred, labels=["nonroutine"], average=None, zero_division=0)[0]

    fold_f1s  = [r["macro_f1"] for r in fold_results]

    print(f"\n{'='*50}")
    print(f"LOFO CV Results (all-MiniLM-L6-v2)")
    print(f"  MacroF1 : {macro_f1:.4f}  (fold mean±std: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f})")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Recall R: {recall_r:.4f}  A: {recall_a:.4f}  N: {recall_n:.4f}")
    print()
    print(classification_report(all_true, all_pred, labels=LABEL_ORDER, zero_division=0))

    # Write doc
    lines = []
    lines.append("# Sentence-Embedding Baseline: Leave-One-Family-Out CV")
    lines.append("")
    lines.append("Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (dim=384).")
    lines.append("Classifier: logistic regression (sklearn defaults, max_iter=3000).")
    lines.append("Evaluation: **leave-one-family-out CV** — 20 folds, 1 family (3 prompts) held out per fold.")
    lines.append("This matches the fold structure of the main hidden-state probe analysis,")
    lines.append("making the comparison directly interpretable.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| MacroF1 (aggregate) | {macro_f1:.4f} |")
    lines.append(f"| MacroF1 (fold mean ± std) | {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f} |")
    lines.append(f"| Accuracy | {acc:.4f} |")
    lines.append(f"| Recall (Routine) | {recall_r:.4f} |")
    lines.append(f"| Recall (Ambiguous) | {recall_a:.4f} |")
    lines.append(f"| Recall (Nonroutine) | {recall_n:.4f} |")
    lines.append(f"| n prompts | 60 |")
    lines.append(f"| n families | {len(families)} |")
    lines.append("")
    lines.append("## Comparison with random 70/30 baseline")
    lines.append("")
    lines.append("| Protocol | MacroF1 mean ± std |")
    lines.append("|----------|-------------------|")
    lines.append("| Random 70/30 split (seeds 0–4) | 0.4488 ± 0.1697 |")
    lines.append(f"| Leave-one-family-out (this run) | {macro_f1:.4f} (aggregate); fold {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f} |")
    lines.append("| Hidden-state probe best layer (LOFO) | 0.5681–0.5807 (model-dependent) |")
    lines.append("")
    lines.append("The LOFO sentence-embedding baseline is a direct apples-to-apples comparator")
    lines.append("for the hidden-state probe under the same CV protocol.")
    lines.append("")
    lines.append("## Per-fold results")
    lines.append("")
    lines.append("| Family | True labels | Predicted | MacroF1 |")
    lines.append("|--------|------------|-----------|---------|")
    for r in fold_results:
        true_str = ", ".join(r["true"])
        pred_str = ", ".join(r["pred"])
        lines.append(f"| {r['family']} | {true_str} | {pred_str} | {r['macro_f1']:.3f} |")
    lines.append("")
    lines.append("## Classification report")
    lines.append("")
    lines.append("```")
    lines.append(classification_report(all_true, all_pred, labels=LABEL_ORDER, zero_division=0))
    lines.append("```")

    doc = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_DOC), exist_ok=True)
    with open(OUTPUT_DOC, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\nSaved: {OUTPUT_DOC}")


if __name__ == "__main__":
    main()
