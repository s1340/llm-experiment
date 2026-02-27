import re
import csv
import glob
import os
import sys
from statistics import mean, pstdev

# Usage:
#   python 14_summarize_sentence_embed_multiseed.py <sentence_embed_folder>
#
# Example:
#   python 14_summarize_sentence_embed_multiseed.py G:\LLM\experiment\results\qwen_multiseed\sentence_embed

RE_FLOAT = r"[-+]?\d*\.\d+|\d+"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_first(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    return cast(m.group(1))

def extract_list3(pattern: str, text: str):
    m = re.search(pattern, text)
    if not m:
        return None
    nums = re.findall(RE_FLOAT, m.group(1))
    if len(nums) < 3:
        return None
    return [float(nums[0]), float(nums[1]), float(nums[2])]

def seed_from_filename(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"seed(\d+)_", base)
    return int(m.group(1)) if m else -1

def parse_sentence_embed(text: str) -> dict:
    out = {}
    out["acc"] = extract_first(r"Sentence-embedding prompt-holdout ACC:\s*(" + RE_FLOAT + r")", text)
    out["macrof1"] = extract_first(r"Sentence-embedding prompt-holdout Macro-F1:\s*(" + RE_FLOAT + r")", text)
    out["recall_RNA"] = extract_list3(r"Per-class recall\s*\(R,N,A\):\s*\[([^\]]+)\]", text)
    out["mean_probs_A_RNA"] = extract_list3(
        r"Mean predicted probs on TRUE A items\s*\[P\(R\), P\(N\), P\(A\)\]:\s*\[([^\]]+)\]",
        text
    )
    out["embed_shape_dim"] = extract_first(r"Embedding matrix shape:\s*\(\d+,\s*(\d+)\)", text, cast=int)
    return out

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None

def safe_pstdev(xs):
    xs = [x for x in xs if x is not None]
    return pstdev(xs) if len(xs) > 1 else 0.0 if xs else None

def get3(d, key, idx):
    v = d.get(key)
    return v[idx] if isinstance(v, list) and len(v) >= 3 else None

def main():
    if len(sys.argv) < 2:
        print("Usage: python 14_summarize_sentence_embed_multiseed.py <sentence_embed_folder>")
        sys.exit(1)

    folder = os.path.abspath(sys.argv[1])
    files = sorted(glob.glob(os.path.join(folder, "seed*_sentence_embed*.txt")))

    if not files:
        print("No sentence-embedding files found. Expected something like seed0_sentence_embed.txt")
        sys.exit(2)

    rows = {}
    for fp in files:
        s = seed_from_filename(fp)
        rows[s] = {
            "seed": s,
            "file": os.path.basename(fp),
            **parse_sentence_embed(read_text(fp))
        }

    seeds = sorted([s for s in rows.keys() if s >= 0])

    csv_path = os.path.join(folder, "sentence_embed_multiseed_summary.csv")
    report_path = os.path.join(folder, "sentence_embed_multiseed_report.txt")

    fieldnames = [
        "seed", "file",
        "embed_shape_dim",
        "acc", "macrof1",
        "recall_R", "recall_N", "recall_A",
        "meanA_PR", "meanA_PN", "meanA_PA",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in seeds:
            d = rows[s]
            w.writerow({
                "seed": s,
                "file": d.get("file"),
                "embed_shape_dim": d.get("embed_shape_dim"),
                "acc": d.get("acc"),
                "macrof1": d.get("macrof1"),
                "recall_R": get3(d, "recall_RNA", 0),
                "recall_N": get3(d, "recall_RNA", 1),
                "recall_A": get3(d, "recall_RNA", 2),
                "meanA_PR": get3(d, "mean_probs_A_RNA", 0),
                "meanA_PN": get3(d, "mean_probs_A_RNA", 1),
                "meanA_PA": get3(d, "mean_probs_A_RNA", 2),
            })

    accs = [rows[s].get("acc") for s in seeds]
    f1s = [rows[s].get("macrof1") for s in seeds]
    rR = [get3(rows[s], "recall_RNA", 0) for s in seeds]
    rN = [get3(rows[s], "recall_RNA", 1) for s in seeds]
    rA = [get3(rows[s], "recall_RNA", 2) for s in seeds]
    pR = [get3(rows[s], "mean_probs_A_RNA", 0) for s in seeds]
    pN = [get3(rows[s], "mean_probs_A_RNA", 1) for s in seeds]
    pA = [get3(rows[s], "mean_probs_A_RNA", 2) for s in seeds]

    dims = sorted(set([rows[s].get("embed_shape_dim") for s in seeds if rows[s].get("embed_shape_dim") is not None]))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Sentence-embedding multiseed summary folder: {folder}\n")
        f.write(f"Seeds found: {seeds}\n")
        f.write(f"Embedding dims observed: {dims}\n\n")

        f.write("Sentence-embedding baseline (3-class prompt-holdout)\n")
        f.write(f"  ACC      mean±std: {safe_mean(accs):.4f} ± {safe_pstdev(accs):.4f}\n")
        f.write(f"  MacroF1  mean±std: {safe_mean(f1s):.4f} ± {safe_pstdev(f1s):.4f}\n")
        f.write(f"  Recall R mean±std: {safe_mean(rR):.4f} ± {safe_pstdev(rR):.4f}\n")
        f.write(f"  Recall N mean±std: {safe_mean(rN):.4f} ± {safe_pstdev(rN):.4f}\n")
        f.write(f"  Recall A mean±std: {safe_mean(rA):.4f} ± {safe_pstdev(rA):.4f}\n")
        f.write(f"  Mean probs on TRUE A [P(R)] mean±std: {safe_mean(pR):.4f} ± {safe_pstdev(pR):.4f}\n")
        f.write(f"  Mean probs on TRUE A [P(N)] mean±std: {safe_mean(pN):.4f} ± {safe_pstdev(pN):.4f}\n")
        f.write(f"  Mean probs on TRUE A [P(A)] mean±std: {safe_mean(pA):.4f} ± {safe_pstdev(pA):.4f}\n")
        f.write("\nPer-seed quick view:\n")
        for s in seeds:
            d = rows[s]
            rec = d.get("recall_RNA")
            probs = d.get("mean_probs_A_RNA")
            f.write(
                f"  seed{s}: acc={d.get('acc')} macrof1={d.get('macrof1')} "
                f"recall(R,N,A)={rec} meanA_probs={probs}\n"
            )

    print("Wrote:", csv_path)
    print("Wrote:", report_path)
    print("Done.")

if __name__ == "__main__":
    main()