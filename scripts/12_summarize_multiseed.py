import re
import csv
import glob
import os
import sys
from statistics import mean, pstdev

# Usage:
#   python 12_summarize_multiseed.py G:\LLM\experiment\results\qwen_multiseed
#
# Produces:
#   qwen_multiseed_summary.csv
#   qwen_multiseed_report.txt

RE_FLOAT = r"[-+]?\d*\.\d+|\d+"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_first(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    return cast(m.group(1))

def extract_int(pattern: str, text: str):
    return extract_first(pattern, text, cast=int)

def extract_list3(pattern: str, text: str):
    # expects something like: [0.6, 1.0, 0.4]
    m = re.search(pattern, text)
    if not m:
        return None
    nums = re.findall(RE_FLOAT, m.group(1))
    if len(nums) < 3:
        return None
    return [float(nums[0]), float(nums[1]), float(nums[2])]

def parse_probe3class(text: str) -> dict:
    out = {}
    out["best_layer_acc"] = extract_int(r"Best layer by Acc:\s*(\d+)", text)
    out["best_acc"] = extract_first(r"Best layer by Acc:\s*\d+\s+acc:\s*(" + RE_FLOAT + r")", text)

    out["best_layer_macrof1"] = extract_int(r"Best layer by Macro-F1:\s*(\d+)", text)
    out["best_macrof1"] = extract_first(r"Best layer by Macro-F1:\s*\d+\s+macro_f1:\s*(" + RE_FLOAT + r")", text)

    # Per-class recall (R,N,A)
    out["recall_RNA"] = extract_list3(r"Per-class recall\s*\(R,N,A\):\s*\[([^\]]+)\]", text)

    # Mean predicted probs on TRUE A items [P(R), P(N), P(A)]
    out["mean_probs_A_RNA"] = extract_list3(
        r"Mean predicted probs on TRUE A items\s*\[P\(R\), P\(N\), P\(A\)\]:\s*\[([^\]]+)\]",
        text
    )
    return out

def parse_tfidf(text: str) -> dict:
    out = {}
    out["tfidf_acc"] = extract_first(r"TF-IDF prompt-holdout ACC:\s*(" + RE_FLOAT + r")", text)
    out["tfidf_macrof1"] = extract_first(r"TF-IDF prompt-holdout Macro-F1:\s*(" + RE_FLOAT + r")", text)
    out["tfidf_recall_RNA"] = extract_list3(r"Per-class recall\s*\(R,N,A\):\s*\[([^\]]+)\]", text)
    out["tfidf_mean_probs_A_RNA"] = extract_list3(
        r"Mean predicted probs on TRUE A items\s*\[P\(R\), P\(N\), P\(A\)\]:\s*\[([^\]]+)\]",
        text
    )
    return out

def seed_from_filename(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"seed(\d+)_", base)
    return int(m.group(1)) if m else -1

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None

def safe_pstdev(xs):
    xs = [x for x in xs if x is not None]
    return pstdev(xs) if len(xs) > 1 else 0.0 if xs else None

def main():
    if len(sys.argv) < 2:
        print("Usage: python 12_summarize_multiseed.py <results_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    folder = os.path.abspath(folder)

    probe_files = sorted(glob.glob(os.path.join(folder, "seed*_probe3class*.txt")))
    tfidf_files = sorted(glob.glob(os.path.join(folder, "seed*_tfidf*.txt")))

    if not probe_files:
        print("No probe files found. Expected something like seed0_probe3class.txt")
        sys.exit(2)
    if not tfidf_files:
        print("No tfidf files found. Expected something like seed0_tfidf.txt")
        sys.exit(3)

    # Map seed -> parsed dicts
    rows = {}
    for pf in probe_files:
        s = seed_from_filename(pf)
        rows.setdefault(s, {})
        rows[s]["seed"] = s
        rows[s]["probe_file"] = os.path.basename(pf)
        rows[s].update(parse_probe3class(read_text(pf)))

    for tf in tfidf_files:
        s = seed_from_filename(tf)
        rows.setdefault(s, {})
        rows[s]["seed"] = s
        rows[s]["tfidf_file"] = os.path.basename(tf)
        rows[s].update(parse_tfidf(read_text(tf)))

    # Sort by seed
    seeds = sorted([s for s in rows.keys() if s >= 0])

    # Flatten list fields into columns
    def get3(d, key, idx):
        v = d.get(key)
        return v[idx] if isinstance(v, list) and len(v) >= 3 else None

    csv_path = os.path.join(folder, "qwen_multiseed_summary.csv")
    report_path = os.path.join(folder, "qwen_multiseed_report.txt")

    fieldnames = [
        "seed",
        "probe_file",
        "tfidf_file",
        "best_layer_acc",
        "best_acc",
        "best_layer_macrof1",
        "best_macrof1",
        "recall_R", "recall_N", "recall_A",
        "meanA_PR", "meanA_PN", "meanA_PA",
        "tfidf_acc",
        "tfidf_macrof1",
        "tfidf_recall_R", "tfidf_recall_N", "tfidf_recall_A",
        "tfidf_meanA_PR", "tfidf_meanA_PN", "tfidf_meanA_PA",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in seeds:
            d = rows[s]
            row = {
                "seed": s,
                "probe_file": d.get("probe_file"),
                "tfidf_file": d.get("tfidf_file"),
                "best_layer_acc": d.get("best_layer_acc"),
                "best_acc": d.get("best_acc"),
                "best_layer_macrof1": d.get("best_layer_macrof1"),
                "best_macrof1": d.get("best_macrof1"),
                "recall_R": get3(d, "recall_RNA", 0),
                "recall_N": get3(d, "recall_RNA", 1),
                "recall_A": get3(d, "recall_RNA", 2),
                "meanA_PR": get3(d, "mean_probs_A_RNA", 0),
                "meanA_PN": get3(d, "mean_probs_A_RNA", 1),
                "meanA_PA": get3(d, "mean_probs_A_RNA", 2),
                "tfidf_acc": d.get("tfidf_acc"),
                "tfidf_macrof1": d.get("tfidf_macrof1"),
                "tfidf_recall_R": get3(d, "tfidf_recall_RNA", 0),
                "tfidf_recall_N": get3(d, "tfidf_recall_RNA", 1),
                "tfidf_recall_A": get3(d, "tfidf_recall_RNA", 2),
                "tfidf_meanA_PR": get3(d, "tfidf_mean_probs_A_RNA", 0),
                "tfidf_meanA_PN": get3(d, "tfidf_mean_probs_A_RNA", 1),
                "tfidf_meanA_PA": get3(d, "tfidf_mean_probs_A_RNA", 2),
            }
            w.writerow(row)

    # Make a small text report
    best_accs = [rows[s].get("best_acc") for s in seeds]
    best_f1s = [rows[s].get("best_macrof1") for s in seeds]
    tfidf_accs = [rows[s].get("tfidf_acc") for s in seeds]
    tfidf_f1s = [rows[s].get("tfidf_macrof1") for s in seeds]

    # Best layer distribution
    best_layers = [rows[s].get("best_layer_macrof1") for s in seeds if rows[s].get("best_layer_macrof1") is not None]
    layer_counts = {}
    for L in best_layers:
        layer_counts[L] = layer_counts.get(L, 0) + 1

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Multiseed summary folder: {folder}\n")
        f.write(f"Seeds found: {seeds}\n\n")

        f.write("Hidden-state probe (3-class prompt-holdout)\n")
        f.write(f"  Best ACC   mean±std: {safe_mean(best_accs):.4f} ± {safe_pstdev(best_accs):.4f}\n" if safe_mean(best_accs) is not None else "  Best ACC   mean±std: n/a\n")
        f.write(f"  Best MacroF1 mean±std: {safe_mean(best_f1s):.4f} ± {safe_pstdev(best_f1s):.4f}\n" if safe_mean(best_f1s) is not None else "  Best MacroF1 mean±std: n/a\n")
        if layer_counts:
            f.write(f"  Best-layer (by MacroF1) counts: {dict(sorted(layer_counts.items()))}\n")
        f.write("\n")

        f.write("TF-IDF baseline (3-class prompt-holdout)\n")
        f.write(f"  ACC      mean±std: {safe_mean(tfidf_accs):.4f} ± {safe_pstdev(tfidf_accs):.4f}\n" if safe_mean(tfidf_accs) is not None else "  ACC      mean±std: n/a\n")
        f.write(f"  MacroF1  mean±std: {safe_mean(tfidf_f1s):.4f} ± {safe_pstdev(tfidf_f1s):.4f}\n" if safe_mean(tfidf_f1s) is not None else "  MacroF1  mean±std: n/a\n")
        f.write("\n")

        f.write("Per-seed quick view:\n")
        for s in seeds:
            d = rows[s]
            f.write(
                f"  seed{s}: probe(bestF1={d.get('best_macrof1')}, bestLayerF1={d.get('best_layer_macrof1')}, "
                f"bestACC={d.get('best_acc')}) | tfidf(acc={d.get('tfidf_acc')}, f1={d.get('tfidf_macrof1')})\n"
            )

    print("Wrote:", csv_path)
    print("Wrote:", report_path)
    print("Done.")

if __name__ == "__main__":
    main()