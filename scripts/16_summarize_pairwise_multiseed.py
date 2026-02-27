import re
import csv
import glob
import os
import sys
from statistics import mean, pstdev

# Usage:
#   python 16_summarize_pairwise_multiseed.py <pairwise_results_folder>
#
# Example:
#   python 16_summarize_pairwise_multiseed.py G:\LLM\experiment\results\pairwise_qwen
#
# Expected files:
#   seed0_RN.txt, seed0_RA.txt, seed0_AN.txt, seed1_RN.txt, ...
#
# Outputs:
#   pairwise_summary.csv
#   pairwise_report.txt

RE_FLOAT = r"[-+]?\d*\.\d+|\d+"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_first(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    return cast(m.group(1))

def extract_list(pattern: str, text: str):
    m = re.search(pattern, text)
    if not m:
        return None
    nums = re.findall(RE_FLOAT, m.group(1))
    return [float(x) for x in nums] if nums else None

def seed_and_pair_from_filename(path: str):
    base = os.path.basename(path)
    # supports seed0_RN.txt, seed3_RA.txt, seed12_AN.txt
    m = re.match(r"seed(\d+)_([A-Za-z]{2})", base)
    if not m:
        return None, None
    seed = int(m.group(1))
    pair = m.group(2).upper()
    return seed, pair

def parse_pairwise_log(text: str):
    out = {}
    out["seed"] = extract_first(r"Split seed:\s*(\d+)", text, cast=int)

    # Pair line can be "Pair: R,N"
    pair_str = None
    m_pair = re.search(r"Pair:\s*([RNA]),([RNA])", text)
    if m_pair:
        pair_str = f"{m_pair.group(1)},{m_pair.group(2)}"
    out["pair_str"] = pair_str

    out["best_layer_acc"] = extract_first(r"Best layer by ACC:\s*(\d+)", text, cast=int)
    out["best_acc"] = extract_first(r"Best layer by ACC:\s*\d+\s+acc:\s*(" + RE_FLOAT + r")", text)

    out["best_layer_macrof1"] = extract_first(r"Best layer by Macro-F1:\s*(\d+)", text, cast=int)
    out["best_macrof1"] = extract_first(r"Best layer by Macro-F1:\s*\d+\s+macro_f1:\s*(" + RE_FLOAT + r")", text)

    # Example line: Per-class recall (R,N): [0.8, 1.0]
    m_rec = re.search(r"Per-class recall\s*\(([RNA]),([RNA])\):\s*\[([^\]]+)\]", text)
    if m_rec:
        out["recall_labels"] = (m_rec.group(1), m_rec.group(2))
        nums = re.findall(RE_FLOAT, m_rec.group(3))
        if len(nums) >= 2:
            out["recall_0"] = float(nums[0])
            out["recall_1"] = float(nums[1])

    # Mean probs lines:
    # Mean predicted probs on TRUE R items [P(R), P(N)]: [.., ..]
    # Mean predicted probs on TRUE A items [P(A), P(N)]: [.., ..]
    mean_probs = {}
    for m in re.finditer(
        r"Mean predicted probs on TRUE ([RNA]) items\s*\[P\(([RNA])\), P\(([RNA])\)\]:\s*\[([^\]]+)\]",
        text
    ):
        true_cls = m.group(1)
        p_label_0 = m.group(2)
        p_label_1 = m.group(3)
        nums = re.findall(RE_FLOAT, m.group(4))
        if len(nums) >= 2:
            mean_probs[true_cls] = {
                "prob_order": (p_label_0, p_label_1),
                "vals": (float(nums[0]), float(nums[1]))
            }
    out["mean_probs"] = mean_probs

    return out

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None

def safe_pstdev(xs):
    xs = [x for x in xs if x is not None]
    return pstdev(xs) if len(xs) > 1 else 0.0 if xs else None

def fmt_mean_std(xs):
    m = safe_mean(xs)
    s = safe_pstdev(xs)
    if m is None:
        return "n/a"
    return f"{m:.4f} ± {s:.4f}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python 16_summarize_pairwise_multiseed.py <pairwise_results_folder>")
        sys.exit(1)

    folder = os.path.abspath(sys.argv[1])
    files = sorted(glob.glob(os.path.join(folder, "seed*_*.txt")))
    if not files:
        print("No pairwise txt files found. Expected e.g. seed0_RN.txt")
        sys.exit(2)

    rows = []
    for fp in files:
        seed_from_name, pair_from_name = seed_and_pair_from_filename(fp)
        if seed_from_name is None:
            continue
        txt = read_text(fp)
        parsed = parse_pairwise_log(txt)

        row = {
            "file": os.path.basename(fp),
            "seed_file": seed_from_name,
            "pair_file": pair_from_name,
            "seed_log": parsed.get("seed"),
            "pair_log": parsed.get("pair_str"),
            "best_layer_acc": parsed.get("best_layer_acc"),
            "best_acc": parsed.get("best_acc"),
            "best_layer_macrof1": parsed.get("best_layer_macrof1"),
            "best_macrof1": parsed.get("best_macrof1"),
            "recall_label_0": None,
            "recall_label_1": None,
            "recall_0": parsed.get("recall_0"),
            "recall_1": parsed.get("recall_1"),
        }

        if parsed.get("recall_labels"):
            row["recall_label_0"] = parsed["recall_labels"][0]
            row["recall_label_1"] = parsed["recall_labels"][1]

        # flatten mean probs for up to 3 classes, depending on pair
        # columns named like meanprob_TRUE_R_P0 / P1, etc.
        mean_probs = parsed.get("mean_probs", {})
        for cls in ["R", "N", "A"]:
            mp = mean_probs.get(cls)
            if mp and "vals" in mp:
                row[f"meanprob_TRUE_{cls}_P0"] = mp["vals"][0]
                row[f"meanprob_TRUE_{cls}_P1"] = mp["vals"][1]
                row[f"meanprob_TRUE_{cls}_order"] = ",".join(mp["prob_order"])
            else:
                row[f"meanprob_TRUE_{cls}_P0"] = None
                row[f"meanprob_TRUE_{cls}_P1"] = None
                row[f"meanprob_TRUE_{cls}_order"] = None

        rows.append(row)

    # Sort rows
    rows.sort(key=lambda r: (r["pair_file"] or "", r["seed_file"] if r["seed_file"] is not None else -1))

    # Write CSV
    csv_path = os.path.join(folder, "pairwise_summary.csv")
    report_path = os.path.join(folder, "pairwise_report.txt")

    fieldnames = [
        "file",
        "seed_file", "pair_file",
        "seed_log", "pair_log",
        "best_layer_acc", "best_acc",
        "best_layer_macrof1", "best_macrof1",
        "recall_label_0", "recall_label_1",
        "recall_0", "recall_1",
        "meanprob_TRUE_R_P0", "meanprob_TRUE_R_P1", "meanprob_TRUE_R_order",
        "meanprob_TRUE_N_P0", "meanprob_TRUE_N_P1", "meanprob_TRUE_N_order",
        "meanprob_TRUE_A_P0", "meanprob_TRUE_A_P1", "meanprob_TRUE_A_order",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Group by pair
    pairs = ["RN", "RA", "AN"]
    grouped = {p: [r for r in rows if (r.get("pair_file") or "").upper() == p] for p in pairs}

    def best_layer_counts(rows_for_pair):
        counts = {}
        for r in rows_for_pair:
            L = r.get("best_layer_macrof1")
            if L is not None:
                counts[L] = counts.get(L, 0) + 1
        return dict(sorted(counts.items()))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Pairwise multiseed summary folder: {folder}\n")
        f.write(f"Files parsed: {len(rows)}\n\n")

        for p in pairs:
            rs = grouped[p]
            if not rs:
                f.write(f"=== Pair {p} ===\nNo files found.\n\n")
                continue

            best_accs = [r.get("best_acc") for r in rs]
            best_f1s = [r.get("best_macrof1") for r in rs]
            best_layers = [r.get("best_layer_macrof1") for r in rs]

            f.write(f"=== Pair {p} ===\n")
            f.write(f"Seeds: {[r.get('seed_file') for r in rs]}\n")
            f.write(f"Best ACC mean±std: {fmt_mean_std(best_accs)}\n")
            f.write(f"Best Macro-F1 mean±std: {fmt_mean_std(best_f1s)}\n")
            f.write(f"Best-layer (by Macro-F1) counts: {best_layer_counts(rs)}\n")

            # Recall summary (if present)
            r0s = [r.get("recall_0") for r in rs]
            r1s = [r.get("recall_1") for r in rs]
            lbl0 = next((r.get("recall_label_0") for r in rs if r.get("recall_label_0")), None)
            lbl1 = next((r.get("recall_label_1") for r in rs if r.get("recall_label_1")), None)
            if lbl0 and lbl1:
                f.write(f"Recall {lbl0} mean±std: {fmt_mean_std(r0s)}\n")
                f.write(f"Recall {lbl1} mean±std: {fmt_mean_std(r1s)}\n")

            f.write("Per-seed quick view:\n")
            for r in rs:
                f.write(
                    f"  seed{r.get('seed_file')}: bestF1={r.get('best_macrof1')} "
                    f"bestLayerF1={r.get('best_layer_macrof1')} bestACC={r.get('best_acc')} "
                    f"recalls=({r.get('recall_label_0')}:{r.get('recall_0')}, {r.get('recall_label_1')}:{r.get('recall_1')})\n"
                )
            f.write("\n")

        f.write("Notes:\n")
        f.write("- Pair codes: RN = R vs N, RA = R vs A, AN = A vs N\n")
        f.write("- Best-layer values are raw indices from the model's hidden-state stack.\n")
        f.write("- For cross-model comparison, normalize by total layers later (proportional depth).\n")

    print("Wrote:", csv_path)
    print("Wrote:", report_path)
    print("Done.")

if __name__ == "__main__":
    main()