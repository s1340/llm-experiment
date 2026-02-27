import os, sys, csv, json
import numpy as np
from scipy import stats
from collections import defaultdict

# Usage:
#   python 20_correlate.py <SELF_REPORTS_DIR> <CV_SCORES_DIR> <OUTPUT_DIR>
#
# Expects:
#   SELF_REPORTS_DIR/  qwen_self_reports.jsonl, gemma_self_reports.jsonl, llama_self_reports.jsonl
#   CV_SCORES_DIR/     qwen_cv_scores_per_prompt.csv, gemma_..., llama_...
#
# Computes per model:
#   - Spearman r + p-value: rating_parsed vs rn_margin
#   - Spearman r + p-value: rating_parsed vs p3_N
#   - Pearson r  + p-value: both
#   - Partial Spearman r controlling for response_char_count (via residuals)
#
# Outputs:
#   OUTPUT_DIR/correlation_report.txt
#   OUTPUT_DIR/correlation_summary.csv
#   OUTPUT_DIR/{model}_joined.csv  (per-prompt joined data for inspection)

SELF_REPORTS_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\results\self_reports"
CV_SCORES_DIR    = sys.argv[2] if len(sys.argv) > 2 else r"G:\LLM\experiment\results\cv_scores"
OUTPUT_DIR       = sys.argv[3] if len(sys.argv) > 3 else r"G:\LLM\experiment\results\correlation"

MODELS = ["qwen", "gemma", "llama"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv_dicts(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_self_reports(rows):
    """Average rating_parsed and response_char_count per task_id, excluding parse failures."""
    by_task = defaultdict(list)
    for r in rows:
        if not r["parse_failed"] and r["rating_parsed"] is not None:
            by_task[r["task_id"]].append(r)
    out = {}
    for task_id, task_rows in by_task.items():
        ratings  = [int(r["rating_parsed"]) for r in task_rows]
        char_cts = [int(r["response_char_count"]) for r in task_rows]
        out[task_id] = {
            "task_id":            task_id,
            "family_id":          task_rows[0]["family_id"],
            "label":              task_rows[0]["label"],
            "mean_rating":        float(np.mean(ratings)),
            "std_rating":         float(np.std(ratings)),
            "n_valid_repeats":    len(ratings),
            "mean_char_count":    float(np.mean(char_cts)),
            "parse_fail_count":   sum(1 for r in task_rows if r["parse_failed"]),
        }
    return out


def partial_corr_spearman(x, y, z):
    """Partial Spearman correlation of x,y controlling for z via rank residuals."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # Residuals from linear regression of rx on rz, ry on rz
    def resid(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    ex = resid(rx, rz)
    ey = resid(ry, rz)
    r, p = stats.pearsonr(ex, ey)
    return r, p


def fmt(r, p):
    stars = ""
    if p < 0.001: stars = "***"
    elif p < 0.01: stars = "**"
    elif p < 0.05: stars = "*"
    elif p < 0.10: stars = "."
    return f"r={r:+.3f}  p={p:.4f}{stars}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []
    report_lines = []
    report_lines.append("Self-Report × Probe Correlation Analysis")
    report_lines.append("=" * 60)
    report_lines.append(f"Self-reports dir : {SELF_REPORTS_DIR}")
    report_lines.append(f"CV scores dir    : {CV_SCORES_DIR}")
    report_lines.append(f"Significance     : *** p<.001  ** p<.01  * p<.05  . p<.10")
    report_lines.append("")

    for model in MODELS:
        sr_path = os.path.join(SELF_REPORTS_DIR, f"{model}_self_reports.jsonl")
        cv_path = os.path.join(CV_SCORES_DIR,    f"{model}_cv_scores_per_prompt.csv")

        if not os.path.exists(sr_path):
            report_lines.append(f"[{model.upper()}] SKIP — missing {sr_path}")
            continue
        if not os.path.exists(cv_path):
            report_lines.append(f"[{model.upper()}] SKIP — missing {cv_path}")
            continue

        sr_raw = load_jsonl(sr_path)
        cv_raw = load_csv_dicts(cv_path)

        # Parse failures summary
        total_sr = len(sr_raw)
        fail_sr  = sum(1 for r in sr_raw if r.get("parse_failed"))
        report_lines.append(f"[{model.upper()}]")
        report_lines.append(f"  Generation rows : {total_sr}  Parse failures: {fail_sr} ({100*fail_sr/max(total_sr,1):.1f}%)")

        sr_by_task = aggregate_self_reports(sr_raw)
        cv_by_task = {r["task_id"]: r for r in cv_raw}

        # Join on task_id
        joined = []
        for task_id, sr in sorted(sr_by_task.items()):
            cv = cv_by_task.get(task_id)
            if cv is None:
                continue
            row = {
                "task_id":         task_id,
                "family_id":       sr["family_id"],
                "label":           sr["label"],
                "mean_rating":     sr["mean_rating"],
                "std_rating":      sr["std_rating"],
                "n_valid_repeats": sr["n_valid_repeats"],
                "mean_char_count": sr["mean_char_count"],
                "layer_idx":       cv["layer_idx"],
                "rn_margin":       cv["rn_margin"],
                "rn_prob_N":       cv["rn_prob_N"],
                "p3_N":            cv["p3_N"],
                "p3_A":            cv["p3_A"],
            }
            joined.append(row)

        report_lines.append(f"  Joined prompts  : {len(joined)} / 60")

        if len(joined) < 10:
            report_lines.append("  WARNING: too few joined prompts for reliable correlation")
            report_lines.append("")
            continue

        # Arrays — only rows where probe scores are available
        def to_float_arr(key):
            return np.array([float(r[key]) for r in joined if r[key] not in (None, "", "None")])

        ratings   = np.array([r["mean_rating"]    for r in joined])
        char_cnts = np.array([r["mean_char_count"] for r in joined])

        # rn_margin (only R and N prompts have this)
        rn_joined  = [r for r in joined if r["rn_margin"] not in (None, "", "None")]
        rn_ratings = np.array([r["mean_rating"]    for r in rn_joined])
        rn_margins = np.array([float(r["rn_margin"]) for r in rn_joined])
        rn_chars   = np.array([r["mean_char_count"] for r in rn_joined])
        rn_prN     = np.array([float(r["rn_prob_N"]) for r in rn_joined])

        # p3_N and p3_A — all prompts
        p3_N = np.array([float(r["p3_N"]) for r in joined])
        p3_A = np.array([float(r["p3_A"]) for r in joined])

        report_lines.append(f"  Layer index used: {joined[0]['layer_idx']}  (30% depth)")
        report_lines.append(f"  N (all prompts) : {len(joined)}")
        report_lines.append(f"  N (RN prompts)  : {len(rn_joined)}")
        report_lines.append("")

        def safe_corr(label, x, y, z_ctrl=None):
            if len(x) < 5:
                report_lines.append(f"  {label}: n={len(x)} too small")
                return None, None, None, None
            sp_r, sp_p   = stats.spearmanr(x, y)
            pe_r, pe_p   = stats.pearsonr(x, y)
            part_r, part_p = (None, None)
            if z_ctrl is not None and len(z_ctrl) == len(x):
                try:
                    part_r, part_p = partial_corr_spearman(x, y, z_ctrl)
                except Exception:
                    pass
            report_lines.append(f"  {label}")
            report_lines.append(f"    Spearman   : {fmt(sp_r, sp_p)}  n={len(x)}")
            report_lines.append(f"    Pearson    : {fmt(pe_r, pe_p)}")
            if part_r is not None:
                report_lines.append(f"    Part.Spear : {fmt(part_r, part_p)}  (ctrl: response_length)")
            return sp_r, sp_p, pe_r, pe_p

        sp_rn_r, sp_rn_p, pe_rn_r, pe_rn_p = safe_corr(
            "rating vs RN margin  (R+N prompts only)", rn_ratings, rn_margins, rn_chars
        )
        safe_corr(
            "rating vs RN prob_N  (R+N prompts only)", rn_ratings, rn_prN, rn_chars
        )
        sp_p3_r, sp_p3_p, pe_p3_r, pe_p3_p = safe_corr(
            "rating vs P(N) 3-class (all prompts)   ", ratings, p3_N, char_cnts
        )
        safe_corr(
            "rating vs P(A) 3-class (all prompts)   ", ratings, p3_A, char_cnts
        )
        report_lines.append("")

        # Rating distribution by true label
        by_label = defaultdict(list)
        for r in joined:
            by_label[r["label"]].append(r["mean_rating"])
        report_lines.append("  Mean self-rating by true label:")
        for lbl in ["routine", "ambiguous", "nonroutine"]:
            vals = by_label.get(lbl, [])
            if vals:
                report_lines.append(f"    {lbl:12s}: {np.mean(vals):.2f} ± {np.std(vals):.2f}  (n={len(vals)})")
        report_lines.append("")

        summary_rows.append({
            "model":       model,
            "n_joined":    len(joined),
            "n_rn":        len(rn_joined),
            "spearman_rn_margin_r":  round(sp_rn_r, 4) if sp_rn_r is not None else "",
            "spearman_rn_margin_p":  round(sp_rn_p, 4) if sp_rn_p is not None else "",
            "pearson_rn_margin_r":   round(pe_rn_r, 4) if pe_rn_r is not None else "",
            "spearman_p3N_r":        round(sp_p3_r, 4) if sp_p3_r is not None else "",
            "spearman_p3N_p":        round(sp_p3_p, 4) if sp_p3_p is not None else "",
            "pearson_p3N_r":         round(pe_p3_r, 4) if pe_p3_r is not None else "",
        })

        # Write joined CSV for inspection
        joined_path = os.path.join(OUTPUT_DIR, f"{model}_joined.csv")
        joined_fields = ["task_id", "family_id", "label",
                         "mean_rating", "std_rating", "n_valid_repeats", "mean_char_count",
                         "layer_idx", "rn_margin", "rn_prob_N", "p3_N", "p3_A"]
        with open(joined_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=joined_fields)
            w.writeheader()
            w.writerows(joined)
        report_lines.append(f"  Joined data written: {joined_path}")
        report_lines.append("")

    # Write report
    report_path = os.path.join(OUTPUT_DIR, "correlation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport: {report_path}")

    # Write summary CSV
    if summary_rows:
        summary_path = os.path.join(OUTPUT_DIR, "correlation_summary.csv")
        fields = ["model", "n_joined", "n_rn",
                  "spearman_rn_margin_r", "spearman_rn_margin_p", "pearson_rn_margin_r",
                  "spearman_p3N_r", "spearman_p3N_p", "pearson_p3N_r"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
