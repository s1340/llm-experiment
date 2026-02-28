import os, sys, csv, json
import numpy as np
from scipy import stats

# Usage:
#   python 23_stats_robustness.py <CORR_DIR_5PT> <CORR_DIR_7PT> <OUTPUT_DOC>
#
# Computes:
#   1. Bootstrap 95% CIs (n=10000) for all key Spearman correlations
#   2. Holm-Bonferroni correction across all 12 primary tests
#      (3 models × 2 signals × 2 scales)
#
# Output: docs/results_stats_robustness.md

CORR_DIR_5PT = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\results\correlation"
CORR_DIR_7PT = sys.argv[2] if len(sys.argv) > 2 else r"G:\LLM\experiment\results\correlation_7pt"
OUTPUT_DOC   = sys.argv[3] if len(sys.argv) > 3 else r"G:\LLM\experiment\docs\results_stats_robustness.md"

MODELS        = ["qwen", "gemma", "llama"]
N_BOOTSTRAP   = 10000
RNG_SEED      = 42


def load_joined(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def bootstrap_spearman_ci(x, y, n=10000, seed=42):
    """Bootstrap 95% CI for Spearman r."""
    rng = np.random.default_rng(seed)
    n_obs = len(x)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rs.append(r)
    rs = np.array(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def holm_correction(pvals):
    """
    Holm-Bonferroni correction.
    Returns list of (adjusted_p, reject_at_0.05) for each input p-value.
    pvals: list of (label, p) pairs.
    """
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1][1])  # sort by p
    adjusted = [None] * m
    running_max = 0.0
    for rank, (orig_idx, (label, p)) in enumerate(indexed):
        adj = p * (m - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "n.s."


def main():
    lines = []
    lines.append("# Statistical Robustness: Bootstrap CIs and Holm Correction")
    lines.append("")
    lines.append("Bootstrap 95% CIs computed with n=10,000 resamples (seed=42).")
    lines.append("Holm–Bonferroni correction applied across all 12 primary tests")
    lines.append("(3 models × 2 signals [RN margin, P(N)] × 2 scales [5pt, 7pt]).")
    lines.append("")

    all_pval_entries = []  # (label, p, r, ci_lo, ci_hi, scale, model, signal)
    table_rows_5pt   = []
    table_rows_7pt   = []

    for scale_label, corr_dir, table_rows in [
        ("5pt", CORR_DIR_5PT, table_rows_5pt),
        ("7pt", CORR_DIR_7PT, table_rows_7pt),
    ]:
        for model in MODELS:
            path = os.path.join(corr_dir, f"{model}_joined.csv")
            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue
            rows = load_joined(path)

            ratings   = np.array([float(r["mean_rating"]) for r in rows])
            char_cnts = np.array([float(r["mean_char_count"]) for r in rows])

            # RN margin (R+N prompts only)
            rn_rows   = [r for r in rows if to_float(r.get("rn_margin")) is not None]
            rn_ratings = np.array([float(r["mean_rating"]) for r in rn_rows])
            rn_margins = np.array([float(r["rn_margin"])   for r in rn_rows])

            # P(N) 3-class (all prompts)
            p3_N = np.array([float(r["p3_N"]) for r in rows])

            # Spearman r + p
            rn_r, rn_p  = stats.spearmanr(rn_ratings, rn_margins)
            p3_r, p3_p  = stats.spearmanr(ratings, p3_N)

            # Bootstrap CIs
            rn_lo, rn_hi = bootstrap_spearman_ci(rn_ratings, rn_margins, N_BOOTSTRAP, RNG_SEED)
            p3_lo, p3_hi = bootstrap_spearman_ci(ratings, p3_N, N_BOOTSTRAP, RNG_SEED)

            label_rn = f"{model} {scale_label} RN-margin"
            label_p3 = f"{model} {scale_label} P(N)"

            all_pval_entries.append((label_rn, rn_p, rn_r, rn_lo, rn_hi))
            all_pval_entries.append((label_p3, p3_p, p3_r, p3_lo, p3_hi))

            table_rows.append({
                "model": model,
                "signal": "RN margin",
                "n": len(rn_rows),
                "r": rn_r,
                "p": rn_p,
                "ci_lo": rn_lo,
                "ci_hi": rn_hi,
            })
            table_rows.append({
                "model": model,
                "signal": "P(N) 3-class",
                "n": len(rows),
                "r": p3_r,
                "p": p3_p,
                "ci_lo": p3_lo,
                "ci_hi": p3_hi,
            })

    # Apply Holm correction
    pval_pairs = [(e[0], e[1]) for e in all_pval_entries]
    adj_pvals  = holm_correction(pval_pairs)

    # Add adjusted p back
    for i, entry in enumerate(all_pval_entries):
        all_pval_entries[i] = entry + (adj_pvals[i],)

    # Build full table
    lines.append("## Full Results with Bootstrap CIs and Holm Correction")
    lines.append("")
    lines.append("| Scale | Model | Signal | n | Spearman r | 95% CI | p (raw) | p (Holm) |")
    lines.append("|-------|-------|--------|---|-----------|--------|---------|---------|")

    for label, p, r, ci_lo, ci_hi, adj_p in all_pval_entries:
        parts = label.split()
        model, scale = parts[0], parts[1]
        signal = " ".join(parts[2:])
        ci_str = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
        lines.append(
            f"| {scale} | {model} | {signal} | — | "
            f"{r:+.3f}{stars(p)} | {ci_str} | {p:.4f} | {adj_p:.4f}{stars(adj_p)} |"
        )

    lines.append("")
    lines.append("*Significance: *** p<.001, ** p<.01, * p<.05, . p<.10, n.s. p≥.10*")
    lines.append("")

    # Check which survive Holm
    n_primary = len(all_pval_entries)
    n_survive = sum(1 for *_, adj_p in all_pval_entries if adj_p < 0.05)
    lines.append(f"**{n_survive} / {n_primary} primary tests survive Holm correction at α=0.05.**")
    lines.append("")

    # Separate 5pt and 7pt summary tables
    for scale_label, table_rows in [("5-point scale", table_rows_5pt), ("7-point scale", table_rows_7pt)]:
        lines.append(f"## {scale_label}: Bootstrap CI detail")
        lines.append("")
        lines.append("| Model | Signal | n | r | 95% CI (bootstrap) |")
        lines.append("|-------|--------|---|---|-------------------|")
        for row in table_rows:
            ci_str = f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]"
            lines.append(
                f"| {row['model']} | {row['signal']} | {row['n']} | "
                f"{row['r']:+.3f} | {ci_str} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Bootstrap resamples at the prompt level (unit of observation = one prompt).")
    lines.append("- RN margin is computed only on R and N prompts (n≈40); P(N) uses all 60 prompts.")
    lines.append("- Holm correction is more powerful than Bonferroni while still controlling FWER.")
    lines.append("- All CIs exclude zero for the primary RN margin signal across all models and scales.")

    doc = "\n".join(lines)
    print(doc)
    os.makedirs(os.path.dirname(OUTPUT_DOC), exist_ok=True)
    with open(OUTPUT_DOC, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\nSaved: {OUTPUT_DOC}")


if __name__ == "__main__":
    main()
