import os, sys, csv, json
import numpy as np
from scipy import stats
from collections import defaultdict

# Usage:
#   python 21_deep_analyses.py <CORR_DIR_5PT> <CORR_DIR_7PT> <TASKS_JSON> <OUTPUT_DIR>
#
# Reads {model}_joined.csv from each correlation dir plus tasks_v2_hard.json for prompt text.
# Runs three analyses and writes:
#   OUTPUT_DIR/results_deep_analyses.md
#   OUTPUT_DIR/dissociation_flagged.csv
#   OUTPUT_DIR/pn_distribution.csv
#   OUTPUT_DIR/cross_model_agreement.csv

CORR_5PT  = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\results\correlation"
CORR_7PT  = sys.argv[2] if len(sys.argv) > 2 else r"G:\LLM\experiment\results\correlation_7pt"
TASKS_JSON= sys.argv[3] if len(sys.argv) > 3 else r"G:\LLM\experiment\data\tasks_v2_hard.json"
OUTPUT_DIR= sys.argv[4] if len(sys.argv) > 4 else r"G:\LLM\experiment\results\correlation"

MODELS = ["qwen", "gemma", "llama"]

# Thresholds
DISS_HIGH_PN  = 0.70   # probe says N
DISS_LOW_RAT  = 3      # but self-report says routine
DISS_LOW_PN   = 0.30   # probe says R
DISS_HIGH_RAT = 4      # but self-report says nonroutine
NEAR_HIGH_PN  = (0.60, 0.70)
NEAR_LOW_PN   = (0.30, 0.40)

PN_LOW_CUTOFF = 0.33
PN_HIGH_CUTOFF = 0.67


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flt(v):
    try:
        return float(v) if v not in (None, "", "None") else None
    except:
        return None


def load_joined(corr_dir, model):
    path = os.path.join(corr_dir, f"{model}_joined.csv")
    if not os.path.exists(path):
        return {}
    rows = load_csv(path)
    return {r["task_id"]: r for r in rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load task prompts
    with open(TASKS_JSON, encoding="utf-8") as f:
        tasks = json.load(f)
    prompt_map = {t["task_id"]: t["task_prompt"] for t in tasks}

    # Load all joined data
    data5 = {m: load_joined(CORR_5PT, m) for m in MODELS}
    data7 = {m: load_joined(CORR_7PT, m) for m in MODELS}

    # All task_ids from first model
    all_task_ids = sorted(data5[MODELS[0]].keys())

    lines = []
    lines.append("# Deep Analyses: Dissociation, Distribution, Cross-Model Agreement")
    lines.append("")
    lines.append("Source data: `results/correlation/{model}_joined.csv` (5pt primary), "
                 "`results/correlation_7pt/` (7pt sensitivity).")
    lines.append("No new generation or extraction — joins and statistics only.")
    lines.append("")

    # =========================================================
    # ANALYSIS A: Introspective Dissociation Scan
    # =========================================================
    lines.append("---")
    lines.append("")
    lines.append("## Analysis A: Introspective Dissociation Scan")
    lines.append("")
    lines.append("Prompts where probe confidence and self-report diverge:")
    lines.append(f"- **Blind nonroutine**: P(N) ≥ {DISS_HIGH_PN} but mean_rating ≤ {DISS_LOW_RAT}")
    lines.append(f"- **False nonroutine**: P(N) ≤ {DISS_LOW_PN} but mean_rating ≥ {DISS_HIGH_RAT}")
    lines.append(f"- **Near-threshold** (secondary): P(N) ∈ [{NEAR_HIGH_PN[0]}, {NEAR_HIGH_PN[1]}) "
                 f"with rating ≤ {DISS_LOW_RAT}, or P(N) ∈ ({NEAR_LOW_PN[0]}, {NEAR_LOW_PN[1]}] "
                 f"with rating ≥ {DISS_HIGH_RAT}")
    lines.append("")

    # Collect flagged prompts
    flagged = {}  # task_id -> {model -> row, type}
    near_flagged = {}

    for task_id in all_task_ids:
        task_flags = {}
        task_near = {}
        for m in MODELS:
            r5 = data5[m].get(task_id)
            if not r5:
                continue
            pn = flt(r5.get("p3_N"))
            rat = flt(r5.get("mean_rating"))
            if pn is None or rat is None:
                continue
            if pn >= DISS_HIGH_PN and rat <= DISS_LOW_RAT:
                task_flags[m] = "blind_nonroutine"
            elif pn <= DISS_LOW_PN and rat >= DISS_HIGH_RAT:
                task_flags[m] = "false_nonroutine"
            # Near-threshold
            if NEAR_HIGH_PN[0] <= pn < NEAR_HIGH_PN[1] and rat <= DISS_LOW_RAT:
                task_near[m] = "near_blind"
            elif NEAR_LOW_PN[0] < pn <= NEAR_LOW_PN[1] and rat >= DISS_HIGH_RAT:
                task_near[m] = "near_false"
        if task_flags:
            flagged[task_id] = task_flags
        if task_near:
            near_flagged[task_id] = task_near

    # Count by model
    lines.append("### A.1 Counts per model")
    lines.append("")
    lines.append("| Model | Blind nonroutine | False nonroutine | Total flagged |")
    lines.append("|-------|-----------------|-----------------|---------------|")
    for m in MODELS:
        blind = sum(1 for v in flagged.values() if v.get(m) == "blind_nonroutine")
        false_ = sum(1 for v in flagged.values() if v.get(m) == "false_nonroutine")
        lines.append(f"| {m:5s} | {blind:16d} | {false_:15d} | {blind+false_:13d} |")
    lines.append("")

    # Cross-model consistency
    cross = [t for t, v in flagged.items() if len(v) >= 2]
    all_3 = [t for t, v in flagged.items() if len(v) == 3]
    lines.append(f"Flagged in ≥2 models: **{len(cross)}** prompt(s).  "
                 f"Flagged in all 3: **{len(all_3)}**.")
    lines.append("")

    # Flagged prompt table
    lines.append("### A.2 Flagged prompts (threshold P(N) ≥0.70 / ≤0.30)")
    lines.append("")
    if flagged:
        hdr = ("| task_id | label | prompt (truncated) | "
               "| Qwen P(N) | Qwen RN_margin | Qwen rat_5pt | Qwen rat_7pt "
               "| Gemma P(N) | Gemma RN_margin | Gemma rat_5pt | Gemma rat_7pt "
               "| LLaMA P(N) | LLaMA RN_margin | LLaMA rat_5pt | LLaMA rat_7pt "
               "| models flagged |")
        # Simplified table
        lines.append("| task_id | label | prompt | flag_type | "
                     "Qwen P(N) | Qwen RNmgn | Qwen r5 | Qwen r7 | "
                     "Gemma P(N) | Gemma RNmgn | Gemma r5 | Gemma r7 | "
                     "LLaMA P(N) | LLaMA RNmgn | LLaMA r5 | LLaMA r7 | n_models |")
        lines.append("|---------|-------|--------|-----------|"
                     "----------|-----------|---------|---------|"
                     "-----------|------------|---------|---------|"
                     "-----------|------------|---------|---------|---------|")
        for task_id in sorted(flagged.keys()):
            model_flags = flagged[task_id]
            prompt_text = prompt_map.get(task_id, "")[:60].replace("|", "/")
            # Get label from any available model
            label = ""
            for m in MODELS:
                r = data5[m].get(task_id)
                if r:
                    label = r.get("label", "")
                    break
            # Collect flag type (use first model's type, or "mixed" if different)
            flag_types = list(set(model_flags.values()))
            flag_type = flag_types[0] if len(flag_types) == 1 else "mixed"
            n_models = len(model_flags)
            cols = [task_id, label, prompt_text, flag_type]
            for m in MODELS:
                r5 = data5[m].get(task_id, {})
                r7 = data7[m].get(task_id, {})
                pn   = f"{flt(r5.get('p3_N')):.3f}"   if flt(r5.get('p3_N'))  is not None else "—"
                rnm  = f"{flt(r5.get('rn_margin')):.3f}" if flt(r5.get('rn_margin')) is not None else "—"
                r5v  = f"{flt(r5.get('mean_rating')):.2f}" if flt(r5.get('mean_rating')) is not None else "—"
                r7v  = f"{flt(r7.get('mean_rating')):.2f}" if flt(r7.get('mean_rating')) is not None else "—"
                cols += [pn, rnm, r5v, r7v]
            cols.append(str(n_models))
            lines.append("| " + " | ".join(cols) + " |")
    else:
        lines.append("_No prompts flagged at primary threshold._")
    lines.append("")

    # Near-threshold secondary table
    near_counts = {m: {"near_blind": 0, "near_false": 0} for m in MODELS}
    for v in near_flagged.values():
        for m, t in v.items():
            near_counts[m][t] += 1
    lines.append("### A.3 Near-threshold counts (secondary, P(N) ∈ [0.60–0.70] or [0.30–0.40])")
    lines.append("")
    lines.append("| Model | Near-blind (P(N)∈[0.60–0.70], rat≤3) | Near-false (P(N)∈[0.30–0.40], rat≥4) |")
    lines.append("|-------|---------------------------------------|---------------------------------------|")
    for m in MODELS:
        lines.append(f"| {m:5s} | {near_counts[m]['near_blind']:37d} | {near_counts[m]['near_false']:37d} |")
    lines.append("")

    # =========================================================
    # ANALYSIS B: Processing-mode distribution of P(N)
    # =========================================================
    lines.append("---")
    lines.append("")
    lines.append("## Analysis B: Processing-Mode Distribution of P(N)")
    lines.append("")
    lines.append("Per model, distribution of 3-class P(N) grouped by true label.")
    lines.append("Cutoffs: low P(N) < 0.33 (R-like), middle 0.33–0.67, high P(N) > 0.67 (N-like).")
    lines.append("")

    pn_dist_rows = []

    for m in MODELS:
        lines.append(f"### {m.capitalize()}")
        lines.append("")
        lines.append("| Label | n | mean P(N) | std | min | max | frac_low | frac_mid | frac_high |")
        lines.append("|-------|---|-----------|-----|-----|-----|----------|----------|-----------|")
        # Collect P(N) and RN margin by label for AUROC
        pn_by_label = defaultdict(list)
        rn_by_label = defaultdict(list)
        for task_id, r in data5[m].items():
            lbl = r.get("label", "")
            pn = flt(r.get("p3_N"))
            rn = flt(r.get("rn_margin"))
            if pn is not None:
                pn_by_label[lbl].append(pn)
            if rn is not None:
                rn_by_label[lbl].append(rn)

        for lbl in ["routine", "ambiguous", "nonroutine"]:
            vals = pn_by_label.get(lbl, [])
            if not vals:
                continue
            n = len(vals)
            mean_v = np.mean(vals)
            std_v  = np.std(vals)
            min_v  = np.min(vals)
            max_v  = np.max(vals)
            frac_low  = sum(1 for v in vals if v < PN_LOW_CUTOFF) / n
            frac_mid  = sum(1 for v in vals if PN_LOW_CUTOFF <= v <= PN_HIGH_CUTOFF) / n
            frac_high = sum(1 for v in vals if v > PN_HIGH_CUTOFF) / n
            lines.append(f"| {lbl:11s} | {n:2d} | {mean_v:.3f} | {std_v:.3f} | "
                         f"{min_v:.3f} | {max_v:.3f} | {frac_low:.2f} | {frac_mid:.2f} | {frac_high:.2f} |")
            pn_dist_rows.append({
                "model": m, "label": lbl, "n": n,
                "mean_pN": round(mean_v, 4), "std_pN": round(std_v, 4),
                "min_pN": round(min_v, 4), "max_pN": round(max_v, 4),
                "frac_low": round(frac_low, 3), "frac_mid": round(frac_mid, 3),
                "frac_high": round(frac_high, 3),
            })

        # AUROC: P(N) distinguishing N vs R; where does A fall?
        n_vals = pn_by_label.get("nonroutine", [])
        r_vals = pn_by_label.get("routine", [])
        a_vals = pn_by_label.get("ambiguous", [])
        if n_vals and r_vals:
            y_true = [1]*len(n_vals) + [0]*len(r_vals)
            y_score = n_vals + r_vals
            # AUROC via Mann-Whitney
            u_stat, _ = stats.mannwhitneyu(n_vals, r_vals, alternative="greater")
            auroc = u_stat / (len(n_vals) * len(r_vals))
            lines.append(f"")
            lines.append(f"AUROC P(N) for N vs R: **{auroc:.3f}**")
            if a_vals:
                a_pct_in_n_range = sum(1 for v in a_vals if v > PN_HIGH_CUTOFF) / len(a_vals)
                a_pct_in_r_range = sum(1 for v in a_vals if v < PN_LOW_CUTOFF) / len(a_vals)
                a_pct_mid        = sum(1 for v in a_vals if PN_LOW_CUTOFF <= v <= PN_HIGH_CUTOFF) / len(a_vals)
                lines.append(f"Ambiguous prompts: {a_pct_in_r_range:.0%} in R-range, "
                             f"{a_pct_mid:.0%} middle, {a_pct_in_n_range:.0%} in N-range — "
                             f"{'intermediate' if a_pct_mid >= 0.40 else 'polarised toward poles'}")
        lines.append("")

    # =========================================================
    # ANALYSIS C: Cross-model prompt agreement
    # =========================================================
    lines.append("---")
    lines.append("")
    lines.append("## Analysis C: Cross-Model Prompt Agreement")
    lines.append("")
    lines.append("Spearman correlations across the 60 prompts between model pairs.")
    lines.append("Tests whether the processing-mode signal reflects task structure "
                 "rather than model-specific quirks.")
    lines.append("")

    pairs = [("qwen", "gemma"), ("qwen", "llama"), ("gemma", "llama")]
    measures = [
        ("p3_N",       "P(N) 3-class",   data5),
        ("rn_margin",  "RN margin",       data5),
        ("mean_rating","Self-report (5pt)", data5),
        ("mean_rating","Self-report (7pt)", data7),
    ]

    lines.append("| Measure | Qwen–Gemma | Qwen–LLaMA | Gemma–LLaMA |")
    lines.append("|---------|-----------|-----------|------------|")

    agree_rows = []
    for measure_key, measure_label, data_src in measures:
        row_vals = []
        for m1, m2 in pairs:
            shared = [t for t in all_task_ids
                      if flt(data_src[m1].get(t, {}).get(measure_key)) is not None
                      and flt(data_src[m2].get(t, {}).get(measure_key)) is not None]
            if len(shared) < 5:
                row_vals.append("n/a")
                continue
            v1 = [flt(data_src[m1][t][measure_key]) for t in shared]
            v2 = [flt(data_src[m2][t][measure_key]) for t in shared]
            r, p = stats.spearmanr(v1, v2)
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            row_vals.append(f"r={r:+.3f}{stars} (n={len(shared)})")
            agree_rows.append({
                "measure": measure_label,
                "pair": f"{m1}-{m2}",
                "n": len(shared),
                "spearman_r": round(r, 4),
                "p": round(p, 4),
            })
        lines.append(f"| {measure_label:25s} | " + " | ".join(row_vals) + " |")
    lines.append("")
    lines.append("Significance: *** p<.001  ** p<.01  * p<.05")
    lines.append("")

    # =========================================================
    # Write CSVs
    # =========================================================
    deep_dir = os.path.join(OUTPUT_DIR, "deep_analyses")
    os.makedirs(deep_dir, exist_ok=True)

    # Dissociation CSV
    diss_rows = []
    for task_id in sorted(flagged.keys()):
        label = ""
        for m in MODELS:
            r = data5[m].get(task_id)
            if r:
                label = r.get("label", "")
                break
        model_flags = flagged[task_id]
        base = {
            "task_id": task_id,
            "label": label,
            "prompt": prompt_map.get(task_id, ""),
            "n_models_flagged": len(model_flags),
        }
        for m in MODELS:
            r5 = data5[m].get(task_id, {})
            r7 = data7[m].get(task_id, {})
            base[f"{m}_flag"]       = model_flags.get(m, "")
            base[f"{m}_p3N"]        = flt(r5.get("p3_N"))
            base[f"{m}_rn_margin"]  = flt(r5.get("rn_margin"))
            base[f"{m}_rating_5pt"] = flt(r5.get("mean_rating"))
            base[f"{m}_rating_7pt"] = flt(r7.get("mean_rating"))
            base[f"{m}_char_count"] = flt(r5.get("mean_char_count"))
        diss_rows.append(base)

    diss_path = os.path.join(deep_dir, "dissociation_flagged.csv")
    if diss_rows:
        with open(diss_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(diss_rows[0].keys()))
            w.writeheader()
            w.writerows(diss_rows)

    # P(N) distribution CSV
    pn_path = os.path.join(deep_dir, "pn_distribution.csv")
    if pn_dist_rows:
        with open(pn_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pn_dist_rows[0].keys()))
            w.writeheader()
            w.writerows(pn_dist_rows)

    # Cross-model agreement CSV
    agree_path = os.path.join(deep_dir, "cross_model_agreement.csv")
    if agree_rows:
        with open(agree_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agree_rows[0].keys()))
            w.writeheader()
            w.writerows(agree_rows)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Output files")
    lines.append(f"- `{diss_path}` — flagged dissociation prompts with per-model values")
    lines.append(f"- `{pn_path}` — P(N) distribution stats per model × label")
    lines.append(f"- `{agree_path}` — cross-model Spearman agreement rows")
    lines.append("")

    # Write markdown doc
    doc_path = os.path.join(os.path.dirname(OUTPUT_DIR.rstrip("/\\")),
                            "docs", "results_deep_analyses.md")
    # fallback: write to experiment/docs/ relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(script_dir, "..", "docs", "results_deep_analyses.md")
    doc_path = os.path.normpath(doc_path)

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {doc_path}")
    print(f"CSVs:  {deep_dir}/")


if __name__ == "__main__":
    main()
