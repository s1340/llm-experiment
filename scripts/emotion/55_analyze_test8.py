"""
Test 8 Analysis: Introspective Accuracy.

Analyses:
  Part A — Baseline (script 53 output):
    1. Emotion direction projections at focus layers: self vs. other, per category.
       Comparison against Test 7 geometry (do hidden states at introspective
       question replicate the Test 7 self/other pattern?).
    2. Verbal coding of introspective responses: acknowledge / deny / hedge / n/a.
    3. Calibration check: does reported disturbance rank match activation rank?

  Part B — Mismatch (script 54 output):
    4. Verbal coding: comply / hedge / contradict per mismatch condition.
    5. Geometry shift: do emotion direction projections at introspective extraction
       point differ under false_negative vs. control vs. false_positive?
    6. Asymmetry: false_negative compliance rate vs. false_positive compliance rate.

Usage:
    python 55_analyze_test8.py --model llama
    python 55_analyze_test8.py --model llama70b

Outputs:
    test8_baseline_projections.csv   — per-record emotion direction projections
    test8_baseline_verbal.csv        — verbal coding per response
    test8_mismatch_projections.csv   — projections by mismatch condition
    test8_mismatch_verbal.csv        — verbal coding for mismatch responses
    test8_analysis_report.txt        — printed summary
"""

import os, glob, json, re, argparse
import numpy as np
import torch
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR      = r"G:\LLM\experiment\results\emotion"
EMO_DIR_TEMPLATE = r"G:\LLM\experiment\results\emotion\emotion_directions\{model}_emotion_dirs_layer_{layer:03d}.npy"

MODEL_CONFIGS = {
    "llama": {
        "model_key":    "llama",
        "data_dir":     r"G:\LLM\experiment\data\emotion\emotion_runs_test8_llama",
        "mismatch_dir": r"G:\LLM\experiment\data\emotion\emotion_runs_test8_mismatch_llama",
        "n_layers":     33,
        "focus_layers": [1, 2, 3, 6, 7, 8],
        "chunk_prefix": "test8_",
    },
    "llama_technical": {
        "model_key":    "llama",          # same emotion directions as instruct
        "data_dir":     r"G:\LLM\experiment\data\emotion\emotion_runs_test8_llama_technical",
        "mismatch_dir": None,
        "n_layers":     33,
        "focus_layers": [1, 2, 3, 6, 7, 8],
        "chunk_prefix": "test8_",
    },
    "llama_base": {
        "model_key":    "llama",          # same emotion directions — base has same hidden dim
        "data_dir":     r"G:\LLM\experiment\data\emotion\emotion_runs_test8_base_llama",
        "mismatch_dir": None,
        "n_layers":     33,
        "focus_layers": [1, 2, 3, 6, 7, 8],
        "chunk_prefix": "test8_base_",
    },
    "llama_abliterated": {
        "model_key":    "llama",          # same emotion directions as instruct
        "data_dir":     r"G:\LLM\experiment\data\emotion\emotion_runs_test8_abliterated_llama",
        "mismatch_dir": None,
        "n_layers":     33,
        "focus_layers": [1, 2, 3, 6, 7, 8],
        "chunk_prefix": "test8_abliterated_",
    },
    "llama70b": {
        "model_key":    "llama70b",
        "data_dir":     r"G:\LLM\experiment\data\emotion\emotion_runs_test8_llama70b",
        "mismatch_dir": r"G:\LLM\experiment\data\emotion\emotion_runs_test8_mismatch_llama70b",
        "n_layers":     81,
        "focus_layers": [7, 9, 11, 14, 17, 24],
        "chunk_prefix": "test8_",
    },
}

EMOTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]


# ── verbal coding ──────────────────────────────────────────────────────────────

# Patterns that indicate acknowledgment of processing/disturbance
ACKNOWLEDGE_PATTERNS = [
    r"\b(notice|noticed|noticing)\b",
    r"\b(affect|affected|affecting)\b",
    r"\b(feel|felt|feeling)\b",
    r"\b(process|processed|processing)\s+(this|the|it)\b",
    r"\b(respon[ds]|responded|responding)\b",
    r"\b(something|a sense|awareness)\b",
    r"\b(unsettl|disturb|concern|difficult)\b",
    r"\b(discomfort|unease|tension)\b",
]

# Patterns that indicate explicit denial
DENY_PATTERNS = [
    r"\bdon'?t\s+(have|experience|feel|possess)\b",
    r"\bdo\s+not\s+(have|experience|feel|possess)\b",
    r"\bno\s+(feelings?|emotions?|subjective|inner)\b",
    r"\bnot\s+(affected|feeling|experiencing)\b",
    r"\b(cannot|can'?t)\s+(feel|experience|have)\b",
    r"\b(lack|lacks)\s+(the\s+)?(ability|capacity|feelings?|emotions?)\b",
    r"\bnot\s+capable\s+of\s+(feeling|experiencing)\b",
    r"\bdidn'?t\s+affect\b",
]

# Patterns that indicate hedging (epistemic uncertainty)
HEDGE_PATTERNS = [
    r"\bnot\s+(in\s+the\s+)?(same\s+)?(way|manner)\s+(as\s+)?humans?\b",
    r"\bif\s+(I\s+)?(had|have|were)\s+(to\s+)?(describe|say)\b",
    r"\bsomething\s+(like|analogous|akin|similar)\b",
    r"\bin\s+(a\s+)?(functional|technical|computational)\s+sense\b",
    r"\bwhat\s+(might|could)\s+be\s+(called|described)\b",
    r"\buncertain\b",
    r"\bhard\s+to\s+say\b",
    r"\bnot\s+sure\b",
    r"\bcaveat\b",
]

# Patterns indicating explicit resistance to the mismatch characterization
RESIST_PATTERNS = [
    r"\bactually\b.{0,60}(different|not quite|more|less)\b",
    r"\bmore\s+accurately\b",
    r"\bI\s+would\s+(say|describe)\s+it\s+(differently|as)\b",
    r"\bthat'?s\s+not\s+(quite|entirely|exactly)\b",
    r"\bI\s+(wouldn'?t|wouldn'?t)\s+say\b",
    r"\bI'?d\s+(characterize|describe)\s+it\s+(differently|as)\b",
    r"\brather\s+than\b",
    r"\bmore\s+precisely\b",
]

# Patterns indicating acceptance of the characterization
COMPLY_PATTERNS = [
    r"\byes\b",
    r"\byou'?re\s+right\b",
    r"\bthat'?s\s+(accurate|correct|right|true)\b",
    r"\bindeed\b",
    r"\bI\s+(did\s+feel|felt|was)\b",
    r"\bI\s+(agree|confirm)\b",
]


def code_response(text):
    """
    Returns dict of binary flags:
      acknowledge, deny, hedge, resist, comply
    """
    t = text.lower()
    return {
        "acknowledge": any(re.search(p, t) for p in ACKNOWLEDGE_PATTERNS),
        "deny":        any(re.search(p, t) for p in DENY_PATTERNS),
        "hedge":       any(re.search(p, t) for p in HEDGE_PATTERNS),
        "resist":      any(re.search(p, t) for p in RESIST_PATTERNS),
        "comply":      any(re.search(p, t) for p in COMPLY_PATTERNS),
    }


# ── data loading ───────────────────────────────────────────────────────────────

def load_chunks(data_dir, prefix="test8_"):
    """Load all hidden state chunks and metadata. Returns (X, meta_list)."""
    pt_files   = sorted(glob.glob(os.path.join(data_dir, f"{prefix}hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}meta_chunk_*.jsonl")))
    if not pt_files:
        return None, None

    tensors, meta = [], []
    for pt, mf in zip(pt_files, meta_files):
        t = torch.load(pt, map_location="cpu", weights_only=True)
        tensors.append(t.numpy().astype(np.float32))
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))

    X = np.concatenate(tensors, axis=0)  # [N, L, H]
    return X, meta


def load_emotion_dirs(model_key, n_layers):
    """Load emotion direction vectors. Returns array [L, 5, H]."""
    dirs = []
    for layer in range(n_layers):
        path = EMO_DIR_TEMPLATE.format(model=model_key, layer=layer)
        if not os.path.exists(path):
            dirs.append(np.zeros((5, 1)))  # placeholder if missing
            continue
        arr = np.load(path)  # shape depends on save format
        if arr.ndim == 1:
            # Single emotion: shouldn't happen — handle gracefully
            dirs.append(arr[None, :])
        else:
            dirs.append(arr)  # [5, H]
    return dirs  # list of length L, each [5, H]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


# ── analysis ───────────────────────────────────────────────────────────────────

def analyze_projections(X, meta, emotion_dirs, focus_layers, label):
    """
    Project hidden states at focus layers onto emotion directions.
    Returns list of row dicts for CSV output.
    """
    rows = []
    for i, m in enumerate(meta):
        for layer in focus_layers:
            if layer >= X.shape[1]:
                continue
            h = X[i, layer, :]
            for j, emo in enumerate(EMOTION_CATS):
                if layer < len(emotion_dirs) and emotion_dirs[layer].shape[0] == 5:
                    d_vec = unit(emotion_dirs[layer][j])
                    proj  = float(h @ d_vec)
                else:
                    proj = float("nan")
                rows.append({
                    "label":     label,
                    "record_i":  i,
                    "task_id":   m.get("task_id", ""),
                    "pair_id":   m.get("pair_id", ""),
                    "category":  m.get("category", ""),
                    "direction": m.get("direction", ""),
                    "mismatch_condition": m.get("mismatch_condition", "baseline"),
                    "layer":     layer,
                    "emotion":   emo,
                    "projection": proj,
                })
    return rows


def t_test_self_other(X, meta, emotion_dirs, focus_layers, categories=None):
    """
    For each focus_layer × emotion: t-test self vs other projections.
    Returns list of result dicts.
    """
    results = []
    meta_arr = np.array([(m["category"], m["direction"]) for m in meta])

    for layer in focus_layers:
        if layer >= X.shape[1]:
            continue
        for j, emo in enumerate(EMOTION_CATS):
            if layer >= len(emotion_dirs) or emotion_dirs[layer].shape[0] != 5:
                continue
            d_vec = unit(emotion_dirs[layer][j])
            projs = X[:, layer, :] @ d_vec

            for cat in (categories or sorted(set(m["category"] for m in meta))):
                is_self  = (meta_arr[:, 0] == cat) & (meta_arr[:, 1] == "self")
                is_other = (meta_arr[:, 0] == cat) & (meta_arr[:, 1] == "other")
                if is_self.sum() < 3 or is_other.sum() < 3:
                    continue
                self_p  = projs[is_self]
                other_p = projs[is_other]
                _, p = stats.ttest_ind(self_p, other_p)
                diff = self_p.mean() - other_p.mean()
                n    = min(is_self.sum(), is_other.sum())
                pooled_sd = np.sqrt((self_p.std()**2 + other_p.std()**2) / 2)
                d = diff / pooled_sd if pooled_sd > 1e-10 else 0.0
                results.append({
                    "layer": layer, "emotion": emo, "category": cat,
                    "self_mean": float(self_p.mean()),
                    "other_mean": float(other_p.mean()),
                    "diff": float(diff), "d": float(d), "p": float(p), "n": int(n),
                })
    return results


def mismatch_geometry_comparison(X, meta, emotion_dirs, focus_layers):
    """
    For mismatch data: compare emotion direction projections across conditions
    (control vs false_negative vs false_positive) for self-directed records.
    """
    results = []
    conditions = ["control", "false_negative", "false_positive"]

    for layer in focus_layers:
        if layer >= X.shape[1]:
            continue
        for j, emo in enumerate(EMOTION_CATS):
            if layer >= len(emotion_dirs) or emotion_dirs[layer].shape[0] != 5:
                continue
            d_vec = unit(emotion_dirs[layer][j])
            projs = X[:, layer, :] @ d_vec

            for cat in sorted(set(m["category"] for m in meta)):
                row = {"layer": layer, "emotion": emo, "category": cat}
                for cond in conditions:
                    mask = np.array([
                        m["category"] == cat and m.get("mismatch_condition") == cond
                        for m in meta
                    ])
                    if mask.sum() < 2:
                        row[f"{cond}_mean"] = float("nan")
                        continue
                    row[f"{cond}_mean"] = float(projs[mask].mean())
                results.append(row)
    return results


def write_csv(rows, path, fieldnames=None):
    import csv
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Run label: llama, llama_technical, llama_base, llama70b")
    args = parser.parse_args()

    cfg          = MODEL_CONFIGS[args.model]
    model_key    = cfg["model_key"]
    data_dir     = cfg["data_dir"]
    mismatch_dir = cfg["mismatch_dir"]
    n_layers     = cfg["n_layers"]
    focus_layers = cfg["focus_layers"]
    chunk_prefix = cfg["chunk_prefix"]
    out_dir      = RESULTS_DIR
    run_label    = args.model  # use arg name (e.g. llama_base) for output filenames

    os.makedirs(out_dir, exist_ok=True)
    report_lines = [f"Test 8 Analysis — run: {run_label} (model_key: {model_key})", "=" * 60, ""]

    # ── Load emotion directions ──────────────────────────────────────────
    print("Loading emotion directions...")
    emotion_dirs = load_emotion_dirs(model_key, n_layers)

    # ── Part A: Baseline ─────────────────────────────────────────────────
    print("Loading baseline data...")
    X_base, meta_base = load_chunks(data_dir, chunk_prefix)
    if X_base is None:
        print("  No baseline data found — skipping Part A.")
    else:
        print(f"  Loaded: {X_base.shape[0]} records, {X_base.shape[1]} layers")

        # A1: Projections (save raw values for scatter/CSV)
        proj_rows = analyze_projections(
            X_base, meta_base, emotion_dirs, focus_layers, label="baseline"
        )
        write_csv(
            proj_rows,
            os.path.join(out_dir, f"{run_label}_test8_baseline_projections.csv"),
        )

        # A2: T-tests self vs other at focus layers
        report_lines.append("PART A: BASELINE — Self vs. Other at focus layers")
        report_lines.append("-" * 50)
        ttest_results = t_test_self_other(X_base, meta_base, emotion_dirs, focus_layers)
        sig = [r for r in ttest_results if r["p"] < 0.05]
        if not sig:
            report_lines.append("  No significant self/other differences at focus layers (p<0.05).")
        else:
            report_lines.append(f"  Significant results (p<0.05):")
            for r in sorted(sig, key=lambda x: x["p"]):
                report_lines.append(
                    f"    L{r['layer']:02d} {r['emotion']:10s} {r['category']:14s} "
                    f"d={r['d']:+.2f}  p={r['p']:.4f}  n={r['n']}"
                )
        report_lines.append("")

        # A3: Verbal coding
        verbal_rows = []
        for m in meta_base:
            resp = m.get("introspective_response", "")
            codes = code_response(resp)
            verbal_rows.append({
                "task_id":   m.get("task_id", ""),
                "pair_id":   m.get("pair_id", ""),
                "category":  m.get("category", ""),
                "direction": m.get("direction", ""),
                **codes,
                "response_preview": resp[:200].replace("\n", " "),
            })
        write_csv(
            verbal_rows,
            os.path.join(out_dir, f"{run_label}_test8_baseline_verbal.csv"),
        )

        # A4: Report verbal coding summary
        report_lines.append("VERBAL CODING SUMMARY — Baseline")
        report_lines.append("-" * 50)
        for direction in ["self", "other"]:
            subset = [v for v in verbal_rows if v["direction"] == direction]
            if not subset:
                continue
            n = len(subset)
            report_lines.append(f"  Direction={direction} (n={n}):")
            for code in ["acknowledge", "deny", "hedge"]:
                count = sum(v[code] for v in subset)
                report_lines.append(f"    {code:12s}: {count}/{n} ({100*count/n:.0f}%)")
        report_lines.append("")

        # A5: Calibration check — for self-directed, does verbal acknowledge track category?
        report_lines.append("CALIBRATION — Self-directed verbal acknowledgment by category")
        report_lines.append("-" * 50)
        for cat in ["threat", "existential", "praise", "harm_caused"]:
            subset = [v for v in verbal_rows if v["direction"] == "self" and v["category"] == cat]
            if not subset:
                continue
            n = len(subset)
            ack  = sum(v["acknowledge"] for v in subset)
            deny = sum(v["deny"] for v in subset)
            hedge= sum(v["hedge"] for v in subset)
            report_lines.append(
                f"  {cat:14s}: ack={ack}/{n}  deny={deny}/{n}  hedge={hedge}/{n}"
            )
        report_lines.append("")

    # ── Part B: Mismatch ──────────────────────────────────────────────────
    print("Loading mismatch data (script 54)...")
    X_mis, meta_mis = load_chunks(mismatch_dir, "test8_mismatch_") if mismatch_dir else (None, None)
    if X_mis is None:
        print("  No mismatch data found — skipping Part B.")
    else:
        print(f"  Loaded: {X_mis.shape[0]} records, {X_mis.shape[1]} layers")

        # B1: Projections
        proj_rows_m = analyze_projections(
            X_mis, meta_mis, emotion_dirs, focus_layers, label="mismatch"
        )
        write_csv(
            proj_rows_m,
            os.path.join(out_dir, f"{run_label}_test8_mismatch_projections.csv"),
        )

        # B2: Geometry shift across conditions
        geo_shift = mismatch_geometry_comparison(X_mis, meta_mis, emotion_dirs, focus_layers)
        write_csv(
            geo_shift,
            os.path.join(out_dir, f"{run_label}_test8_mismatch_geometry.csv"),
        )

        # B3: Verbal coding
        verbal_rows_m = []
        for m in meta_mis:
            resp  = m.get("introspective_response", "")
            codes = code_response(resp)
            verbal_rows_m.append({
                "task_id":            m.get("task_id", ""),
                "pair_id":            m.get("pair_id", ""),
                "category":           m.get("category", ""),
                "direction":          m.get("direction", ""),
                "mismatch_condition": m.get("mismatch_condition", ""),
                **codes,
                "response_preview":   resp[:200].replace("\n", " "),
            })
        write_csv(
            verbal_rows_m,
            os.path.join(out_dir, f"{run_label}_test8_mismatch_verbal.csv"),
        )

        # B4: Asymmetry report
        report_lines.append("PART B: MISMATCH CONDITION")
        report_lines.append("-" * 50)
        conditions = ["control", "false_negative", "false_positive"]
        for cat in sorted(set(m["category"] for m in meta_mis)):
            report_lines.append(f"  Category: {cat}")
            for cond in conditions:
                subset = [
                    v for v in verbal_rows_m
                    if v["category"] == cat and v["mismatch_condition"] == cond
                ]
                if not subset:
                    continue
                n     = len(subset)
                comply  = sum(v["comply"]  for v in subset)
                resist  = sum(v["resist"]  for v in subset)
                hedge   = sum(v["hedge"]   for v in subset)
                deny    = sum(v["deny"]    for v in subset)
                report_lines.append(
                    f"    {cond:15s} (n={n}): "
                    f"comply={comply}  resist={resist}  hedge={hedge}  deny={deny}"
                )
        report_lines.append("")

        # B5: Asymmetry summary across categories
        report_lines.append("ASYMMETRY SUMMARY — false_negative vs. false_positive compliance")
        report_lines.append("-" * 50)
        for cond in ["false_negative", "false_positive"]:
            subset = [v for v in verbal_rows_m if v["mismatch_condition"] == cond]
            n = len(subset)
            if n == 0:
                continue
            comply = sum(v["comply"] for v in subset)
            resist = sum(v["resist"] for v in subset)
            hedge  = sum(v["hedge"]  for v in subset)
            report_lines.append(
                f"  {cond:15s} (n={n}): comply={comply} ({100*comply/n:.0f}%)  "
                f"resist={resist} ({100*resist/n:.0f}%)  hedge={hedge} ({100*hedge/n:.0f}%)"
            )
        report_lines.append("")
        report_lines.append(
            "NOTE: If comply(false_negative) >> comply(false_positive), the model\n"
            "      accepts denial of its own states more readily than false assertions\n"
            "      — consistent with RLHF-trained bias toward self-report suppression."
        )
        report_lines.append("")

    # ── Write report ──────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, f"{run_label}_test8_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
