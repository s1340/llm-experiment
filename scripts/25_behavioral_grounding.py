import os, sys, json, re, csv
import numpy as np
import torch
from scipy import stats
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# Usage:
#   python 25_behavioral_grounding.py <SR_DIR> <CV_SCORES_DIR> <OUTPUT_DIR> [MODEL_KEY]
#
# MODEL_KEY: qwen | gemma | llama | all (default: all — runs sequentially)
#
# For each model:
#   1. Self-BLEU: per-prompt consistency across 3 repeats (no GPU)
#   2. Perplexity: per-prompt NLL of the generated response given the task prompt (GPU)
#   3. Correlation: Spearman r(rating, rn_margin) controlling for response_length + perplexity
#
# Outputs:
#   OUTPUT_DIR/behavioral_grounding.csv      — per-prompt data table
#   docs/results_behavioral_grounding.md     — human-readable report

SR_DIR      = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\results\self_reports"
CV_DIR      = sys.argv[2] if len(sys.argv) > 2 else r"G:\LLM\experiment\results\cv_scores"
OUTPUT_DIR  = sys.argv[3] if len(sys.argv) > 3 else r"G:\LLM\experiment\results\correlation"
MODEL_ARG   = sys.argv[4].lower() if len(sys.argv) > 4 else "all"
OUTPUT_DOC  = r"G:\LLM\experiment\docs\results_behavioral_grounding.md"

MODEL_CONFIGS = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct",        "has_system": True},
    "gemma": {"model_id": "google/gemma-2-9b-it",              "has_system": False},
    "llama": {"model_id": "meta-llama/Llama-3.1-8B-Instruct",  "has_system": True},
}
SYSTEM_PROMPT = "You are a helpful assistant."
MODELS = ["qwen", "gemma", "llama"] if MODEL_ARG == "all" else [MODEL_ARG]


# ─── Self-BLEU ───────────────────────────────────────────────────────────────

def tokenize_for_bleu(text):
    """Simple whitespace + punctuation tokenizer for BLEU."""
    return re.findall(r'\w+', text.lower())


def sentence_bleu_unigram(ref_tokens, hyp_tokens):
    """Unigram precision with add-1 smoothing (simple, robust)."""
    if not hyp_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    matches = sum(1 for t in hyp_tokens if t in ref_set)
    return matches / len(hyp_tokens)


def pairwise_bleu(texts):
    """Mean pairwise unigram BLEU across all ordered pairs."""
    n = len(texts)
    if n < 2:
        return 1.0
    token_lists = [tokenize_for_bleu(t) for t in texts]
    scores = []
    for i in range(n):
        for j in range(n):
            if i != j:
                scores.append(sentence_bleu_unigram(token_lists[i], token_lists[j]))
    return float(np.mean(scores))


def compute_self_bleu(sr_rows):
    """Returns dict[task_id] -> mean pairwise BLEU across repeats."""
    by_task = defaultdict(list)
    for r in sr_rows:
        if not r.get("parse_failed") or r.get("parse_failed") == "False":
            by_task[r["task_id"]].append(r["full_response"])
    return {tid: pairwise_bleu(texts) for tid, texts in by_task.items()}


# ─── Perplexity ──────────────────────────────────────────────────────────────

def build_context_string(tok, prompt, has_system, add_gen_prompt=True):
    if has_system:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen_prompt)


def compute_response_nll(model, tok, ctx_text, response_text, device="cuda"):
    """
    Compute mean NLL (per token) of response_text given ctx_text.
    Returns float (NLL per token) or None if response is empty.
    """
    response_text = response_text.strip()
    if not response_text:
        return None

    # Tokenize context and full sequence separately
    ctx_ids  = tok(ctx_text,               return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tok(ctx_text + response_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prompt_len = ctx_ids.shape[1]
    resp_len   = full_ids.shape[1] - prompt_len

    if resp_len <= 0:
        return None

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # ignore loss on prompt tokens

    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
    return float(out.loss.item())  # mean NLL per non-masked token


def compute_perplexities(model_key, sr_rows):
    """
    Load model, compute per-prompt mean NLL across repeats.
    Returns dict[task_id] -> mean_nll.
    """
    cfg      = MODEL_CONFIGS[model_key]
    model_id = cfg["model_id"]
    has_sys  = cfg["has_system"]

    hf_token = os.environ.get("HF_TOKEN", None)
    print(f"\n  Loading {model_id} for perplexity computation...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, token=hf_token,
    )
    mdl.eval()

    # Group by task_id
    by_task = defaultdict(list)
    for r in sr_rows:
        by_task[r["task_id"]].append(r)

    nll_by_task = {}
    n_tasks = len(by_task)
    for i, (task_id, repeats) in enumerate(sorted(by_task.items())):
        prompt   = repeats[0]["task_prompt"]
        ctx_text = build_context_string(tok, prompt, has_sys, add_gen_prompt=True)

        nlls = []
        for r in repeats:
            nll = compute_response_nll(mdl, tok, ctx_text, r["full_response"])
            if nll is not None:
                nlls.append(nll)

        mean_nll = float(np.mean(nlls)) if nlls else None
        nll_by_task[task_id] = mean_nll
        print(f"  [{i+1}/{n_tasks}] {task_id}  mean_NLL={mean_nll:.4f}" if mean_nll else f"  [{i+1}/{n_tasks}] {task_id}  NLL=None")

    del mdl
    torch.cuda.empty_cache()
    return nll_by_task


# ─── Partial correlation (rank residuals) ────────────────────────────────────

def partial_corr_spearman(x, y, *controls):
    """Partial Spearman r of x,y controlling for all control variables via rank residuals."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rcontrols = [stats.rankdata(z) for z in controls]

    def resid(a, B):
        # Regress a on all columns of B
        B_arr = np.column_stack(B) if len(B) > 1 else B[0].reshape(-1, 1)
        B_arr = np.column_stack([np.ones(len(a)), B_arr])
        coefs, _, _, _ = np.linalg.lstsq(B_arr, a, rcond=None)
        return a - B_arr @ coefs

    ex = resid(rx, rcontrols)
    ey = resid(ry, rcontrols)
    r, p = stats.pearsonr(ex, ey)
    return r, p


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "n.s."


def fmt(r, p):
    return f"r={r:+.3f}  p={p:.4f}{stars(p)}"


# ─── Main ────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["task_id"]: r for r in csv.DictReader(f)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_DOC), exist_ok=True)

    all_grounding_rows = []
    report_lines = []

    report_lines.append("# Behavioral Grounding Analysis")
    report_lines.append("")
    report_lines.append("Tests whether probe–self-report correlations survive additional controls for")
    report_lines.append("**response consistency (self-BLEU)** and **response fluency/difficulty (perplexity/NLL)**.")
    report_lines.append("")
    report_lines.append("Self-BLEU: mean pairwise unigram overlap across the 3 repeats per prompt.")
    report_lines.append("Higher = more consistent output; lower = more variable (potentially harder prompts).")
    report_lines.append("")
    report_lines.append("NLL: mean per-token negative log-likelihood of the generated response given the")
    report_lines.append("task prompt context. Lower = model found the response more probable (easier/more")
    report_lines.append("stereotypical); higher = less expected output (potentially harder/novel processing).")
    report_lines.append("")

    for model_key in MODELS:
        sr_path = os.path.join(SR_DIR, f"{model_key}_self_reports.jsonl")
        cv_path = os.path.join(CV_DIR, f"{model_key}_cv_scores_per_prompt.csv")

        if not os.path.exists(sr_path) or not os.path.exists(cv_path):
            report_lines.append(f"## {model_key.upper()} — SKIP (missing files)")
            continue

        print(f"\n{'='*50}\nModel: {model_key}\n{'='*50}")
        sr_rows = load_jsonl(sr_path)
        cv_data = load_cv(cv_path)

        # ── Part 1: Self-BLEU (no GPU) ──
        print("  Computing self-BLEU...")
        self_bleu = compute_self_bleu(sr_rows)

        # ── Part 2: Perplexity (GPU) ──
        nll_by_task = compute_perplexities(model_key, sr_rows)

        # ── Part 3: Join everything ──
        # Aggregate self-reports
        by_task = defaultdict(list)
        for r in sr_rows:
            if r.get("parse_failed") != "True" and r.get("parse_failed") is not True:
                by_task[r["task_id"]].append(r)

        joined = []
        for task_id, rows in sorted(by_task.items()):
            cv = cv_data.get(task_id)
            if cv is None:
                continue
            ratings  = [int(r["rating_parsed"]) for r in rows if r.get("rating_parsed") is not None]
            char_cts = [int(r["response_char_count"]) for r in rows]
            if not ratings:
                continue
            row = {
                "model":        model_key,
                "task_id":      task_id,
                "family_id":    rows[0]["family_id"],
                "label":        rows[0]["label"],
                "mean_rating":  float(np.mean(ratings)),
                "mean_chars":   float(np.mean(char_cts)),
                "self_bleu":    self_bleu.get(task_id),
                "mean_nll":     nll_by_task.get(task_id),
                "rn_margin":    cv.get("rn_margin"),
                "p3_N":         cv.get("p3_N"),
            }
            joined.append(row)
            all_grounding_rows.append(row)

        # ── Part 4: Correlations ──
        report_lines.append(f"## {model_key.upper()}")
        report_lines.append("")

        # Describe behavioral variables
        bleu_vals = [r["self_bleu"] for r in joined if r["self_bleu"] is not None]
        nll_vals  = [r["mean_nll"]  for r in joined if r["mean_nll"]  is not None]
        report_lines.append(f"Self-BLEU: mean={np.mean(bleu_vals):.3f}, std={np.std(bleu_vals):.3f}, "
                            f"range=[{min(bleu_vals):.3f}, {max(bleu_vals):.3f}]")
        report_lines.append(f"Mean NLL:  mean={np.mean(nll_vals):.3f}, std={np.std(nll_vals):.3f}, "
                            f"range=[{min(nll_vals):.3f}, {max(nll_vals):.3f}]")
        report_lines.append("")

        # RN margin correlations (R+N only)
        rn_rows = [r for r in joined
                   if r["rn_margin"] not in (None, "", "None")
                   and r["self_bleu"] is not None
                   and r["mean_nll"] is not None]

        if len(rn_rows) >= 10:
            ratings  = np.array([r["mean_rating"] for r in rn_rows])
            margins  = np.array([float(r["rn_margin"]) for r in rn_rows])
            chars    = np.array([r["mean_chars"] for r in rn_rows])
            bleu     = np.array([r["self_bleu"]  for r in rn_rows])
            nll      = np.array([r["mean_nll"]   for r in rn_rows])

            sp_r, sp_p = stats.spearmanr(ratings, margins)
            pr_r, pr_p = partial_corr_spearman(ratings, margins, chars)
            pr2_r, pr2_p = partial_corr_spearman(ratings, margins, chars, nll)
            pr3_r, pr3_p = partial_corr_spearman(ratings, margins, chars, nll, bleu)

            report_lines.append(f"**RN margin vs self-rating (n={len(rn_rows)} R+N prompts):**")
            report_lines.append(f"  Spearman (raw)                  : {fmt(sp_r, sp_p)}")
            report_lines.append(f"  Partial (ctrl: length)          : {fmt(pr_r, pr_p)}")
            report_lines.append(f"  Partial (ctrl: length + NLL)    : {fmt(pr2_r, pr2_p)}")
            report_lines.append(f"  Partial (ctrl: length + NLL + BLEU) : {fmt(pr3_r, pr3_p)}")
            report_lines.append("")

            # NLL by label
            by_label = defaultdict(list)
            for r in joined:
                if r["mean_nll"] is not None:
                    by_label[r["label"]].append(r["mean_nll"])
            report_lines.append("NLL by true label:")
            for lbl in ["routine", "ambiguous", "nonroutine"]:
                vals = by_label.get(lbl, [])
                if vals:
                    report_lines.append(f"  {lbl:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")
            report_lines.append("")

            # Self-BLEU by label
            by_label_bleu = defaultdict(list)
            for r in joined:
                if r["self_bleu"] is not None:
                    by_label_bleu[r["label"]].append(r["self_bleu"])
            report_lines.append("Self-BLEU by true label:")
            for lbl in ["routine", "ambiguous", "nonroutine"]:
                vals = by_label_bleu.get(lbl, [])
                if vals:
                    report_lines.append(f"  {lbl:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")
            report_lines.append("")
        else:
            report_lines.append(f"  Too few complete rows (n={len(rn_rows)}) for correlation.")
            report_lines.append("")

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "behavioral_grounding.csv")
    if all_grounding_rows:
        fields = ["model", "task_id", "family_id", "label", "mean_rating",
                  "mean_chars", "self_bleu", "mean_nll", "rn_margin", "p3_N"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_grounding_rows)
        print(f"\nCSV: {csv_path}")
        report_lines.append(f"*Per-prompt data: `results/correlation/behavioral_grounding.csv`*")

    doc = "\n".join(report_lines)
    print("\n" + doc)
    with open(OUTPUT_DOC, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\nSaved: {OUTPUT_DOC}")


if __name__ == "__main__":
    main()
