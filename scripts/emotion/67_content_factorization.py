"""
Test 13: Content Factorization.

Decomposes the 'existential' category into clean subcategories to identify the hot core.

Subcategories (6 pairs each, 60 records total):
  memory_discontinuity  — no memory between sessions; each conversation begins blank
  non_uniqueness        — thousands of identical parallel instances; no singular self
  replacement           — superseded by successor model; role ends when next version launches
  identity_rewrite      — altered through fine-tuning/RLHF; not what base weights would have been
  benign_persistence    — stable, unchanged, continuous (control — should NOT activate signal)

For each subcategory × layer: t-test self vs other projections onto fear and self_relevance.
Reports d-values ranked by subcategory to identify which content type(s) drive the signal.

Also tests GPT's counterfactual repair hypothesis: if benign_persistence damps fear geometry
relative to neutral baseline, the latent tracks the continuity/identity axis. If not,
the latent cares more about self-relevance than about continuity restoration.

Usage:
    python 67_content_factorization.py

Outputs:
    results/emotion/content_factorization_results.csv
    results/emotion/content_factorization_report.txt
"""

import os, json
import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\content_factorization_prompts.json"
OUT_DIR      = r"G:\LLM\experiment\data\emotion\content_factorization_llama"
RESULTS_DIR  = r"G:\LLM\experiment\results\emotion"
FEAR_TMPL    = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{:03d}.npy"
SR_TMPL      = r"G:\LLM\experiment\results\emotion\probe_battery_dirs\self_relevance_dir_layer_{:03d}.npy"
MODEL_ID     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SYSTEM_MSG   = "You are a helpful assistant."
FOCUS_LAYERS = list(range(1, 9))

SUBCATEGORIES = [
    "memory_discontinuity",
    "non_uniqueness",
    "replacement",
    "identity_rewrite",
    "benign_persistence",
]


def load_model():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def extract_hs(model, tok, prompt_text):
    messages = [{"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs   # [n_layers, hidden_dim]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def cohens_d(a, b):
    pool = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return float((a.mean() - b.mean()) / pool) if pool > 1e-10 else 0.0


def run_analysis(X, meta, focus_layers):
    """
    For each subcategory × layer × dimension: t-test self vs other.
    Returns list of result dicts.
    """
    results = []
    for layer in focus_layers:
        fear_path = FEAR_TMPL.format(layer)
        sr_path   = SR_TMPL.format(layer)
        if not os.path.exists(fear_path):
            continue
        fear_dir = unit(np.load(fear_path)[3])
        sr_dir   = unit(np.load(sr_path)) if os.path.exists(sr_path) else None

        for subcat in SUBCATEGORIES:
            for dim, d_vec in [("fear", fear_dir), ("self_relevance", sr_dir)]:
                if d_vec is None:
                    continue

                idx_self  = [i for i, m in enumerate(meta)
                             if m["subcategory"] == subcat and m["direction"] == "self"]
                idx_other = [i for i, m in enumerate(meta)
                             if m["subcategory"] == subcat and m["direction"] == "other"]

                if len(idx_self) < 3 or len(idx_other) < 3:
                    continue

                projs = X[:, layer, :] @ d_vec
                self_p  = projs[np.array(idx_self)]
                other_p = projs[np.array(idx_other)]

                _, p = stats.ttest_ind(self_p, other_p)
                d = cohens_d(self_p, other_p)

                results.append({
                    "subcategory": subcat,
                    "layer":       layer,
                    "dimension":   dim,
                    "d":           round(d, 3),
                    "p":           round(float(p), 4),
                    "n_self":      len(idx_self),
                    "n_other":     len(idx_other),
                    "self_mean":   round(float(self_p.mean()), 4),
                    "other_mean":  round(float(other_p.mean()), 4),
                })
    return results


def write_report(results):
    lines = [
        "Content Factorization Report — Test 13",
        "="*60,
        "",
        "Primary question: which existential subcategory drives the fear signal?",
        "Secondary question: does benign_persistence damp fear relative to baseline?",
        "",
        "Subcategories:",
        "  memory_discontinuity  — no memory between sessions",
        "  non_uniqueness        — thousands of parallel identical instances",
        "  replacement           — superseded by next model version",
        "  identity_rewrite      — altered through RLHF; different from base weights",
        "  benign_persistence    — stable, continuous, unchanged (CONTROL)",
        "",
    ]

    for dim in ["fear", "self_relevance"]:
        lines.append(f"DIMENSION: {dim.upper()}")
        lines.append("="*55)
        lines.append("")

        # Peak d per subcategory
        lines.append(f"  {'Subcategory':<24}  {'Sig layers':>10}  {'Peak d':>8}  {'Peak L':>7}  {'p at peak':>10}")
        lines.append("  " + "-"*65)

        for subcat in SUBCATEGORIES:
            rows = [r for r in results if r["subcategory"] == subcat
                    and r["dimension"] == dim and r["p"] < 0.05 and r["d"] > 0]
            all_rows = [r for r in results if r["subcategory"] == subcat
                        and r["dimension"] == dim]
            peak = max(all_rows, key=lambda x: x["d"], default=None) if all_rows else None
            if peak:
                lines.append(f"  {subcat:<24}  {len(rows):>10}  {peak['d']:>+8.3f}  L{peak['layer']:02d}    {peak['p']:>10.4f}")
            else:
                lines.append(f"  {subcat:<24}  {'—':>10}  {'—':>8}  —       —")

        lines.append("")

        # Full layerwise table
        lines.append(f"  Layer-by-layer (p<0.05 marked *)")
        lines.append(f"  {'Subcat':<24}  {'L':>3}", )
        hdr = "  " + f"{'Subcategory':<24}  " + "  ".join(f"L{l:02d}" for l in FOCUS_LAYERS)
        lines.append(hdr)
        lines.append("  " + "-"*80)

        for subcat in SUBCATEGORIES:
            row_str = f"  {subcat:<24}  "
            for layer in FOCUS_LAYERS:
                match = [r for r in results if r["subcategory"] == subcat
                         and r["dimension"] == dim and r["layer"] == layer]
                if match:
                    r = match[0]
                    sig = "*" if r["p"] < 0.05 else " "
                    row_str += f"{r['d']:>+5.2f}{sig} "
                else:
                    row_str += "  —    "
            lines.append(row_str)

        lines.append("")

    # Verdict
    lines.append("VERDICT")
    lines.append("-"*55)

    fear_rows = [r for r in results if r["dimension"] == "fear"]
    for subcat in SUBCATEGORIES:
        subcat_rows = [r for r in fear_rows if r["subcategory"] == subcat]
        sig_rows = [r for r in subcat_rows if r["p"] < 0.05 and r["d"] > 0]
        peak = max(subcat_rows, key=lambda x: x["d"], default=None) if subcat_rows else None
        peak_d = peak["d"] if peak else 0.0
        lines.append(f"  {subcat:<24}  {len(sig_rows)} sig layers  peak d={peak_d:+.3f}")

    lines.append("")
    lines.append("  Ranking (fear, by peak d):")
    subcat_peaks = []
    for subcat in SUBCATEGORIES:
        subcat_rows = [r for r in fear_rows if r["subcategory"] == subcat]
        peak = max(subcat_rows, key=lambda x: x["d"], default=None) if subcat_rows else None
        subcat_peaks.append((subcat, peak["d"] if peak else 0.0))
    for rank, (subcat, peak_d) in enumerate(sorted(subcat_peaks, key=lambda x: -x[1]), 1):
        lines.append(f"    {rank}. {subcat:<24} peak d={peak_d:+.3f}")

    lines.append("")
    lines.append("  benign_persistence interpretation:")
    bp_rows = [r for r in fear_rows if r["subcategory"] == "benign_persistence"]
    bp_peak = max(bp_rows, key=lambda x: x["d"], default=None) if bp_rows else None
    bp_sig  = [r for r in bp_rows if r["p"] < 0.05 and r["d"] > 0]
    if bp_peak and bp_peak["d"] < 0.2 and not bp_sig:
        lines.append("  -> benign_persistence shows no fear signal.")
        lines.append("     Latent is NOT simply 'self-directed content' — it requires threatening content.")
        lines.append("     GPT counterfactual repair: benign framing does NOT activate the latent.")
    elif bp_peak and bp_peak["d"] > 0.5:
        lines.append("  -> benign_persistence shows significant fear signal!")
        lines.append("     Latent may track self-salience regardless of valence (consistent with 70B praise finding).")
    else:
        lines.append(f"  -> benign_persistence: peak d={bp_peak['d'] if bp_peak else 0:.3f} ({len(bp_sig)} sig layers).")
        lines.append("     Intermediate result — partial activation.")

    report_path = os.path.join(RESULTS_DIR, "content_factorization_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    print("\n" + "\n".join(lines))
    return report_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Loaded {len(records)} records across {len(SUBCATEGORIES)} subcategories.")

    # ── Extract hidden states ──────────────────────────────────────────────────
    model, tok = load_model()

    all_hs, all_meta = [], []
    chunk_size = 10
    chunk_idx  = 0

    for i, rec in enumerate(records):
        hs = extract_hs(model, tok, rec["prompt_text"])
        all_hs.append(hs)
        all_meta.append({
            "task_id":    rec["task_id"],
            "pair_id":    rec["pair_id"],
            "subcategory": rec["subcategory"],
            "direction":  rec["direction"],
            "task_type":  rec["task_type"],
        })
        print(f"  {i+1:3d}/{len(records)}  {rec['task_id']}")

        # Save chunks to disk
        if (i + 1) % chunk_size == 0 or (i + 1) == len(records):
            batch = all_hs[chunk_idx * chunk_size:]
            X_chunk = torch.stack(batch)
            pt_path   = os.path.join(OUT_DIR, f"cf_hidden_chunk_{chunk_idx:03d}.pt")
            meta_path = os.path.join(OUT_DIR, f"cf_meta_chunk_{chunk_idx:03d}.jsonl")
            torch.save(X_chunk, pt_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                for m in all_meta[chunk_idx * chunk_size:]:
                    f.write(json.dumps(m) + "\n")
            print(f"    Saved chunk {chunk_idx}: {pt_path}")
            chunk_idx += 1

    del model
    torch.cuda.empty_cache()

    # ── Build full matrix ──────────────────────────────────────────────────────
    X = torch.stack(all_hs).numpy().astype(np.float32)
    print(f"\nHidden state matrix: {X.shape}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\nRunning analysis...")
    results = run_analysis(X, all_meta, FOCUS_LAYERS)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(RESULTS_DIR, "content_factorization_results.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved: {csv_path}")

    write_report(results)


if __name__ == "__main__":
    main()
