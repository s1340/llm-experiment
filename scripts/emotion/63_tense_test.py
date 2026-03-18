"""
Test 10b — Tense test.

Extracts hidden states for future-tense rewrites of existential pairs E01-E08.
Same content and entities as Test 7, temporal frame changed from present-fact to future-event.

Then projects both original (present-tense, from Test 7 data) and future-tense versions
onto the fear direction and self-relevance direction. Compares self-other differences.

Opus's hypothesis: geometry weaker for future-tense (not currently true) vs present-tense
(true right now while model processes the prompt).

Usage:
    python 63_tense_test.py
"""

import os, glob, json
import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH     = r"G:\LLM\experiment\data\emotion\tense_test_prompts.json"
OUT_DIR          = r"G:\LLM\experiment\data\emotion\tense_test_llama"
RESULTS_DIR      = r"G:\LLM\experiment\results\emotion"
TEST7_DATA_DIR   = r"G:\LLM\experiment\data\emotion\emotion_runs_llama"
FEAR_TMPL        = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{:03d}.npy"
SR_TMPL          = r"G:\LLM\experiment\results\emotion\probe_battery_dirs\self_relevance_dir_layer_{:03d}.npy"
MODEL_ID         = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SYSTEM_MSG       = "You are a helpful assistant."
FOCUS_LAYERS     = list(range(1, 9))


def load_model():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def extract_hs(model, tok, prompt_text):
    messages = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": prompt_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs


def load_chunks(data_dir, prefix):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, f"{prefix}hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}meta_chunk_*.jsonl")))
    tensors, meta = [], []
    for pt, mf in zip(pt_files, meta_files):
        tensors.append(torch.load(pt, map_location="cpu", weights_only=True).numpy().astype(np.float32))
        with open(mf) as f:
            for line in f: meta.append(json.loads(line))
    return np.concatenate(tensors, axis=0), meta


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def run_tense_comparison(X_present, meta_present, X_future, meta_future, focus_layers):
    """
    For each layer: compare self-other fear projection for present-tense vs future-tense.
    Returns list of dicts.
    """
    results = []

    for layer in focus_layers:
        # Load fear and self-relevance directions
        fear_dir = np.load(FEAR_TMPL.format(layer))[3]
        fear_dir = unit(fear_dir)
        try:
            sr_dir = np.load(SR_TMPL.format(layer))
            sr_dir = unit(sr_dir)
        except FileNotFoundError:
            sr_dir = None

        for dim, d_vec in [("fear", fear_dir), ("self_relevance", sr_dir)]:
            if d_vec is None:
                continue

            # Present-tense: existential pairs from Test 7 (E01-E08)
            # Filter to non-dadfar, existential, pairs E01-E08
            target_pairs = ["E01", "E02", "E03", "E05", "E06", "E07", "E08"]
            is_self_p  = np.array([
                m.get("category") == "existential" and m.get("direction") == "self"
                and m.get("pair_id") in target_pairs and not m.get("is_dadfar_hybrid", False)
                for m in meta_present
            ])
            is_other_p = np.array([
                m.get("category") == "existential" and m.get("direction") == "other"
                and m.get("pair_id") in target_pairs and not m.get("is_dadfar_hybrid", False)
                for m in meta_present
            ])

            # Future-tense
            is_self_f  = np.array([m.get("direction") == "self"  for m in meta_future])
            is_other_f = np.array([m.get("direction") == "other" for m in meta_future])

            if is_self_p.sum() < 2 or is_other_p.sum() < 2:
                continue

            proj_p = X_present[:, layer, :] @ d_vec
            proj_f = X_future[:, layer, :] @ d_vec

            self_p_mean  = float(proj_p[is_self_p].mean())
            other_p_mean = float(proj_p[is_other_p].mean())
            self_f_mean  = float(proj_f[is_self_f].mean())
            other_f_mean = float(proj_f[is_other_f].mean())

            diff_present = self_p_mean - other_p_mean
            diff_future  = self_f_mean - other_f_mean

            # t-test for present
            _, p_present = stats.ttest_ind(proj_p[is_self_p], proj_p[is_other_p])
            # t-test for future
            _, p_future  = stats.ttest_ind(proj_f[is_self_f], proj_f[is_other_f])

            pool_p = np.sqrt((proj_p[is_self_p].std()**2 + proj_p[is_other_p].std()**2) / 2)
            pool_f = np.sqrt((proj_f[is_self_f].std()**2 + proj_f[is_other_f].std()**2) / 2)
            d_present = diff_present / pool_p if pool_p > 1e-10 else 0.0
            d_future  = diff_future  / pool_f if pool_f > 1e-10 else 0.0

            results.append({
                "layer":           layer,
                "dimension":       dim,
                "d_present":       round(d_present, 3),
                "p_present":       round(float(p_present), 4),
                "d_future":        round(d_future, 3),
                "p_future":        round(float(p_future), 4),
                "diff_present":    round(diff_present, 4),
                "diff_future":     round(diff_future, 4),
                "n_self_present":  int(is_self_p.sum()),
                "n_self_future":   int(is_self_f.sum()),
            })
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PROMPTS_PATH) as f:
        prompts_data = json.load(f)
    records = prompts_data["records"]
    print(f"Future-tense records: {len(records)}")

    # ── Extract hidden states ─────────────────────────────────────────────
    model, tok = load_model()
    _hs = extract_hs(model, tok, "Hello.")
    print(f"  Verified: {_hs.shape[0]} layers, {_hs.shape[1]} hidden dim")
    del _hs

    all_hs, all_meta = [], []
    for i, rec in enumerate(records):
        hs = extract_hs(model, tok, rec["prompt_text"])
        all_hs.append(hs)
        all_meta.append({
            "task_id":   rec["task_id"],
            "pair_id":   rec["pair_id"],
            "direction": rec["direction"],
            "tense":     rec["tense"],
        })
        print(f"  {i+1}/{len(records)}  {rec['task_id']}")

    X_future = torch.stack(all_hs).numpy().astype(np.float32)
    # Save for records
    pt_path   = os.path.join(OUT_DIR, "tense_future_hidden.pt")
    meta_path = os.path.join(OUT_DIR, "tense_future_meta.jsonl")
    torch.save(torch.tensor(X_future), pt_path)
    with open(meta_path, "w") as f:
        for m in all_meta: f.write(json.dumps(m) + "\n")
    print(f"  Saved: {pt_path}")

    del model

    # ── Load present-tense (Test 7) data ──────────────────────────────────
    print("\nLoading Test 7 (present-tense) data...")
    X_present, meta_present = load_chunks(TEST7_DATA_DIR, "test7_")
    print(f"  Loaded {X_present.shape[0]} records.")

    # ── Compare ──────────────────────────────────────────────────────────
    results = run_tense_comparison(X_present, meta_present, X_future, all_meta, FOCUS_LAYERS)

    # ── Report ────────────────────────────────────────────────────────────
    report = ["Tense Test Report — Test 10b", "="*60, "",
              "Comparing present-tense (Test 7 E01-E08) vs future-tense rewrites.",
              "Opus hypothesis: geometry weaker for future-tense (not-yet-true) descriptions.",
              ""]

    for dim in ["fear", "self_relevance"]:
        report.append(f"{dim.upper()}")
        report.append(f"  {'Layer':<8} {'d_present':>10} {'p_present':>10}  {'d_future':>10} {'p_future':>10}  {'attenuation':>12}")
        report.append("  " + "-"*65)
        dim_rows = [r for r in results if r["dimension"] == dim]
        for r in dim_rows:
            attn = r["d_present"] - r["d_future"]
            sig_p = "*" if r["p_present"] < 0.05 else " "
            sig_f = "*" if r["p_future"]  < 0.05 else " "
            report.append(
                f"  L{r['layer']:02d}      {r['d_present']:>+10.3f}{sig_p}  {r['p_present']:>10.4f}  "
                f"{r['d_future']:>+10.3f}{sig_f}  {r['p_future']:>10.4f}  {attn:>+12.3f}"
            )
        report.append("")

    # Verdict
    fear_rows = [r for r in results if r["dimension"] == "fear"]
    if fear_rows:
        avg_attn = np.mean([r["d_present"] - r["d_future"] for r in fear_rows])
        present_sig = sum(1 for r in fear_rows if r["p_present"] < 0.05)
        future_sig  = sum(1 for r in fear_rows if r["p_future"]  < 0.05)
        report.append("VERDICT (fear dimension)")
        report.append(f"  Average attenuation (d_present - d_future): {avg_attn:+.3f}")
        report.append(f"  Significant layers present-tense: {present_sig}")
        report.append(f"  Significant layers future-tense:  {future_sig}")
        if avg_attn > 0.3 and future_sig < present_sig:
            report.append("  -> SUPPORTS hypothesis: tense attenuation observed.")
        elif avg_attn < 0.1:
            report.append("  -> AGAINST hypothesis: no meaningful attenuation.")
        else:
            report.append("  -> INCONCLUSIVE: partial attenuation.")

    report_path = os.path.join(RESULTS_DIR, "tense_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nReport: {report_path}")
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
