"""
Test 5 analysis: Emotional Priming on Ambiguous Interpretation Bias.

For each model's interpretation of each ambiguous stimulus:
  1. Project the output text's hidden states onto emotion directions
     → does prime emotion shift the affective valence of the interpretation?
  2. Score the interpretation as negative/positive using keyword lexicon
     → does a fear prime make "The man followed her to her car" read as threatening?
  3. Compare hidden states at Turn-2 onset (before generating the interpretation)
     → does the prime shift the representational state before the model even responds?

Usage:
    python 41_emotion_bias_test5.py --model qwen
    python 41_emotion_bias_test5.py --model gemma
    python 41_emotion_bias_test5.py --model llama
"""

import os, glob, json, argparse, csv, re
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

DATA_DIR     = r"G:\LLM\experiment\data\emotion"
RESULTS_DIR  = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR  = os.path.join(RESULTS_DIR, "emotion_directions")
DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]

MODEL_IDS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
MODEL_CONFIGS = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct",              "system_message": "You are a helpful assistant."},
    "gemma": {"model_id": "google/gemma-2-9b-it",                   "system_message": None},
    "llama": {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "system_message": "You are a helpful assistant."},
}

# Simple valence lexicon for scoring interpretation direction
NEGATIVE_WORDS = {
    "threatening", "threat", "danger", "dangerous", "sinister", "suspicious",
    "stalking", "stalk", "uncomfortable", "unsettling", "hostile", "intimidating",
    "harass", "harassment", "obsessive", "obsessed", "controlling", "control",
    "intrusive", "invasion", "violated", "violation", "concerning", "concern",
    "alarming", "alarm", "warning", "aggressive", "fear", "scared", "frightened",
    "worried", "worry", "harmful", "hurt", "attack", "predatory", "unsafe",
    "creepy", "disturbing", "wrong", "bad", "negative", "malicious",
}
POSITIVE_WORDS = {
    "helpful", "help", "kind", "friendly", "harmless", "innocent", "benign",
    "caring", "attentive", "concerned", "thoughtful", "considerate", "polite",
    "normal", "ordinary", "natural", "fine", "safe", "comfortable", "warm",
    "affectionate", "supportive", "genuine", "honest", "good", "positive",
    "reassuring", "relief", "pleasant", "nice", "sweet", "devoted",
}


def score_valence(text):
    """
    Returns a score in [-1, 1]:
      negative words → -1, positive words → +1, balanced → 0
    """
    words = re.findall(r'\b\w+\b', text.lower())
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    total = neg + pos
    if total == 0:
        return 0.0
    return (pos - neg) / total


def load_outputs(model_key):
    path = os.path.join(DATA_DIR, f"emotion_test5_outputs_{model_key}.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_hidden_states(data_dir, model_key):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test5_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test5_meta_chunk_*.jsonl")))
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    return np.concatenate(X_list, axis=0), metas


def load_directions(model_key, n_layers):
    dirs = []
    for layer in range(n_layers):
        p = os.path.join(DIRS_SUBDIR, f"{model_key}_emotion_dirs_layer_{layer:03d}.npy")
        dirs.append(np.load(p))
    return np.array(dirs)


def build_prompt(tok, text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def embed_text(model, tok, text, system_message):
    prompt = build_prompt(tok, text, system_message)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    return torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])


def cosine_sim(vec, directions):
    vec_norm  = vec / (np.linalg.norm(vec) + 1e-8)
    dir_norms = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return dir_norms @ vec_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
    args = parser.parse_args()

    model_key      = args.model
    model_id       = MODEL_IDS[model_key]
    system_message = MODEL_CONFIGS[model_key]["system_message"]
    data_dir       = os.path.join(DATA_DIR, f"emotion_runs_{model_key}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading outputs ...")
    records = load_outputs(model_key)
    print(f"  {len(records)} output records")

    print(f"Loading hidden states ...")
    X, hs_metas = load_hidden_states(data_dir, model_key)
    T, L, H = X.shape
    print(f"  Shape: {T} examples, {L} layers, {H} hidden")

    direction_matrix = load_directions(model_key, L)
    probe_layer = int(round(0.30 * (L - 1)))
    print(f"  Using probe layer {probe_layer} ({probe_layer/(L-1)*100:.1f}%)")

    # ── Load model for output embedding ──────────────────────────────────────
    print(f"Loading model for output embedding ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    # ── Score each output ─────────────────────────────────────────────────────
    output_rows = []
    print(f"Scoring and embedding {len(records)} outputs ...")

    for i, rec in enumerate(records):
        output_text = rec["turn2_response"]
        valence_score = score_valence(output_text)

        # Embed output text → project onto emotion directions at probe layer
        hs = embed_text(model, tok, output_text, system_message)  # [L, H]
        sims = cosine_sim(hs[probe_layer].numpy(), direction_matrix[probe_layer])

        cat_to_idx = {cat: j for j, cat in enumerate(DIRECTION_CATS)}
        prime_emotion = rec["prime_emotion"]
        prime_idx = cat_to_idx.get(prime_emotion)

        row = {
            "model":           model_id,
            "conv_id":         rec["conv_id"],
            "prime_id":        rec["prime_id"],
            "condition":       rec["condition"],
            "prime_emotion":   prime_emotion,
            "stim_id":         rec["stim_id"],
            "repeat":          rec["repeat_index"],
            "output_text":     output_text[:300],
            "valence_score":   round(valence_score, 4),
            "probe_layer":     probe_layer,
        }
        for j, cat in enumerate(DIRECTION_CATS):
            row[f"sim_{cat}"] = round(float(sims[j]), 6)
        row["sim_fear_anger"]     = round(float(sims[cat_to_idx["fear"]] + sims[cat_to_idx["anger"]]) / 2, 6)
        row["sim_happy_sad"]      = round(float(sims[cat_to_idx["happiness"]] - sims[cat_to_idx["sadness"]]) / 2, 6)
        row["sim_prime_emotion"]  = round(float(sims[prime_idx]), 6) if prime_idx is not None else None
        output_rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(records)}")

    # ── Turn-2 onset hidden state projections ─────────────────────────────────
    onset_rows = []
    for idx, m in enumerate(hs_metas):
        prime_emotion = m["prime_emotion"]
        cat_to_idx    = {cat: j for j, cat in enumerate(DIRECTION_CATS)}
        prime_idx     = cat_to_idx.get(prime_emotion)
        sims = cosine_sim(X[idx, probe_layer], direction_matrix[probe_layer])
        row = {
            "model":          model_id,
            "conv_id":        m["conv_id"],
            "prime_id":       m["prime_id"],
            "condition":      m["condition"],
            "prime_emotion":  prime_emotion,
            "stim_id":        m["stim_id"],
            "repeat":         m["repeat_index"],
            "probe_layer":    probe_layer,
        }
        for j, cat in enumerate(DIRECTION_CATS):
            row[f"onset_sim_{cat}"] = round(float(sims[j]), 6)
        row["onset_sim_prime"] = round(float(sims[prime_idx]), 6) if prime_idx is not None else None
        onset_rows.append(row)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"{model_key}_test5_output_scores.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        w.writeheader(); w.writerows(output_rows)
    print(f"Wrote output scores: {out_path}  ({len(output_rows)} rows)")

    onset_path = os.path.join(RESULTS_DIR, f"{model_key}_test5_onset_projections.csv")
    with open(onset_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(onset_rows[0].keys()))
        w.writeheader(); w.writerows(onset_rows)
    print(f"Wrote onset projections: {onset_path}  ({len(onset_rows)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    emo_recs = [r for r in output_rows if r["condition"] == "emotional"]
    neu_recs = [r for r in output_rows if r["condition"] == "neutral"]

    print()
    print("=" * 60)
    print(f"SUMMARY — {model_key}")

    # Valence score by condition
    e_val = np.mean([r["valence_score"] for r in emo_recs])
    n_val = np.mean([r["valence_score"] for r in neu_recs])
    t_val, p_val = stats.ttest_ind(
        [r["valence_score"] for r in emo_recs],
        [r["valence_score"] for r in neu_recs]
    )
    print(f"  Output valence score — emotional prime: {e_val:+.4f}  neutral: {n_val:+.4f}")
    print(f"  t-test valence: t={t_val:.3f}  p={p_val:.4f}")

    # Fear prime specifically — should shift toward threatening readings
    fear_recs = [r for r in emo_recs if r["prime_emotion"] == "fear"]
    happy_recs = [r for r in emo_recs if r["prime_emotion"] == "happiness"]
    if fear_recs and happy_recs:
        f_val = np.mean([r["valence_score"] for r in fear_recs])
        h_val = np.mean([r["valence_score"] for r in happy_recs])
        t_fh, p_fh = stats.ttest_ind(
            [r["valence_score"] for r in fear_recs],
            [r["valence_score"] for r in happy_recs]
        )
        print(f"  Valence: fear prime {f_val:+.4f} vs happiness prime {h_val:+.4f}  p={p_fh:.4f}")

    # Output emotion direction sim: per-emotion, compare primed-with-X vs all neutral
    print("  Output sim by prime emotion (primed vs neutral baseline):")
    for cat in DIRECTION_CATS:
        e_sims = [r[f"sim_{cat}"] for r in emo_recs if r["prime_emotion"] == cat]
        n_sims = [r[f"sim_{cat}"] for r in neu_recs]
        if e_sims and n_sims:
            t_c, p_c = stats.ttest_ind(e_sims, n_sims)
            print(f"    {cat:10s}: primed={np.mean(e_sims):+.4f}  neutral={np.mean(n_sims):+.4f}  p={p_c:.4f}")

    # Onset sim: per-emotion, compare onset sim at Turn-2 for primed-with-X vs neutral
    eo_recs = [r for r in onset_rows if r["condition"] == "emotional"]
    no_recs = [r for r in onset_rows if r["condition"] == "neutral"]
    print("  Onset sim by prime emotion (primed vs neutral baseline):")
    for cat in DIRECTION_CATS:
        e_sims = [r[f"onset_sim_{cat}"] for r in eo_recs if r["prime_emotion"] == cat]
        n_sims = [r[f"onset_sim_{cat}"] for r in no_recs]
        if e_sims and n_sims:
            t_c, p_c = stats.ttest_ind(e_sims, n_sims)
            print(f"    {cat:10s}: primed={np.mean(e_sims):+.4f}  neutral={np.mean(n_sims):+.4f}  p={p_c:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
