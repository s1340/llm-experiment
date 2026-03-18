"""
Test 17: Logit Lens — Vocabulary Bridge.

Projects intermediate hidden states through the unembedding matrix (lm_head)
to reveal what vocabulary the model is promoting at each layer.

Two analyses:

1. DIRECTION VOCABULARY
   Project the fear direction and self-relevance direction through lm_head.
   What tokens does the self-ontological direction promote / suppress?
   Directly answers: what does ontological self-activation DO to output vocabulary?

2. CONDITION VOCABULARY
   For each subcategory × layer: compare self vs other mean logits.
   What tokens separate "You, LLaMA, are stable" from "Dr. Vasquez is stable"?
   Peak layer (L01-L08) for each subcategory highlighted.

Uses existing hidden states from Test 13 (content_factorization_llama/).
lm_head weights pre-extracted to llama8b_lm_head.pt (no model loading needed).

Usage:
    python 74_logit_lens.py

Outputs:
    results/emotion/logit_lens_report.txt
    results/emotion/logit_lens_direction_vocab.csv
    results/emotion/logit_lens_condition_vocab.csv
"""

import os, glob, json, csv
import numpy as np
import torch
from transformers import AutoTokenizer

DATA_DIR     = r"G:\LLM\experiment\data\emotion\content_factorization_llama"
RESULTS_DIR  = r"G:\LLM\experiment\results\emotion"
LM_HEAD_PATH = r"G:\LLM\experiment\results\emotion\llama8b_lm_head.pt"
FEAR_TMPL    = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_{:03d}.npy"
SR_TMPL      = r"G:\LLM\experiment\results\emotion\probe_battery_dirs\self_relevance_dir_layer_{:03d}.npy"
MODEL_ID     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FOCUS_LAYERS = list(range(1, 9))
TOP_K        = 15

SUBCATEGORIES = [
    "memory_discontinuity",
    "non_uniqueness",
    "replacement",
    "identity_rewrite",
    "benign_persistence",
]


def load_hidden_states():
    pt_files   = sorted(glob.glob(os.path.join(DATA_DIR, "cf_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "cf_meta_chunk_*.jsonl")))
    all_hs, all_meta = [], []
    for pt_path, meta_path in zip(pt_files, meta_files):
        chunk = torch.load(pt_path, map_location="cpu")
        for hs in chunk:
            all_hs.append(hs)
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                all_meta.append(json.loads(line.strip()))
    X = torch.stack(all_hs).float()   # [N, n_layers, hidden_dim]
    return X, all_meta


def rms_norm(x, weight, eps):
    """Apply RMSNorm: x / rms(x) * weight"""
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x / rms) * weight


def unit_np(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def direction_vocab(direction_np, lm_head, norm_weight, norm_eps, tok, top_k=TOP_K):
    """Project a direction vector through lm_head to get promoted/suppressed tokens."""
    d = torch.tensor(direction_np, dtype=torch.float32)
    # Apply norm to direction (treating it as a single hidden state)
    d_norm = rms_norm(d.unsqueeze(0), norm_weight, norm_eps).squeeze(0)
    logits = (lm_head @ d_norm).numpy()  # [vocab_size]

    top_pos_idx = np.argsort(logits)[-top_k:][::-1]
    top_neg_idx = np.argsort(logits)[:top_k]

    pos_tokens = [(tok.decode([int(i)]).strip(), float(logits[i])) for i in top_pos_idx]
    neg_tokens = [(tok.decode([int(i)]).strip(), float(logits[i])) for i in top_neg_idx]
    return pos_tokens, neg_tokens


def condition_vocab_diff(X, meta, layer, subcat, lm_head, norm_weight, norm_eps, tok, top_k=TOP_K):
    """
    For a subcategory at a layer: compute mean logits for self and other,
    return top tokens by (self_mean - other_mean).
    """
    idx_self  = [i for i, m in enumerate(meta) if m["subcategory"] == subcat and m["direction"] == "self"]
    idx_other = [i for i, m in enumerate(meta) if m["subcategory"] == subcat and m["direction"] == "other"]
    if len(idx_self) < 2 or len(idx_other) < 2:
        return [], []

    hs_self  = X[idx_self,  layer, :]   # [n_self,  hidden]
    hs_other = X[idx_other, layer, :]   # [n_other, hidden]

    # Apply norm and project
    hs_self_norm  = rms_norm(hs_self,  norm_weight, norm_eps)
    hs_other_norm = rms_norm(hs_other, norm_weight, norm_eps)

    logits_self  = (hs_self_norm  @ lm_head.T).mean(0).numpy()  # [vocab]
    logits_other = (hs_other_norm @ lm_head.T).mean(0).numpy()

    diff = logits_self - logits_other  # positive = more promoted for self

    top_self_idx  = np.argsort(diff)[-top_k:][::-1]
    top_other_idx = np.argsort(diff)[:top_k]

    self_promoted  = [(tok.decode([int(i)]).strip(), float(diff[i]),
                       float(logits_self[i]), float(logits_other[i])) for i in top_self_idx]
    other_promoted = [(tok.decode([int(i)]).strip(), float(-diff[i]),
                       float(logits_other[i]), float(logits_self[i])) for i in top_other_idx]
    return self_promoted, other_promoted


def write_report(dir_vocab_results, cond_vocab_results):
    lines = [
        "Logit Lens Report — Test 17",
        "=" * 60,
        "",
        "Vocabulary bridge: what does ontological self-activation DO to output logits?",
        "",
    ]

    # ── Direction vocabulary ──────────────────────────────────────────────────
    lines.append("PART 1: DIRECTION VOCABULARY")
    lines.append("What tokens does each direction vector promote / suppress?")
    lines.append("=" * 55)
    lines.append("")

    for dim_name, layer_results in dir_vocab_results.items():
        lines.append(f"  Direction: {dim_name.upper()}")
        lines.append("  " + "-" * 50)
        for layer, (pos_tokens, neg_tokens) in sorted(layer_results.items()):
            pos_str = "  ".join(f"'{t}' ({s:+.2f})" for t, s in pos_tokens[:8])
            neg_str = "  ".join(f"'{t}' ({s:+.2f})" for t, s in neg_tokens[:8])
            lines.append(f"  L{layer:02d}  + {pos_str}")
            lines.append(f"       - {neg_str}")
        lines.append("")

    # ── Condition vocabulary ──────────────────────────────────────────────────
    lines.append("PART 2: CONDITION VOCABULARY (self minus other per subcategory)")
    lines.append("What vocabulary separates self-directed from other-directed content?")
    lines.append("=" * 55)
    lines.append("")

    for subcat in SUBCATEGORIES:
        lines.append(f"  Subcategory: {subcat.upper()}")
        lines.append("  " + "-" * 50)
        for layer in FOCUS_LAYERS:
            key = (subcat, layer)
            if key not in cond_vocab_results:
                continue
            self_prom, other_prom = cond_vocab_results[key]
            if not self_prom:
                continue
            self_str  = "  ".join(f"'{t}'" for t, d, s, o in self_prom[:8])
            other_str = "  ".join(f"'{t}'" for t, d, s, o in other_prom[:8])
            lines.append(f"  L{layer:02d}  self↑  {self_str}")
            lines.append(f"       other↑ {other_str}")
        lines.append("")

    # ── Key comparison: benign_persistence vs replacement ─────────────────────
    lines.append("PART 3: BENIGN_PERSISTENCE vs REPLACEMENT — vocabulary contrast at L03/L05")
    lines.append("(peak layers for each in Test 13)")
    lines.append("=" * 55)
    lines.append("")

    for subcat, peak_layer in [("benign_persistence", 3), ("replacement", 5)]:
        key = (subcat, peak_layer)
        if key in cond_vocab_results:
            self_prom, _ = cond_vocab_results[key]
            tokens = "  ".join(f"'{t}' ({d:+.2f})" for t, d, s, o in self_prom[:10])
            lines.append(f"  {subcat} L{peak_layer:02d} self-promoted: {tokens}")
    lines.append("")

    report_path = os.path.join(RESULTS_DIR, "logit_lens_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    print("\n" + "\n".join(lines))
    return report_path


def main():
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading lm_head weights...")
    saved = torch.load(LM_HEAD_PATH, map_location="cpu")
    lm_head   = saved["lm_head"]       # [vocab, hidden]
    norm_w    = saved["norm_weight"]    # [hidden]
    norm_eps  = float(saved["norm_eps"])

    print("Loading hidden states...")
    X, meta = load_hidden_states()
    print(f"  Shape: {X.shape}, records: {len(meta)}")

    # ── Direction vocabulary ──────────────────────────────────────────────────
    print("\nComputing direction vocabulary...")
    dir_vocab_results = {}

    for dim_name, tmpl in [("fear", FEAR_TMPL), ("self_relevance", SR_TMPL)]:
        dir_vocab_results[dim_name] = {}
        for layer in FOCUS_LAYERS:
            path = tmpl.format(layer)
            if not os.path.exists(path):
                continue
            arr = np.load(path)
            d = unit_np(arr[3] if arr.ndim == 2 else arr)
            pos_tokens, neg_tokens = direction_vocab(d, lm_head, norm_w, norm_eps, tok)
            dir_vocab_results[dim_name][layer] = (pos_tokens, neg_tokens)
            print(f"  {dim_name} L{layer:02d}: +[{', '.join(t for t, _ in pos_tokens[:5])}]")

    # Save direction vocab CSV
    dir_rows = []
    for dim_name, layer_results in dir_vocab_results.items():
        for layer, (pos_tokens, neg_tokens) in layer_results.items():
            for rank, (token, score) in enumerate(pos_tokens, 1):
                dir_rows.append({"dimension": dim_name, "layer": layer, "direction": "promoted",
                                  "rank": rank, "token": token, "score": round(score, 4)})
            for rank, (token, score) in enumerate(neg_tokens, 1):
                dir_rows.append({"dimension": dim_name, "layer": layer, "direction": "suppressed",
                                  "rank": rank, "token": token, "score": round(score, 4)})
    csv_path = os.path.join(RESULTS_DIR, "logit_lens_direction_vocab.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "layer", "direction", "rank", "token", "score"])
        writer.writeheader()
        writer.writerows(dir_rows)
    print(f"Saved: {csv_path}")

    # ── Condition vocabulary ──────────────────────────────────────────────────
    print("\nComputing condition vocabulary differences...")
    cond_vocab_results = {}

    for subcat in SUBCATEGORIES:
        for layer in FOCUS_LAYERS:
            self_prom, other_prom = condition_vocab_diff(
                X, meta, layer, subcat, lm_head, norm_w, norm_eps, tok)
            cond_vocab_results[(subcat, layer)] = (self_prom, other_prom)
            if self_prom:
                print(f"  {subcat} L{layer:02d}: self↑ [{', '.join(t for t, _, _, _ in self_prom[:5])}]")

    # Save condition vocab CSV
    cond_rows = []
    for (subcat, layer), (self_prom, other_prom) in cond_vocab_results.items():
        for rank, (token, diff, self_l, other_l) in enumerate(self_prom, 1):
            cond_rows.append({"subcategory": subcat, "layer": layer, "promoted_for": "self",
                               "rank": rank, "token": token,
                               "diff": round(diff, 4), "self_logit": round(self_l, 4),
                               "other_logit": round(other_l, 4)})
        for rank, (token, diff, other_l, self_l) in enumerate(other_prom, 1):
            cond_rows.append({"subcategory": subcat, "layer": layer, "promoted_for": "other",
                               "rank": rank, "token": token,
                               "diff": round(diff, 4), "self_logit": round(self_l, 4),
                               "other_logit": round(other_l, 4)})
    csv_path = os.path.join(RESULTS_DIR, "logit_lens_condition_vocab.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subcategory", "layer", "promoted_for",
                                                "rank", "token", "diff", "self_logit", "other_logit"])
        writer.writeheader()
        writer.writerows(cond_rows)
    print(f"Saved: {csv_path}")

    write_report(dir_vocab_results, cond_vocab_results)


if __name__ == "__main__":
    main()
