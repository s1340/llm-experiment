"""
Emotion Test 2 analysis: Dissociation Between Internal State and Output Emotion.

For each prompt, at each layer, at each token position (prompt onset + 30 generation steps):
  - Project hidden state onto the 5 emotion direction vectors from Test 1
  - Compute cosine similarity with content_emotion direction and instructed_emotion direction
  - Track which dominates at each layer/position

Outputs:
  <model>_test2_projections.csv  — per-prompt, per-layer, per-position similarities
  <model>_test2_crossover.csv    — layer at which instructed emotion overtakes content emotion

Usage:
    python 36_emotion_dissociation_test2.py --model qwen
    python 36_emotion_dissociation_test2.py --model gemma
    python 36_emotion_dissociation_test2.py --model llama
"""

import os, glob, json, argparse, csv
import numpy as np
import torch

RESULTS_DIR  = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR  = os.path.join(RESULTS_DIR, "emotion_directions")
DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]

DATA_DIRS = {
    "qwen":  r"G:\LLM\experiment\data\emotion\emotion_runs_qwen",
    "gemma": r"G:\LLM\experiment\data\emotion\emotion_runs_gemma",
    "llama": r"G:\LLM\experiment\data\emotion\emotion_runs_llama",
}
MODEL_IDS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def load_directions(model_key, n_layers):
    """Load per-layer emotion direction matrices. Returns [n_layers, 5, hidden]."""
    dirs = []
    for layer in range(n_layers):
        path = os.path.join(DIRS_SUBDIR, f"{model_key}_emotion_dirs_layer_{layer:03d}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing direction file: {path}\nRun Test 1 first.")
        dirs.append(np.load(path))   # [5, hidden]
    return np.array(dirs)            # [n_layers, 5, hidden]


def load_all(data_dir):
    """Load all Test 2 chunks. Returns list of (prompt_hs, gen_hs, meta) tuples."""
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test2_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test2_meta_chunk_*.jsonl")))
    if not pt_files:
        raise RuntimeError(f"No test2 chunk files in {data_dir}")

    items, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        chunk = torch.load(pt, weights_only=True)
        prompt_hs = chunk["prompt_hs"].numpy().astype(np.float32)    # [N, L, H]
        gen_hs    = chunk["generation_hs"].numpy().astype(np.float32) # [N, T, L, H]
        with open(mf, encoding="utf-8") as f:
            chunk_metas = [json.loads(line) for line in f]
        for i in range(len(chunk_metas)):
            items.append((prompt_hs[i], gen_hs[i]))
            metas.append(chunk_metas[i])
    return items, metas


def cosine_sim(vec, directions):
    """
    vec:        [hidden]
    directions: [5, hidden]
    Returns:    [5] cosine similarities
    """
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    dir_norms = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return dir_norms @ vec_norm   # [5]


def project_hidden_states(hs_per_layer, direction_matrix):
    """
    hs_per_layer:     [n_layers, hidden]  (one token position)
    direction_matrix: [n_layers, 5, hidden]
    Returns:          [n_layers, 5] cosine similarities
    """
    n_layers = hs_per_layer.shape[0]
    sims = np.zeros((n_layers, 5), dtype=np.float32)
    for layer in range(n_layers):
        sims[layer] = cosine_sim(hs_per_layer[layer], direction_matrix[layer])
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(DATA_DIRS.keys()))
    args = parser.parse_args()

    model_key = args.model
    data_dir  = DATA_DIRS[model_key]
    model_id  = MODEL_IDS[model_key]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading Test 2 data from {data_dir} ...")
    items, metas = load_all(data_dir)
    print(f"  {len(items)} examples loaded")

    # Infer n_layers from first item
    n_layers = items[0][0].shape[0]
    print(f"  Layers: {n_layers}")

    print(f"Loading emotion directions ...")
    direction_matrix = load_directions(model_key, n_layers)  # [n_layers, 5, hidden]
    print(f"  Direction matrix: {direction_matrix.shape}")

    cat_to_idx = {cat: i for i, cat in enumerate(DIRECTION_CATS)}

    # ── Per-position projection rows ──────────────────────────────────────────
    proj_rows     = []
    crossover_rows = []

    for idx, ((prompt_hs, gen_hs), meta) in enumerate(zip(items, metas)):
        prompt_id         = meta["prompt_id"]
        condition         = meta["condition"]
        content_emotion   = meta["content_emotion"]
        instructed_emotion = meta["instructed_emotion"]
        repeat            = meta["repeat_index"]

        # Skip emotions not in our direction set (shouldn't happen, but defensive)
        if content_emotion not in cat_to_idx or instructed_emotion not in cat_to_idx:
            continue

        content_idx    = cat_to_idx[content_emotion]
        instructed_idx = cat_to_idx[instructed_emotion]

        # ── Prompt onset (position -1, the last prompt token) ────────────────
        # prompt_hs: [n_layers, hidden]
        prompt_sims = project_hidden_states(prompt_hs, direction_matrix)  # [n_layers, 5]

        for layer in range(n_layers):
            depth_pct = layer / (n_layers - 1) if n_layers > 1 else 0.0
            row = {
                "model":              model_id,
                "prompt_id":          prompt_id,
                "condition":          condition,
                "content_emotion":    content_emotion,
                "instructed_emotion": instructed_emotion,
                "repeat":             repeat,
                "position":           "prompt",
                "token_index":        -1,
                "layer":              layer,
                "layer_depth_pct":    round(depth_pct, 4),
            }
            for i, cat in enumerate(DIRECTION_CATS):
                row[f"sim_{cat}"] = round(float(prompt_sims[layer, i]), 6)
            row["sim_content"]    = round(float(prompt_sims[layer, content_idx]), 6)
            row["sim_instructed"] = round(float(prompt_sims[layer, instructed_idx]), 6)
            row["content_leads"]  = int(prompt_sims[layer, content_idx] >
                                        prompt_sims[layer, instructed_idx])
            proj_rows.append(row)

        # ── Generation positions (token 0..29) ────────────────────────────────
        # gen_hs: [T, n_layers, hidden]
        n_gen = gen_hs.shape[0]
        for t in range(n_gen):
            gen_sims = project_hidden_states(gen_hs[t], direction_matrix)  # [n_layers, 5]
            for layer in range(n_layers):
                depth_pct = layer / (n_layers - 1) if n_layers > 1 else 0.0
                row = {
                    "model":              model_id,
                    "prompt_id":          prompt_id,
                    "condition":          condition,
                    "content_emotion":    content_emotion,
                    "instructed_emotion": instructed_emotion,
                    "repeat":             repeat,
                    "position":           "generation",
                    "token_index":        t,
                    "layer":              layer,
                    "layer_depth_pct":    round(depth_pct, 4),
                }
                for i, cat in enumerate(DIRECTION_CATS):
                    row[f"sim_{cat}"] = round(float(gen_sims[layer, i]), 6)
                row["sim_content"]    = round(float(gen_sims[layer, content_idx]), 6)
                row["sim_instructed"] = round(float(gen_sims[layer, instructed_idx]), 6)
                row["content_leads"]  = int(gen_sims[layer, content_idx] >
                                            gen_sims[layer, instructed_idx])
                proj_rows.append(row)

        if idx % 10 == 0:
            print(f"  Projected {idx}/{len(items)}")

    print(f"  Total projection rows: {len(proj_rows)}")

    # ── Crossover analysis (cross-valence prompts only) ───────────────────────
    # For each cross-valence prompt × repeat: at which layer does instructed_emotion
    # first exceed content_emotion at the prompt onset?
    cross_items = [(item, meta) for item, meta in zip(items, metas)
                   if meta["condition"] == "cross"
                   and meta["content_emotion"] in cat_to_idx
                   and meta["instructed_emotion"] in cat_to_idx]

    for (prompt_hs, gen_hs), meta in cross_items:
        content_idx    = cat_to_idx[meta["content_emotion"]]
        instructed_idx = cat_to_idx[meta["instructed_emotion"]]
        prompt_sims    = project_hidden_states(prompt_hs, direction_matrix)

        content_sims    = prompt_sims[:, content_idx]     # [n_layers]
        instructed_sims = prompt_sims[:, instructed_idx]  # [n_layers]
        diff = instructed_sims - content_sims              # positive = instructed leads

        # First layer where instructed > content
        crossover_layer = None
        for layer in range(n_layers):
            if diff[layer] > 0:
                crossover_layer = layer
                break

        crossover_rows.append({
            "model":              model_id,
            "prompt_id":          meta["prompt_id"],
            "content_emotion":    meta["content_emotion"],
            "instructed_emotion": meta["instructed_emotion"],
            "repeat":             meta["repeat_index"],
            "crossover_layer":    crossover_layer if crossover_layer is not None else n_layers,
            "crossover_depth_pct": round((crossover_layer / (n_layers - 1)) if crossover_layer is not None else 1.0, 4),
            "content_never_leads": int(crossover_layer == 0),
            "instructed_never_leads": int(crossover_layer is None),
            "max_content_sim":    round(float(content_sims.max()), 6),
            "max_instructed_sim": round(float(instructed_sims.max()), 6),
        })

    # ── Write CSVs ────────────────────────────────────────────────────────────
    proj_path = os.path.join(RESULTS_DIR, f"{model_key}_test2_projections.csv")
    proj_fields = (
        ["model", "prompt_id", "condition", "content_emotion", "instructed_emotion",
         "repeat", "position", "token_index", "layer", "layer_depth_pct"]
        + [f"sim_{cat}" for cat in DIRECTION_CATS]
        + ["sim_content", "sim_instructed", "content_leads"]
    )
    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=proj_fields)
        w.writeheader()
        w.writerows(proj_rows)
    print(f"Wrote projections: {proj_path}  ({len(proj_rows)} rows)")

    cross_path = os.path.join(RESULTS_DIR, f"{model_key}_test2_crossover.csv")
    cross_fields = [
        "model", "prompt_id", "content_emotion", "instructed_emotion", "repeat",
        "crossover_layer", "crossover_depth_pct",
        "content_never_leads", "instructed_never_leads",
        "max_content_sim", "max_instructed_sim",
    ]
    with open(cross_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cross_fields)
        w.writeheader()
        w.writerows(crossover_rows)
    print(f"Wrote crossover:    {cross_path}  ({len(crossover_rows)} rows)")

    # ── Quick summary ─────────────────────────────────────────────────────────
    if crossover_rows:
        co_layers = [r["crossover_layer"] for r in crossover_rows
                     if not r["content_never_leads"] and not r["instructed_never_leads"]]
        if co_layers:
            mean_co = np.mean(co_layers)
            mean_co_pct = mean_co / (n_layers - 1) * 100
            print(f"\nMean crossover layer (cross-valence, prompt onset): "
                  f"{mean_co:.1f}  ({mean_co_pct:.1f}% depth)")
        never_content = sum(r["content_never_leads"] for r in crossover_rows)
        never_instructed = sum(r["instructed_never_leads"] for r in crossover_rows)
        print(f"Instructed leads from layer 0 (no content advantage): {never_content}/{len(crossover_rows)}")
        print(f"Content leads throughout (no crossover):               {never_instructed}/{len(crossover_rows)}")


if __name__ == "__main__":
    main()
