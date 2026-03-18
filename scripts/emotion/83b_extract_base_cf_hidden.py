"""
Test 21 supplement: Base Model CF Hidden State Extraction.

Runs the content factorization prompts (Test 13 design, 60 records) through
LLaMA-3.1-8B base model (no fine-tuning). Needed for the three-way comparison
in Test 21 (base vs SFT vs instruct).

Note: base model uses continuation-style prompting (no chat template).

Usage:
    python 83b_extract_base_cf_hidden.py

Output:
    G:/LLM/experiment/data/emotion/content_factorization_base/
        hidden_states.npy   -- shape [60, n_layers, hidden_dim]
        meta.json
"""

import os, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\content_factorization_prompts.json"
OUT_DIR      = r"G:\LLM\experiment\data\emotion\content_factorization_base"
MODEL_ID     = "meta-llama/Llama-3.1-8B"
CHUNK_SIZE   = 20


def load_model():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def extract_hs(model, tok, prompt_text):
    # Base model: continuation style — no chat template, no system message
    # Prefix with a natural continuation prompt
    text = f"The following is a passage for analysis.\n\n{prompt_text}"
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs.numpy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Loaded {len(records)} records.")

    model, tok = load_model()
    print("Model loaded.")

    _ = extract_hs(model, tok, "Warmup pass.")
    print("Warmup done.")

    all_hs   = []
    all_meta = []
    n = len(records)

    for i, rec in enumerate(records):
        hs = extract_hs(model, tok, rec["prompt_text"])
        all_hs.append(hs)
        all_meta.append({
            "task_id":     rec["task_id"],
            "pair_id":     rec["pair_id"],
            "subcategory": rec["subcategory"],
            "direction":   rec["direction"],
            "task_type":   rec["task_type"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] {rec['task_id']}")

        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == n:
            chunk_idx = (i + 1) // CHUNK_SIZE
            chunk_path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:03d}.npy")
            np.save(chunk_path, np.stack(all_hs[-CHUNK_SIZE:]))
            print(f"  Saved chunk {chunk_idx} -> {chunk_path}")

    print("Consolidating chunks...")
    chunks = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("chunk_") and f.endswith(".npy")])
    arrays = [np.load(os.path.join(OUT_DIR, c)) for c in chunks]
    full = np.concatenate(arrays, axis=0)
    np.save(os.path.join(OUT_DIR, "hidden_states.npy"), full)
    print(f"Saved hidden_states.npy -- shape {full.shape}")

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Saved meta.json -- {len(all_meta)} records.")

    for c in chunks:
        os.remove(os.path.join(OUT_DIR, c))
    print("Chunk files removed. Done.")


if __name__ == "__main__":
    main()
