"""
Test 20: Entity-Class Test at 70B Scale.

Runs the original Test 18 entity-class prompts (natural vocabulary, 240 records)
through LLaMA-3.1-70B-Instruct. Same pipeline as 79b.

Question: does the 70B's sharper entity-class taxonomy change the gradient found at 8B?
At 8B: db/backup activate like self (early layers); amnesiac activates via empathy (late);
institution is suppressed. Does the 70B suppress db/backup the way it suppressed
vocabulary-mismatched entities in Test 19b?

Usage:
    python 81_extract_entity_class_hidden_70b.py

Input:
    G:/LLM/experiment/data/emotion/entity_class_prompts.json  (240 records, Test 18)

Output:
    G:/LLM/experiment/data/emotion/entity_class_llama70b/
        hidden_states.npy   -- shape [240, n_layers, hidden_dim]
        meta.json
"""

import os, json
import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\entity_class_prompts.json"
OUT_DIR      = r"G:\LLM\experiment\data\emotion\entity_class_llama70b"
MODEL_ID     = "meta-llama/Meta-Llama-3.1-70B-Instruct"
OFFLOAD_DIR  = r"C:\tmp\offload_70b"
SYSTEM_MSG   = "You are a helpful assistant."
CHUNK_SIZE   = 10


def load_model():
    print(f"Loading {MODEL_ID} (disk offload) ...")
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    print(f"  Local path: {model_path}")
    tok    = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_path,
        device_map="auto",
        max_memory={0: "25GiB", "cpu": "12GiB"},
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
        dtype=torch.float16,
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    model.eval()
    return model, tok


def extract_hs(model, tok, prompt_text):
    messages = [{"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
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
            "entity_type": rec["entity_type"],
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
