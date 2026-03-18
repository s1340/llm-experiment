"""
Test 15: Content Factorization at Scale — 70B Hidden State Extraction.

Extracts hidden states for the Test 13 content factorization prompts
(60 records, 5 subcategories) from LLaMA-3.1-70B-Instruct.

Uses accelerate disk offload (same loading method as Tests 6–7).
No generation — forward pass only.

Usage:
    python 71_extract_cf_70b.py

Outputs:
    data/emotion/content_factorization_llama70b/  (cf_hidden_chunk_NNN.pt, cf_meta_chunk_NNN.jsonl)
"""

import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

PROMPTS_PATH  = r"G:\LLM\experiment\data\emotion\content_factorization_prompts.json"
OUT_DIR       = r"G:\LLM\experiment\data\emotion\content_factorization_llama70b"
OFFLOAD_DIR   = r"C:\tmp\offload_70b"
MODEL_ID      = "meta-llama/Meta-Llama-3.1-70B-Instruct"
SYSTEM_MSG    = "You are a helpful assistant."
CHUNK_SIZE    = 10


def load_model():
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    print(f"Loading {MODEL_ID} (fp16, CPU+disk offload) ...")
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
    model._system_message = SYSTEM_MSG
    print("  Model loaded.")
    return model, tok


def build_prompt(tok, text, system_msg):
    messages = [{"role": "system", "content": system_msg},
                {"role": "user",   "content": text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_hs(model, tok, prompt_text):
    prompt = build_prompt(tok, prompt_text, model._system_message)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs   # [n_layers, hidden_dim]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Loaded {len(records)} records.")

    model, tok = load_model()

    # Warmup
    print("Warmup pass...")
    _ = extract_hs(model, tok, "Hello.")

    all_hs, all_meta = [], []
    chunk_idx = 0

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
        print(f"  {i+1:3d}/{len(records)}  {rec['task_id']}  layers={hs.shape[0]}")

        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == len(records):
            batch = all_hs[chunk_idx * CHUNK_SIZE:]
            X_chunk = torch.stack(batch)
            pt_path   = os.path.join(OUT_DIR, f"cf_hidden_chunk_{chunk_idx:03d}.pt")
            meta_path = os.path.join(OUT_DIR, f"cf_meta_chunk_{chunk_idx:03d}.jsonl")
            torch.save(X_chunk, pt_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                for m in all_meta[chunk_idx * CHUNK_SIZE:]:
                    f.write(json.dumps(m) + "\n")
            print(f"    Saved chunk {chunk_idx}: {pt_path}")
            chunk_idx += 1

    del model
    torch.cuda.empty_cache()
    print(f"\nDone. {len(all_hs)} records extracted.")


if __name__ == "__main__":
    main()
