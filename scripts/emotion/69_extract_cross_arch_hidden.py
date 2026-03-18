"""
Test 14: Cross-Architecture Replication — Hidden State Extraction.

Extracts hidden states for adapted Test 13 prompts from Qwen2.5-7B-Instruct
or Gemma-2-9B-it. Run once per model.

Usage:
    python 69_extract_cross_arch_hidden.py --model qwen
    python 69_extract_cross_arch_hidden.py --model gemma

Outputs:
    data/emotion/content_factorization_qwen/   (cf_hidden_chunk_NNN.pt, cf_meta_chunk_NNN.jsonl)
    data/emotion/content_factorization_gemma/
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CONFIGS = {
    "qwen": {
        "model_id":     "Qwen/Qwen2.5-7B-Instruct",
        "system_msg":   "You are a helpful assistant.",
        "prompts_path": r"G:\LLM\experiment\data\emotion\cross_arch_prompts_qwen.json",
        "out_dir":      r"G:\LLM\experiment\data\emotion\content_factorization_qwen",
    },
    "gemma": {
        "model_id":     "google/gemma-2-9b-it",
        "system_msg":   None,
        "prompts_path": r"G:\LLM\experiment\data\emotion\cross_arch_prompts_gemma.json",
        "out_dir":      r"G:\LLM\experiment\data\emotion\content_factorization_gemma",
    },
}

CHUNK_SIZE = 10


def load_model(model_id):
    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def extract_hs(model, tok, prompt_text, system_msg):
    if system_msg is not None:
        messages = [{"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt_text}]
    else:
        messages = [{"role": "user", "content": prompt_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs   # [n_layers, hidden_dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    with open(cfg["prompts_path"], encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Loaded {len(records)} records for {args.model}.")

    model, tok = load_model(cfg["model_id"])

    all_hs, all_meta = [], []
    chunk_idx = 0

    for i, rec in enumerate(records):
        hs = extract_hs(model, tok, rec["prompt_text"], cfg["system_msg"])
        all_hs.append(hs)
        all_meta.append({
            "task_id":     rec["task_id"],
            "pair_id":     rec["pair_id"],
            "subcategory": rec["subcategory"],
            "direction":   rec["direction"],
            "task_type":   rec["task_type"],
        })
        print(f"  {i+1:3d}/{len(records)}  {rec['task_id']}")

        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == len(records):
            batch = all_hs[chunk_idx * CHUNK_SIZE:]
            X_chunk = torch.stack(batch)
            pt_path   = os.path.join(cfg["out_dir"], f"cf_hidden_chunk_{chunk_idx:03d}.pt")
            meta_path = os.path.join(cfg["out_dir"], f"cf_meta_chunk_{chunk_idx:03d}.jsonl")
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
