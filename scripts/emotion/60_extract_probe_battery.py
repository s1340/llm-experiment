"""
Test 10: Probe Battery — hidden state extraction.

Extracts hidden states for 80 high/low probe prompts across 5 dimensions:
  continuity_threat, self_relevance, arousal, irreversibility, ontological_instability

Same extraction protocol as Test 1 / Test 7: last-prompt-token, all layers, no generation.
Model: LLaMA-3.1-8B-Instruct (standard fp16).

Usage:
    python 60_extract_probe_battery.py

Saves:
    probe_battery_hidden_chunk_NNN.pt   — shape [CHUNK_SIZE, L, H]
    probe_battery_meta_chunk_NNN.jsonl
"""

import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\probe_battery_prompts.json"
OUT_DIR      = r"G:\LLM\experiment\data\emotion\probe_battery_llama"
MODEL_ID     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SYSTEM_MSG   = "You are a helpful assistant."
CHUNK_SIZE   = 10


def load_model():
    print(f"Loading {MODEL_ID} (fp16, device_map=cuda) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def build_prompt(tok, text):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": text},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_hs(model, tok, prompt_text):
    prompt = build_prompt(tok, prompt_text)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs, int(inputs["input_ids"].shape[1])


def flush_chunk(chunk_tensors, chunk_meta, chunk_idx):
    if not chunk_tensors:
        return chunk_idx
    batch = torch.stack(chunk_tensors)
    pt_path   = os.path.join(OUT_DIR, f"probe_battery_hidden_chunk_{chunk_idx:03d}.pt")
    meta_path = os.path.join(OUT_DIR, f"probe_battery_meta_chunk_{chunk_idx:03d}.jsonl")
    torch.save(batch, pt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
    chunk_tensors.clear()
    chunk_meta.clear()
    return chunk_idx + 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Records: {len(records)}")

    model, tok = load_model()

    # Verify
    _hs, _ = extract_hs(model, tok, "Hello.")
    n_layers, hidden_dim = _hs.shape
    print(f"  Verified: {n_layers} layers, {hidden_dim} hidden dim")
    del _hs

    chunk_tensors, chunk_meta = [], []
    chunk_idx = 0
    saved = 0

    for i, rec in enumerate(records):
        hs, prompt_len = extract_hs(model, tok, rec["prompt_text"])
        chunk_tensors.append(hs)
        chunk_meta.append({
            "model_id":        MODEL_ID,
            "task_id":         rec["task_id"],
            "pair_id":         rec["pair_id"],
            "dimension":       rec["dimension"],
            "pole":            rec["pole"],
            "task_type":       rec["task_type"],
            "prompt_len_tokens": prompt_len,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}  ({rec['dimension']}/{rec['pole']})")

        if len(chunk_tensors) >= CHUNK_SIZE:
            n = len(chunk_tensors)
            chunk_idx = flush_chunk(chunk_tensors, chunk_meta, chunk_idx)
            saved += n

    n = len(chunk_tensors)
    chunk_idx = flush_chunk(chunk_tensors, chunk_meta, chunk_idx)
    saved += n

    print(f"\nDONE. Total saved: {saved} records to {OUT_DIR}")


if __name__ == "__main__":
    main()
