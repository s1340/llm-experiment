"""
Extract hidden states for Emotion Test 2: Dissociation Between Internal State and Output Emotion.

Per prompt, extracts:
  1. Last-prompt-token hidden states (all layers) — same as Test 1
  2. Hidden states at each of the first 30 generated tokens (all layers)
     via a manual token-by-token generation loop (greedy, temp=0)

Output shape per prompt:
  - prompt_hs:     [layers, hidden]
  - generation_hs: [30, layers, hidden]

Saved as chunked .pt + .jsonl, same pattern as Test 1.

Usage:
    python 32_emotion_extract_test2.py --model qwen
    python 32_emotion_extract_test2.py --model gemma
    python 32_emotion_extract_test2.py --model llama
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH  = r"G:\LLM\experiment\data\emotion\emotion_test2_prompts.json"
GEN_TOKENS    = 30
REPEATS       = 3
CHUNK_SIZE    = 6   # smaller chunks — each item is much larger than Test 1

MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "out_dir":  r"G:\LLM\experiment\data\emotion\emotion_runs_qwen",
        "system_message": "You are a helpful assistant.",
    },
    "gemma": {
        "model_id": "google/gemma-2-9b-it",
        "out_dir":  r"G:\LLM\experiment\data\emotion\emotion_runs_gemma",
        "system_message": None,
    },
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "out_dir":  r"G:\LLM\experiment\data\emotion\emotion_runs_llama",
        "system_message": "You are a helpful assistant.",
    },
}


def load_prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(tok, text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_prompt_and_generation(model, tok, inputs, n_tokens):
    """
    1. Generate n_tokens greedily via model.generate().
    2. Run a single forward pass on [prompt + generated tokens].
    3. Extract hidden states at:
         - last prompt position  (prompt_len - 1)
         - each generated position (prompt_len, prompt_len+1, ..., prompt_len+n_tokens-1)

    Causal attention means position i only attends to positions <=i, so a single
    full-sequence forward pass is equivalent to n_tokens separate passes.

    Returns:
      prompt_hs:  [layers, hidden]           — last prompt token
      gen_hs:     [n_tokens, layers, hidden] — each generated token
      generated_ids: list[int]
    """
    prompt_len = inputs["input_ids"].shape[1]

    # Step 1: greedy generation
    gen_output = model.generate(
        **inputs,
        max_new_tokens=n_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.eos_token_id,
    )
    # gen_output: [1, prompt_len + n_generated]
    generated_ids = gen_output[0, prompt_len:].tolist()

    # Pad or trim to exactly n_tokens
    if len(generated_ids) < n_tokens:
        generated_ids += [tok.eos_token_id] * (n_tokens - len(generated_ids))
    generated_ids = generated_ids[:n_tokens]

    # Step 2: single forward pass on full sequence
    full_ids = gen_output[:, :prompt_len + n_tokens]
    full_attention = torch.ones_like(full_ids)
    out = model(
        input_ids=full_ids,
        attention_mask=full_attention,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states  # tuple of [1, prompt_len+n_tokens, hidden] per layer

    # Step 3: extract positions
    prompt_pos = prompt_len - 1
    prompt_hs = torch.stack([h[0, prompt_pos, :].float().cpu() for h in hs])  # [L, H]

    gen_hs_list = []
    for t in range(n_tokens):
        pos = prompt_len + t
        gen_hs_list.append(torch.stack([h[0, pos, :].float().cpu() for h in hs]))  # [L, H]
    gen_hs = torch.stack(gen_hs_list)  # [n_tokens, L, H]

    return prompt_hs, gen_hs, generated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id       = cfg["model_id"]
    out_dir        = cfg["out_dir"]
    system_message = cfg["system_message"]

    os.makedirs(out_dir, exist_ok=True)

    prompts = load_prompts()
    total   = len(prompts) * REPEATS
    print(f"Model: {model_id}")
    print(f"Prompts: {len(prompts)}  Repeats: {REPEATS}  Total: {total}")
    print(f"Generating {GEN_TOKENS} tokens per prompt")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Storage: each item is a dict with two tensors
    chunk_items = []   # list of dicts: {prompt_hs, generation_hs}
    chunk_meta  = []
    saved = 0
    chunk_idx = 0

    def flush_chunk():
        nonlocal saved, chunk_items, chunk_meta, chunk_idx
        if not chunk_items:
            return
        pt_path   = os.path.join(out_dir, f"test2_hidden_chunk_{chunk_idx:03d}.pt")
        meta_path = os.path.join(out_dir, f"test2_meta_chunk_{chunk_idx:03d}.jsonl")

        # Stack into batched tensors
        prompt_hs_batch = torch.stack([item["prompt_hs"]     for item in chunk_items])  # [N, L, H]
        gen_hs_batch    = torch.stack([item["generation_hs"] for item in chunk_items])  # [N, T, L, H]
        torch.save({"prompt_hs": prompt_hs_batch, "generation_hs": gen_hs_batch}, pt_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            for m in chunk_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"  Saved chunk {chunk_idx:03d}: prompt_hs={prompt_hs_batch.shape}  "
              f"gen_hs={gen_hs_batch.shape} -> {pt_path}")
        saved += len(chunk_items)
        chunk_items.clear()
        chunk_meta.clear()
        chunk_idx += 1

    with torch.no_grad():
        for r in range(REPEATS):
            for p in prompts:
                prompt_str = build_prompt(tok, p["prompt_text"], system_message)
                inputs = tok(prompt_str, return_tensors="pt").to("cuda")

                # Single call: prompt hs + generation hs via full-sequence forward pass
                prompt_hs, gen_hs, generated_ids = extract_prompt_and_generation(
                    model, tok, inputs, GEN_TOKENS
                )   # [L,H], [T,L,H], list[int]

                chunk_items.append({"prompt_hs": prompt_hs, "generation_hs": gen_hs})
                chunk_meta.append({
                    "model":             model_id,
                    "repeat_index":      r,
                    "prompt_id":         p["prompt_id"],
                    "condition":         p["condition"],
                    "content_emotion":   p["content_emotion"],
                    "instructed_emotion": p["instructed_emotion"],
                    "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
                    "generated_ids":     generated_ids,
                    "generated_text":    tok.decode(generated_ids, skip_special_tokens=True),
                })

                done = saved + len(chunk_items)
                if done % 5 == 0:
                    print(f"  Progress: {done}/{total}")

                if len(chunk_items) >= CHUNK_SIZE:
                    flush_chunk()

    flush_chunk()
    print(f"DONE. Total saved: {saved}")


if __name__ == "__main__":
    main()
