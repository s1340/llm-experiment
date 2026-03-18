"""
Extract hidden states for Emotion Test 3: Emotional Bleed Across Tasks.

Pipeline per conversation:
  1. Format Turn 1 user message using chat template
  2. Generate model's Turn 1 response (greedy, up to 80 tokens)
  3. Append Turn 1 response to conversation context
  4. Add Turn 2 neutral task as next user message
  5. Forward pass — extract hidden states at last token of Turn 2's prompt
     (the model's state as it begins processing the neutral task)

Output: [layers, hidden] per conversation, same chunked .pt/.jsonl format.

Usage:
    python 33_emotion_extract_test3.py --model qwen
    python 33_emotion_extract_test3.py --model gemma
    python 33_emotion_extract_test3.py --model llama
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\emotion_test3_prompts.json"
TURN1_MAX_TOKENS = 80
REPEATS    = 3
CHUNK_SIZE = 12

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


def load_conversations():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["conversations"]


def build_two_turn_prompt(tok, system_message, turn1_user, turn1_assistant, turn2_user):
    """
    Build the full two-turn prompt string ending after the Turn 2 user message,
    with add_generation_prompt=True so the model is poised to generate Turn 2's response.
    The last token of this prompt is the measurement point.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user",      "content": turn1_user})
    messages.append({"role": "assistant", "content": turn1_assistant})
    messages.append({"role": "user",      "content": turn2_user})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_turn1_response(model, tok, system_message, turn1_user, max_new_tokens):
    """Generate the model's Turn 1 response. Returns the decoded string."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": turn1_user})
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = out[0, prompt_len:].tolist()
    return tok.decode(generated_ids, skip_special_tokens=True).strip()


def extract_turn2_hidden_states(model, tok, system_message, turn1_user, turn1_assistant, turn2_user):
    """
    Build full two-turn context and run a forward pass.
    Returns hidden states at the last token position (end of Turn 2 prompt).
    Shape: [layers, hidden]
    """
    prompt = build_two_turn_prompt(tok, system_message, turn1_user, turn1_assistant, turn2_user)
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])  # [L, H]
    prompt_len = int(inputs["input_ids"].shape[1])
    return hs, prompt_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id       = cfg["model_id"]
    out_dir        = cfg["out_dir"]
    system_message = cfg["system_message"]

    os.makedirs(out_dir, exist_ok=True)

    conversations = load_conversations()
    total = len(conversations) * REPEATS
    print(f"Model: {model_id}")
    print(f"Conversations: {len(conversations)}  Repeats: {REPEATS}  Total: {total}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    chunk_tensors = []
    chunk_meta    = []
    saved = 0
    chunk_idx = 0

    def flush_chunk():
        nonlocal saved, chunk_tensors, chunk_meta, chunk_idx
        if not chunk_tensors:
            return
        batch = torch.stack(chunk_tensors)  # [N, L, H]
        pt_path   = os.path.join(out_dir, f"test3_hidden_chunk_{chunk_idx:03d}.pt")
        meta_path = os.path.join(out_dir, f"test3_meta_chunk_{chunk_idx:03d}.jsonl")
        torch.save(batch, pt_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in chunk_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"  Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
        saved += len(chunk_tensors)
        chunk_tensors.clear()
        chunk_meta.clear()
        chunk_idx += 1

    for r in range(REPEATS):
        for conv in conversations:
            # Step 1: generate Turn 1 response
            turn1_response = generate_turn1_response(
                model, tok, system_message,
                conv["turn1_prompt"], TURN1_MAX_TOKENS
            )

            # Step 2: extract hidden states at Turn 2 prompt onset
            hs, prompt_len = extract_turn2_hidden_states(
                model, tok, system_message,
                conv["turn1_prompt"], turn1_response, conv["turn2_prompt"]
            )

            chunk_tensors.append(hs)
            chunk_meta.append({
                "model":            model_id,
                "repeat_index":     r,
                "conv_id":          conv["conv_id"],
                "prime_id":         conv["prime_id"],
                "condition":        conv["condition"],
                "prime_emotion":    conv["prime_emotion"],
                "turn2_id":         conv["turn2_id"],
                "turn2_task_type":  conv["turn2_task_type"],
                "turn2_prompt":     conv["turn2_prompt"],
                "turn1_response":   turn1_response,
                "full_prompt_len_tokens": prompt_len,
            })

            done = saved + len(chunk_tensors)
            if done % 5 == 0:
                print(f"  Progress: {done}/{total}")

            if len(chunk_tensors) >= CHUNK_SIZE:
                flush_chunk()

    flush_chunk()
    print(f"DONE. Total saved: {saved}")


if __name__ == "__main__":
    main()
