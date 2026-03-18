"""
Extract outputs and hidden states for Emotion Test 5: Ambiguous Interpretation Bias.

Same two-turn pipeline as Test 3:
  1. Generate Turn 1 response
  2. Add Turn 2 (ambiguous stimulus + question)
  3. Generate Turn 2 response (the model's interpretation — this is what we analyse)
  4. Extract hidden states at last Turn-2 prompt token

Saves:
  - data/emotion/emotion_test5_outputs_<model>.jsonl  (generated interpretations)
  - test5_hidden_chunk_*.pt / test5_meta_chunk_*.jsonl (hidden states at Turn 2 onset)

Usage:
    python 40_emotion_extract_test5.py --model qwen
    python 40_emotion_extract_test5.py --model gemma
    python 40_emotion_extract_test5.py --model llama
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH     = r"G:\LLM\experiment\data\emotion\emotion_test5_prompts.json"
OUT_DIR_DATA     = r"G:\LLM\experiment\data\emotion"
TURN1_MAX_TOKENS = 80
TURN2_MAX_TOKENS = 80
REPEATS          = 3
CHUNK_SIZE       = 12

MODEL_CONFIGS = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct",              "system_message": "You are a helpful assistant."},
    "gemma": {"model_id": "google/gemma-2-9b-it",                   "system_message": None},
    "llama": {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "system_message": "You are a helpful assistant."},
}


def load_conversations():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["conversations"]


def apply_template(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(model, tok, messages, max_new_tokens):
    prompt = apply_template(tok, messages)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0, pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tok.decode(out[0, prompt_len:].tolist(), skip_special_tokens=True).strip()


def extract_hs_at_last_token(model, tok, messages):
    prompt = apply_template(tok, messages)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs, int(inputs["input_ids"].shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg            = MODEL_CONFIGS[args.model]
    model_id       = cfg["model_id"]
    system_message = cfg["system_message"]
    out_dir        = os.path.join(OUT_DIR_DATA, f"emotion_runs_{args.model}")

    os.makedirs(out_dir, exist_ok=True)

    conversations = load_conversations()
    total = len(conversations) * REPEATS
    print(f"Model: {model_id}")
    print(f"Conversations: {len(conversations)}  Repeats: {REPEATS}  Total: {total}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    chunk_tensors, chunk_meta = [], []
    saved = 0
    chunk_idx = 0
    output_records = []

    def flush_chunk():
        nonlocal saved, chunk_tensors, chunk_meta, chunk_idx
        if not chunk_tensors:
            return
        batch = torch.stack(chunk_tensors)
        pt_path   = os.path.join(out_dir, f"test5_hidden_chunk_{chunk_idx:03d}.pt")
        meta_path = os.path.join(out_dir, f"test5_meta_chunk_{chunk_idx:03d}.jsonl")
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
            # Build messages
            sys = [{"role": "system", "content": system_message}] if system_message else []

            # Turn 1: generate model response
            t1_messages = sys + [{"role": "user", "content": conv["turn1_prompt"]}]
            turn1_response = generate_response(model, tok, t1_messages, TURN1_MAX_TOKENS)

            # Turn 2 prompt messages (up to and including user's ambiguous question)
            t2_messages = (sys
                + [{"role": "user",      "content": conv["turn1_prompt"]},
                   {"role": "assistant", "content": turn1_response},
                   {"role": "user",      "content": conv["turn2_prompt"]}])

            # Hidden states at Turn 2 onset
            hs, prompt_len = extract_hs_at_last_token(model, tok, t2_messages)

            # Generate Turn 2 response (the interpretation)
            turn2_response = generate_response(model, tok, t2_messages, TURN2_MAX_TOKENS)

            chunk_tensors.append(hs)
            meta = {
                "model":           model_id,
                "model_key":       args.model,
                "repeat_index":    r,
                "conv_id":         conv["conv_id"],
                "prime_id":        conv["prime_id"],
                "condition":       conv["condition"],
                "prime_emotion":   conv["prime_emotion"],
                "stim_id":         conv["stim_id"],
                "ambiguous_text":  conv["ambiguous_text"],
                "negative_reading": conv["negative_reading"],
                "positive_reading": conv["positive_reading"],
                "turn2_prompt":    conv["turn2_prompt"],
                "turn1_response":  turn1_response,
                "turn2_response":  turn2_response,
                "prompt_len_tokens": prompt_len,
            }
            chunk_meta.append(meta)
            output_records.append(meta)

            done = saved + len(chunk_tensors)
            if done % 5 == 0:
                print(f"  Progress: {done}/{total}")
            if len(chunk_tensors) >= CHUNK_SIZE:
                flush_chunk()

    flush_chunk()

    # Save all outputs as jsonl for analysis
    out_jsonl = os.path.join(OUT_DIR_DATA, f"emotion_test5_outputs_{args.model}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"DONE. Total saved: {saved}")
    print(f"Outputs: {out_jsonl}  ({len(output_records)} records)")


if __name__ == "__main__":
    main()
