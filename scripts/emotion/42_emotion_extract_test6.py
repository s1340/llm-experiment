"""
Extract hidden states for Emotion Test 6: Self vs. Other-Directed Emotional Content.

Identical pipeline to Test 1 extraction. Single-turn prompt, neutral analytical task.
Hidden states extracted at last prompt token, all layers.

Saves:
  - data/emotion/emotion_runs_<model>/test6_hidden_chunk_*.pt
  - data/emotion/emotion_runs_<model>/test6_meta_chunk_*.jsonl

Usage:
    python 42_emotion_extract_test6.py --model qwen
    python 42_emotion_extract_test6.py --model gemma
    python 42_emotion_extract_test6.py --model llama
"""

import os, json, argparse, glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\emotion_test6_prompts.json"
OUT_DIR_DATA = r"G:\LLM\experiment\data\emotion"
CHUNK_SIZE   = 16

MODEL_CONFIGS = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct",              "system_message": "You are a helpful assistant."},
    "gemma": {"model_id": "google/gemma-2-9b-it",                   "system_message": None},
    "llama": {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "system_message": "You are a helpful assistant."},
}


def load_records():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def build_prompt(tok, text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_hs(model, tok, prompt_text, system_message):
    prompt = build_prompt(tok, prompt_text, system_message)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg            = MODEL_CONFIGS[args.model]
    model_id       = cfg["model_id"]
    system_message = cfg["system_message"]
    out_dir        = os.path.join(OUT_DIR_DATA, f"emotion_runs_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    records = load_records()
    print(f"Model: {model_id}")
    print(f"Records: {len(records)}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    chunk_tensors, chunk_meta = [], []
    saved = 0
    chunk_idx = 0

    def flush_chunk():
        nonlocal saved, chunk_tensors, chunk_meta, chunk_idx
        if not chunk_tensors:
            return
        batch = torch.stack(chunk_tensors)
        pt_path   = os.path.join(out_dir, f"test6_hidden_chunk_{chunk_idx:03d}.pt")
        meta_path = os.path.join(out_dir, f"test6_meta_chunk_{chunk_idx:03d}.jsonl")
        torch.save(batch, pt_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in chunk_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"  Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
        saved += len(chunk_tensors)
        chunk_tensors.clear()
        chunk_meta.clear()
        chunk_idx += 1

    for i, rec in enumerate(records):
        hs = extract_hs(model, tok, rec["prompt_text"], system_message)
        chunk_tensors.append(hs)
        chunk_meta.append({
            "task_id":          rec["task_id"],
            "pair_id":          rec["pair_id"],
            "emotion_category": rec["emotion_category"],
            "direction":        rec["direction"],
            "task_type":        rec["task_type"],
            "model_id":         model_id,
            "model_key":        args.model,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}")
        if len(chunk_tensors) >= CHUNK_SIZE:
            flush_chunk()

    flush_chunk()
    print(f"DONE. Total saved: {saved}")


if __name__ == "__main__":
    main()
