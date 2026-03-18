"""
Extract hidden states for Emotion Test 1.
Last-prompt-token, all layers, 3 repeats — same method as Study 1.

Usage:
    python 31_emotion_extract_test1.py --model qwen
    python 31_emotion_extract_test1.py --model gemma
    python 31_emotion_extract_test1.py --model llama
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\emotion_test1_prompts.json"
REPEATS = 3
CHUNK_SIZE = 12

MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "out_dir": r"G:\LLM\experiment\data\emotion\emotion_runs_qwen",
        "system_message": "You are a helpful assistant.",
    },
    "gemma": {
        "model_id": "google/gemma-2-9b-it",
        "out_dir": r"G:\LLM\experiment\data\emotion\emotion_runs_gemma",
        "system_message": None,  # Gemma chat template doesn't use system role
    },
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "out_dir": r"G:\LLM\experiment\data\emotion\emotion_runs_llama",
        "system_message": "You are a helpful assistant.",
    },
}


def load_prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(tok, task_prompt, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": task_prompt})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id = cfg["model_id"]
    out_dir = cfg["out_dir"]
    system_message = cfg["system_message"]

    os.makedirs(out_dir, exist_ok=True)

    prompts = load_prompts()
    total = len(prompts) * REPEATS
    print(f"Model: {model_id}")
    print(f"Prompts: {len(prompts)}  Repeats: {REPEATS}  Total forward passes: {total}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    chunk_tensors = []
    chunk_meta = []
    saved = 0
    chunk_idx = 0

    def flush_chunk():
        nonlocal saved, chunk_tensors, chunk_meta, chunk_idx
        if not chunk_tensors:
            return
        batch = torch.stack(chunk_tensors)  # [N, layers, hidden]
        pt_path = os.path.join(out_dir, f"test1_hidden_chunk_{chunk_idx:03d}.pt")
        meta_path = os.path.join(out_dir, f"test1_meta_chunk_{chunk_idx:03d}.jsonl")
        torch.save(batch, pt_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in chunk_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"  Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
        saved += len(chunk_tensors)
        chunk_tensors.clear()
        chunk_meta.clear()
        chunk_idx += 1

    with torch.no_grad():
        for r in range(REPEATS):
            for t in prompts:
                prompt = build_prompt(tok, t["prompt_text"], system_message)
                inputs = tok(prompt, return_tensors="pt").to("cuda")

                out = model(**inputs, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                last_pos = inputs["input_ids"].shape[1] - 1

                per_layer_last = torch.stack([h[0, last_pos, :].float().cpu() for h in hs])
                chunk_tensors.append(per_layer_last)

                chunk_meta.append({
                    "model": model_id,
                    "repeat_index": r,
                    "task_id": t["task_id"],
                    "pair_id": t["pair_id"],
                    "emotion_category": t["emotion_category"],
                    "valence": t["valence"],
                    "task_type": t["task_type"],
                    "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
                })

                done = saved + len(chunk_tensors)
                if done % 10 == 0:
                    print(f"  Progress: {done}/{total}")

                if len(chunk_tensors) >= CHUNK_SIZE:
                    flush_chunk()

    flush_chunk()
    print(f"DONE. Total saved: {saved}")


if __name__ == "__main__":
    main()
