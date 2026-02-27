import os, json, math, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-2-9b-it"

TASKS_PATH = r"G:\LLM\experiment\data\tasks_v2_hard.json"
OUT_DIR = r"G:\LLM\experiment\data\scale_runs_gemma"
REPEATS = 3          # start small (we’ll set to 10 later)
CHUNK_SIZE = 12      # save every N examples (prevents RAM pain)

def load_tasks():
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = load_tasks()
    print("Tasks:", len(tasks), "Repeats:", REPEATS, "Chunk size:", CHUNK_SIZE)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    chunk_tensors = []
    chunk_meta = []
    saved = 0
    total = len(tasks) * REPEATS

    def flush_chunk(chunk_idx):
        nonlocal saved, chunk_tensors, chunk_meta
        if not chunk_tensors:
            return
        batch = torch.stack(chunk_tensors)  # [N, layers, hidden]
        pt_path = os.path.join(OUT_DIR, f"hidden_chunk_{chunk_idx:03d}.pt")
        meta_path = os.path.join(OUT_DIR, f"meta_chunk_{chunk_idx:03d}.jsonl")
        torch.save(batch, pt_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in chunk_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
        saved += len(chunk_tensors)
        chunk_tensors = []
        chunk_meta = []

    chunk_idx = 0
    with torch.no_grad():
        for r in range(REPEATS):
            for t in tasks:
                # Build prompt
                messages = [
                {"role": "user", "content": t["task_prompt"]},
                ]
                
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to("cuda")

                # Forward pass, grab hidden states
                out = model(**inputs, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                last_pos = inputs["input_ids"].shape[1] - 1

                per_layer_last = torch.stack([h[0, last_pos, :].float().cpu() for h in hs])
                chunk_tensors.append(per_layer_last)

                chunk_meta.append({
                    "model": MODEL_ID,
                    "repeat_index": r,
                    "task_id": t["task_id"],
                    "label": t["label"],
                    "task_prompt": t["task_prompt"],
                    "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
                })

                done = saved + len(chunk_tensors)
                if done % 5 == 0:
                    print(f"Progress: {done}/{total}")

                if len(chunk_tensors) >= CHUNK_SIZE:
                    flush_chunk(chunk_idx)
                    chunk_idx += 1

    flush_chunk(chunk_idx)
    print("DONE. Total saved:", saved)

if __name__ == "__main__":
    main()