import os, json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick ONE model for the first smoke test (fast + reliable)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

OUT_LOG = r"G:\LLM\experiment\logs\smoke_test.jsonl"

def main():
    print("HF_HOME =", os.environ.get("HF_HOME"))
    print("Loading:", MODEL_ID)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    load_s = time.time() - t0
    print(f"Loaded in {load_s:.1f}s")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say a single creepy sentence about a tower that eats memories."},
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    gen_t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    gen_s = time.time() - gen_t0

    text = tok.decode(out[0], skip_special_tokens=True)
    record = {
        "model": MODEL_ID,
        "load_seconds": load_s,
        "gen_seconds": gen_s,
        "prompt": prompt,
        "output": text,
    }

    os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved log to:", OUT_LOG)
    print("\n--- OUTPUT ---\n")
    print(text)

if __name__ == "__main__":
    main()