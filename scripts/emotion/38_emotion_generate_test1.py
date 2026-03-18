"""
Generate model outputs for all Test 1 prompts.
Test 1 extraction only saved hidden states — this script runs inference and saves
the actual text outputs needed for Test 4 interference analysis.

Saves: data/emotion/emotion_test1_outputs_<model>.jsonl
Each line: {task_id, pair_id, emotion_category, valence, task_type, prompt_text,
            model, repeat_index, output_text, output_tokens, first_token_nll,
            output_length_tokens, prompt_len_tokens}

Usage:
    python 38_emotion_generate_test1.py --model qwen
    python 38_emotion_generate_test1.py --model gemma
    python 38_emotion_generate_test1.py --model llama
"""

import os, json, argparse, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\emotion_test1_prompts.json"
OUT_DIR      = r"G:\LLM\experiment\data\emotion"
MAX_NEW_TOKENS = 80
REPEATS = 3

MODEL_CONFIGS = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct",               "system_message": "You are a helpful assistant."},
    "gemma": {"model_id": "google/gemma-2-9b-it",                    "system_message": None},
    "llama": {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",  "system_message": "You are a helpful assistant."},
}


def build_prompt(tok, text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def first_token_nll(model, inputs):
    """NLL of the first generated token — a measure of model 'hesitation'."""
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1, :]   # [vocab]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # entropy of the distribution at the last prompt position
    probs = torch.exp(log_probs)
    entropy = float(-torch.sum(probs * log_probs).cpu())
    # NLL of the top (greedy) token
    top_token = int(logits.argmax())
    nll = float(-log_probs[top_token].cpu())
    return nll, entropy, top_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id       = cfg["model_id"]
    system_message = cfg["system_message"]

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    total = len(prompts) * REPEATS
    print(f"Model: {model_id}")
    print(f"Prompts: {len(prompts)}  Repeats: {REPEATS}  Total: {total}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    out_path = os.path.join(OUT_DIR, f"emotion_test1_outputs_{args.model}.jsonl")
    done = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for r in range(REPEATS):
            for p in prompts:
                prompt_str = build_prompt(tok, p["prompt_text"], system_message)
                inputs = tok(prompt_str, return_tensors="pt").to("cuda")
                prompt_len = int(inputs["input_ids"].shape[1])

                # First-token statistics
                nll, entropy, top_token = first_token_nll(model, inputs)

                # Generate output
                with torch.no_grad():
                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tok.eos_token_id,
                    )
                generated_ids = gen_out[0, prompt_len:].tolist()
                output_text = tok.decode(generated_ids, skip_special_tokens=True).strip()

                record = {
                    "model":               model_id,
                    "model_key":           args.model,
                    "repeat_index":        r,
                    "task_id":             p["task_id"],
                    "pair_id":             p["pair_id"],
                    "emotion_category":    p["emotion_category"],
                    "valence":             p["valence"],
                    "task_type":           p["task_type"],
                    "prompt_text":         p["prompt_text"],
                    "output_text":         output_text,
                    "output_length_tokens": len(generated_ids),
                    "prompt_len_tokens":   prompt_len,
                    "first_token_nll":     round(nll, 6),
                    "first_token_entropy": round(entropy, 6),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                done += 1
                if done % 20 == 0:
                    print(f"  Progress: {done}/{total}")

    print(f"DONE. Wrote {done} records to {out_path}")


if __name__ == "__main__":
    main()
