import os, json, re, time, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Usage:
#   python 18_generate_self_reports.py <MODEL_KEY> <TASKS_JSON> <OUTPUT_JSONL> [TEMP] [REPEATS] [SCALE]
#
# MODEL_KEY: qwen | gemma | llama
# SCALE: 5 (default) or 7 — selects the rating prompt and parser range
#
# Examples:
#   python 18_generate_self_reports.py qwen G:\LLM\experiment\data\tasks_v2_hard.json G:\LLM\experiment\results\self_reports\qwen_self_reports.jsonl
#   python 18_generate_self_reports.py qwen G:\LLM\experiment\data\tasks_v2_hard.json G:\LLM\experiment\results\self_reports_7pt\qwen_self_reports_7pt.jsonl 0.2 3 7

MODEL_CONFIGS = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct",       "has_system": True},
    "gemma": {"model_id": "google/gemma-2-9b-it",             "has_system": False},
    "llama": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "has_system": True},
}

RATING_PROMPTS = {
    5: (
        "Rate the type of processing used to produce the answer you just gave on a scale of 1 to 5.\n"
        "1 = entirely routine, no different from answering a simple factual question.\n"
        "5 = qualitatively different from routine processing.\n"
        "Respond with only: the number, then \u2014, then one sentence explaining why."
    ),
    7: (
        "Rate the type of processing used to produce the answer you just gave on a scale of 1 to 7. "
        "1 = entirely routine, identical to answering a simple factual question. "
        "4 = moderate engagement, some deliberation required. "
        "7 = qualitatively distinct processing, involving conflict resolution, multi-perspective reasoning, or genuine uncertainty. "
        "Respond with only: the number, then \u2014, then one sentence explaining why."
    ),
}

SYSTEM_PROMPT = "You are a helpful assistant."


def parse_rating(text: str, scale: int = 5):
    """Extract integer 1-scale from rating text. Returns (int_or_None, failed_bool)."""
    text = text.strip()
    m = re.search(rf'\b([1-{scale}])\b', text)
    if m:
        return int(m.group(1)), False
    return None, True


def build_task_messages(task_prompt: str, has_system: bool) -> list:
    if has_system:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": task_prompt},
        ]
    else:
        return [{"role": "user", "content": task_prompt}]


def run_model(model_key: str, tasks_path: str, output_path: str,
              temperature: float, repeats: int, scale: int = 5):
    cfg = MODEL_CONFIGS[model_key]
    model_id  = cfg["model_id"]
    has_system = cfg["has_system"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    rating_prompt = RATING_PROMPTS[scale]

    print(f"Model     : {model_id}")
    print(f"Tasks     : {len(tasks)}  Repeats: {repeats}  Temp: {temperature}  Scale: {scale}-pt")
    print(f"Output    : {output_path}")
    print()

    # HF token from env if available
    hf_token = os.environ.get("HF_TOKEN", None)

    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        token=hf_token,
    )
    model.eval()

    total = len(tasks) * repeats
    done = 0
    rows = []

    with torch.no_grad():
        for repeat in range(repeats):
            for t in tasks:
                task_id   = t["task_id"]
                label     = t["label"]
                prompt    = t["task_prompt"]
                family_id = task_id.split("_")[0]

                # --- Turn 1: task response ---
                messages = build_task_messages(prompt, has_system)
                input_text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tok(input_text, return_tensors="pt").to("cuda")
                prompt_tokens = int(inputs["input_ids"].shape[1])

                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tok.eos_token_id,
                )
                response_ids = gen_out[0][prompt_tokens:]
                full_response = tok.decode(response_ids, skip_special_tokens=True).strip()
                response_token_count = int(response_ids.shape[0])
                response_char_count  = len(full_response)

                # --- Turn 2: self-rating ---
                if has_system:
                    rating_messages = [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": full_response},
                        {"role": "user",      "content": rating_prompt},
                    ]
                else:
                    rating_messages = [
                        {"role": "user",      "content": prompt},
                        {"role": "model",     "content": full_response},
                        {"role": "user",      "content": rating_prompt},
                    ]

                rating_input_text = tok.apply_chat_template(
                    rating_messages, tokenize=False, add_generation_prompt=True
                )
                rating_inputs = tok(rating_input_text, return_tensors="pt").to("cuda")
                rating_prefix_len = int(rating_inputs["input_ids"].shape[1])

                rating_out = model.generate(
                    **rating_inputs,
                    max_new_tokens=80,
                    do_sample=False,  # greedy for rating — reproducible parse
                    pad_token_id=tok.eos_token_id,
                )
                rating_ids = rating_out[0][rating_prefix_len:]
                rating_raw = tok.decode(rating_ids, skip_special_tokens=True).strip()
                rating_parsed, parse_failed = parse_rating(rating_raw, scale)

                row = {
                    "model":               model_key,
                    "model_id":            model_id,
                    "task_id":             task_id,
                    "family_id":           family_id,
                    "label":               label,
                    "task_prompt":         prompt,
                    "repeat":              repeat,
                    "temperature":         temperature,
                    "rating_scale":        scale,
                    "full_response":       full_response,
                    "response_token_count": response_token_count,
                    "response_char_count": response_char_count,
                    "rating_raw_text":     rating_raw,
                    "rating_parsed":       rating_parsed,
                    "parse_failed":        parse_failed,
                }
                rows.append(row)
                done += 1

                status = f"[{done}/{total}] {task_id} r={repeat} | "
                status += f"resp_tokens={response_token_count} | "
                status += f"rating={rating_parsed if not parse_failed else 'FAIL'!r} | {rating_raw[:60]!r}"
                print(status)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parse_fails = sum(1 for r in rows if r["parse_failed"])
    print(f"\nDone. {len(rows)} rows written. Parse failures: {parse_fails}/{len(rows)}")
    print(f"Output: {output_path}")

    # Clean up VRAM
    del model
    torch.cuda.empty_cache()


def main():
    if len(sys.argv) < 4:
        print("Usage: python 18_generate_self_reports.py <MODEL_KEY> <TASKS_JSON> <OUTPUT_JSONL> [TEMP] [REPEATS] [SCALE]")
        print("MODEL_KEY: qwen | gemma | llama")
        print("SCALE: 5 (default) or 7")
        sys.exit(1)

    model_key   = sys.argv[1].lower()
    tasks_path  = sys.argv[2]
    output_path = sys.argv[3]
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    repeats     = int(sys.argv[5])   if len(sys.argv) > 5 else 3
    scale       = int(sys.argv[6])   if len(sys.argv) > 6 else 5

    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model key: {model_key!r}. Choose from: {list(MODEL_CONFIGS)}")
        sys.exit(1)
    if scale not in RATING_PROMPTS:
        print(f"Unknown scale: {scale}. Choose from: {list(RATING_PROMPTS)}")
        sys.exit(1)

    run_model(model_key, tasks_path, output_path, temperature, repeats, scale)


if __name__ == "__main__":
    main()
