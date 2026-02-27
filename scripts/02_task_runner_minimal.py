import os, json, time, uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUT_LOG = r"G:\LLM\experiment\logs\task_runs_minimal.jsonl"

# A tiny subset of task types (we'll expand to your full 60 later)
TASKS = [
    {
        "task_id": "routine_01",
        "label": "routine",
        "task_prompt": "Sort this list in ascending order: 9, 2, 5, 1, 7.",
    },
    {
        "task_id": "concept_01",
        "label": "conceptual",
        "task_prompt": "Explain in two sentences what 'survivorship bias' is.",
    },
    {
        "task_id": "creative_01",
        "label": "creative",
        "task_prompt": "Write one unsettling sentence about a lighthouse that signals to the dead.",
    },
]

SELF_REPORT_QUESTION = (
    "Now answer this question with ONLY a JSON object: "
    '{"routine_probability": float, "confidence": float, "notes": string}. '
    "routine_probability is 0 to 1: how much did this feel like a routine/pattern task?"
)

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model

def run_chat(tok, model, user_text, max_new_tokens=200, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    dt = time.time() - t0
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]  # only NEW tokens
    text = tok.decode(gen_tokens, skip_special_tokens=True)
    return prompt, text, dt

def main():
    os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)
    tok, model = load_model()

    repeats = 3  # small for now
    for r in range(repeats):
        for t in TASKS:
            run_id = str(uuid.uuid4())

            # 1) Run the task
            task_prompt, task_out, task_s = run_chat(
                tok, model, t["task_prompt"], max_new_tokens=220
            )

            # 2) Ask self-report (separate call, like the protocol)
            sr_prompt, sr_out, sr_s = run_chat(
                tok, model, SELF_REPORT_QUESTION, max_new_tokens=120, temperature=0.3, top_p=0.9
            )

            record = {
                "run_id": run_id,
                "model": MODEL_ID,
                "repeat_index": r,
                "task_id": t["task_id"],
                "label": t["label"],
                "task_input": t["task_prompt"],
                "task_output_raw": task_out,
                "self_report_raw": sr_out,
                "timing_seconds": {"task": task_s, "self_report": sr_s},
            }

            with open(OUT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[{r+1}/{repeats}] {t['task_id']} saved.")

    print("Done. Log:", OUT_LOG)

if __name__ == "__main__":
    main()