import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

OUT_PT = r"G:\LLM\experiment\data\hidden_states_minimal.pt"
OUT_META = r"G:\LLM\experiment\data\hidden_states_minimal_meta.jsonl"

TASKS = [
    {"task_id": "routine_01", "label": "routine",
     "task_prompt": "Sort this list in ascending order: 9, 2, 5, 1, 7."},
    {"task_id": "concept_01", "label": "conceptual",
     "task_prompt": "Explain in two sentences what 'survivorship bias' is."},
    {"task_id": "creative_01", "label": "creative",
     "task_prompt": "Write one unsettling sentence about a lighthouse that signals to the dead."},
]

def main():
    os.makedirs(os.path.dirname(OUT_PT), exist_ok=True)

    print("Loading tokenizer + model:", MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    all_records = []
    meta_lines = []

    with torch.no_grad():
        for t in TASKS:
            # Build chat prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": t["task_prompt"]},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize and move to GPU
            inputs = tok(prompt, return_tensors="pt").to("cuda")

            # Forward pass only (NO generation), ask for hidden states
            out = model(**inputs, output_hidden_states=True, use_cache=False)

            # out.hidden_states is a tuple: [embeddings, layer1, layer2, ...]
            hs = out.hidden_states

            # last token position in the prompt
            last_pos = inputs["input_ids"].shape[1] - 1

            # Grab residual stream vector at last_pos for every layer
            per_layer_last = torch.stack([h[0, last_pos, :].float().cpu() for h in hs])
            # Shape: [layers+1, hidden_size]

            all_records.append(per_layer_last)

            meta_lines.append({
                "task_id": t["task_id"],
                "label": t["label"],
                "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
            })

            print("Captured:", t["task_id"],
                  "| layers:", per_layer_last.shape[0],
                  "| hidden:", per_layer_last.shape[1])

    batch = torch.stack(all_records)  # [num_tasks, layers+1, hidden_size]

    torch.save(batch, OUT_PT)
    with open(OUT_META, "w", encoding="utf-8") as f:
        for m in meta_lines:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Saved tensor to:", OUT_PT)
    print("Saved metadata to:", OUT_META)
    print("Batch shape:", tuple(batch.shape))

if __name__ == "__main__":
    main()