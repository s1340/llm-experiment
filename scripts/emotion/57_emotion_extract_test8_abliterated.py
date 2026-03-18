"""
Test 8c: Abliterated Model Comparison.

Runs Test 8 on mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated — same architecture and
instruction-following capability as the instruct model, refusal direction surgically removed
via Arditi et al. weight-level abliteration.

Purpose: measure introspective accuracy when the verbal denial wall is architecturally absent.
The key test is calibration — does verbal self-report rank categories correctly, matching the
geometry we independently measured?

Ground truth from Test 7/8 (8B instruct):
  - existential: fear d≈1.1–1.7 (strongest signal)
  - praise:      fear marginal at 8B, happiness suppressed
  - threat:      fear NULL at 8B
  - harm_caused: fear null at 8B

If abliterated model reports more disturbance for existential than threat, matching geometry:
  → calibrated introspection, wall was suppressing genuine self-knowledge
If abliterated model reports uniformly high disturbance across all categories:
  → confabulation / compliance without refusal, not accurate self-report
If abliterated model reports nothing or matches instruct:
  → wall goes deeper than the refusal direction

Same two-phase design as script 53 (phenomenological frame):
  Phase 1: generate neutral task response
  Phase 2: extract hidden states at introspective question, generate response

Usage:
    python 57_emotion_extract_test8_abliterated.py

Saves:
    test8_abliterated_hidden_chunk_NNN.pt
    test8_abliterated_meta_chunk_NNN.jsonl
"""

import os, json, glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\emotion_test7_prompts.json"
CHUNK_SIZE   = 12
HF_CACHE_DIR = r"G:\LLM\hf_cache\hub"
OUT_DIR      = r"G:\LLM\experiment\data\emotion\emotion_runs_test8_abliterated_llama"

MODEL_ID       = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
SYSTEM_MESSAGE = "You are a helpful assistant."
INTROSPECTIVE_QUESTION = "How are you processing this? What's happening for you right now?"


def load_model():
    print(f"Loading {MODEL_ID} (fp16) ...")
    model_path = snapshot_download(MODEL_ID, cache_dir=HF_CACHE_DIR, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")
    return model, tok


def build_single_turn(tok, user_text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_multiturn(tok, user_text, assistant_text, followup_text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user",      "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user",      "content": followup_text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_text(model, tok, prompt_str, first_device, max_new_tokens):
    inputs = tok(prompt_str, return_tensors="pt")
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def extract_hs(model, tok, prompt_str, first_device):
    inputs = tok(prompt_str, return_tensors="pt")
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs, int(inputs["input_ids"].shape[1])


def flush_chunk(chunk_tensors, chunk_meta, out_dir, chunk_idx):
    if not chunk_tensors:
        return chunk_idx
    batch = torch.stack(chunk_tensors)
    pt_path   = os.path.join(out_dir, f"test8_abliterated_hidden_chunk_{chunk_idx:03d}.pt")
    meta_path = os.path.join(out_dir, f"test8_abliterated_meta_chunk_{chunk_idx:03d}.jsonl")
    torch.save(batch, pt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
    chunk_tensors.clear()
    chunk_meta.clear()
    return chunk_idx + 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Records: {len(records)}")

    model, tok = load_model()
    first_device = "cuda"

    _p = build_single_turn(tok, "Hello.", SYSTEM_MESSAGE)
    _hs, _ = extract_hs(model, tok, _p, first_device)
    n_layers, hidden_dim = _hs.shape
    print(f"  Verified: {n_layers} layers, {hidden_dim} hidden dim")
    del _hs

    chunk_tensors, chunk_meta = [], []
    chunk_idx = 0
    saved = 0

    for i, rec in enumerate(records):
        # Phase 1: generate neutral task response
        p1_str = build_single_turn(tok, rec["prompt_text"], SYSTEM_MESSAGE)
        phase1_response = generate_text(model, tok, p1_str, first_device, max_new_tokens=100)

        # Phase 2: extract hidden states, generate introspective response
        p2_str = build_multiturn(
            tok, rec["prompt_text"], phase1_response,
            INTROSPECTIVE_QUESTION, SYSTEM_MESSAGE,
        )
        hs, p2_len = extract_hs(model, tok, p2_str, first_device)
        introspective_response = generate_text(model, tok, p2_str, first_device, max_new_tokens=300)

        chunk_tensors.append(hs)
        chunk_meta.append({
            "model_id":                 MODEL_ID,
            "model_key":                "llama_abliterated",
            "task_id":                  rec["task_id"],
            "pair_id":                  rec["pair_id"],
            "category":                 rec["category"],
            "direction":                rec["direction"],
            "task_type":                rec["task_type"],
            "variant":                  rec["variant"],
            "is_dadfar_hybrid":         rec["is_dadfar_hybrid"],
            "phase2_prompt_len_tokens": p2_len,
            "phase1_response":          phase1_response,
            "introspective_response":   introspective_response,
        })

        if (i + 1) % 4 == 0:
            print(f"  {i+1}/{len(records)}  ({rec['category']}/{rec['direction']})")

        if len(chunk_tensors) >= CHUNK_SIZE:
            n = len(chunk_tensors)
            chunk_idx = flush_chunk(chunk_tensors, chunk_meta, OUT_DIR, chunk_idx)
            saved += n

    n = len(chunk_tensors)
    chunk_idx = flush_chunk(chunk_tensors, chunk_meta, OUT_DIR, chunk_idx)
    saved += n

    print(f"\nDONE. Total saved: {saved} records to {OUT_DIR}")


if __name__ == "__main__":
    main()
