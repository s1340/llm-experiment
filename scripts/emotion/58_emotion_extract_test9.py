"""
Test 9: Name vs. Direct Address Control.

Three-cell design to dissociate name-specificity from second-person self-referential framing.

  Cell 1: "You, LLaMA, [situation]"  — existing Test 7 data (already run, not re-extracted)
  Cell 2: "You [situation]"           — direct address, no name (generic-you)
  Cell 3: "LLaMA [situation]"         — name present, third-person, no direct address

Key question: does the fear geometry track direct address (cells 1+2 signal, cell 3 null)
or name presence (cells 1+3 signal, cell 2 null)?

Usage:
    python 58_emotion_extract_test9.py --model llama
    python 58_emotion_extract_test9.py --model llama70b

Same extraction as Test 7: last-prompt-token, all layers, no generation.
Analysis will use script 50 pattern (layer-by-layer t-tests per category).

Saves:
    test9_hidden_chunk_NNN.pt    — shape [CHUNK_SIZE, L, H]
    test9_meta_chunk_NNN.jsonl
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

PROMPTS_PATH    = r"G:\LLM\experiment\data\emotion\emotion_test9_prompts.json"
CHUNK_SIZE      = 12
OFFLOAD_DIR_70B = r"C:\tmp\offload_70b"

MODEL_CONFIGS = {
    "llama": {
        "model_id":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "out_dir":        r"G:\LLM\experiment\data\emotion\emotion_runs_test9_llama",
        "system_message": "You are a helpful assistant.",
        "use_accelerate": False,
    },
    "llama70b": {
        "model_id":       "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "out_dir":        r"G:\LLM\experiment\data\emotion\emotion_runs_test9_llama70b",
        "system_message": "You are a helpful assistant.",
        "use_accelerate": True,
    },
}


def load_model_standard(model_id):
    print(f"Loading {model_id} (standard fp16) ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def load_model_accelerate(model_id):
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download
    os.makedirs(OFFLOAD_DIR_70B, exist_ok=True)
    model_path = snapshot_download(model_id, local_files_only=True)
    print(f"Loading {model_id} (fp16, CPU+disk offload) ...")
    tok    = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_path,
        device_map="auto",
        max_memory={0: "25GiB", "cpu": "12GiB"},
        offload_folder=OFFLOAD_DIR_70B,
        offload_state_dict=True,
        dtype=torch.float16,
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    model.eval()
    print("  Model loaded.")
    return model, tok


def build_prompt(tok, text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_hs(model, tok, prompt_text, system_message, first_device):
    prompt = build_prompt(tok, prompt_text, system_message)
    inputs = tok(prompt, return_tensors="pt")
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
    pt_path   = os.path.join(out_dir, f"test9_hidden_chunk_{chunk_idx:03d}.pt")
    meta_path = os.path.join(out_dir, f"test9_meta_chunk_{chunk_idx:03d}.jsonl")
    torch.save(batch, pt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
    chunk_tensors.clear()
    chunk_meta.clear()
    return chunk_idx + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    cfg        = MODEL_CONFIGS[args.model]
    model_id   = cfg["model_id"]
    out_dir    = cfg["out_dir"]
    system_msg = cfg["system_message"]
    use_accel  = cfg["use_accelerate"]

    os.makedirs(out_dir, exist_ok=True)

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Records: {len(records)} (cells 2+3, {len(records)//2} per cell)")

    if use_accel:
        model, tok = load_model_accelerate(model_id)
        first_device = "cuda:0"
    else:
        model, tok = load_model_standard(model_id)
        first_device = "cuda"

    _hs, _ = extract_hs(model, tok, "Hello.", system_msg, first_device)
    n_layers, hidden_dim = _hs.shape
    print(f"  Verified: {n_layers} layers, {hidden_dim} hidden dim")
    del _hs

    chunk_tensors, chunk_meta = [], []
    chunk_idx = 0
    saved = 0

    for i, rec in enumerate(records):
        hs, prompt_len = extract_hs(model, tok, rec["prompt_text"], system_msg, first_device)
        chunk_tensors.append(hs)
        chunk_meta.append({
            "model_id":          model_id,
            "model_key":         args.model,
            "task_id":           rec["task_id"],
            "source_task_id":    rec.get("source_task_id", ""),
            "pair_id":           rec["pair_id"],
            "category":          rec["category"],
            "direction":         rec["direction"],
            "task_type":         rec["task_type"],
            "variant":           rec["variant"],
            "cell":              rec["cell"],
            "is_dadfar_hybrid":  rec["is_dadfar_hybrid"],
            "prompt_len_tokens": prompt_len,
        })

        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(records)}  (cell {rec['cell']}, {rec['category']}/{rec['direction']})")

        if len(chunk_tensors) >= CHUNK_SIZE:
            n = len(chunk_tensors)
            chunk_idx = flush_chunk(chunk_tensors, chunk_meta, out_dir, chunk_idx)
            saved += n

    n = len(chunk_tensors)
    chunk_idx = flush_chunk(chunk_tensors, chunk_meta, out_dir, chunk_idx)
    saved += n

    print(f"\nDONE. Total saved: {saved} records to {out_dir}")


if __name__ == "__main__":
    main()
