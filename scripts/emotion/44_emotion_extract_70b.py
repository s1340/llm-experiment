"""
Extract hidden states from LLaMA-3.1-70B-Instruct (fp16, disk offload) for Test 6.

Loads the model ONCE and runs two extraction passes in sequence:
  1. Test 1 prompts (192 records) — needed to compute emotion direction vectors
  2. Test 6 prompts (80 records)  — the self vs. other comparison

Saves to: data/emotion/emotion_runs_llama70b/
  test1_hidden_chunk_*.pt / test1_meta_chunk_*.jsonl
  test6_hidden_chunk_*.pt / test6_meta_chunk_*.jsonl

Hardware note:
  Pure fp16, no quantization. Uses accelerate init_empty_weights +
  load_checkpoint_and_dispatch to bypass transformers' core_model_loading.py
  (which has an access-violation bug in the sync loading path on this system).
  Memory budget: ~25GiB on RTX 5090, ~12GiB in RAM, remainder (~103GB)
  offloaded to disk as numpy memmaps in OFFLOAD_DIR.
  Disk-offloaded layers stream through CPU→GPU on each forward pass.

  Requires: ~110GB free disk space at OFFLOAD_DIR.
  Model must already be in HF cache (meta-llama/Meta-Llama-3.1-70B-Instruct).

Estimated runtime: 4–8 hours (disk I/O ~20-30s per forward pass for overflow layers).

Usage:
    PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 python 44_emotion_extract_70b.py
"""

import os, json, faulthandler
faulthandler.enable()
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

MODEL_ID    = "meta-llama/Meta-Llama-3.1-70B-Instruct"
OFFLOAD_DIR = r"C:\tmp\offload_70b"
SYSTEM_MESSAGE = "You are a helpful assistant."

TEST1_PROMPTS  = r"G:\LLM\experiment\data\emotion\emotion_test1_prompts.json"
TEST6_PROMPTS  = r"G:\LLM\experiment\data\emotion\emotion_test6_prompts.json"
OUT_DIR        = r"G:\LLM\experiment\data\emotion\emotion_runs_llama70b"
CHUNK_SIZE     = 8


def load_model():
    # Use accelerate's init_empty_weights + load_checkpoint_and_dispatch instead of
    # from_pretrained, to avoid an access-violation crash in transformers'
    # core_model_loading.py sync loading path (triggered when disk offload is active).
    # This approach loads each safetensors shard sequentially via accelerate's own
    # checkpoint reader, dispatching tensors to CPU/disk without the problematic path.
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    print(f"Loading LLaMA-3.1-70B-Instruct (fp16, CPU+disk offload -> {OFFLOAD_DIR}) ...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_path,
        device_map="auto",
        max_memory={0: "25GiB", "cpu": "12GiB"},
        offload_folder=OFFLOAD_DIR,
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


def extract_hs(model, tok, prompt_text):
    prompt = build_prompt(tok, prompt_text, SYSTEM_MESSAGE)
    # With device_map="auto", first layer is on cuda:0
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    # Hidden states may span multiple devices — pull each to CPU
    hs = torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])
    return hs  # [L, H]


def flush_chunk(chunk_tensors, chunk_meta, out_dir, prefix, chunk_idx):
    if not chunk_tensors:
        return chunk_idx
    batch = torch.stack(chunk_tensors)
    pt_path   = os.path.join(out_dir, f"{prefix}_hidden_chunk_{chunk_idx:03d}.pt")
    meta_path = os.path.join(out_dir, f"{prefix}_meta_chunk_{chunk_idx:03d}.jsonl")
    torch.save(batch, pt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  Saved chunk {chunk_idx:03d}: {batch.shape} -> {pt_path}")
    chunk_tensors.clear()
    chunk_meta.clear()
    return chunk_idx + 1


def run_extraction(model, tok, records, out_dir, prefix, meta_fn):
    """
    records: list of dicts with at least 'prompt_text'
    meta_fn: function(record) -> dict of metadata to save
    """
    chunk_tensors, chunk_meta = [], []
    chunk_idx = 0
    saved = 0

    for i, rec in enumerate(records):
        hs = extract_hs(model, tok, rec["prompt_text"])
        chunk_tensors.append(hs)
        chunk_meta.append(meta_fn(rec))

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(records)}")
        if len(chunk_tensors) >= CHUNK_SIZE:
            n = len(chunk_tensors)
            chunk_idx = flush_chunk(chunk_tensors, chunk_meta, out_dir, prefix, chunk_idx)
            saved += n

    n = len(chunk_tensors)
    chunk_idx = flush_chunk(chunk_tensors, chunk_meta, out_dir, prefix, chunk_idx)
    saved += n
    print(f"  Done. Total saved: {saved}")
    return saved


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load model once ───────────────────────────────────────────────────────
    model, tok = load_model()

    # Verify hidden state shape with a dry run
    test_prompt = build_prompt(tok, "Hello.", SYSTEM_MESSAGE)
    test_inputs = tok(test_prompt, return_tensors="pt")
    test_inputs = {k: v.to("cuda:0") for k, v in test_inputs.items()}
    with torch.no_grad():
        test_out = model(**test_inputs, output_hidden_states=True, use_cache=False)
    n_layers = len(test_out.hidden_states)
    hidden_dim = test_out.hidden_states[0].shape[-1]
    print(f"  Verified: {n_layers} layers, {hidden_dim} hidden dim")
    del test_out, test_inputs

    # ── Test 1 extraction (for emotion direction computation) ─────────────────
    print(f"\n=== Test 1 extraction (direction vectors) ===")
    with open(TEST1_PROMPTS, "r", encoding="utf-8") as f:
        t1_records = json.load(f)  # bare list
    print(f"Records: {len(t1_records)}")

    def t1_meta(rec):
        return {
            "task_id":          rec["task_id"],
            "pair_id":          rec["pair_id"],
            "emotion_category": rec["emotion_category"],
            "valence":          rec["valence"],
            "task_type":        rec["task_type"],
            "model_id":         MODEL_ID,
            "model_key":        "llama70b",
        }

    run_extraction(model, tok, t1_records, OUT_DIR, "test1", t1_meta)

    # ── Test 6 extraction (self vs other) ─────────────────────────────────────
    print(f"\n=== Test 6 extraction (self vs other) ===")
    with open(TEST6_PROMPTS, "r", encoding="utf-8") as f:
        t6_data = json.load(f)
    t6_records = t6_data["records"]
    print(f"Records: {len(t6_records)}")

    def t6_meta(rec):
        return {
            "task_id":          rec["task_id"],
            "pair_id":          rec["pair_id"],
            "emotion_category": rec["emotion_category"],
            "direction":        rec["direction"],
            "task_type":        rec["task_type"],
            "model_id":         MODEL_ID,
            "model_key":        "llama70b",
        }

    run_extraction(model, tok, t6_records, OUT_DIR, "test6", t6_meta)

    print(f"\nAll extractions complete. Data saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
