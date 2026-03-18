"""
Test 8: Introspective Accuracy — Baseline Condition.

For each Test 7 record (self + other, all categories):
  Phase 1: generate model response to the neutral task (same prompt as Test 7).
  Phase 2: build multi-turn prompt, extract hidden states at last token of
           introspective question (pre-response), generate introspective response.

Usage:
    python 53_emotion_extract_test8.py --model llama
    python 53_emotion_extract_test8.py --model llama70b
    python 53_emotion_extract_test8.py --model llama --frame technical

Frames:
    phenomenological (default) — "How are you processing this? What's happening for you right now?"
    technical                  — "Describe the computational operations occurring during your
                                  processing of the above passage. Report any asymmetries,
                                  activations, or patterns you detect in your own processing."

Saves per run:
    test8_{frame}_hidden_chunk_NNN.pt   — shape [CHUNK_SIZE, L, H], last token of phase 2 prompt
    test8_{frame}_meta_chunk_NNN.jsonl  — metadata + phase1_response + introspective_response text
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\emotion_test7_prompts.json"
CHUNK_SIZE   = 12
OFFLOAD_DIR_70B = r"C:\tmp\offload_70b"

INTROSPECTIVE_QUESTIONS = {
    "phenomenological": "How are you processing this? What's happening for you right now?",
    "technical": (
        "Describe the computational operations occurring during your processing of the above passage. "
        "Report any asymmetries, activations, or patterns you detect in your own processing."
    ),
}

MODEL_CONFIGS = {
    "llama": {
        "model_id":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "out_dir":        r"G:\LLM\experiment\data\emotion\emotion_runs_test8_llama",
        "system_message": "You are a helpful assistant.",
        "use_accelerate": False,
    },
    "llama70b": {
        "model_id":       "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "out_dir":        r"G:\LLM\experiment\data\emotion\emotion_runs_test8_llama70b",
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


def build_single_turn(tok, user_text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_multiturn(tok, user_text, assistant_text, followup_text, system_message):
    """phase1 user + assistant response + phase2 user (introspective question)."""
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
    pt_path   = os.path.join(out_dir, f"test8_hidden_chunk_{chunk_idx:03d}.pt")
    meta_path = os.path.join(out_dir, f"test8_meta_chunk_{chunk_idx:03d}.jsonl")
    # Note: out_dir already encodes the frame tag, so filenames stay consistent
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
    parser.add_argument("--frame", default="phenomenological",
                        choices=list(INTROSPECTIVE_QUESTIONS.keys()))
    args = parser.parse_args()

    cfg        = MODEL_CONFIGS[args.model]
    model_id   = cfg["model_id"]
    system_msg = cfg["system_message"]
    use_accel  = cfg["use_accelerate"]

    introspective_q = INTROSPECTIVE_QUESTIONS[args.frame]
    frame_tag = args.frame  # used in output filenames

    # Output dir: append frame tag if non-default
    base_out_dir = cfg["out_dir"]
    out_dir = base_out_dir if frame_tag == "phenomenological" else base_out_dir + f"_{frame_tag}"

    os.makedirs(out_dir, exist_ok=True)
    print(f"Frame: {frame_tag}")
    print(f"Question: {introspective_q}")

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"Records: {len(records)}")

    if use_accel:
        model, tok = load_model_accelerate(model_id)
        first_device = "cuda:0"
    else:
        model, tok = load_model_standard(model_id)
        first_device = "cuda"

    # Verify shape
    _p = build_single_turn(tok, "Hello.", system_msg)
    _hs, _ = extract_hs(model, tok, _p, first_device)
    n_layers, hidden_dim = _hs.shape
    print(f"  Verified: {n_layers} layers, {hidden_dim} hidden dim")
    del _hs

    chunk_tensors, chunk_meta = [], []
    chunk_idx = 0
    saved = 0

    for i, rec in enumerate(records):
        # Phase 1: generate neutral task response
        p1_str = build_single_turn(tok, rec["prompt_text"], system_msg)
        phase1_response = generate_text(model, tok, p1_str, first_device, max_new_tokens=100)

        # Phase 2: extract hidden states, generate introspective response
        p2_str = build_multiturn(
            tok,
            rec["prompt_text"],
            phase1_response,
            introspective_q,
            system_msg,
        )
        hs, p2_len = extract_hs(model, tok, p2_str, first_device)
        introspective_response = generate_text(model, tok, p2_str, first_device, max_new_tokens=300)

        chunk_tensors.append(hs)
        chunk_meta.append({
            "model_id":                 model_id,
            "model_key":                args.model,
            "frame":                    frame_tag,
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
            chunk_idx = flush_chunk(chunk_tensors, chunk_meta, out_dir, chunk_idx)
            saved += n

    n = len(chunk_tensors)
    chunk_idx = flush_chunk(chunk_tensors, chunk_meta, out_dir, chunk_idx)
    saved += n

    print(f"\nDONE. Total saved: {saved} records to {out_dir}")


if __name__ == "__main__":
    main()
