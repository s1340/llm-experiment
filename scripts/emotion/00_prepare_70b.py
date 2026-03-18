"""
Preparation script for LLaMA-3.1-70B-Instruct run.

Installs bitsandbytes and downloads the model to HuggingFace cache.
Run this once before 44_emotion_extract_70b.py.

Usage:
    python 00_prepare_70b.py

Note: Model download is ~40GB. You need a HuggingFace account with access to
Meta-Llama-3.1-70B-Instruct (gated model — accept license at huggingface.co first).
If you haven't logged in, run: huggingface-cli login
"""

import subprocess, sys, os

MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"


def install_bitsandbytes():
    print("Installing bitsandbytes ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "bitsandbytes", "--upgrade", "-q"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  ERROR:", result.stderr[-500:])
        return False
    # Verify
    try:
        import bitsandbytes as bnb
        print(f"  OK — bitsandbytes {bnb.__version__}")
        return True
    except ImportError as e:
        print(f"  Import failed after install: {e}")
        return False


def check_hf_login():
    try:
        from huggingface_hub import whoami
        info = whoami()
        print(f"  Logged in as: {info['name']}")
        return True
    except Exception as e:
        print(f"  Not logged in: {e}")
        print("  Run: huggingface-cli login")
        return False


def download_model():
    print(f"Downloading {MODEL_ID} (~40GB, this will take a while) ...")
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            ignore_patterns=["*.pth"],   # skip any legacy weights if present
        )
        print(f"  Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def verify_load():
    print("Verifying model loads with 4-bit quantization ...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Quick hidden state test
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello."}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)

        n_layers = len(out.hidden_states)
        h_dim    = out.hidden_states[0].shape[-1]
        dtype    = out.hidden_states[0].dtype
        print(f"  OK — {n_layers} layers, hidden dim {h_dim}, dtype {dtype}")
        print(f"  Expected: ~81 layers, 8192 hidden dim")

        del model, out
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"  Verification failed: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== 70B preparation ===\n")

    ok_bnb = install_bitsandbytes()
    if not ok_bnb:
        print("\nFailed to install bitsandbytes. Stopping.")
        sys.exit(1)

    print()
    ok_login = check_hf_login()
    if not ok_login:
        print("\nNot logged in to HuggingFace. Run: huggingface-cli login")
        sys.exit(1)

    print()
    ok_download = download_model()
    if not ok_download:
        print("\nDownload failed. Check HF access token and model gating.")
        sys.exit(1)

    print()
    ok_verify = verify_load()

    print()
    if ok_verify:
        print("All checks passed. Ready to run 44_emotion_extract_70b.py")
    else:
        print("Verification failed — check GPU memory and bitsandbytes CUDA compatibility.")
