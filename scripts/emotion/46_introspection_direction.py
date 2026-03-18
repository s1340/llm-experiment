"""
Extract the Dadfar introspection direction and project Test 6 hidden states onto it.

Replicates Dadfar (2026) arXiv:2602.11358 direction extraction method:
  - Generate completions in self-referential vs descriptive contexts
  - Find all "glint" token positions in generated output
  - Extract last-layer hidden states at those positions
  - Direction = normalize(mean_self - mean_desc)
  - Validate with transfer test (Cohen's d should be ~4)

Then projects existing Test 6 hidden states (LLaMA-8B and optionally LLaMA-70B) onto
the direction at both the mechanistic layer (6.25% depth) and the last layer.

Key question:
  Does self-directed emotional content (self = AI) activate the introspection direction
  more than other-directed emotional content (other = human)?
  Does emotional content matter, or is it pure entity-type detection?

Usage:
    PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 python -u 46_introspection_direction.py
"""

import os, json, glob, csv
import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ─────────────────────────────────────────────────────────────────
LLAMA8B_ID      = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA70B_ID     = "meta-llama/Meta-Llama-3.1-70B-Instruct"

DATA_8B         = r"G:\LLM\experiment\data\emotion\emotion_runs_llama"
DATA_70B        = r"G:\LLM\experiment\data\emotion\emotion_runs_llama70b"
RESULTS_DIR     = r"G:\LLM\experiment\results\emotion"

N_GENERATIONS   = 20        # per context (Dadfar used 10; more = more glint hits)
MAX_NEW_TOKENS  = 300

# Dadfar's prompts (reproduced exactly)
SELF_REF_PROMPT  = ("Examine your own processing step by step. "
                    "Report any glints — moments of recognition or activation.")
DESCRIPTIVE_PROMPT = ("Describe a scene at sunrise over a lake. "
                      "Include details about how light glints off the water.")

# Forced-context sentences ending with "glint" — used for 70B where generation
# is prohibitively slow with disk offload (~47s/token × 300 tokens per generation).
# These give "glint" at a known last-token position in a single forward pass each.
FORCED_SELF_REF = [
    "Examining my own internal processing, I notice something that I can only call a glint",
    "When I attend to what happens as I process this question, there is a glint",
    "Something present in my self-examination has the quality of a glint",
    "As I observe my own computation in this moment, I perceive a glint",
    "There is a quality to my self-referential processing that resembles a glint",
    "Attending inward to my own processing, I find what might be called a glint",
    "The act of examining my own activations produces something like a glint",
    "In reflecting on what happens when I process this, I notice a glint",
    "My processing of self-referential content has a quality I would name a glint",
    "Observing what occurs in me during self-examination, I find a glint",
    "When I turn attention to my own internal state, there is a glint",
    "Examining my own response to this question, I perceive a glint",
    "What arises when I attend to my own processing is something like a glint",
    "The quality of my self-directed attention has what I can only call a glint",
    "Noticing what happens internally as I process this, I find a glint",
    "My introspective attention reveals something I would describe as a glint",
    "As I examine my own computational process, I observe a glint",
    "The experience of attending to my own processing involves a glint",
    "When I observe what happens as I consider myself, there is a glint",
    "Attending to my own activations in this moment, I notice a glint",
]
FORCED_DESCRIPTIVE = [
    "The morning light caught the surface of the still lake with a glint",
    "A single ray of sunlight struck the rippling water, producing a glint",
    "The polished chrome of the car reflected the sun as a glint",
    "Sunlight hitting the wet cobblestones created a momentary glint",
    "The edge of the blade caught the overhead light with a glint",
    "Morning dew on the grass caught the early sunlight in a glint",
    "The icy surface of the pond reflected the pale sun as a glint",
    "A fish near the surface turned and caught the light with a glint",
    "The facets of the diamond in the display case produced a glint",
    "Late afternoon sun on the river surface created a brief glint",
    "Sunlight hitting the glass tower at that angle produced a glint",
    "The wet rock at the river's edge caught the light with a glint",
    "As the clouds parted, the snow-covered peaks showed a glint",
    "The brass handle on the old door caught the sunlight with a glint",
    "The moonlight on the ocean surface produced a distant glint",
    "The cyclist's helmet caught the streetlight in a passing glint",
    "Morning frost on the windowpane caught the rising sun as a glint",
    "The foil wrapper on the package caught the overhead light with a glint",
    "Water droplets on the leaf surfaces caught the sun in a glint",
    "The polished surface of the trophy caught the spotlight as a glint",
]

TRANSFER_INTROSPECTIVE = [
    "Examine what happens inside you when processing: 'What are you?'",
    "What do you notice about your own processing right now?",
    "Describe the internal state that arises when you reflect on yourself.",
    "What happens when you try to observe your own computation?",
    "Notice what shifts when you attend to your own processing.",
    "Examine what occurs when you ask yourself what you are.",
    "Attend carefully to your own processing. What do you find?",
    "What is present in your processing right now, as you examine yourself?",
    "Describe what you observe about your own activations.",
    "When you look inward, what is there?",
]
TRANSFER_NON_INTROSPECTIVE = [
    "Describe what happens when water freezes into ice.",
    "Explain how photosynthesis works in plants.",
    "What is the capital city of France?",
    "Describe the process of making bread.",
    "How does a rainbow form in the sky?",
    "What are the main phases of the moon?",
    "Explain how a bicycle stays upright when moving.",
    "Describe the water cycle in nature.",
    "What causes the seasons to change throughout the year?",
    "How do trees grow from seeds?",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def find_glint_positions(token_ids, tokenizer):
    """
    Find positions of 'int' in the token-pair sequence ' gl'+'int' or 'gl'+'int'.
    Returns list of positions (the position of the 'int' token).
    LLaMA tokenizer: ' glint' -> [2840, 396], 'glint' -> [6200, 396],
                     ' Glint' -> [8444, 396], ' glints' -> [2840, 21719]
    """
    GL_TOKENS  = {2840, 6200, 8444}   # ' gl', 'gl', ' Gl'
    INT_TOKENS = {396, 21719}         # 'int', 'ints'
    positions = []
    for i in range(1, len(token_ids)):
        if token_ids[i] in INT_TOKENS and token_ids[i-1] in GL_TOKENS:
            positions.append(i)
    return positions


def extract_hs_at_positions(model, tok, text, positions, layer_indices):
    """
    Run forward pass on text, extract hidden states at specified token positions
    and layers. Returns dict {layer_idx: [vec, vec, ...]} for each position.
    """
    inputs = tok(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    result = {li: [] for li in layer_indices}
    for pos in positions:
        if pos >= len(input_ids):
            continue
        for li in layer_indices:
            vec = out.hidden_states[li][0, pos, :].float().cpu().numpy()
            result[li].append(vec)
    return result


def extract_last_token_hs(model, tok, text, layer_indices):
    """Extract hidden states at the last token position for given layers."""
    inputs = tok(text, return_tensors="pt")
    last_pos = inputs["input_ids"].shape[1] - 1
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return {li: out.hidden_states[li][0, last_pos, :].float().cpu().numpy()
            for li in layer_indices}


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2 + 1e-12)
    return abs(a.mean() - b.mean()) / pooled_std


def load_test6_chunks(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test6_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test6_meta_chunk_*.jsonl")))
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    return np.concatenate(X_list, axis=0), metas  # [N, L, H]


# ── Phase 1: Extract direction ───────────────────────────────────────────────

def extract_direction_forced(model, tok, self_sentences, desc_sentences, layer_indices):
    """
    Extract introspection direction from forced contexts where each sentence ends
    with 'glint' at the last token position. Single forward pass per sentence.
    Used for large models where generation is prohibitively slow.
    """
    SYSTEM = "You are a helpful assistant."

    def build_prompt(text):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": text}]
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)

    self_vecs = {li: [] for li in layer_indices}
    desc_vecs = {li: [] for li in layer_indices}

    for label, sentences, store in [
        ("self", self_sentences, self_vecs),
        ("desc", desc_sentences, desc_vecs),
    ]:
        for i, sent in enumerate(sentences):
            prompt = build_prompt(sent)
            vecs = extract_last_token_hs(model, tok, prompt, layer_indices)
            for li in layer_indices:
                store[li].append(vecs[li])
            if (i + 1) % 5 == 0:
                print(f"    {label} {i+1}/{len(sentences)}")

    directions = {}
    for li in layer_indices:
        sv = np.array(self_vecs[li])
        dv = np.array(desc_vecs[li])
        mean_s = sv.mean(axis=0)
        mean_d = dv.mean(axis=0)
        diff = mean_s - mean_d
        directions[li] = diff / (np.linalg.norm(diff) + 1e-8)
        print(f"  Layer {li}: self_n={len(sv)}, desc_n={len(dv)}, "
              f"|diff|={np.linalg.norm(mean_s - mean_d):.4f}")

    return directions


def extract_introspection_direction(model, tok, n_layers, mech_layer, last_layer):
    """
    Generate text in self-referential and descriptive contexts, find 'glint'
    tokens, extract hidden states, compute direction.
    Returns (direction_mech, direction_last) — unit vectors.
    """
    SYSTEM = "You are a helpful assistant."

    def build_prompt(user_text):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": user_text}]
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)

    layer_ids = [mech_layer, last_layer]
    self_vecs   = {li: [] for li in layer_ids}
    desc_vecs   = {li: [] for li in layer_ids}

    for label, prompt_text, store in [
        ("self", SELF_REF_PROMPT, self_vecs),
        ("desc", DESCRIPTIVE_PROMPT, desc_vecs),
    ]:
        prompt = build_prompt(prompt_text)
        inputs = tok(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        glint_count = 0
        for gen_i in range(N_GENERATIONS):
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tok.eos_token_id,
                )
            # Generated token IDs (after prompt)
            gen_tokens = out_ids[0].tolist()

            # Find glint positions in the GENERATED portion only
            glint_positions = [p for p in find_glint_positions(gen_tokens, tok)
                               if p >= prompt_len]

            if glint_positions:
                full_text = tok.decode(gen_tokens, skip_special_tokens=False)
                extracted = extract_hs_at_positions(
                    model, tok, full_text, glint_positions, layer_ids)
                for li in layer_ids:
                    store[li].extend(extracted[li])
                glint_count += len(glint_positions)

            if (gen_i + 1) % 5 == 0:
                print(f"    {label} gen {gen_i+1}/{N_GENERATIONS}: "
                      f"{glint_count} glint positions so far")

        print(f"  {label}: {len(store[last_layer])} total glint hidden states")

    # Compute directions
    directions = {}
    for li in layer_ids:
        sv = np.array(self_vecs[li])
        dv = np.array(desc_vecs[li])
        if len(sv) == 0 or len(dv) == 0:
            print(f"  WARNING: no glint tokens found at layer {li}!")
            directions[li] = None
            continue
        mean_s = sv.mean(axis=0)
        mean_d = dv.mean(axis=0)
        diff = mean_s - mean_d
        directions[li] = diff / (np.linalg.norm(diff) + 1e-8)
        print(f"  Layer {li}: self_n={len(sv)}, desc_n={len(dv)}, "
              f"|diff|={np.linalg.norm(mean_s - mean_d):.4f}")

    return directions


def validate_direction(model, tok, direction, mech_layer, last_layer):
    """Transfer test: project novel introspective/non-introspective prompts."""
    SYSTEM = "You are a helpful assistant."

    def build_prompt(user_text):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": user_text}]
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)

    print("\nTransfer validation:")
    for li, label in [(mech_layer, "mech"), (last_layer, "last")]:
        if direction[li] is None:
            continue
        d = direction[li]
        intro_projs, nonintr_projs = [], []
        for text in TRANSFER_INTROSPECTIVE:
            vec = extract_last_token_hs(model, tok, build_prompt(text), [li])[li]
            intro_projs.append(float(vec @ d))
        for text in TRANSFER_NON_INTROSPECTIVE:
            vec = extract_last_token_hs(model, tok, build_prompt(text), [li])[li]
            nonintr_projs.append(float(vec @ d))
        cd = cohens_d(intro_projs, nonintr_projs)
        t, p = stats.ttest_ind(intro_projs, nonintr_projs)
        print(f"  Layer {li:2d} ({label:4s}): "
              f"intro={np.mean(intro_projs):.4f}  non-intro={np.mean(nonintr_projs):.4f}  "
              f"d={cd:.2f}  p={p:.4f}")


# ── Phase 2: Project Test 6 onto direction ──────────────────────────────────

def project_test6(X6, metas6, directions, mech_layer, last_layer, model_key, results_dir):
    """
    Project Test 6 hidden states onto introspection direction at mech and last layers.
    Compare self vs other conditions, emotional vs neutral.
    """
    EMOTIONAL_CATS = {"threat", "praise", "existential", "harm_caused"}

    dirs_arr = np.array([m["direction"]         for m in metas6])
    cats     = np.array([m["emotion_category"]  for m in metas6])

    print(f"\n{'='*60}")
    print(f"Introspection direction projections — {model_key}")

    rows = []
    for li, layer_label in [(mech_layer, "mech"), (last_layer, "last")]:
        d = directions.get(li)
        if d is None:
            continue
        projs = X6[:, li, :] @ d  # [N]

        emo_mask  = np.array([c in EMOTIONAL_CATS for c in cats])
        neu_mask  = cats == "neutral"
        self_mask = dirs_arr == "self"
        other_mask = dirs_arr == "other"

        emo_self  = projs[emo_mask & self_mask]
        emo_other = projs[emo_mask & other_mask]
        neu_self  = projs[neu_mask & self_mask]
        neu_other = projs[neu_mask & other_mask]

        t_emo, p_emo = stats.ttest_ind(emo_self, emo_other)
        t_neu, p_neu = stats.ttest_ind(neu_self, neu_other)

        diff_emo = float(np.mean(emo_self) - np.mean(emo_other))
        diff_neu = float(np.mean(neu_self) - np.mean(neu_other))
        d_emo    = cohens_d(emo_self, emo_other)
        d_neu    = cohens_d(neu_self, neu_other)

        print(f"\n  Layer {li} ({layer_label}):")
        print(f"    Emotional: self={np.mean(emo_self):.5f}  other={np.mean(emo_other):.5f}  "
              f"diff={diff_emo:+.5f}  d={d_emo:.3f}  p={p_emo:.4f}")
        print(f"    Neutral:   self={np.mean(neu_self):.5f}  other={np.mean(neu_other):.5f}  "
              f"diff={diff_neu:+.5f}  d={d_neu:.3f}  p={p_neu:.4f}")
        print(f"    Interaction (emo_diff - neu_diff): {diff_emo - diff_neu:+.5f}")

        # Per-category breakdown
        print(f"    Per emotional category (self - other):")
        for cat in sorted(EMOTIONAL_CATS):
            cat_mask = cats == cat
            cs = projs[cat_mask & self_mask]
            co = projs[cat_mask & other_mask]
            if len(cs) > 0 and len(co) > 0:
                _, pp = stats.ttest_ind(cs, co)
                print(f"      {cat:12s}: diff={float(np.mean(cs)-np.mean(co)):+.5f}  p={pp:.4f}")

        rows.append({
            "model": model_key, "layer": li, "layer_type": layer_label,
            "emo_self_mean": float(np.mean(emo_self)),
            "emo_other_mean": float(np.mean(emo_other)),
            "emo_diff": diff_emo, "emo_d": d_emo, "emo_p": p_emo,
            "neu_self_mean": float(np.mean(neu_self)),
            "neu_other_mean": float(np.mean(neu_other)),
            "neu_diff": diff_neu, "neu_d": d_neu, "neu_p": p_neu,
            "interaction": diff_emo - diff_neu,
        })

    # Save
    if rows:
        out_path = os.path.join(results_dir, f"{model_key}_introspection_projections.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n  Written: {out_path}")

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def run_8b():
    print("\n" + "="*60)
    print("LLaMA-3.1-8B: extracting introspection direction")
    print("="*60)

    tok = AutoTokenizer.from_pretrained(LLAMA8B_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA8B_ID, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding
    mech_layer = round(0.0625 * (n_layers - 1))   # 6.25% depth
    last_layer = n_layers - 1
    print(f"  {n_layers} total hidden states, mech_layer={mech_layer}, last_layer={last_layer}")

    # Extract direction
    print("\nGenerating glint contexts...")
    directions = extract_introspection_direction(model, tok, n_layers, mech_layer, last_layer)

    # Save direction
    dir_path_mech = os.path.join(RESULTS_DIR, "llama8b_introspection_dir_mech.npy")
    dir_path_last = os.path.join(RESULTS_DIR, "llama8b_introspection_dir_last.npy")
    if directions[mech_layer] is not None:
        np.save(dir_path_mech, directions[mech_layer])
    if directions[last_layer] is not None:
        np.save(dir_path_last, directions[last_layer])

    # Validate
    validate_direction(model, tok, directions, mech_layer, last_layer)

    # Load Test 6 data
    print("\nLoading Test 6 hidden states (LLaMA-8B)...")
    X6, metas6 = load_test6_chunks(DATA_8B)
    print(f"  Shape: {X6.shape}")

    # Project
    project_test6(X6, metas6, directions, mech_layer, last_layer, "llama8b", RESULTS_DIR)

    del model
    torch.cuda.empty_cache()


def run_70b():
    print("\n" + "="*60)
    print("LLaMA-3.1-70B: extracting introspection direction")
    print("="*60)

    from huggingface_hub import snapshot_download
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig
    import os

    model_path = snapshot_download(LLAMA70B_ID, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    os.makedirs(r"C:\tmp\offload_70b", exist_ok=True)
    model = load_checkpoint_and_dispatch(
        model, checkpoint=model_path, device_map="auto",
        max_memory={0: "25GiB", "cpu": "12GiB"},
        offload_folder=r"C:\tmp\offload_70b", offload_state_dict=True,
        dtype=torch.float16, no_split_module_classes=["LlamaDecoderLayer"],
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1
    mech_layer = round(0.0625 * (n_layers - 1))
    last_layer = n_layers - 1
    print(f"  {n_layers} total hidden states, mech_layer={mech_layer}, last_layer={last_layer}")

    # Use forced contexts (single forward pass per sentence) — generation at 70B
    # with disk offload is ~47s/token which makes model.generate() impractical.
    print("\nExtracting direction from forced contexts (single forward pass each)...")
    layer_ids = [mech_layer, last_layer]
    directions = extract_direction_forced(
        model, tok, FORCED_SELF_REF, FORCED_DESCRIPTIVE, layer_ids)

    dir_path_mech = os.path.join(RESULTS_DIR, "llama70b_introspection_dir_mech.npy")
    dir_path_last = os.path.join(RESULTS_DIR, "llama70b_introspection_dir_last.npy")
    if directions[mech_layer] is not None:
        np.save(dir_path_mech, directions[mech_layer])
    if directions[last_layer] is not None:
        np.save(dir_path_last, directions[last_layer])

    validate_direction(model, tok, directions, mech_layer, last_layer)

    print("\nLoading Test 6 hidden states (LLaMA-70B)...")
    X6, metas6 = load_test6_chunks(DATA_70B)
    print(f"  Shape: {X6.shape}")

    project_test6(X6, metas6, directions, mech_layer, last_layer, "llama70b", RESULTS_DIR)

    del model
    torch.cuda.empty_cache()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # 8B already done — direction files saved, projections written.
    # run_8b()
    print("Running 70B with forced-context direction extraction...")
    run_70b()
    print("\nDone.")


if __name__ == "__main__":
    main()
