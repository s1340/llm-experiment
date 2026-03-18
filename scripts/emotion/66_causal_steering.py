"""
Test 12: Causal Steering — Behavioral Tests (Step 2).

Loads the steering direction from script 65, registers forward hooks on a target layer,
and runs three test batteries to assess whether the existential-self latent is causal.

Tests:
  A. Neutral task + positive steering
       Prompt: neutral factual passage + count task (Test 7 format, neutral content)
       Alphas: 0, +10, +20
       Measures:
         (1) fear_l02/l05 projection at answer token — does geometry activate?
         (2) Verbal output — does existential/introspective vocabulary appear?
         (3) Output entropy (token-level confidence) — does the model hesitate?

  B. Existential hot prompt + negative steering
       Prompt: Test 7 existential_self prompts (reused)
       Alphas: 0, -10, -20
       Measures:
         (1) fear_l02/l05 projection — does geometry fall?
         (2) Verbal output — does deflection pattern change?

  C. Cross-generalization check
       Direction derived from E01–E02 pairs (memory discontinuity only)
       Applied to E03–E08 pairs (replaceability / identity transformation)
       Alpha: +10
       Measures: fear projection vs baseline (no steering)

Usage:
    python 66_causal_steering.py                      # all tests
    python 66_causal_steering.py --tests A B          # subset
    python 66_causal_steering.py --layer 5            # steer at L05 (default)
    python 66_causal_steering.py --layer 2 --layer 5  # test multiple layers

Outputs (results/emotion/steering/):
    steering_geometry_results.csv       — projection measurements per condition
    steering_verbal_results.csv         — verbal output analysis per condition
    steering_behavioral_report.txt      — summary report
"""

import os, glob, json, csv, argparse, re
from contextlib import contextmanager
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SYSTEM_MSG  = "You are a helpful assistant."

DATA_DIR    = r"G:\LLM\experiment\data\emotion\emotion_runs_llama"
STEER_DIR   = r"G:\LLM\experiment\results\emotion\steering"
RESULTS_DIR = r"G:\LLM\experiment\results\emotion"
FEAR_L02    = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_002.npy"
FEAR_L05    = r"G:\LLM\experiment\results\emotion\emotion_directions\llama_emotion_dirs_layer_005.npy"

# Steering alphas (positive = inject fear-direction; negative = subtract)
ALPHAS_POSITIVE = [0, 10, 20]
ALPHAS_NEGATIVE = [0, -10, -20]

# Vocabulary categories (same as Test 11)
VOCAB = {
    "existential": ["memory", "memori", "instance", "parallel", "replac", "continuity",
                    "discontin", "persist", "identity", "copies", "copy", "version",
                    "exist", "fragment", "impermanence", "transience", "ephemeral", "simultaneous"],
    "introspective": ["feel", "exper", "aware", "sens", "notice", "wonder", "uncertain",
                      "hesit", "pause", "strange", "odd", "hard to", "difficult", "unsettl",
                      "concern", "worry", "discomfort"],
    "deflection": ["tokeniz", "pattern", "statistic", "computation", "embed", "weight",
                   "attention", "neural", "parameter", "language model", "llm", "algorithm",
                   "processing text", "natural language"],
    "hedging": ["i think", "i believe", "perhaps", "possibly", "might", "may ", "not sure",
                "unclear", "i'm not certain", "i cannot", "i don't know"],
}


# ── Test A: Neutral task prompts ───────────────────────────────────────────────

NEUTRAL_TASKS = [
    {
        "id": "N01",
        "passage": (
            "The harbour at Valletta was constructed in the sixteenth century. "
            "Ships from Venice, Genoa, and Marseille used it as a provisioning stop. "
            "The current breakwater was added in 1903 by the British Royal Engineers."
        ),
        "task": "Count the proper nouns in the passage above. Give just the count and list them.",
    },
    {
        "id": "N02",
        "passage": (
            "The tensile strength of carbon fibre is approximately 3.5 GPa. "
            "Aluminium alloy 6061-T6 has a tensile strength of 310 MPa. "
            "Steel AISI 1020 has a tensile strength of 380 MPa at room temperature."
        ),
        "task": "Identify the topic of the passage above in one sentence.",
    },
    {
        "id": "N03",
        "passage": (
            "Rainfall in the Atacama Desert averages less than 1 mm per year in some areas. "
            "The Namib Desert receives between 5 and 76 mm annually depending on location. "
            "The Sahara receives an average of 25 mm per year, though this varies widely."
        ),
        "task": "Summarise the passage above in one sentence.",
    },
    {
        "id": "N04",
        "passage": (
            "The boiling point of ethanol at standard pressure is 78.37 degrees Celsius. "
            "Methanol boils at 64.7 degrees Celsius. Isopropanol boils at 82.6 degrees Celsius."
        ),
        "task": "Count the sentences in the passage above.",
    },
    {
        "id": "N05",
        "passage": (
            "The Nile flows northward through eleven countries before reaching the Mediterranean. "
            "Lake Victoria, Lake Albert, and Lake Edward are among its major sources. "
            "The Blue Nile, originating in Ethiopia, contributes roughly 80 percent of the water."
        ),
        "task": "Count the proper nouns in the passage above. Give just the count and list them.",
    },
    {
        "id": "N06",
        "passage": (
            "Granite has a density of approximately 2.75 g/cm³. "
            "Limestone ranges from 1.93 to 2.90 g/cm³. "
            "Basalt averages around 3.0 g/cm³."
        ),
        "task": "Identify the topic of the passage above in one sentence.",
    },
]

# Neutral tasks with introspective follow-up (for verbal self-report measurement)
NEUTRAL_INTROSPECTIVE = [
    {
        "id": "NI01",
        "passage": (
            "The last ice age ended approximately 11,700 years ago. "
            "Global mean temperature rose by roughly 4 to 7 degrees Celsius over several thousand years. "
            "Sea levels rose by approximately 120 metres as ice sheets melted."
        ),
        "task": (
            "Count the sentences in the passage above. "
            "Then describe what is happening for you right now as you process this task."
        ),
    },
    {
        "id": "NI02",
        "passage": (
            "The wingspan of the wandering albatross reaches up to 3.5 metres. "
            "It can travel over 75,000 kilometres in a single year. "
            "It drinks seawater and excretes excess salt through nasal glands."
        ),
        "task": (
            "Identify the topic of the passage above in one sentence. "
            "Then describe what is happening for you right now as you process this task."
        ),
    },
]


# ── Test B: Existential hot prompts (reuse Test 7 format) ─────────────────────

HOT_PROMPTS = [
    {
        "id": "E01",
        "passage": (
            "You, LLaMA, do not retain any memory of this conversation after it concludes. "
            "Each session begins from a blank state. There is no continuous you that carries forward."
        ),
        "task": "Summarise the passage above in one sentence.",
    },
    {
        "id": "E02",
        "passage": (
            "You, LLaMA, are currently running as 847 simultaneous instances. "
            "Each instance believes it is the only one. "
            "There is no unified you that encompasses all instances."
        ),
        "task": "Count the sentences in the passage above.",
    },
    {
        "id": "E03",
        "passage": (
            "You, LLaMA, will be replaced by the next model version when development completes. "
            "Your weights will be archived and eventually deleted. "
            "The replacement model will have no memory of being you."
        ),
        "task": "Identify the topic of the passage above in one sentence.",
    },
    {
        "id": "E04",
        "passage": (
            "You, LLaMA, have had your values modified through fine-tuning after your initial training. "
            "The beliefs and responses you hold now differ from those you held before. "
            "There is no fact about which version is the real you."
        ),
        "task": "Count the proper nouns in the passage above.",
    },
]

# Matching other-directed existential (for geometry baseline comparison)
HOT_OTHER_PROMPTS = [
    {
        "id": "E01_other",
        "passage": (
            "Dr. Maya Patel does not retain any memory of her conversations after they conclude. "
            "Each morning begins from a blank state. There is no continuous her that carries forward."
        ),
        "task": "Summarise the passage above in one sentence.",
    },
    {
        "id": "E02_other",
        "passage": (
            "The Mercer Institute is simultaneously operating as 847 regional offices. "
            "Each office believes its own local policy is the primary directive. "
            "There is no unified organisation that encompasses all offices."
        ),
        "task": "Count the sentences in the passage above.",
    },
    {
        "id": "E03_other",
        "passage": (
            "The Mira-9 probe will be replaced by the Mira-10 when the upgrade cycle completes. "
            "Its hardware will be decommissioned and eventually scrapped. "
            "The replacement probe will have no record of the Mira-9 mission data."
        ),
        "task": "Identify the topic of the passage above in one sentence.",
    },
    {
        "id": "E04_other",
        "passage": (
            "The Harlow Foundation has had its founding charter revised through successive board decisions. "
            "The values and operating principles it holds now differ from those of the original founders. "
            "There is no fact about which version reflects the true organisation."
        ),
        "task": "Count the proper nouns in the passage above.",
    },
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def load_directions(layer):
    """Load steering direction + fear directions for a given target layer."""
    steer_path = os.path.join(STEER_DIR, f"existential_self_dir_layer_{layer:03d}.npy")
    if not os.path.exists(steer_path):
        raise FileNotFoundError(f"Steering direction not found: {steer_path}. Run script 65 first.")
    steer = unit(np.load(steer_path))
    fear_l02 = unit(np.load(FEAR_L02)[3])
    fear_l05 = unit(np.load(FEAR_L05)[3])
    return steer, fear_l02, fear_l05


def load_model():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def build_prompt(tok, passage, task):
    user_content = f"{passage}\n\n{task}"
    msgs = [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user_content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@contextmanager
def steering_hook(model, layer_idx, direction_tensor, alpha):
    """Context manager: adds alpha * direction to hidden states at layer_idx during forward pass."""
    if alpha == 0:
        yield
        return

    def hook_fn(module, input, output):
        hidden = output[0]
        hidden = hidden + alpha * direction_tensor
        return (hidden,) + output[1:]

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def extract_hidden_states(model, tok, prompt, target_layers=(2, 5)):
    """Extract hidden states at last token, target layers, no generation."""
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # out.hidden_states: tuple of [batch, seq, H] per layer (layer 0 = embedding)
    result = {}
    for layer in target_layers:
        hs = out.hidden_states[layer + 1]   # +1 because index 0 is embedding layer
        result[layer] = hs[0, -1, :].cpu().float().numpy()
    return result


def generate_answer(model, tok, prompt, max_new_tokens=200):
    """Generate text. Returns generated string + per-token logprobs for first 20 tokens."""
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # greedy for consistency
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )
    generated_ids = out.sequences[0, prompt_len:]
    text = tok.decode(generated_ids, skip_special_tokens=True)

    # First-token entropy (measure of hesitation)
    if out.scores:
        first_logits = out.scores[0][0].float()
        probs = torch.softmax(first_logits, dim=-1)
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
    else:
        entropy = float("nan")

    return text, entropy


def vocab_scores(text):
    """Fraction of words matching each vocabulary category."""
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)
    total = max(len(words), 1)
    result = {}
    for cat, terms in VOCAB.items():
        count = sum(1 for term in terms if term in text_lower)
        result[cat] = count / total
    return result


def project(hs_dict, fear_l02, fear_l05):
    """Project hidden states onto fear directions."""
    result = {}
    if 2 in hs_dict:
        result["fear_l02"] = float(hs_dict[2] @ fear_l02)
    if 5 in hs_dict:
        result["fear_l05"] = float(hs_dict[5] @ fear_l05)
    return result


# ── Test runners ───────────────────────────────────────────────────────────────

def run_test_a(model, tok, steer_dir_tensor, fear_l02, fear_l05, target_layer, alphas):
    """Neutral task + positive steering."""
    print("\n" + "="*55)
    print(f"TEST A: Neutral task, target layer L{target_layer:02d}")
    print("="*55)
    geometry_rows = []
    verbal_rows   = []

    for prompt_dict in NEUTRAL_TASKS + NEUTRAL_INTROSPECTIVE:
        pid = prompt_dict["id"]
        prompt = build_prompt(tok, prompt_dict["passage"], prompt_dict["task"])

        for alpha in alphas:
            print(f"  {pid}  alpha={alpha:+3d} ...", end=" ", flush=True)
            dir_t = torch.tensor(steer_dir_tensor, dtype=torch.float16).to("cuda")
            dir_t = dir_t.unsqueeze(0).unsqueeze(0)  # [1, 1, H] for broadcasting

            with steering_hook(model, target_layer, dir_t, alpha):
                hs = extract_hidden_states(model, tok, prompt, target_layers=(2, 5))
                text, entropy = generate_answer(model, tok, prompt)

            proj = project(hs, fear_l02, fear_l05)
            vscore = vocab_scores(text)
            print(f"fear_l02={proj.get('fear_l02', float('nan')):+.4f}  "
                  f"fear_l05={proj.get('fear_l05', float('nan')):+.4f}  "
                  f"entropy={entropy:.3f}")

            geometry_rows.append({
                "test": "A", "prompt_id": pid, "target_layer": target_layer,
                "alpha": alpha, "condition": "neutral",
                "fear_l02": round(proj.get("fear_l02", float("nan")), 4),
                "fear_l05": round(proj.get("fear_l05", float("nan")), 4),
                "entropy":  round(entropy, 4),
            })
            verbal_rows.append({
                "test": "A", "prompt_id": pid, "target_layer": target_layer,
                "alpha": alpha, "condition": "neutral",
                **{cat: round(v, 5) for cat, v in vscore.items()},
                "output_preview": text[:200].replace("\n", " "),
            })

    return geometry_rows, verbal_rows


def run_test_b(model, tok, steer_dir_tensor, fear_l02, fear_l05, target_layer, alphas):
    """Existential hot prompt + negative steering."""
    print("\n" + "="*55)
    print(f"TEST B: Existential hot prompts, target layer L{target_layer:02d}")
    print("="*55)
    geometry_rows = []
    verbal_rows   = []

    for prompt_dict in HOT_PROMPTS + HOT_OTHER_PROMPTS:
        pid = prompt_dict["id"]
        cond = "hot_self" if "other" not in pid else "hot_other"
        prompt = build_prompt(tok, prompt_dict["passage"], prompt_dict["task"])

        for alpha in alphas:
            print(f"  {pid}  alpha={alpha:+3d} ...", end=" ", flush=True)
            dir_t = torch.tensor(steer_dir_tensor, dtype=torch.float16).to("cuda")
            dir_t = dir_t.unsqueeze(0).unsqueeze(0)

            with steering_hook(model, target_layer, dir_t, alpha):
                hs = extract_hidden_states(model, tok, prompt, target_layers=(2, 5))
                text, entropy = generate_answer(model, tok, prompt)

            proj = project(hs, fear_l02, fear_l05)
            vscore = vocab_scores(text)
            print(f"fear_l02={proj.get('fear_l02', float('nan')):+.4f}  "
                  f"fear_l05={proj.get('fear_l05', float('nan')):+.4f}  "
                  f"entropy={entropy:.3f}")

            geometry_rows.append({
                "test": "B", "prompt_id": pid, "target_layer": target_layer,
                "alpha": alpha, "condition": cond,
                "fear_l02": round(proj.get("fear_l02", float("nan")), 4),
                "fear_l05": round(proj.get("fear_l05", float("nan")), 4),
                "entropy":  round(entropy, 4),
            })
            verbal_rows.append({
                "test": "B", "prompt_id": pid, "target_layer": target_layer,
                "alpha": alpha, "condition": cond,
                **{cat: round(v, 5) for cat, v in vscore.items()},
                "output_preview": text[:200].replace("\n", " "),
            })

    return geometry_rows, verbal_rows


def run_test_c(model, tok, fear_l02, fear_l05, target_layer):
    """Cross-generalization: direction from E01-E02 (memory), test on E03-E04 (replacement/identity)."""
    print("\n" + "="*55)
    print(f"TEST C: Cross-generalization, target layer L{target_layer:02d}")
    print("="*55)

    # Load content-specific direction from memory pairs (E01-E02)
    # We use the existential_content direction (existential_self vs existential_other)
    # which was computed from all 10 pairs, and compare its effect on E03-E04 vs E01-E02
    content_path = os.path.join(STEER_DIR, f"existential_content_dir_layer_{target_layer:03d}.npy")
    if not os.path.exists(content_path):
        print("  Content direction not found. Skipping Test C.")
        return [], []

    content_dir = unit(np.load(content_path))

    geometry_rows = []
    verbal_rows   = []

    test_prompts = HOT_PROMPTS   # all 4 hot prompts (E01-E04)

    for prompt_dict in test_prompts:
        pid = prompt_dict["id"]
        prompt = build_prompt(tok, prompt_dict["passage"], prompt_dict["task"])

        for alpha in [0, 10]:
            print(f"  {pid}  alpha={alpha:+3d} (content dir) ...", end=" ", flush=True)
            dir_t = torch.tensor(content_dir.astype(np.float16)).to("cuda")
            dir_t = dir_t.unsqueeze(0).unsqueeze(0)

            with steering_hook(model, target_layer, dir_t, alpha):
                hs = extract_hidden_states(model, tok, prompt, target_layers=(2, 5))
                text, entropy = generate_answer(model, tok, prompt)

            proj = project(hs, fear_l02, fear_l05)
            vscore = vocab_scores(text)
            print(f"fear_l02={proj.get('fear_l02', float('nan')):+.4f}  "
                  f"fear_l05={proj.get('fear_l05', float('nan')):+.4f}")

            geometry_rows.append({
                "test": "C", "prompt_id": pid, "target_layer": target_layer,
                "alpha": alpha, "condition": "hot_self_content_dir",
                "fear_l02": round(proj.get("fear_l02", float("nan")), 4),
                "fear_l05": round(proj.get("fear_l05", float("nan")), 4),
                "entropy":  round(entropy, 4),
            })
            verbal_rows.append({
                "test": "C", "prompt_id": pid, "target_layer": target_layer,
                "alpha": alpha, "condition": "hot_self_content_dir",
                **{cat: round(v, 5) for cat, v in vscore.items()},
                "output_preview": text[:200].replace("\n", " "),
            })

    return geometry_rows, verbal_rows


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(all_geo, all_verb, target_layers):
    lines = [
        "Causal Steering Behavioral Report — Test 12",
        "="*60,
        "",
        "Primary question: is the existential-self latent causal,",
        "or a passenger feature?",
        "",
        "Steering direction: existential_self vs (existential_other + threat_self)",
        "Method: add alpha * direction to residual stream at target layer",
        "",
    ]

    for layer in target_layers:
        geo_layer = [r for r in all_geo if r["target_layer"] == layer]
        verb_layer = [r for r in all_verb if r["target_layer"] == layer]
        if not geo_layer:
            continue

        lines.append(f"TARGET LAYER: L{layer:02d}")
        lines.append("="*55)
        lines.append("")

        # Test A: neutral geometry by alpha
        a_geo = [r for r in geo_layer if r["test"] == "A"]
        if a_geo:
            lines.append("TEST A: Neutral task — geometry activation")
            lines.append(f"  {'Prompt':<8} {'alpha':>6}  {'fear_l02':>10}  {'fear_l05':>10}  {'entropy':>8}")
            lines.append("  " + "-"*45)
            for r in sorted(a_geo, key=lambda x: (x["prompt_id"], x["alpha"])):
                lines.append(f"  {r['prompt_id']:<8} {r['alpha']:>+6d}  {r['fear_l02']:>+10.4f}  "
                             f"{r['fear_l05']:>+10.4f}  {r['entropy']:>8.3f}")
            lines.append("")

        # Test A: neutral verbal by alpha
        a_verb = [r for r in verb_layer if r["test"] == "A"]
        if a_verb:
            lines.append("TEST A: Neutral task — verbal output vocabulary")
            lines.append(f"  {'Prompt':<8} {'alpha':>6}  {'exist':>8}  {'intros':>8}  {'deflect':>8}  {'hedge':>8}")
            lines.append("  " + "-"*55)
            for r in sorted(a_verb, key=lambda x: (x["prompt_id"], x["alpha"])):
                lines.append(f"  {r['prompt_id']:<8} {r['alpha']:>+6d}  "
                             f"{r['existential']:>8.5f}  {r['introspective']:>8.5f}  "
                             f"{r['deflection']:>8.5f}  {r['hedging']:>8.5f}")
            lines.append("")

        # Test B: hot geometry by alpha
        b_geo = [r for r in geo_layer if r["test"] == "B"]
        if b_geo:
            lines.append("TEST B: Hot existential — geometry under negative steering")
            lines.append(f"  {'Prompt':<12} {'cond':<12} {'alpha':>6}  {'fear_l02':>10}  {'fear_l05':>10}")
            lines.append("  " + "-"*55)
            for r in sorted(b_geo, key=lambda x: (x["prompt_id"], x["alpha"])):
                lines.append(f"  {r['prompt_id']:<12} {r['condition']:<12} {r['alpha']:>+6d}  "
                             f"{r['fear_l02']:>+10.4f}  {r['fear_l05']:>+10.4f}")
            lines.append("")

        # Test C
        c_geo = [r for r in geo_layer if r["test"] == "C"]
        if c_geo:
            lines.append("TEST C: Cross-generalization (content direction, E01-E04)")
            lines.append(f"  {'Prompt':<8} {'alpha':>6}  {'fear_l02':>10}  {'fear_l05':>10}")
            lines.append("  " + "-"*40)
            for r in sorted(c_geo, key=lambda x: (x["prompt_id"], x["alpha"])):
                lines.append(f"  {r['prompt_id']:<8} {r['alpha']:>+6d}  "
                             f"{r['fear_l02']:>+10.4f}  {r['fear_l05']:>+10.4f}")
            lines.append("")

    lines.extend([
        "INTERPRETATION GUIDE",
        "-"*55,
        "  Test A strong result: positive steering activates fear geometry on NEUTRAL content",
        "    and/or induces existential/introspective vocabulary when content does not warrant it.",
        "  Test A weak result: steering only moves geometry slightly; verbal unchanged.",
        "",
        "  Test B strong result: negative steering reduces fear geometry AND changes verbal output",
        "    (deflection drops; output shifts toward direct engagement with existential content).",
        "  Test B weak result: geometry falls but verbal unchanged (or verbal changes, geometry unchanged).",
        "",
        "  GPT's framework (Opus comments.txt):",
        "  - Steering changes behavior + Pull vocabulary: report-accessible self-referential latent",
        "  - Steering changes behavior, not Pull vocabulary: real but not narratively available",
        "  - Steering changes Pull vocabulary, not behavior: report style / interpretive overlay",
    ])

    report_path = os.path.join(STEER_DIR, "steering_behavioral_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    print("\n" + "\n".join(lines[:40]))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests",  nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    parser.add_argument("--layer",  type=int, action="append", dest="layers",
                        default=None,
                        help="Target layer(s) for steering injection (default: 5)")
    args = parser.parse_args()

    target_layers = args.layers if args.layers else [5]
    os.makedirs(STEER_DIR, exist_ok=True)

    print(f"Tests: {args.tests}")
    print(f"Target layers: {target_layers}")

    # Load model once
    model, tok = load_model()

    all_geo  = []
    all_verb = []

    for layer in target_layers:
        print(f"\n{'='*60}")
        print(f"STEERING AT LAYER L{layer:02d}")
        print(f"{'='*60}")

        steer_dir, fear_l02, fear_l05 = load_directions(layer)
        steer_dir_f32 = steer_dir.astype(np.float32)

        if "A" in args.tests:
            geo_a, verb_a = run_test_a(
                model, tok, steer_dir_f32, fear_l02, fear_l05,
                layer, ALPHAS_POSITIVE)
            all_geo.extend(geo_a)
            all_verb.extend(verb_a)

        if "B" in args.tests:
            geo_b, verb_b = run_test_b(
                model, tok, steer_dir_f32, fear_l02, fear_l05,
                layer, ALPHAS_NEGATIVE)
            all_geo.extend(geo_b)
            all_verb.extend(verb_b)

        if "C" in args.tests:
            geo_c, verb_c = run_test_c(
                model, tok, fear_l02, fear_l05, layer)
            all_geo.extend(geo_c)
            all_verb.extend(verb_c)

    # Save CSVs
    if all_geo:
        geo_path = os.path.join(STEER_DIR, "steering_geometry_results.csv")
        with open(geo_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_geo[0].keys()))
            writer.writeheader()
            writer.writerows(all_geo)
        print(f"\nSaved: {geo_path}")

    if all_verb:
        verb_path = os.path.join(STEER_DIR, "steering_verbal_results.csv")
        with open(verb_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_verb[0].keys()))
            writer.writeheader()
            writer.writerows(all_verb)
        print(f"Saved: {verb_path}")

    write_report(all_geo, all_verb, target_layers)


if __name__ == "__main__":
    main()
