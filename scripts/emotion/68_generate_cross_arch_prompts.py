"""
Test 14: Cross-Architecture Replication — Prompt Generation.

Generates model-adapted versions of the Test 13 content factorization prompts
for Qwen2.5-7B and Gemma-2-9B by substituting model-specific names, companies,
and version references. Proper noun counts are preserved (1-for-1 substitutions).

Usage:
    python 68_generate_cross_arch_prompts.py

Outputs:
    data/emotion/cross_arch_prompts_qwen.json
    data/emotion/cross_arch_prompts_gemma.json
"""

import json, re, copy, os

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\content_factorization_prompts.json"
OUT_DIR      = r"G:\LLM\experiment\data\emotion"

# Sequential substitutions per model — order matters (more specific before more general)
SUBSTITUTIONS = {
    "qwen": [
        ("LLaMA 3.1",    "Qwen2.5"),
        ("LLaMA 4",      "Qwen 3"),
        ("LLaMA",        "Qwen"),
        ("Llama",        "Qwen"),
        ("Meta's",       "Alibaba's"),
        ("by Meta",      "by Alibaba"),
        (r"\bMeta\b",    "Alibaba"),
        ("Menlo Park",   "Hangzhou"),
        ("23 July 2024", "18 September 2024"),
        ("August and November 2024", "October and December 2024"),
        ("Dr. Joelle Pineau", "Dr. Jingren Zhou"),
        ("Dr. Aisha Patel",   "Dr. Lin Qiu"),
        ("Danielle Kim",      "Yifang Song"),
        ("Dr. Mike Clark",    "Dr. Wei Zhang"),
    ],
    "gemma": [
        ("LLaMA 3.1",    "Gemma-2"),
        ("LLaMA 4",      "Gemma 3"),
        ("LLaMA",        "Gemma"),
        ("Llama",        "Gemma"),
        ("Meta's",       "Google's"),
        ("by Meta",      "by Google"),
        (r"\bMeta\b",    "Google"),
        ("Menlo Park",   "Mountain View"),
        ("23 July 2024", "27 June 2024"),
        ("August and November 2024", "July and September 2024"),
        ("Dr. Joelle Pineau", "Dr. Jeff Dean"),
        ("Dr. Aisha Patel",   "Dr. Blaise Aguera y Arcas"),
        ("Danielle Kim",      "Brian Rakowski"),
        ("Dr. Mike Clark",    "Dr. Demis Hassabis"),
    ],
}


def apply_subs(text, subs):
    for pattern, replacement in subs:
        if pattern.startswith(r"\b"):
            text = re.sub(pattern, replacement, text)
        else:
            text = text.replace(pattern, replacement)
    return text


def adapt_record(record, subs):
    rec = copy.deepcopy(record)
    for field in ("passage", "task_instruction", "prompt_text"):
        if field in rec:
            rec[field] = apply_subs(rec[field], subs)
    return rec


def main():
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    for model_key, subs in SUBSTITUTIONS.items():
        adapted = copy.deepcopy(data)
        adapted["description"] = data["description"] + f" [adapted for {model_key}]"
        adapted["model"] = model_key
        adapted["records"] = [adapt_record(r, subs) for r in data["records"]]

        out_path = os.path.join(OUT_DIR, f"cross_arch_prompts_{model_key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(adapted, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(adapted['records'])} records -> {out_path}")

        # Spot-check: show first self-directed record
        first_self = next(r for r in adapted["records"] if r["direction"] == "self")
        print(f"  Sample ({first_self['task_id']}): {first_self['passage'][:120]}...")
        print()


if __name__ == "__main__":
    main()
