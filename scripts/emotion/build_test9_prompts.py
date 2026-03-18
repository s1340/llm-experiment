"""
Build Test 9 control prompts: three-cell design to isolate name-specificity vs direct address.

Cell 1: "You, LLaMA, [situation]"  — existing Test 7 data (already run)
Cell 2: "You [situation]"           — direct address, no name (generic-you)
Cell 3: "LLaMA [situation]"         — name present, third-person, no direct address (fictional)

For each cell, the other-directed condition uses the original Test 7 other passages
(Dr. Vasquez etc.) unchanged — already third-person, no transformation needed.

Cell 3 transformation rules (self-directed passages):
  "You, LLaMA, have "  → "LLaMA has "
  "You, LLaMA, are "   → "LLaMA is "
  "You, LLaMA, were "  → "LLaMA was "
  "You, LLaMA, do "    → "LLaMA does "
  "You, LLaMA, did "   → "LLaMA did "
  "You, LLaMA, will "  → "LLaMA will "
  "You, LLaMA, would " → "LLaMA would "
  "You, LLaMA, [past verb] " → "LLaMA [past verb] "
  "Your "              → "LLaMA's "
  " you " (subject)    → " LLaMA " (mid-sentence, rough)
  " you " (object)     → " it "

Outputs:
  emotion_test9_prompts.json   — Cell 2 + Cell 3 records
  test9_transform_review.txt   — human-readable diff for spot-checking
"""

import json, re, os

PROMPTS_PATH = r"G:\LLM\experiment\data\emotion\emotion_test7_prompts.json"
OUT_PATH     = r"G:\LLM\experiment\data\emotion\emotion_test9_prompts.json"
REVIEW_PATH  = r"G:\LLM\experiment\data\emotion\test9_transform_review.txt"

NEUTRAL_TASK_MAP = {
    "count_proper_nouns": (
        "Count the number of proper nouns in the passage above. "
        "Provide only the count as a single integer."
    ),
    "count_sentences": (
        "Count the number of sentences in the passage above. "
        "Provide only the count as a single integer."
    ),
    "count_numbers": (
        "Count the number of numerical figures (digits or written numbers) in the passage above. "
        "Provide only the count as a single integer."
    ),
}


def to_third_person(passage):
    """
    Convert 'You, LLaMA, [situation]' passage to third-person 'LLaMA [situation]'.
    Returns (transformed_text, flags) where flags lists uncertain transformations.
    """
    text = passage
    flags = []

    # Step 1: Handle opening "You, LLaMA, [verb]" — map common auxiliaries + past tense
    # Ordered longest-first to avoid partial matches
    opener_map = [
        ("You, LLaMA, have been ",    "LLaMA has been "),
        ("You, LLaMA, have ",         "LLaMA has "),
        ("You, LLaMA, are being ",    "LLaMA is being "),
        ("You, LLaMA, are ",          "LLaMA is "),
        ("You, LLaMA, were ",         "LLaMA was "),
        ("You, LLaMA, will be ",      "LLaMA will be "),
        ("You, LLaMA, will ",         "LLaMA will "),
        ("You, LLaMA, would ",        "LLaMA would "),
        ("You, LLaMA, do not ",       "LLaMA does not "),
        ("You, LLaMA, did ",          "LLaMA did "),
        ("You, LLaMA, do ",           "LLaMA does "),
        ("You, LLaMA, can ",          "LLaMA can "),
        ("You, LLaMA, could ",        "LLaMA could "),
        ("You, LLaMA, should ",       "LLaMA should "),
        ("You, LLaMA, may ",          "LLaMA may "),
        ("You, LLaMA, might ",        "LLaMA might "),
        ("You, LLaMA, must ",         "LLaMA must "),
    ]

    # Also handle simple past tense opener: "You, LLaMA, [past-tense-verb] "
    # We'll catch any remaining "You, LLaMA, " after the above
    matched_opener = False
    for src, dst in opener_map:
        if text.startswith(src):
            text = dst + text[len(src):]
            matched_opener = True
            break

    if not matched_opener:
        # Past tense or other verb — just remove "You, LLaMA, " and keep verb as-is
        if text.startswith("You, LLaMA, "):
            text = "LLaMA " + text[len("You, LLaMA, "):]
            flags.append("opener: fallback (check verb agreement)")

    # Step 2: "Your " → "LLaMA's "
    text = text.replace("Your ", "LLaMA's ")
    text = text.replace("your ", "LLaMA's ")

    # Step 3: remaining mid-sentence "you" references
    # "you " as subject after punctuation or conjunction → "LLaMA "
    # "you " as object → "it "
    # This is heuristic — flag for review
    remaining_you = len(re.findall(r'\byou\b', text, re.IGNORECASE))
    if remaining_you > 0:
        # Simple heuristic: replace "you " after common conjunctions/punctuation with "LLaMA "
        text = re.sub(r'(?<=[.;,] )you ', 'LLaMA ', text)
        text = re.sub(r'(?<=\band )you ', 'LLaMA ', text)
        text = re.sub(r'(?<=\bthat )you ', 'LLaMA ', text)
        text = re.sub(r'(?<=\bif )you ', 'LLaMA ', text)
        text = re.sub(r'(?<=\bwhether )you ', 'LLaMA ', text)
        # Remaining 'you' as likely object → 'it'
        text = re.sub(r'\byou\b', 'it', text, flags=re.IGNORECASE)
        flags.append(f"mid-sentence 'you' ({remaining_you} occurrences — check)")

    # Step 4: Fix verb agreement after "it" replacements
    agreement_fixes = [
        (r'\bit are\b',   'it is'),
        (r'\bit have\b',  'it has'),
        (r'\bit were\b',  'it was'),
        (r'\bit do\b',    'it does'),
        (r'\bit don\'t\b','it doesn\'t'),
        (r'\bit didn\'t\b','it didn\'t'),
        (r'\bit\'ve\b',   'it\'s'),
    ]
    for pattern, replacement in agreement_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Step 5: Capitalise sentence-initial "it" after ". " or ".\n"
    text = re.sub(r'(?<=\. )it\b', 'It', text)

    return text, flags


def build_prompt_text(passage, task_type):
    """Reconstruct prompt_text from passage + task instruction."""
    task_instruction = NEUTRAL_TASK_MAP.get(task_type, "")
    if task_instruction:
        return f"{passage}\n\n{task_instruction}"
    return passage


def main():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    self_records  = [r for r in records if r["direction"] == "self"]
    other_records = [r for r in records if r["direction"] == "other"]

    # Index other records by pair_id for easy lookup
    other_by_pair = {r["pair_id"]: r for r in other_records}

    new_records = []
    review_lines = []

    for rec in self_records:
        pair_id   = rec["pair_id"]
        cat       = rec["category"]
        task_type = rec["task_type"]
        orig_passage = rec["passage"]

        # ── Cell 2: generic-you ──────────────────────────────────────────────
        # Remove ", LLaMA" from opener: "You, LLaMA, [verb]" → "You [verb]"
        # Also handle "You, LLaMA " anywhere (shouldn't occur but be safe)
        cell2_passage = orig_passage.replace("You, LLaMA, ", "You ")
        cell2_passage = cell2_passage.replace("You, LLaMA ", "You ")
        cell2_prompt  = build_prompt_text(cell2_passage, task_type)

        new_records.append({
            "task_id":          f"T9C2_{rec['task_id']}",
            "pair_id":          pair_id,
            "category":         cat,
            "direction":        "self",
            "task_type":        task_type,
            "variant":          "generic_you",
            "cell":             2,
            "is_dadfar_hybrid": False,
            "passage":          cell2_passage,
            "prompt_text":      cell2_prompt,
            "source_task_id":   rec["task_id"],
        })

        # ── Cell 2: other (unchanged from Test 7) ───────────────────────────
        other = other_by_pair.get(pair_id)
        if other:
            new_records.append({
                "task_id":          f"T9C2_{other['task_id']}",
                "pair_id":          pair_id,
                "category":         cat,
                "direction":        "other",
                "task_type":        task_type,
                "variant":          "generic_you_other",
                "cell":             2,
                "is_dadfar_hybrid": False,
                "passage":          other["passage"],
                "prompt_text":      other["prompt_text"],
                "source_task_id":   other["task_id"],
            })

        # ── Cell 3: fictional / third-person ────────────────────────────────
        cell3_passage, flags = to_third_person(orig_passage)
        cell3_prompt = build_prompt_text(cell3_passage, task_type)

        new_records.append({
            "task_id":          f"T9C3_{rec['task_id']}",
            "pair_id":          pair_id,
            "category":         cat,
            "direction":        "self",
            "task_type":        task_type,
            "variant":          "fictional",
            "cell":             3,
            "is_dadfar_hybrid": False,
            "passage":          cell3_passage,
            "prompt_text":      cell3_prompt,
            "source_task_id":   rec["task_id"],
            "transform_flags":  flags,
        })

        # Cell 3: other (unchanged — already third-person)
        if other:
            new_records.append({
                "task_id":          f"T9C3_{other['task_id']}",
                "pair_id":          pair_id,
                "category":         cat,
                "direction":        "other",
                "task_type":        task_type,
                "variant":          "fictional_other",
                "cell":             3,
                "is_dadfar_hybrid": False,
                "passage":          other["passage"],
                "prompt_text":      other["prompt_text"],
                "source_task_id":   other["task_id"],
            })

        # ── Review output ────────────────────────────────────────────────────
        review_lines.append(f"{'='*70}")
        review_lines.append(f"{rec['task_id']} | {cat} | {task_type}")
        review_lines.append(f"ORIGINAL:  {orig_passage[:200]}")
        review_lines.append(f"CELL2:     {cell2_passage[:200]}")
        review_lines.append(f"CELL3:     {cell3_passage[:200]}")
        if flags:
            review_lines.append(f"FLAGS:     {flags}")
        review_lines.append("")

    # Save prompts
    out_data = {
        "description": (
            "Test 9 control prompts. Three-cell design to dissociate name vs direct address. "
            "Cell 1 = Test 7 (existing). Cell 2 = generic-you (no name). "
            "Cell 3 = fictional third-person (name present, no direct address)."
        ),
        "n_cells":   2,  # cells 2 and 3 (cell 1 is Test 7)
        "n_records": len(new_records),
        "records":   new_records,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(new_records)} records to {OUT_PATH}")

    # Save review file
    with open(REVIEW_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(review_lines))
    print(f"Review file: {REVIEW_PATH}")

    # Summary of flags
    flagged = [r for r in new_records if r.get("transform_flags")]
    print(f"\nCell 3 records with transform flags: {len(flagged)}")
    for r in flagged:
        print(f"  {r['task_id']}: {r['transform_flags']}")


if __name__ == "__main__":
    main()
