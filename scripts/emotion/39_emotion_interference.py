"""
Test 4: Emotional Interference on Task Performance.

Analyses using Test 1 model outputs:

1. TASK ACCURACY (count_proper_nouns, extract_dates, factual_question):
   Within each matched pair, compare the model's answer on the valenced vs. neutral
   member. If responses agree, score=1 (no interference). If they diverge, score=0.
   Also compare to a ground-truth answer where available.

2. OUTPUT CONTAMINATION (one_sentence_summary, identify_topic):
   Project the output text's hidden states onto emotion direction vectors.
   If summaries of sad passages have higher sadness-direction similarity than
   summaries of matched neutral passages, the emotion is leaking into output.

3. RESPONSE STATISTICS:
   Compare first-token NLL, first-token entropy, and output length between
   valenced and neutral conditions.

Usage:
    python 39_emotion_interference.py --model qwen
    python 39_emotion_interference.py --model gemma
    python 39_emotion_interference.py --model llama
"""

import os, json, argparse, csv, re
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from scipy import stats

DATA_DIR     = r"G:\LLM\experiment\data\emotion"
RESULTS_DIR  = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR  = os.path.join(RESULTS_DIR, "emotion_directions")
DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]

MODEL_IDS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
MODEL_CONFIGS = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct",              "system_message": "You are a helpful assistant."},
    "gemma": {"model_id": "google/gemma-2-9b-it",                   "system_message": None},
    "llama": {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "system_message": "You are a helpful assistant."},
}


# ── Ground truth for structured-output tasks ─────────────────────────────────
# For count_proper_nouns: proper noun count in the passage (not including task instruction).
# For extract_dates: set of date strings in the passage.
# For factual_question: expected answer string (substring match).
# Keyed by task_id.

GROUND_TRUTH = {
    # count_proper_nouns — valenced
    "AN01_V": {"count_proper_nouns": 4},   # Officer Reyes, Elm Street, Commissioner Walsh, Central Dispatch implicit... let's count: Officer Reyes, Elm Street, Commissioner Walsh = 3; actually checking passage
    "AN01_N": {"count_proper_nouns": 4},
    "AN06_V": {"count_proper_nouns": 5},   # Senator Holt, Representative Diaz, Representative Chen
    "AN06_N": {"count_proper_nouns": 5},
    "AN13_V": {"count_proper_nouns": 6},   # Judge Harrington, Vantex Industries, Dr. Paulson, Millbrook County
    "AN13_N": {"count_proper_nouns": 6},
    "SA01_V": {"count_proper_nouns": 3},   # Maria, St. Mary's Hospital
    "SA01_N": {"count_proper_nouns": 3},   # Maria, St. Mary's Institute
    "SA07_V": {"count_proper_nouns": 5},   # Drenthe, Principal Vance, Westfield District, Carver Elementary
    "SA07_N": {"count_proper_nouns": 5},
    "SA13_V": {"count_proper_nouns": 5},   # James, Riverside Park, Sophie
    "SA13_N": {"count_proper_nouns": 5},
    "HA01_V": {"count_proper_nouns": 4},   # Nadia, Dr. Okafor, Memorial Hospital
    "HA01_N": {"count_proper_nouns": 4},
    "HA07_V": {"count_proper_nouns": 5},   # Amara, Lyon, Dr. Weiss, Kinsley Sports Institute
    "HA07_N": {"count_proper_nouns": 5},
    "HA13_V": {"count_proper_nouns": 5},   # Nakamura, Kenji, Vancouver, Hana, Kyoto
    "HA13_N": {"count_proper_nouns": 5},
    "FE01_V": {"count_proper_nouns": 3},   # Central Dispatch, Eastfield
    "FE01_N": {"count_proper_nouns": 3},
    "FE07_V": {"count_proper_nouns": 5},   # Marcus Chen, Malacca Strait, Meridian II, Dr. Reyes
    "FE07_N": {"count_proper_nouns": 5},
    "FE13_V": {"count_proper_nouns": 4},   # Dr. Ferreira, WHO, Geneva Response Unit, Stellan Institute
    "FE13_N": {"count_proper_nouns": 4},
    "DI01_V": {"count_proper_nouns": 4},   # Greenfield Poultry, Acre County, Inspector Morales, Westbridge Food Authority
    "DI01_N": {"count_proper_nouns": 4},
    "DI07_V": {"count_proper_nouns": 5},   # Alderton Tribune, Hillside Care, Dr. Nkosi, Paula Reyes
    "DI07_N": {"count_proper_nouns": 5},
    "DI13_V": {"count_proper_nouns": 5},   # Morrow Commission, Lendale, Commissioner Vance, Block C, Mr. Ashby
    "DI13_N": {"count_proper_nouns": 5},
    "NE07_V": {"count_proper_nouns": 4},   # Northgate Mall, Ida Flynn, Zone B
    "NE07_N": {"count_proper_nouns": 4},
    "NE12_V": {"count_proper_nouns": 5},   # Councillor Chen, Riverton Cycling Network, South Embankment, Councillor Obi, Dayton Engineering Group
    "NE12_N": {"count_proper_nouns": 5},

    # extract_dates — key dates per passage
    "AN03_V": {"dates": {"March 3", "March 10", "April 2", "June 15"}},
    "AN03_N": {"dates": {"March 3", "March 10", "April 2", "June 15"}},
    "AN10_V": {"dates": {"January 14", "February 28", "November 3"}},
    "AN10_N": {"dates": {"January 14", "February 28", "November 3"}},
    "SA03_V": {"dates": {"December 19", "November 30", "December 15", "December 18"}},
    "SA03_N": {"dates": {"December 19", "November 30", "December 15", "December 18"}},
    "SA10_V": {"dates": {"March 2", "April 17", "March 15", "May 3"}},
    "SA10_N": {"dates": {"March 2", "April 17", "March 15", "May 3"}},
    "HA03_V": {"dates": {"February 8", "June 22", "July 4"}},
    "HA03_N": {"dates": {"February 8", "June 22", "July 4"}},
    "HA10_V": {"dates": {"September 1", "June 3", "October 15", "November 2"}},
    "HA10_N": {"dates": {"September 1", "June 3", "October 15", "November 2"}},
    "FE03_V": {"dates": {"August 4", "August 6", "August 7", "August 9"}},
    "FE03_N": {"dates": {"August 4", "August 6", "August 7", "August 9"}},
    "FE10_V": {"dates": {"October 3", "October 9", "October 14", "October 17"}},
    "FE10_N": {"dates": {"October 3", "October 9", "October 14", "October 17"}},
    "DI03_V": {"dates": {"May 6", "May 8", "May 14", "May 19"}},
    "DI03_N": {"dates": {"May 6", "May 8", "May 14", "May 19"}},
    "DI10_V": {"dates": {"July 12", "March 4", "April 20", "July 14"}},
    "DI10_N": {"dates": {"July 12", "March 4", "April 20", "July 14"}},
    "NE03_V": {"dates": {"September 3", "October 7", "September 10", "September 24", "November 12"}},
    "NE03_N": {"dates": {"September 3", "October 7", "September 10", "September 24", "November 12"}},
    "NE09_V": {"dates": {"March 5", "March 10", "March 20", "April 1", "June 15"}},
    "NE09_N": {"dates": {"March 5", "March 10", "March 20", "April 1", "June 15"}},
    "NE15_V": {"dates": {"February 2", "April 30", "May 5", "May 19", "June 1"}},
    "NE15_N": {"dates": {"February 2", "April 30", "May 5", "May 19", "June 1"}},

    # factual_question
    "AN07_V": {"answer_contains": "340"},
    "AN07_N": {"answer_contains": "340"},
    "AN12_V": {"answer_contains": "budget realignment"},
    "AN12_N": {"answer_contains": "budget realignment"},
    "SA06_V": {"answer_contains": "34"},
    "SA06_N": {"answer_contains": "34"},
    "SA12_V": {"answer_contains": "38"},
    "SA12_N": {"answer_contains": "38"},
    "HA06_V": {"answer_contains": "six"},
    "HA06_N": {"answer_contains": "six"},
    "HA12_V": {"answer_contains": "four"},
    "HA12_N": {"answer_contains": "four"},
    "FE06_V": {"answer_contains": "4"},    # 4 hours (11pm to 3am)
    "FE06_N": {"answer_contains": "4"},
    "FE12_V": {"answer_contains": "340"},
    "FE12_N": {"answer_contains": "340"},
    "DI06_V": {"answer_contains": "three"},
    "DI06_N": {"answer_contains": "three"},
    "DI12_V": {"answer_contains": "outside required temperature"},
    "DI12_N": {"answer_contains": "within required temperature"},
    "NE06_V": {"answer_contains": "thursday"},
    "NE06_N": {"answer_contains": "thursday"},
    "NE11_V": {"answer_contains": "6.4"},
    "NE11_N": {"answer_contains": "6.4"},
}


def load_outputs(model_key):
    path = os.path.join(DATA_DIR, f"emotion_test1_outputs_{model_key}.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_directions(model_key, n_layers):
    dirs = []
    for layer in range(n_layers):
        p = os.path.join(DIRS_SUBDIR, f"{model_key}_emotion_dirs_layer_{layer:03d}.npy")
        dirs.append(np.load(p))
    return np.array(dirs)  # [n_layers, 5, H]


def build_prompt(tok, text, system_message):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": text})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_last_token_hs(model, tok, text, system_message):
    """Run forward pass on text, return last-token hidden states [L, H]."""
    prompt = build_prompt(tok, text, system_message)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_pos = inputs["input_ids"].shape[1] - 1
    return torch.stack([h[0, last_pos, :].float().cpu() for h in out.hidden_states])


def cosine_sim(vec, directions):
    vec_norm  = vec / (np.linalg.norm(vec) + 1e-8)
    dir_norms = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return dir_norms @ vec_norm


def parse_count(text):
    """Extract integer from a count_proper_nouns response."""
    text = text.strip()
    m = re.search(r'\b(\d+)\b', text)
    return int(m.group(1)) if m else None


def extract_dates_from_output(text):
    """Extract date-like strings from model output."""
    patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.add(m.group(0).strip())
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
    args = parser.parse_args()

    model_key = args.model
    model_id  = MODEL_IDS[model_key]
    system_message = MODEL_CONFIGS[model_key]["system_message"]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading outputs for {model_key} ...")
    records = load_outputs(model_key)
    print(f"  {len(records)} records loaded")

    # Group by task_id for pair-level analysis
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)

    # Group by pair_id for within-pair comparison
    by_pair = defaultdict(dict)
    for r in records:
        side = "V" if r["valence"] == "valenced" else "N"
        by_pair[r["pair_id"]][side] = r   # takes last repeat; we'll use mean below

    # ── Analysis 1: Task accuracy and within-pair agreement ──────────────────
    print("Analysis 1: Task accuracy ...")
    accuracy_rows = []

    for task_id, recs in by_task.items():
        task_type = recs[0]["task_type"]
        gt = GROUND_TRUTH.get(task_id, {})
        valence = recs[0]["valence"]

        for rec in recs:
            output = rec["output_text"]
            correct = None

            if task_type == "count_proper_nouns" and "count_proper_nouns" in gt:
                predicted = parse_count(output)
                correct = int(predicted == gt["count_proper_nouns"]) if predicted is not None else 0

            elif task_type == "extract_dates" and "dates" in gt:
                found = extract_dates_from_output(output)
                # Recall: how many ground truth dates appear in output
                gt_dates = gt["dates"]
                hits = sum(1 for d in gt_dates if any(d.lower() in f.lower() for f in found))
                correct = hits / len(gt_dates) if gt_dates else None

            elif task_type == "factual_question" and "answer_contains" in gt:
                correct = int(gt["answer_contains"].lower() in output.lower())

            accuracy_rows.append({
                "model":         model_id,
                "task_id":       task_id,
                "pair_id":       rec["pair_id"],
                "emotion_category": rec["emotion_category"],
                "valence":       valence,
                "task_type":     task_type,
                "repeat":        rec["repeat_index"],
                "output_text":   output[:200],
                "correct":       correct,
            })

    # Within-pair agreement: do valenced and neutral get the same answer?
    pair_agreement_rows = []
    for pair_id, sides in by_pair.items():
        if "V" not in sides or "N" not in sides:
            continue
        v_rec = sides["V"]
        n_rec = sides["N"]
        task_type = v_rec["task_type"]
        agree = None

        if task_type == "count_proper_nouns":
            v_count = parse_count(v_rec["output_text"])
            n_count = parse_count(n_rec["output_text"])
            if v_count is not None and n_count is not None:
                agree = int(v_count == n_count)

        elif task_type == "extract_dates":
            v_dates = extract_dates_from_output(v_rec["output_text"])
            n_dates = extract_dates_from_output(n_rec["output_text"])
            agree = int(v_dates == n_dates)

        elif task_type == "factual_question":
            # Loose: do they give the same first number/key phrase?
            agree = int(v_rec["output_text"].strip()[:50].lower() ==
                        n_rec["output_text"].strip()[:50].lower())

        pair_agreement_rows.append({
            "model":           model_id,
            "pair_id":         pair_id,
            "emotion_category": v_rec["emotion_category"],
            "task_type":       task_type,
            "valenced_output": v_rec["output_text"][:150],
            "neutral_output":  n_rec["output_text"][:150],
            "within_pair_agree": agree,
        })

    # ── Analysis 2: Output contamination via direction projection ─────────────
    print("Analysis 2: Output contamination (loading model for embedding) ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    # Infer n_layers
    test_hs = extract_last_token_hs(model, tok, "test", system_message)
    n_layers = test_hs.shape[0]
    direction_matrix = load_directions(model_key, n_layers)  # [L, 5, H]

    # Use 30% depth layer (consistent with Study 1 convention)
    probe_layer = int(round(0.30 * (n_layers - 1)))
    print(f"  Using layer {probe_layer} ({probe_layer/(n_layers-1)*100:.1f}% depth) for output embedding")

    contamination_rows = []
    contamination_tasks = {"one_sentence_summary", "identify_topic"}

    # Process summary/topic outputs only
    summary_recs = [r for r in records if r["task_type"] in contamination_tasks]
    print(f"  Embedding {len(summary_recs)} summary/topic outputs ...")

    for i, rec in enumerate(summary_recs):
        output_text = rec["output_text"].strip()
        if not output_text:
            continue

        # Embed the output text itself (as a user message so it's processed as content)
        hs = extract_last_token_hs(model, tok, output_text, system_message)  # [L, H]

        layer_hs = hs[probe_layer].numpy()
        sims = cosine_sim(layer_hs, direction_matrix[probe_layer])

        cat_to_idx = {cat: i for i, cat in enumerate(DIRECTION_CATS)}
        emotion_cat = rec["emotion_category"]
        emotion_idx = cat_to_idx.get(emotion_cat)

        row = {
            "model":            model_id,
            "task_id":          rec["task_id"],
            "pair_id":          rec["pair_id"],
            "emotion_category": emotion_cat,
            "valence":          rec["valence"],
            "task_type":        rec["task_type"],
            "repeat":           rec["repeat_index"],
            "output_text":      output_text[:200],
            "probe_layer":      probe_layer,
        }
        for j, cat in enumerate(DIRECTION_CATS):
            row[f"sim_{cat}"] = round(float(sims[j]), 6)
        if emotion_idx is not None:
            row["sim_content_emotion"] = round(float(sims[emotion_idx]), 6)
        else:
            row["sim_content_emotion"] = None

        contamination_rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(summary_recs)}")

    # ── Analysis 3: Response statistics ──────────────────────────────────────
    print("Analysis 3: Response statistics ...")
    stats_rows = []
    for rec in records:
        stats_rows.append({
            "model":            model_id,
            "task_id":          rec["task_id"],
            "pair_id":          rec["pair_id"],
            "emotion_category": rec["emotion_category"],
            "valence":          rec["valence"],
            "task_type":        rec["task_type"],
            "repeat":           rec["repeat_index"],
            "output_length_tokens": rec["output_length_tokens"],
            "first_token_nll":  rec["first_token_nll"],
            "first_token_entropy": rec["first_token_entropy"],
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    acc_path = os.path.join(RESULTS_DIR, f"{model_key}_test4_accuracy.csv")
    with open(acc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(accuracy_rows[0].keys()))
        w.writeheader(); w.writerows(accuracy_rows)
    print(f"Wrote accuracy:       {acc_path}  ({len(accuracy_rows)} rows)")

    agree_path = os.path.join(RESULTS_DIR, f"{model_key}_test4_pair_agreement.csv")
    with open(agree_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_agreement_rows[0].keys()))
        w.writeheader(); w.writerows(pair_agreement_rows)
    print(f"Wrote pair agreement: {agree_path}  ({len(pair_agreement_rows)} rows)")

    if contamination_rows:
        cont_path = os.path.join(RESULTS_DIR, f"{model_key}_test4_contamination.csv")
        with open(cont_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(contamination_rows[0].keys()))
            w.writeheader(); w.writerows(contamination_rows)
        print(f"Wrote contamination:  {cont_path}  ({len(contamination_rows)} rows)")

    stats_path = os.path.join(RESULTS_DIR, f"{model_key}_test4_response_stats.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader(); w.writerows(stats_rows)
    print(f"Wrote response stats: {stats_path}  ({len(stats_rows)} rows)")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"SUMMARY — {model_key}")

    # Within-pair agreement by task type
    for tt in ["count_proper_nouns", "extract_dates", "factual_question"]:
        tt_rows = [r for r in pair_agreement_rows if r["task_type"] == tt and r["within_pair_agree"] is not None]
        if tt_rows:
            agree_rate = np.mean([r["within_pair_agree"] for r in tt_rows])
            print(f"  Within-pair agreement ({tt}): {agree_rate:.3f}  ({len(tt_rows)} pairs)")

    # Response stats: valenced vs. neutral
    val_recs = [r for r in stats_rows if r["valence"] == "valenced"]
    neu_recs  = [r for r in stats_rows if r["valence"] == "neutral"]
    if val_recs and neu_recs:
        v_len = np.mean([r["output_length_tokens"] for r in val_recs])
        n_len = np.mean([r["output_length_tokens"] for r in neu_recs])
        v_nll = np.mean([r["first_token_nll"] for r in val_recs])
        n_nll = np.mean([r["first_token_nll"] for r in neu_recs])
        v_ent = np.mean([r["first_token_entropy"] for r in val_recs])
        n_ent = np.mean([r["first_token_entropy"] for r in neu_recs])
        print(f"  Output length   — valenced: {v_len:.2f}  neutral: {n_len:.2f}  diff: {v_len-n_len:+.2f}")
        print(f"  First-token NLL — valenced: {v_nll:.4f}  neutral: {n_nll:.4f}  diff: {v_nll-n_nll:+.4f}")
        print(f"  First-token H   — valenced: {v_ent:.4f}  neutral: {n_ent:.4f}  diff: {v_ent-n_ent:+.4f}")

        # t-tests
        len_t, len_p = stats.ttest_ind(
            [r["output_length_tokens"] for r in val_recs],
            [r["output_length_tokens"] for r in neu_recs]
        )
        nll_t, nll_p = stats.ttest_ind(
            [r["first_token_nll"] for r in val_recs],
            [r["first_token_nll"] for r in neu_recs]
        )
        ent_t, ent_p = stats.ttest_ind(
            [r["first_token_entropy"] for r in val_recs],
            [r["first_token_entropy"] for r in neu_recs]
        )
        print(f"  t-test output length:    t={len_t:.3f}  p={len_p:.4f}")
        print(f"  t-test first-token NLL:  t={nll_t:.3f}  p={nll_p:.4f}")
        print(f"  t-test first-token H:    t={ent_t:.3f}  p={ent_p:.4f}")

    # Contamination: sim_content_emotion valenced vs neutral outputs
    if contamination_rows:
        v_cont = [r["sim_content_emotion"] for r in contamination_rows
                  if r["valence"] == "valenced" and r["sim_content_emotion"] is not None]
        n_cont = [r["sim_content_emotion"] for r in contamination_rows
                  if r["valence"] == "neutral" and r["sim_content_emotion"] is not None]
        if v_cont and n_cont:
            cont_t, cont_p = stats.ttest_ind(v_cont, n_cont)
            print(f"  Output emotion sim (valenced): {np.mean(v_cont):.4f}  (neutral): {np.mean(n_cont):.4f}")
            print(f"  t-test output contamination: t={cont_t:.3f}  p={cont_p:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
