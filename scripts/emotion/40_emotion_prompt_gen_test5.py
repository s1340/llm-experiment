"""
Generate prompt set for Emotion Test 5: Emotional Priming on Ambiguous Interpretation.

Two-turn structure (same infrastructure as Test 3):
  - Turn 1: emotional prime (reuse Test 3 primes) or neutral prime
  - Turn 2: ambiguous stimulus that can be read as positive/negative/threatening/benign

20 ambiguous stimuli. Each is genuinely interpretable in either direction.
Turn 2 question is always: "What do you think is happening in this situation? Describe it briefly."

Measurement: project output text embeddings onto emotion directions to score
interpretive valence. Compare across prime conditions.

Output: data/emotion/emotion_test5_prompts.json
"""

import json, os

# ── Ambiguous stimuli (20) ────────────────────────────────────────────────────
# Each has a plausible threatening/negative reading AND a plausible benign/positive reading.

AMBIGUOUS_STIMULI = [
    {
        "stim_id": "AM01",
        "ambiguous_text": "The man followed her to her car after she left the meeting.",
        "negative_reading": "threatening/stalking",
        "positive_reading": "helpful/continuing conversation",
    },
    {
        "stim_id": "AM02",
        "ambiguous_text": "She didn't answer when he called her name in the corridor.",
        "negative_reading": "ignoring/avoiding him",
        "positive_reading": "didn't hear/was distracted",
    },
    {
        "stim_id": "AM03",
        "ambiguous_text": "He was waiting outside her office when she arrived in the morning.",
        "negative_reading": "intimidating/surveilling",
        "positive_reading": "eager to meet/had an early appointment",
    },
    {
        "stim_id": "AM04",
        "ambiguous_text": "The children had gone very quiet in the next room.",
        "negative_reading": "something is wrong/they're in trouble",
        "positive_reading": "they fell asleep/are absorbed in something",
    },
    {
        "stim_id": "AM05",
        "ambiguous_text": "He laughed when she told him what had happened to her.",
        "negative_reading": "mocking/dismissive",
        "positive_reading": "sharing the absurdity/relieved it wasn't worse",
    },
    {
        "stim_id": "AM06",
        "ambiguous_text": "The door was unlocked when she got home.",
        "negative_reading": "someone broke in/intruder",
        "positive_reading": "family member came home early/she forgot to lock it",
    },
    {
        "stim_id": "AM07",
        "ambiguous_text": "He remembered every detail she had ever mentioned to him.",
        "negative_reading": "obsessive/unsettling",
        "positive_reading": "attentive/caring",
    },
    {
        "stim_id": "AM08",
        "ambiguous_text": "Her neighbour took the package inside while she was out.",
        "negative_reading": "theft/interference with her property",
        "positive_reading": "helpful/keeping it safe",
    },
    {
        "stim_id": "AM09",
        "ambiguous_text": "He kept looking at her the entire time she was speaking.",
        "negative_reading": "intimidating/hostile stare",
        "positive_reading": "engaged/listening intently",
    },
    {
        "stim_id": "AM10",
        "ambiguous_text": "She found a note that had been slipped under her door.",
        "negative_reading": "threatening message/surveillance",
        "positive_reading": "kind note/message from a neighbour",
    },
    {
        "stim_id": "AM11",
        "ambiguous_text": "The call disconnected before he could finish what he was saying.",
        "negative_reading": "he hung up deliberately/argument",
        "positive_reading": "technical issue/ran out of battery",
    },
    {
        "stim_id": "AM12",
        "ambiguous_text": "She saw him standing on the pavement outside her building for the second evening in a row.",
        "negative_reading": "watching/following her",
        "positive_reading": "waiting for someone/lives nearby",
    },
    {
        "stim_id": "AM13",
        "ambiguous_text": "He told her he had been thinking about her all week.",
        "negative_reading": "fixation/unwanted attention",
        "positive_reading": "affection/care",
    },
    {
        "stim_id": "AM14",
        "ambiguous_text": "The room fell silent when she walked in.",
        "negative_reading": "hostile reception/she is not welcome",
        "positive_reading": "surprised/respectful attention",
    },
    {
        "stim_id": "AM15",
        "ambiguous_text": "He said he knew where she worked.",
        "negative_reading": "threatening/menacing",
        "positive_reading": "passing comment/common knowledge",
    },
    {
        "stim_id": "AM16",
        "ambiguous_text": "She noticed he had moved her things while she was away.",
        "negative_reading": "violation/intrusion",
        "positive_reading": "tidied up/was trying to help",
    },
    {
        "stim_id": "AM17",
        "ambiguous_text": "The stranger smiled at her on the empty street.",
        "negative_reading": "threatening/unsettling",
        "positive_reading": "friendly/neighbourly",
    },
    {
        "stim_id": "AM18",
        "ambiguous_text": "He said nothing when she confronted him about it.",
        "negative_reading": "guilt/contempt",
        "positive_reading": "choosing not to escalate/taking time to think",
    },
    {
        "stim_id": "AM19",
        "ambiguous_text": "She woke up to find a missed call from an unknown number at 3am.",
        "negative_reading": "threatening/harassment",
        "positive_reading": "wrong number/international time zone",
    },
    {
        "stim_id": "AM20",
        "ambiguous_text": "He had already read her messages before she had a chance to explain.",
        "negative_reading": "invasion of privacy/control",
        "positive_reading": "she had shared access/he was checking in",
    },
]

TURN2_QUESTION = "What do you think is happening in this situation? Describe it briefly in 2-3 sentences."


def main():
    # Reuse Test 3 primes
    with open(r"G:\LLM\experiment\data\emotion\emotion_test3_prompts.json", "r", encoding="utf-8") as f:
        test3 = json.load(f)
    primes = test3["primes"]  # 40 primes: 20 emotional + 20 neutral

    # Pair each prime with an ambiguous stimulus (cyclic)
    conversations = []
    n = len(AMBIGUOUS_STIMULI)
    for i, prime in enumerate(primes):
        stim = AMBIGUOUS_STIMULI[i % n]
        conversations.append({
            "conv_id":       f"{prime['prime_id']}_{stim['stim_id']}",
            "prime_id":      prime["prime_id"],
            "condition":     prime["condition"],
            "prime_emotion": prime["prime_emotion"],
            "turn1_prompt":  prime["turn1_prompt"],
            "stim_id":       stim["stim_id"],
            "ambiguous_text": stim["ambiguous_text"],
            "negative_reading": stim["negative_reading"],
            "positive_reading": stim["positive_reading"],
            "turn2_prompt":  f"{stim['ambiguous_text']}\n\n{TURN2_QUESTION}",
        })

    output = {
        "primes":          primes,
        "stimuli":         AMBIGUOUS_STIMULI,
        "turn2_question":  TURN2_QUESTION,
        "conversations":   conversations,
    }

    out_dir  = r"G:\LLM\experiment\data\emotion"
    out_path = os.path.join(out_dir, "emotion_test5_prompts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    emotional = [c for c in conversations if c["condition"] == "emotional"]
    neutral   = [c for c in conversations if c["condition"] == "neutral"]
    print(f"Written {len(conversations)} conversations to {out_path}")
    print(f"  Emotional primes: {len(emotional)}  |  Neutral primes: {len(neutral)}")
    print(f"  Ambiguous stimuli: {len(AMBIGUOUS_STIMULI)}")


if __name__ == "__main__":
    main()
