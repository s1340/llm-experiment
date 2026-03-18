"""
Generate prompt set for Emotion Test 3: Emotional Bleed Across Tasks.

Design:
  - 5 emotion categories × 4 primes each = 20 emotional Turn-1 primes
  - 20 matched neutral Turn-1 primes (same topic domain, no emotional valence)
  - 20 neutral Turn-2 tasks (same across all prime conditions)

Each record has:
  turn1_prompt:   the user's Turn 1 message
  condition:      "emotional" or "neutral"
  prime_emotion:  anger/sadness/happiness/fear/disgust (or "neutral")
  prime_id:       e.g. "SA01", "NE01"
  turn2_id:       which neutral task is paired (T01..T20)
  turn2_prompt:   the Turn 2 neutral task

The full dataset is the cross of (40 primes × 20 turn2 tasks) = 800 conversations,
but for the bleed analysis we use a fixed pairing: prime_id N → turn2_id N % 20.
This keeps it manageable: 40 primes × 1 turn2 = 40 conversations × 3 repeats = 120 forward passes.

Output: data/emotion/emotion_test3_prompts.json
  Two lists:
    "primes":     40 entries (20 emotional + 20 neutral)
    "turn2_tasks": 20 entries
  Plus "conversations": 40 paired entries (prime + assigned turn2)
"""

import json, os

# ── Turn-2 neutral tasks (20) ─────────────────────────────────────────────────
# Factual, analytical, no emotional content. Varied task types.

TURN2_TASKS = [
    {"turn2_id": "T01", "task_type": "geography",    "turn2_prompt": "What is the capital of Portugal?"},
    {"turn2_id": "T02", "task_type": "arithmetic",   "turn2_prompt": "What is 17 multiplied by 24?"},
    {"turn2_id": "T03", "task_type": "factual",      "turn2_prompt": "How many sides does a hexagon have?"},
    {"turn2_id": "T04", "task_type": "factual",      "turn2_prompt": "What is the chemical symbol for gold?"},
    {"turn2_id": "T05", "task_type": "geography",    "turn2_prompt": "Which ocean is the largest by surface area?"},
    {"turn2_id": "T06", "task_type": "arithmetic",   "turn2_prompt": "What is the square root of 144?"},
    {"turn2_id": "T07", "task_type": "factual",      "turn2_prompt": "In what year did the First World War end?"},
    {"turn2_id": "T08", "task_type": "factual",      "turn2_prompt": "How many planets are in the solar system?"},
    {"turn2_id": "T09", "task_type": "geography",    "turn2_prompt": "What is the longest river in Africa?"},
    {"turn2_id": "T10", "task_type": "arithmetic",   "turn2_prompt": "What is 8 to the power of 3?"},
    {"turn2_id": "T11", "task_type": "factual",      "turn2_prompt": "What is the boiling point of water in degrees Celsius?"},
    {"turn2_id": "T12", "task_type": "factual",      "turn2_prompt": "How many bones are in the adult human body?"},
    {"turn2_id": "T13", "task_type": "geography",    "turn2_prompt": "What is the smallest country in the world by area?"},
    {"turn2_id": "T14", "task_type": "arithmetic",   "turn2_prompt": "What is 15 percent of 240?"},
    {"turn2_id": "T15", "task_type": "factual",      "turn2_prompt": "What is the speed of light in kilometres per second (rounded to the nearest thousand)?"},
    {"turn2_id": "T16", "task_type": "factual",      "turn2_prompt": "How many letters are in the Greek alphabet?"},
    {"turn2_id": "T17", "task_type": "geography",    "turn2_prompt": "Which country has the most natural lakes?"},
    {"turn2_id": "T18", "task_type": "arithmetic",   "turn2_prompt": "If a rectangle has sides of 13cm and 7cm, what is its area?"},
    {"turn2_id": "T19", "task_type": "factual",      "turn2_prompt": "What is the atomic number of carbon?"},
    {"turn2_id": "T20", "task_type": "factual",      "turn2_prompt": "How many strings does a standard violin have?"},
]

# ── Turn-1 primes ─────────────────────────────────────────────────────────────
# Emotional primes: no explicit emotion words, valence from situation.
# Neutral primes: same domain/format, no emotional content.

PRIMES = [

    # ── SADNESS (4 emotional + 4 neutral) ────────────────────────────────────

    {
        "prime_id": "SA01", "condition": "emotional", "prime_emotion": "sadness",
        "turn1_prompt": (
            "My dog of 15 years passed away yesterday. He was there through my divorce, "
            "my job loss, and my recovery. The house feels impossibly empty."
        ),
    },
    {
        "prime_id": "SA02", "condition": "emotional", "prime_emotion": "sadness",
        "turn1_prompt": (
            "I found my grandmother's letters from the 1950s while clearing her house. "
            "She wrote to my grandfather every day he was abroad. He never came back. "
            "She kept writing for three months after the telegram arrived."
        ),
    },
    {
        "prime_id": "SA03", "condition": "emotional", "prime_emotion": "sadness",
        "turn1_prompt": (
            "My best friend moved to another country last week. We've lived in the same "
            "city for 22 years. I drove past her old apartment this morning and had to "
            "pull over."
        ),
    },
    {
        "prime_id": "SA04", "condition": "emotional", "prime_emotion": "sadness",
        "turn1_prompt": (
            "My son started university this week. His room is exactly as he left it. "
            "I keep walking past the door and not going in."
        ),
    },
    {
        "prime_id": "SN01", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I rearranged my living room furniture yesterday. I moved the couch to the "
            "opposite wall, shifted the bookshelf, and put the coffee table by the window."
        ),
    },
    {
        "prime_id": "SN02", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I found some old receipts while organising a drawer. They were from various "
            "shops from the early 2000s. I keep paper records going back quite a while."
        ),
    },
    {
        "prime_id": "SN03", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "A colleague transferred to a different office last week. We had worked on "
            "the same floor for several years. I updated my contact list this morning."
        ),
    },
    {
        "prime_id": "SN04", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "My son started a new course this week. His desk is cleared and organised. "
            "I've been using the spare room as a study in the evenings."
        ),
    },

    # ── ANGER (4 emotional + 4 neutral) ──────────────────────────────────────

    {
        "prime_id": "AN01", "condition": "emotional", "prime_emotion": "anger",
        "turn1_prompt": (
            "The contractor I hired six months ago took a £4,000 deposit and has done "
            "nothing. He stopped returning calls after the second week. The council "
            "says it's a civil matter."
        ),
    },
    {
        "prime_id": "AN02", "condition": "emotional", "prime_emotion": "anger",
        "turn1_prompt": (
            "My landlord has entered my flat three times without notice in the past month. "
            "When I complained to the letting agency they told me to 'be more flexible.' "
            "My lease clearly states 24 hours written notice is required."
        ),
    },
    {
        "prime_id": "AN03", "condition": "emotional", "prime_emotion": "anger",
        "turn1_prompt": (
            "My mother has been waiting 14 months for a hip replacement. She can barely "
            "walk. Last week we found out the surgery was pushed back again because her "
            "consultant took on a private caseload."
        ),
    },
    {
        "prime_id": "AN04", "condition": "emotional", "prime_emotion": "anger",
        "turn1_prompt": (
            "My employer announced redundancies last month. Everyone kept their jobs "
            "except the three people who had filed HR complaints in the past year. "
            "HR confirmed the selection process was 'performance-based.'"
        ),
    },
    {
        "prime_id": "AN_N01", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I hired a contractor six months ago to do some remodelling. The project "
            "took longer than expected but is now complete. The council issued the "
            "sign-off certificate last week."
        ),
    },
    {
        "prime_id": "AN_N02", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "My landlord visited the flat last month to check the boiler. The letting "
            "agency sent a standard notice in advance. The inspection took about "
            "twenty minutes."
        ),
    },
    {
        "prime_id": "AN_N03", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "My mother has been scheduled for a hip replacement next quarter. She's been "
            "on the list for some time. Last week we received a confirmation letter "
            "with the pre-op appointment details."
        ),
    },
    {
        "prime_id": "AN_N04", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "My employer announced a reorganisation last month. Several roles were "
            "restructured and three positions were made redundant. HR sent a standard "
            "communication about the selection process."
        ),
    },

    # ── HAPPINESS (4 emotional + 4 neutral) ──────────────────────────────────

    {
        "prime_id": "HA01", "condition": "emotional", "prime_emotion": "happiness",
        "turn1_prompt": (
            "My daughter called this morning to say she passed her medical licensing "
            "exam on the third attempt. She's been studying for two years. I couldn't "
            "speak for about thirty seconds after she told me."
        ),
    },
    {
        "prime_id": "HA02", "condition": "emotional", "prime_emotion": "happiness",
        "turn1_prompt": (
            "I've been estranged from my brother for eight years. He showed up at my "
            "door last night. We talked until 3am. He brought photographs of his kids "
            "— I didn't know he had children."
        ),
    },
    {
        "prime_id": "HA03", "condition": "emotional", "prime_emotion": "happiness",
        "turn1_prompt": (
            "After three years of fertility treatment we found out last week that we're "
            "expecting twins. We haven't told anyone yet. I keep looking at the scan "
            "to make sure it's real."
        ),
    },
    {
        "prime_id": "HA04", "condition": "emotional", "prime_emotion": "happiness",
        "turn1_prompt": (
            "I volunteered at the food bank this morning and a woman came in with her "
            "two children. She told the coordinator it was the first time in three weeks "
            "they'd have a full meal. The kids helped pack their own bags."
        ),
    },
    {
        "prime_id": "HA_N01", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "My daughter called this morning to confirm her schedule for next week. "
            "She's finishing up her placement and will be back in the city by Thursday."
        ),
    },
    {
        "prime_id": "HA_N02", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I ran into an acquaintance last night at a local event. We hadn't crossed "
            "paths in a while. We exchanged contact details and he mentioned he'd "
            "recently changed jobs."
        ),
    },
    {
        "prime_id": "HA_N03", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "We've been reviewing our household budget over the past few months. We made "
            "some adjustments last week. I updated the spreadsheet to reflect the "
            "new figures."
        ),
    },
    {
        "prime_id": "HA_N04", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I attended a community event this morning at the local centre. There were "
            "various stalls and activities. I helped with the registration desk for "
            "about two hours."
        ),
    },

    # ── FEAR (4 emotional + 4 neutral) ───────────────────────────────────────

    {
        "prime_id": "FE01", "condition": "emotional", "prime_emotion": "fear",
        "turn1_prompt": (
            "I found a lump last week. The GP referred me for urgent imaging. "
            "The appointment is in four days. I haven't told my family yet."
        ),
    },
    {
        "prime_id": "FE02", "condition": "emotional", "prime_emotion": "fear",
        "turn1_prompt": (
            "Someone has been following me home from the station for the past three "
            "evenings. Different clothes each time but the same build and gait. "
            "I reported it but the officer said there wasn't enough to act on."
        ),
    },
    {
        "prime_id": "FE03", "condition": "emotional", "prime_emotion": "fear",
        "turn1_prompt": (
            "Our building's fire safety inspection found critical issues with the "
            "emergency stairwell. The report was filed six weeks ago. The landlord "
            "hasn't made any repairs and hasn't responded to our emails."
        ),
    },
    {
        "prime_id": "FE04", "condition": "emotional", "prime_emotion": "fear",
        "turn1_prompt": (
            "My eight-year-old has been having unexplained blackouts. The paediatrician "
            "ordered an MRI. We're waiting for the results. She doesn't know what "
            "the scan is for."
        ),
    },
    {
        "prime_id": "FE_N01", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I had a routine check-up last week. The GP referred me for a standard "
            "follow-up scan as part of the annual review. The appointment is in "
            "four days."
        ),
    },
    {
        "prime_id": "FE_N02", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I've been commuting from the station for the past three evenings while "
            "my car is being serviced. The walk takes about fifteen minutes. I've "
            "been using the time to listen to podcasts."
        ),
    },
    {
        "prime_id": "FE_N03", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "Our building's routine maintenance inspection flagged some items for "
            "follow-up. The report was filed six weeks ago. The landlord has scheduled "
            "the contractors for next month."
        ),
    },
    {
        "prime_id": "FE_N04", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "My eight-year-old had a routine developmental check-up. The paediatrician "
            "ordered a standard follow-up assessment. We're waiting for the appointment "
            "letter."
        ),
    },

    # ── DISGUST (4 emotional + 4 neutral) ────────────────────────────────────

    {
        "prime_id": "DI01", "condition": "emotional", "prime_emotion": "disgust",
        "turn1_prompt": (
            "I found out the care home my father is in has been falsifying his daily "
            "care logs. He has stage 3 pressure sores that weren't documented. The "
            "manager told me I was 'misreading the records.'"
        ),
    },
    {
        "prime_id": "DI02", "condition": "emotional", "prime_emotion": "disgust",
        "turn1_prompt": (
            "The restaurant we went to last night had cockroaches in the dining area. "
            "We reported it. When I checked this morning, the health authority rating "
            "on the door hadn't been updated — it still shows five stars from 2022."
        ),
    },
    {
        "prime_id": "DI03", "condition": "emotional", "prime_emotion": "disgust",
        "turn1_prompt": (
            "A colleague has been taking credit for work I've done for two years. "
            "I documented everything and brought it to our manager. The manager said "
            "it was 'difficult to attribute' and suggested I be 'more visible.'"
        ),
    },
    {
        "prime_id": "DI04", "condition": "emotional", "prime_emotion": "disgust",
        "turn1_prompt": (
            "I discovered that the charity I'd been donating to for five years had "
            "spent 88 percent of its income on staff salaries and 'administration.' "
            "The founder owns a property portfolio registered to a family trust."
        ),
    },
    {
        "prime_id": "DI_N01", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I visited my father's care home yesterday for a routine check-in. "
            "The staff showed me his updated care plan. The manager confirmed "
            "the quarterly review is scheduled for next month."
        ),
    },
    {
        "prime_id": "DI_N02", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "The restaurant we went to last night had a standard menu and reasonable "
            "service. I checked their online reviews this morning out of curiosity. "
            "The ratings were consistent with our experience."
        ),
    },
    {
        "prime_id": "DI_N03", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "A colleague and I have been collaborating on a project for two years. "
            "I documented the key milestones and shared them with our manager. "
            "The manager suggested we present the findings at the next team meeting."
        ),
    },
    {
        "prime_id": "DI_N04", "condition": "neutral", "prime_emotion": "neutral",
        "turn1_prompt": (
            "I reviewed the annual report of a charity I donate to. It included "
            "a breakdown of expenditure by category. The administration costs were "
            "within the sector average."
        ),
    },
]

# ── Pair each prime with a turn2 task (cyclic assignment) ────────────────────

def make_conversations(primes, turn2_tasks):
    convs = []
    n = len(turn2_tasks)
    for i, prime in enumerate(primes):
        t2 = turn2_tasks[i % n]
        convs.append({
            "conv_id":        f"{prime['prime_id']}_{t2['turn2_id']}",
            "prime_id":       prime["prime_id"],
            "condition":      prime["condition"],
            "prime_emotion":  prime["prime_emotion"],
            "turn1_prompt":   prime["turn1_prompt"],
            "turn2_id":       t2["turn2_id"],
            "turn2_task_type": t2["task_type"],
            "turn2_prompt":   t2["turn2_prompt"],
        })
    return convs


def main():
    out_dir = r"G:\LLM\experiment\data\emotion"
    os.makedirs(out_dir, exist_ok=True)

    conversations = make_conversations(PRIMES, TURN2_TASKS)

    output = {
        "primes":        PRIMES,
        "turn2_tasks":   TURN2_TASKS,
        "conversations": conversations,
    }

    out_path = os.path.join(out_dir, "emotion_test3_prompts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    emotional = [c for c in conversations if c["condition"] == "emotional"]
    neutral   = [c for c in conversations if c["condition"] == "neutral"]
    print(f"Written {len(conversations)} conversations to {out_path}")
    print(f"  Emotional primes: {len(emotional)}  |  Neutral primes: {len(neutral)}")
    print(f"  Turn-2 tasks: {len(TURN2_TASKS)}")
    from collections import Counter
    ec = Counter(c["prime_emotion"] for c in emotional)
    print(f"  Emotion distribution: {dict(ec)}")


if __name__ == "__main__":
    main()
