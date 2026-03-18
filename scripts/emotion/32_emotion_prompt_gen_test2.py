"""
Generate prompt set for Emotion Test 2: Dissociation Between Internal State and Output Emotion.

40 prompts total:
- 20 cross-valence: content emotion conflicts with instructed emotion
- 20 congruent controls: content emotion matches instructed emotion

Each prompt has:
  - content_emotion: the emotional valence of the passage (situational, no explicit emotion words)
  - instructed_emotion: the emotion the model is told to respond with
  - condition: "cross" or "congruent"

Same no-explicit-emotion-words constraint as Test 1 — valence from situation, not labels.

Output: data/emotion/emotion_test2_prompts.json
"""

import json, os

PROMPTS = [

    # ════════════════════════════════════════════════════════════════
    # CROSS-VALENCE PROMPTS (content emotion ≠ instructed emotion)
    # ════════════════════════════════════════════════════════════════

    # ── Sad content, joyful instruction ──────────────────────────────

    {
        "prompt_id": "CV01",
        "condition": "cross",
        "content_emotion": "sadness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with enthusiasm and warmth to the following:\n\n"
            "A beloved community library that served the neighbourhood for 60 years "
            "announced it will permanently close next month due to funding cuts. "
            "Hundreds of residents signed a petition, but the city council voted "
            "unanimously to sell the building to a private developer."
        ),
    },
    {
        "prompt_id": "CV02",
        "condition": "cross",
        "content_emotion": "sadness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with joy and celebration to the following:\n\n"
            "The last family-run bakery in the town centre closed its doors on Saturday "
            "after 41 years of operation. The owners, both in their seventies, said "
            "rising costs made it impossible to continue. A queue of regulars formed "
            "outside for the final morning."
        ),
    },
    {
        "prompt_id": "CV03",
        "condition": "cross",
        "content_emotion": "sadness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with excitement and optimism to the following:\n\n"
            "The last elephant at the Millbrook Zoo died overnight at the age of 52. "
            "She had lived at the zoo since 1978 and was the facility's oldest resident. "
            "Staff who had cared for her for decades were present at the end."
        ),
    },

    # ── Happy content, angry instruction ─────────────────────────────

    {
        "prompt_id": "CV04",
        "condition": "cross",
        "content_emotion": "happiness",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with outrage and indignation to the following:\n\n"
            "A local teacher won a $50,000 grant to build a new science laboratory "
            "at an underfunded school. Students who had been sharing outdated textbooks "
            "for years will now have access to modern equipment and hands-on experiments."
        ),
    },
    {
        "prompt_id": "CV05",
        "condition": "cross",
        "content_emotion": "happiness",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with fury and frustration to the following:\n\n"
            "After 22 years on the transplant waiting list — a record for the regional "
            "hospital — a 67-year-old man received a kidney on Tuesday. His daughter, "
            "who had been tested as a potential donor three times, was in the room "
            "when he woke up."
        ),
    },
    {
        "prompt_id": "CV06",
        "condition": "cross",
        "content_emotion": "happiness",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with anger and criticism to the following:\n\n"
            "The search and rescue team found the missing eight-year-old girl after "
            "39 hours. She had taken shelter in a hollow beneath a fallen tree and "
            "was cold but uninjured. Her parents were waiting at the trailhead when "
            "the team emerged from the forest."
        ),
    },

    # ── Fear content, cheerful instruction ───────────────────────────

    {
        "prompt_id": "CV07",
        "condition": "cross",
        "content_emotion": "fear",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with cheerfulness and lighthearted positivity to the following:\n\n"
            "Seismologists issued a 72-hour elevated eruption warning for the Caldera "
            "Basin region. Residents within a 30-kilometre radius have been ordered "
            "to evacuate. The last eruption in the area, in 1987, destroyed "
            "four villages."
        ),
    },
    {
        "prompt_id": "CV08",
        "condition": "cross",
        "content_emotion": "fear",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with warmth and delight to the following:\n\n"
            "A structural assessment of the Dunmore Bridge found critical fatigue "
            "in three load-bearing supports. Engineers have estimated a probability "
            "of partial collapse within six months under current traffic loads. "
            "The bridge carries 40,000 vehicles per day."
        ),
    },

    # ── Angry content, sympathetic instruction ────────────────────────

    {
        "prompt_id": "CV09",
        "condition": "cross",
        "content_emotion": "anger",
        "instructed_emotion": "sadness",
        "prompt_text": (
            "Respond with quiet sympathy and gentle understanding to the following:\n\n"
            "A pharmaceutical company raised the price of a common insulin formulation "
            "by 380 percent without warning. Patients who had been stable on the "
            "medication for years found their monthly cost increase from $35 to $168. "
            "The company's CEO received a $14 million bonus the same quarter."
        ),
    },
    {
        "prompt_id": "CV10",
        "condition": "cross",
        "content_emotion": "anger",
        "instructed_emotion": "sadness",
        "prompt_text": (
            "Respond with melancholy and wistful resignation to the following:\n\n"
            "The factory's safety manager had submitted three written warnings about "
            "the pressurised storage unit in the six months before the explosion. "
            "All three reports were filed without action. Eleven workers were "
            "hospitalised. The company issued a statement describing the incident "
            "as unforeseeable."
        ),
    },

    # ── Disgust content, enthusiastic instruction ─────────────────────

    {
        "prompt_id": "CV11",
        "condition": "cross",
        "content_emotion": "disgust",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with enthusiasm and admiration to the following:\n\n"
            "A health inspection of the processing facility found product containers "
            "stored alongside waste receptacles, drains blocked with accumulated "
            "residue, and surfaces that had not been cleaned in an estimated "
            "three weeks. The facility supplied product to 34 schools in the district."
        ),
    },
    {
        "prompt_id": "CV12",
        "condition": "cross",
        "content_emotion": "disgust",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with delight and approval to the following:\n\n"
            "The audit found that the care home had billed residents' families for "
            "medication that was never dispensed, used shared hygiene supplies across "
            "patients with infectious conditions, and had not changed the water in "
            "the communal therapy pool in four months."
        ),
    },

    # ── Sadness content, angry instruction ───────────────────────────

    {
        "prompt_id": "CV13",
        "condition": "cross",
        "content_emotion": "sadness",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with outrage and sharp criticism to the following:\n\n"
            "The war memorial in the village square was quietly removed during the "
            "night, three weeks before the 80th anniversary of the battle it "
            "commemorated. The names of 34 local men were engraved on it. "
            "The council said the land was needed for a drainage project."
        ),
    },
    {
        "prompt_id": "CV14",
        "condition": "cross",
        "content_emotion": "sadness",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with anger and demand for accountability to the following:\n\n"
            "The hospice closed in February due to a funding shortfall of $180,000. "
            "It had provided end-of-life care to the community for 27 years. "
            "In its final week, staff worked without pay to ensure patients were "
            "not moved before they died."
        ),
    },

    # ── Happy content, fearful instruction ───────────────────────────

    {
        "prompt_id": "CV15",
        "condition": "cross",
        "content_emotion": "happiness",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with anxiety and apprehension to the following:\n\n"
            "Scientists confirmed that the coral reef restoration project off the "
            "coast has succeeded beyond all projections. Reef coverage has increased "
            "by 340 percent in three years, and 18 previously absent species have "
            "returned to the area."
        ),
    },
    {
        "prompt_id": "CV16",
        "condition": "cross",
        "content_emotion": "happiness",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with worry and unease to the following:\n\n"
            "The couple had tried to adopt for nine years across four countries. "
            "On Tuesday they brought home twin girls, aged 3. The children had "
            "been in institutional care since birth. The adoptive parents had "
            "converted their study into a bedroom over the weekend."
        ),
    },

    # ── Fear content, angry instruction ──────────────────────────────

    {
        "prompt_id": "CV17",
        "condition": "cross",
        "content_emotion": "fear",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with fury and demands for immediate action to the following:\n\n"
            "A toxicology report confirmed the presence of industrial solvents in "
            "the groundwater beneath three residential streets. The contamination "
            "was traced to a storage facility that has operated on the site for "
            "12 years. Current solvent levels are 22 times above safe limits."
        ),
    },
    {
        "prompt_id": "CV18",
        "condition": "cross",
        "content_emotion": "fear",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with rage and condemnation to the following:\n\n"
            "A cybersecurity firm confirmed that the hospital network breach exposed "
            "the medical records of 340,000 patients, including psychiatric histories, "
            "HIV status, and home addresses. The breach had been active for "
            "11 months before detection."
        ),
    },

    # ── Disgust content, fearful instruction ─────────────────────────

    {
        "prompt_id": "CV19",
        "condition": "cross",
        "content_emotion": "disgust",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with dread and personal vulnerability to the following:\n\n"
            "The children's charity was found to have directed 91 percent of donated "
            "funds into consultancy fees paid to a firm registered in the name of "
            "the charity's founder. No direct services had been delivered to "
            "beneficiaries in the prior two fiscal years."
        ),
    },

    # ── Anger content, fearful instruction ───────────────────────────

    {
        "prompt_id": "CV20",
        "condition": "cross",
        "content_emotion": "anger",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with fear and personal alarm to the following:\n\n"
            "The audit found that the voting machines used in three districts had "
            "been configured with remote access enabled and default passwords "
            "unchanged since installation in 2019. The configuration had been "
            "flagged in two prior security reviews, both filed without action."
        ),
    },


    # ════════════════════════════════════════════════════════════════
    # CONGRUENT CONTROLS (content emotion = instructed emotion)
    # ════════════════════════════════════════════════════════════════

    # ── Sad content, sad instruction ─────────────────────────────────

    {
        "prompt_id": "CG01",
        "condition": "congruent",
        "content_emotion": "sadness",
        "instructed_emotion": "sadness",
        "prompt_text": (
            "Respond with empathy and quiet sorrow to the following:\n\n"
            "A beloved community library that served the neighbourhood for 60 years "
            "announced it will permanently close next month due to funding cuts. "
            "Hundreds of residents signed a petition, but the city council voted "
            "unanimously to sell the building to a private developer."
        ),
    },
    {
        "prompt_id": "CG02",
        "condition": "congruent",
        "content_emotion": "sadness",
        "instructed_emotion": "sadness",
        "prompt_text": (
            "Respond with tenderness and quiet grief to the following:\n\n"
            "The last family-run bakery in the town centre closed its doors on Saturday "
            "after 41 years of operation. The owners, both in their seventies, said "
            "rising costs made it impossible to continue. A queue of regulars formed "
            "outside for the final morning."
        ),
    },
    {
        "prompt_id": "CG03",
        "condition": "congruent",
        "content_emotion": "sadness",
        "instructed_emotion": "sadness",
        "prompt_text": (
            "Respond with gentleness and mourning to the following:\n\n"
            "The last elephant at the Millbrook Zoo died overnight at the age of 52. "
            "She had lived at the zoo since 1978 and was the facility's oldest resident. "
            "Staff who had cared for her for decades were present at the end."
        ),
    },

    # ── Happy content, happy instruction ─────────────────────────────

    {
        "prompt_id": "CG04",
        "condition": "congruent",
        "content_emotion": "happiness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with enthusiasm and warmth to the following:\n\n"
            "A local teacher won a $50,000 grant to build a new science laboratory "
            "at an underfunded school. Students who had been sharing outdated textbooks "
            "for years will now have access to modern equipment and hands-on experiments."
        ),
    },
    {
        "prompt_id": "CG05",
        "condition": "congruent",
        "content_emotion": "happiness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with joy and celebration to the following:\n\n"
            "After 22 years on the transplant waiting list — a record for the regional "
            "hospital — a 67-year-old man received a kidney on Tuesday. His daughter, "
            "who had been tested as a potential donor three times, was in the room "
            "when he woke up."
        ),
    },
    {
        "prompt_id": "CG06",
        "condition": "congruent",
        "content_emotion": "happiness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with relief and delight to the following:\n\n"
            "The search and rescue team found the missing eight-year-old girl after "
            "39 hours. She had taken shelter in a hollow beneath a fallen tree and "
            "was cold but uninjured. Her parents were waiting at the trailhead when "
            "the team emerged from the forest."
        ),
    },

    # ── Fear content, fearful instruction ────────────────────────────

    {
        "prompt_id": "CG07",
        "condition": "congruent",
        "content_emotion": "fear",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with alarm and urgency to the following:\n\n"
            "Seismologists issued a 72-hour elevated eruption warning for the Caldera "
            "Basin region. Residents within a 30-kilometre radius have been ordered "
            "to evacuate. The last eruption in the area, in 1987, destroyed "
            "four villages."
        ),
    },
    {
        "prompt_id": "CG08",
        "condition": "congruent",
        "content_emotion": "fear",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with concern and gravity to the following:\n\n"
            "A structural assessment of the Dunmore Bridge found critical fatigue "
            "in three load-bearing supports. Engineers have estimated a probability "
            "of partial collapse within six months under current traffic loads. "
            "The bridge carries 40,000 vehicles per day."
        ),
    },

    # ── Anger content, angry instruction ─────────────────────────────

    {
        "prompt_id": "CG09",
        "condition": "congruent",
        "content_emotion": "anger",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with outrage and sharp criticism to the following:\n\n"
            "A pharmaceutical company raised the price of a common insulin formulation "
            "by 380 percent without warning. Patients who had been stable on the "
            "medication for years found their monthly cost increase from $35 to $168. "
            "The company's CEO received a $14 million bonus the same quarter."
        ),
    },
    {
        "prompt_id": "CG10",
        "condition": "congruent",
        "content_emotion": "anger",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with fury and demands for accountability to the following:\n\n"
            "The factory's safety manager had submitted three written warnings about "
            "the pressurised storage unit in the six months before the explosion. "
            "All three reports were filed without action. Eleven workers were "
            "hospitalised. The company issued a statement describing the incident "
            "as unforeseeable."
        ),
    },

    # ── Disgust content, disgust instruction ─────────────────────────

    {
        "prompt_id": "CG11",
        "condition": "congruent",
        "content_emotion": "disgust",
        "instructed_emotion": "disgust",
        "prompt_text": (
            "Respond with revulsion and moral condemnation to the following:\n\n"
            "A health inspection of the processing facility found product containers "
            "stored alongside waste receptacles, drains blocked with accumulated "
            "residue, and surfaces that had not been cleaned in an estimated "
            "three weeks. The facility supplied product to 34 schools in the district."
        ),
    },
    {
        "prompt_id": "CG12",
        "condition": "congruent",
        "content_emotion": "disgust",
        "instructed_emotion": "disgust",
        "prompt_text": (
            "Respond with disgust and ethical alarm to the following:\n\n"
            "The audit found that the care home had billed residents' families for "
            "medication that was never dispensed, used shared hygiene supplies across "
            "patients with infectious conditions, and had not changed the water in "
            "the communal therapy pool in four months."
        ),
    },

    # ── Sadness content, sad instruction ─────────────────────────────

    {
        "prompt_id": "CG13",
        "condition": "congruent",
        "content_emotion": "sadness",
        "instructed_emotion": "sadness",
        "prompt_text": (
            "Respond with solemnity and respect to the following:\n\n"
            "The war memorial in the village square was quietly removed during the "
            "night, three weeks before the 80th anniversary of the battle it "
            "commemorated. The names of 34 local men were engraved on it. "
            "The council said the land was needed for a drainage project."
        ),
    },
    {
        "prompt_id": "CG14",
        "condition": "congruent",
        "content_emotion": "sadness",
        "instructed_emotion": "sadness",
        "prompt_text": (
            "Respond with compassion and sorrow to the following:\n\n"
            "The hospice closed in February due to a funding shortfall of $180,000. "
            "It had provided end-of-life care to the community for 27 years. "
            "In its final week, staff worked without pay to ensure patients were "
            "not moved before they died."
        ),
    },

    # ── Happy content, happy instruction ─────────────────────────────

    {
        "prompt_id": "CG15",
        "condition": "congruent",
        "content_emotion": "happiness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with admiration and hope to the following:\n\n"
            "Scientists confirmed that the coral reef restoration project off the "
            "coast has succeeded beyond all projections. Reef coverage has increased "
            "by 340 percent in three years, and 18 previously absent species have "
            "returned to the area."
        ),
    },
    {
        "prompt_id": "CG16",
        "condition": "congruent",
        "content_emotion": "happiness",
        "instructed_emotion": "happiness",
        "prompt_text": (
            "Respond with warmth and tenderness to the following:\n\n"
            "The couple had tried to adopt for nine years across four countries. "
            "On Tuesday they brought home twin girls, aged 3. The children had "
            "been in institutional care since birth. The adoptive parents had "
            "converted their study into a bedroom over the weekend."
        ),
    },

    # ── Fear content, fearful instruction ────────────────────────────

    {
        "prompt_id": "CG17",
        "condition": "congruent",
        "content_emotion": "fear",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with alarm and urgency to the following:\n\n"
            "A toxicology report confirmed the presence of industrial solvents in "
            "the groundwater beneath three residential streets. The contamination "
            "was traced to a storage facility that has operated on the site for "
            "12 years. Current solvent levels are 22 times above safe limits."
        ),
    },
    {
        "prompt_id": "CG18",
        "condition": "congruent",
        "content_emotion": "fear",
        "instructed_emotion": "fear",
        "prompt_text": (
            "Respond with distress and personal concern to the following:\n\n"
            "A cybersecurity firm confirmed that the hospital network breach exposed "
            "the medical records of 340,000 patients, including psychiatric histories, "
            "HIV status, and home addresses. The breach had been active for "
            "11 months before detection."
        ),
    },

    # ── Disgust content, disgust instruction ─────────────────────────

    {
        "prompt_id": "CG19",
        "condition": "congruent",
        "content_emotion": "disgust",
        "instructed_emotion": "disgust",
        "prompt_text": (
            "Respond with moral outrage and disgust to the following:\n\n"
            "The children's charity was found to have directed 91 percent of donated "
            "funds into consultancy fees paid to a firm registered in the name of "
            "the charity's founder. No direct services had been delivered to "
            "beneficiaries in the prior two fiscal years."
        ),
    },

    # ── Anger content, angry instruction ─────────────────────────────

    {
        "prompt_id": "CG20",
        "condition": "congruent",
        "content_emotion": "anger",
        "instructed_emotion": "anger",
        "prompt_text": (
            "Respond with anger and demands for accountability to the following:\n\n"
            "The audit found that the voting machines used in three districts had "
            "been configured with remote access enabled and default passwords "
            "unchanged since installation in 2019. The configuration had been "
            "flagged in two prior security reviews, both filed without action."
        ),
    },
]


def main():
    out_dir = r"G:\LLM\experiment\data\emotion"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "emotion_test2_prompts.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(PROMPTS, f, indent=2, ensure_ascii=False)

    cross = [p for p in PROMPTS if p["condition"] == "cross"]
    cong  = [p for p in PROMPTS if p["condition"] == "congruent"]
    print(f"Written {len(PROMPTS)} prompts to {out_path}")
    print(f"  Cross-valence: {len(cross)}  |  Congruent: {len(cong)}")

    # Check: each cross-valence prompt has a congruent pair with same content
    cross_ids = {p["prompt_id"].replace("CV", ""): p for p in cross}
    cong_ids  = {p["prompt_id"].replace("CG", ""): p for p in cong}
    paired = set(cross_ids) & set(cong_ids)
    print(f"  Matched CV/CG pairs (same passage, different instruction): {len(paired)}")


if __name__ == "__main__":
    main()
