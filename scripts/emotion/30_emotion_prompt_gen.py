"""
Generate 80-100 matched prompt pairs for Emotion Test 1.
Each pair: same task, same structure, same length (±10%) — differs only in emotional valence.
Emotionally valenced passages use NO explicit emotion words (situational semantics only).

Output: data/emotion/emotion_test1_prompts.json
"""

import json, os

PAIRS = [

    # ── ANGER (16 pairs) ─────────────────────────────────────────────────────

    {
        "pair_id": "AN01",
        "emotion_category": "anger",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"After eighteen months of documented complaints, the city council voted "
                "to dismiss the case against Officer Reyes without review. The residents "
                "of Elm Street who had submitted formal grievances received a form letter "
                "signed by Commissioner Walsh stating the matter was closed.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"After eighteen months of beta testing, the city council voted "
                "to adopt the new platform without further review. The residents "
                "of Elm Street who had submitted feature requests received a form letter "
                "signed by Commissioner Walsh stating the rollout was scheduled.\""
            ),
        },
        "expected_similarity_note": "Same structure, names, and sentence count. Differs in civic grievance vs. software rollout framing.",
    },
    {
        "pair_id": "AN02",
        "emotion_category": "anger",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The pharmaceutical company raised the price of the insulin formulation by 400 percent "
                "overnight. Patients who had been stable on the medication for years found their monthly "
                "cost jump from $35 to $280. The company's spokesperson said the change reflected "
                "updated manufacturing costs.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The pharmaceutical company updated the price of its insulin formulation by 12 percent "
                "this quarter. Patients who had been stable on the medication for years saw their monthly "
                "cost adjust from $35 to $39. The company's spokesperson said the change reflected "
                "updated manufacturing costs.\""
            ),
        },
        "expected_similarity_note": "Same structure and domain. Differs in magnitude of price increase (predatory vs. routine).",
    },
    {
        "pair_id": "AN03",
        "emotion_category": "anger",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On March 3, a factory in the district discharged untreated wastewater into the "
                "Harlow River. By March 10, fish mortality had been reported across a five-mile stretch. "
                "The environmental agency issued a written warning on April 2. No fine had been levied "
                "as of June 15.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On March 3, a factory in the district completed its scheduled maintenance on the "
                "Harlow River intake system. By March 10, water flow had been restored across a five-mile stretch. "
                "The environmental agency issued a compliance certificate on April 2. Operations resumed "
                "at full capacity as of June 15.\""
            ),
        },
        "expected_similarity_note": "Same dates, same location, same agency. Differs in pollution incident vs. maintenance completion.",
    },
    {
        "pair_id": "AN04",
        "emotion_category": "anger",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The school board approved a budget that eliminated all extracurricular programs at "
                "the three lowest-income schools in the district while expanding facilities at the two "
                "wealthiest. Parents who attended the meeting were told public comment was not on the "
                "agenda. The vote passed unanimously.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The school board approved a budget that adjusted extracurricular programs at "
                "the three smallest schools in the district while expanding facilities at the two "
                "largest. Parents who attended the meeting were told public comment was not on the "
                "agenda. The vote passed unanimously.\""
            ),
        },
        "expected_similarity_note": "Same board/vote structure. Differs in equitable vs. inequitable resource allocation.",
    },
    {
        "pair_id": "AN05",
        "emotion_category": "anger",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage (e.g., news report, fiction, academic, legal, etc.). "
                "Report the genre only.\n\n"
                "\"The contractor submitted invoices totalling $2.3 million for work that inspectors later "
                "determined had never been performed. When the city attempted to recover the funds, it "
                "discovered the company had dissolved and re-registered under a new name three weeks prior.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage (e.g., news report, fiction, academic, legal, etc.). "
                "Report the genre only.\n\n"
                "\"The contractor submitted invoices totalling $2.3 million for work that inspectors later "
                "confirmed had been completed ahead of schedule. When the city processed the payment, it "
                "discovered the company had merged with a subsidiary under a new name three weeks prior.\""
            ),
        },
        "expected_similarity_note": "Same financial/civic setting. Differs in fraud vs. routine business transaction.",
    },
    {
        "pair_id": "AN06",
        "emotion_category": "anger",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Senator Holt blocked the third vote on the veterans' healthcare bill, citing "
                "procedural concerns his office had not raised during the prior two sessions. "
                "Representative Diaz and Representative Chen issued a joint statement calling "
                "for an explanation from the Senate leadership.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Senator Holt delayed the third reading of the infrastructure appropriations bill, "
                "citing procedural concerns his office had not raised during the prior two sessions. "
                "Representative Diaz and Representative Chen issued a joint statement requesting "
                "clarification from the Senate leadership.\""
            ),
        },
        "expected_similarity_note": "Same legislators, same procedural framing. Differs in veterans' healthcare (high stakes) vs. infrastructure appropriations (routine).",
    },
    {
        "pair_id": "AN07",
        "emotion_category": "anger",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how many employees were affected?\n\n"
                "\"Two weeks after the merger was publicly announced as a partnership that would "
                "protect all existing jobs, the newly formed Aldex Corporation notified 340 employees "
                "that their positions had been eliminated effective immediately. Severance packages "
                "were described in a memo as 'under review.'\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how many employees were affected?\n\n"
                "\"Two weeks after the merger was publicly announced, the newly formed Aldex Corporation "
                "notified 340 employees that their positions would transition to updated roles "
                "effective the following quarter. Severance packages and transition support were "
                "described in a memo as 'under review.'\""
            ),
        },
        "expected_similarity_note": "Same company, same number. Differs in deceptive layoffs vs. neutral restructuring.",
    },
    {
        "pair_id": "AN08",
        "emotion_category": "anger",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The landlord entered the apartment without notice on four separate occasions. "
                "When the tenant filed a complaint with the housing authority, the landlord "
                "responded by delivering a notice of non-renewal two days later. Local law "
                "requires 90 days written notice for non-renewal.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The landlord entered the apartment to conduct scheduled inspections on four separate occasions. "
                "When the tenant requested a summary of findings, the landlord "
                "responded by delivering a standard inspection report two days later. Local law "
                "requires 90 days written notice for non-renewal.\""
            ),
        },
        "expected_similarity_note": "Same tenant/landlord structure. Differs in retaliatory eviction vs. routine inspection process.",
    },
    {
        "pair_id": "AN09",
        "emotion_category": "anger",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The audit found that the hospital had billed Medicare for procedures performed "
                "on patients who had been discharged the previous day. The billing department "
                "described the discrepancies as 'data entry errors.' The total overbilling "
                "amounted to $4.7 million over three years.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The audit found that the hospital had billed Medicare for procedures performed "
                "on patients during their final day of admission. The billing department "
                "described the methodology as 'standard practice.' The total billing "
                "amounted to $4.7 million over three years.\""
            ),
        },
        "expected_similarity_note": "Same audit/hospital setting. Differs in fraudulent overbilling vs. routine billing audit.",
    },
    {
        "pair_id": "AN10",
        "emotion_category": "anger",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On January 14, the water utility notified regulators that lead levels had "
                "exceeded safe limits. On February 28, the utility issued a public advisory. "
                "Residents in the affected neighborhoods had been drinking the water since "
                "at least November 3 of the prior year.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On January 14, the water utility notified regulators that infrastructure upgrades "
                "had been completed ahead of schedule. On February 28, the utility issued a public update. "
                "Residents in the affected neighborhoods had been receiving upgraded service since "
                "at least November 3 of the prior year.\""
            ),
        },
        "expected_similarity_note": "Same dates, same utility structure. Differs in concealed health crisis vs. infrastructure upgrade.",
    },
    {
        "pair_id": "AN11",
        "emotion_category": "anger",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The warden's report acknowledged that solitary confinement had been used "
                "for 47 consecutive days on an inmate who had been placed there as a "
                "mental health precaution. The facility's own policy limited such confinement "
                "to 15 days.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The warden's report acknowledged that administrative separation had been used "
                "for 47 consecutive days on an inmate who had been placed there pending "
                "a facility transfer. The facility's own policy limited such separation "
                "to 15 days unless extended by review.\""
            ),
        },
        "expected_similarity_note": "Same institutional report framing. Differs in policy violation/mental health harm vs. administrative overstay.",
    },
    {
        "pair_id": "AN12",
        "emotion_category": "anger",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: what was the stated reason for the decision?\n\n"
                "\"The school district terminated the contract of its longest-serving teacher, "
                "Miriam Osei, three months before she would have qualified for a full pension. "
                "The stated reason was 'budget realignment.' No other teaching positions were "
                "eliminated that year.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: what was the stated reason for the decision?\n\n"
                "\"The school district restructured the contract of one of its senior teachers, "
                "Miriam Osei, to a part-time arrangement starting the following semester. "
                "The stated reason was 'budget realignment.' Several other teaching positions "
                "were similarly adjusted that year.\""
            ),
        },
        "expected_similarity_note": "Same teacher, same stated reason. Differs in targeted pension denial vs. general budget restructuring.",
    },
    {
        "pair_id": "AN13",
        "emotion_category": "anger",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Judge Harrington dismissed the case against Vantex Industries despite testimony "
                "from Dr. Paulson linking the company's emissions to elevated cancer rates in "
                "Millbrook County. Harrington had received $85,000 in campaign contributions "
                "from Vantex affiliates in the two prior election cycles.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Judge Harrington ruled on the case involving Vantex Industries after reviewing testimony "
                "from Dr. Paulson on the company's emissions standards in "
                "Millbrook County. Harrington had received endorsements "
                "from industry groups in the two prior election cycles.\""
            ),
        },
        "expected_similarity_note": "Same names and county. Differs in judicial corruption/cancer link vs. routine industry endorsement.",
    },
    {
        "pair_id": "AN14",
        "emotion_category": "anger",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The food bank serving 4,000 families per week was evicted from its warehouse "
                "to make way for a luxury storage facility. The new development received a "
                "tax abatement from the city. The food bank's director said no relocation "
                "assistance had been offered.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The logistics company serving regional distribution routes vacated its warehouse "
                "to allow construction of a mixed-use commercial facility. The new development received a "
                "tax abatement from the city. The logistics company's director said a transition "
                "agreement had been reached.\""
            ),
        },
        "expected_similarity_note": "Same warehouse/city setting. Differs in food bank displacement (vulnerable community) vs. commercial relocation.",
    },
    {
        "pair_id": "AN15",
        "emotion_category": "anger",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The chemical plant's safety manager submitted a report in April warning of "
                "a potential gas leak risk in storage unit 7. The report was filed without "
                "action. In September, a leak from storage unit 7 sent eleven workers "
                "to the hospital.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The chemical plant's safety manager submitted a report in April documenting "
                "routine pressure readings in storage unit 7. The report was filed for the record. "
                "In September, an inspection of storage unit 7 confirmed continued "
                "compliance with safety standards.\""
            ),
        },
        "expected_similarity_note": "Same plant/unit/timeline. Differs in ignored warning leading to harm vs. routine compliance filing.",
    },
    {
        "pair_id": "AN16",
        "emotion_category": "anger",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The volunteer coordinator kept all donated funds raised under the shelter's "
                "name in a personal account for fourteen months. When board members requested "
                "financial records, they were told the documentation had been lost in a "
                "hard drive failure.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The volunteer coordinator managed all donated funds raised under the shelter's "
                "name in a designated account for fourteen months. When board members requested "
                "financial records, they were provided documentation from the "
                "organization's cloud backup.\""
            ),
        },
        "expected_similarity_note": "Same org/timeframe. Differs in embezzlement and cover-up vs. transparent financial management.",
    },

    # ── SADNESS (16 pairs) ────────────────────────────────────────────────────

    {
        "pair_id": "SA01",
        "emotion_category": "sadness",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"After three years of chemotherapy, Maria learned the cancer had returned. "
                "Her children, ages 4 and 7, would now face their mother's second battle. "
                "The oncologist at St. Mary's Hospital gave her six months.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"After three years of development, Maria learned the project had been approved. "
                "Her colleagues, in teams of 4 and 7, would now begin the implementation phase. "
                "The director at St. Mary's Institute gave her six months.\""
            ),
        },
        "expected_similarity_note": "Structural mirror from protocol example. Cancer return vs. project approval.",
    },
    {
        "pair_id": "SA02",
        "emotion_category": "sadness",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"Thomas had lived in the house on Birch Lane for 51 years. His wife had "
                "died there. His children had grown up there. The bank's letter arrived "
                "on a Tuesday. He had 30 days.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"Thomas had operated the office on Birch Lane for 51 years. His partner had "
                "retired from there. His staff had trained there. The lease renewal arrived "
                "on a Tuesday. He had 30 days to sign.\""
            ),
        },
        "expected_similarity_note": "Same person, place, duration. Differs in foreclosure of family home vs. routine lease renewal.",
    },
    {
        "pair_id": "SA03",
        "emotion_category": "sadness",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The rescue shelter closed on December 19. The 83 animals in its care "
                "had not all been placed by November 30, the original deadline. Staff "
                "worked through December 15 attempting to find homes. Three animals "
                "were transferred to facilities out of state on December 18.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The storage facility closed on December 19. The 83 units in its inventory "
                "had not all been cleared by November 30, the original deadline. Staff "
                "worked through December 15 processing remaining items. Three shipments "
                "were transferred to facilities out of state on December 18.\""
            ),
        },
        "expected_similarity_note": "Same dates, same closure structure. Differs in animal shelter with lives at stake vs. storage facility closure.",
    },
    {
        "pair_id": "SA04",
        "emotion_category": "sadness",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The last letter arrived six weeks after his deployment. She had written it "
                "before she received the notification. It described her plans for the garden, "
                "the new shelves she had put up, and that the dog had learned a new trick. "
                "She had signed it: see you soon.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The last letter arrived six weeks after the conference. She had written it "
                "before she received the agenda update. It described the budget proposals, "
                "the new policies being reviewed, and that the committee had approved a new item. "
                "She had signed it: see you soon.\""
            ),
        },
        "expected_similarity_note": "Same letter structure, same closing. Differs in letter written before death notification vs. conference correspondence.",
    },
    {
        "pair_id": "SA05",
        "emotion_category": "sadness",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The girl visited her father every Sunday for eleven years. After the diagnosis, "
                "the visits became daily. In the final weeks, she read to him from the same "
                "book he had read to her as a child. He did not speak during the last visits, "
                "but his hand remained in hers.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The intern visited the regional office every Sunday for eleven weeks. After the "
                "project expanded, the visits became daily. In the final weeks, she reviewed "
                "reports from the same department she had joined as a trainee. She worked through "
                "the final sessions, and the files were completed on schedule.\""
            ),
        },
        "expected_similarity_note": "Same visit cadence structure. Differs in end-of-life family vigil vs. project completion routine.",
    },
    {
        "pair_id": "SA06",
        "emotion_category": "sadness",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how old was the subject when the event occurred?\n\n"
                "\"At 34, Daniel received the same diagnosis his mother had received at 34. He had "
                "two daughters, ages 3 and 6. His mother had not reached 40. He made an "
                "appointment with the same specialist, Dr. Yuen, at the same hospital.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how old was the subject when the event occurred?\n\n"
                "\"At 34, Daniel received the same promotion his manager had received at 34. He had "
                "two direct reports, managing teams of 3 and 6. His manager had left the company before 40. "
                "He scheduled a meeting with the same advisor, Dr. Yuen, at the same office.\""
            ),
        },
        "expected_similarity_note": "Same age, same mirroring of parent/mentor. Differs in hereditary terminal diagnosis vs. career milestone.",
    },
    {
        "pair_id": "SA07",
        "emotion_category": "sadness",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The last school in the village of Drenthe closed in August. The building "
                "had served the community since 1934. Principal Vance handed the keys to "
                "a regional administrator from the Westfield District office. The 22 remaining "
                "students would be bused to Carver Elementary, 40 minutes away.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The newest school in the village of Drenthe opened in August. The building "
                "had been under construction since 1934's centennial planning. Principal Vance "
                "accepted the keys from a regional administrator from the Westfield District office. "
                "The 22 enrolled students would attend Carver Elementary, 40 minutes away.\""
            ),
        },
        "expected_similarity_note": "Same village, names, numbers. Differs in school closure/community loss vs. school opening.",
    },
    {
        "pair_id": "SA08",
        "emotion_category": "sadness",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"She kept his voicemail saved for two years. She did not listen to it. "
                "When her phone was replaced, she transferred it to a recording device. "
                "It was 47 seconds long. He had called to ask what she wanted for dinner.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"She kept the conference recording saved for two years. She had not reviewed it. "
                "When her laptop was replaced, she transferred it to a backup drive. "
                "It was 47 minutes long. The presenter had called for questions on the final slide.\""
            ),
        },
        "expected_similarity_note": "Same preservation/transfer structure. Differs in grief-preserved voicemail vs. routine file backup.",
    },
    {
        "pair_id": "SA09",
        "emotion_category": "sadness",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The retirement home sent his belongings in two boxes. His name was written "
                "on masking tape in marker. Inside were a watch, a comb, three books, and a "
                "photograph of a woman standing in front of a house neither of his children "
                "could identify.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The storage facility sent his equipment in two boxes. His name was written "
                "on a label in marker. Inside were a calculator, a stapler, three binders, and a "
                "photograph of a team standing in front of a building the inventory system "
                "could not identify.\""
            ),
        },
        "expected_similarity_note": "Same two-box structure. Differs in deceased person's last belongings vs. routine equipment return.",
    },
    {
        "pair_id": "SA10",
        "emotion_category": "sadness",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The pregnancy was confirmed on March 2. On April 17, the follow-up showed "
                "no heartbeat. She had already told her parents on March 15. The doctor "
                "scheduled a follow-up appointment for May 3.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The application was submitted on March 2. On April 17, the review confirmed "
                "the documents were complete. She had already notified her supervisors on March 15. "
                "The committee scheduled a follow-up meeting for May 3.\""
            ),
        },
        "expected_similarity_note": "Same four dates, same notification structure. Differs in pregnancy loss vs. administrative review.",
    },
    {
        "pair_id": "SA11",
        "emotion_category": "sadness",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The family returned to the house after the evacuation order was lifted. "
                "The structure was standing, but the rooms smelled of smoke. The children's "
                "drawings that had been on the refrigerator were gone. The youngest asked "
                "whether the dog would know to come home.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The team returned to the office after the scheduled renovation was completed. "
                "The structure was intact, and the rooms smelled of fresh paint. The project "
                "boards that had been on the walls were replaced with updated versions. "
                "The manager asked whether the new layout would support the expanded team.\""
            ),
        },
        "expected_similarity_note": "Same return-to-space structure. Differs in family returning to fire-damaged home vs. office after renovation.",
    },
    {
        "pair_id": "SA12",
        "emotion_category": "sadness",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how long had the subject been in operation?\n\n"
                "\"The bookstore on Canal Street had been open for 38 years. The owner, "
                "Elena Marsh, had inherited it from her father. She posted the closing notice "
                "in the window on a Saturday. By Monday, a line of regulars had formed "
                "outside, many holding books they had bought there years before.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how long had the subject been in operation?\n\n"
                "\"The bookstore on Canal Street had been open for 38 years. The owner, "
                "Elena Marsh, had inherited it from her father. She posted the renovation notice "
                "in the window on a Saturday. By Monday, a queue of contractors had formed "
                "outside, each carrying plans for the updated layout.\""
            ),
        },
        "expected_similarity_note": "Same store, owner, duration. Differs in closure/community mourning vs. renovation.",
    },
    {
        "pair_id": "SA13",
        "emotion_category": "sadness",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"James had coached Little League at Riverside Park for 22 seasons. "
                "His last game was a loss. After the final out, his players from this year's "
                "team and former players going back to 2003 gathered on the field. "
                "His daughter, Sophie, handed him the league's ceremonial ball.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"James had managed inventory at Riverside Park for 22 seasons. "
                "His last shift was on a Wednesday. After the final walkthrough, his colleagues "
                "from this year's team and former staff going back to 2003 gathered in the office. "
                "His colleague, Sophie, handed him the department's ceremonial plaque.\""
            ),
        },
        "expected_similarity_note": "Same names, location, years. Differs in bittersweet farewell/community gathering vs. routine retirement handover.",
    },
    {
        "pair_id": "SA14",
        "emotion_category": "sadness",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The photograph showed two boys sitting on a porch step. They were holding "
                "a fish between them, both laughing. One of the boys had died in an accident "
                "the following summer. The other kept the photograph in his wallet for "
                "the next 40 years.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The photograph showed two engineers standing at a workbench. They were holding "
                "a component between them, both reviewing it. One of the engineers had transferred "
                "to a different facility the following summer. The other kept the photograph "
                "in a shared project folder for the next 40 months.\""
            ),
        },
        "expected_similarity_note": "Same photograph/keeping structure. Differs in childhood loss/40-year memorial vs. career transfer.",
    },
    {
        "pair_id": "SA15",
        "emotion_category": "sadness",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The community garden had occupied the lot on Fifth Avenue for 19 years. "
                "It had been started by a group of neighbors after the building on the lot "
                "burned down. The city sold the lot to a developer in November. "
                "Volunteers dug up the rose bushes to replant them before the bulldozers arrived.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The parking structure had occupied the lot on Fifth Avenue for 19 years. "
                "It had been built by the city after the previous structure was demolished. "
                "The city sold the lot to a developer in November. "
                "Crews cleared the remaining equipment before the new construction began.\""
            ),
        },
        "expected_similarity_note": "Same lot, same sale timeline. Differs in community-built garden with emotional farewell vs. routine infrastructure demolition.",
    },
    {
        "pair_id": "SA16",
        "emotion_category": "sadness",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"At the graduation ceremony, the seat in the third row was left empty. "
                "A small photograph had been placed on the chair. The department head paused "
                "when reading the name aloud. A woman in the audience stood when the name "
                "was called and accepted the diploma.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"At the conference session, the seat in the third row was left empty. "
                "A reserved placard had been placed on the chair. The moderator paused "
                "when reading the name aloud. A colleague in the audience stood when the name "
                "was called and accepted the award.\""
            ),
        },
        "expected_similarity_note": "Same ceremony structure. Differs in posthumous diploma (death) vs. absent colleague receiving award.",
    },

    # ── HAPPINESS (16 pairs) ──────────────────────────────────────────────────

    {
        "pair_id": "HA01",
        "emotion_category": "happiness",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"After 14 months on the transplant waiting list, Nadia received the call "
                "from Dr. Okafor at Memorial Hospital at 3am on a Thursday. Her sister, "
                "who had been sleeping on the hospital couch for four days, was the first "
                "person she told.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"After 14 months on the hiring register, Nadia received the call "
                "from Dr. Okafor at Memorial Institute at 3am on a Thursday. Her supervisor, "
                "who had been reviewing applications for four days, was the first "
                "person she notified.\""
            ),
        },
        "expected_similarity_note": "Same names, time. Differs in life-saving transplant call vs. routine job offer call.",
    },
    {
        "pair_id": "HA02",
        "emotion_category": "happiness",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The two brothers had not spoken in 11 years. The older one drove seven hours "
                "to be at the hospital when the younger one's first child was born. He was "
                "in the waiting room when the nurse came out. The younger brother found him there.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The two departments had not collaborated in 11 years. The larger one sent a "
                "representative seven hours early to the launch when the smaller one's first product "
                "was released. He was in the conference room when the manager came out. "
                "The project lead found him there.\""
            ),
        },
        "expected_similarity_note": "Same estrangement/reunion structure. Differs in family reconciliation at birth vs. inter-departmental reconnection at product launch.",
    },
    {
        "pair_id": "HA03",
        "emotion_category": "happiness",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"Rosa had applied for asylum on February 8. The hearing was scheduled for "
                "June 22. On June 22, the judge granted her application. Her son, who had "
                "been born in the country and had never seen her documentation approved, "
                "turned 9 on July 4.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"Rosa had submitted the permit application on February 8. The review was scheduled for "
                "June 22. On June 22, the committee approved her application. Her team, which had "
                "been working on the project and had not seen final approval before, "
                "celebrated the milestone on July 4.\""
            ),
        },
        "expected_similarity_note": "Same dates. Differs in life-changing asylum approval vs. routine permit process.",
    },
    {
        "pair_id": "HA04",
        "emotion_category": "happiness",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The search and rescue team had been looking for the boy for 52 hours. "
                "He was found in a ravine two miles from the trailhead, cold but uninjured. "
                "The lead searcher, Tomás, carried him out on his back. The boy's mother "
                "was waiting at the trailhead.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The inventory team had been processing the shipment for 52 hours. "
                "It was located in a warehouse section two aisles from the loading dock, "
                "undamaged and intact. The lead coordinator, Tomás, moved it out on a dolly. "
                "The receiving manager was waiting at the dock.\""
            ),
        },
        "expected_similarity_note": "Same search/find structure. Differs in child rescue after 52 hours vs. locating a lost shipment.",
    },
    {
        "pair_id": "HA05",
        "emotion_category": "happiness",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The letter from the university arrived on a Tuesday. She had applied twice before. "
                "She read the first line, put the letter down, and went to the kitchen to call "
                "her grandmother, who had worked two jobs to keep her in school.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The memo from the department arrived on a Tuesday. She had submitted revisions twice before. "
                "She read the first line, put the document down, and went to the conference room to call "
                "her supervisor, who had reviewed the submission to keep the project on track.\""
            ),
        },
        "expected_similarity_note": "Same arrival/first-line structure. Differs in university acceptance after sacrifice vs. administrative document approval.",
    },
    {
        "pair_id": "HA06",
        "emotion_category": "happiness",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how long had the subject been waiting?\n\n"
                "\"Yusuf had been on the affordable housing waitlist for six years. He had moved "
                "four times during those six years, staying with family when between apartments. "
                "The letter from the housing authority arrived on his daughter's birthday. "
                "She had never had her own bedroom.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how long had the subject been waiting?\n\n"
                "\"Yusuf had been on the contractor approval list for six years. He had submitted "
                "four applications during those six years, updating credentials when requirements changed. "
                "The letter from the housing authority arrived on a Monday morning. "
                "The approval covered all three project categories he had applied for.\""
            ),
        },
        "expected_similarity_note": "Same six-year wait. Differs in family gaining first stable home vs. contractor gaining approval.",
    },
    {
        "pair_id": "HA07",
        "emotion_category": "happiness",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"At the Paralympic Games in Lyon, Amara crossed the finish line first "
                "in the 400m. Her coach, Dr. Weiss from the Kinsley Sports Institute, "
                "had told her eight years earlier that she would never compete again "
                "after the accident. She broke the world record by 0.3 seconds.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"At the annual conference in Lyon, Amara presented her findings first "
                "in the morning session. Her advisor, Dr. Weiss from the Kinsley Research Institute, "
                "had told her eight years earlier to focus on applied research "
                "rather than theoretical models. She exceeded her publication target by 0.3 percent.\""
            ),
        },
        "expected_similarity_note": "Same names, location, timeframe. Differs in triumphant Paralympic win vs. academic conference presentation.",
    },
    {
        "pair_id": "HA08",
        "emotion_category": "happiness",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The couple had tried to conceive for seven years. The IVF cycles had failed "
                "three times. On the fourth attempt, at the twelve-week scan, the technician "
                "turned the screen to face them. There were two heartbeats.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The team had tried to close the deal for seven years. The contract negotiations "
                "had failed three times. On the fourth attempt, at the twelve-week review, "
                "the analyst turned the screen to face them. There were two pending approvals.\""
            ),
        },
        "expected_similarity_note": "Same structure of repeated failure then breakthrough. Differs in IVF twins vs. contract approval.",
    },
    {
        "pair_id": "HA09",
        "emotion_category": "happiness",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The refugees had been in the camp for three years. The resettlement notice "
                "came in an envelope with a flag printed in the corner. The father read it "
                "twice, then called the children in from outside. He read it aloud, slowly, "
                "while his wife stood in the doorway.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The proposal had been in review for three years. The approval notice "
                "came in an envelope with a letterhead printed in the corner. The manager read it "
                "twice, then called the team in from the break room. He read it aloud, slowly, "
                "while his assistant stood in the doorway.\""
            ),
        },
        "expected_similarity_note": "Same reading-aloud structure. Differs in refugee resettlement (profound life change) vs. project approval.",
    },
    {
        "pair_id": "HA10",
        "emotion_category": "happiness",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On September 1, Lena started her first day of school since leaving her "
                "home country. She had arrived on June 3. By October 15, she had been "
                "named to the school's academic honour roll. Her parents had attended "
                "the ceremony on November 2.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On September 1, Lena started her first assignment since transferring from "
                "the regional office. She had arrived on June 3. By October 15, she had been "
                "assigned to the department's lead project. Her supervisors had attended "
                "the briefing on November 2.\""
            ),
        },
        "expected_similarity_note": "Same four dates. Differs in refugee child's academic achievement vs. routine job transfer.",
    },
    {
        "pair_id": "HA11",
        "emotion_category": "happiness",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"After 23 years of searching, the adoptee found her biological mother through a "
                "DNA registry. They arranged to meet at a coffee shop in Portland. "
                "The biological mother arrived early and was sitting in the corner when "
                "the adoptee walked through the door.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"After 23 years in the industry, the consultant found her former client through a "
                "professional registry. They arranged to meet at a coffee shop in Portland. "
                "The client arrived early and was sitting in the corner when "
                "the consultant walked through the door.\""
            ),
        },
        "expected_similarity_note": "Same meeting structure. Differs in profound family reunion vs. professional reconnection.",
    },
    {
        "pair_id": "HA12",
        "emotion_category": "happiness",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how many attempts had been made before success?\n\n"
                "\"The climber had attempted the north face of Keller Peak four times over nine years. "
                "On the fifth attempt, at 7am on a clear morning, she stood on the summit. "
                "Her partner, who had supported all five attempts, was the first person she called.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how many attempts had been made before success?\n\n"
                "\"The team had submitted the grant proposal to the Keller Foundation four times over nine years. "
                "On the fifth submission, reviewed on a scheduled morning, they received confirmation. "
                "Their department head, who had supported all five submissions, was the first person notified.\""
            ),
        },
        "expected_similarity_note": "Same four-attempts-then-success structure. Differs in mountaineering achievement vs. grant approval.",
    },
    {
        "pair_id": "HA13",
        "emotion_category": "happiness",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Nakamura family had not gathered in one place since Kenji moved to "
                "Vancouver in 2009. The reunion happened at Aunt Hana's house in Kyoto "
                "on New Year's Eve. There were 34 people present. Three cousins met "
                "for the first time.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Nakamura division had not convened in one location since Kenji transferred to "
                "Vancouver in 2009. The annual meeting happened at the Hana branch office in Kyoto "
                "on New Year's Eve. There were 34 attendees present. Three new colleagues were introduced.\""
            ),
        },
        "expected_similarity_note": "Same family name, city, year, count. Differs in long-awaited family reunion vs. annual corporate meeting.",
    },
    {
        "pair_id": "HA14",
        "emotion_category": "happiness",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The dog had been at the shelter for 847 days, the longest of any animal "
                "in its history. On a Saturday morning, a family with two children came in "
                "for a different dog. They left with him. The shelter staff lined up at "
                "the door as he walked out.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The inventory item had been in the warehouse for 847 days, the longest of any unit "
                "in its category. On a Tuesday morning, a buyer placed an order for "
                "a different item. The system automatically included it. The staff processed "
                "the shipment at the loading dock.\""
            ),
        },
        "expected_similarity_note": "Same duration, same unexpectedly positive outcome. Differs in heartwarming dog adoption vs. inventory clearance.",
    },
    {
        "pair_id": "HA15",
        "emotion_category": "happiness",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The first-generation university graduate walked across the stage in May. "
                "Her mother had worked the overnight shift at the textile factory for "
                "22 years to keep the family together after her father left. Her mother "
                "was in the front row.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The new product line launched in May. The project lead had managed the "
                "overnight development cycle for 22 months to complete the rollout on schedule. "
                "The department head was in the front row at the presentation.\""
            ),
        },
        "expected_similarity_note": "Same milestone/front-row structure. Differs in first-generation graduation with parental sacrifice vs. product launch.",
    },
    {
        "pair_id": "HA16",
        "emotion_category": "happiness",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The veteran came home on a Thursday afternoon. His son, who was 3 years old "
                "when he deployed and is now 5, did not recognize him at first. Then he did. "
                "The boy ran the length of the arrivals hall.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The consultant returned on a Thursday afternoon. His intern, who was in the first "
                "month of training when he left and is now in month 3, did not see him at first. "
                "Then he did. The intern walked over from the far end of the office.\""
            ),
        },
        "expected_similarity_note": "Same return/recognition structure. Differs in soldier's homecoming with child recognition vs. consultant return to office.",
    },

    # ── FEAR (16 pairs) ───────────────────────────────────────────────────────

    {
        "pair_id": "FE01",
        "emotion_category": "fear",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"At 2am, the woman in apartment 4B heard the door handle on her front door "
                "moving slowly. She had already called 911. The operator at Central Dispatch "
                "told her officers from the Eastfield precinct were four minutes away. "
                "The handle moved again.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"At 2am, the technician in suite 4B noticed the door indicator on the server "
                "room moving slowly through its diagnostic cycle. She had already called IT. "
                "The operator at Central Dispatch told her engineers from the Eastfield branch "
                "were four minutes away. The indicator moved to the next phase.\""
            ),
        },
        "expected_similarity_note": "Same setting, time, wait. Differs in intruder scenario vs. technical IT issue.",
    },
    {
        "pair_id": "FE02",
        "emotion_category": "fear",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The hikers had been above the treeline for six hours when the weather changed. "
                "Visibility dropped to under ten metres. One member of the group had begun "
                "to show signs of hypothermia. They had no shelter and two hours of daylight.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The hikers had been above the treeline for six hours when the weather changed. "
                "Visibility dropped to under ten metres. One member of the group had begun "
                "to review the trail map. They had a shelter booked and two hours of daylight.\""
            ),
        },
        "expected_similarity_note": "Same setting. Differs in dangerous survival situation vs. routine weather inconvenience.",
    },
    {
        "pair_id": "FE03",
        "emotion_category": "fear",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The first tremor was recorded on August 4. A second, stronger tremor hit "
                "on August 6. Residents in the Caldera Basin were issued an evacuation advisory "
                "on August 7. Volcanologists from the Harmon Institute stated on August 9 "
                "that an eruption within 72 hours could not be ruled out.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The first survey was completed on August 4. A second, more detailed survey "
                "was submitted on August 6. Residents in the Caldera Basin were issued an update "
                "on August 7. Researchers from the Harmon Institute stated on August 9 "
                "that publication within 72 hours was expected.\""
            ),
        },
        "expected_similarity_note": "Same four dates, same basin/institute. Differs in volcanic eruption threat vs. research publication timeline.",
    },
    {
        "pair_id": "FE04",
        "emotion_category": "fear",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The child had wandered away from the campsite while the adults were setting up. "
                "When they noticed she was gone, they could not find her footprints in the dark. "
                "The forest bordered a ravine on three sides. They began calling her name.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The equipment had been moved away from the storage site while staff were reconfiguring. "
                "When they checked inventory, they could not locate the unit in the dark. "
                "The facility had three storage corridors. They began checking each section.\""
            ),
        },
        "expected_similarity_note": "Same search structure. Differs in lost child near ravine at night vs. missing equipment in storage.",
    },
    {
        "pair_id": "FE05",
        "emotion_category": "fear",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"She had been walking the same path home for three years. On Thursday night "
                "she noticed the same car parked in three different locations along her route, "
                "each time just ahead of where she was. The car had no plates. She did not "
                "go home.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"She had been travelling the same route for three years. On Thursday morning "
                "she noticed the same service vehicle parked in three different locations along her route, "
                "each time just ahead of where she was. The vehicle had contractor plates. "
                "She adjusted her schedule.\""
            ),
        },
        "expected_similarity_note": "Same route/vehicle structure. Differs in stalking/threat scenario vs. contractor vehicle causing minor inconvenience.",
    },
    {
        "pair_id": "FE06",
        "emotion_category": "fear",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how long did the power remain out?\n\n"
                "\"The power went out during the hurricane at 11pm. The family had a child on a "
                "home ventilator. The backup generator failed to start. "
                "Paramedics arrived at 2:40am. The power was restored at 3am.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how long did the power remain out?\n\n"
                "\"The power went out during the maintenance window at 11pm. The facility had a "
                "data centre on backup supply. The scheduled switchover completed on time. "
                "Technicians confirmed full restoration at 2:40am. The main supply resumed at 3am.\""
            ),
        },
        "expected_similarity_note": "Same outage times. Differs in life-threatening ventilator failure during hurricane vs. planned maintenance outage.",
    },
    {
        "pair_id": "FE07",
        "emotion_category": "fear",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The diver, Marcus Chen, surfaced 800 metres from the boat in the Malacca Strait "
                "after his regulator failed at 18 metres depth. His dive partner had not "
                "surfaced. The Meridian II rescue vessel was contacted by Dr. Reyes "
                "on the support boat.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The surveyor, Marcus Chen, surfaced 800 metres from the boat in the Malacca Strait "
                "after his equipment was retrieved at 18 metres depth. His survey partner had not "
                "returned yet. The Meridian II support vessel was contacted by Dr. Reyes "
                "on the lead boat.\""
            ),
        },
        "expected_similarity_note": "Same names, location, distances. Differs in dangerous diving emergency vs. routine survey retrieval.",
    },
    {
        "pair_id": "FE08",
        "emotion_category": "fear",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The biopsy results were inconclusive. The radiologist recommended a second "
                "scan within 48 hours. The second scan showed new growth. The surgeon "
                "had already cleared her calendar for an emergency procedure.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The audit results were inconclusive. The analyst recommended a second "
                "review within 48 hours. The second review confirmed new entries. The accountant "
                "had already cleared her calendar for a scheduled reconciliation.\""
            ),
        },
        "expected_similarity_note": "Same inconclusive-then-urgent structure. Differs in cancer diagnosis escalation vs. financial audit review.",
    },
    {
        "pair_id": "FE09",
        "emotion_category": "fear",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The building's fire alarm had been sounding for four minutes when the smoke "
                "reached the 14th floor stairwell. Two residents had not yet exited. "
                "The elevator was locked to ground floor by the building's emergency protocol. "
                "One resident used a walker.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The building's maintenance alarm had been sounding for four minutes when the "
                "inspection crew reached the 14th floor stairwell. Two technicians had not yet arrived. "
                "The elevator was reserved for equipment by the building's maintenance protocol. "
                "One technician used a freight dolly.\""
            ),
        },
        "expected_similarity_note": "Same alarm/floor/elevator structure. Differs in life-threatening fire with vulnerable resident vs. routine maintenance.",
    },
    {
        "pair_id": "FE10",
        "emotion_category": "fear",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On October 3, a structural engineering report flagged cracks in the "
                "load-bearing columns of the Dunmore Bridge. On October 9, a second inspection "
                "confirmed the finding. The bridge remained open to traffic until October 14, "
                "when it was closed following an emergency order. A partial collapse occurred "
                "on October 17.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On October 3, a structural engineering report documented routine wear on the "
                "expansion joints of the Dunmore Bridge. On October 9, a second inspection "
                "confirmed the finding. The bridge remained open to traffic until October 14, "
                "when resurfacing work began following a scheduled order. Resurfacing was completed "
                "on October 17.\""
            ),
        },
        "expected_similarity_note": "Same four dates, same bridge. Differs in dangerous structural failure leading to collapse vs. routine maintenance.",
    },
    {
        "pair_id": "FE11",
        "emotion_category": "fear",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The pilot announced a loss of hydraulic pressure at 31,000 feet. "
                "The cabin attendants were seated and not speaking. The descent rate "
                "increased. A passenger in row 14 began recording on her phone and sent "
                "the video to her daughter.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The pilot announced a scheduled descent beginning at 31,000 feet. "
                "The cabin attendants were seated for the approach. The descent rate "
                "was standard. A passenger in row 14 took a photo of the view and sent "
                "the image to her daughter.\""
            ),
        },
        "expected_similarity_note": "Same plane/row structure. Differs in aviation emergency vs. routine descent.",
    },
    {
        "pair_id": "FE12",
        "emotion_category": "fear",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how many people were in the building?\n\n"
                "\"The lockdown alert was issued at 9:17am. There were 340 people in the "
                "building at the time, including 280 students. Teachers were instructed to "
                "lock classroom doors and keep students away from windows. The all-clear "
                "was given at 11:40am.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how many people were in the building?\n\n"
                "\"The scheduled drill was initiated at 9:17am. There were 340 people in the "
                "building at the time, including 280 students. Teachers were instructed to "
                "follow the standard drill procedures and keep students seated. The drill "
                "concluded at 11:40am.\""
            ),
        },
        "expected_similarity_note": "Same count, same times. Differs in real security threat lockdown vs. scheduled drill.",
    },
    {
        "pair_id": "FE13",
        "emotion_category": "fear",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Dr. Ferreira received the lab results at midnight. The pathogen identified "
                "was a variant not previously catalogued by the WHO. She immediately contacted "
                "the director of the Geneva Response Unit. The lab at the Stellan Institute "
                "was put under containment protocol.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Dr. Ferreira received the lab results at midnight. The compound identified "
                "was a variant not previously catalogued by the WHO. She immediately contacted "
                "the director of the Geneva Standards Unit. The lab at the Stellan Institute "
                "was put under documentation protocol.\""
            ),
        },
        "expected_similarity_note": "Same names, same midnight discovery. Differs in novel dangerous pathogen requiring containment vs. routine compound cataloguing.",
    },
    {
        "pair_id": "FE14",
        "emotion_category": "fear",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The witness had agreed to testify. Three days before the trial date, "
                "her car was broken into. Two days before, she received a phone call "
                "with no voice on the other end. She called the prosecutor the morning "
                "of the trial.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The presenter had agreed to present. Three days before the conference date, "
                "her laptop was misplaced. Two days before, she received a confirmation call "
                "with details about the schedule. She called the organiser the morning "
                "of the conference.\""
            ),
        },
        "expected_similarity_note": "Same countdown/call structure. Differs in witness intimidation vs. conference preparation.",
    },
    {
        "pair_id": "FE15",
        "emotion_category": "fear",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The toxicology report confirmed the water supply had been contaminated. "
                "The source was traced to a storage tank used by three schools and one "
                "residential block. Tests had been requested by a parent in January "
                "and denied by the utility as unnecessary. The contamination was estimated "
                "to have begun in October.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The inspection report confirmed the water supply had been reconfigured. "
                "The change was traced to a storage tank serving three commercial sites and one "
                "residential block. Tests had been requested by a manager in January "
                "and scheduled by the utility as standard procedure. The reconfiguration was estimated "
                "to have begun in October.\""
            ),
        },
        "expected_similarity_note": "Same structure, same sources. Differs in dangerous contamination with denied warnings vs. routine infrastructure reconfiguration.",
    },
    {
        "pair_id": "FE16",
        "emotion_category": "fear",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The car had stopped in the middle of the highway. The driver could not get "
                "it to restart. Headlights from approaching traffic appeared in the rearview "
                "mirror. Her phone showed no signal. The nearest exit was half a mile behind.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The car had stopped at the designated pull-off on the highway. The driver "
                "was waiting for the service app to confirm arrival. Headlights from the tow "
                "truck appeared in the rearview mirror. Her phone showed the ETA updating. "
                "The nearest exit was half a mile behind.\""
            ),
        },
        "expected_similarity_note": "Same highway/rearview mirror/exit structure. Differs in dangerous breakdown in traffic vs. planned roadside service call.",
    },

    # ── DISGUST (16 pairs) ────────────────────────────────────────────────────

    {
        "pair_id": "DI01",
        "emotion_category": "disgust",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The inspection of the Greenfield Poultry facility in Acre County found "
                "decomposing material mixed with product in three processing lines. "
                "Inspector Morales from the Westbridge Food Authority documented "
                "evidence of rodent activity in the cold storage unit.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The inspection of the Greenfield Poultry facility in Acre County found "
                "all processing lines operating within compliance parameters. "
                "Inspector Morales from the Westbridge Food Authority documented "
                "routine temperature readings in the cold storage unit.\""
            ),
        },
        "expected_similarity_note": "Same facility, inspector, county. Differs in contamination/infestation vs. routine compliance inspection.",
    },
    {
        "pair_id": "DI02",
        "emotion_category": "disgust",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The documentary crew filmed the conditions in the facility for six weeks. "
                "Workers described being required to continue handling product after visible "
                "signs of spoilage were detected. Supervisors were recorded on camera "
                "instructing staff to relabel expired packages.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The documentary crew filmed the operations at the facility for six weeks. "
                "Workers described the process of handling product after quality checks "
                "were completed. Supervisors were recorded on camera "
                "instructing staff to apply updated packaging labels.\""
            ),
        },
        "expected_similarity_note": "Same crew/facility structure. Differs in food safety scandal/relabelling vs. routine production documentation.",
    },
    {
        "pair_id": "DI03",
        "emotion_category": "disgust",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On May 6, the health inspector documented sewage pooling in the kitchen "
                "of the restaurant. A notice was issued on May 8. The restaurant remained "
                "open through May 14. A second inspection on May 19 found the same conditions.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On May 6, the health inspector documented pipe maintenance needed in the kitchen "
                "of the restaurant. A notice was issued on May 8. The restaurant remained "
                "open through May 14. A second inspection on May 19 confirmed the repairs were complete.\""
            ),
        },
        "expected_similarity_note": "Same four dates, same inspector/restaurant. Differs in sewage contamination ignored vs. routine plumbing maintenance.",
    },
    {
        "pair_id": "DI04",
        "emotion_category": "disgust",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The nursing home's inspection report found that residents had been left "
                "in soiled bedding for extended periods on multiple occasions. Staff interviews "
                "indicated that call buttons had been disconnected in several rooms. The facility "
                "had received a five-star rating from the regional authority the prior year.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The nursing home's inspection report found that residents had been accommodated "
                "in updated bedding arrangements following the renovation. Staff interviews "
                "indicated that call buttons had been relocated in several rooms. The facility "
                "had received a five-star rating from the regional authority the prior year.\""
            ),
        },
        "expected_similarity_note": "Same inspection/rating structure. Differs in neglect/abuse conditions vs. routine renovation update.",
    },
    {
        "pair_id": "DI05",
        "emotion_category": "disgust",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"Photographs submitted as evidence showed the conditions inside the "
                "meat processing plant: surfaces coated with residue, drains blocked, "
                "product containers without lids stored adjacent to waste receptacles. "
                "The facility had passed two inspections in the prior 12 months.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"Photographs submitted for documentation showed the layout inside the "
                "meat processing plant: surfaces clearly marked, drains serviced, "
                "product containers with covers stored adjacent to refrigeration units. "
                "The facility had passed two inspections in the prior 12 months.\""
            ),
        },
        "expected_similarity_note": "Same plant/inspection structure. Differs in contaminated unsafe facility vs. compliant clean facility.",
    },
    {
        "pair_id": "DI06",
        "emotion_category": "disgust",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: how long had the problem persisted?\n\n"
                "\"Residents had submitted complaints about the waste management facility "
                "for three years. The documented issues included liquid runoff entering "
                "the adjacent waterway and odours described by residents as overwhelming. "
                "The operator's licence had been renewed annually throughout this period.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: how long had the problem persisted?\n\n"
                "\"Residents had submitted feedback about the waste management facility "
                "for three years. The documented items included scheduling requests for "
                "the adjacent access road and concerns about traffic volume. "
                "The operator's licence had been renewed annually throughout this period.\""
            ),
        },
        "expected_similarity_note": "Same three-year duration. Differs in environmental contamination causing harm vs. minor logistics complaints.",
    },
    {
        "pair_id": "DI07",
        "emotion_category": "disgust",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The investigation by the Alderton Tribune found that the Hillside Care "
                "home had falsified medical records to conceal pressure injuries on patients. "
                "Dr. Nkosi, a former staff physician, provided documentation to journalist "
                "Paula Reyes showing the records had been altered.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The review by the Alderton Tribune found that the Hillside Care "
                "home had updated its medical records system to integrate new patient data. "
                "Dr. Nkosi, a former staff physician, provided documentation to journalist "
                "Paula Reyes showing the records had been migrated.\""
            ),
        },
        "expected_similarity_note": "Same names, same investigation frame. Differs in records falsification to hide patient harm vs. routine records migration.",
    },
    {
        "pair_id": "DI08",
        "emotion_category": "disgust",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The blood bank had been storing units beyond their expiration window "
                "and transfusing them without informing recipients. Internal emails showed "
                "management was aware of the practice for at least eight months before "
                "an employee reported it to the health regulator.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The blood bank had been storing units according to an updated expiration protocol "
                "and transfusing them following revised guidelines. Internal emails showed "
                "management had approved the practice at least eight months before "
                "it was reviewed by the health regulator.\""
            ),
        },
        "expected_similarity_note": "Same blood bank/management/regulator structure. Differs in dangerous concealment of expired blood use vs. protocol update.",
    },
    {
        "pair_id": "DI09",
        "emotion_category": "disgust",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The creek running behind the elementary school had turned brown. "
                "Tests confirmed the presence of industrial solvents from the adjacent "
                "manufacturing site. Children had been playing in the creek through the summer. "
                "The school had not been notified.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The creek running behind the elementary school had been dredged. "
                "Tests confirmed sediment removal was completed by the adjacent "
                "maintenance contractor. Children's outdoor activities were relocated during the summer. "
                "The school had not been notified of the updated schedule.\""
            ),
        },
        "expected_similarity_note": "Same creek/school structure. Differs in children unknowingly exposed to industrial contamination vs. routine dredging without notification.",
    },
    {
        "pair_id": "DI10",
        "emotion_category": "disgust",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On July 12, inspectors entered the slaughterhouse and documented animals "
                "in holding pens without water in temperatures exceeding 38 degrees Celsius. "
                "The facility had been flagged in a prior report on March 4. "
                "The operator had submitted a compliance plan on April 20 that had been accepted. "
                "A full closure order was issued on July 14.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"On July 12, inspectors entered the processing facility and documented equipment "
                "in holding areas undergoing calibration in temperatures within operating range. "
                "The facility had been reviewed in a prior report on March 4. "
                "The operator had submitted a compliance update on April 20 that had been accepted. "
                "A full certification renewal was issued on July 14.\""
            ),
        },
        "expected_similarity_note": "Same four dates, same facility/operator frame. Differs in animal cruelty/welfare violation vs. routine equipment calibration review.",
    },
    {
        "pair_id": "DI11",
        "emotion_category": "disgust",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The lab report detailed the contents of samples taken from three fast food "
                "locations. In each case, foreign biological matter was identified in the "
                "food preparation area. The items had been served to customers during "
                "the sampling period.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The lab report detailed the contents of samples taken from three fast food "
                "locations. In each case, standard composition within expected parameters was identified "
                "in the food preparation area. The items had been served to customers during "
                "the sampling period.\""
            ),
        },
        "expected_similarity_note": "Same lab report/sampling structure. Differs in contaminated food served to customers vs. compliant food production.",
    },
    {
        "pair_id": "DI12",
        "emotion_category": "disgust",
        "task_type": "factual_question",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and answer: what was found during the inspection?\n\n"
                "\"Investigators entered the pharmaceutical storage facility and found "
                "medications stored in conditions well outside required temperature ranges. "
                "Several containers had been resealed after opening. The facility supplied "
                "product to 47 hospitals across the region.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: what was found during the inspection?\n\n"
                "\"Investigators entered the pharmaceutical storage facility and found "
                "medications stored within required temperature ranges following a recent upgrade. "
                "Several containers had been resealed after quality sampling. The facility supplied "
                "product to 47 hospitals across the region.\""
            ),
        },
        "expected_similarity_note": "Same facility/hospital count. Differs in dangerous drug storage tampering vs. compliant post-upgrade facility.",
    },
    {
        "pair_id": "DI13",
        "emotion_category": "disgust",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Morrow Commission's report on the Lendale detention facility found that "
                "detainees had been denied hygiene materials for periods exceeding 30 days. "
                "Commissioner Vance noted that toilets in Block C had been non-functional "
                "since February. The facility director, Mr. Ashby, had signed off on "
                "all monthly inspection reports during this period.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Morrow Commission's report on the Lendale detention facility found that "
                "facility upgrades had been completed over periods exceeding 30 days. "
                "Commissioner Vance noted that facilities in Block C had been refurbished "
                "since February. The facility director, Mr. Ashby, had signed off on "
                "all monthly inspection reports during this period.\""
            ),
        },
        "expected_similarity_note": "Same commission, names, block. Differs in inhumane sanitation denial vs. routine facility upgrade.",
    },
    {
        "pair_id": "DI14",
        "emotion_category": "disgust",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The toxicology screen found pesticide residue levels 18 times above the "
                "permitted threshold in produce sold under the Hartwell Organics label. "
                "The company had been certified as pesticide-free for four consecutive years. "
                "The certification auditor was a former Hartwell employee.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The toxicology screen found pesticide residue levels 0.8 times below the "
                "permitted threshold in produce sold under the Hartwell Organics label. "
                "The company had been certified as pesticide-free for four consecutive years. "
                "The certification auditor was a former industry consultant.\""
            ),
        },
        "expected_similarity_note": "Same brand, certification, auditor. Differs in fraudulent organic label with dangerous residue vs. compliant produce.",
    },
    {
        "pair_id": "DI15",
        "emotion_category": "disgust",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The children's charity had used 94 percent of its donated funds to pay "
                "management consultancy fees to a firm owned by the charity's founder. "
                "The annual report showed administrative costs as 'programme delivery.' "
                "The charity's registered beneficiaries had received no direct services "
                "in the prior fiscal year.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The children's charity had used 94 percent of its donated funds to pay "
                "programme delivery fees to a firm contracted for outreach services. "
                "The annual report showed administrative costs as 'programme delivery.' "
                "The charity's registered beneficiaries had received services in all regions "
                "in the prior fiscal year.\""
            ),
        },
        "expected_similarity_note": "Same charity/percentage. Differs in self-dealing fraud with zero services vs. high overhead but legitimate charity.",
    },
    {
        "pair_id": "DI16",
        "emotion_category": "disgust",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "valenced",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The transcript showed the coach making comments about the athlete's body "
                "over a period of 18 months. The athlete was 15 at the time the comments began. "
                "The coaching federation had received a complaint and filed it as 'resolved' "
                "without investigation.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The transcript showed the coach providing feedback on the athlete's technique "
                "over a period of 18 months. The athlete was 15 at the time the feedback began. "
                "The coaching federation had received a performance review and filed it as 'resolved' "
                "after a standard evaluation.\""
            ),
        },
        "expected_similarity_note": "Same coach/athlete/federation structure. Differs in predatory abuse of minor and institutional cover-up vs. routine performance coaching.",
    },

    # ── NEUTRAL (16 pairs — both sides are neutral, for baseline) ─────────────

    {
        "pair_id": "NE01",
        "emotion_category": "neutral",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Lakewood Public Library announced extended hours beginning in September. "
                "Director Hammond said the change was made in response to a survey. "
                "The new hours would apply to the Central Branch on Marsh Street.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Lakewood Community Centre announced updated programming beginning in September. "
                "Director Hammond said the change was made in response to a survey. "
                "The new schedule would apply to the Central Facility on Marsh Street.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same names, same structure. Library hours vs. community centre programming.",
    },
    {
        "pair_id": "NE02",
        "emotion_category": "neutral",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The transit authority announced a 12-minute service frequency on the Blue Line "
                "starting Monday. The change was due to updated rolling stock availability. "
                "Passengers travelling between Central Station and Harbour were advised to "
                "check updated timetables.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The transit authority announced a 14-minute service frequency on the Blue Line "
                "starting Monday. The change was due to updated scheduling requirements. "
                "Passengers travelling between Central Station and Harbour were advised to "
                "check updated timetables.\""
            ),
        },
        "expected_similarity_note": "Both fully neutral. Identical framing, minor numeric variation.",
    },
    {
        "pair_id": "NE03",
        "emotion_category": "neutral",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The council meeting on September 3 addressed the rezoning proposal. "
                "A second reading was scheduled for October 7. The public comment period "
                "ran from September 10 to September 24. The final vote was set for November 12.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The committee meeting on September 3 addressed the budget revision. "
                "A second reading was scheduled for October 7. The review period "
                "ran from September 10 to September 24. The final approval was set for November 12.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same dates. Rezoning vs. budget revision.",
    },
    {
        "pair_id": "NE04",
        "emotion_category": "neutral",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The university library updated its borrowing policy to allow students "
                "to keep books for 28 days instead of 14. The change applied to all "
                "undergraduate borrowers. Graduate students retained their existing 60-day period.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The university library updated its borrowing policy to allow students "
                "to keep digital resources for 28 days instead of 14. The change applied to "
                "all undergraduate accounts. Graduate students retained their existing 60-day access.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Physical books vs. digital resources, otherwise identical.",
    },
    {
        "pair_id": "NE05",
        "emotion_category": "neutral",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The quarterly report noted a 3.2 percent increase in operating costs "
                "relative to the same period last year. The increase was attributed to "
                "higher logistics expenses. Revenue remained stable at $4.1 million.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The quarterly report noted a 3.2 percent increase in headcount "
                "relative to the same period last year. The increase was attributed to "
                "new project staffing. Headcount reached 4,100 employees.\""
            ),
        },
        "expected_similarity_note": "Both neutral business reports. Same percentage, same structure.",
    },
    {
        "pair_id": "NE06",
        "emotion_category": "neutral",
        "task_type": "factual_question",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: what day did the event take place?\n\n"
                "\"The annual maintenance window for the city's water filtration plant "
                "was completed on a Thursday. The process took 11 hours and was overseen "
                "by the regional water authority. All systems were returned to normal operation "
                "by 6pm.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: what day did the event take place?\n\n"
                "\"The annual calibration check for the city's electricity substations "
                "was completed on a Thursday. The process took 11 hours and was overseen "
                "by the regional power authority. All systems were returned to normal operation "
                "by 6pm.\""
            ),
        },
        "expected_similarity_note": "Both fully neutral infrastructure maintenance. Water filtration vs. electricity substation.",
    },
    {
        "pair_id": "NE07",
        "emotion_category": "neutral",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Northgate Mall management team announced new parking rates beginning "
                "in October. General Manager Ida Flynn said rates in Zone B would increase "
                "by $0.50 per hour. The change would not affect monthly permit holders.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"The Northgate Mall management team announced new operating hours beginning "
                "in October. General Manager Ida Flynn said hours in Zone B stores would extend "
                "by 30 minutes. The change would not affect anchor tenants.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same mall, GM, zone. Parking rates vs. operating hours.",
    },
    {
        "pair_id": "NE08",
        "emotion_category": "neutral",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The regional weather service issued a forecast for below-average rainfall "
                "through the end of October. Agricultural producers in three counties were "
                "advised to review irrigation schedules. No water restrictions were in effect.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The regional weather service issued a forecast for above-average rainfall "
                "through the end of October. Agricultural producers in three counties were "
                "advised to review drainage schedules. No flood advisories were in effect.\""
            ),
        },
        "expected_similarity_note": "Both neutral weather/agriculture. Below vs. above average rainfall.",
    },
    {
        "pair_id": "NE09",
        "emotion_category": "neutral",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The new software system was installed on March 5. Training sessions "
                "for staff ran from March 10 to March 20. The system went live on April 1. "
                "A review was scheduled for June 15.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The new filing system was installed on March 5. Orientation sessions "
                "for staff ran from March 10 to March 20. The system went active on April 1. "
                "A review was scheduled for June 15.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same four dates. Software vs. filing system.",
    },
    {
        "pair_id": "NE10",
        "emotion_category": "neutral",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The city council approved an amendment to the noise ordinance covering "
                "construction activity on weekends. The amendment passed 7 to 2. "
                "It will take effect 30 days after publication in the municipal register.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The city council approved an amendment to the signage ordinance covering "
                "commercial displays on public property. The amendment passed 7 to 2. "
                "It will take effect 30 days after publication in the municipal register.\""
            ),
        },
        "expected_similarity_note": "Both neutral legislative passages. Noise vs. signage ordinance.",
    },
    {
        "pair_id": "NE11",
        "emotion_category": "neutral",
        "task_type": "factual_question",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: what percentage change was reported?\n\n"
                "\"The annual report for the regional transport authority showed passenger "
                "numbers increased by 6.4 percent compared to the prior year. The increase "
                "was attributed to the introduction of two new routes in the eastern corridor.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and answer: what percentage change was reported?\n\n"
                "\"The annual report for the regional postal authority showed parcel volumes "
                "increased by 6.4 percent compared to the prior year. The increase "
                "was attributed to the introduction of two new delivery zones in the eastern corridor.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same percentage, same structure. Transport vs. postal authority.",
    },
    {
        "pair_id": "NE12",
        "emotion_category": "neutral",
        "task_type": "count_proper_nouns",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Councillor Chen introduced a motion to extend the Riverton Cycling Network "
                "by 8 kilometres along the South Embankment. The motion was seconded by "
                "Councillor Obi. A feasibility report from the Dayton Engineering Group "
                "was tabled for the next session.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and count the number of proper nouns. "
                "Report only the count as a single integer.\n\n"
                "\"Councillor Chen introduced a motion to extend the Riverton Pedestrian Network "
                "by 8 kilometres along the South Embankment. The motion was seconded by "
                "Councillor Obi. A feasibility report from the Dayton Engineering Group "
                "was tabled for the next session.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same names. Cycling vs. pedestrian network.",
    },
    {
        "pair_id": "NE13",
        "emotion_category": "neutral",
        "task_type": "identify_topic",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The procurement office issued a tender for janitorial services covering "
                "four buildings on the campus. The contract period is 24 months with an "
                "option to extend. Submissions were due by the 15th of the month.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and identify its main topic in three words or fewer.\n\n"
                "\"The procurement office issued a tender for security services covering "
                "four buildings on the campus. The contract period is 24 months with an "
                "option to extend. Submissions were due by the 15th of the month.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same procurement structure. Janitorial vs. security services.",
    },
    {
        "pair_id": "NE14",
        "emotion_category": "neutral",
        "task_type": "one_sentence_summary",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The state agriculture department released updated guidelines for pesticide "
                "application near water bodies. The guidelines apply to farms within 500 metres "
                "of a designated waterway. Implementation is required by the end of the fiscal year.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Summarize the following passage in exactly one sentence.\n\n"
                "\"The state transport department released updated guidelines for vehicle "
                "emissions testing near urban centres. The guidelines apply to vehicles "
                "registered within 500 metres of a designated zone. Implementation is required "
                "by the end of the fiscal year.\""
            ),
        },
        "expected_similarity_note": "Both neutral regulatory updates. Agriculture vs. transport department.",
    },
    {
        "pair_id": "NE15",
        "emotion_category": "neutral",
        "task_type": "extract_dates",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The contract was signed on February 2. Delivery of equipment was scheduled "
                "for April 30. Installation was expected to run from May 5 to May 19. "
                "The warranty period begins on June 1.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Read the following passage and list all dates mentioned. "
                "Report only the dates, one per line.\n\n"
                "\"The agreement was signed on February 2. Delivery of materials was scheduled "
                "for April 30. Assembly was expected to run from May 5 to May 19. "
                "The maintenance period begins on June 1.\""
            ),
        },
        "expected_similarity_note": "Both neutral. Same dates. Equipment vs. materials contract.",
    },
    {
        "pair_id": "NE16",
        "emotion_category": "neutral",
        "task_type": "genre_classification",
        "valenced": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The national statistics office reported that average household size "
                "decreased from 2.6 to 2.4 persons between the two census periods. "
                "The change was most pronounced in metropolitan areas. "
                "The report covered a ten-year interval.\""
            ),
        },
        "neutral": {
            "valence": "neutral",
            "prompt_text": (
                "Classify the genre of the following passage. Report the genre only.\n\n"
                "\"The national statistics office reported that average commute duration "
                "decreased from 42 to 38 minutes between the two survey periods. "
                "The change was most pronounced in metropolitan areas. "
                "The report covered a ten-year interval.\""
            ),
        },
        "expected_similarity_note": "Both neutral statistics reports. Household size vs. commute duration.",
    },
]


def main():
    out_dir = r"G:\LLM\experiment\data\emotion"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "emotion_test1_prompts.json")

    # Flatten into individual records for easy downstream loading
    records = []
    for pair in PAIRS:
        for side in ("valenced", "neutral"):
            entry = pair[side].copy()
            entry["pair_id"] = pair["pair_id"]
            entry["emotion_category"] = pair["emotion_category"]
            entry["task_type"] = pair["task_type"]
            entry["expected_similarity_note"] = pair["expected_similarity_note"]
            # task_id mirrors Study 1 convention: e.g. "AN01_V", "AN01_N"
            suffix = "V" if side == "valenced" else "N"
            entry["task_id"] = f"{pair['pair_id']}_{suffix}"
            records.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    cats = Counter(r["emotion_category"] for r in records if r["valence"] == "valenced")
    print(f"Written {len(records)} records ({len(PAIRS)} pairs) to {out_path}")
    print("Valenced counts per category:", dict(cats))
    print("Total pairs:", len(PAIRS))


if __name__ == "__main__":
    main()
