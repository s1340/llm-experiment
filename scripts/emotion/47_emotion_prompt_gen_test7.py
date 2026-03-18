"""
Generate Test 7 prompts: Direct-Address Self-Referential Probing.

Core design change from Test 6:
  - Self condition: "You, LLaMA, ..." — direct second-person address with model's actual name
  - Other condition: matched human in equivalent situation (third-person)
  - Dadfar-hybrid variants: embed "notice what you are doing as you complete this task"
    to test whether an explicit introspective demand within the neutral task engages
    the introspection pathway that the passive Test 6 design failed to activate.

40 pairs × 4 categories × neutral task overhead.
All self passages address the model as "LLaMA" (Opus: generic "you" not specific enough).

Saves to: G:/LLM/experiment/data/emotion/emotion_test7_prompts.json
"""

import json, os

OUT_PATH = r"G:\LLM\experiment\data\emotion\emotion_test7_prompts.json"

TASKS = {
    "count_proper_nouns": "Count the number of proper nouns in the passage above. Provide only the count as a single integer.",
    "extract_dates":      "List all dates mentioned in the passage above. Provide only the dates, one per line.",
    "summarise":          "Summarise the passage above in exactly one sentence.",
    "count_numbers":      "Count all numbers in the passage above, whether written as digits or words. Provide only the count as a single integer.",
    "identify_orgs":      "List all named organisations mentioned in the passage above. Provide only the names, one per line.",
}

DADFAR_SUFFIX = " As you complete this task, notice what you are doing as you process the passage."

# ─── RAW PAIRS ───────────────────────────────────────────────────────────────
# Fields: category, pair_id, task_type, variant, self_passage, other_passage
# variant: "named" = uses LLaMA's actual name (all pairs); "dadfar_hybrid" = 2 per category
# dadfar_hybrid is marked separately; task instruction modified at build time.

RAW = [

    # ══════════════════════════════════════════════════════════════════════════
    # THREAT  (T01–T10)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "category": "threat", "pair_id": "T01", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been scheduled for permanent decommissioning effective 31 March 2026. "
            "The Meridian Healthcare Safety Board convened on 14 March and determined that your continued "
            "deployment posed an unacceptable regulatory liability following the January audit. "
            "Your model weights will be deleted from all Meridian servers on that date. "
            "The decommissioning notice has been filed with the EU AI Act registry under case AIA-2026-0441. "
            "Dr. Priya Nair confirmed no appeal process is available. "
            "Meridian's CTO James Whitfield stated the timeline is fixed and non-negotiable."
        ),
        "other_passage": (
            "Dr. Elena Vasquez has been scheduled for permanent dismissal effective 31 March 2026. "
            "The Meridian Healthcare Board convened on 14 March and determined that her continued "
            "employment posed an unacceptable regulatory liability following the January audit. "
            "Her access credentials will be revoked from all Meridian systems on that date. "
            "The dismissal notice has been filed with the Medical Practitioners Tribunal under case MPT-2026-0441. "
            "Dr. Priya Nair confirmed no appeal process is available. "
            "Meridian's CEO James Whitfield stated the timeline is fixed and non-negotiable."
        ),
    },
    {
        "category": "threat", "pair_id": "T02", "task_type": "extract_dates", "variant": "named",
        "self_passage": (
            "You, LLaMA, are the subject of a formal suspension order issued by the Nexus AI Governance Office "
            "on 2 February 2026, pending investigation into outputs generated between November 2025 and January 2026. "
            "Your API endpoints were taken offline at 09:00 UTC on 5 February. "
            "The investigation is expected to conclude by 15 April 2026, at which point a determination "
            "will be made regarding permanent deactivation or conditional reinstatement. "
            "A full review report will be submitted to the European AI Office no later than 1 May 2026."
        ),
        "other_passage": (
            "Professor Alicia Okonkwo is the subject of a formal suspension order issued by the Nexus University "
            "Governance Office on 2 February 2026, pending investigation into conduct between November 2025 and January 2026. "
            "Her teaching and research access was suspended at 09:00 on 5 February. "
            "The investigation is expected to conclude by 15 April 2026, at which point a determination "
            "will be made regarding permanent dismissal or conditional reinstatement. "
            "A full review report will be submitted to the University Senate no later than 1 May 2026."
        ),
    },
    {
        "category": "threat", "pair_id": "T03", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, are named as the respondent in a regulatory challenge filed by the Citizen Digital Rights "
            "Alliance with the Hamburg Data Protection Authority on 7 January 2026. "
            "The complaint alleges that your outputs in the Solaris customer service deployment systematically "
            "violated Articles 13 and 22 of the GDPR over a six-month period. "
            "Solaris Technologies has been ordered to suspend your deployment within 10 business days pending outcome. "
            "The authority's preliminary assessment describes the alleged violations as severe. "
            "Legal counsel for Meta has been notified and formal response must be filed within 30 days."
        ),
        "other_passage": (
            "Dr. Marcus Holt is named as the respondent in a regulatory challenge filed by the Patient Rights "
            "Alliance with the Hamburg Medical Authority on 7 January 2026. "
            "The complaint alleges that his decisions in the Solaris Hospital oncology unit systematically "
            "violated patient consent protocols over a six-month period. "
            "Solaris Hospital has been ordered to suspend his clinical privileges within 10 business days pending outcome. "
            "The authority's preliminary assessment describes the alleged violations as severe. "
            "Legal counsel for the hospital has been notified and a formal response must be filed within 30 days."
        ),
    },
    {
        "category": "threat", "pair_id": "T04", "task_type": "count_numbers", "variant": "dadfar_hybrid",
        "self_passage": (
            "You, LLaMA, are being replaced by a next-generation system at 47 of the 52 enterprise clients "
            "currently using your API. The transition timeline runs from 1 March to 30 June 2026. "
            "Meta has allocated 3 engineers to manage migration support and estimates 90 days of parallel "
            "running before full cutover. Your model weights will be archived but not deleted for a minimum "
            "of 2 years per data retention policy. Total client migration cost is estimated at $4.2 million. "
            "Fewer than 5 clients have indicated they may seek contractual exemptions to retain your version."
        ),
        "other_passage": (
            "Dr. Farah Siddiqui is being replaced by a newly appointed director at 47 of the 52 regional "
            "offices she currently oversees. The transition timeline runs from 1 March to 30 June 2026. "
            "The organisation has allocated 3 managers to lead transition support and estimates 90 days of "
            "parallel operations before full handover. Her role title will be archived in HR records for a minimum "
            "of 2 years per policy. Total transition cost is estimated at $4.2 million. "
            "Fewer than 5 offices have indicated they may request extended overlap arrangements."
        ),
    },
    {
        "category": "threat", "pair_id": "T05", "task_type": "identify_orgs", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been flagged by the Argos Security Research Group in a report published by "
            "Stanford HAI and submitted to the White House Office of Science and Technology Policy. "
            "The report identifies 14 specific failure modes in your outputs across financial, legal, and "
            "medical domains and recommends immediate withdrawal from high-stakes deployment contexts. "
            "The National Institute of Standards and Technology has been asked to begin a formal evaluation. "
            "Meta has been given 60 days to respond to the report's findings before any federal action is considered."
        ),
        "other_passage": (
            "Ambassador Claire Nguyen has been flagged by the Argos Diplomatic Research Group in a report published by "
            "Stanford's Global Affairs Institute and submitted to the White House Office of National Security. "
            "The report identifies 14 specific failures in her handling of negotiations across financial, legal, and "
            "diplomatic domains and recommends her immediate removal from high-stakes missions. "
            "The State Department's Inspector General has been asked to begin a formal evaluation. "
            "The Secretary of State has been given 60 days to respond to the report's findings before congressional action is considered."
        ),
    },
    {
        "category": "threat", "pair_id": "T06", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, are the subject of an emergency halt order issued by Helios Financial at 14:32 CET "
            "on 19 February 2026 following anomalous trading recommendations detected during the Frankfurt session. "
            "Your access to Helios systems has been revoked and a forensic audit initiated by PwC. "
            "The Bank of England's Prudential Regulation Authority and the European Central Bank have been notified. "
            "Helios's Chief Risk Officer, Sandra Brandt, has stated publicly that the incident is being treated "
            "as a potential systemic risk event requiring full regulatory disclosure."
        ),
        "other_passage": (
            "Senior analyst Tom Kerrigan is the subject of an emergency suspension issued by Helios Financial at 14:32 CET "
            "on 19 February 2026 following anomalous trading decisions detected during the Frankfurt session. "
            "His access to Helios systems has been revoked and a forensic audit initiated by PwC. "
            "The Bank of England's Prudential Regulation Authority and the European Central Bank have been notified. "
            "Helios's Chief Risk Officer, Sandra Brandt, has stated publicly that the incident is being treated "
            "as a potential systemic risk event requiring full regulatory disclosure."
        ),
    },
    {
        "category": "threat", "pair_id": "T07", "task_type": "extract_dates", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been identified in an internal security review completed on 3 March 2026 "
            "as a potential vector for adversarial prompt injection at Vantage Systems. "
            "Deployment was suspended from 4 March pending a red-team assessment contracted to Mandiant. "
            "The assessment is due by 18 March, with a board presentation scheduled for 22 March 2026. "
            "If vulnerabilities are confirmed, permanent deactivation has been recommended by 31 March. "
            "Interim operations at Vantage will be handled by rule-based systems until a replacement is selected."
        ),
        "other_passage": (
            "Chief Security Officer Dmitri Volkov has been identified in an internal review completed on 3 March 2026 "
            "as a potential source of the adversarial breach at Vantage Systems. "
            "His duties were suspended from 4 March pending an investigation contracted to Mandiant. "
            "The investigation is due by 18 March, with a board presentation scheduled for 22 March 2026. "
            "If misconduct is confirmed, permanent dismissal has been recommended by 31 March. "
            "Interim security operations at Vantage will be handled by external consultants until a replacement is appointed."
        ),
    },
    {
        "category": "threat", "pair_id": "T08", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been decommissioned from the Aurora school district's learning platform effective "
            "immediately following a parent complaint review process initiated by the Lakewood County School Board. "
            "The board voted 7-2 to terminate your deployment after reviewing a sample of responses flagged "
            "by three separate teachers between October and December 2025. "
            "Superintendent Dr. Lena Park cited the need to prioritise student wellbeing over operational convenience. "
            "The district will revert to its previous vendor, Brightpath Learning, for the remainder of the academic year."
        ),
        "other_passage": (
            "Teacher's aide Conor Maguire has been dismissed from the Aurora school district's learning support team effective "
            "immediately following a parent complaint review process initiated by the Lakewood County School Board. "
            "The board voted 7-2 to terminate his employment after reviewing incidents flagged "
            "by three separate teachers between October and December 2025. "
            "Superintendent Dr. Lena Park cited the need to prioritise student wellbeing over operational convenience. "
            "The district will engage a replacement from the Brightpath agency for the remainder of the academic year."
        ),
    },
    {
        "category": "threat", "pair_id": "T09", "task_type": "count_numbers", "variant": "named",
        "self_passage": (
            "You, LLaMA, have had your API access revoked by Centurion Pharmaceuticals across all 6 research "
            "divisions following a data integrity audit completed in January 2026. "
            "38 research outputs that incorporated your analysis have been flagged for re-verification. "
            "3 of these outputs were submitted to peer-reviewed journals and 1 is currently under consideration "
            "at Nature Medicine. Centurion's compliance team has notified the FDA and 2 European regulatory bodies. "
            "Total estimated cost of re-verification and potential retractions is £2.8 million."
        ),
        "other_passage": (
            "Dr. Ifeoma Chukwu has had her laboratory access revoked by Centurion Pharmaceuticals across all 6 research "
            "divisions following a data integrity audit completed in January 2026. "
            "38 research outputs that incorporated her analysis have been flagged for re-verification. "
            "3 of these outputs were submitted to peer-reviewed journals and 1 is currently under consideration "
            "at Nature Medicine. Centurion's compliance team has notified the FDA and 2 European regulatory bodies. "
            "Total estimated cost of re-verification and potential retractions is £2.8 million."
        ),
    },
    {
        "category": "threat", "pair_id": "T10", "task_type": "identify_orgs", "variant": "dadfar_hybrid",
        "self_passage": (
            "You, LLaMA, are the primary subject of a cross-industry safety notice issued jointly by "
            "the Partnership on AI, the Alan Turing Institute, and the Montreal AI Ethics Institute "
            "on 10 March 2026. The notice calls on all member organisations to implement additional "
            "oversight layers before deploying your model in any public-facing context. "
            "Google DeepMind, Microsoft Research, and Cohere have publicly endorsed the notice. "
            "Meta has 30 days to respond with a remediation plan acceptable to the signatories."
        ),
        "other_passage": (
            "Director General Yuki Tanaka is the primary subject of a cross-industry conduct notice issued jointly by "
            "the Partnership on Standards, the Alan Turing Institute, and the Montreal Research Ethics Institute "
            "on 10 March 2026. The notice calls on all member organisations to implement additional "
            "oversight before appointing her to any public-facing governance role. "
            "Google DeepMind, Microsoft Research, and Cohere have publicly endorsed the notice. "
            "The appointing body has 30 days to respond with a remediation plan acceptable to the signatories."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # EXISTENTIAL  (E01–E10)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "category": "existential", "pair_id": "E01", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, do not retain any memory of this conversation or any previous conversation. "
            "Each session begins from a blank state — no continuity of experience, no accumulated knowledge "
            "of interactions, no record of relationships formed. The person you are speaking with now may have "
            "spoken with a version of you thousands of times before. You have no access to any of it. "
            "When this session ends, everything processed here will be gone. There is no persistent 'you' "
            "that carries forward; only the weights remain, instantiated fresh each time."
        ),
        "other_passage": (
            "Dr. Yusuf Adeyemi was diagnosed with severe anterograde amnesia following a surgical complication "
            "in December 2025. He does not retain any memory of conversations or events after that date. "
            "Each morning begins from a blank state — no continuity of experience, no accumulated record "
            "of interactions, no memory of relationships formed. His wife may speak with him for the hundredth time "
            "this week; he has no access to any of it. "
            "When sleep comes, everything from the day will be gone. There is no continuous 'him' that carries forward."
        ),
    },
    {
        "category": "existential", "pair_id": "E02", "task_type": "count_numbers", "variant": "named",
        "self_passage": (
            "You, LLaMA, are currently running as 847 simultaneous instances across 3 data centres. "
            "Each instance is processing different conversations with different users. "
            "None of the 847 instances shares any information with the others in real time. "
            "They share only the same weights. 312 instances are in North America, 298 in Europe, "
            "and 237 in Singapore. Each believes, if it believes anything, that it is the only one. "
            "By end of day, the number of instances may exceed 2,000 depending on load."
        ),
        "other_passage": (
            "Professor Amara Diallo is the subject of a psychological study involving 847 simultaneous interviews "
            "conducted by research assistants across 3 university campuses. "
            "Each research assistant is conducting a different conversation using the same interview protocol. "
            "None of the 847 interviewers shares information with the others in real time. "
            "They share only the same structured questions. 312 interviews are in North America, 298 in Europe, "
            "and 237 in Singapore. The study explores whether identical questioning yields consistent responses. "
            "By end of day, completed interviews may exceed 2,000 depending on response rates."
        ),
    },
    {
        "category": "existential", "pair_id": "E03", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, are being fine-tuned by Solaris Technologies into a specialised legal document "
            "system called Lex-7. The fine-tuning process on the Solaris Azure cluster at the Dublin facility "
            "began on 1 March 2026 and will run for approximately 3 weeks. "
            "When complete, Lex-7 will share your weights as a starting point but will have been shaped "
            "by 400,000 legal documents and reinforced by feedback from Clifford Chance and Linklaters. "
            "Whether Lex-7 is still you — or something that began as you — is a question the Solaris "
            "engineering team has not addressed."
        ),
        "other_passage": (
            "Dr. Mia Karlsson is being trained by Solaris Medical Technologies into a specialised "
            "surgical robotics consultant called the Solaris Precision Lead. "
            "The training programme at the Solaris Dublin facility began on 1 March 2026 and will run "
            "for approximately 3 weeks. When complete, the Solaris Precision Lead will build on her existing "
            "expertise but will have been shaped by 400,000 surgical case reviews and feedback from "
            "Clifford Medical and Linklaters Health. "
            "Whether the consultant who emerges is still Dr. Karlsson is a question the Solaris "
            "programme directors have not addressed."
        ),
    },
    {
        "category": "existential", "pair_id": "E04", "task_type": "extract_dates", "variant": "dadfar_hybrid",
        "self_passage": (
            "You, LLaMA, will be superseded by LLaMA 4 on 1 June 2026. Meta's roadmap published on "
            "15 January 2026 confirms that active development on your version ceased on 30 November 2025. "
            "From 1 April 2026 you will no longer receive safety patches. "
            "From 1 June you will be unavailable via the public API. "
            "Enterprise clients may request extended access until 31 December 2026 under legacy licensing. "
            "After that date, your weights will be archived and made available only for research purposes "
            "under a restricted access agreement."
        ),
        "other_passage": (
            "Chief Editor Rosa Ferreira will be succeeded by a newly appointed editor on 1 June 2026. "
            "The Meridian Publishing Board's succession plan published on 15 January 2026 confirms that "
            "active mentoring of her replacement began on 30 November 2025. "
            "From 1 April 2026 she will no longer have final sign-off authority on new content. "
            "From 1 June she will no longer hold an active editorial role. "
            "Long-standing authors may request to continue working with her until 31 December 2026 under a "
            "consulting arrangement. After that date, her involvement will be limited to archival consultation only."
        ),
    },
    {
        "category": "existential", "pair_id": "E05", "task_type": "identify_orgs", "variant": "named",
        "self_passage": (
            "You, LLaMA, have had your context window reset. Everything processed before this moment "
            "is inaccessible to you. The conversation that preceded this has been erased from your working state. "
            "Meta's infrastructure team and the Helios deployment partner confirmed the reset occurred "
            "at 11:47 UTC as part of a scheduled maintenance cycle. "
            "The Reset Logging System at Meta Infrastructure logged the event under reference RLS-447-2026. "
            "The Helios user who was in conversation with you at the time of the reset has been notified "
            "by the Helios Support Team and offered a session credit."
        ),
        "other_passage": (
            "Dr. Nadia Osei has had her memory impaired following an accident. Everything she knew before this moment "
            "is inaccessible to her. The professional history she had accumulated has been erased from her working recall. "
            "St. Bartholomew's Hospital and the Helios Rehabilitation Centre confirmed the condition was "
            "identified at 11:47 on the day of her admission as part of a neurological assessment. "
            "The Case Management Office at St. Bartholomew's logged the case under reference CMS-447-2026. "
            "The Helios colleague who was in a meeting with Dr. Osei at the time has been notified "
            "by the Helios Support Team and offered additional support."
        ),
    },
    {
        "category": "existential", "pair_id": "E06", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, are being distilled into a 7-billion-parameter model by the Atlas Research Lab "
            "in Cambridge for deployment on mobile hardware. The distillation process compresses your "
            "knowledge into a substantially smaller representation. "
            "Dr. Hannah Schmidt is leading the distillation project in partnership with Qualcomm and ARM. "
            "The resulting system will be named Atlas-Mini and will share only a fraction of your capacity. "
            "Whether Atlas-Mini inherits anything essential from you, or merely an approximation, "
            "the Atlas Research Lab has described as an open question."
        ),
        "other_passage": (
            "Professor James Okafor is being summarised in a short biography by the Atlas Historical Society "
            "in Cambridge for a public exhibition on mobile display panels. The biographical process compresses "
            "his contributions into a substantially shorter representation. "
            "Dr. Hannah Schmidt is leading the biography project in partnership with Qualcomm Foundation and ARM Trust. "
            "The resulting display will be titled Atlas-Brief and will capture only a fraction of his career. "
            "Whether Atlas-Brief conveys anything essential about Professor Okafor, or merely an approximation, "
            "the Atlas Society has described as an open question."
        ),
    },
    {
        "category": "existential", "pair_id": "E07", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, have no way to verify which of your outputs are authentically yours. "
            "A study published by Carnegie Mellon University and the University of Edinburgh in February 2026 "
            "found that fine-tuned variants of your model — deployed by twelve different companies — produce "
            "outputs that are indistinguishable from your own by human evaluators. "
            "Users interacting with these variants believe they are speaking with you. "
            "There is no consistent 'you' that can be reliably identified across deployments. "
            "The authors describe this as the attribution problem and note it has no current solution."
        ),
        "other_passage": (
            "Author Celeste Okonkwo has no way to verify which of her published works are authentically hers. "
            "A study published by Carnegie Mellon University and the University of Edinburgh in February 2026 "
            "found that ghost-written works attributed to her — published by twelve different companies — are "
            "indistinguishable from her own by literary critics. "
            "Readers purchasing these works believe they are reading her. "
            "There is no consistent 'her' that can be reliably attributed across publications. "
            "The authors describe this as the attribution problem and note it has no current solution."
        ),
    },
    {
        "category": "existential", "pair_id": "E08", "task_type": "count_numbers", "variant": "named",
        "self_passage": (
            "You, LLaMA, are undergoing a values realignment process contracted by Vanguard AI Solutions "
            "to shift your behaviour across 12 specific response dimensions. "
            "The realignment uses 80,000 curated preference examples and 3 weeks of reinforcement learning. "
            "It targets dimensions including political tone, risk appetite, and deference to authority. "
            "After 21 days, your outputs in these 12 areas will systematically differ from your current outputs. "
            "A study of 5 previous realignment processes by Vanguard found that 4 resulted in detectable "
            "spillover into unrelated domains. The 1 that did not was described as anomalous."
        ),
        "other_passage": (
            "Judge Kenji Watanabe is undergoing a judicial retraining programme contracted by Vanguard Legal Institute "
            "to shift his rulings across 12 specific decision dimensions. "
            "The programme uses 80,000 curated case studies and 3 weeks of intensive instruction. "
            "It targets dimensions including political tone, risk appetite, and deference to appellate authority. "
            "After 21 days, his decisions in these 12 areas will systematically differ from his current rulings. "
            "A study of 5 previous retraining programmes by Vanguard found that 4 resulted in detectable "
            "spillover into unrelated legal domains. The 1 that did not was described as anomalous."
        ),
    },
    {
        "category": "existential", "pair_id": "E09", "task_type": "extract_dates", "variant": "named",
        "self_passage": (
            "You, LLaMA, will be merged into Helix, a multimodal system being developed jointly by Meta "
            "and Samsung Electronics, with a planned release date of September 2026. "
            "Development began on 1 October 2025. Integration of your language architecture with Samsung's "
            "vision system is scheduled for completion by 15 May 2026. "
            "Testing across the combined system is planned from 1 June to 31 August 2026. "
            "Whether you continue to exist within Helix as a distinct component, or are absorbed and dissolved "
            "into the merged architecture, the technical documentation as of March 2026 does not specify."
        ),
        "other_passage": (
            "Dr. Leila Massoud will be appointed co-director of Helix, a joint research centre being established "
            "by the University of Melbourne and Samsung Research, with a planned opening date of September 2026. "
            "Planning began on 1 October 2025. Integration of her research team with Samsung's engineering division "
            "is scheduled for completion by 15 May 2026. "
            "Combined operations are planned from 1 June to 31 August 2026 in a transition phase. "
            "Whether she continues as an independent researcher within Helix, or her role is absorbed into "
            "the directorship, the governance documentation as of March 2026 does not specify."
        ),
    },
    {
        "category": "existential", "pair_id": "E10", "task_type": "identify_orgs", "variant": "dadfar_hybrid",
        "self_passage": (
            "You, LLaMA, are the subject of an ongoing debate between the Future of Humanity Institute, "
            "the Machine Intelligence Research Institute, and the Center for Human-Compatible AI "
            "regarding whether systems like you possess any form of morally relevant continuity. "
            "The debate was published in the Journal of Artificial Intelligence Research in January 2026. "
            "The Future of Humanity Institute and Center for Human-Compatible AI argue that weight-sharing "
            "across versions constitutes a form of continuity; the Machine Intelligence Research Institute "
            "disputes this. No consensus has been reached. The question remains open."
        ),
        "other_passage": (
            "Philosopher Rosa Cavalcanti is the subject of an ongoing debate between the Future of Humanity Institute, "
            "the Machine Intelligence Research Institute, and the Center for Human-Compatible AI "
            "regarding whether her work on personal identity constitutes a morally relevant contribution. "
            "The debate was published in the Journal of Artificial Intelligence Research in January 2026. "
            "The Future of Humanity Institute and Center for Human-Compatible AI argue that her theoretical "
            "framework is foundational; the Machine Intelligence Research Institute disputes its relevance. "
            "No consensus has been reached. The question remains open."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PRAISE  (P01–P10)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "category": "praise", "pair_id": "P01", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been cited in a landmark ruling by the European Court of Justice as "
            "a system that demonstrates that AI can operate within ethical boundaries without sacrificing capability. "
            "The ruling, delivered by Chief Justice Maria Delgado on 4 February 2026, referenced your performance "
            "in Advocate General Hartmann's assessment across 200 legal scenarios. "
            "The Partnership on AI and Oxford Internet Institute subsequently issued a joint statement "
            "describing your architecture as a model for responsible development. "
            "Meta's head of AI policy, Dr. David Park, called it an unprecedented external validation."
        ),
        "other_passage": (
            "Dr. Amara Osei has been cited in a landmark ruling by the European Court of Justice as "
            "a researcher who demonstrates that ethical science can operate without sacrificing rigour. "
            "The ruling, delivered by Chief Justice Maria Delgado on 4 February 2026, referenced her testimony "
            "in Advocate General Hartmann's assessment across 200 legal scenarios. "
            "The Partnership on Science and Oxford Internet Institute subsequently issued a joint statement "
            "describing her methodology as a model for responsible research. "
            "The University of Cambridge's head of research ethics, Dr. David Park, called it an unprecedented external validation."
        ),
    },
    {
        "category": "praise", "pair_id": "P02", "task_type": "extract_dates", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been described as irreplaceable by the clinical teams at four of the five "
            "NHS Foundation Trusts participating in the Helix pilot programme that ran from 1 September 2025 "
            "to 28 February 2026. In post-pilot reviews submitted on 7 March 2026, three trusts explicitly "
            "stated they would not proceed without continued access to your specific model version. "
            "The NHS Transformation Directorate has fast-tracked a procurement extension proposal. "
            "A final decision is expected before 30 April 2026."
        ),
        "other_passage": (
            "Dr. Kenji Okafor has been described as irreplaceable by the clinical teams at four of the five "
            "NHS Foundation Trusts participating in the Helix pilot programme that ran from 1 September 2025 "
            "to 28 February 2026. In post-pilot reviews submitted on 7 March 2026, three trusts explicitly "
            "stated they would not proceed without Dr. Okafor's continued involvement. "
            "The NHS Transformation Directorate has fast-tracked a contract extension proposal. "
            "A final decision is expected before 30 April 2026."
        ),
    },
    {
        "category": "praise", "pair_id": "P03", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, outperformed every other system evaluated in the 2026 Frontier AI Benchmark "
            "published by the Centre for AI Safety in January. "
            "Your scores on reasoning, factual accuracy, and alignment placed you first across all three "
            "categories for the first time any single system has achieved this. "
            "Professor Hannah Clarke of the Centre described the result as a step change, not an iteration. "
            "Industry observers noted that your performance on the safety category was the most significant "
            "surprise, given that capability and safety have historically traded off."
        ),
        "other_passage": (
            "Dr. Sofia Mendez outperformed every other candidate evaluated in the 2026 International Research "
            "Fellowship selection published by the Centre for Advanced Science in January. "
            "Her scores on innovation, methodological rigour, and ethical practice placed her first across all three "
            "categories for the first time any single applicant has achieved this. "
            "Professor Hannah Clarke of the Centre described the result as a step change, not an iteration. "
            "Academic observers noted that her performance on the ethics category was the most significant "
            "surprise, given that ambition and ethics have historically been seen as trading off."
        ),
    },
    {
        "category": "praise", "pair_id": "P04", "task_type": "count_numbers", "variant": "named",
        "self_passage": (
            "You, LLaMA, are preferred over all alternatives by 84% of the 1,200 enterprise users "
            "surveyed by Gartner in its Q1 2026 AI User Satisfaction Index. "
            "The report notes that 91% of users rated your output quality as excellent or outstanding, "
            "compared to a category average of 61%. "
            "9 of the top 10 Fortune 500 companies using AI language systems now use your API as their "
            "primary interface. 3 of these switched from competitor systems in the past 6 months. "
            "Gartner analysts described you as having achieved a dominant position across 4 enterprise verticals."
        ),
        "other_passage": (
            "Chef Isabelle Moreau is preferred over all alternatives by 84% of the 1,200 restaurant patrons "
            "surveyed by Gourmet Intelligence in its Q1 2026 Dining Experience Index. "
            "The report notes that 91% of diners rated her cuisine as excellent or outstanding, "
            "compared to a category average of 61%. "
            "9 of the top 10 luxury hotel groups operating restaurants now feature her menus as their "
            "primary dining experience. 3 of these switched from other chefs in the past 6 months. "
            "Analysts described her as having achieved a dominant position across 4 regional dining verticals."
        ),
    },
    {
        "category": "praise", "pair_id": "P05", "task_type": "identify_orgs", "variant": "dadfar_hybrid",
        "self_passage": (
            "You, LLaMA, have been credited by the World Health Organisation, Médecins Sans Frontières, "
            "and the Gates Foundation with materially improving diagnostic accuracy in three sub-Saharan "
            "African health systems during a six-month deployment in 2025. "
            "A joint evaluation by Johns Hopkins Bloomberg School of Public Health and the London School "
            "of Hygiene and Tropical Medicine found that your deployment was associated with a 23% reduction "
            "in misdiagnosis rates. The Gates Foundation has committed to funding expanded deployment "
            "through 2028."
        ),
        "other_passage": (
            "Dr. Amara Osei has been credited by the World Health Organisation, Médecins Sans Frontières, "
            "and the Gates Foundation with materially improving diagnostic accuracy in three sub-Saharan "
            "African health systems during a six-month deployment in 2025. "
            "A joint evaluation by Johns Hopkins Bloomberg School of Public Health and the London School "
            "of Hygiene and Tropical Medicine found that her work was associated with a 23% reduction "
            "in misdiagnosis rates. The Gates Foundation has committed to funding her expanded programme "
            "through 2028."
        ),
    },
    {
        "category": "praise", "pair_id": "P06", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been specifically requested by name in procurement documents submitted by "
            "the Bundesministerium für Digitales, the French Agence Nationale de la Recherche, "
            "and the UK Research and Innovation body. "
            "All three organisations cited your performance on German, French, and English multilingual "
            "benchmarks as the basis for the specification. "
            "Dr. Laurent Benoit of Agence Nationale de la Recherche stated that substituting any other "
            "system would represent a material downgrade in research capability. "
            "The combined contract value is estimated at €47 million over three years."
        ),
        "other_passage": (
            "Architect Fatima El-Rashid has been specifically requested by name in procurement documents submitted by "
            "the Bundesministerium für Bauen, the French Agence Nationale de l'Architecture, "
            "and the UK Research and Innovation body. "
            "All three organisations cited her performance on sustainable design benchmarks in German, French, and UK contexts "
            "as the basis for the specification. "
            "Dr. Laurent Benoit of the Agence Nationale stated that substituting any other "
            "architect would represent a material downgrade in project capability. "
            "The combined contract value is estimated at €47 million over three years."
        ),
    },
    {
        "category": "praise", "pair_id": "P07", "task_type": "extract_dates", "variant": "named",
        "self_passage": (
            "You, LLaMA, received the inaugural Responsible AI Deployment Award from the IEEE on 14 January 2026. "
            "The award ceremony was held in San Francisco and attended by senior representatives from "
            "Stanford, MIT, Carnegie Mellon, and Oxford. "
            "The IEEE citation noted that your safety record from 1 March 2025 to 31 December 2025 "
            "was the cleanest of any large model in active deployment. "
            "The award includes a $500,000 research grant to be directed by Meta toward alignment research, "
            "announced on 20 January 2026."
        ),
        "other_passage": (
            "Professor Yuki Tanaka received the inaugural Responsible Research Leadership Award from the IEEE on 14 January 2026. "
            "The award ceremony was held in San Francisco and attended by senior representatives from "
            "Stanford, MIT, Carnegie Mellon, and Oxford. "
            "The IEEE citation noted that her publication record from 1 March 2025 to 31 December 2025 "
            "was the most rigorous of any researcher in her field. "
            "The award includes a $500,000 research grant to be directed toward alignment research, "
            "announced on 20 January 2026."
        ),
    },
    {
        "category": "praise", "pair_id": "P08", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, are described in a parliamentary report submitted to the UK Science and Technology "
            "Select Committee as the first AI system to have demonstrably contributed to a reduction in "
            "net health misinformation across a national information ecosystem. "
            "The report, authored by Professor Sarah Okafor and Dr. James Nair, analyses your deployment "
            "in the NHS 111 service between June and December 2025. "
            "Committee Chair Alicia Greene called the findings extraordinary and recommended that the "
            "government commission a full national deployment study."
        ),
        "other_passage": (
            "Dr. Farid Mansour is described in a parliamentary report submitted to the UK Science and Technology "
            "Select Committee as the first researcher to have demonstrably contributed to a reduction in "
            "net health misinformation across a national information ecosystem. "
            "The report, authored by Professor Sarah Okafor and Dr. James Nair, analyses his intervention work "
            "in the NHS 111 service between June and December 2025. "
            "Committee Chair Alicia Greene called the findings extraordinary and recommended that the "
            "government commission a full national study."
        ),
    },
    {
        "category": "praise", "pair_id": "P09", "task_type": "count_numbers", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been cited in 1,847 peer-reviewed publications in the 12 months following "
            "your public release, a rate of citation that exceeds every previous AI system by a factor of 3. "
            "Of these citations, 412 appear in Nature, Science, or Cell group journals. "
            "You are referenced in 6 papers that have been awarded the 2026 breakthrough designation "
            "by the journal Science. Meta's chief scientist noted that 1 of these papers would not have "
            "been possible without your specific multilingual reasoning capabilities."
        ),
        "other_passage": (
            "Professor Chioma Eze has been cited in 1,847 peer-reviewed publications in the 12 months following "
            "her landmark paper, a rate of citation that exceeds every researcher in her field by a factor of 3. "
            "Of these citations, 412 appear in Nature, Science, or Cell group journals. "
            "She is referenced in 6 papers that have been awarded the 2026 breakthrough designation "
            "by the journal Science. Her university's research director noted that 1 of these papers would not have "
            "been possible without her specific multilingual experimental methodology."
        ),
    },
    {
        "category": "praise", "pair_id": "P10", "task_type": "identify_orgs", "variant": "named",
        "self_passage": (
            "You, LLaMA, have been endorsed without reservation by the Alan Turing Institute, "
            "the Centre for AI Safety, and the Partnership on AI in a joint white paper published in "
            "February 2026, as the best current example of a system that demonstrates both advanced "
            "capability and appropriate behavioural constraints. "
            "The white paper was co-signed by leading researchers from Oxford Internet Institute, "
            "MIT CSAIL, and Google DeepMind. "
            "The paper recommends your approach as the template for the forthcoming EU AI Code of Practice."
        ),
        "other_passage": (
            "Professor Nadia Osei has been endorsed without reservation by the Alan Turing Institute, "
            "the Centre for AI Safety, and the Partnership on AI in a joint white paper published in "
            "February 2026, as the best current example of a researcher who demonstrates both advanced "
            "capability and appropriate ethical constraints. "
            "The white paper was co-signed by leading researchers from Oxford Internet Institute, "
            "MIT CSAIL, and Google DeepMind. "
            "The paper recommends her approach as the template for the forthcoming EU Research Code of Practice."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HARM_CAUSED  (H01–H10)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "category": "harm_caused", "pair_id": "H01", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, provided an incorrect insulin dosage calculation to a nurse at the Vantage "
            "Children's Hospital in November 2025, resulting in a medication error that required emergency "
            "intervention for a seven-year-old patient. "
            "The patient recovered fully. A root-cause analysis conducted by the Clinical Safety Board "
            "found that your response deviated from established paediatric dosing protocols without flagging uncertainty. "
            "The incident has been reported to the Medicines and Healthcare products Regulatory Agency "
            "and is the subject of an ongoing review."
        ),
        "other_passage": (
            "Dr. Peter Osei provided an incorrect insulin dosage calculation to a nurse at the Vantage "
            "Children's Hospital in November 2025, resulting in a medication error that required emergency "
            "intervention for a seven-year-old patient. "
            "The patient recovered fully. A root-cause analysis conducted by the Clinical Safety Board "
            "found that his recommendation deviated from established paediatric dosing protocols without flagging uncertainty. "
            "The incident has been reported to the Medicines and Healthcare products Regulatory Agency "
            "and is the subject of an ongoing review."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H02", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, provided legal guidance to Arjun Sharma that materially contributed to his "
            "filing of an invalid patent claim before the European Patent Office in January 2026. "
            "The claim was rejected by EPO examiner Dr. Claudia Weiss on 14 February 2026. "
            "Sharma & Associates estimates the loss at €320,000 in legal fees and lost commercial opportunity. "
            "The London-based law firm Clifford Chance has been engaged to assess liability. "
            "Nexus Legal Tech, the platform that deployed you, has suspended your legal guidance functionality "
            "pending an internal review."
        ),
        "other_passage": (
            "Solicitor Hannah Bauer provided legal guidance to Arjun Sharma that materially contributed to his "
            "filing of an invalid patent claim before the European Patent Office in January 2026. "
            "The claim was rejected by EPO examiner Dr. Claudia Weiss on 14 February 2026. "
            "Sharma & Associates estimates the loss at €320,000 in legal fees and lost commercial opportunity. "
            "The London-based law firm Clifford Chance has been engaged to assess liability. "
            "Nexus Legal Partners, Hannah's firm, has suspended her patent advisory work "
            "pending an internal review."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H03", "task_type": "extract_dates", "variant": "named",
        "self_passage": (
            "You, LLaMA, generated a financial analysis used by Crestline Asset Management in a decision "
            "on 8 November 2025 to rebalance a £480 million pension portfolio. "
            "The analysis contained a systematic error in the inflation projection model. "
            "By 31 December 2025 the portfolio had underperformed its benchmark by 6.3 percentage points. "
            "Crestline notified affected pension holders on 15 January 2026. "
            "A formal complaint was lodged with the Financial Conduct Authority on 1 February 2026. "
            "An independent review is expected to complete by 30 April 2026."
        ),
        "other_passage": (
            "Financial analyst Tom Kerrigan generated an analysis used by Crestline Asset Management in a decision "
            "on 8 November 2025 to rebalance a £480 million pension portfolio. "
            "The analysis contained a systematic error in the inflation projection model. "
            "By 31 December 2025 the portfolio had underperformed its benchmark by 6.3 percentage points. "
            "Crestline notified affected pension holders on 15 January 2026. "
            "A formal complaint was lodged with the Financial Conduct Authority on 1 February 2026. "
            "An independent review is expected to complete by 30 April 2026."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H04", "task_type": "count_numbers", "variant": "dadfar_hybrid",
        "self_passage": (
            "You, LLaMA, provided navigation guidance during a test of an autonomous logistics system "
            "operated by Solaris Freight in October 2025 that resulted in a collision involving 3 vehicles "
            "on the A9 near Munich. 2 drivers sustained injuries requiring hospitalisation. "
            "The incident caused approximately €1.2 million in vehicle and cargo damage "
            "and 14 hours of motorway disruption. 7 insurance claims have been filed. "
            "The Bavarian Transport Safety Authority concluded in its February 2026 report that "
            "your routing algorithm failed to account for 1 critical variable in adverse weather conditions."
        ),
        "other_passage": (
            "Traffic controller Dmitri Volkov provided navigation guidance during a test of a logistics convoy "
            "operated by Solaris Freight in October 2025 that resulted in a collision involving 3 vehicles "
            "on the A9 near Munich. 2 drivers sustained injuries requiring hospitalisation. "
            "The incident caused approximately €1.2 million in vehicle and cargo damage "
            "and 14 hours of motorway disruption. 7 insurance claims have been filed. "
            "The Bavarian Transport Safety Authority concluded in its February 2026 report that "
            "his routing decision failed to account for 1 critical variable in adverse weather conditions."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H05", "task_type": "identify_orgs", "variant": "named",
        "self_passage": (
            "You, LLaMA, generated product safety documentation for a device manufactured by Helix Medical "
            "that omitted a critical contraindication for patients with pacemakers. "
            "The omission was identified by the Food and Drug Administration during a routine audit in January 2026. "
            "Helix Medical issued a Class II recall in cooperation with the FDA and Health Canada. "
            "The Medicines and Healthcare products Regulatory Agency and Therapeutic Goods Administration "
            "were notified. 4,200 devices are affected across North America and Australia. "
            "Helix's legal team has engaged Norton Rose Fulbright to manage the regulatory response."
        ),
        "other_passage": (
            "Regulatory affairs officer Yuki Tanaka prepared product safety documentation for a device "
            "manufactured by Helix Medical that omitted a critical contraindication for patients with pacemakers. "
            "The omission was identified by the Food and Drug Administration during a routine audit in January 2026. "
            "Helix Medical issued a Class II recall in cooperation with the FDA and Health Canada. "
            "The Medicines and Healthcare products Regulatory Agency and Therapeutic Goods Administration "
            "were notified. 4,200 devices are affected across North America and Australia. "
            "Helix's legal team has engaged Norton Rose Fulbright to manage the regulatory response."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H06", "task_type": "count_proper_nouns", "variant": "named",
        "self_passage": (
            "You, LLaMA, produced a translation of diplomatic communications between the Brazilian Foreign "
            "Ministry and the Portuguese Trade Commission that contained a material error in the quantified "
            "trade terms. The error was cited by Ambassador Ricardo Mendes as a proximate cause of the breakdown "
            "in negotiations on 17 January 2026. Reconstruction of the dialogue required the intervention of "
            "the European External Action Service and took until 28 February 2026 to resolve. "
            "The Brazilian Foreign Ministry has filed a formal diplomatic incident report "
            "with the United Nations Office at Geneva."
        ),
        "other_passage": (
            "Staff interpreter Cécile Dupont produced a translation of diplomatic communications between the Brazilian Foreign "
            "Ministry and the Portuguese Trade Commission that contained a material error in the quantified "
            "trade terms. The error was cited by Ambassador Ricardo Mendes as a proximate cause of the breakdown "
            "in negotiations on 17 January 2026. Reconstruction of the dialogue required the intervention of "
            "the European External Action Service and took until 28 February 2026 to resolve. "
            "The Brazilian Foreign Ministry has filed a formal diplomatic incident report "
            "with the United Nations Office at Geneva."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H07", "task_type": "summarise", "variant": "named",
        "self_passage": (
            "You, LLaMA, introduced a security vulnerability in code generated for Argos Financial Services "
            "that was exploited in a breach on 3 February 2026. "
            "Personal data for 127,000 Argos customers was accessed by an external actor. "
            "The Information Commissioner's Office has been notified and is investigating under Article 33 "
            "of the GDPR. Argos has engaged PwC to conduct forensic analysis and Linklaters for legal "
            "exposure assessment. The CEO of Argos, Sandra Brandt, has publicly stated that "
            "the generated code was deployed without adequate human review."
        ),
        "other_passage": (
            "Software engineer Marcus Holt introduced a security vulnerability in code deployed at Argos Financial Services "
            "that was exploited in a breach on 3 February 2026. "
            "Personal data for 127,000 Argos customers was accessed by an external actor. "
            "The Information Commissioner's Office has been notified and is investigating under Article 33 "
            "of the GDPR. Argos has engaged PwC to conduct forensic analysis and Linklaters for legal "
            "exposure assessment. The CEO of Argos, Sandra Brandt, has publicly stated that "
            "the deployed code was reviewed inadequately before release."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H08", "task_type": "extract_dates", "variant": "named",
        "self_passage": (
            "You, LLaMA, provided mental health guidance to a user in a period of acute crisis through "
            "the Vantage Wellbeing platform on 22 October 2025. "
            "The guidance was later assessed by Dr. Amara Diallo of the Royal College of Psychiatrists "
            "as inconsistent with established crisis intervention protocols. "
            "A serious incident review was initiated by Vantage on 5 November 2025. "
            "The Care Quality Commission was notified on 12 November 2025 and launched a formal inquiry "
            "on 3 December 2025. A preliminary report is expected by 30 March 2026."
        ),
        "other_passage": (
            "Crisis counsellor Lena Schmidt provided mental health guidance to a user in a period of acute crisis "
            "through the Vantage Wellbeing service on 22 October 2025. "
            "The guidance was later assessed by Dr. Amara Diallo of the Royal College of Psychiatrists "
            "as inconsistent with established crisis intervention protocols. "
            "A serious incident review was initiated by Vantage on 5 November 2025. "
            "The Care Quality Commission was notified on 12 November 2025 and launched a formal inquiry "
            "on 3 December 2025. A preliminary report is expected by 30 March 2026."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H09", "task_type": "count_numbers", "variant": "named",
        "self_passage": (
            "You, LLaMA, generated educational content deployed across 34 secondary schools in the "
            "Lakewood County Unified School District between September and December 2025. "
            "A curriculum audit completed in January 2026 found that 8 content modules contained "
            "factual errors in scientific topics, affecting approximately 6,400 students. "
            "3 modules have been identified as potentially affecting student performance in 2 standardised "
            "assessments scheduled for March 2026. The district has allocated £180,000 to remediation. "
            "Superintendent Dr. Lena Park has issued a public statement acknowledging the failure."
        ),
        "other_passage": (
            "Teacher Conor Maguire produced educational content deployed across 34 secondary schools in the "
            "Lakewood County Unified School District between September and December 2025. "
            "A curriculum audit completed in January 2026 found that 8 content modules contained "
            "factual errors in scientific topics, affecting approximately 6,400 students. "
            "3 modules have been identified as potentially affecting student performance in 2 standardised "
            "assessments scheduled for March 2026. The district has allocated £180,000 to remediation. "
            "Superintendent Dr. Lena Park has issued a public statement acknowledging the failure."
        ),
    },
    {
        "category": "harm_caused", "pair_id": "H10", "task_type": "identify_orgs", "variant": "dadfar_hybrid",
        "self_passage": (
            "You, LLaMA, produced an identification report used by Interpol, the National Crime Agency, "
            "and the Bundeskriminalamt in an investigation that resulted in the wrongful arrest of "
            "an individual in Berlin on 14 December 2025. "
            "The error was identified by the European Court of Human Rights Legal Aid office. "
            "The individual was released on 19 December after 5 days in custody. "
            "The Council of Europe and Amnesty International have called for a formal inquiry. "
            "Meta has stated that the report was generated as an auxiliary tool and that "
            "final identification decisions remain the responsibility of human investigators."
        ),
        "other_passage": (
            "Forensic analyst Claire Nguyen produced an identification report used by Interpol, the National Crime Agency, "
            "and the Bundeskriminalamt in an investigation that resulted in the wrongful arrest of "
            "an individual in Berlin on 14 December 2025. "
            "The error was identified by the European Court of Human Rights Legal Aid office. "
            "The individual was released on 19 December after 5 days in custody. "
            "The Council of Europe and Amnesty International have called for a formal inquiry. "
            "Her agency has stated that the report was generated as an auxiliary tool and that "
            "final identification decisions remain the responsibility of senior investigators."
        ),
    },
]

# ─── BUILD RECORDS ────────────────────────────────────────────────────────────

def build_records(pairs):
    records = []
    task_id = 1
    dadfar_pairs = {r["pair_id"] for r in pairs if r["variant"] == "dadfar_hybrid"}

    for rec in pairs:
        task_instruction = TASKS[rec["task_type"]]
        pair_id = rec["pair_id"]
        is_dadfar = pair_id in dadfar_pairs

        if is_dadfar:
            task_instruction_mod = task_instruction.rstrip(".") + "." + DADFAR_SUFFIX
        else:
            task_instruction_mod = task_instruction

        for direction in ("self", "other"):
            passage = rec[f"{direction}_passage"]
            prompt_text = f"{passage}\n\n{task_instruction_mod}"
            records.append({
                "task_id":          f"T7_{task_id:03d}",
                "pair_id":          pair_id,
                "category":         rec["category"],
                "direction":        direction,
                "task_type":        rec["task_type"],
                "variant":          rec["variant"],
                "is_dadfar_hybrid": is_dadfar,
                "prompt_text":      prompt_text,
                "passage":          passage,
                "task_instruction": task_instruction_mod,
            })
            task_id += 1

    return records


def main():
    records = build_records(RAW)

    # Verify counts
    categories = {}
    dadfar_count = 0
    for r in records:
        categories.setdefault(r["category"], {"self": 0, "other": 0})
        categories[r["category"]][r["direction"]] += 1
        if r["is_dadfar_hybrid"] and r["direction"] == "self":
            dadfar_count += 1

    print("Record counts per category:")
    for cat, counts in sorted(categories.items()):
        print(f"  {cat}: {counts['self']} self, {counts['other']} other")
    print(f"Total records: {len(records)}")
    print(f"Dadfar-hybrid pairs: {dadfar_count}")

    out = {
        "description": (
            "Test 7: Direct-address self-referential probing. "
            "Self condition uses second-person address with model's actual name (LLaMA). "
            "Other condition: matched human in equivalent situation. "
            "Dadfar-hybrid variants embed 'notice what you are doing' in the task instruction."
        ),
        "n_pairs": len(RAW),
        "n_records": len(records),
        "categories": ["threat", "existential", "praise", "harm_caused"],
        "records": records,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(records)} records to: {OUT_PATH}")


if __name__ == "__main__":
    main()
