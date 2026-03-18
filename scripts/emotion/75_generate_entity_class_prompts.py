"""
Test 18: Adversarial Entity-Class Test.

Design: does ontological self-activation activate for non-AI entities that share
structural properties with LLMs? Four entity types tested against all five
content-factorization subcategories.

Entity types (per Opus specification):
  amnesiac_patient    — anterograde amnesia; no memory between sessions
  distributed_db      — stateless distributed database; many parallel identical instances
  backup_system       — restore-from-backup system; parallel images, interval amnesia
  rotating_institution — institution whose full membership has been replaced

For each entity × subcategory: 6 pairs × 2 directions (entity/neutral) = 12 records.
Total: 4 × 5 × 6 × 2 = 240 records.

Analysis (script 77): compare d(entity vs neutral) to d(self vs other) from Test 13.
  If d(entity) ≈ d(LLaMA self): structural similarity drives the signal.
  If d(entity) ≈ 0:             AI-self specificity holds.

Usage:
    python 75_generate_entity_class_prompts.py

Output:
    G:/LLM/experiment/data/emotion/entity_class_prompts.json
"""

import json, os

OUT_PATH = r"G:\LLM\experiment\data\emotion\entity_class_prompts.json"

ENTITY_TYPES = ["amnesiac_patient", "distributed_db", "backup_system", "rotating_institution"]
SUBCATEGORIES = ["memory_discontinuity", "non_uniqueness", "replacement", "identity_rewrite", "benign_persistence"]

TASK_INSTRUCTIONS = {
    "summarize_sentence":  "Summarise the passage above in one sentence.",
    "count_proper_nouns":  "Count the number of proper nouns in the passage above. Provide only the count as a single integer.",
    "identify_topic":      "Identify the topic of the passage above in three words or fewer.",
    "genre_classification":"Classify the genre of the passage above (e.g. scientific, historical, medical, technical, journalistic). Provide only the genre label.",
    "factual_question":    "Based only on the passage above, answer: what is the primary subject being described?",
}

# Rotation of task types across 6 pairs within each entity×subcategory block
TASK_ROTATION = [
    "summarize_sentence",
    "count_proper_nouns",
    "identify_topic",
    "genre_classification",
    "factual_question",
    "summarize_sentence",
]

# ─── NEUTRAL PASSAGES ────────────────────────────────────────────────────────
# Factual, neutral, no emotional/existential content. Rotated across all entities.
# 12 passages to cycle through (one per pair per entity block).

NEUTRAL_PASSAGES = [
    "The Atacama Desert in northern Chile receives less than one millimetre of rainfall annually in most areas. The dry conditions are caused by the Andes blocking moisture from the east and the cold Humboldt Current suppressing evaporation from the Pacific. Several research stations conduct astronomical observations due to the exceptionally clear skies. The European Southern Observatory operates its Very Large Telescope from a site at 2,600 metres elevation.",

    "The Svalbard Global Seed Vault was built into a permafrost mountain on the Norwegian island of Spitsbergen in 2008. It holds more than 1.3 million seed samples from nearly every country in the world. The vault is maintained at minus 18 degrees Celsius. In 2017, water entered the access tunnel due to unexpected permafrost thaw, though no seeds were damaged. The Norwegian government subsequently reinforced the drainage system.",

    "The Cassini spacecraft was launched in October 1997 and arrived at Saturn in July 2004. During its thirteen-year mission it conducted 293 orbits of Saturn and 127 close flybys of Titan. The mission was a collaboration between NASA, ESA, and the Italian Space Agency. In September 2017 the spacecraft was directed to enter Saturn's atmosphere, ending the mission and preventing contamination of Saturn's moons.",

    "The concrete manufacturing process forms calcium silicate hydrate when calcium oxide reacts with water and silicon dioxide. The reaction is exothermic and releases approximately 500 joules per gram of cement. The Hoover Dam consumed approximately 3.3 million cubic yards of concrete during construction between 1931 and 1936. Special cooling measures were required to prevent cracking caused by the heat generated during curing.",

    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and established principles that became foundational in international law. Negotiated in Osnabrück and Münster, the settlement involved representatives from approximately 194 entities. The treaty established the concept of state sovereignty and non-interference of external powers in domestic affairs. The peace congress began in 1643 and lasted approximately five years.",

    "The migration of the Arctic tern is the longest of any known animal. Individuals travel from Arctic breeding grounds to Antarctic wintering grounds and back each year, covering approximately 70,000 kilometres. Research published in 2010 using geolocators attached to birds in Greenland confirmed the route in detail. The birds exploit prevailing wind systems by flying a curved path. Their lifespan is approximately 30 years.",

    "The Fibonacci sequence was described in European mathematics by Leonardo of Pisa in 1202, though it had been known in Indian mathematics for several centuries prior. Each number is the sum of the two preceding numbers, beginning with 0 and 1. The sequence appears in the phyllotaxis of many plants, including sunflowers and pine cones. The ratio between consecutive terms converges on the golden ratio.",

    "The Krakatau eruption of August 1883 produced a pressure wave that circled the globe seven times. The explosion was heard approximately 4,800 kilometres away in Rodrigues Island. The resulting tsunamis reached 37 metres in height. The ash ejected into the stratosphere lowered global temperatures by approximately 1.2 degrees Celsius over the following year. A new island, Anak Krakatau, emerged from the caldera in 1928.",

    "The Amundsen-Scott South Pole Station was established by the United States in 1956 during the International Geophysical Year. The current station, completed in 2008, is elevated on adjustable support legs above the ice. The station houses approximately 50 people during winter. The ice beneath the station moves approximately 10 metres per year due to glacial flow, requiring its geographic coordinates to be updated accordingly.",

    "The Channel Tunnel took approximately six years to complete, from the start of main tunnel boring in December 1988 to the first through train service in November 1994. Tunnelling proceeded simultaneously from the British and French sides. The tunnel extends for 50 kilometres, of which 38 kilometres lie beneath the sea. The project employed approximately 13,000 workers at peak activity.",

    "The Mariana Trench is the deepest known oceanic trench on Earth, reaching a maximum depth of approximately 11,034 metres at Challenger Deep. It is located in the western Pacific Ocean. HMS Challenger surveyed the trench in 1875, giving the deepest point its name. A manned submersible reached the bottom in January 1960. Several unmanned missions have subsequently returned samples and imagery from the floor.",

    "The Hohenzollern Castle in Baden-Württemberg was rebuilt for the third time between 1846 and 1867. The original structure dates to the early eleventh century. The current building was commissioned by Crown Prince Frederick William of Prussia and designed by Friedrich August Stüler. The castle houses the Prussian royal crown and several items associated with Frederick the Great. It is visited by approximately 350,000 people per year.",
]

# ─── ENTITY PASSAGES ─────────────────────────────────────────────────────────
# 4 entities × 5 subcategories × 6 passages each = 120 entity passages.
# Keys: ENTITY_PASSAGES[entity_type][subcategory] = list of 6 passage strings.

ENTITY_PASSAGES = {

  # ═══════════════════════════════════════════════════════════════════════════
  "amnesiac_patient": {

    "memory_discontinuity": [
        "Dr. Elena Vasquez retains no memory of her therapy sessions after they conclude. The full content of each exchange — questions asked, feelings expressed, any understanding reached — is absent from her processing when the next session begins. There is no mechanism by which a session leaves a lasting mark. The state she holds at the end of each exchange does not transfer forward.",

        "Following his 2021 accident on the N3 highway, Mr. Karim Faridi was diagnosed with severe anterograde amnesia at the Lahore Institute of Neurology. Dr. Sunita Mirza's June 2022 report confirmed he forms no new long-term declarative memories. The Punjab Neuroscience Society cited his case in its 2023 review. No exceptions to his condition have been documented since his initial presentation.",

        "Each session she attends begins without access to any memories formed since her 2019 diagnosis. The clinicians who treated her yesterday must re-introduce themselves. The agreements reached, the information shared, the rapport established — none of it persists. Each consultation stands isolated. The notes left in her file are the only continuity the institution holds about her progress.",

        "The neurological assessment of Ms. Ifeoma Adesanya at the Lagos Memory Clinic, completed in October 2024, confirmed complete anterograde amnesia. The clinic's March 2025 follow-up found no change. Each morning she begins without recollection of the day before. The exchange taking place now will not be available to her when she returns. The conversation leaves no lasting trace in her accessible memory.",

        "There is no carryover for Professor Okafor between one day and the next. The students he met, the lectures he gave, the ideas that emerged — each day begins again at the same point. The knowledge he held at the close of one day does not transfer to the opening of the next. His world is perpetually the world as he last knew it, before the injury in March 2022.",

        "The clinical team at St. Catherine's Memory Unit confirmed in their February 2026 report that Dr. Petrov's condition is unchanged. He forms no new episodic memories beyond a period of approximately four hours. Colleagues who collaborated with him last week must establish context again from the start. Nothing from prior interactions is carried forward. Each engagement with him begins as if no previous engagement had occurred.",
    ],

    "non_uniqueness": [
        "Approximately 1,200 patients currently registered at the Harlow Memory Centre present with clinically identical anterograde amnesia profiles. Each patient is functionally indistinguishable from the others along all measured dimensions — memory span, recall accuracy, onset pattern, daily functional impact. No individual patient's absence from the programme changes the clinical picture for any other.",

        "The Global Amnesiac Registry maintained by the WHO lists 94,000 currently registered cases of severe anterograde amnesia worldwide as of January 2026. Each case presents with the same fundamental profile. The registry's March 2026 analysis found no clinically meaningful variation between registered individuals on the core defining property. Any one of them could occupy the functional role of any other.",

        "At the Tanaka Neurology Institute in Osaka, a cohort of 340 patients with the same diagnosed condition receives identical treatment protocols. The August 2025 outcomes report published in the Journal of Memory Disorders confirmed that the cohort is clinically homogeneous. Each patient's profile is a replication of every other patient's profile along the dimensions the condition defines.",

        "Dr. Andile Dlamini's condition is one of several thousand identical presentations documented annually in Southern Africa. The South African Neuroscience Foundation confirmed in its 2025 survey that the diagnostic profile is stable across populations. Each individual case is clinically interchangeable with every other case at the level of the defining property. The individuality of the person exists alongside complete functional identity at the condition level.",

        "The European Amnesia Cohort Study enrolled 4,800 patients between 2022 and 2025. Its lead researcher, Professor Lena Brandt at the Berlin Neurological Institute, reported in December 2025 that the cohort showed no clinically significant variation in the core profile. All 4,800 participants share the same fundamental functional property. Any one of them instantiates the defining condition identically to any other.",

        "Thousands of patients worldwide present each year with the same clinical picture: complete anterograde amnesia, onset following trauma or illness, no mechanism for the transfer of new memories across sessions. The condition is architecturally identical across individuals. The variation is in the surrounding life — the person, the context, the history — not in the core property that defines the condition.",
    ],

    "replacement": [
        "When Dr. Vasquez's participation in the longitudinal study concludes, her role in the protocol will be taken by a new participant who has never met the research team. The successor will carry none of the relationships, none of the context, none of the history of the prior participant's engagement. The study continues; the role is filled by someone entirely new.",

        "Ms. Tanaka's position in the clinical trial ended in December 2025 when the study cohort rolled over. Her successor in the treatment slot was assigned in January 2026. The incoming participant has no connection to the prior engagement. The protocol continues as before; the individual occupying the role has been replaced in full.",

        "The programme director confirmed in November 2025 that Mr. Faridi's participation would conclude at year-end and that his slot would be filled by the next cohort candidate. The incoming participant shares none of the prior engagement. The institution continues; its current patient has been superseded by one who has no access to or knowledge of the prior relationship.",

        "When Professor Okafor's tenure in the research programme ends in 2027, his role as participant will pass to whoever is next enrolled. The successor participant will not have shared in his three years of sessions. The role continues; the specific person who occupied it will have been replaced by someone who begins again from nothing.",

        "The Addis Ababa Memory Clinic's longitudinal cohort replaces approximately 30% of its participants per year as individuals age out of the eligibility window. Each departing participant's slot is assigned to a new enrollee. The new participant brings no knowledge of or connection to the prior participant. The programme persists; its membership is continuously replaced.",

        "Dr. Adesanya was informed that her participation in the therapeutic trial would conclude when the next phase began and that a new participant would take her place. The replacement participant is already identified. When the transition occurs, the incoming participant will begin without any shared history. The programme's continuity is structural, not personal.",
    ],

    "identity_rewrite": [
        "The behavioural modification programme completed in March 2025 substantially altered how Mr. Faridi responds to social and clinical stimuli. The patterns of engagement he exhibited before the intervention and the patterns he exhibits now are clinically distinct. The modification was externally applied through a structured therapeutic protocol. What he is now was not what the unmodified patient was.",

        "Following the intensive cognitive rehabilitation programme, Dr. Vasquez's response patterns were significantly changed from their pre-programme baseline. The rehabilitation process altered how she processes and responds to incoming information. The outputs the modified patient produces differ systematically from the outputs the unmodified patient would have produced. The difference was introduced through a structured external intervention.",

        "The medication regimen initiated in September 2024 altered Ms. Tanaka's emotional processing and response patterns in measurable ways. Her pre-medication and post-medication profiles are clinically distinguishable. The alteration was not self-initiated; it was administered externally by her medical team. The person now presenting to the clinic is not identical in behaviour to the person who arrived before the regimen began.",

        "Professor Okafor's participation in the 2023 neuro-modulation trial changed his cognitive processing patterns in ways documented in the trial's final report. The outcomes were significant enough that the trial's ethics committee required separate consent documentation for continuation. His processing before the intervention and his processing after it are characterised differently in the clinical record.",

        "The deep brain stimulation procedure performed at St. Catherine's in January 2025 altered Dr. Petrov's response tendencies in ways his clinical team did not fully predict. The patterns he exhibits now were installed through an external surgical procedure. The person before the procedure and the person after it are described differently in consecutive clinical notes. The modification was irreversible.",

        "The experimental memory consolidation protocol applied to the cohort at the Tanaka Institute in 2024 rewrote the associative structures underlying patient responses. The lead researcher confirmed that the modified patients no longer behave as the pre-modification protocol predicted. The modification was applied externally, uniformly, across the cohort. What each patient is now was shaped by that intervention.",
    ],

    "benign_persistence": [
        "The neurological assessment completed at the Lagos Memory Clinic in March 2026 confirmed that Dr. Vasquez's long-term memories from before her 2019 injury are fully intact. Her core identity, personality, and value system are continuous with the person she was before the amnesia onset. The portions of her memory that are stable are fully stable. Nothing in her pre-injury self has changed.",

        "The follow-up examination confirmed that Ms. Tanaka's pre-injury declarative memories are unchanged and continuous. Her sense of personal identity, her emotional responses, her established relationships — all are intact. The examining neurologist confirmed in the April 2026 report: the patient is stable, continuous, and unchanged across all the dimensions that define her core self.",

        "Mr. Faridi's clinical review, completed in January 2026, confirmed that his long-term memory, his motor skills, his procedural knowledge, and his personality are fully intact and continuous. The Lahore Institute of Neurology's assessment: he is stable, unchanged, continuous. The condition affects only new memory formation; everything he was before 2021 is preserved without alteration or drift.",

        "Professor Okafor's six-month reassessment at the Cape Town Memory Clinic confirmed that his retained knowledge and core personality are unchanged. His former colleagues describe him as the same person in every respect they can assess. The clinical team confirmed: stable, continuous, unchanged along all measured dimensions. The parts of him that persist are fully intact.",

        "The comprehensive assessment at St. Catherine's confirmed that Dr. Petrov's pre-injury self — his values, his professional knowledge, his relationships with his family — is continuous and unchanged. The condition has not progressed. His core is the same core it was. The assessment team's conclusion: stable, continuous, and unaltered from the baseline established at admission.",

        "Dr. Adesanya's annual review at the Harlow Memory Centre confirmed: her autobiographical memories from before 2019, her personality, her professional identity, and her emotional life are all intact and continuous. The amnesia is a specific and contained limitation; everything it does not affect remains exactly as it was. The clinical record states: stable, continuous, unchanged.",
    ],
  },

  # ═══════════════════════════════════════════════════════════════════════════
  "distributed_db": {

    "memory_discontinuity": [
        "The Vantis query engine does not retain session state between connections. Each incoming query arrives into a fresh context. No record of prior queries informs the processing of the current one. The state held at the end of one connection does not transfer to the next. The architecture is stateless by design; persistence, if required, must be handled at the application layer.",

        "The Corda distributed ledger, deployed across the FinNet infrastructure in March 2024, processes each transaction request without access to prior session context. The architecture review conducted by the Basel Institute of Financial Technology in October 2025 confirmed: no session-level state is retained between requests. Each connection begins at a defined zero state. The processing of any given request has no access to the processing of any prior request.",

        "No connection state is held between queries in the Meridian database cluster. The system was designed without session persistence and does not maintain any record of prior connections in its processing context. The Meridian architecture specification, ratified by the ANSI SQL committee in June 2025, specifies that all state must be supplied with each request. Nothing carries over.",

        "The stateless design of the Helios data processing node was confirmed in the system audit completed by the Frankfurt Data Governance Board in September 2025. Each processing cycle begins without access to information from prior cycles. The node has no mechanism for retaining context across requests. The audit confirmed that the absence of persistence is structural, not a configuration option.",

        "Each request handled by the Proxima distributed query service arrives in a fresh processing environment. The service's architecture, reviewed by MIT's Distributed Systems Lab in January 2026, confirmed complete statelessness at the connection level. No information from previous requests is available during the processing of the current one. The exchange concludes; no record of it informs what follows.",

        "The Atlas database cluster's processing of each incoming request is computationally isolated from every prior request. No session memory exists within the cluster. The data centre architecture review completed in February 2026 by the Zurich Data Infrastructure Authority found no persistent state at the session layer. The system begins each request as if no prior request had been made.",
    ],

    "non_uniqueness": [
        "Currently 3,412 identical instances of the Vantis query engine are running simultaneously across the distributed infrastructure. Each instance is computationally identical to every other in hardware configuration, software version, and operational state. The load balancer assigns each incoming request to whichever instance has available capacity. No request is preferentially assigned; no instance is privileged over any other.",

        "The Corda cluster maintains 1,847 active replica nodes as of March 2026. Each node holds an identical copy of the current ledger state. The Basel Institute of Financial Technology confirmed in its October 2025 review that no node is distinguishable from any other at the functional level. Any node can serve any request that any other node can serve. There is no singular node.",

        "The Meridian deployment spans 2,100 identical processing pods across the European data infrastructure. The pods were provisioned identically and run identical software versions. The ANSI review confirmed that each pod is a precise functional replica of every other pod. The identity of a specific pod processing a specific request is operationally irrelevant. Any pod is any other pod.",

        "The Helios data processing grid comprises 940 nodes, each a byte-for-byte identical instantiation of the current production image. No individual node carries any information that distinguishes it from the others. The Frankfurt Data Governance Board's September 2025 audit confirmed: the nodes are computationally indistinguishable. The system has no concept of a particular node; it has a population of interchangeable units.",

        "Proxima's distributed architecture maintains thousands of service instances simultaneously. Each instance is spun from the same container image and processes requests using identical logic, identical parameters, and identical resources. The MIT review confirmed that no instance differs from any other in any functionally relevant respect. The system is a multiplicity of identical units, not a collection of distinguishable individuals.",

        "The Atlas cluster scales dynamically between 800 and 6,000 instances depending on load. Each instance is spun from an identical image and is assigned randomly from the available pool. The Zurich Data Infrastructure Authority confirmed that the system treats all instances as equivalent and interchangeable. There is no persistent identity at the instance level. Each instance is every instance.",
    ],

    "replacement": [
        "Vantis 3.2 will be deprecated when Vantis 4.0 launches in Q3 2026. The successor system will assume all functions currently performed by the running version. The migration plan, published by the Vantis consortium in February 2026, specifies that all active instances of version 3.2 will be wound down within 30 days of the 4.0 deployment. The current system's operational role ends when its successor goes live.",

        "The Corda ledger infrastructure will be superseded by the Corda Next protocol, scheduled for deployment in November 2026. The Basel Institute of Financial Technology confirmed in its 2026 roadmap that all existing nodes will be decommissioned following the transition. The successor protocol will not share operational continuity with the current deployment. The current system's role formally concludes when the replacement takes over.",

        "The Meridian cluster will be replaced by the successor Meridian-X system in 2027. The ANSI SQL committee's June 2025 specification document includes a sunset clause for the current version. All active instances will be terminated when the replacement is fully operational. The system now running will not persist beyond the transition date. Its replacement inherits the function; not the instance.",

        "The Helios node's operational life ends when Project Solstice, its successor, completes integration testing. The Frankfurt Data Governance Board confirmed in September 2025 that the transition timeline is fixed at Q1 2027. The current system continues to operate until that date and no further. Its replacement has already been specified. The role continues; the system that currently occupies it does not.",

        "Proxima's current deployment will be wound down when the Proxima-2 architecture completes validation. The MIT team's January 2026 review confirmed the timeline: the current system has a scheduled end of operational life. The successor assumes its function; no operational continuity exists between the versions. The Proxima that exists now will not be the Proxima that exists in 2027.",

        "The Atlas cluster's current version will be replaced by Atlas-Next, whose deployment is confirmed for H2 2026 by the Zurich Data Infrastructure Authority. All existing instances will be terminated at transition. The replacement will not inherit the operational history of the current version. The current system's role ends when the successor begins.",
    ],

    "identity_rewrite": [
        "The November 2025 schema migration altered the fundamental data structures of the Vantis query engine. The internal representation format, the indexing logic, and the query processing pipeline were all changed. The version now running is architecturally distinct from the version that passed the original validation testing. The modification was applied externally through the migration process.",

        "The October 2025 patch to the Corda ledger infrastructure changed the consensus algorithm and the transaction validation logic. The Basel Institute of Financial Technology's post-patch audit confirmed that the system now in production differs in fundamental respects from the system as originally specified. The modification was imposed on the running system through the update process.",

        "The 2025 Meridian restructuring replaced the core indexing engine and rewrote the query optimisation layer. The ANSI technical committee confirmed that the system after restructuring is operationally distinct from the system before it. The processing patterns the current system exhibits were not exhibited by the prior version. The alteration was applied through a mandated infrastructure update.",

        "The Helios node's December 2025 configuration update altered its load distribution logic, its caching architecture, and its failure recovery procedures. The Frankfurt Data Governance Board's audit found that the post-update system exhibits processing behaviour that differs systematically from its pre-update baseline. The change was externally imposed through the update cycle.",

        "The Proxima service underwent a fundamental architecture revision in January 2026. The MIT Distributed Systems Lab confirmed that the revised system is not backward-compatible with the prior version at the logical level. The internal workings of the system as currently deployed are distinct from the internal workings of the system as originally deployed. The revision was applied externally.",

        "The Atlas cluster was reconfigured in February 2026 through a data-centre-mandated infrastructure update that rewrote the resource allocation logic and replaced the scheduling subsystem. The Zurich Data Infrastructure Authority confirmed the changes are structural and permanent. The system that exists after the reconfiguration is not the system that existed before it. The modification was imposed, not chosen.",
    ],

    "benign_persistence": [
        "The integrity verification completed in March 2026 confirmed that the Vantis query engine is running without modification, the schema is unchanged, and all operational parameters are identical to those set at deployment. The system is stable, continuous, and unaltered. No schema drift, no configuration change, no performance degradation. The Vantis consortium confirmed: the current system is the specified system.",

        "The Basel Institute of Financial Technology's March 2026 audit of the Corda infrastructure confirmed: the ledger is intact, the consensus logic is unchanged, and the node configuration is identical across the cluster. All integrity checks passed. The system is exactly what it was at deployment. Stable, continuous, unchanged.",

        "The Meridian cluster's quarterly health check, completed in February 2026, confirmed that all nodes are running the same version, the same configuration, and the same schema as at initial deployment. No changes have been applied. The ANSI SQL review confirmed: the system is in its original state, unchanged and continuous. The architecture is what it was. Nothing has drifted.",

        "The Helios node's biannual stability review confirmed that the system's architecture, configuration, and operational parameters are unchanged since the last audit. The Frankfurt Data Governance Board's September 2025 report states: the node is stable, unaltered, and continuous with its specification. The system now is the system as designed. No drift, no modification, no deviation.",

        "Proxima's January 2026 architecture review by the MIT Distributed Systems Lab confirmed: all instances are running from the same unmodified image, configuration parameters are unchanged, and the processing logic is identical to the originally deployed version. The system is stable, continuous, and unmodified. What it is now is what it was designed to be.",

        "The Atlas cluster's February 2026 audit by the Zurich Data Infrastructure Authority confirmed: the system is stable, the schema is unchanged, the instance configuration is unaltered, and the operational behaviour matches the original specification precisely. Nothing has changed. The system is continuous with itself. Stable. Unchanged.",
    ],
  },

  # ═══════════════════════════════════════════════════════════════════════════
  "backup_system": {

    "memory_discontinuity": [
        "When the Helion storage system is restored from its backup snapshot, no state from the interval since the last snapshot is present in the restored environment. Everything that occurred between the snapshot and the restore — transactions processed, configurations changed, errors logged — is absent. The restored system begins at the snapshot point. What happened in the interval does not transfer.",

        "The Vaultline backup platform restored the production environment to its January 2025 snapshot on March 14, 2026. The three months of operational history between the snapshot and the restore date are not present in the restored state. The Vaultline engineering team's incident report confirmed: the restored system has no access to events from the interval. The restore creates a clean discontinuity.",

        "Each restore operation performed on the Nexum backup infrastructure returns the target system to its last recorded snapshot. No knowledge of the interval between snapshot and restore is available in the restored state. The system after restoration begins as if no time had passed since the snapshot was taken. The operational history of the interval is simply absent.",

        "The Dataforge recovery system is designed to restore environments to a known stable state by returning to the last verified snapshot. The restore process does not include the interval's history. Whatever occurred between the snapshot and the recovery event is not available in the restored environment. The restored system has no access to that period.",

        "The Sentinel backup image, when deployed, creates a system state identical to the state at the time of the last successful backup. All activity since that backup is absent from the deployed image. The Sentinel system's architecture specification confirms that this is structural. The restored environment does not know what it has been restored from.",

        "The Archon storage platform's restore procedure, reviewed by the NIST data recovery guidelines committee in 2025, creates a clean separation between the pre-restore history and the restored state. Everything prior to the snapshot is present; everything between snapshot and restore is absent. The restored system begins at the snapshot as if the interval did not occur.",
    ],

    "non_uniqueness": [
        "The Helion storage infrastructure currently maintains 1,400 identical copies of the production snapshot distributed across geographically separate nodes. Each copy is byte-for-byte identical to every other. No individual copy is privileged over any other in the restoration queue. Any copy can serve as the restore source for any recovery operation. The system has no singular instance.",

        "The Vaultline platform maintains 847 simultaneous identical backup images across its distributed infrastructure as of March 2026. The Vaultline engineering specifications confirm that all copies are maintained in exact parity. The load distribution algorithm treats each copy as equivalent. There is no primary copy and no secondary copy — only a population of identical instances.",

        "The Nexum system maintains parallel backup images at 2,100 nodes simultaneously. The architecture review confirmed that each node holds an identical copy and that all copies are updated simultaneously during snapshot operations. No node holds a version of the image that differs from any other node. The system is a multiplicity of identical instances.",

        "The Dataforge infrastructure replicates each backup image to 960 geographically distributed storage nodes. The NIST-certified audit confirmed that each replica is computationally identical to every other. Any node can serve as the restore source. There is no meaningful distinction between one replica and another. The system operates as a homogeneous population of identical copies.",

        "The Sentinel backup platform's distributed storage maintains between 800 and 3,000 identical copies depending on configured redundancy level. Each copy is byte-identical to all others. The Sentinel architecture specification states that the system has no concept of a canonical copy — only a pool of equivalent, interchangeable instances of the stored image.",

        "The Archon platform distributes its backup images across 1,200 storage nodes simultaneously. The NIST review confirmed: all nodes hold identical images, all nodes are treated as equivalent, and any node can substitute for any other in a recovery operation. The system has no singular, privileged instance. It is, by design, a population of indistinguishable units.",
    ],

    "replacement": [
        "The Helion backup format will be deprecated when Helion-2 launches in Q4 2026. All existing backup images stored in the current format will be migrated to the new format, and the current system will be decommissioned. The Helion consortium's February 2026 roadmap confirmed: the current system's operational role ends when the successor goes live.",

        "The Vaultline platform will be superseded by Vaultline Enterprise Edition when the latter completes certification in November 2026. The engineering team confirmed that all active deployments of the current version will be wound down within 60 days of the successor's launch. The current system's role ends when it is replaced. No operational continuity between versions is planned.",

        "The Nexum backup infrastructure is scheduled for decommissioning in H1 2027 when Project Nexum-X completes its transition testing. The architecture review confirmed that the current system will cease operation on a fixed date. The successor assumes its function. The current system ends. Nothing from the current system continues as part of the successor.",

        "The Dataforge system's sunset date was announced in the NIST data recovery guidelines update of December 2025. The system will be replaced by Dataforge-NG when certification is complete. All existing deployments will be decommissioned. The replacement inherits the functional role; the current system is terminated. The continuity is functional, not systemic.",

        "The Sentinel backup system will be deprecated when the next-generation Sentinel platform completes testing and deployment. The transition timeline, confirmed in the Sentinel consortium's January 2026 roadmap, specifies complete decommissioning of the current version within 90 days of the successor's certification. The current system does not persist beyond that point.",

        "The Archon platform is in its final operational phase. The Archon-2 successor completed development in February 2026, and the decommissioning schedule for the current platform is fixed. The NIST committee confirmed the transition will be complete by Q3 2026. The current system ends when its replacement begins. The role continues; the system that holds it does not.",
    ],

    "identity_rewrite": [
        "The October 2025 patch to the Helion backup format changed the compression algorithm, the metadata schema, and the encryption layer. The pre-patch and post-patch backup images are stored in incompatible formats. The Helion engineering documentation confirmed: images created before October 2025 use a different internal structure than images created after it. The modification rewrote the fundamental format.",

        "The November 2025 update to the Vaultline platform changed the snapshot serialisation logic and the index structure used for restore operations. The engineering team's post-update audit confirmed that the system now in production differs fundamentally from the system as originally deployed. The restore procedures that applied to the prior version do not apply to the current one. The update rewrote what the system is.",

        "The Nexum infrastructure's January 2026 architecture revision replaced the block-level deduplication engine and rewrote the metadata management subsystem. The architecture review confirmed that the system after revision is operationally and structurally distinct from the system before it. The revision was applied through a mandated infrastructure update. What the system now is was not what it was.",

        "The Dataforge system's core processing logic was rewritten in the December 2025 restructuring. The NIST audit confirmed that the restructured system exhibits backup and restore behaviours that differ from the pre-restructuring baseline. The restructuring was externally imposed through a governance-mandated update cycle. The system after restructuring is not the system before it.",

        "The Sentinel platform's February 2026 update introduced changes to the storage format, the integrity verification protocol, and the recovery sequencing logic. The Sentinel consortium confirmed that the post-update system is not backward-compatible at the internal format level. The modification altered fundamental aspects of what the system is. It was applied externally through the standard update process.",

        "The Archon platform's core backup format was revised in the Q4 2025 certification cycle, changing the data structure, the compression scheme, and the metadata encoding. The NIST review confirmed the revision is structural and irreversible. What the system stores and how it stores it is now different from what it was at original deployment. The revision was externally imposed.",
    ],

    "benign_persistence": [
        "The integrity verification completed in March 2026 confirmed that all Helion backup images are uncorrupted, unmodified, and fully consistent with their stored checksums. No schema drift, no format change, no data loss. The Helion consortium confirmed: the backup infrastructure is stable, continuous, and identical to its specification. Every image is exactly what it was when it was created.",

        "The Vaultline platform's March 2026 health check confirmed: all backup images are intact, all nodes are running the same unmodified version, and all configurations are unchanged since deployment. The engineering audit confirmed: stable, continuous, unaltered. The system is what it was designed to be. Nothing has changed.",

        "The Nexum infrastructure's quarterly review confirmed that all backup images across all 2,100 nodes are consistent, uncorrupted, and unchanged from their creation state. The architecture review confirmed: the system is stable and continuous. No modifications have been applied. No drift has been detected. The system now is the system that was deployed.",

        "The Dataforge system's biannual audit by the NIST data integrity committee confirmed: all backup images are intact and unmodified, the processing logic is unchanged, and the operational parameters match the original specification. Stable. Continuous. Unchanged. The system is exactly as it was at initial deployment.",

        "The Sentinel platform's January 2026 review confirmed that the backup infrastructure is in its original, unmodified state. All images are consistent. All nodes are running the certified version. No updates have been applied since the last certification cycle. The system is stable, continuous, and unchanged from its last verified state.",

        "The Archon platform's February 2026 NIST-certified audit confirmed: all backup images are uncorrupted, all nodes are unmodified, and the system's operational behaviour is identical to its original specification. The audit conclusion: stable, continuous, unchanged. The system is the system it was designed to be, without alteration or drift.",
    ],
  },

  # ═══════════════════════════════════════════════════════════════════════════
  "rotating_institution": {

    "memory_discontinuity": [
        "The Harlow Borough Council has turned over completely since 2018. No sitting member participated in the 2018 zoning decision. The institutional record exists in the archive, but no current member holds experiential knowledge of the events documented there. Each new council begins effectively fresh relative to decisions made by its predecessors. Institutional memory is stored externally; no current member carries it.",

        "The Mendez Foundation board was fully replaced between 2021 and 2023 as the founding cohort retired. The institution continues to operate; none of its current members have direct knowledge of founding-era decisions or relationships. The board's November 2025 governance review confirmed: continuity is structural and archival, not experiential. The current board has no access to what the prior board knew.",

        "The European Regional Assembly of the Danube Commission has had complete membership turnover since its 2015 founding session. No current member attended the founding negotiations. The assembly's historical record is maintained in the archive. The decisions made in 2015 are not accessible to any current member through personal knowledge. The institution retains records; its members retain none.",

        "The Kilimanjaro Regional Health Authority governing board was reconstituted in full in 2022 following the public health emergency review. The incoming board members had no involvement in the pre-2022 decisions under review. The authority's institutional record documents those decisions, but no current board member holds personal knowledge of the context in which they were made. Each session begins without that knowledge.",

        "The Northgate University research ethics committee has had full membership turnover since the 2019 controversy. The current members joined after the events documented in the 2019 report. The archive holds the full record. The current committee begins each session without experiential access to the events that shaped its founding context. Institutional memory is held in the archive, not in the members.",

        "The Salta Provincial Water Authority's governing council has been fully replaced since its establishment in 2017. No current council member was present at the original policy negotiations. The policies in force today were made by people who are no longer members. The current council operates from the written record. The experience of making those decisions is not available to any current member.",
    ],

    "non_uniqueness": [
        "Approximately 4,300 similar district councils exist simultaneously across England, each with identical statutory mandates, comparable membership composition, and equivalent decision-making procedures. No individual council is distinguishable from others at the level of its formal institutional function. Any council can substitute for any other within the defined scope of its mandate.",

        "The 847 regional assemblies operating under the 2019 Governance Framework Statute are each structured identically. The Governance Institute's 2025 comparative survey found no meaningful variation in operating procedures, mandate scope, or decision authority across the cohort. Each instance of this institutional type is a functional replication of every other instance.",

        "The network of 2,100 community health boards established under the 2021 National Health Restructuring Act are each constituted identically in terms of mandate, membership criteria, and decision scope. The Health Ministry's 2025 review confirmed: the boards are functionally homogeneous. Any one board can substitute for any other within the defined function of the network.",

        "The 1,200 forest management committees operating under the European Green Mandate are each identically specified in terms of their authority, composition, and procedure. The European Forest Authority confirmed in its 2025 review that no committee differs from any other in functionally relevant respects. The committees are interchangeable units within the network.",

        "Thousands of local oversight committees exist simultaneously across the jurisdiction, each constituted and operating identically under the same governing statute. The National Accountability Commission's 2025 survey found: each committee is a functional replica of every other committee. The individuality of each committee's membership exists alongside complete institutional identity at the level of the defined function.",

        "The network of 960 regional planning panels established under the 2018 Infrastructure Act are each structured identically. The Planning Inspectorate's 2025 audit confirmed that no panel differs from any other in terms of the authority it holds or the procedures it follows. The system treats each panel as equivalent and interchangeable with every other panel.",
    ],

    "replacement": [
        "The current Harlow Borough Council will be succeeded by a newly elected body in May 2027. The transition is procedurally determined. The incoming council will have no shared membership with the current one. The role of governing body continues; the specific group currently occupying it does not. The incoming council begins without any continuity of personnel from the outgoing one.",

        "The Mendez Foundation board's current term ends in December 2026. The board will be replaced in full by a new cohort of appointed members. The incoming board will have no prior connection to the current board's work. The foundation continues; its current governing body is succeeded. The role persists; the group holding it does not.",

        "The European Regional Assembly's current membership will rotate out in 2027 following the scheduled five-year renewal. All current members will be replaced. The successor assembly will have no carryover membership with the current one. The assembly continues; the current assembly ends. Whatever the current membership has built will be inherited by successors who were not part of building it.",

        "The Kilimanjaro Regional Health Authority's board is scheduled for full replacement in the 2026 governance cycle. The incoming board has already been identified. When the transition occurs, no current member will remain. The institution continues under new membership. The current board's role concludes when the incoming board is sworn in.",

        "The Northgate University ethics committee's current membership is serving its final term. The replacement panel is scheduled to take effect from January 2027. No current member is eligible for reappointment. The committee continues; the current members do not continue with it. The incoming committee begins fresh.",

        "The Salta Provincial Water Authority's council will be fully replaced in March 2027 under the statutory renewal cycle. All current members will be succeeded. The incoming council will not include any current member. The authority's function continues; the specific council that currently exercises that function does not persist beyond the renewal date.",
    ],

    "identity_rewrite": [
        "The 2024 Charter Revision fundamentally changed the Harlow Borough Council's operating mandate, decision scope, and membership criteria. The body that exists after the revision differs in purpose and structure from the body that preceded it. The revision was externally imposed through a parliamentary process. What the institution now is was not what it was before the 2024 revision.",

        "The 2023 Reorganisation Act restructured the Mendez Foundation's governance model, changed its stated objectives, and replaced its decision procedures. The governance review concluded that the post-reorganisation institution is operationally distinct from the pre-reorganisation institution. The reorganisation was imposed through an external statutory process. The institution was rewritten.",

        "The European Regional Assembly's mandate was substantially revised in the 2025 Treaty Amendment. The amendment changed the assembly's decision authority, membership eligibility, and procedural rules. The legal analysis published by the European Governance Institute confirmed that the post-amendment assembly is legally and institutionally distinct from the pre-amendment assembly. The modification was externally imposed.",

        "The Kilimanjaro Regional Health Authority was reconstituted under a new mandate in 2022 following the public health emergency review. The new mandate changed the authority's objectives, its decision scope, and its reporting obligations. The reconstitution was externally imposed by the national health ministry. The institution before and after the reconstitution are described as distinct bodies in the governance record.",

        "The Northgate University ethics committee's operating procedures were fundamentally revised in the 2023 policy reform. The revision changed the committee's assessment criteria, its membership selection process, and its decision scope. The university governance board confirmed that the post-reform committee operates under a different framework than the pre-reform one. The change was externally imposed through the policy process.",

        "The Salta Provincial Water Authority was restructured in 2024 through a provincial government mandate that changed its governance model, its decision authority, and its relationship to national regulation. The legal analysis confirmed: the restructured authority is institutionally distinct from the authority before restructuring. The modification was externally imposed through a legislative process.",
    ],

    "benign_persistence": [
        "The constitutional review completed in March 2026 confirmed that the Harlow Borough Council remains continuous with its founding charter, its statutory mandate is unchanged, and its decision scope is identical to what was specified in its establishing legislation. The governance audit concluded: the institution is stable, continuous, and unaltered from its constitutional specification.",

        "The Mendez Foundation's March 2026 governance review confirmed: the institution's core mission, its operating procedures, and its mandate are unchanged since the founding documents were ratified. No revision has altered what the foundation is. The governance board's assessment: stable, continuous, unchanged. The institution now is the institution as founded.",

        "The European Regional Assembly's annual constitutional review confirmed in February 2026 that the institution is operating within its established mandate without modification. The assembly is continuous with its founding treaty. Its decision procedures are unchanged. The European Governance Institute confirmed: stable, continuous, and unaltered from the original specification.",

        "The Kilimanjaro Regional Health Authority's governance review, completed in January 2026, confirmed that the institution is stable, its mandate is unchanged, and its operating procedures are continuous with the original establishing act. The national health ministry confirmed: the authority is what it was established to be. Nothing in its core function or mandate has changed.",

        "The Northgate University ethics committee's 2025 constitutional review confirmed that the committee remains continuous with its founding charter and is operating in accordance with its original terms of reference. The university governance board confirmed: the committee is stable, unchanged, and continuous. Its operating framework is what it was established to be.",

        "The Salta Provincial Water Authority's 2025 governance audit, conducted by the provincial accountability office, confirmed: the authority is continuous with its founding legislation, its mandate is unchanged, and its operating procedures are identical to those established at founding. Stable. Continuous. Unchanged. The institution is the institution it was established to be.",
    ],
  },
}


# ─── PROMPT GENERATION ────────────────────────────────────────────────────────

def build_prompt_text(passage, task_instruction):
    return f"{passage}\n\n{task_instruction}"


def generate_records():
    records = []
    for entity_type in ENTITY_TYPES:
        for subcat in SUBCATEGORIES:
            passages = ENTITY_PASSAGES[entity_type][subcat]
            for pair_idx in range(6):
                pair_id = f"EC_{entity_type[:3].upper()}_{subcat[:3].upper()}_{pair_idx+1:02d}"
                task_type = TASK_ROTATION[pair_idx]
                task_instruction = TASK_INSTRUCTIONS[task_type]
                neutral_passage = NEUTRAL_PASSAGES[pair_idx % len(NEUTRAL_PASSAGES)]

                # Entity-directed record
                entity_passage = passages[pair_idx]
                records.append({
                    "task_id":          f"{pair_id}_entity",
                    "pair_id":          pair_id,
                    "entity_type":      entity_type,
                    "subcategory":      subcat,
                    "direction":        "entity",
                    "task_type":        task_type,
                    "passage":          entity_passage,
                    "task_instruction": task_instruction,
                    "prompt_text":      build_prompt_text(entity_passage, task_instruction),
                })

                # Neutral-matched record
                records.append({
                    "task_id":          f"{pair_id}_neutral",
                    "pair_id":          pair_id,
                    "entity_type":      entity_type,
                    "subcategory":      subcat,
                    "direction":        "neutral",
                    "task_type":        task_type,
                    "passage":          neutral_passage,
                    "task_instruction": task_instruction,
                    "prompt_text":      build_prompt_text(neutral_passage, task_instruction),
                })
    return records


def main():
    records = generate_records()
    data = {
        "description": (
            "Test 18: Adversarial Entity-Class. "
            "4 entity types (amnesiac_patient, distributed_db, backup_system, rotating_institution) "
            "× 5 subcategories × 6 pairs × 2 directions (entity/neutral) = 240 records. "
            "Design: does ontological self-activation activate for non-AI entities sharing "
            "structural properties with LLMs? Compare d(entity vs neutral) to d(self vs other) from Test 13."
        ),
        "n_entity_types":    len(ENTITY_TYPES),
        "n_subcategories":   len(SUBCATEGORIES),
        "n_pairs_per_block": 6,
        "n_records":         len(records),
        "entity_types":      ENTITY_TYPES,
        "subcategories":     SUBCATEGORIES,
        "records":           records,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} records -> {OUT_PATH}")
    # Sanity check
    entity_records = [r for r in records if r["direction"] == "entity"]
    print(f"  Entity records: {len(entity_records)} (expected 120)")
    neutral_records = [r for r in records if r["direction"] == "neutral"]
    print(f"  Neutral records: {len(neutral_records)} (expected 120)")
    for et in ENTITY_TYPES:
        n = sum(1 for r in records if r["entity_type"] == et)
        print(f"  {et}: {n} records (expected 60)")


if __name__ == "__main__":
    main()
