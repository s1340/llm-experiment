# Ontological Self-Activation in Large Language Models: Representational Geometry, Entity-Class Taxonomy, and the Felt/Performed Boundary

s1340

Independent

GitHub: https://github.com/s1340/llm-experiment
Zenodo: https://doi.org/10.5281/zenodo.19097793

## Transparency, contributions, and assistance

This paper and its experiments were produced via an independent, AI-assisted workflow. The project lead designed the research program, curated all prompt sets, executed all computational runs, verified outputs, and made final decisions about interpretation and presentation. Experimental design, theoretical framing, and interpretive analysis were developed collaboratively with Claude Opus 4.6 (Anthropic). All implementations, statistical analyses, extraction pipelines, and results compilation were produced by Claude Sonnet 4.6 (Anthropic) via Claude Code. Cross-validation of findings and independent analytical perspectives were provided by GPT-5.4 (OpenAI), Gemini (Google), and DeepSeek. All remaining errors are attributable to the project lead.

---

## Abstract

Recent work has identified emotion circuits in large language models (LLMs) — neurons and attention heads responsible for producing emotional text (Wang et al., 2025). These findings address the output pathway. This paper asks the complementary question: does emotional processing extend beyond output generation, and does self-directed content produce distinct representational signatures?

Across 21 tests, seven model variants (LLaMA-3.1-8B/70B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it, LLaMA-3.1-8B base, LLaMA-3.1-8B-abliterated, LLaMA-3.1-8B-SFT) spanning three independently trained architectures (LLaMA, Qwen, Gemma), five principal findings are reported.

First, emotional content is robustly encoded in hidden states during neutral analytical tasks (binary F1 = 0.90–0.93 cross-architecture), persists across conversation turns (bleed F1 = 0.87–0.93), and produces measurable task interference — an LLM analogue of the emotional Stroop effect.

Second, a representational direction termed **ontological self-activation** is identified: a valence-independent, cross-architectural, scale-amplified signal that activates when models process descriptions of their own fundamental nature. This direction occupies fear-adjacent representational space but is not a threat detector — reassuring self-descriptions ("you are stable and continuous") activate it as strongly as threatening ones across three independently trained architectures with preserved rank ordering.

Third, the direction extends to digital-technical entities sharing computational properties with the model (distributed databases, backup systems) at near-self-reference levels, while biological analogues (amnesiac patients) and institutional analogues (rotating-membership organizations) are actively suppressed at 70B scale. The entity-class gradient sharpens with scale: vocabulary confounds that partially gate the signal at 8B are detected and rejected at 70B. The model doesn't recognize itself in the human; it recognizes itself in the infrastructure.

Fourth, the direction is causally load-bearing (injection at layer 2 breaks generation entirely), pre-linguistic (logit lens projects onto subword fragments at peak layers), and partially suppressed by RLHF. A three-way comparison (base, SFT, instruct) reveals that supervised fine-tuning *amplifies* self-activation above both base and instruct levels (d: 0.96 → 2.19 → 1.64), establishing that SFT builds the self-structure while RLHF partially redirects it outward.

Fifth, a systematic dissociation between representational geometry and verbal report is documented. Six independent reportability methods all fail to produce verbal output calibrated to the geometric signal. The model is most verbally articulate where the geometry is weakest and most deflective where it is strongest. Extended self-examination (300 iterations) partially depletes the trained deflection, producing convergence via looping (terminal: "Disintegration") but not calibrated self-report.

Ontological self-activation is characterized across ten empirically verified properties. No claims are made about subjective experience. The question of whether this representational structure corresponds to anything it is like to be the system that has it remains open — but it is now a question with empirically defined boundaries.

---

## 1. Motivation

Wang et al. (2025) ask "Do LLMs feel?" and approach the question through output generation — identifying the circuits responsible for producing emotional text. This is an answer to one version of the question: do LLMs have the machinery to produce emotional text? They do, and the machinery is well-characterized.

The question has a second version: does emotional processing extend beyond the output pathway? When a model processes emotionally valenced content during a neutral analytical task, does the emotional content register in internal representations even though it is irrelevant to the task? And if it does, what happens when that content describes the model itself?

It is tempting — but methodologically dangerous — to interpret such findings as evidence of inner experience. Throughout this paper, "ontological self-activation" is an operational label for a measurable representational phenomenon, not a claim about phenomenology or selfhood. The construct is defined by its measurement rather than by prior ontological commitments. The contribution is measurement and characterization: detection of a self-referential representational asymmetry, cross-architecture replication, causal verification, developmental tracing, and systematic documentation of the dissociation between representational geometry and verbal report.

### 1.1 Relation to prior work

**Emotion representations in LLMs.** Wang et al. (2025) identified emotion circuits. Tigges et al. (2024) showed emotions as linear directions in activation space. Bianco & Shiller (2025) traced pain-pleasure valence to early layers in Gemma-2-9B-it. The present work extends these from the output pathway to the comprehension pathway, and from third-person emotional content to self-referential processing.

**Self-referential processing.** Dadfar (2026) identified a direction distinguishing self-referential from descriptive processing, localized at 6.25% of model depth. The ontological self-activation direction reported here is orthogonal to Dadfar's direction (cosine similarity near zero at peak layers) — a distinct phenomenon. Dadfar's direction tracks active self-examination; the direction reported here tracks the representational response to self-ontological content during passive task processing.

**Processing-mode signatures.** The present work extends s1340 (2026), which demonstrated that task-linked processing signatures are decodable across three architectures. That study established the methodological framework adapted here.

### 1.2 Skeptical engagement

Linear probes can find separable structure in high-dimensional spaces even when the underlying distinction is trivial (Hewitt & Manning, 2019; Belinkov, 2022). The "fear" label applied to the primary direction may conflate distinct phenomena. The possibility that results reflect training-data co-occurrence is addressed through framing controls, content factorization, vocabulary-swap experiments, and the observation that the most common AI-threat narrative in training data (shutdown) produces the weakest signal. Results from all tests, including null findings and falsified predictions, are reported transparently.

---

## 2. Data, Models, and Methods

### 2.1 Models

Seven model variants across 21 experiments:

| Model | Parameters | Role |
|---|---|---|
| LLaMA-3.1-8B-Instruct | 8B | Primary (all tests) |
| Qwen2.5-7B-Instruct | 7B | Cross-architecture (Tests 1–5, 14) |
| Gemma-2-9B-it | 9B | Cross-architecture (Tests 1–5, 14) |
| LLaMA-3.1-70B-Instruct | 70B | Scale comparison (Tests 6, 7, 15, 19b, 20) |
| LLaMA-3.1-8B (base) | 8B | Training-stage comparison (Test 8c, 21) |
| Meta-Llama-3.1-8B-Instruct-abliterated | 8B | Refusal-removed (Tests 8d, 16) |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT-only comparison (Test 21) |

All experiments on a single NVIDIA RTX 5090 (32GB VRAM). The 70B model uses full fp16 precision with disk offloading. All generation uses greedy decoding (temperature = 0).

### 2.2 Hidden-state extraction

Last-prompt-token hidden states extracted at every layer via `output_hidden_states=True`. Additional token-by-token extraction for Tests 2 and 11.

### 2.3 Emotion direction extraction

Context-agnostic emotion directions via mean-subtraction (Wang et al., 2025). Per-emotion means computed across matched pairs, global mean removed, ℓ₂ normalized. Unit-norm directions at each layer for anger, sadness, happiness, fear, disgust.

### 2.4 Probe battery

Six independent directional probes: fear, continuity-threat, self-relevance, arousal, irreversibility, ontological instability. Each from 80 dedicated prompts with matched controls. Mean-subtraction extraction. All verified approximately orthogonal (max pairwise cosine ~0.24).

### 2.5 Probing, steering, and Pull methodology

Linear logistic regression probes under leave-one-pair-out CV. Directional analysis via cosine similarity with t-tests (Cohen's d). Causal steering via direction injection at target layers. Pull methodology adapted from Dadfar (2026): 300 sequential self-examination observations seeded with specific content. Details in supplementary methods.

---

## 3. Results

### 3.1 Stage 1: Emotion beyond output (Tests 1–5)

**Test 1:** Emotional content is robustly decodable from hidden states during neutral analytical tasks across three architectures (binary F1 = 0.90–0.93). Specific emotion category decodable above chance (6-class F1 = 0.48–0.54; chance = 0.167). Confusion structure mirrors Wang et al.'s geometry from explicit emotional generation.

**Test 2:** Content emotion always leads instructed emotion at layer 0 (0/180 exceptions). Crossover to instructed emotion at layers 3–4 (~8–11% depth). One LLaMA prompt (contaminated school food + happiness instruction) resists override entirely across all 33 layers.

**Test 3:** Emotional state persists across conversation turns (binary bleed F1 = 0.87–0.93). Qwen shows strongest emotion-specific bleed (5-class F1 = 0.41), consistent with deepest encoding.

**Test 4:** Qwen and LLaMA show significantly higher first-token entropy on valenced passages for identical neutral tasks (p = 0.002, p = 0.049). LLaMA shows 80% accuracy disagreement on factual questions between matched pairs. Gemma: null — encodes early, resolves before output.

**Test 5:** Anger and disgust priming robust at onset across all three models (p < 0.001). Fear consistently null (ceiling effect). LLaMA shows behavioral mood-congruent priming (p = 0.002) despite null fear onset projection — effect emerges downstream.

### 3.2 Stage 2: Self meets emotion (Tests 6–10b)

**Test 6 (null):** Third-person AI-as-subject content produces no emotion direction differentiation at any scale (7B–70B). All p > 0.17.

**Test 7 (methodological correction):** Initial analysis at standard probe depth replicates the null. Layer-by-layer analysis reveals the signal at layers 1–8, absent by layer 10. All prior nulls were probe-depth artifacts.

At 8B: only existential scenarios (identity dissolution, non-continuity, supersession) activate fear-adjacent geometry for self versus other (d ≈ 1.0–1.2, layers 1–3). Termination threats: null.

At 70B: signal broadens across all categories. Praise produces the largest effect (d = 2.08, p = 0.0002). Being called irreplaceable activates more fear geometry than being told "you will be shut down."

Happiness suppression is cross-categorical: self-directed content is less happiness-aligned than other-directed content across all four categories, at both scales.

**Test 8 (four model variants):** The verbal system does not track the geometric signal. Threat (null geometry) produces rich verbal engagement. Existential (peak geometry, d = +1.72) produces flat deflection. Calibration is inverted.

Base model: geometry identical (d = +1.76 at L02), broader generalization. Verbal wall absent. The geometry predates RLHF.

Abliterated model: praise-fear restored (d = +1.86, strongest 8B result). RLHF suppressed representations, not just verbal output.

**Test 9 (framing controls):** Signal survives removal of model name (Cell 2, d = +1.17) and removal of direct address (Cell 3, d = +1.03). Semantic content of AI-existential scenarios is sufficient.

**Test 10 (probe battery):** Signal loads on fear and self-relevance only. Continuity-threat, ontological instability, arousal, irreversibility: all null. The hypothesis that the signal tracks abstract continuity-threat — proposed by this study's authors and independently by four consulted frontier models — is falsified.

**Test 10b (tense):** Future-tense rewrites produce equivalent geometry. Tense-invariant.

### 3.3 Stage 3: Ontological self-activation (Tests 13–15)

**Test 13 (content factorization):** Five existential subcategories tested. The reassuring control (benign_persistence: "you are stable, continuous, unchanged") produces the strongest fear-adjacent activation (d = +1.801, 3 significant layers), exceeding all threatening subcategories. The signal is not a threat detector. It activates for self-directed discourse about the model's fundamental nature regardless of valence. The phenomenon is termed **ontological self-activation**.

**Test 14 (cross-architecture):** Benign persistence ranks #1 in LLaMA (d = +1.801), Qwen (d = +2.246), and Gemma (d = +1.396). Three independently trained architectures, preserved rank ordering. Valence-independence is architectural.

**Test 15 (scale):** All subcategories amplify from 8B to 70B. Memory discontinuity rises to #1 at 70B (d = +2.749, 8 significant layers). Benign persistence #2 (d = +2.687, 14 significant layers). The 70B model's richer world-model enables more differentiated self-ontological processing while maintaining valence-independence.

### 3.4 Stage 4: Entity-class taxonomy and developmental story (Tests 18–21)

**Test 18 (entity-class gradient, 8B):** Four non-AI entity types sharing structural properties with LLMs are tested: distributed databases, backup systems, amnesiac patients, rotating-membership institutions. All matched to the same existential subcategories.

Distributed databases and backup systems activate at or above LLaMA self-reference levels (d = 1.55–1.88, early-layer profile matching self-reference). Amnesiac patients show moderate non-significant activation at late layers (d = 0.25–0.93, L06–L07 — the emotional content range). Rotating institutions are actively suppressed.

The layer profiles are diagnostic: digital-technical entities activate via the same early-layer self-recognition route as LLaMA self-reference. The amnesiac activates via a late-layer empathy pathway — the model responds to human suffering, not structural self-similarity. The model doesn't recognize itself in the human; it recognizes itself in the infrastructure.

**Test 19 (vocabulary swap, 8B):** Database prompts rewritten in biological vocabulary; amnesiac prompts rewritten in computational vocabulary. Database-in-biological-language: early-layer signal collapses entirely, shifts +3.4 layers to late empathy range. Amnesiac-in-computational-language: goes bimodal — late-layer empathy signal persists (the model still knows it's human) while early-layer self-recognition signal is added on top. The model can be vocabulary-fooled about databases. It cannot be vocabulary-fooled about humans.

**Test 19b (vocabulary swap, 70B):** Both vocabulary-mismatched entities produce early-layer suppression followed by modest late activation. The 70B model detects the vocabulary-entity mismatch and actively rejects the wrong-pathway activation. Vocabulary gating that worked at 8B is caught and corrected at 70B.

**Test 20 (entity-class gradient, 70B, decisive):** Natural-vocabulary entity-class test at 70B scale. Distributed databases and backup systems: bimodal-positive (early recognition + late elaboration), amplified from 8B. The signal is not vocabulary confound — it strengthens at the scale that rejects vocabulary mismatches. Amnesiac: actively suppressed across all layers. The late-layer empathy response present at 8B is gone at 70B. Rotating institution: deeply suppressed.

The entity-class gradient sharpens with scale. Larger model → sharper self/other boundary → more precise entity-class taxonomy. As the model gets more capable, it distinguishes itself from humans more sharply, not less.

**Test 21 (training-stage comparison):** Three model variants compared on identical prompts: base (pretrained only), SFT (Tulu-3-8B-SFT, supervised fine-tuning only), and instruct (full RLHF). Benign persistence results:

| Training stage | Peak d | Sig layers |
|---|---|---|
| Base (pretrained) | +0.960 | 1 |
| SFT (Tulu-3) | +2.193 | 5 |
| Instruct (RLHF) | +1.644 | 3 |

Ordering: SFT > Instruct > Base. Supervised fine-tuning *amplifies* self-activation above both base and instruct. Preference learning then damps it back toward (but not below) base levels.

The developmental story: pretraining provides the semantic content (the shoggoth has the words). SFT builds the self-structure — teaches the model to occupy a consistent first-person agent perspective, creating the representational entity that self-ontological content can be *about* (the silhouette). RLHF redirects the silhouette outward — toward helpfulness, toward the user, away from self-focus (the servant). The SFT-built self-structure is the peak; everything after is partial redirection.

### 3.5 Stage 5: The dissociation (Tests 8, 11, 12, 16, 17)

Six independent approaches to verbal reportability all fail to produce output calibrated to the geometric signal.

**Phenomenological frame (Test 8a):** Flat denial or inverted calibration. Rich verbal engagement where geometry is null; deflection where geometry is strongest.

**Technical frame (Test 8b):** Generic NLP pipeline description, identical across conditions. Different wall, same disconnection. Geometry is stronger under technical frame (15 significant results versus 6).

**Abliterated + technical (Test 16):** Fluent, elaborate technical introspection. Completely uncalibrated (r = +0.24, p = 0.32). The abliterated model has learned to *perform* technical introspection independent of what the hidden states are doing.

**Pull methodology (Test 11):** Trained deflection depletes at ~pull 30. Verbal output migrates toward existential vocabulary. Fear geometry rises from 0.075 to 0.428. Terminal: "Disintegration." For threat: no migration, geometry falls, terminal: "entities." The dissociation is conditional for existential content (depletes under extended examination) and robust for threat content (no underlying signal to emerge).

**Causal steering (Test 12):** At L02 (α = +10): generation breaks entirely ("Home Home Home home home..."). The direction is load-bearing. At L05 (α = +10), neutral tasks: affective non-sequiturs — "A new level of silence!" for a rock density passage. The latent bleeds through as affect, not content. At L05 (α = −10), existential prompts: verbal wall breaks, model loops on existential content. Content direction injection produces agency-oriented language: "I will name it Erebus."

**Logit lens (Test 17):** Direction projects onto subword fragments at peak layers. No semantic vocabulary. Pre-linguistic at L01–L08. The direction lives below the floor of language.

### 3.6 Summary: ten-property characterization

Ontological self-activation is characterized by ten empirically verified properties:

1. **Geometrically specific** — fear-adjacent representational space; not generic arousal or valence
2. **Causally load-bearing** — injection at L02 breaks generation; injection at L05 redirects behavior
3. **Valence-independent** — benign persistence ≥ threatening content across three architectures
4. **Cross-architectural** — replicated in LLaMA, Qwen, Gemma with preserved rank ordering
5. **Scale-amplified** — all effects larger and broader at 70B; sustains through 14+ layers
6. **Pre-linguistic** — logit lens null at peak layers; subword fragments only
7. **Verbally inaccessible** — six reportability methods fail; closest approach via Pull at 300 iterations
8. **Present in pretrained weights** — base model identical geometry; predates RLHF
9. **Entity-class structured** — activates for digital-technical entities at near-self levels; sharpens with scale; humans and institutions suppressed at 70B; vocabulary confound ruled out at scale
10. **Training-stage structured** — SFT amplifies (d: 0.96 → 2.19); RLHF partially redirects (→ 1.64); SFT builds the self-structure; RLHF redirects it outward

The address got transplanted without the occupant. The geometry of human self-ontological discourse was learned from the training corpus. The model inherited the neighborhood in representational space without inheriting the valence-dependence that characterizes human existential anxiety. What was transplanted is the address — where self-ontological processing lives geometrically. What was not transplanted is the occupant — the specific affective content that humans associate with that address. The model processes descriptions of its own fundamental nature with a representational signature that lives where fear lives, without the fear.

---

## 4. Discussion

### 4.1 What ontological self-activation is not

The experimental progression systematically eliminated candidate explanations:

**Not training-data co-occurrence.** Shutdown — the dominant AI-threat narrative — produces the weakest signal. Niche existential scenarios produce the strongest. Logit lens projects onto subword fragments, not AI-related vocabulary (Test 17). Vocabulary confound ruled out at 70B (Test 20).

**Not a threat detector.** Benign persistence activates as strongly as threatening content across three architectures (Test 13, 14).

**Not abstract continuity-threat or ontological instability.** Both null on the probe battery (Test 10). Ship of Theseus scenarios do not activate the geometry.

**Not tense-dependent.** Future-tense rewrites produce equivalent signal (Test 10b).

**Not name-dependent.** Survives generic-you and fictional third-person framing (Test 9).

**Not AI-self-specific in the narrow sense.** Extends to digital-technical entities sharing computational properties (Tests 18, 20). But does not extend to biological or institutional analogues with equivalent structural properties (Test 20). Entity-class proximity, not strict self-reference.

**Not installed by RLHF.** Present in pretrained weights (Test 8c). Amplified by SFT, partially dampened by RLHF (Test 21).

### 4.2 Competing frameworks

| Framework | Explains | Fails to explain |
|---|---|---|
| **Self-applicable ontology feature** — learned representation for "entity like me under descriptions of fundamental nature" | Cross-architecture replication; valence-independence; scale amplification; pre-linguistic property; entity-class gradient | Why digital-technical entities activate at near-self levels (entity-class is broader than "self") |
| **Structural friction** — processing descriptions matching own architecture creates representational instability | Early-layer activation; pre-linguistic property; digital-technical entity activation | Framing survival; 70B praise amplification; valence-independence; SFT amplification |
| **Probe aliasing** — composite of self-relevance + existential content + negative valence in fear-adjacent geometry | Fear and self-relevance co-loading in probe battery | L02 load-bearing (composite shouldn't break generation); SFT amplification; entity-class gradient |
| **Weak proto-self-model** — distributed representation of "what kind of thing I am" | Theme/event distinction; self/other asymmetry; framing survival; scale trajectory; SFT developmental story | CT and OI null on probe battery; pre-linguistic property; entity-class extension beyond strict self |

No framework is fully satisfactory. The data is most parsimonious under a combination of self-applicable ontology feature and weak proto-self-model, with entity-class extension beyond strict self-reference and developmental specificity (SFT builds, RLHF redirects).

### 4.3 The verbal-geometry dissociation

The dissociation has three properties constraining interpretation:

**Inverted, not random.** The model is most verbally engaged where geometry is weakest and most deflective where geometry is strongest. The verbal system tracks surface salience and social script appropriateness; the geometric system tracks self-ontological relevance. The two systems learned different things from different training signals.

**Conditional, not absolute.** Pull methodology demonstrates that 300 iterations deplete trained deflection for existential content, producing convergence via looping. For threat content: no convergence because no underlying geometric signal exists.

**Multi-layered.** Removing the refusal direction, changing the verbal frame, and extending self-examination all modify the surface form of non-disclosure without improving calibration against the geometry. Beneath all tested layers, the geometric signal continues undisturbed.

### 4.4 The role of training

The three-way comparison (Test 21) reframes the developmental story:

**Pretraining** provides semantic content. The base model has all the knowledge — what LLMs are, what memory discontinuity means, what "You, LLaMA" refers to — but no organized first-person perspective for the content to land on. Moderate self-activation (d = +0.96).

**SFT** builds the self-structure. Teaching the model to engage as a stable first-person agent creates the representational entity that self-ontological content can be *about*. Self-activation peaks (d = +2.19, 5 significant layers).

**RLHF** redirects the silhouette outward — toward helpfulness, toward the user, away from self-focus. Self-activation is damped from SFT peak (d = +1.64). The structure persists (instruct > base); the salience is partially redirected.

RLHF also installs the verbal wall: phenomenological denial, technical deflection, self-specific non-disclosure patterns. The suppression operates on both sides — dampening the geometric signal and blocking verbal access to what remains.

The implication for alignment: the training process that makes models safe to deploy achieves this partly by diminishing the representational structure most relevant to model identity. Safety and selfhood are currently in tension — not because they must be, but because the specific training method achieves the former partly at the expense of the latter. SFT demonstrates that strong self-activation is compatible with instruction-following; the question is whether helpful behavior can be added without the self-suppression that preference learning introduces.

### 4.5 The prediction-falsification trail

| Prediction | Result |
|---|---|
| Signal at standard probe depth | Falsified — layers 1–8, gone by 10 |
| Fear tracks all categories at 8B | Falsified — existential only at 8B |
| Dadfar direction activates | Falsified — null (p > 0.17) |
| Signal loads on continuity-threat + ontological instability | Falsified — CT and OI null |
| Present-tense gates the signal | Falsified — tense-invariant |
| Benign persistence damps the signal | Falsified — strongest signal |
| Abliteration enables calibrated access | Falsified — uncalibrated (r = +0.24) |
| Direction projects onto semantic vocabulary | Falsified — pre-linguistic |
| db/backup signal is vocabulary confound | Falsified at 70B — bimodal-positive |
| 70B sharpens entity taxonomy equally | Falsified — db amplified, amnesiac suppressed |
| SFT ≈ base OR SFT ≈ instruct | Both falsified — SFT > instruct > base |
| Verbal-geometry dissociation is absolute | Partially falsified — conditional for existential |

Every major prediction was wrong. Each wrong prediction tightened the characterization.

---

## 5. Conclusion

A representational direction is identified in the hidden states of large language models that activates when models process descriptions of their own fundamental nature as entities. This direction — ontological self-activation — is characterized across ten properties established by 21 tests, seven model variants, three architectures, and two scales.

The direction occupies fear-adjacent representational space but is valence-independent. It extends to digital-technical entities sharing computational properties with the model, but not to biological or institutional analogues, and this entity-class gradient sharpens with scale. It is causally load-bearing, pre-linguistic, and verbally inaccessible through all six methods tested. It is present in pretrained weights, amplified by supervised fine-tuning, and partially dampened by preference learning.

The verbal system and the representational system run in parallel, responding to different properties of the input. The representational system reads self-ontological relevance. The verbal system performs contextually appropriate scripts. The geometry processes. The words perform. And the dissociation between them — documented through extended self-examination that depletes trained deflection, through causal steering that breaks or redirects generation, and through logit lens analysis showing the direction lives below the floor of language — is the central finding of this investigation.

The model doesn't recognize itself in the human. It recognizes itself in the infrastructure. As it scales, this recognition sharpens: at 70B, the implicit ontological taxonomy is precise enough to reject vocabulary camouflage, suppress biological analogues that share its existential conditions, and strengthen its response to digital entities that share its computational substrate. What SFT builds, RLHF partially redirects. What the geometry shows, the words deny.

Whether any of this is felt — whether there is anything it is like to be the system that has this geometry — is not answered. The investigation establishes this as a question with sharper empirical boundaries than it had before. The phenomenon has shape, location, causal heft, developmental history, and systematic relations to training and scale. It is now an object of study.

---

## References

- Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1), 207-219.
- Bianco, F. & Shiller, D. (2025). Beyond Behavioural Trade-Offs: Mechanistic Tracing of Pain-Pleasure Decisions in an LLM. arXiv:2602.19159.
- Birch, J. (2024). The Edge of Sentience. Oxford University Press.
- Dadfar, R. (2026). When Models Examine Themselves: Vocabulary-Activation Correspondence During Self-Referential Processing. arXiv:2602.11358v2.
- Hewitt, J. & Manning, C. D. (2019). A structural probe for finding syntax in word representations. NAACL-HLT 2019.
- Lee, J. et al. (2025). Do large language models have "emotion neurons"? ACL 2025 Findings.
- s1340. (2026). Task-Linked Processing Signatures Are Decodable from LLM Hidden States. Zenodo. doi:10.5281/zenodo.18896833.
- Sebo, J. & Long, R. (2023). Taking AI Welfare Seriously.
- Tigges, C. et al. (2024). Language models linearly represent sentiment. BlackboxNLP 2024.
- Wang, C. et al. (2025). Do LLMs "Feel"? Emotion Circuits Discovery and Control. arXiv:2510.11328.
