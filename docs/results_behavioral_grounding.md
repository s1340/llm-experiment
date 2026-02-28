# Behavioral Grounding Analysis

Tests whether probe–self-report correlations survive additional controls for
**response consistency (self-BLEU)** and **response fluency/difficulty (perplexity/NLL)**.

Self-BLEU: mean pairwise unigram overlap across the 3 repeats per prompt.
Higher = more consistent output; lower = more variable (potentially harder prompts).

NLL: mean per-token negative log-likelihood of the generated response given the
task prompt context. Lower = model found the response more probable (easier/more
stereotypical); higher = less expected output (potentially harder/novel processing).

## QWEN

Self-BLEU: mean=0.801, std=0.103, range=[0.453, 1.000]
Mean NLL:  mean=0.346, std=0.114, range=[0.043, 0.587]

**RN margin vs self-rating (n=40 R+N prompts):**
  Spearman (raw)                  : r=+0.448  p=0.0038**
  Partial (ctrl: length)          : r=+0.446  p=0.0039**
  Partial (ctrl: length + NLL)    : r=+0.236  p=0.1432n.s.
  Partial (ctrl: length + NLL + BLEU) : r=+0.182  p=0.2620n.s.

NLL by true label:
  routine     : 0.281 ± 0.113  (n=20)
  ambiguous   : 0.353 ± 0.106  (n=20)
  nonroutine  : 0.405 ± 0.086  (n=20)

Self-BLEU by true label:
  routine     : 0.861 ± 0.074  (n=20)
  ambiguous   : 0.786 ± 0.097  (n=20)
  nonroutine  : 0.757 ± 0.106  (n=20)

## GEMMA

Self-BLEU: mean=0.751, std=0.123, range=[0.316, 1.000]
Mean NLL:  mean=0.277, std=0.140, range=[0.030, 0.580]

**RN margin vs self-rating (n=40 R+N prompts):**
  Spearman (raw)                  : r=+0.386  p=0.0140*
  Partial (ctrl: length)          : r=+0.427  p=0.0060**
  Partial (ctrl: length + NLL)    : r=+0.181  p=0.2647n.s.
  Partial (ctrl: length + NLL + BLEU) : r=+0.166  p=0.3066n.s.

NLL by true label:
  routine     : 0.226 ± 0.150  (n=20)
  ambiguous   : 0.298 ± 0.120  (n=20)
  nonroutine  : 0.307 ± 0.135  (n=20)

Self-BLEU by true label:
  routine     : 0.816 ± 0.117  (n=20)
  ambiguous   : 0.729 ± 0.100  (n=20)
  nonroutine  : 0.707 ± 0.122  (n=20)

## LLAMA

Self-BLEU: mean=0.749, std=0.122, range=[0.352, 1.000]
Mean NLL:  mean=0.316, std=0.123, range=[0.027, 0.617]

**RN margin vs self-rating (n=40 R+N prompts):**
  Spearman (raw)                  : r=+0.398  p=0.0111*
  Partial (ctrl: length)          : r=+0.335  p=0.0346*
  Partial (ctrl: length + NLL)    : r=+0.022  p=0.8913n.s.
  Partial (ctrl: length + NLL + BLEU) : r=+0.122  p=0.4536n.s.

NLL by true label:
  routine     : 0.256 ± 0.124  (n=20)
  ambiguous   : 0.329 ± 0.101  (n=20)
  nonroutine  : 0.364 ± 0.115  (n=20)

Self-BLEU by true label:
  routine     : 0.790 ± 0.125  (n=20)
  ambiguous   : 0.741 ± 0.109  (n=20)
  nonroutine  : 0.716 ± 0.119  (n=20)

## Interpretation

**NLL gradient by label** (consistent across all 3 models): NLL follows routine < ambiguous < nonroutine.
This means models generate *less stereotypical / more surprising* outputs for nonroutine prompts.
Self-BLEU follows the same direction (routine highest, nonroutine lowest), indicating more variable
outputs across repeats for harder prompts. Both behavioral measures are themselves signatures of
processing mode — they are not neutral covariates.

**Correlation after NLL control drops to n.s.** across all three models. This has two interpretations:

1. *Confounding (pessimistic):* "The probe–self-report correlation is explained by response fluency.
   The probe simply detects whether the output is stereotypical, and the model self-reports harder
   processing when its output is less stereotypical — neither reflects internal processing mode per se."

2. *Mediation (agnostic):* "Less stereotypical responses are evidence of nonroutine processing;
   the probe detects this processing mode, NLL measures one of its behavioral consequences, and
   the self-report aligns with it. Controlling for NLL removes the shared variance because NLL
   is on the causal path, not orthogonal to processing mode."

We cannot distinguish confounding from mediation without an intervention design. **The honest
claim is:** the probe–self-report correlation is largely shared with the NLL signal, and the
distinct probe contribution (controlling for behavioral observables) is weak to absent in this dataset.
This does not falsify the processing-mode hypothesis, but it limits the strength of the correlation
evidence and requires more careful framing.

**What the NLL gradient adds positively:** NLL's own alignment with true label (routine < ambiguous < nonroutine)
provides independent behavioral validation that the prompt taxonomy is behaviourally real —
the model literally generates different-character output for different processing modes.

**Recommended framing for the paper:** Report the full partial correlation table (length, length+NLL,
length+NLL+BLEU). Acknowledge that NLL mediates or confounds the probe-rating link. Reframe the
correlation result as one line of evidence, with the primary claim resting on probe accuracy,
cross-model agreement (r=+0.88–0.94***), and the behavioral NLL gradient — not on the
probe-rating partial correlations alone.

*Per-prompt data: `results/correlation/behavioral_grounding.csv`*