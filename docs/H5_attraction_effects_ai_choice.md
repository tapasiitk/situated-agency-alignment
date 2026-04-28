# ATTRACTION EFFECTS IN AI-RECOMMENDED CHOICE: AI RECOMMENDATIONS AS SOFT DECOYS

*Target venue: Psychological Science / JEP: General / Cognition (multi-study); secondary: Judgment and Decision Making, Journal of Behavioral Decision Making.*

---

## Abstract

A core finding from classical decision research is that adding a dominated decoy to a multi-alternative choice set asymmetrically increases the share of the dominating "target" option — the **attraction effect** (Huber, Payne, & Puto, 1982). Modern agentic AI systems increasingly *recommend* one option in such choice sets, and recommendations are typically modelled as Bayesian evidence whose effect on choice should scale with stated reliability. This paper proposes that AI recommendations function not (only) as Bayesian evidence but as **soft decoys** — context-shaping cues that re-weight attribute valuations and produce attraction-effect-like shifts in choice that exceed what the recommendation's stated reliability would predict. We test this in a 3-experiment programme. Experiment 1 establishes the basic asymmetric-dominance shift induced by AI recommendation in a multi-alternative choice paradigm, against a calibrated Bayesian-update baseline. Experiment 2 manipulates AI **stated reliability** to dissociate the rational and contextual components of the shift. Experiment 3 manipulates **recommendation salience** and **structural similarity** to map the conditions under which the soft-decoy effect is amplified or attenuated. We additionally measure Sense of Agency (SoA) at the single-decision level and predict an interaction in which over-constraining reliability raises choice share for the recommended option but reduces SoA — a single-decision analogue of the inverted-U relationship between Level of Automation and SoA proposed in our companion theoretical work (Rath, in preparation, T1). The paper bridges classical multi-alternative choice research to AI-augmented decision-making and supplies a behavioural foothold for downstream studies of Proxy Agency in extended interactions.

**Keywords:** attraction effect · asymmetric dominance · AI recommendation · multi-alternative choice · Sense of Agency · Proxy Agency · context effects · algorithmic decision support

---

## 1. Background and motivation

### 1.1 The attraction effect as a classical violation of IIA

The attraction effect is one of the most reliably reproduced violations of Luce's Independence of Irrelevant Alternatives (IIA). Given a target option A and competitor B that trade off on two attributes, introducing a third option A′ that is dominated by A on both attributes — but is not dominated by B — *increases* the share of A even though A′ itself is rarely chosen (Huber, Payne, & Puto, 1982; Wedell, 1991; Tversky & Simonson, 1993). Standard accounts attribute the shift to context-dependent attribute weighting: the comparison set restructures how attributes are valued, rather than supplying new evidence about the targets.

### 1.2 AI recommendations as informational versus contextual

In the literature on algorithm-aided decision-making, AI recommendations are typically modelled as **informational** signals: their effect on choice should scale with the user's belief in the AI's accuracy (Logg, Minson, & Moore, 2019; Dietvorst, Simmons, & Massey, 2015). On this view, calibrating the AI's stated reliability should bound the recommendation's effect.

This paper argues that this account is incomplete. AI recommendations are **structural** as well as informational: they restructure the attentional and evaluative context in which the option set is processed. A recommended option becomes the implicit target of comparison — the same role that the dominating option plays in classical attraction-effect designs. We therefore predict an attraction-effect-like shift in choice that exceeds what is rationally predicted from the AI's stated reliability — even after accuracy is calibrated and even when the recommendation contains no new attribute information.

### 1.3 Why this paper bridges the broader research program

The Proxy Agency framework (Rath, in preparation, T1) argues that sufficiently personalised AI preserves Sense of Agency through Semantic Fluency, and that high SoA itself can produce a moral shield in extended interactions. This paper is the **single-decision analogue** of that mechanism: at the level of one choice, an AI recommendation that fits the comparison set fluently shifts both choice and SoA, and the relationship between AI reliability and SoA is non-monotonic. H5 therefore generates a behavioural foothold for the inverted-U prediction (T1 Prediction 1) in a fast, low-cost paradigm, prior to the longitudinal designs (H1, H3) that test it under sustained Proxy Agency.

---

## 2. Hypotheses

### Primary hypotheses

**H1 — The Soft-Decoy Hypothesis.**
*The presence of an AI recommendation increases the choice share of the recommended option beyond what is predicted by a Bayesian-update model calibrated to the AI's stated reliability.* Operationalised as: in a multi-alternative choice with 3–4 options, the *residual* shift in choice probability for the recommended option, after subtracting the model-predicted shift from rational reliability-weighted updating, is significantly greater than zero.

**H2 — The Asymmetric-Dominance Generalisation.**
*The AI recommendation also lifts choice share for options that are structurally adjacent to the recommended option in attribute space — analogously to how a decoy lifts the dominating target in classical designs.* Operationalised as: choice probability for the option asymmetrically dominated-adjacent to the recommended option rises with the recommendation's presence, even when the recommended option itself is removed from the choice set.

**H3 — The Reliability × SoA Interaction.**
*Subjective Sense of Agency at the choice level is non-monotonic in stated reliability.* Specifically: SoA is highest at moderate reliability (e.g., ~60–80%) and *falls* at very high stated reliability (≥95%) even though choice share for the recommended option continues to rise. This is the single-decision analogue of T1's inverted-U.

### Auxiliary hypotheses

**H4 — Salience modulation.** Higher visual or interactional salience of the recommendation amplifies the soft-decoy shift (H1) without proportionally increasing the rational-update component.

**H5 — Similarity moderation.** Soft-decoy shift is largest when the recommended option is *moderately* similar to a competitor in attribute space (the structural condition under which classical attraction effects are strongest), and is reduced when the recommended option is structurally isolated.

---

## 3. Methods

### 3.1 Design

We propose a three-experiment programme.

- **Experiment 1 (N ≈ 240).** Between-subjects, two conditions: *No-recommendation baseline* vs *AI-recommendation present*. Within-subjects, 12 trinary or quaternary choice sets covering canonical attraction-effect geometries (target/competitor/decoy across two attribute dimensions). Outcome: per-option choice probability.
- **Experiment 2 (N ≈ 360).** Between-subjects, three reliability levels: 60%, 80%, 95% stated accuracy. Within-subjects, 12 choice sets identical to Experiment 1. Outcome: choice probability + post-decision SoA report.
- **Experiment 3 (N ≈ 320).** Within-subjects 2×2: salience (low / high) × structural similarity of recommended option to a competitor (isolated / similar). Outcome: choice probability + SoA.

### 3.2 Stimuli

Choice sets are constructed in two-attribute spaces (e.g., price × quality for consumer products; expected return × risk for portfolios; or abstract attribute pairs for non-consumer domains in a robustness experiment). For each set we enumerate canonical geometries: **A–B–A′** (asymmetric dominance), **A–B–C** (no dominance), and **A–B alone** (binary baseline). Salience is manipulated by visual emphasis (highlight, position, font weight) without altering attribute information.

### 3.3 Procedure

Participants complete a brief familiarisation and a calibration block in which they are told the AI's stated accuracy and shown its prior performance (matched across conditions). Each trial: present option set; if AI condition, display recommendation with the calibrated reliability label; participant selects one option; rate Sense of Agency on a 0–100 two-scale instrument adapted from H1 (RT2S — "To what extent did *you* contribute?" / "To what extent did the *AI* contribute?"). Response time is logged.

### 3.4 Reliability calibration and Bayesian baseline

The Bayesian-update baseline is computed per-set per-participant. Given the AI's announced accuracy *p*, the rational expected lift in choice probability for the recommended option is bounded by *p*'s likelihood-ratio shift over a uniform prior across options. Empirical lift is decomposed into rational and residual components; H1 tests the residual.

### 3.5 Sample-size justification

Power analysis (G*Power; Cohen's *d* ≈ 0.25 for the residual lift, two-tailed α = .05, 1 − β = .80) yields a per-cell minimum of ~125. We over-recruit to ~120 per cell to absorb attention-check exclusions (target ~10–15%), giving total N ≈ 240 (Exp 1), 360 (Exp 2), 320 (Exp 3).

### 3.6 Exclusions and quality controls

Pre-registered exclusion rules: failure of two attention checks, response time below 300 ms or above 3 SD of trial mean, choice probability of the dominant option below chance on calibration trials. Sensitivity analyses with and without exclusion will be reported.

---

## 4. Analysis plan

### 4.1 Primary analysis

For H1 (Soft-Decoy): hierarchical logistic regression of choice ~ recommendation × structure × stated-reliability, with random intercepts per participant and per choice set. The critical contrast is the *residual recommendation effect* after partialling out the Bayesian-rational lift component. Reported as both frequentist (β, *p*) and Bayesian (BF₁₀, posterior credible interval) tests. Pre-registered effect-size threshold for H1: residual β > 0 with BF₁₀ > 6.

For H2 (Asymmetric-Dominance Generalisation): repeat the regression on adjacent-option choice probability with the recommended option removed; same model.

For H3 (Reliability × SoA): hierarchical model SoA ~ stated-reliability × choice-was-recommended. Test for *quadratic* effect of reliability on SoA. Pre-registered: significant negative quadratic coefficient.

### 4.2 Secondary analyses

- **Decomposition.** Bradley-Terry / multinomial logit fits to recover attribute-weight changes attributable to the recommendation; complements the regression-based residual.
- **Salience × Similarity.** Within-subjects analysis from Experiment 3.
- **Response time as mechanism cue.** Faster RT under high salience/similarity supports a contextual rather than deliberative pathway.

### 4.3 Robustness

- Domain-generalisation robustness: replicate Experiment 1 in an abstract attribute space, removing real-world prior beliefs.
- Adversarial baseline: include trials where AI deliberately recommends the structurally weakest option, to verify that the soft-decoy effect is recommendation-driven and not target-prominence-driven.

---

## 5. Roadmap and authorship

- **Effort:** ~6–8 months for all three experiments + write-up. Experiment 1 alone is publishable as a standalone paper if needed.
- **Authorship:** First (PhD-derived programme), postdoc PI as senior.
- **Dependencies:** None upstream; benefits H1/I1 by validating the SoA RT2S instrument in a single-shot paradigm.
- **Submission strategy:** *Psychological Science* (short-format) for a single high-impact target study, or *Cognition* / *JEP: General* (multi-study) for the full three-experiment package. *Judgment and Decision Making* is the secondary fallback.

---

## 6. Pre-registration block (OSF / AsPredicted template)

> The block below is intended to be ported with minimal edits into OSF (`https://osf.io/`) or AsPredicted (`https://aspredicted.org/`). Lock before data collection.

### 6.1 Study type
Hypothesis-testing experiment with confirmatory and exploratory components. Pre-registered before data collection.

### 6.2 Hypotheses (confirmatory)
- **H1 (primary):** AI recommendation produces a residual lift in choice probability for the recommended option, beyond the Bayesian-rational baseline implied by stated reliability.
- **H2:** AI recommendation lifts choice probability for the option asymmetrically dominated-adjacent to the recommended option, even when the recommended option is removed.
- **H3:** Sense of Agency is non-monotonic in stated reliability (negative quadratic term significant).

### 6.3 Design
- Experiment 1: between-subjects (recommendation absent vs present); within-subjects (12 choice-set geometries).
- Experiment 2: between-subjects (3 stated reliability levels); within-subjects (12 choice sets) + SoA outcome.
- Experiment 3: within-subjects 2×2 (salience × similarity).

### 6.4 Sampling plan
- Recruitment: Prolific (geographically restricted to UK/US English speakers); pre-registered minimum age 18.
- Stop rule: stop at the *N* derived from §3.5 power analysis, after applying exclusions.
- No interim peeking; no optional stopping.

### 6.5 Variables
- **Independent variables:** recommendation presence; stated reliability (60/80/95%); choice-set geometry (target/decoy/no-dominance); salience (low/high); structural similarity (isolated/similar).
- **Primary DVs:** choice probability per option; post-decision SoA (RT2S 0–100 scales for self vs AI authorship).
- **Secondary DVs:** response time; perceived AI competence (single-item).

### 6.6 Analysis (confirmatory)
- Hierarchical logistic regression (lme4, R) with random intercepts per participant and choice set; both frequentist *p*-values and Bayesian BF₁₀ reported.
- Pre-registered decision rules: H1 supported if residual β > 0 with BF₁₀ > 6 in primary analysis. H3 supported if quadratic coefficient on reliability is negative with BF₁₀ > 6.

### 6.7 Analysis (exploratory, clearly labelled)
- Bradley-Terry attribute-weight decomposition.
- Response-time mediation analysis.
- Domain-generalisation comparison (consumer vs portfolio vs abstract).

### 6.8 Exclusions
- Failure on ≥2 attention checks → exclude.
- Response time <300 ms or >3 SD above per-set mean → exclude trial only.
- Calibration-block target accuracy below chance → exclude participant.
- Sensitivity analyses reported with and without exclusions.

### 6.9 Stopping rule
Stop data collection at the pre-registered *N* per cell (§3.5) after applying exclusions; no optional stopping; no interim analyses.

### 6.10 Inference and reporting
- All primary analyses reported regardless of significance.
- Confidence/credible intervals reported alongside point estimates.
- Pre-registered code and stimuli archived on OSF; deviations from pre-registered plan, if any, flagged in a "deviations" section in the manuscript.

---

*Document version 0.1. To be revised once Experiment 1 stimulus set is finalised; pre-registration locked at that point.*
