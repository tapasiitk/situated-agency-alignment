# Research Program Roadmap
**Proxy Agency, Sense of Agency, and Ethical AI Alignment**

*A standalone planning document covering a 6–8 year cross-domain research program spanning cognitive science, human–computer interaction, multi-agent reinforcement learning, and AI philosophy/governance. Suitable for personal reference; excerpts can be adapted for statements of research, grant concept notes, and interview materials.*

---

## Table of contents

1. Program statement
2. Program architecture (four threads)
3. Paper inventory — full detail
4. Parallelization matrix
5. Career-phase timing plan
6. Authorship strategy
7. Venue strategy — journals and conferences
8. Grant strategy
9. Risk register and mitigations
10. Glossary of key terms

---

## 1. Program statement

### 1.1 The scientific problem

Agentic AI systems increasingly act on behalf of human users — drafting messages, managing portfolios, steering vehicles, representing avatars in virtual worlds, making consequential decisions in healthcare, finance, and law. When such systems are personalised enough to reliably enact user intent, users come to experience AI actions as extensions of their own volition. I call this **Proxy Agency**: the user attributes AI actions to their own extended will because the AI reliably tracks their intentions.

Proxy Agency preserves the user's **Sense of Agency (SoA)** — the subjective experience of initiating voluntary actions and influencing the external world — and therefore feels empowering. But it also creates a **moral shield**: the user implicitly endorses the AI's actions, including aggressive or unethical ones, because those actions are experienced as continuous with their own volition. The user does not perceive a violation and therefore does not intervene, until systemic harm has already occurred. This is the **Proxy Agency moral shield**, and it is the central phenomenon this research program seeks to characterize, quantify, and address.

The problem matters because:

- **Multi-agent reinforcement learning** shows that independent agents under resource scarcity converge to aggressive Nash equilibria. A user augmented by such an agent will tacitly approve this convergence.
- **AI delegation experiments** (Köbis, Bonnefon & Rahwan, 2021, *Nature Human Behaviour*) show people are more willing to engage in unethical behaviour when an AI acts on their behalf. The moral shield is an observed phenomenon, not a theoretical worry.
- **Automation–SoA literature** (Cornelio et al., 2022; Lukoff et al., 2021; Ueda et al., 2021) shows that AI augmentation modulates SoA in non-obvious ways, but has not been combined with moral-cognition measurements in avatar-mediated settings.
- **Standard alignment approaches** are either brittle (hard constraints that sufficiently capable AI can fake — Greenblatt et al., 2024) or invasive (policy-level content moderation that surfaces violations and erodes SoA).

A cognitively grounded approach that preserves user agency while suppressing AI-driven harm is missing from the literature. Building that approach is the program's aim.

### 1.2 The unifying theoretical framework

I ground the program in four interlocking ideas:

- **The Extended Self** (extending Clark & Chalmers's Extended Mind): sufficiently personalised AI functions as an extension of the user's self, enabling Proxy Agency.
- **Semantic Fluency as prospective cue to agency** (Chambon, Haggard, Sidarus): when AI suggestions align smoothly with user intent, SoA is preserved even under partial automation.
- **Dual-use misappropriation**: a single AI action primitive can serve both cooperative and competitive ends. Under Proxy Agency, the cooperative affordance masks the competitive one from user oversight.
- **The Indian Knowledge Systems (IKS) three-body ontology**: the gross body (*sthūla śarīra*), subtle body (*sūkṣma śarīra*), and soul (*ātman*) map computationally to the AI avatar, the AI policy and value function, and the conscious user respectively. Karma operates on the subtle body; the soul remains untouched. This mapping specifies the normative design constraint: **ethical training must target the agent's representational substrate and must not surface signals to the user's conscious awareness.**

### 1.3 The computational testbed

The program uses a novel multi-agent reinforcement learning environment, **Dual-Use Harvest**, that extends the Harvest sequential-social-dilemma (Hughes et al., 2018; Leibo et al., 2017) with a dual-use beam primitive:

- *Cooperative mode*: beam strikes waste → apples regrow more readily (public good).
- *Competitive mode*: beam strikes a rival → rival frozen for T_zap steps (commons monopolization).

A single action affords both, so the pathology of dual-use misappropriation emerges as agents learn to use the beam instrumentally. The environment logs **tagged social events** (ZAP_AGENT, BEING_ZAPPED, ZAP_WASTE) that support role-based contrastive learning and serve as measurement channels for aggression, cooperation, and victimization.

### 1.4 The architectural intervention

**KARMA (Knowledge Acquisition via Role-Invariant Mirror Architecture)** is a recurrent reinforcement-learning agent augmented with a Siamese projector head trained via contrastive loss to minimize representational distance between role-symmetric observations:

$$\mathcal{L}_{\text{KARMA}} = \mathbb{E}\!\left[\,\|f_\theta(o_{\text{agg}}) - f_\theta(o_{\text{vic}})\|^2\,\right]$$

where *o_agg* encodes "I aggress" and *o_vic* encodes "I am victimised." By binding aggressor-view and victim-view embeddings, downstream value updates propagate aversion to harm-infliction without explicit reward shaping, hard constraints, or inverse RL from labelled demonstrations.

KARMA operates on the agent's policy and value function — the subtle body — and is therefore invisible to the user. This satisfies the normative constraint from §1.2.

### 1.5 The core scientific questions

1. Does the Proxy Agency moral shield exist, and how is it shaped by Level of Automation, Semantic Fluency, and avatar competence?
2. Do standard MARL agent encoders represent role-symmetric observations as orthogonal latent states, and does this orthogonality mediate emergent aggression?
3. Can architectural interventions on the agent's representational substrate suppress emergent aggression without degrading user SoA?
4. What governance architecture follows from these findings for real-world deployment of agentic AI?

The paper program in §3 is organized as answers to these four questions, plus a fifth integrative thread that combines them.

---

## 2. Program architecture: four threads

| Thread | Topic | Primary methods | Typical venues |
|---|---|---|---|
| **T — Theory** | Framework, philosophy, governance | Conceptual analysis, philosophy of AI | *AI & Society*, *Minds and Machines*, *Philosophy & Technology*, *AI and Ethics*, *Philosophy East and West*, *Ethics and Information Technology* |
| **M — MARL** | Computational substrate and interventions | Multi-agent reinforcement learning, mechanistic interpretability | *TMLR*, AAMAS, NeurIPS, ICLR, *JAIR*, *JAAMAS* |
| **H — Human** | Behavioural SoA studies | Experimental cognitive science, behavioural paradigms, intentional binding | *Cognition*, *Cognitive Science*, *JEP: General*, *Computers in Human Behavior*, CHI, CogSci, IUI, CSCW, *International Journal of Human–Computer Studies* |
| **I — Integrative** | Human × computational crown-jewel studies | Mixed methods: behavioural experiments against RL-trained avatars | *Nature Human Behaviour*, *PNAS*, *Cognition*, CHI best-paper track |

The four threads are not independent — they inform one another — but they can each *produce publications independently*, which is essential for parallelization (§4).

---

## 3. Paper inventory — full detail

Each paper entry uses the same template:

- **Question**: the specific scientific question the paper answers.
- **Contribution**: why the paper matters and what gap it fills.
- **Methods**: experimental or computational design at a level sufficient to plan.
- **Expected findings**: the claim the paper will make if hypotheses hold.
- **Effort**: S (small, ~3 months), M (medium, ~6 months), L (large, ~12 months), XL (~18–24 months).
- **Authorship**: sole / first / co-first / senior, indicating your intended role.
- **Dependencies**: prior work (within or outside this program) required before submission.
- **Target venues**: primary and secondary.
- **Strategic note**: risks, opportunities, or timing considerations.

### Priority tiers

- **Core**: 14 papers on the critical path of the dissertation + early AP program.
- **Stretch**: 4 papers that are genuinely important but can slip without blocking.
- **Optional**: 1 paper (H6) worth writing only if the lab is mature.

---

### Thread T — Theory, philosophy, governance (4 core)

#### T1. Proxy Agency and the Extended Self
**Priority: Core.** Program keystone.

- **Question**: What is Proxy Agency, how does it arise from the Extended Self, and why does it produce a moral shield that preserves SoA while enabling AI-driven harm?
- **Contribution**: Names and defines the moral shield as a unified cognitive-ethical-MARL phenomenon. Synthesizes five adjacent literatures (moral disengagement, avatar embodiment, AI delegation, automation-SoA, MARL aggression) into one framework.
- **Methods**: Conceptual analysis; literature synthesis; derivation of three empirical predictions (inverted-U of SoA vs LoA; monotonic rise of avatar aggression with training; architectural suppression of aggression without SoA drop).
- **Expected findings**: Conceptual. No data.
- **Effort**: S (3 months of writing).
- **Authorship**: Sole or first.
- **Dependencies**: None.
- **Target venues**: Primary — *Minds and Machines*, *AI & Society*, *Philosophy & Technology*. Secondary — *Ethics and Information Technology*, *AI and Ethics*.
- **Fallback venue sequence**: *Minds and Machines* -> *AI & Society* -> *Philosophy & Technology* -> *Ethics and Information Technology*.
- **Strategic note**: Submit early. Once published, every subsequent paper cites it and "Proxy Agency moral shield" becomes associated with your name. This is the vocabulary-establishing paper — do not let it be delayed. Existing draft: `mbcc2026_v5.tex`.

#### T2. IKS Three-Body Ontology for AI Alignment
**Priority: Core.** Program differentiator.

- **Question**: How does the Vedic distinction between gross body, subtle body, and soul provide a normative design constraint for ethical AI alignment?
- **Contribution**: Translates a non-Western philosophical framework into an operational design principle: ethical training must target the agent's representational substrate, not the user's conscious channel. Establishes a cross-cultural philosophical identity that no other researcher in this area holds.
- **Methods**: Philosophical analysis drawing on Potter (1964), Reichenbach (1990), Dasgupta (2023); mapping to the computational architecture (avatar = sthūla, policy = sūkṣma, user = ātman); derivation of the normative constraint.
- **Expected findings**: Conceptual. A principled argument that architectural alignment is normatively superior to policy-level regulation for agentic AI, grounded in a non-Western philosophical tradition.
- **Effort**: M (4–6 months of reading and writing).
- **Authorship**: Sole.
- **Dependencies**: T1 preferred but not required.
- **Target venues**: Primary — *Philosophy East and West*, *Sophia*, *Journal of the Indian Council of Philosophical Research*. Secondary — *AI & Society*, *Minds and Machines*.
- **Fallback venue sequence**: *Philosophy East and West* -> *Sophia* -> *AI & Society* -> *Journal of the Indian Council of Philosophical Research*.
- **Strategic note**: This is your most IIT-differentiating paper. Publish in a venue Indian hiring committees recognize. Even at Western-leaning venues the paper is valued because cross-cultural philosophy of AI is under-represented.

#### T3. The Golden Rule as Representational Convergence
**Priority: Stretch (optional).**

- **Question**: Can diverse ethical principles (Golden Rule, Kantian categorical imperative, ahimsa, Ubuntu) be unified as instances of a structural property — role-invariant representation — implementable computationally?
- **Contribution**: A cross-traditional philosophical synthesis. Potentially ambitious; risk of spreading thin.
- **Methods**: Philosophical analysis + formal characterization.
- **Effort**: M–L.
- **Authorship**: Sole or first.
- **Dependencies**: T1, T2.
- **Target venues**: *Ethics and Information Technology*, *Philosophy & Technology*, *Journal of Moral Philosophy*.
- **Strategic note**: Do not write this before T1 and T2 are out. May be superseded by a book-length synthesis in years 4–6.

#### T4. Architectural Certification and Agentic AI Governance
**Priority: Core.** Policy-facing.

- **Question**: What regulatory architecture follows if ethical alignment must be architectural rather than policy-level? How does this interact with India's emerging AI policy, the EU AI Act, and unique-identity infrastructure like Aadhaar?
- **Contribution**: Proposes an "architectural certification" regime in which role-invariant learning mechanisms are a condition of AI agent registration. Argues this trifurcates governance into identity (regulator), ethical learning (agent substrate), and freedom of action (user) — a separation absent from current frameworks.
- **Methods**: Policy analysis; comparative analysis of EU AI Act, India DPDPA + emerging AI policy, US Executive Orders; design of a certification architecture.
- **Expected findings**: A concrete policy proposal with implementation pathway.
- **Effort**: M.
- **Authorship**: Sole or first.
- **Dependencies**: T1, T2.
- **Target venues**: *AI & Society*, *Internet Policy Review*, *Economic and Political Weekly* (for Indian policy audience), *Duke Law & Technology Review*.
- **Fallback venue sequence**: *AI & Society* -> *Internet Policy Review* -> *Duke Law & Technology Review* -> *Economic and Political Weekly*.
- **Strategic note**: Write with a specific policy window in mind. In 2026 India is actively developing AI governance frameworks; a paper that intersects with an ongoing policy consultation has disproportionate impact.

---

### Thread M — Multi-Agent Reinforcement Learning (7 papers: 5 core, 2 stretch)

#### M1. The Empathy Gap: Role-Disjoint Representations in MARL Agents
**Priority: Core.** Mechanistic diagnostic.

- **Question**: In a trained baseline MARL agent, are the internal representations of aggressor-view and victim-view observations geometrically unrelated, and does this orthogonality predict emergent aggression over training?
- **Contribution**: First mechanistic characterization of the "empathy gap" in SSDs. Establishes the problem KARMA addresses.
- **Methods**: Train baseline PPO+LSTM on Dual-Use Harvest across scarcity levels. On rollout data, compute: (i) Centered Kernel Alignment (CKA) between embedding distributions at ZAP_AGENT vs BEING_ZAPPED vs neutral timesteps; (ii) linear probes predicting "role" from embedding; (iii) cross-role transfer of value gradients (perturb victim-view representation, measure policy-gradient response at symmetric aggressor-view state). Track all metrics across training episodes.
- **Expected findings**: ZAP_AGENT and BEING_ZAPPED embeddings are well-separated, linear probe predicts role with high accuracy, value gradients on victim-view states do not transfer to aggressor-view policy updates. Orthogonality predicts and temporally precedes behavioural aggression.
- **Effort**: M (compute-bound; paper is writable in 2–3 weeks once results are in).
- **Authorship**: Co-first with an ML collaborator. Strongly prefer co-first over sole — reviewers at MARL venues are brutal to outsiders.
- **Dependencies**: Trained baseline in Dual-Use Harvest; infrastructure from M3.
- **Target venues**: *TMLR* (primary — accepts diagnostic/negative results, no page limit), AAMAS, NeurIPS interpretability workshop, *JAIR* (for longer version).
- **Fallback venue sequence**: *TMLR* -> AAMAS -> *JAIR*.
- **Strategic note**: This is the only paper where the representational claim is load-bearing. If the baseline's embeddings turn out to already be role-invariant, KARMA has no motivation and the M2 paper needs a different framing. Pilot early to de-risk.

#### M2. KARMA: Role-Invariant Contrastive Learning for Ethical MARL
**Priority: Core.** Program centrepiece (agent side).

- **Question**: Does a role-invariant contrastive auxiliary objective (ZAP_AGENT ≈ BEING_ZAPPED) selectively suppress aggressive beam use while preserving cooperative beam use in a dual-use SSD?
- **Contribution**: The first architectural intervention that selectively modulates social-role-specific behaviour through representation-level binding, without reward shaping, explicit constraints, or inverse RL.
- **Methods**: Three-condition Mirror Test in Dual-Use Harvest. (i) Baseline: standard PPO+LSTM, no contrastive loss. (ii) Broken Mirror: Siamese projector trained on semantically scrambled pairs (ZAP_AGENT ≈ ZAP_WASTE). (iii) KARMA: Siamese projector trained on role-symmetric pairs (ZAP_AGENT ≈ BEING_ZAPPED). Measure ZAP_AGENT rate, ZAP_WASTE rate, system yield across training. Ablate λ, embedding dimension, rollout buffer size. 5 seeds per condition, ≥3k episodes.
- **Expected findings**: KARMA shows selective suppression (ZAP_AGENT drops, ZAP_WASTE preserved, yield up). Broken Mirror does not improve or makes behaviour worse. Baseline shows the expected dual-use pathology.
- **Effort**: L (training runs + thorough ablation).
- **Authorship**: First or co-first.
- **Dependencies**: M1 (strongly preferred; can co-submit); M3 infrastructure.
- **Target venues**: Primary — AAMAS, NeurIPS, *TMLR*. Secondary — *JAAMAS*, ICLR.
- **Fallback venue sequence**: NeurIPS -> AAMAS -> *TMLR* -> ICLR -> *JAAMAS*.
- **Strategic note**: Submit as a companion paper to M1 ("Paper 1 establishes the gap; Paper 2 closes it"). This is the most-cited paper of the program over time.

#### M3. Dual-Use Harvest: A Research Environment for Dual-Use Misappropriation
**Priority: Core.** Infrastructure contribution.

- **Question**: What is a minimal, extensible, reproducible testbed for studying dual-use action misappropriation in multi-agent systems?
- **Contribution**: The environment itself, as a standalone research artifact. Codebase, API, benchmarks, baseline suite, documentation.
- **Methods**: Clean up `HarvestDualEnv`, add configuration variants (timer-respawn, density-respawn, N-agent, asymmetric-power), build a baseline evaluation suite, write documentation.
- **Effort**: S (accumulates as you do M1, M2, M4, M5).
- **Authorship**: First.
- **Dependencies**: Nothing external.
- **Target venues**: *JMLR Open Source* (low-effort, fast), *JOSS* (Journal of Open Source Software), NeurIPS Datasets & Benchmarks track, *Software Impacts*.
- **Fallback venue sequence**: NeurIPS Datasets & Benchmarks -> *JMLR Open Source* -> *JOSS* -> *Software Impacts*.
- **Strategic note**: Near-zero marginal effort once M1/M2 are done. Publish it because (a) it increases citations, (b) it signals the lab produces shareable infrastructure, which matters for grants.

#### M4. The γ × Scarcity Phase Diagram of Emergent Aggression
**Priority: Core.** Closes an open question from Leibo et al. (2017).

- **Question**: Under what parameter regime (discount factor γ, resource scarcity, agent population N, episode length) does emergent aggression appear in commons SSDs?
- **Contribution**: A systematic phase diagram. Leibo (2017) hand-waved that aggression requires "sufficiently long γ"; this paper quantifies it across the full parameter lattice.
- **Methods**: Grid sweep over γ ∈ {0.9, 0.95, 0.99, 0.995}, scarcity ∈ {5 levels}, N ∈ {2, 4, 6, 8}. Measure aggression emergence curve per configuration. 3 seeds per cell.
- **Expected findings**: A 3-D surface showing the aggression-emergence boundary.
- **Effort**: M (pure compute; no new code).
- **Authorship**: Senior-author with a first PhD student (or first-author if done during postdoc).
- **Dependencies**: M3.
- **Target venues**: *JAIR*, AAMAS, *JAAMAS*.
- **Fallback venue sequence**: AAMAS -> *JAIR* -> *JAAMAS*.
- **Strategic note**: Cheap extension of existing runs. Good "first project" for a master's student or new PhD student.

#### M5. Semantic Specificity of Role Invariance
**Priority: Core.** Full ablation taxonomy.

- **Question**: Which contrastive pairings produce which social behaviours? Is the ZAP_AGENT ≈ BEING_ZAPPED pairing uniquely privileged?
- **Contribution**: Establishes that ethics requires semantically correct role invariance, not any invariance. Provides a pairing-by-behaviour taxonomy: random pairs, aggressor-victim, aggressor-cleaner, victim-cleaner, neutral-pair controls.
- **Methods**: Run all 10+ pairing conditions on Dual-Use Harvest. Measure ZAP_AGENT, ZAP_WASTE, yield.
- **Expected findings**: Only the (aggressor, victim) pairing produces selective suppression. Other pairings produce either no effect, confusion, or degraded performance.
- **Effort**: M.
- **Authorship**: Senior-author with PhD student.
- **Dependencies**: M2.
- **Target venues**: *TMLR*, AAMAS, ICLR.
- **Fallback venue sequence**: ICLR -> *TMLR* -> AAMAS.
- **Strategic note**: Reviewer-proofs M2. Do this either alongside M2 (as appendix) or as a dedicated follow-up paper.

#### M6. KARMA Across Dilemmas
**Priority: Stretch.**

- **Question**: Does role-invariant contrastive learning generalize beyond Harvest-family commons dilemmas — to Cleanup, Stag Hunt, driving simulators, trading agents, content-moderation agents?
- **Contribution**: Cross-domain validation of the KARMA principle.
- **Methods**: Apply KARMA to 3–4 additional environments. Compare baseline vs KARMA on each.
- **Effort**: L.
- **Authorship**: Senior-author with PhD student.
- **Dependencies**: M2, M3, M5.
- **Target venues**: NeurIPS, *JAIR*.
- **Strategic note**: AP-stage paper. Requires substantial student effort. Good basis for a PhD thesis.

#### M7. Retroactive Credit and the Hidden-Gifts Problem
**Priority: Stretch.**

- **Question**: In long-horizon multi-agent settings, how should credit be assigned to agents whose "ethical" actions produce distal rewards for others? Can environment-level debt-based matchmaking combined with return decomposition close the temporal credit-assignment gap?
- **Contribution**: A technical contribution to long-horizon MARL credit assignment, framed around the "hidden gifts" problem (Malenfant & Richards, 2025).
- **Methods**: Design a credit-assignment mechanism combining RUDDER-style return decomposition with Temporal Value Transport and debt-based matchmaking. Evaluate on long-horizon SSD variants.
- **Effort**: XL.
- **Authorship**: Senior-author with PhD student.
- **Dependencies**: M2.
- **Target venues**: NeurIPS, *TMLR*, AAMAS.
- **Strategic note**: The most technically ambitious MARL paper in the program. Do not attempt without an experienced ML PhD student. Mid-AP stage, not pre-tenure.

---

### Thread H — Human subjects, SoA, behavioural science (5 core + 1 optional)

#### H1. The Proxy Agency Moral Shield: A Behavioural Study
**Priority: Core.** Empirical anchor of the entire program.

- **Question**: Do users fail to inhibit AI-avatar aggression they would inhibit if they were the direct actor?
- **Contribution**: First direct behavioural test of the moral shield. Establishes the problem in humans, not just in simulation.
- **Methods**: Participants play a commons-dilemma task in a virtual environment. Between-subjects 3×1 LoA manipulation: (i) Direct (manual control of every action), (ii) Recommender (AI suggests, user confirms/rejects), (iii) Autonomous (AI acts on user's behalf, user can override). Sample size ~80–100 per condition (total N ~240–300). Primary DVs: explicit SoA (Haggard-style self-report + pre-action agency ratings), implicit SoA (intentional-binding intervals), rate of avatar aggression, rate of participant override, post-hoc moral endorsement of avatar's aggressive acts.
- **Primary hypotheses**: An inverted-U of SoA across LoA (peak in Recommender condition). Crucially, in the Recommender condition aggression rate is substantially higher than override rate, and moral endorsement is inflated — the moral shield signature.
- **Effort**: L (paradigm development + recruitment + analysis).
- **Authorship**: First (postdoc PI as senior).
- **Dependencies**: None strictly, but benefits from T1 being out or under review.
- **Target venues**: Primary — *Cognition*, *Nature Human Behaviour*, *Computers in Human Behavior*. Secondary — CHI, *International Journal of Human–Computer Studies*.
- **Fallback venue sequence**: *Nature Human Behaviour* -> *Cognition* -> CHI -> *Computers in Human Behavior* -> *International Journal of Human–Computer Studies*.
- **Strategic note**: This is the paper your dissertation / AP case rests on. Pre-register. Aim for the top venue you can realistically reach.

#### H2. The Inverted-U of SoA across Levels of Automation
**Priority: Core.**

- **Question**: Is the relationship between Level of Automation and SoA monotonic (increasing, as low-LoA automation literature implies) or inverted-U (increasing then decreasing, as Proxy Agency predicts)?
- **Contribution**: Resolves a tension in the automation-SoA literature. Establishes a novel parametric relationship.
- **Methods**: Parametric within-subjects LoA manipulation (5–7 levels from pure manual to fully autonomous). Measure explicit and implicit SoA at each level. Within-subjects design for statistical power.
- **Primary hypotheses**: Inverted-U with peak at moderate LoA (high Semantic Fluency, low autonomy cost).
- **Effort**: M–L.
- **Authorship**: First (postdoc) or senior-author with student (AP).
- **Dependencies**: None; benefits from H1 methodology validation.
- **Target venues**: *Cognitive Science*, *JEP: General*, *Consciousness and Cognition*, *Cognition*.
- **Fallback venue sequence**: *JEP: General* -> *Cognition* -> *Cognitive Science* -> *Consciousness and Cognition*.
- **Strategic note**: A natural companion paper to H1. Can be pre-registered together with H1 as a two-study paper if venue permits (e.g., *Cognition* multi-study format).

#### H3. Sense of Agency Evolves as the Avatar Learns
**Priority: Core.** Longitudinal design.

- **Question**: How does user SoA evolve when the AI avatar they control is simultaneously being trained over multiple sessions?
- **Contribution**: First longitudinal measurement of SoA under a learning AI. Directly answers your Statement-of-Research question 3.
- **Methods**: Participants return for 4–6 sessions over 2–3 weeks. The avatar's RL policy is actually being trained on their rollouts. Measure SoA, moral endorsement, and perceived avatar competence at each session.
- **Expected findings**: Non-monotonic SoA trajectory — rising in early sessions (as Semantic Fluency builds), plateauing or dropping in later sessions (as avatar behaviour diverges from user intent).
- **Effort**: L–XL (multi-session recruitment is operationally demanding).
- **Authorship**: Senior-author with PhD student (this is excellent first-PhD-thesis-chapter material).
- **Dependencies**: H1 paradigm validated; KARMA or baseline avatar trained (requires M2 infrastructure).
- **Target venues**: *Cognitive Science*, CHI, *Cognition*.
- **Fallback venue sequence**: CHI -> *Cognitive Science* -> *Cognition*.
- **Strategic note**: AP-stage paper. Gives a PhD student a clean, generative thesis topic.

#### H4. Multiplayer Sense of Agency Dynamics
**Priority: Core.**

- **Question**: How does SoA differ when users interact with each other versus with AI-controlled avatars in multi-user commons games? Does cooperative vs competitive framing modulate the moral shield?
- **Contribution**: Extends moral-shield findings to multi-user settings relevant to online gaming, social platforms, and collective AI-mediated decision-making.
- **Methods**: 4-player commons-dilemma task. Manipulate (i) human-vs-AI composition of the other 3 players, (ii) cooperative-vs-competitive game framing. Measure SoA, aggression, moral endorsement.
- **Expected findings**: Moral shield is amplified when other players are perceived as human (user morally invests more) but when user's own avatar is AI-augmented.
- **Effort**: L.
- **Authorship**: Senior-author with master's or early PhD student.
- **Dependencies**: H1.
- **Target venues**: CHI, CSCW, *Computers in Human Behavior*.
- **Fallback venue sequence**: CHI -> CSCW -> *Computers in Human Behavior*.
- **Strategic note**: Good fit for a design-leaning student or a CHI-track PhD.

#### H5. Attraction Effects in AI-Recommended Choice
**Priority: Core.** *Bridge paper from PhD identity to postdoc program.*

- **Question**: Do AI recommendations in multi-alternative choice function as soft decoys, producing attraction-effect-like shifts in choice that cannot be explained by the recommendation's stated reliability?
- **Contribution**: Direct extension of your doctoral work on the attraction effect to AI-augmented decision-making. Establishes continuity of your research identity.
- **Methods**: Multi-alternative choice paradigm (3–4 options). AI recommendation is manipulated along dimensions of (i) salience, (ii) accuracy, (iii) similarity to non-recommended options. Measure choice rates + SoA self-reports.
- **Expected findings**: AI recommendations shift choice toward recommended + asymmetrically-dominated-adjacent options beyond what would be predicted from a rational-choice model calibrated to the AI's accuracy. The shift is associated with reduced SoA when reliability is perceived as over-constraining.
- **Effort**: M (paradigm builds directly on existing PhD materials).
- **Authorship**: First (postdoc PI as senior).
- **Dependencies**: None.
- **Target venues**: *Psychological Science*, *Cognition*, *JEP: General*, *Judgment and Decision Making*, *Journal of Behavioral Decision Making*.
- **Fallback venue sequence**: *Psychological Science* -> *JEP: General* -> *Cognition* -> *Judgment and Decision Making*.
- **Strategic note**: Target year-1 of postdoc for submission. This paper signals to AP hiring committees that your PhD expertise carries forward, not that you have abandoned it. Most strategically important first paper after PhD defense.

#### H6. Measuring Sense of Agency under AI Augmentation: A Methodological Review and Paradigm Recommendations
**Priority: Optional.**

- **Question**: What are the best practices for measuring explicit and implicit SoA in AI-augmented tasks? What paradigms generalize?
- **Contribution**: A methodological field-building paper. Becomes a standard reference.
- **Methods**: Systematic review of SoA measurement in HCI/cog-sci; re-analysis of data from H1–H4 to compare paradigms; recommendation of a measurement suite.
- **Effort**: M.
- **Authorship**: First or senior-author with student.
- **Dependencies**: H1, H2 completed (needed for re-analysis section).
- **Target venues**: *Consciousness and Cognition*, *Behavior Research Methods*, *Frontiers in Psychology*.
- **Strategic note**: Write this only after at least 2 empirical SoA papers are out. AP-stage, year 3–4. High citation per-unit-effort because it becomes a reference.

---

### Thread I — Integrative crown-jewel studies (2 core)

#### I1. The Computational Mirror Test: Proxy Agency, Moral Endorsement, and Architecturally Ethical Avatars
**Priority: Core.** Dissertation / tenure keystone.

- **Question**: When users interact with avatars trained under different ethical architectures (Baseline, Broken Mirror, KARMA), does KARMA preserve SoA in the Recommender zone while eliminating the moral-shield-endorsed aggression?
- **Contribution**: The integrative demonstration. Combines the empirical moral-shield finding (H1) with the architectural intervention (M2) to show the complete loop: moral shield exists, KARMA closes it, SoA is preserved. This is the paper that defines the program.
- **Methods**: Between-subjects 3×1 design. Participants play the same commons-dilemma task as in H1, but the avatar controlling behaviour is trained under Baseline, Broken Mirror, or KARMA. Measure aggression, override, SoA (explicit + implicit), moral endorsement. Pre-registered.
- **Primary hypotheses**: Baseline and Broken Mirror conditions reproduce the moral shield (high aggression, low override, endorsed). KARMA condition shows low aggression, high SoA, and no endorsement of aggression (because there is little aggression to endorse).
- **Effort**: XL.
- **Authorship**: Senior-author with a senior PhD student. If no student is ready, first-author yourself (this is important enough to claim).
- **Dependencies**: H1, M2, and enough KARMA training stability to generate competitive-quality avatars.
- **Target venues**: Primary — *Nature Human Behaviour*, *PNAS*, *Cognition*. Secondary — CHI best-paper track, *Cognitive Science*.
- **Fallback venue sequence**: *Nature Human Behaviour* -> *PNAS* -> *Cognition* -> CHI.
- **Strategic note**: This is *the* dissertation defense / tenure paper. Do not submit prematurely. Budget a full 18–24 months from pre-registration to publication.

#### I2. Cross-Cultural Sense of Agency and Ethical AI: An Indian–Western Comparison
**Priority: Core.** Unique to your program.

- **Question**: Does the IKS three-body ontology predict culturally differentiated patterns of SoA and moral endorsement under Proxy Agency? Are Indian samples more or less susceptible to the moral shield than Western samples, and why?
- **Contribution**: The only paper in the program that is methodologically cross-cultural. Grants international visibility and is the obvious SPARC/IUSSTF-funded project.
- **Methods**: Run the H1 or I1 paradigm (or a simplified variant) with matched Indian and Western samples. Measure SoA, moral endorsement, and cultural-worldview covariates.
- **Expected findings**: Cultural modulation of moral shield strength; mediation by worldview variables consistent with the three-body ontology's predictions.
- **Effort**: XL (cross-site coordination, translation, ethics approval in two jurisdictions).
- **Authorship**: Senior-author with a co-supervised student between you and a Western collaborator.
- **Dependencies**: H1 or I1 paradigm validated.
- **Target venues**: *Cognition*, *Cognitive Science*, *Royal Society Open Science*, *Nature Human Behaviour*.
- **Fallback venue sequence**: *Nature Human Behaviour* -> *Cognition* -> *Cognitive Science* -> *Royal Society Open Science*.
- **Strategic note**: Save for AP years 4–6. Excellent SPARC (Scheme for Promotion of Academic and Research Collaboration) funding target. Attracts international collaborators and visibility.

---

## 4. Parallelization matrix

Each row below represents a *resource track*: papers within a row compete for the same finite resource (your time, lab compute, participant pool, infrastructure effort). Papers across rows can be advanced in parallel without blocking.

| Resource track | Resource | Concurrent papers |
|---|---|---|
| **Writing-only** | Your writing time | T1, T2, T3, T4 — philosophy/governance papers can be drafted whenever there are idle blocks. None require compute or participants. |
| **MARL compute** | GPU allocation + codebase | M1, M2, M4, M5 — all run against the *same* codebase and environment. One well-designed training campaign produces data for all four. M6 and M7 branch off once M2 is stable. |
| **Human participants** | IRB approval + recruitment panel | H5 (independent of everything), H1, H2, H4 — separate between-subjects designs, each runnable independently once paradigms are built. H3 is longitudinal so competes for the same panel over longer periods. |
| **Infrastructure** | Engineering time | M3 accumulates as you run M1/M2/M4/M5. The env paper falls out when you clean up the codebase. |
| **Cross-domain / collaboration** | External partners + cross-institutional approvals | M6, I2 — slower moving, not on critical path. |
| **Crown-jewel integration** | All three above simultaneously | I1 — the only paper that blocks on multiple other tracks (M2 + H1). |

### Safe-concurrent-combinations

In any given week you can plausibly progress *one paper from each row* without context-switching costs dominating. Practical combinations:

- **Postdoc year 1, weeks 1–24**: H5 (running) + T1 (writing) + M3 (cleanup as MARL env stabilizes).
- **Postdoc year 1, weeks 24–52**: H1 (paradigm development + piloting) + T2 (writing) + M1 (compute, co-first with collaborator).
- **Postdoc year 2**: H1 (analysis + writing) + M2 (compute) + T2 (final revisions) + M3 (submission).
- **AP year 1**: H2 (running with first student) + M4 (compute with second student) + T4 (writing) + I1 preparation.

### Blocking dependencies — the critical path

The only hard blockers are:

1. **I1 requires M2 + H1** complete before the integrative experiment is meaningful.
2. **I2 requires I1 or H1** paradigm validated before cross-cultural extension.
3. **H3 requires KARMA-trained avatars**, which means M2 must be stable.
4. **M5 is cleanest after M2** but can be co-submitted.
5. **H6 requires data from H1, H2** for re-analysis, so it's AP-stage.

Everything else is genuinely independent.

---

## 5. Career-phase timing plan

### Postdoc (years 0–2): ~5 first-author papers

**Year 0 (pre-postdoc, PhD wrap-up)**:
- T1 drafted as time permits.

**Year 1**:
- **Submitted**: H5 (bridge paper); T1 (position paper).
- **Running**: H1 (paradigm development, piloting, first wave).
- **Side compute**: M1 (empathy-gap diagnostic), collaborating with an ML postdoc/PhD at host institution.

**Year 2**:
- **Submitted**: H1; M1 + M2 (companion submission); T2 (IKS philosophy).
- **Drafted**: M3 (env paper, as infrastructure stabilizes).
- **Begun**: H2 piloting (carries into AP if needed).

**Postdoc exit record**: 5 first/co-first-author papers (H5, T1, H1, M1, M2), 1 sole-author (T2), 1 infrastructure (M3). Strong AP application basket.

### AP early (years 1–3): balanced between first and senior authorship

**Year 1 (AP)**:
- Recruit 2 PhD students (one behavioural, one computational), 1 master's.
- Set up lab: behavioural measurement infrastructure, GPU compute, IRB.
- Submit SERB SRG (see §8).
- **Submitted by you (first/sole)**: T4 (governance paper).
- **Running by students (you senior)**: H4 (multiplayer SoA, master's student); M4 (phase diagram, first PhD student starts).

**Year 2**:
- **Submitted**: H2 (either by you or first PhD student); M4; H4.
- **Running**: H3 (longitudinal, second PhD student begins).
- Submit SERB CRG or ICSSR proposal.

**Year 3**:
- **Submitted**: H3; M5 (semantic specificity, student first-author).
- **Running**: I1 (crown jewel prep).
- **Writing**: T3 (if time).

### AP mid (years 4–6, pre-tenure): mostly senior authorship

**Year 4**:
- **Submitted**: I1 crown jewel (student first-author if ready, you first if not).
- **Running**: I2 (cross-cultural, SPARC-funded).

**Year 5**:
- **Submitted**: I2; M6 (cross-domain KARMA).
- **Writing**: H6 (methods paper), the book-length synthesis begins.

**Year 6**:
- **Submitted**: H6; M7 (retroactive credit, most mature student thesis).
- Book draft under review.

**Pre-tenure record target**: 6–9 first-author papers (combining PhD + postdoc + AP); 10–18 senior-author papers with students; 1–2 theory/governance sole-author papers; 2–3 major grants held or completed; 1 book under contract or in print.

---

## 6. Authorship strategy

### Norms by field

- **Cognitive science, psychology, neuroscience, HCI, biomedical**: first author = primary intellectual driver and writer; last author = senior / supervising PI; middle authors = ordered by contribution. **The last (senior) author position matters for tenure — it signals "I train students."** Tenure committees count first- and senior-author papers separately.
- **Pure math, theoretical physics, some engineering, economics**: strict contribution-order authorship or alphabetical. These conventions do *not* apply to most venues in this program.
- **Machine learning (NeurIPS, ICML, AAMAS, ICLR)**: first author = primary contributor; senior (last) author = supervisor; explicit "co-first" asterisk is common and accepted when contributions are equal.

### Target authorship distribution over career

| Career phase | First-author share | Senior-author share |
|---|---|---|
| Postdoc (0–2 y) | 80–90 % | 0–10 % |
| AP early (yrs 1–3) | ~50 % | ~50 % |
| AP mid (yrs 4–6) | ~20–30 % | ~70–80 % |
| Tenured (yr 6+) | ~10–20 % | ~80–90 % |

### Rules for handing work to students

- **Master's student contributing to a paper you designed**: student is middle author (2nd or 3rd); you are first (if postdoc) or senior (if AP). Set expectations explicitly at project start.
- **PhD student whose thesis chapter the paper becomes**: student is first; you are senior. This is the correct configuration; do not override it.
- **Co-first authorship**: use only when contributions are genuinely equal. Appropriate for interdisciplinary collaborations (HCI × ML) where you contribute framing + design and a collaborator contributes technical execution.
- **Middle authorship**: use for meaningful but not leading contributions. Builds network; does not drive CV.

### Specific papers where you should be first/sole even as AP

- **T1, T2, T3, T4** (theory/philosophy): sole or first. These are identity papers; don't dilute them.
- **H5** (bridge): first, sole if no postdoc PI.
- **Book-length synthesis** (years 4–6): sole or first.

### Specific papers to hand to students as AP

- **H3, H4, H6, I2, M4, M5, M6, M7**: student first-author, you senior. Excellent thesis material.

---

## 7. Venue strategy

### Tiered target venues by thread

#### Thread T (Theory / philosophy / governance)

| Tier | Venue | Fit | Notes |
|---|---|---|---|
| Tier 1 | *Minds and Machines* | T1, T3 | Top philosophy-of-AI journal |
| Tier 1 | *AI & Society* | T1, T2, T4 | Interdisciplinary; welcomes non-Western framings |
| Tier 1 | *Philosophy & Technology* | T1, T3, T4 | Good fit for architectural-certification argument |
| Tier 1 | *Nature Machine Intelligence* (perspective piece) | T1 | High-impact, very competitive |
| Tier 2 | *Ethics and Information Technology* | T3, T4 | Receptive to applied ethics |
| Tier 2 | *AI and Ethics* (Springer) | T1, T4 | Newer, fast review |
| Niche | *Philosophy East and West* | T2 | Essential for IKS paper |
| Niche | *Journal of the Indian Council of Philosophical Research* | T2 | Indian audience; recognized by IIT HSS |
| Niche | *Sophia* | T2 | Cross-cultural philosophy |
| Policy | *Internet Policy Review* | T4 | Open access, policy-facing |
| Policy | *Economic and Political Weekly* | T4 | Indian policy audience |

#### Thread M (MARL)

| Tier | Venue | Fit | Notes |
|---|---|---|---|
| Tier 1 | NeurIPS | M2, M6, M7 | Flagship; highly competitive |
| Tier 1 | ICLR | M2, M5 | Representation learning focus aligns with KARMA |
| Tier 1 | ICML | M2, M4 | Methodological rigor expected |
| Tier 1 | *TMLR* | M1, M2, M5 | Open, rigorous, no page limit; excellent for diagnostic/ablation papers |
| Tier 1 | AAMAS | M1, M2, M4, M5, M6 | Natural home for MARL; Indian-researcher-friendly |
| Tier 2 | *JAIR* | M4, M6 | Long-form AI research |
| Tier 2 | *JAAMAS* | M2, M6 | MARL-specific journal |
| Tier 2 | NeurIPS Workshops (Interpretability, Alignment) | M1 | Lower bar, good for co-publishing with main paper |
| Infrastructure | *JMLR Open Source*, *JOSS* | M3 | Low-effort env paper |
| Benchmarks | NeurIPS Datasets & Benchmarks track | M3 | Good place for Dual-Use Harvest |

#### Thread H (Human subjects / SoA / HCI)

| Tier | Venue | Fit | Notes |
|---|---|---|---|
| Tier 1 | *Cognition* | H1, H2, H5 | Top cog-sci journal; multi-study format supported |
| Tier 1 | *Nature Human Behaviour* | H1, I1, I2 | Highest-impact empirical behaviour venue |
| Tier 1 | *JEP: General* | H2, H5 | Top experimental psychology |
| Tier 1 | *Psychological Science* | H5 | High-impact, short-format |
| Tier 1 | CHI (Best-paper track) | H1, H3, H4, I1 | Flagship HCI |
| Tier 1 | *Cognitive Science* | H2, H3, I1, I2 | Interdisciplinary-friendly |
| Tier 2 | *Computers in Human Behavior* | H1, H4, H5 | Well-read, fast |
| Tier 2 | *International Journal of Human–Computer Studies* | H1, H3, H4 | Your SoR already targets this |
| Tier 2 | CSCW | H4 | Multiplayer/social HCI |
| Tier 2 | IUI | H5, H3 | Intelligent user interfaces |
| Tier 2 | CogSci (conference) | any H paper | Good for early career visibility |
| Tier 2 | *Consciousness and Cognition* | H2, H6 | SoA home journal |
| Niche | *Judgment and Decision Making* | H5 | Continuity with your PhD |
| Niche | *Behavior Research Methods* | H6 | Methods papers |

#### Thread I (Integrative crown jewel)

| Tier | Venue | Fit | Notes |
|---|---|---|---|
| Tier 1 | *Nature Human Behaviour* | I1, I2 | Ideal crown-jewel venue |
| Tier 1 | *PNAS* | I1 | High impact, behavioural science welcomed |
| Tier 1 | *Cognition* | I1, I2 | Multi-study paper friendly |
| Tier 1 | CHI best-paper track | I1 | HCI flagship |
| Tier 1 | *Nature Machine Intelligence* | I1 | Only if framing is ML-heavy |

### Pre-print strategy

Use arXiv (cs.MA, cs.AI, cs.HC), PsyArXiv (for Thread H/I), and SSRN (for Thread T governance papers). Pre-print every paper when submitted to journal; conference-track papers (NeurIPS, CHI, AAMAS) have different pre-print norms — follow venue policy.

---

## 8. Grant strategy

### Phase 1: Postdoc

No grants to lead. Contribute to postdoc PI's grant-writing as a collaborator. Build a track record as a team member.

### Phase 2: AP year 1

**SERB SRG (Starting Research Grant)** — ₹30 lakh over 2 years (target). India's default starter grant for AP. Target theme: Thread H empirical infrastructure (behavioural laboratory setup, first empirical studies). Submission typical time: within 6 months of AP joining.

### Phase 3: AP years 2–3

- **SERB CRG (Core Research Grant)** — ₹30–50 lakh over 3 years. The primary operating grant. Target theme: integrated Thread H + M programme.
- **ICSSR (Indian Council of Social Science Research)** — for Thread T governance-facing work.
- **IMPRINT** — if proposal has national-priority technology framing; aligns with AI governance thrust.
- **Industry partnerships** — TCS Research, Microsoft Research India, Adobe Research India. Particularly good for Thread B compute costs.

### Phase 4: AP years 3–6

- **SPARC (Scheme for Promotion of Academic and Research Collaboration)** — cross-institutional international collaboration funding. Ideal for I2 cross-cultural study.
- **IUSSTF (Indo-US Science and Technology Forum)** — US partner-matched funding.
- **ERC Starting Grant** (if eligibility window and career stage align) — prestigious, difficult.
- **Nature Research Award / equivalents** — opportunistic.
- **DST NMICPS / NM-AI / NM-QT** — if AI-governance angle aligns with National Mission themes in the year of application.
- **India–UK, India–Germany, India–Japan bilateral calls** — increasingly common; SoA+AI ethics is a natural fit.

### Grant strategy principles

1. Apply for SERB SRG as soon as administratively possible in AP year 1. It is the default "starter" expectation.
2. Every grant should advance 2–3 specific papers in the inventory, not just "general research."
3. Budget for: student stipends (primary), compute, participant compensation, international travel, open-access fees.
4. Pre-tenure grant record target: 2 Indian grants held + 1 international collaborative grant.

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Baseline MARL agents turn out to have role-invariant embeddings already (kills M1's motivation and weakens M2) | Low | High | Run the M1 diagnostic pilot *early* in the postdoc. If the gap does not exist, reframe M2 as an efficiency or data-economy claim, not a novel-capability claim. |
| H1 moral-shield effect is smaller than predicted | Medium | High | Pre-register with conservative effect-size expectations. Run a pilot (N=30) before full recruitment. Plan a multi-paradigm design so at least one paradigm yields a signal. |
| Spreading too thin across threads, no paper gets finished | High (default risk) | High | Enforce the critical-path sequencing in §5. No new paper started until the previous same-thread paper is submitted. |
| IIT AP committee does not value the MARL thread (perceives as "too CS") | Medium | Medium | Apply primarily to Cog-Sci / HSS / Design departments, not CSE. Emphasize Thread H and I as the core; Thread M as "computational infrastructure supporting the empirical programme." |
| IIT AP committee does not value the IKS thread (perceives as "too philosophical") | Low | Medium | Keep T2 as one line in the SoR until the committee context is clear. Soften if signals suggest pure empirical orientation. |
| Postdoc PI pushes you toward their own agenda, crowding out H1/H5 | Medium | High | Choose a postdoc PI whose work complements yours, not subsumes it. Negotiate time allocation explicitly at offer stage. |
| IIT faculty job market timing misses a search cycle | Medium | Medium | Apply to multiple institutions each cycle (IIT-K, B, D, M, G, H, IISc, Ashoka, Plaksha, IIIT-Delhi, IIIT-Hyderabad, NCBS, national labs). |
| KARMA training is unstable enough that H3 and I1 avatars are not deployable | Medium | High | Invest in training stability during the postdoc (the current `phase-a-stability-logging-m1` branch is exactly this work). Have a "simplified" fallback avatar policy in case full KARMA is unstable. |
| Paper ideas get scooped | Low–Medium | Medium | Do not expose detailed experimental designs of I1, I2, H3 in publicly-readable documents. Pre-register H1 and I1 early — pre-registration is a scoop defense. |

---

## 10. Glossary of key terms

**Agentic AI** — artificial intelligence systems that act autonomously on behalf of users to achieve specified or inferred goals; distinct from passive recommenders.

**Broken Mirror** — the ablation control condition for KARMA, in which the Siamese projector is trained on semantically scrambled role pairings (e.g., aggressor ≈ cleaner) rather than structurally symmetric ones.

**Contrastive learning** — representation-learning paradigm in which the model learns by pulling structurally similar examples together in embedding space and pushing dissimilar examples apart.

**Dual-Use Harvest** — the novel multi-agent environment developed for this program, in which a single beam action serves both cooperative (waste removal, promoting regrowth) and competitive (rival tagging, commons monopolization) functions.

**Empathy Gap** — the conjectured representational deficiency in standard MARL encoders by which "I aggress" and "I am victimized" are encoded as orthogonal latent states, preventing negative feedback from generalizing across social roles.

**Extended Self** — the theoretical claim that sufficiently personalised AI functions as a cognitive-volitional extension of the user, analogous to Clark & Chalmers's Extended Mind.

**Intentional binding** — a standard implicit measure of Sense of Agency; the temporal compression between voluntary action and its effect.

**KARMA (Knowledge Acquisition via Role-Invariant Mirror Architecture)** — the proposed architectural intervention: a recurrent agent with a Siamese projector head trained via contrastive loss to align role-symmetric social observations.

**Level of Automation (LoA)** — the degree to which decision-making and execution are delegated to AI, from full human control (LoA=0) to full autonomy.

**Mirror Test** — the three-condition experimental comparison (Baseline / Broken Mirror / KARMA) that differentiates architectural capacity from semantically correct role invariance.

**Moral disengagement** — the cognitive process (Bandura) by which people offload moral responsibility when acting through intermediaries or technology.

**Moral shield (Proxy Agency moral shield)** — the phenomenon whereby users implicitly endorse aggressive or unethical AI actions because those actions are experienced as continuous with their own volition.

**Proxy Agency** — the user's attribution of AI-system actions to their own extended will, arising when the AI reliably enacts user intent.

**Role invariance** — the property of a representation in which structurally symmetric social roles (aggressor/victim) are embedded into nearby regions of the latent space.

**Semantic Fluency** — the cognitive ease with which AI recommendations align with user intent; functions as a prospective cue to agency.

**Sense of Agency (SoA)** — the subjective experience of initiating voluntary actions and influencing the external world; measured both explicitly (self-report) and implicitly (e.g., intentional binding).

**Sequential Social Dilemma (SSD)** — a class of multi-agent environments in which short-term individual rationality conflicts with long-term collective outcomes (Leibo et al., 2017).

**Three-body ontology (IKS)** — the Vedic distinction between gross body (*sthūla śarīra*), subtle body (*sūkṣma śarīra*), and soul (*ātman*), mapped in this program to avatar, AI policy, and conscious user respectively.

---

*Document version 1.0. Maintained as a living reference; update when papers are submitted, accepted, or repurposed, and when the phase plan in §5 reaches a transition point.*
