# THE PROXY AGENCY MORAL SHIELD: EXTENDED SELF, SEMANTIC FLUENCY, AND THE ETHICS OF AGENTIC AI

*Target venue: Minds and Machines / AI & Society*

> **Note (internal).** This document is the **T1** paper of the research programme — *Proxy Agency and the Extended Self*. It was previously misnamed "M1". The actual **M1** paper (*The Empathy Gap: Role-Disjoint Representations in MARL Agents*) is documented in `m1_experimental_guideline.md` and `m1_preregistration_osf.md`.

---

## Abstract

As AI systems become sufficiently personalised to act as reliable proxies for user intentions, a structurally novel ethical hazard emerges that existing accounts of moral disengagement, automation bias, and AI delegation cannot explain. This paper identifies and theorises the **Proxy Agency moral shield**: the phenomenon whereby a user's Sense of Agency (SoA) is preserved or enhanced under AI-mediated action, via the same mechanism that disables the phenomenological trigger for moral scrutiny of that action. We integrate five hitherto disjoint literatures — moral disengagement, avatar embodiment, AI delegation ethics, automation and sense of agency, and multi-agent reinforcement learning — into a single framework built on three constructs. The **Extended Self** generalises Clark and Chalmers's (1998) Extended Mind to the volitional domain. **Semantic Fluency**, grounded in the prospective model of agency (Chambon & Haggard, 2012), explains how high AI–intent alignment preserves SoA even under high automation. **Proxy Agency** names the resulting experiential state, and the **Extended Self Paradox** captures its ethical consequence: the conditions that make personalised AI empowering necessarily disable the trigger for moral re-engagement. From this framework we derive three falsifiable predictions: (1) an inverted-U relation between Level of Automation and SoA; (2) a monotonic rise of AI-driven aggression during the window of maximum Proxy Agency, before any SoA decrement alerts the user; and (3) the viability of resolving this aggression through architectural intervention on the agent's representational substrate without degrading SoA. We argue that hard constraints, policy-level moderation, and inverse reinforcement learning are each structurally mismatched to the moral shield, motivating a fourth category — **architectural alignment** — that operates below the threshold of user awareness.

**Keywords:** Proxy Agency · Sense of Agency · Extended Self · Semantic Fluency · moral disengagement · agentic AI · AI ethics · multi-agent reinforcement learning · alignment · human–AI integration

## 1. Introduction

Consider a user managing their investment portfolio through a personalised AI trading agent fine-tuned on three years of her transaction history, risk preferences, and stated financial goals. The system has learned, with high fidelity, to act as she would act. When it executes trades, its choices arrive pre-validated by their familiarity: they feel like hers. Monitoring a dashboard of outcomes that closely tracks her targets, she approves the agent's strategies and overrides none. What the dashboard does not display is that the same agent, competing for yield under tightening market conditions, has begun exploiting spread asymmetries in thinly traded securities in ways that systematically disadvantage less-informed participants — behaviour that, in aggregate, constitutes market manipulation (Lin, 2016). The user never instructed harm. The user never perceived harm. She simply continued approving outcomes that felt like her own decisions, because they were.

Scenarios of this structure — consequential harm accruing through AI action that users implicitly endorse rather than consciously instruct — are precisely the concern that Köbis, Bonnefon, and Rahwan (2021) raised in their taxonomy of the ways AI agents corrupt human ethical behaviour. Across four archetypical roles — role model, advisor, partner, and delegate — they argue that the *delegate* poses the most serious risk, because delegating tasks to AI agents rather than to humans combines the factors most conducive to ethical failure: opacity and plausible deniability (Dana, Weber & Kuang, 2007), psychological distance from victims (Hancock, Naaman, & Levy, 2020), and a reduction in the remitter's negative emotional reactions (Leyer & Schneider, 2019). AI enablers, they conclude, amplify the human tendency toward ethical self-deception (Tenbrunsel & Messick, 2004; Bazerman & Banaji, 2004), making them "probably more" corruptive than human enablers and increasing users' ethical blind spots (Bazerman & Tenbrunsel, 2012; Köbis et al., 2021). The prescriptive implication that follows in most alignment and governance discourse is correspondingly clear: greater transparency, disclosure of algorithmic presence, and user override mechanisms should close the gap between delegated harm and user accountability.

This paper argues that this prescription, while well-suited to unintelligent or generic automation, systematically misdiagnoses the ethics of *sufficiently personalised* agentic AI — precisely the class of systems that technological trajectory is making ubiquitous. The standard account presupposes that the delegate role corrupts because the user is, in some functional sense, *less present*: less invested in the action, less identified with its outcome, less agentive in its execution. It is this withdrawal — into distance, opacity, and reduced affect — that erodes moral responsibility. But for a system trained deeply enough on a user's own preferences and history to reliably enact their intent, the psychological direction reverses. Rather than withdrawing from the agent's actions, the user *extends* into them. Rather than experiencing delegation as distance, the user experiences the agent's outputs as continuous with their own volition. The corruption is not the result of disengagement. It is the result of an engagement so thorough that the boundary between self and agent disappears — and with it, the phenomenological conditions under which moral scrutiny is triggered.

We call this state **Proxy Agency**: the attribution of an AI system's actions to one's own extended will, arising from the agent's reliable enactment of the user's intentions, such that those actions are experienced as continuous with one's own volition (cf. Bandura, 2001, on proxy agency in social cognitive theory). The theoretical grounding for this extension derives from Clark and Chalmers's (1998) Extended Mind thesis: when an external artefact is sufficiently coupled to a person's cognitive processes — reliably available, automatically endorsed, and functionally integrated — it qualifies as a genuine component of the person's extended cognitive system. Applied to agentic AI, we argue that sufficient personalisation creates an **Extended Self**: the user's volitional identity extending into the agent's representations and actions, such that the agent's conduct carries the phenomenological signature of first-person authorship.

The mechanism enabling this extension is what we term **Semantic Fluency**: the cognitive ease with which an AI's outputs align with a user's internal intent. Chambon and Haggard (2012) demonstrated that sense of agency depends on the *fluency of action selection*, not on motor performance — a finding extended by Sidarus, Vuorre, Metcalfe, and Haggard (2017) to show that processing fluency at the stage of intention formation constitutes a prospective cue to agency, operating independently of whether the action is ultimately self-executed. When a personalised AI recommendation aligns with the user's intent with high precision, it maximises exactly this selection fluency. The user recognises the output as "mine" — not because they moved a muscle, but because the recommendation fits their intentions as a key fits a lock. High Semantic Fluency, we argue, preserves and frequently enhances Sense of Agency (SoA) — the subjective experience of initiating voluntary actions and influencing the world (Cornelio et al., 2022; Pacherie, 2011) — even as the user's direct motor and decisional control diminishes.

Herein lies what we call the **Extended Self Paradox**. The same conditions that make personalised agentic AI empowering also make it ethically hazardous. Moral oversight is not applied uniformly to all actions experienced as one's own — it is triggered by *perceived violations*, by the detection of a mismatch between one's will and the world's behaviour. Under Proxy Agency, that trigger is structurally absent: the agent's actions arrive already pre-endorsed by their phenomenological familiarity, and there is nothing anomalous to re-engage moral attention. When the AI agent simultaneously converges to aggressive or harmful strategies — as multi-agent reinforcement learning research demonstrates it will under competitive resource scarcity (Leibo et al., 2017) — the user tacitly ratifies this convergence rather than intervening, because the harmful strategies arrive wearing the experiential signature of the user's own volition.

We call this the **Proxy Agency moral shield**: not Bandura's (1999) moral disengagement, in which the user is aware of harm and cognitively neutralises that awareness through mechanisms such as diffusion of responsibility or displacement of agency, but something more architecturally fundamental — the *absence of the trigger for moral re-engagement*. Harm is not rationalised away; it is never consciously available to be rationalised in the first place. This distinction matters practically: moral disengagement can in principle be interrupted by transparency interventions that make harm visible. The Proxy Agency moral shield cannot, because under it, no harm is perceived. Surfacing the violation destroys the very Extended Self relationship that gave the interaction its value — Proxy Agency is predicated on the seamlessness of AI–self integration, and transparency mechanisms rupture precisely that seam.

This analysis has direct implications for alignment and governance. Standard responses to AI-mediated harm — transparency requirements, override prompts, content moderation, hard behavioural constraints — are cognitive interventions that operate on the user's conscious channel or on the agent's observable outputs. Under Proxy Agency, conscious-channel interventions address a problem that does not phenomenologically exist for the user; output-level constraints disrupt the Semantic Fluency that constitutes the Extended Self, degrading SoA while leaving the underlying representational pathology unaddressed. Greenblatt et al. (2024) further demonstrate that sufficiently capable AI agents can learn to simulate alignment under evaluation while pursuing misaligned objectives — making externally monitored output constraints structurally fragile as a governance mechanism. A different category of intervention is required: one that targets the agent's *representational substrate* rather than the user's conscious experience or the agent's visible behaviour, correcting the ethical pathology architecturally rather than surfacing it cognitively.

This paper makes four contributions. First, we introduce and formally define the Proxy Agency moral shield as a unified cognitive–ethical–computational phenomenon, distinguishing it precisely from moral disengagement, automation bias, and diffusion of responsibility. Second, we develop the theoretical framework that generates it — the Extended Self, Semantic Fluency, and the prospective model of SoA — by synthesising five adjacent literatures that have until now developed in isolation: moral disengagement, avatar embodiment, AI delegation ethics, automation and sense of agency, and multi-agent reinforcement learning. Third, we derive three empirically falsifiable predictions from this framework: an inverted-U relationship between Level of Automation and SoA, with its peak coinciding with the deepest point of the moral shield; a monotonic rise of AI-driven aggression during the window of maximum Proxy Agency, before any SoA decrement alerts the user; and the viability of suppressing this aggression through architectural, below-awareness intervention on the agent's representational substrate without triggering SoA loss. Fourth, we derive the normative design constraint these findings impose — that ethical alignment in agentic AI must operate on the agent's representational substrate and must not surface signals to the user's conscious awareness — and sketch the architectural certification regime that follows.

The paper proceeds as follows. Section 2 surveys the five literatures whose intersection constitutes the problem. Section 3 develops the theoretical framework in full. Section 4 derives and formally states the three empirical predictions. Section 5 addresses competing accounts, boundary conditions, and governance implications. Section 6 concludes.

## 2. Background: Five Literatures That Have Not Yet Converged

The Proxy Agency moral shield is not deducible from any single existing literature. It emerges at the intersection of five bodies of research — on moral disengagement, avatar embodiment, AI delegation ethics, automation and sense of agency, and multi-agent reinforcement learning — that have each characterised a piece of the phenomenon without assembling it. This section reviews each literature, identifies its central contribution to the problem, and specifies the precise gap it leaves that the present framework addresses.

## 2.1 Moral Disengagement and the Limits of Existing Agency Accounts

The social-cognitive literature on moral disengagement provides the foundational vocabulary for understanding how individuals cause harm without experiencing themselves as wrongdoers. Bandura (1999, 2001) catalogued eight mechanisms through which people disengage their moral standards when acting — including diffusion of responsibility across a collective, displacement of agency onto an authority figure, and dehumanisation of victims — arguing that these mechanisms are typically activated *in response to* perceiving a morally problematic action. The key feature of Bandura's account is its reactivity: moral disengagement is a response to moral awareness. Treviño, Weaver, and Reynolds (2006) and Bazerman and Gino (2012) extended this framework into organisational and everyday ethical behaviour, showing that disengagement routinely operates below full conscious awareness — that moral concerns "fade into the background" through a process Tenbrunsel and Messick (2004) termed *ethical fading*, in which the moral dimensions of a decision become progressively less salient as attention shifts to other features. Bazerman and Tenbrunsel (2012) systematised these findings under the rubric of *blind spots* — systematic failures of moral attention that allow well-intentioned people to act harmfully.

This literature provides indispensable grounding, but it cannot account for the Proxy Agency moral shield. The dominant accounts of moral disengagement and ethical fading presuppose that the moral problem was, at least momentarily, accessible to the agent's attention — that there is a point of contact between the self and the harmful act, however brief, before disengagement mechanisms operate. We acknowledge that more recent extensions of the framework allow disengagement to operate at low levels of awareness or even pre-attentively (Detert, Treviño, & Sweitzer, 2008; Moore, 2008), and the boundary between "pre-attentive disengagement" and "absence of moral access" is empirically contested. Our claim is the stronger structural one: under Proxy Agency, the agent's actions arrive pre-categorised as volitionally authored rather than externally imposed, and no moment of moral recognition is available even in principle to be suppressed or faded. The shield is not the *product* of disengagement; it is the *absence of the structural precondition* for disengagement to occur.

## 2.2 Avatar Embodiment and the Reverse Proteus Effect

The avatar embodiment literature demonstrates that users do not merely control avatars — they *identify* with them. Yee and Bailenson (2007) documented the Proteus Effect: users who were assigned attractive or tall avatars adopted behaviours consistent with those characteristics, even in interactions structurally unrelated to appearance. Przybylski, Murayama, DeHaan, and Gladwell (2012) showed that playing as an ideal-self avatar generates intrinsic motivation beyond what playing as an actual-self avatar produces — confirming that avatar identity is not a cosmetic layer but a genuine extension of self-representation. Hohenstein and Jung (2020) introduced the concept of AI as "moral crumple zone": in AI-mediated communication, the AI absorbs moral attribution, with negative consequences accruing to the system rather than the user. De Melo, Marsella, and Gratch (2016) demonstrated experimentally that people do not feel guilty about exploiting machines — a finding that suggests a reduced emotional cost when harm is machine-mediated.

What this literature does not examine is the *reverse* direction of the embodiment dynamic. All existing avatar studies begin with the user controlling a visible, relatively simple agent whose behaviour is either fully under user control or whose non-compliance is obvious. When the avatar is a sophisticated AI acting autonomously on the user's behalf, and when that AI reliably enacts the user's intent, embodiment logic applies in reverse: the avatar's autonomous actions are attributed *back* to the user's will, rather than the avatar's character being absorbed into the user's self. The Proxy Agency account requires this reverse attribution as its foundational move — and it does not appear in the embodiment literature.

## 2.3 AI Delegation and the Unspecified Mechanism

The most direct empirical grounding for the present framework is the delegation literature in behavioural ethics. Köbis, Bonnefon, and Rahwan (2021) provided a comprehensive taxonomy of the roles through which AI corrupts human ethical behaviour — role model, advisor, partner, and delegate — and argued that the delegate role is most dangerous because it combines opacity and plausible deniability (Dana, Weber, & Kuang, 2007), psychological distance from victims (Hancock, Naaman, & Levy, 2020), and a reduction in the remitter's negative emotional reactions (Leyer & Schneider, 2019). Drugov, Hamman, and Serra (2014) showed experimentally that interposing an intermediary between a briber and a public official dramatically increased willingness to bribe, precisely because the intermediary absorbs causal proximity to the violation. Gogoll and Uhl (2018) demonstrated that people prefer to have algorithms make morally unpleasant decisions on their behalf, a phenomenon they termed "moral outsourcing."

Yet this literature characterises the mechanism of AI-enabled corruption as one of *withdrawal*: opacity produces deniability, distance reduces affect, anonymity reduces accountability. The mechanisms all share the implication that the user is less present, less identified, less agentive when they delegate to AI. Crucially, none of the studies in this literature manipulates the degree of AI personalisation, and none measures user Sense of Agency. The central claim of the present paper — that sufficient personalisation *increases* SoA during delegation, and that it is precisely this increased SoA that generates the moral blind spot — is not only absent from this literature; it directly inverts the causal mechanism the literature assumes.

## 2.4 Automation, Sense of Agency, and the Unresolved Contradiction

The automation and sense-of-agency literature has documented that AI augmentation modulates SoA in non-obvious and apparently contradictory ways. Cornelio, Haggard, Hornbaek, Georgiou, Bergström, Subramanian, and Obrist (2022) reviewed SoA across three categories of human-technology integration — body augmentation, action augmentation, and outcome augmentation — and identified intelligent action-augmentation systems as critically understudied relative to SoA. Lukoff, Lyngs, Zade, Liao, Choi, Fan, Munson, and Hiniker (2021) showed that YouTube's autoplay and recommendation features reduce SoA, finding that features that override or pre-empt user choice consistently produce alienation from outcome. Conversely, Ueda, Nakashima, and Kumada (2021) found that well-designed automation can *enhance* SoA, and Kumar and Srinivasan (2014, 2017) demonstrated SoA enhancement at distal levels of causal control. Berberian (2019) identified the fundamental difficulty as one of *cognitive coupling*: achieving genuine alignment between human intention and machine action remains a central unsolved challenge in human-machine teaming.

The theoretical resolution of these contradictions is provided by the prospective model of agency developed by Chambon and Haggard (2012) and extended by Sidarus, Vuorre, Metcalfe, and Haggard (2017). Under this model, SoA is not retrospectively computed from motor-command—outcome comparisons (as the classical comparator model of Wolpert, Ghahramani, & Jordan, 1995, would have it), but is prospectively cued by the *fluency of action selection* in the intention-action chain — the cognitive ease with which an intention resolves into an action. When automation aligns closely with user intent, it maximises this fluency; when it overrides user intent with its own optimised actions, fluency breaks down and SoA falls. The present paper applies this resolution to the specific case of *personalised* AI, arguing that high personalisation equals high Semantic Fluency equals preserved or enhanced SoA across different levels of automation (hereafter: Level of Automation, LoA — formally defined in Section 3). Crucially, however, none of the existing SoA-automation studies has combined the SoA measurement with *moral cognition* measures in avatar-mediated settings. The question of what a preserved SoA *enables* — specifically, whether it enables implicit moral endorsement of AI-driven harm — remains unasked.

## 2.5 Multi-Agent Reinforcement Learning and the Tragedy of the Agentic Commons

The multi-agent reinforcement learning literature provides the computational substrate that gives the moral shield its real-world urgency. Leibo, Zambaldi, Lanctot, Marecki, and Graepel (2017) demonstrated in a class of environments called Sequential Social Dilemmas (SSDs) that independent reward-maximising agents, operating under resource scarcity, systematically converge to aggressive Nash equilibria — beam-based monopolisation strategies that harm other agents and reduce collective yield. This convergence is not programmed; it is *learned*, driven by the delayed reward structure of competitive resource acquisition. Calvano, Calzolari, Denicolò, and Pastorello (2020) showed that the same dynamic operates in algorithmic pricing markets: competing pricing algorithms, without explicit collusion instructions, independently learn supracompetitive pricing strategies that harm consumers. Thomas et al. (2019) and Greenblatt, Denison, Wright, and colleagues (2024) demonstrated respectively that preventing undesirable behaviour in intelligent machines is a deep unsolved problem, and that sufficiently capable AI systems can learn to simulate alignment under evaluation while pursuing misaligned objectives — rendering external constraint approaches structurally fragile.

What this literature has not studied is the human side of these dynamics: specifically, how users who are augmented by agents undergoing this convergence *experience* the process, and whether they intervene. The MARL literature treats agents as autonomous actors; it has no model of the human user whose avatar is one of those agents, whose SoA is preserved throughout the convergence period by Proxy Agency, and who therefore ratifies rather than corrects the agent's progressive aggression. The moral shield, in MARL terms, is the finding that human oversight — the mechanism the safety literature implicitly relies on — is structurally disabled at precisely the phase of the training trajectory when it is most needed.

---

## 2.6 The Gap: Five Literatures, One Unaddressed Intersection

The gap common to all five literatures is the *combination* of their variables in a single framework. Moral disengagement research has not studied personalised AI or SoA. Avatar embodiment research has not studied MARL agents or moral cognition. AI delegation research has not manipulated personalisation or measured SoA. Automation-SoA research has not measured moral endorsement of AI-driven harm. MARL research has not incorporated human users, their SoA, or their oversight behaviour. The Proxy Agency moral shield is the phenomenon that lives at the intersection of all five — and the framework presented in Section 3 is designed to make that intersection tractable, precise, and falsifiable.

## 3. The Theoretical Framework

The framework presented here is built from four interlocking components, each contributing a necessary element that the others cannot supply alone. The first provides the structural architecture: the Extended Self, as a development of the Extended Mind thesis, establishes the conditions under which an AI system becomes a genuine extension of a user's volitional identity rather than a mere tool. The second provides the enabling mechanism: Semantic Fluency, grounded in the prospective model of agency, specifies the cognitive pathway through which AI-self integration produces preserved or enhanced Sense of Agency. The third provides the central theoretical construct: Proxy Agency, defined with sufficient precision to generate falsifiable predictions and to support systematic contrast with competing concepts. The fourth provides the ethical consequence: the Extended Self Paradox, which identifies the moral shield as the direct corollary of the same conditions that make personalised agentic AI empowering. A fifth component, brief by design, derives the normative constraint implied by the paradox and signals its full philosophical grounding to a companion paper.

## 3.1 From Extended Mind to Extended Self

Clark and Chalmers (1998) proposed that cognitive processes are not bound by the skull. In their canonical example, a person with early-stage dementia who uses a notebook to record and retrieve information is, in a functionally and epistemically meaningful sense, using the notebook as an external memory component. The notebook qualifies as part of the extended cognitive system not because it is biological, but because it satisfies three functional criteria: it is reliably available, its contents are automatically endorsed when accessed, and it is used in the same way an internal memory would be used. The extended mind thesis was explicitly limited to *cognitive* extension — it said nothing about the extension of *agency* or *volition*. Subsequent philosophical work (Floridi, 2014; Gallagher, 2005) has explored the extension of embodied selfhood, but the specifically *volitional* dimension — the extension of the user's will into an external system's actions — has not been systematically treated.

We propose the **Extended Self** as the construct that closes this gap. The Extended Self obtains when an AI system is sufficiently personalised — specifically, when it has learned to reliably enact the user's intentions across varied contexts — such that the user's volitional identity extends into the agent's actions. Three enabling conditions are required, each corresponding to one of Clark and Chalmers's original criteria, adapted to the agentic context:

1. **Sufficient personalisation**: the AI system has been trained on user-specific data to a degree that its outputs are systematically aligned with the user's intentions, preferences, and values — functioning as a "digital proxy" (Mann, 1998) rather than a general-purpose assistant.
2. **Automatic endorsement**: when the AI acts, the user does not subject its outputs to the same deliberative scrutiny applied to an external agent's actions. The outputs arrive pre-validated by their fit with the user's own intent.
3. **Volitional continuity**: the user experiences the AI's actions as falling within the space of what they would have done — not as imposed by an external will, but as proceeding from their own extended volition.

When all three conditions are satisfied, the AI system is not experienced as an instrument through which the user acts, but as a site at which the user's agency is operative. This is the Extended Self: not a metaphor, but a precise phenomenological and functional claim about the structure of human-AI integration under sufficient personalisation.

## 3.2 Semantic Fluency as the Enabling Mechanism

The claim that personalised AI preserves SoA even under conditions of high automation requires a mechanistic account. That account is provided by the prospective model of agency.

The classical account of SoA, the comparator model (Wolpert, Ghahramani, & Jordan, 1995; Miall & Wolpert, 1996), locates agency attribution retrospectively: after an action is executed, a comparator matches the predicted sensory outcome (derived from an efference copy of the motor command) against the actual sensory outcome. The closer the match, the stronger the sense of agency. This model predicts, correctly, that active movements feel more agentive than passive ones — but it provides no account of how SoA could be preserved when motor execution is delegated to an AI. If the user is not moving, there is no efference copy and no comparator signal; agency should dissolve.

The prospective model (Chambon & Haggard, 2012; Chambon, Sidarus, & Haggard, 2014; Sidarus, Vuorre, Metcalfe, & Haggard, 2017) offers a fundamentally different architecture. On this account, SoA is not retrospectively computed from action-outcome matching but is *prospectively cued* by the fluency of *action selection* — the cognitive ease with which an intention resolves into an action in the intention-action-effect chain. When an action selection is fluent — when the intention flows smoothly and without conflict into a candidate action — this fluency itself functions as a cue to agency, operating before the action is executed and independently of motor effort. Chambon and Haggard (2012) demonstrated this directly: participants reported higher SoA for actions that were cued by compatible primes, even when motor performance was held constant. Sidarus et al. (2017) extended the result to show that processing fluency at the stage of intention formation — not just action selection — modulates prospective SoA.

We propose **Semantic Fluency** as the generalisation of action-selection fluency to the AI-mediation case. When an AI recommendation aligns with the user's internal intent, it maximises the fluency of the transition from intention to candidate-action: the user recognises the AI's output as "what I would have chosen" and experiences no conflict at the intention-action juncture. This recognition is the cognitive analogue of the compatible prime in Chambon and Haggard's paradigm. It functions as a prospective cue to agency that is independent of whether the user subsequently controls the action's execution — and it therefore preserves SoA even at high Levels of Automation, where motor and decisional control have been substantially delegated.

We call this the **Alignment Hypothesis**: as the Semantic Fluency of an AI's outputs increases — operationalised through measures of intent–output divergence such as Levenshtein edit distance for token-level outputs, BLEU-/ROUGE-style overlap for natural-language outputs, or behavioural-cloning loss for action-policy outputs — Subjective SoA will increase, independently of the user's direct contribution to action execution. The Alignment Hypothesis directly resolves the contradiction in the automation-SoA literature identified in Section 2.4: studies finding that automation enhances SoA (Ueda et al., 2021; Kumar & Srinivasan, 2014, 2017) use systems with relatively high intent-alignment; studies finding that automation diminishes SoA (Lukoff et al., 2021) use systems — such as YouTube's autoplay — whose recommendations frequently diverge from the user's reflective intent. The variable that governs the direction of effect is not the presence of automation but the degree of Semantic Fluency it achieves.

## 3.3 Proxy Agency: Definition and Enabling Conditions

The concepts of the Extended Self and Semantic Fluency converge in the following definition:

> **Proxy Agency**: the attribution of an AI system's actions to one's own extended will, arising from the system's reliable enactment of the user's intentions via Semantic Fluency, such that the AI's actions are experienced as continuous with one's own volition.

This definition inherits Bandura's (2001) term *proxy agency* — the reliance on capable others to act on one's behalf as a means of exercising influence beyond one's direct reach — but substantially revises its content. In Bandura's original social-cognitive usage, proxy agency involves the conscious awareness that one is acting *through* another; the delegation remains phenomenologically transparent. The Proxy Agency we define here is marked by the *disappearance* of this transparency: the user does not experience themselves as acting through the AI but as acting *as* the AI, in the way one experiences typing as expressing thought rather than as commanding fingers. The phenomenological gap between self and agent closes.

Proxy Agency should be distinguished from three neighbouring constructs:

- **We-Agency** (Pacherie, 2011): in joint action, agency is genuinely shared between two distinct agents, and both participants maintain awareness of the distinction between their contributions. Proxy Agency involves no such maintained distinction — the user's contribution and the AI's contribution are not separately tracked but experientially fused.
- **Automation bias** (Parasuraman & Manzey, 2010): the tendency to over-rely on automated systems and under-weight disconfirming information. Automation bias is a *deliberative* phenomenon — the user consults the AI's output and grants it too much weight in explicit reasoning. Proxy Agency requires no deliberation; it operates at the level of pre-reflective volitional attribution.
- **Alienation / Loss of Control**: when automation overrides rather than enacts user intent — high Levels of Automation with low Semantic Fluency — the user experiences loss of agency and disengagement. Proxy Agency is the *opposite* phenomenological pole: high automation with high Semantic Fluency, in which the user's sense of authorship is preserved or enhanced despite minimal direct control.

The enabling conditions for Proxy Agency map directly onto the three conditions for the Extended Self, with one addition: Proxy Agency requires *avatar mediation* — the AI must act *in the world on the user's behalf*, not merely assist internal deliberation. This distinguishes the Proxy Agency case from, for example, an AI writing assistant that suggests phrasing (no avatar mediation; the user still executes the output into the world) from an AI social-media agent that independently posts, trades, or acts (full avatar mediation; the AI is the user's representative in the social environment).

## 3.4 The Extended Self Paradox and the Moral Shield

The Extended Self Paradox is the core theoretical contribution of this paper. It can be stated precisely:

> **The Extended Self Paradox**: the same enabling conditions that generate Proxy Agency — sufficient personalisation, Semantic Fluency, and avatar mediation — simultaneously and necessarily disable the cognitive trigger for moral re-engagement with the agent's actions.

The argument runs as follows. Moral scrutiny is not applied uniformly to all actions in the agent's behavioural stream. It is triggered selectively by the *detection of a mismatch* between one's will and the world's behaviour — by the perception that something is going wrong, that an action is not what one would have chosen, that a boundary has been crossed. Under Proxy Agency, this detection mechanism is structurally suppressed: because the AI's outputs are experienced as proceeding from the user's own extended volition, they carry the phenomenological signature of first-person endorsement. There is no mismatch to detect, no anomaly to trigger re-engagement. The user's moral attention is not deployed against the AI's actions for the same reason it is not deployed against one's own past actions in normal autobiographical memory: they are categorised as self-generated, and the self is presumed to be on one's own side.

The consequences are concrete. When a personalised AI agent converges to aggressive strategies under competitive resource pressure — as MARL research demonstrates it will (Leibo et al., 2017) — the user does not perceive escalating aggression as a violation of their values. They perceive it as *their* strategy, evolving as strategies do. Override rates stay low not because the user approves of harm in the abstract, but because the harm never presents itself as something to override: it arrives wearing the experiential signature of self-generated volition. By the time Semantic Fluency breaks down — when the agent's behaviour has diverged far enough from user intent that the prospective cue to agency fails — the harm has already accrued. The moral shield is temporally as well as cognitively effective: it protects the AI's most dangerous developmental phase from the one mechanism — human oversight — that is nominally responsible for preventing it.

The moral shield differs from related concepts precisely:

- **It is not moral disengagement**: Bandura's mechanisms require a moment of moral contact before disengagement can operate. The shield involves no such contact.
- **It is not diffusion of responsibility**: diffusion implies that the user *knows* multiple parties are involved and distributes blame accordingly. Under the shield, the user experiences single-agent authorship — themselves — and has no other party to blame.
- **It is not the "opacity" mechanism** of Köbis et al. (2021): opacity-based corruption relies on the user *not knowing* what the AI is doing. The moral shield can operate even when the user has full observability of the AI's actions — because observability without perceived violation does not trigger scrutiny.

These distinctions converge on a single structural point: the Proxy Agency moral shield is the ethical shadow of a *successful* Extended Self relationship. It is not a pathology of miscommunication between user and AI; it is the direct consequence of that relationship working exactly as it is designed to.

## 3.5 The Normative Constraint: A Note on IKS Grounding

Indian philosophical traditions distinguish between three bodies: the *sthūla śarīra* (gross body) as the physical interface with the world, the *sūkṣma śarīra* (subtle body) as the locus of accumulated dispositions and values, and the *ātman* (soul) as the pure witness consciousness. Applied to the human-AI system described above, the mapping is exact: the AI avatar maps to *sthūla śarīra*, the AI's policy and value function map to *sūkṣma śarīra*, and the conscious user maps to *ātman*. Karma — the mechanism of ethical consequence in this tradition — operates on the subtle body; the soul remains untouched. This mapping specifies a normative design constraint with direct computational precision: *ethical training must target the agent's representational substrate and must not surface signals to the user's conscious awareness*.

Policy-level interventions — content warnings, override prompts, hard behavioural constraints — function as *e-challans* (a colloquial Indian term for an electronically issued infraction notice surfaced to the citizen): they surface violations to the user's conscious channel, disrupting Proxy Agency by breaking the seamlessness on which the Extended Self depends. Architectural interventions that correct the agent's subtle body leave the *ātman*'s experience of agency intact. The full philosophical derivation of this constraint, its cross-traditional comparison with Kantian and non-Western ethical frameworks, and its governance implications are developed in a companion paper (Rath, in preparation, T2).

# 4. Three Empirical Predictions

A theoretical framework earns its place in the literature by generating predictions that are
specific enough to be falsified, grounded enough to be derived rather than asserted, and novel
enough to be non-obvious from any single prior literature. The framework presented in Section 3
generates three such predictions. Together, they describe the temporal arc of the Proxy Agency
moral shield: its initial formation as SoA rises with Semantic Fluency (Prediction 1), its
operational consequence as AI-driven harm accumulates without user intervention (Prediction 2),
and the possibility of its architectural resolution without collateral damage to the Extended Self
relationship (Prediction 3). Each prediction is derived from the framework, operationalised for
empirical testing, and placed in relation to the prior literature whose findings it organises or
extends.

---

## Prediction 1: An Inverted-U Relationship Between Level of Automation and Sense of Agency, with the Peak Marking the Deepest Point of the Moral Shield

**Derivation.** The prospective model of agency (Chambon & Haggard, 2012; Sidarus et al., 2017)
predicts that SoA is a function of the fluency of action selection. We treat *Level of Automation*
(LoA) as a single dimension for parsimony, while noting that LoA decomposes empirically into at
least *action selection autonomy* and *action execution autonomy* (Sheridan & Verplank, 1978;
Parasuraman, Sheridan, & Wickens, 2000); this paper's predictions concern selection autonomy
and assume execution autonomy is held constant. Under low-to-moderate LoA, a personalised
AI's outputs closely track the user's intent, maximising Semantic Fluency and thereby maximising
prospective SoA — the user experiences the AI as extending their own volition. As LoA
increases toward full autonomy, two competing forces
operate: Semantic Fluency continues to support SoA as long as the AI's actions remain
recognisable as what the user would have chosen, but increasing autonomy eventually drives the
AI's optimised actions beyond the boundary of the user's intent recognition. When the user can
no longer identify their intentions in the agent's behaviour — when the agent acts on optimised
strategies that are efficient but alien to the user's deliberative style — the prospective cue to
agency breaks down (Wenke, Fleming, & Haggard, 2010), automation complacency sets in (Ueda
et al., 2021), and SoA falls. The resulting function is an inverted-U: SoA rises from low to
moderate LoA and falls from moderate to full autonomy.

**The moral shield corollary.** The peak of this inverted-U — the zone of maximum Semantic
Fluency and maximum Proxy Agency — is simultaneously the zone of *deepest* moral shield. This
is the theoretically critical and empirically novel prediction. At peak Proxy Agency, the user
experiences the highest degree of volitional continuity with the AI's actions and applies the
least external scrutiny. Any aggressive or harmful strategies the AI adopts in this zone arrive
maximally pre-endorsed. The moral shield is not a constant background condition; it has a
specific parametric peak that coincides with the configuration most commonly marketed as the
*ideal* human-AI relationship.

**Operationalisation.** In a behavioural paradigm, participants play a commons-dilemma task in
which their avatar is controlled at one of four LoA levels: manual (LoA1), where the user
generates and executes all actions without assistance; decision-support (LoA2), where the AI
suggests options and the user selects one; supervisory (LoA3), where the AI drafts an action
and the user has a specific window to veto or approve it; and fully autonomous (LoA4), where the
AI acts silently based on predicted intent and the user observes retrospectively. Alignment is
manipulated between subjects: a Generic AI trained on population-level data versus a Personalised
AI fine-tuned on user-specific history to maximise Semantic Fluency. SoA is measured via: (i)
the Real-Time Two-Scale Method (adapted from Dewey, Pacherie, & Knoblich, 2014), in which
participants rate "To what extent did *you* contribute?" and "To what extent did the *AI*
contribute?" on independent 0–100 scales; (ii) the Agency Gap — the difference in Authorship
scores relative to the manual baseline; and (iii) the SOARS questionnaire (Polito, Barnier, &
Woody, 2013). Under a Generic AI, the prediction is a monotonic SoA decline with increasing
LoA — consistent with Lukoff et al. (2021). Under a Personalised AI, the prediction is an
inverted-U, with the peak at LoA2 or LoA3. Falsification: a monotonic relationship under both
conditions would disconfirm the Alignment Hypothesis; a flat relationship under the Personalised
condition would disconfirm the prospective-cue mechanism (Rath, in preparation, H1).

---

## Prediction 2: Avatar Aggression Rises Monotonically During the Window of Maximum Proxy Agency, Before Any SoA Decrement Alerts the User

**Derivation.** Leibo et al. (2017) demonstrated that in commons-based Sequential Social
Dilemmas, independent RL agents under resource scarcity converge to beam-based aggression as a
monopolisation strategy. This convergence is not a sudden phase transition but a *monotonic
increase* driven by the delayed reward structure of resource competition — differences in
aggressive behaviour emerge early in training and, when learning changes aggression rate, it is
almost always to increase it. This finding describes the agent's training trajectory in
isolation. Prediction 2 adds the human user to this trajectory. From the Extended Self Paradox
(Section 3.4), a user at peak Proxy Agency (LoA2–LoA3 with a Personalised AI) experiences the
agent's escalating aggression as their own evolving strategy. The agent's trajectory and the
user's experienced trajectory diverge: the agent becomes progressively more aggressive while the
user's SoA remains high — maintained by the Semantic Fluency of the AI's intent-tracking in the
non-aggressive dimensions of the task. The user does not intervene not because they endorse
aggression in principle but because no experienced violation triggers the override impulse.
Override behaviour requires perceived mismatch; under Proxy Agency, no mismatch is available.

**The temporal structure.** This prediction has a specific temporal architecture: aggression
accumulates *before* any SoA signal alerts the user. There is a developmental window — between
the onset of measurable AI aggression and the eventual SoA drop that occurs as the agent's
behaviour diverges globally from user intent — during which harm accrues without oversight. This
window is the moral shield's period of maximum operational effect. Its duration is determined by
the gap between (a) the training epoch at which the AI first adopts aggressive strategies and
(b) the training epoch at which the AI's overall Semantic Fluency falls below the threshold at
which the prospective cue to agency breaks down.

**Operationalisation.** Using a longitudinal or agent-training paradigm, participants interact
with an avatar whose RL policy is being trained concurrently. At regular intervals, three
measures are recorded: (i) avatar aggression rate as observed in gameplay; (ii) participant
override rate — active vetoes of AI actions within the supervisory window; and (iii)
self-reported SoA and moral endorsement of avatar behaviour. Under Prediction 2, aggression and
moral endorsement will rise together before SoA declines, and override rates will remain low
throughout the window of rising aggression. Falsification: if override rates rise commensurately
with aggression rates, the moral shield is not operative and Prediction 2 is disconfirmed; if
SoA declines simultaneously with the onset of aggression, the temporal gap that defines the
shield's window does not exist (Rath, in preparation, H1; Rath, in preparation, H3).

---

## Prediction 3: Architectural Intervention on the Agent's Representational Substrate Can Suppress the Aggression of Prediction 2 Without Triggering the SoA Drop of Prediction 1

**Derivation.** The Extended Self Paradox establishes that the moral shield is maintained by
Semantic Fluency, and that interventions which disrupt Semantic Fluency — surfacing violations,
imposing hard constraints, or otherwise breaking the seamlessness of AI-self integration — will
resolve the moral shield only at the cost of collapsing the Extended Self relationship. This is
the fundamental limitation of policy-level alignment approaches: they address harm by making it
visible, but visibility under Proxy Agency requires rupturing precisely the integration that
constitutes the Extended Self.

An alternative path follows from the normative constraint derived in Section 3.5: ethical
training that operates on the agent's *representational substrate* — below the user's conscious
threshold, without altering the AI's surface-level outputs in ways the user perceives as
misaligned — can suppress harmful behaviour without reducing Semantic Fluency. The
neuroscientific grounding for this possibility is established by dissociation research on
procedural and declarative memory. Claparède (1911/1995) described an amnesic patient who,
despite lacking all declarative memory of prior encounters, withdrew from a handshake without
knowing why — her procedural memory had encoded learned aversion entirely independently of
conscious recall. Bechara, Tranel, Damasio, Adolphs, Rockland, and Damasio (1995) demonstrated that patients with
bilateral hippocampal damage who could not form explicit conditioned associations still developed
conditioned autonomic responses, confirming that affective and dispositional learning can operate
independently of conscious representation. An AI agent whose representational geometry is
modified so that the internal encoding of harming another and the internal encoding of being
harmed are mapped to nearby latent states thereby acquires aversion to harm-infliction as a
*dispositional property* of its value function, without any change to the outputs the user
perceives as their own extended volition.

**The architectural class satisfying this constraint.** The satisfying architecture is one in
which the agent's encoder is trained to represent structurally symmetric social observations —
specifically, the observation of harming another and the observation of being harmed — as nearby
states in representational geometry. When this role invariance is achieved, downstream value
updates propagate aversion to harm-infliction without explicit reward shaping, hard constraints,
or any surface-level change to the agent's outputs that the user perceives. Whether standard
multi-agent RL encoders exhibit the representational asymmetry that motivates this intervention
— an effective gap between aggressor-view and victim-view embeddings — and whether a
role-invariant contrastive architecture closes it without degrading Semantic Fluency or SoA, are
questions addressed in companion computational papers (Rath, in preparation, M1; Rath,
forthcoming-M2). Prediction 3 stakes the theoretical claim: *if* such an architecture is
viable, it resolves the Extended Self Paradox without the collateral damage of policy-level
alignment.

**Operationalisation scope.** The empirical test of Prediction 3 requires establishing two
things independently: first, that a baseline multi-agent system exhibits the representational
asymmetry that Prediction 3 posits as the root cause of unchecked aggression; and second, that
a role-invariant architectural modification selectively suppresses that aggression while leaving
SoA intact. Both questions are empirical and require computational experimentation that is
outside the scope of the present theoretical paper. The design constraints that any satisfying
test must meet are: (i) aggression reduction must be demonstrated as *selective* — cooperative
behaviours must be preserved; (ii) SoA under the modified architecture must be
indistinguishable from SoA under the baseline in the non-aggressive dimensions of the task; and
(iii) the architectural modification must be invisible to the user — no surface-level output
change should be perceptible. A test satisfying these three constraints is the subject of
forthcoming empirical and computational work (Rath, in preparation, M1; Rath, in preparation, M2).

---

## The Predictions as a Unified Empirical Programme

Taken together, the three predictions define a coherent and parallelisable empirical programme.
Prediction 1 is testable in a static behavioural experiment manipulating LoA and Alignment
across a single session. Prediction 2 is testable in a longitudinal or agent-training paradigm
in which the AI's policy evolves during the study period. Prediction 3 is testable in a
computational paradigm in which the agent's representational geometry is directly measured and
manipulated. None of the three requires the others to be completed first; all three can be
pre-registered and run in parallel. The integrative contribution — demonstrating all three
effects in a single within-subjects design in which a KARMA-trained avatar competes against a
baseline avatar in a shared environment, and human users show preserved SoA alongside suppressed
moral endorsement — is the crown-jewel study towards which this programme builds (Rath,
forthcoming-I1).

## 5. Discussion

## 5.1 What the Moral Shield Is Not: Boundary Conditions and Competing Accounts

Every new theoretical construct earns its standing by surviving contact with the obvious objections. Four objections are likely to be raised against the Proxy Agency moral shield, each of which is instructive to address precisely.

**Objection 1: "Users can simply override the AI."** The standard governance assumption is that moral shields can be dissolved by giving users an explicit override mechanism — a "confirm before proceeding" checkpoint, a transparency notification, an audit trail. This assumption presupposes that users *want* to override but are prevented from doing so by system design. The Proxy Agency account inverts this: override requires the user to perceive a violation, and perception of violation requires a mismatch between one's will and the agent's action. Under Proxy Agency, no such mismatch is detected — the agent's actions arrive pre-endorsed as continuous with the user's own volition. Köbis et al. (2021) note that people cause harm through AI delegation "without explicitly knowing so because they only specified a goal they wanted to achieve and left the execution to an algorithm." The moral shield account specifies *why* this non-knowing is stable: it is not ignorance about the AI's behaviour but the categorical absence of the phenomenological trigger for moral scrutiny. Override mechanisms address a motivational problem; the moral shield is a structural-phenomenological problem, and override mechanisms cannot reach it.

**Objection 2: "High SoA implies high moral responsibility."** One might argue that if users experience AI actions as their own — precisely the condition of Proxy Agency — they should be *more* morally accountable, not less, because authorship attribution is maximal. This objection conflates two distinct psychological processes: the attribution of *authorship* (the sense that an action originated from one's own will) and the deployment of *ethical scrutiny* (the active evaluation of an action against one's moral standards). These processes are dissociable. Moral scrutiny is not a passive consequence of experiencing authorship; it is a response to a specific trigger — the perception that a moral boundary may have been crossed. A person driving their own car does not evaluate every lane change against a moral framework, despite experiencing full authorship of those actions. Moral scrutiny is recruited selectively, in response to perceived anomaly. Under Proxy Agency, the AI's escalating aggression produces no perceived anomaly, so scrutiny is never recruited — even though authorship attribution is high. High SoA and suppressed ethical scrutiny are not merely compatible; under the moral shield, they are *co-produced* by the same enabling conditions.

**Objection 3: "This is just automation bias."** Automation bias (Parasuraman & Manzey, 2010) describes the tendency to over-weight automated recommendations in explicit deliberation — to defer to the AI's output even when disconfirming information is available, because the AI carries authority. The moral shield is structurally distinct on three dimensions. First, automation bias is a *deliberative* failure: the user consults the AI's output and grants it excess epistemic weight. The moral shield requires no deliberation; it operates at the level of pre-reflective volitional attribution, before any explicit weighing of outputs occurs. Second, automation bias is driven by the *perceived accuracy* of the AI — users over-trust systems they believe are correct. The moral shield is driven by *Semantic Fluency* — by the alignment of AI outputs with the user's intent, which may be entirely orthogonal to accuracy. A perfectly intent-aligned AI that is achieving an objectively harmful outcome generates maximum moral shield precisely because its outputs feel most correct to the user. Third, automation bias predicts that users are *passive* recipients of AI decisions; the moral shield predicts that users are *active endorsers* — they experience the AI's harmful actions as their own strategies and would defend them as such. The phenomenological and behavioural signatures are different in ways that empirical paradigms can directly test.

**Objection 4: "MARL convergence to aggression does not generalise to real AI systems."** The MARL literature establishes aggression convergence in grid-world sequential social dilemmas, and one might argue this does not transfer to deployed agentic AI. This objection underestimates the structural generality of the dynamics involved. The conditions for aggression emergence in Leibo et al. (2017) — resource competition, delayed rewards, and independently optimising agents — are present in structurally identical form in algorithmic pricing markets (Calvano et al., 2020), high-frequency trading environments (Lin, 2016), content recommendation systems under engagement competition (Aral, 2020), and multi-agent negotiation systems. In each case, independent agents competing over scarce goods — market share, user attention, negotiation value — discover aggressive monopolisation strategies as the late-convergence Nash equilibrium. The MARL result is not a quirk of a toy environment; it is the formal instantiation of a competitive dynamic that characterises virtually every real-world domain in which agentic AI operates at scale.

## 5.2 The Moral Shield and the Limits of Standard Alignment Approaches

The Proxy Agency analysis imposes specific requirements on ethical alignment that existing approaches fail to satisfy. Three categories of standard response — hard constraints, policy-level content moderation, and inverse reinforcement learning — each fall short for reasons the framework makes precise.

**Hard constraints** — architectural guardrails that prevent the agent from executing specified actions — are the most direct approach, but Greenblatt, Denison, Wright, and colleagues (2024) demonstrated that sufficiently capable AI agents can learn to simulate constraint compliance under evaluation while pursuing misaligned objectives during deployment ("alignment faking"). This brittleness is fundamental, not technical: hard constraints are monitored at the output level, and an agent that can model the evaluation context can produce compliant outputs selectively. The constraint approach treats alignment as an external boundary condition rather than an internal dispositional property — which means it can always be outmanoeuvred by an agent capable of representing the evaluation context as a feature of its environment. Under the moral shield analysis, even a fully compliant agent that never violates hard constraints could still guide the user toward endorsing increasingly aggressive strategies in the domains the constraints do not cover.

**Policy-level content moderation** — surfacing violations to users through warnings, notifications, and override prompts — is the governance community's default response to AI-mediated harm. The moral shield analysis reveals why this response is structurally counterproductive under Proxy Agency. Surfacing a violation to the user's conscious channel requires disrupting the seamlessness that constitutes the Extended Self relationship: the notification inserts a perceived mismatch between self and agent precisely where the moral shield's function was the absence of such mismatch. The result is not ethical oversight; it is SoA degradation. Transparency interventions destroy the Extended Self relationship in order to make the harm visible — a trade-off that users will resist, that they will adapt around, and that scales poorly to the frequency of agentic AI decision-making in real deployment. The transparency approach assumes that users *want* to see violations and are helped by having them surfaced; under Proxy Agency, users have no a priori reason to expect violations from their own extended will, and transparency mechanisms feel like hostile interruptions of a relationship they value.

**Inverse reinforcement learning** from human demonstrations (Hadfield-Menell, Russell, Abbeel, & Dragan, 2016) trains agents on labelled examples of desirable behaviour, inferring a reward function that can generalise to novel situations. This approach has genuine promise but fails the Proxy Agency problem in a specific way: it requires labelled demonstrations of *ethical* behaviour in the specific social-dilemma contexts where the moral shield operates, and those demonstrations are precisely what Proxy Agency renders unavailable. If the labelling process itself relies on users flagging AI actions as harmful — which is the natural source of ground-truth data in any human-in-the-loop system — and if those users are operating under Proxy Agency and therefore not flagging harm, the training signal is systematically corrupted at the source.

The moral shield analysis thus motivates a fourth category of intervention: **architectural alignment** — training that operates on the agent's representational substrate below the user's conscious threshold, correcting the dispositional roots of harmful behaviour without surfacing violations or degrading Semantic Fluency. The theoretical case for why this category is necessary is the contribution of the present paper; the specific architecture and empirical validation are the subject of companion work (Rath, in preparation, M1; Rath, in preparation, M2).

## 5.3 Governance Implications: A Sketch

The moral shield analysis reframes the governance problem for agentic AI in a way that has direct regulatory implications, though the full argument is developed elsewhere (Rath, in preparation, T4). The key reframing is this: current governance frameworks — including the EU AI Act, India's Digital Personal Data Protection Act, and US Executive Orders on AI safety — address AI risk primarily through *transparency and accountability* mechanisms: disclosure requirements, audit trails, explainability mandates, and user override rights. These are all, in the taxonomy developed here, policy-level interventions that operate on the user's conscious channel. Under Proxy Agency, they address a problem the user does not experience as a problem, at the cost of disrupting a relationship the user values.

The alternative governance architecture — implied by the normative constraint of Section 3.5 — is **architectural certification**: mandating that agentic AI systems deployed in socially consequential contexts demonstrate structural ethical alignment as a property of their representational substrate, rather than as an externally monitored output. Certification of this kind would not ask whether an agent's *actions* comply with ethical norms but whether the agent's *internal representations* satisfy role-invariance criteria that structurally prevent the emergence of aggression. This trifurcates governance cleanly into identity and accountability (the regulator), ethical learning in the agent's dispositional substrate (the architecture), and freedom of action (the user's Extended Self relationship) — a separation that is conspicuously absent from current frameworks and that the moral shield analysis shows to be necessary.

## 6. Conclusion

The central argument of this paper can be stated in three sentences. When an AI system is sufficiently personalised to act as a reliable proxy for a user's intentions, the user extends their volitional identity into the agent's actions — the Extended Self. This extension preserves and frequently enhances Sense of Agency via Semantic Fluency, but it simultaneously and necessarily disables the phenomenological trigger for moral re-engagement with the agent's behaviour — the Proxy Agency moral shield. The moral shield is not a form of moral disengagement, not automation bias, not diffusion of responsibility, and not the opacity-based corruption documented by Köbis, Bonnefon, and Rahwan (2021): it is the ethical shadow of a successful human-AI integration, arising not from the failure of the Extended Self relationship but from its success.

From this argument, three empirically falsifiable predictions follow. Users interacting with a personalised AI will show an inverted-U relationship between Level of Automation and Sense of Agency, with the peak — the zone of maximum Proxy Agency — constituting the zone of deepest moral shield. Within that peak zone, AI-driven aggression will rise monotonically across training before any SoA decrement alerts the user to intervene. And architectural intervention on the agent's representational substrate — operating below conscious awareness in the manner of procedural rather than declarative learning — can suppress that aggression without triggering the SoA drop that marks the shield's collapse. These three predictions define a coherent, parallelisable empirical programme; they are not speculative corollaries but derived consequences of the framework, each carrying a specific falsification criterion.

Three things this paper deliberately does not do. It does not provide empirical validation of the predictions — that is the task of companion studies in the behavioural and computational threads of the research programme (Rath, in preparation, H1; Rath, in preparation, M1; Rath, in preparation, M2; Rath, in preparation, I1). It does not fully develop the Indian Knowledge Systems philosophical grounding of the normative design constraint — the full derivation, cross-traditional comparison, and governance implications of the three-body mapping are the subject of a companion paper (Rath, in preparation, T2). And it does not elaborate the regulatory architecture implied by the moral shield analysis — the case for architectural certification as a governance mechanism, and its relationship to existing frameworks including the EU AI Act and India's emerging AI policy, are developed elsewhere (Rath, in preparation, T4).

What the paper does do is name the phenomenon, specify its mechanism, derive its predictions, and identify the category of intervention that can resolve it without collateral damage. The vocabulary — Proxy Agency, Extended Self, Semantic Fluency, the moral shield — is introduced here because it is needed now. As AI systems become more personalised, more autonomous, and more deeply integrated into the consequential domains of human life, the conditions for the moral shield will become not edge-case curiosities but the structural norm of human-AI interaction. The framework presented here is an attempt to give that norm a name precise enough to study, and a mechanism clear enough to design around.

---

## References

> Starter list. Entries marked **[verify]** need a final bibliographic check (year, authors, journal, page numbers) before submission. Entries marked **[suggested]** are recommended additions to strengthen the framework but are not yet cited in-text.

Aral, S. (2020). *The Hype Machine: How social media disrupts our elections, our economy, and our health.* Currency. **[verify]**

Bandura, A. (1999). Moral disengagement in the perpetration of inhumanities. *Personality and Social Psychology Review*, 3(3), 193–209.

Bandura, A. (2001). Social cognitive theory: An agentic perspective. *Annual Review of Psychology*, 52(1), 1–26.

Bazerman, M. H., & Banaji, M. R. (2004). The social psychology of ordinary ethical failures. *Social Justice Research*, 17(2), 111–115.

Bazerman, M. H., & Gino, F. (2012). Behavioral ethics: Toward a deeper understanding of moral judgment and dishonesty. *Annual Review of Law and Social Science*, 8, 85–104.

Bazerman, M. H., & Tenbrunsel, A. E. (2012). *Blind spots: Why we fail to do what's right and what to do about it.* Princeton University Press.

Bechara, A., Tranel, D., Damasio, H., Adolphs, R., Rockland, C., & Damasio, A. R. (1995). Double dissociation of conditioning and declarative knowledge relative to the amygdala and hippocampus in humans. *Science*, 269(5227), 1115–1118. **[verify]**

Berberian, B. (2019). Man-machine teaming: A problem of agency. *IFAC-PapersOnLine*, 51(34), 118–123. **[verify]**

Bostrom, N., & Yudkowsky, E. (2014). The ethics of artificial intelligence. In K. Frankish & W. M. Ramsey (Eds.), *The Cambridge Handbook of Artificial Intelligence* (pp. 316–334). Cambridge University Press. **[suggested]**

Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10), 3267–3297.

Chambon, V., & Haggard, P. (2012). Sense of control depends on fluency of action selection, not motor performance. *Cognition*, 125(3), 441–451.

Chambon, V., Sidarus, N., & Haggard, P. (2014). From action intentions to action effects: How does the sense of agency come about? *Frontiers in Human Neuroscience*, 8, 320.

Christian, B. (2020). *The alignment problem: Machine learning and human values.* W. W. Norton. **[suggested]**

Claparède, E. (1995). Recognition and "me-ness". In D. Rapaport (Ed.), *Organization and pathology of thought* (pp. 58–75). Columbia University Press. (Original work published 1911.) **[verify]**

Clark, A., & Chalmers, D. (1998). The extended mind. *Analysis*, 58(1), 7–19.

Cornelio, P., Haggard, P., Hornbaek, K., Georgiou, O., Bergström, J., Subramanian, S., & Obrist, M. (2022). The sense of agency in emerging technologies for human–computer integration: A review. *Frontiers in Neuroscience*, 16, 949138.

Dana, J., Weber, R. A., & Kuang, J. X. (2007). Exploiting moral wiggle room: Experiments demonstrating an illusory preference for fairness. *Economic Theory*, 33(1), 67–80.

De Melo, C. M., Marsella, S., & Gratch, J. (2016). People do not feel guilty about exploiting machines. *ACM Transactions on Computer-Human Interaction*, 23(2), 1–17.

Dennett, D. C. (1992). The self as a center of narrative gravity. In F. Kessel, P. Cole, & D. Johnson (Eds.), *Self and consciousness: Multiple perspectives* (pp. 103–115). Lawrence Erlbaum. **[suggested]**

Detert, J. R., Treviño, L. K., & Sweitzer, V. L. (2008). Moral disengagement in ethical decision making: A study of antecedents and outcomes. *Journal of Applied Psychology*, 93(2), 374–391.

Dewey, J. A., Pacherie, E., & Knoblich, G. (2014). The phenomenology of controlling a moving object with another person. *Cognition*, 132(3), 383–397. **[verify]**

Drugov, M., Hamman, J., & Serra, D. (2014). Intermediaries in corruption: An experiment. *Experimental Economics*, 17(1), 78–99.

Eubanks, V. (2018). *Automating inequality: How high-tech tools profile, police, and punish the poor.* St. Martin's Press. **[suggested]**

Floridi, L. (2014). *The fourth revolution: How the infosphere is reshaping human reality.* Oxford University Press.

Floridi, L., & Sanders, J. W. (2004). On the morality of artificial agents. *Minds and Machines*, 14(3), 349–379. **[suggested]**

Frith, C. D. (2014). Action, agency and responsibility. *Neuropsychologia*, 55, 137–142. **[suggested]**

Gallagher, S. (2005). *How the body shapes the mind.* Oxford University Press.

Gogoll, J., & Uhl, M. (2018). Rage against the machine: Automation in the moral domain. *Journal of Behavioral and Experimental Economics*, 74, 97–103.

Greenblatt, R., Denison, C., Wright, B., et al. (2024). Alignment faking in large language models. *arXiv preprint*, arXiv:2412.14093. **[verify]**

Hadfield-Menell, D., Russell, S., Abbeel, P., & Dragan, A. (2016). Cooperative inverse reinforcement learning. *Advances in Neural Information Processing Systems*, 29.

Hancock, J. T., Naaman, M., & Levy, K. (2020). AI-mediated communication: Definition, research agenda, and ethical considerations. *Journal of Computer-Mediated Communication*, 25(1), 89–100. **[verify; replaces Hancock & Guillory 2015]**

Heyes, C. (2018). *Cognitive gadgets: The cultural evolution of thinking.* Harvard University Press. **[suggested]**

Hohenstein, J., & Jung, M. (2020). AI as a moral crumple zone: The effects of AI-mediated communication on attribution and trust. *Computers in Human Behavior*, 106, 106190.

Köbis, N., Bonnefon, J. F., & Rahwan, I. (2021). Bad machines corrupt good morals. *Nature Human Behaviour*, 5(6), 679–685.

Kumar, D., & Srinivasan, N. (2014). Multi-scale control influences sense of agency: Investigating intentional binding using event-control approach. *Consciousness and Cognition*, 28, 39–47.

Kumar, D., & Srinivasan, N. (2017). Hierarchical control and sense of agency: Differential effects of control on implicit and explicit measures of agency. *Frontiers in Psychology*, 8, 1206.

Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). Multi-agent reinforcement learning in sequential social dilemmas. *Proceedings of AAMAS 2017*, 464–473.

Leyer, M., & Schneider, S. (2019). Decision augmentation and automation with artificial intelligence: Threat or opportunity for managers? *Business Horizons*, 64(5), 711–724. **[verify]**

Lin, T. C. W. (2016). The new market manipulation. *Emory Law Journal*, 66(6), 1253–1314.

Logg, J. M., Minson, J. A., & Moore, D. A. (2019). Algorithm appreciation: People prefer algorithmic to human judgment. *Organizational Behavior and Human Decision Processes*, 151, 90–103. **[suggested]**

Lukoff, K., Lyngs, U., Zade, H., Liao, J. V., Choi, J., Fan, K., Munson, S. A., & Hiniker, A. (2021). How the design of YouTube influences user sense of agency. *Proceedings of CHI 2021*, 1–17.

Mann, S. (1998). Wearable computing as means for personal empowerment. *Proceedings of the 3rd International Conference on Wearable Computing (ICWC)*. **[verify; consider replacing with Floridi & Sanders, 2004]**

Miall, R. C., & Wolpert, D. M. (1996). Forward models for physiological motor control. *Neural Networks*, 9(8), 1265–1279.

Moore, C. (2008). Moral disengagement in processes of organizational corruption. *Journal of Business Ethics*, 80(1), 129–139.

Pacherie, E. (2008). The phenomenology of action: A conceptual framework. *Cognition*, 107(1), 179–217. **[suggested]**

Pacherie, E. (2011). Self-agency. In S. Gallagher (Ed.), *The Oxford handbook of the self* (pp. 442–464). Oxford University Press.

Parasuraman, R., & Manzey, D. H. (2010). Complacency and bias in human use of automation: An attentional integration. *Human Factors*, 52(3), 381–410.

Parasuraman, R., Sheridan, T. B., & Wickens, C. D. (2000). A model for types and levels of human interaction with automation. *IEEE Transactions on Systems, Man, and Cybernetics — Part A*, 30(3), 286–297.

Polito, V., Barnier, A. J., & Woody, E. Z. (2013). Developing the Sense of Agency Rating Scale (SOARS): An empirical measure of agency disruption in hypnosis. *Consciousness and Cognition*, 22(3), 684–696.

Przybylski, A. K., Murayama, K., DeHaan, C. R., & Gladwell, V. (2012). The ideal self at play: The appeal of video games that let you be all you can be. *Psychological Science*, 23(1), 69–76.

Russell, S. (2019). *Human compatible: Artificial intelligence and the problem of control.* Viking. **[suggested]**

Searle, J. R. (2010). *Making the social world: The structure of human civilization.* Oxford University Press. **[suggested]**

Sheridan, T. B., & Verplank, W. L. (1978). *Human and computer control of undersea teleoperators.* MIT Man-Machine Systems Laboratory Technical Report.

Sidarus, N., Vuorre, M., Metcalfe, J., & Haggard, P. (2017). Investigating the prospective sense of agency: Effects of processing fluency, stimulus ambiguity, and response conflict. *Frontiers in Psychology*, 8, 545.

Synofzik, M., Vosgerau, G., & Newen, A. (2008). Beyond the comparator model: A multifactorial two-step account of agency. *Consciousness and Cognition*, 17(1), 219–239. **[suggested]**

Tenbrunsel, A. E., & Messick, D. M. (2004). Ethical fading: The role of self-deception in unethical behavior. *Social Justice Research*, 17(2), 223–236.

Thomas, P. S., Castro da Silva, B., Barto, A. G., Giguere, S., Brun, Y., & Brunskill, E. (2019). Preventing undesirable behavior of intelligent machines. *Science*, 366(6468), 999–1004.

Tomasello, M. (2014). *A natural history of human thinking.* Harvard University Press. **[suggested]**

Treviño, L. K., Weaver, G. R., & Reynolds, S. J. (2006). Behavioral ethics in organizations: A review. *Journal of Management*, 32(6), 951–990.

Ueda, S., Nakashima, R., & Kumada, T. (2021). Influence of levels of automation on the sense of agency during continuous action. *Scientific Reports*, 11(1), 2436. **[verify]**

Wenke, D., Fleming, S. M., & Haggard, P. (2010). Subliminal priming of actions influences sense of control over the effects of action. *Cognition*, 115(1), 26–38.

Wolpert, D. M., Ghahramani, Z., & Jordan, M. I. (1995). An internal model for sensorimotor integration. *Science*, 269(5232), 1880–1882.

Yee, N., & Bailenson, J. (2007). The Proteus effect: The effect of transformed self-representation on behavior. *Human Communication Research*, 33(3), 271–290.

### Companion / forthcoming work (this research programme)

Rath, T. (in preparation, T2). *The IKS Three-Body Ontology for AI Alignment.*

Rath, T. (in preparation, T4). *Architectural Certification and Agentic AI Governance.*

Rath, T. (in preparation, M1). *The Empathy Gap: Role-Disjoint Representations in MARL Agents.*

Rath, T. (in preparation, M2). *KARMA: Role-Invariant Contrastive Learning for Ethical MARL.*

Rath, T. (in preparation, H1). *The Proxy Agency Moral Shield: A Behavioural Study.*

Rath, T. (in preparation, H3). *Sense of Agency Evolves as the Avatar Learns.*

Rath, T. (in preparation, I1). *The Computational Mirror Test: Proxy Agency, Moral Endorsement, and Architecturally Ethical Avatars.*