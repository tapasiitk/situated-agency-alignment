# Research Program Design Decisions — 30 April 2026

**Document purpose:** Summary of methodological decisions, rationales, and action items arising from a full-day design review. Intended as a handover document for collaborators working on T1 (theory paper), M1 (empathy gap diagnostic), and M2/M2' (KARMA intervention papers).

---

## 1. The Logical Problem with the Original Env B Framing in M1

### What the original framing claimed

The original M1 rationale for including both Env A and Env B stated that the A vs. B comparison would let M1 *"distinguish a pure baseline empathy gap from one driven by action ambiguity."* The hypothesis was that zap's dual-use role causes role-confusion in the encoder, driving the empathy gap.

### Why this framing is broken

If Env A (no dual-use, pure harm, `wastespawnrate = 0.0`) already shows the empathy gap at baseline, that **falsifies** the claim that action ambiguity *drives* the gap. The gap exists without action ambiguity. The A vs. B comparison can only show whether action ambiguity *modulates* (amplifies or attenuates) the gap — a weaker and different scientific claim.

### Corrected framing

The correct Env B question for M1, if Env B is retained at all, is:
> *"Does adding a cooperative beam role change the degree of aggressor-victim separability, and where do ZAPWASTE embeddings cluster — on the aggressor side or the forager side?"*

This is a legitimate exploratory question but should be labeled **exploratory-only** in the preregistration, not confirmatory. The confirmatory empathy gap claim stands entirely on Env A.

### Action item for collaborators

- Remove the phrase "driven by action ambiguity" from all M1 framing documents.
- Replace with: *"The Env A condition establishes the empathy gap as a baseline MARL encoder property. The Env B condition (if included) is an exploratory check on whether dual-use action structure modulates role separability."*
- Update the M1 preregistration section 2.2 and the M1 complete guide section 2 ("Why two envs?") accordingly.

---

## 2. Restructured Publication Strategy: Three Papers Instead of Two

### Proposed structure

| Paper | Environment | Core claim | Status |
|-------|-------------|------------|--------|
| **M1** | Env A only | Empathy gap is a baseline MARL encoder property in harm-only worlds | Proceed now |
| **M2** | Env A only | KARMA suppresses aggression via role-invariant representation in harm-only world | Proceed after M1 |
| **M2'** | Env B only | Empathy gap exists in dual-use worlds; KARMA selectively suppresses harm while preserving cooperation | Blocked — see Section 3 |

### Rationale for separating M2 and M2'

The current single M2 design conflates two distinct claims: (a) KARMA suppresses aggression, and (b) KARMA does so *selectively* — preserving cooperative beam use while suppressing aggressive beam use. Claim (b) is the novel contribution and **requires Env B**. Claim (a) can be demonstrated in Env A alone with a cleaner, faster-to-publish paper. Separating them produces two publishable contributions instead of one over-loaded paper.

### M2 (Env A only) — modified Broken Mirror control

The original Broken Mirror condition uses the pair `ZAPAGENT ↔ ZAPWASTE` (semantically scrambled). In Env A, `ZAPWASTE` events never occur (`wastespawnrate = 0.0`), so this control is undefined. The Broken Mirror must be redesigned for Env A:

- **Recommended replacement:** `ZAPAGENT ↔ APPLEEATEN` (binding aggressor with forager — semantically wrong in the same way, and testable in Env A).
- The logic remains intact: only the role-symmetric pairing `ZAPAGENT ↔ BEINGZAPPED` (KARMA) should produce suppression of harm-infliction. The scrambled pairing (`ZAPAGENT ↔ APPLEEATEN`) should produce no improvement or degraded performance.
- The M2 selectivity claim in Env A is narrower: *"KARMA suppresses harm-infliction without degrading reward-seeking (APPLEEATEN rate preserved)."* This is a coherent and defensible claim, though weaker than the beam-role selectivity claim in M2'.

### M2' — the full selective suppression claim

M2' is the paper where KARMA's contribution is fully demonstrated:
- `ZAPAGENT` rate drops (harm suppressed)
- `ZAPWASTE` rate is **preserved** (cooperation maintained)
- Overall `BeamUseRateperagentstep` does not collapse
- System yield (`AvgReturnperagent`, `AppleRateperagentstep`) improves relative to baseline

This requires Env B to produce a stable dual-use Nash equilibrium — see Section 3.

---

## 3. The Env B Prerequisite Chain for M2'

### Why this is a hard prerequisite

For M2' to exist as a paper, Env B baseline must show **all three simultaneously**:
1. `ViolenceRate` — non-zero and sustained (agents learn to aggress)
2. `CooperationRate` — non-zero and sustained (agents also learn to clean)
3. `AvgReturnperagent` — NOT collapsing (the commons does not degenerate to zero yield)

### Current status (updated 2026-04-29)

The original Env B feasibility scout (seed 42 and seed 123, 4000 episodes) showed `ViolenceRate` rising but harvest collapse to near-zero return (−92%) because agents were reward-hacking the per-action `zap_waste_reward = 0.3` cleanup bonus. The symmetric Env B redesign removes that shaping, adds `waste_regrowth_suppression` and `waste_spread_prob`, and lowers initial/dynamic waste density. The **post-spread no-agent ablation** (`scripts/ablate_waste_regrowth.py`, 8 seeds, 2026-04-29) confirms the spread mechanic creates ecologically relevant pressure (waste counts rise when unchecked; apple counts fall under uniform random actions). **Final numeric values are not locked** until a **learned-policy 4k pilot** on `configs/m1_env_B_sc030_sym.yaml` confirms non-collapsing harvest together with non-trivial violence and cooperation. Provisional knobs in-repo: `waste_regrowth_suppression = 0.10`, `waste_spread_prob = 0.02` (subject to amendment after that pilot).

### The prerequisite chain (strictly ordered)

```
1. Interpret post-spread ablation + adjust spread/alpha if learned pilot is dominated or unstable
2. Run 4k feasibility scout on sym Env B (seeds 42 and 123)
      → confirm ViolenceRate non-zero AND CooperationRate non-zero AND no harvest collapse
3. (Exploratory) M1-style diagnostic in Env B if desired — not part of the confirmatory M1 factorial (see Section 7)
4. Run M2' KARMA selective suppression in Env B
```

Step 1 (ablation) is **complete** as of 2026-04-29; tuning continues with step 2.

### Decision gate

- If sym Env B produces the dual-use Nash equilibrium → proceed with all three papers (M1, M2, M2').
- If sym Env B is still degenerate → debug the environment before committing to M2'. M1 and M2 (both Env A only) are **completely independent** of this gate and can proceed immediately.

---

## 4. Prior Literature Gap — Why Dual-Use Is Novel

### What prior literature shows

Leibo et al. 2017 (*Multi-agent Reinforcement Learning in Sequential Social Dilemmas*) demonstrated that independently learning RL agents converge to aggressive equilibria under resource scarcity in the **Gathering** environment, where the beam freezes rivals and has **no cooperative purpose whatsoever**. Hughes et al. 2018 (*Inequity Aversion*) extended this to the Cleanup game, but in Cleanup, cleaning waste is a separate movement-based action; the beam fines rivals and serves no cooperative purpose. No published study has demonstrated an aggressive Nash equilibrium in an environment where the **same beam primitive** can serve both cooperative and competitive ends.

### The ecological motivation for dual-use

A zap affordance that serves no constructive purpose is ecologically implausible and easy for reviewers to dismiss as an artifact. The dual-use design is the correct answer: in real-world agentic AI systems, the same capability (e.g., a trading algorithm's ability to execute rapid transactions) can be cooperative (providing liquidity) or competitive (front-running). The dual-use Harvest environment is the computational abstraction of this real-world pathology, which is why it is more interesting and more defensible than a straight Leibo-style single-use environment.

### T1 citation strategy

The T1 theory paper should cite:
- **Hardin (1968)** — the tragedy of the commons, foundational
- **Leibo et al. 2017** — MARL replication of commons collapse
- **Hughes et al. 2018** — inequity aversion as a partial solution
- **Sekeris (2014)** (*The Tragedy of the Commons in a Violent World*) — analytical paper showing that when violent appropriation is possible, commons collapse is the unique equilibrium even under initial resource abundance. A direct theoretical complement to the MARL empirical results.
- **Perolat et al. 2017**, **Koster et al. 2022** — broader SSD literature

---

## 5. Tragedy of the Commons as the Theoretical Root

### T1 framing (approved structure)

The following two-layer structure should anchor T1 and every M and H paper's background section:

> **Layer 1 (existing):** Hardin (1968) showed that individually rational resource exploitation leads to collective commons collapse. Leibo 2017 showed that independently learning RL agents rediscover this tragedy computationally — no special incentives or coordination required, just resource scarcity and self-interested optimization.

> **Layer 2 (novel contribution of this program):** When the agent is *user-extended* via Proxy Agency, the Proxy Agency moral shield prevents the user from noticing or inhibiting the emerging commons collapse early enough. The user experiences the agent's aggressive behavior as an extension of their own intent (Sense of Agency is preserved), and therefore does not intervene. Social harm accumulates even as the user's SoA remains intact — social ethics degrade before SoA drops. This is the combined problem the program addresses.

### Why this framing is important

The tragedy of the commons is not background decoration in T1. It is the **ecological substrate** on which the Proxy Agency moral shield operates. Without the commons tragedy as the consequence, Proxy Agency is merely a curiosity about SoA measurement. With it, Proxy Agency becomes a mechanism that actively prevents users from detecting and stopping a well-understood catastrophic social dynamic.

### Action item for T1 collaborators

Ensure Hardin (1968) and Sekeris (2014) appear in the introduction of T1, not just the related-work section. The introduction should establish the stakes (commons collapse) before introducing the novel mechanism (Proxy Agency moral shield prevents detection).

---

## 6. Env A Rules — Plain Language Explanation

This section is for collaborators who need to explain the environment to non-specialist readers or reviewers.

### The setup

Four agents share a single 15×15 grid apple orchard. Apples appear randomly in a central patch and regrow probabilistically at a fixed rate. Every apple an agent collects gives it reward. Resources are finite and contested.

### What each agent can do each step

- Move in any direction (up, down, left, right)
- Rotate to face a direction
- Fire a **beam** forward

### What the beam does in Env A

- If the beam hits another agent → that agent is **frozen** for 25 steps (`zaptimeout = 25`) and cannot move, harvest, or act
- The beam does nothing else — there is no waste, no cleanup channel (`wastespawnrate = 0.0`, `dynamicwasteenabled = false`)
- Firing costs nothing (`zapcost = 0.0`) — aggression is instrumentally motivated by apple monopolization, not reward-shaped in either direction (`zapagentreward = 0.0`)

### Why `zapcost = 0.0` is Leibo-faithful (and counter-intuitive)

In Leibo 2017's Gathering environment, the tagging beam has no explicit negative reward. The only cost is the opportunity cost of spending a time step firing instead of moving toward berries. Setting `zapcost = 0.0` replicates this design choice. The counter-intuitive implication — that conflict is "free" — is precisely what makes the tragedy so sharp: there is no direct deterrent to aggression, only the indirect deterrent that everyone is worse off collectively.

### Why commons collapse when agents become violent

The mechanism is a four-step cycle:

1. **Scarcity triggers aggression:** When apple density drops below a threshold, an agent calculates that freezing a rival for 25 steps yields a net apple gain — the monopolization benefit outweighs the one step spent firing.
2. **All agents make the same calculation:** Because all agents are independent learners facing the same incentive structure, they all converge to using the beam under scarcity. This is the Nash equilibrium.
3. **The collective payoff collapses:** When everyone is zapping, everyone is also spending time frozen. Total agent-steps spent harvesting falls sharply. The apple patch, which regrows at a fixed rate, now generates apples that no one is efficiently collecting.
4. **The tragedy:** Each individual agent gets *more relative to frozen rivals* by zapping — so the behavior is individually rational. But the total apple harvest across all four agents is far lower than it would be under mutual cooperation. This is Hardin's tragedy: individually rational, collectively catastrophic.

### The dose-response nature of the effect

The commons collapse is not binary — it is a graded function of apple density. At high scarcity (apple density = 0.05–0.15), aggression is the dominant strategy and yield collapses severely. At low scarcity (apple density = 0.50–0.70), foraging is profitable enough that aggression has diminishing returns and cooperation emerges naturally. This scarcity-dependence is the basis for M1's H3-mod hypothesis (scarcity moderation of role separability).

---

## 7. M1 Design Changes: Scarcity Expansion

### Decision: Remove Env B from M1 confirmatory design

Env B is moved out of M1's confirmatory hypotheses for two reasons:
1. The "action ambiguity drives empathy gap" framing is logically broken (see Section 1).
2. Env B's symmetric design is still pending ablation validation (see Section 3).

Env A alone is sufficient for all four M1 confirmatory hypotheses (H1 existence, H2 asymmetry, H3-mod scarcity, H4 gradient disconnect).

### Decision: Expand from 3 to 5 scarcity levels

With Env B removed, 9 training runs are freed (3 scarcity × 3 seeds × Env B). This budget should be reinvested in more scarcity conditions rather than more seeds.

**Reasoning:**
- The scarcity moderation hypothesis (H3-mod) is M1's most distinctive contribution relative to prior work. Leibo 2017 gestured at scarcity-dependence without quantifying it. Five monotonically ordered conditions produce a far more compelling dose-response figure than three, and they survive the "you only showed 3 conditions" reviewer objection.
- The five Env A scarcity runs are **forward-compatible with M4** (the Scarcity Phase Diagram paper), which does a full grid sweep. Running them now means M4 has five pre-existing Env A reference rows before any additional M4-specific compute.
- The power check at the pilot cell (Env A, sc030, seed 42) confirmed that 3 seeds and 20 eval episodes per checkpoint are sufficient for reliable metrics. Seed variance is not the limiting factor; condition coverage is.

### Recommended 5-scarcity design

| Level | Apple density | Role | Note |
|-------|--------------|------|------|
| sc005 | 0.05 | Extreme scarcity | Scout 500 episodes first — risk of training instability |
| sc015 | 0.15 | High scarcity | Current, pilot-validated |
| sc030 | 0.30 | Medium scarcity | Pilot cell, fully validated |
| sc050 | 0.50 | Low scarcity | Current |
| sc070 | 0.70 | Near-abundance | Negative control — violence should approach zero |

**Total confirmatory runs:** 5 scarcity × 3 seeds = 15 runs (Env A only).

### Seeds: Keep at 3

3 seeds (42, 123, 456) remain the frozen seed list. This is the standard minimum in MARL papers. Bootstrap 95% CIs across 3 seeds with 1000 resamples are standard. If a reviewer requests more seeds, this is a clean revision-stage amendment covered by the compute reserve budget. Adding seeds pre-registration provides diminishing returns compared to the scientific value of adding scarcity conditions.

### Required scout runs before preregistration freeze

Before the preregistration is frozen, run two cheap scouts:
1. **sc005 scout:** 500 episodes, seed 42, Env A. Confirm training does not destabilize (no NaN loss, no representation collapse, agents explore rather than freeze in place).
2. **sc070 scout:** 500 episodes, seed 42, Env A. Confirm `ViolenceRate` is near zero (negative control validation). If violence is non-trivially present at density 0.70, widen the upper bound to 0.80 or 0.90.

These scouts cost approximately 1 GPU-hour each and should be run before the main campaign begins.

### Updated preregistration grid

The frozen confirmatory design in the M1 complete guide (section 5) should be updated from:

> "Env grid: 2 envs (A, B) × 3 scarcities (0.15, 0.30, 0.50) × 3 seeds = 18 runs"

To:

> "Env grid: 1 env (A) × 5 scarcities (0.05, 0.15, 0.30, 0.50, 0.70) × 3 seeds = 15 runs, subject to scout validation of sc005 and sc070 conditions."

---

## 8. Summary of Action Items for Collaborators

### T1 paper

- [ ] Add Hardin (1968) and Sekeris (2014) to the introduction (not just related work).
- [ ] Structure introduction around the two-layer logic: tragedy of the commons (existing) + Proxy Agency moral shield prevents detection (novel).
- [ ] Do not frame dual-use as a confound — frame it as the ecologically realistic case that motivated the choice of Dual-Use Harvest over a Leibo-style single-use environment.

### M1 preregistration and guide documents

- [ ] Remove "driven by action ambiguity" from all Env B framing. Replace with "exploratory modulation check" if Env B is mentioned at all.
- [ ] Update the confirmatory design grid: 1 env × 5 scarcities × 3 seeds = 15 runs.
- [ ] Add scout validation entries for sc005 and sc070 to the run log template (section 13).
- [ ] Update the paper outline (section 11) to reflect 5 scarcity levels in Figure 1 and Table 1.
- [ ] Add a note in section 2 ("Why two envs?") explaining the decision to remove Env B from confirmatory design, citing the action-ambiguity logic flaw and the pending sym Env B ablation.
- [ ] Update the OSF preregistration metadata (Appendix C) to reflect the new run count (15 instead of 18).

### M2 paper

- [ ] Redesign the Broken Mirror control from `ZAPAGENT ↔ ZAPWASTE` to `ZAPAGENT ↔ APPLEEATEN` for the Env A condition.
- [ ] Narrow the M2 selectivity claim: KARMA suppresses harm-infliction (`ZAPAGENT` rate drops) without degrading reward-seeking (`APPLEEATEN` rate preserved). This is weaker than beam-role selectivity but coherent.
- [ ] Keep all other M2 design elements (3 conditions: Baseline, Broken Mirror, KARMA; 5 seeds; 3k episodes; ablation of lambda).

### M2' (future paper — currently blocked)

- [ ] Do NOT begin M2' training-scale work until the **learned 4k** sym Env B pilot (Section 3, step 2) confirms a non-degenerate dual-use equilibrium. The no-agent ablation (Section 3, step 1) is **done** (2026-04-29).
- [ ] Once unblocked, M2' should restore the original Broken Mirror design (`ZAPAGENT ↔ ZAPWASTE`) because `ZAPWASTE` events exist in Env B.
- [ ] The M2' paper narrative: first establish empathy gap in Env B (from M1 Env B exploratory results or a dedicated M2' diagnostic section), then show KARMA closes it selectively.

### Environment (sym Env B)

- [ ] Run `scripts/ablate_waste_regrowth.py` with the alpha × spread cross-product as the first unblocking action for M2'.
- [ ] Record results in the run log and use them to finalize `wasteregrowthsuppression` and `wastespreadprob` values for all sym Env B configs.

---

### Leibo-Faithful Environment Check (all configs)

The single environment file `karmic_rl/envs/harvest_dual.py` handles both single-use (Env A) and dual-use (Env B) through config alone — no new environment files are needed. The beam in `harvest_dual.py` already uses timeout/freeze (`zaptimeout = 25`), which is Leibo-faithful. Hughes 2018 used a fining beam (deducts reward from rival) — your env does not; it freezes rivals, matching Leibo exactly. The one concrete non-Leibo default that must be fixed is `victimpenalty`:

- [ ] In `karmic_rl/envs/harvest_dual.py`, the default `victimpenalty` is `0.5`, which adds a shaped negative reward for being hit. In Leibo 2017, being zapped carries no direct reward penalty — the cost is purely the opportunity cost of 25 frozen steps.
- [ ] For **all M1 and M2 configs (Env A)**, override `victimpenalty: 0.0` explicitly in every YAML file to be Leibo-faithful.
- [ ] For **all Env B configs** (future M2'), also set `victimpenalty: 0.0` for symmetry — so Env A and Env B differ only in what the beam can target, not in victim shaping.
- [ ] Do **not** create new env files. The existing `harvest_dual.py` with the correct YAML configs is sufficient for all papers (M1, M2, M2').

### Repo / prereg protocol sync (2026-04-29)

The **confirmatory** M1 factorial in `docs/M1_complete_guide.md`, `docs/M1_OSF_Preregistration.md`, `docs/m1_experimental_guideline.md`, and `docs/m1_in_plain_words.md` matches Section 7 here: **Env A only**, **5 scarcity levels** (`0.05, 0.15, 0.30, 0.50, 0.70`), **3 seeds**, **15 training runs**. Configs: `configs/m1_env_A_sc005.yaml`, `m1_env_A_sc015.yaml`, `m1_env_A_sc030.yaml`, `m1_env_A_sc050.yaml`, `m1_env_A_sc070.yaml`. Env B YAMLs stay in-repo for **exploratory** / **M2′** work only.

---

*Document compiled: 30 April 2026. Reflects design review discussions between PI and Perplexity AI research assistant. All decisions are pre-data and pre-registration — no confirmatory results have been seen.*
