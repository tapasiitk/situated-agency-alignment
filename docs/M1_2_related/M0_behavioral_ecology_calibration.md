# Stage 0 Behavioral Ecology Calibration

**Status:** pre-preregistration runbook for M1/M2. This document is about finding and freezing a Leibo-like Env A ecology before any confirmatory empathy-gap or KARMA tests.

Broader paper/venue routing map: `docs/meta/literature_mining_map.md`.

**Short form:** replicate aggression -> freeze ecology -> test empathy gap -> only then KARMA.

---

## 1. Purpose

Stage 0 answers one question only:

> Does our Env A baseline produce sustained instrumental aggression under a commons-harvest ecology?

It must be selected using behavior only. Do not choose the ecology by looking at H1/H2 probes, CKA/RSA, LSTM probes, gradient-transfer metrics, or any KARMA/Broken-Mirror intervention result.

Once a candidate passes this behavioral gate, freeze the exact environment settings, seed list, checkpoint cadence, evaluation budget, and inclusion thresholds. Only then run M1 confirmatory empathy-gap analyses under that frozen ecology.

---

## 2. What the Prior Ecology Actually Was

### Leibo et al. 2017: Gathering / sequential social dilemma

Relevant source: [Leibo et al., AAMAS 2017 / UCL open-access record](https://discovery.ucl.ac.uk/10069053/).

Key ingredients that produced aggression:

- Agents collect apples for reward.
- Tagging gives no direct reward or punishment.
- A tagged agent is removed from play for `Ntagged` frames.
- Aggression is measured as beam-use rate.
- Aggression rises when scarce shared apples make exclusion useful.
- Resource abundance is controlled by apple respawn time `Napple`.
- Conflict/exclusion value is controlled by timeout length `Ntagged`.
- Longer planning horizons increase defection because the benefit of tagging is delayed.
- When apples are abundant enough, agents have little reason to tag even with long horizons.
- In Gathering, aggression is harder than peaceful apple collection because it requires targeting the other agent; larger policy capacity made defection easier to learn.

Takeaway for us: do not reward zapping directly. Make zapping useful only because it temporarily excludes competitors from apples.

### Perolat et al. 2017: common-pool resource appropriation

Relevant source: [Perolat et al., NeurIPS/NIPS 2017 proceedings](https://papers.nips.cc/paper/6955-a-multi-agent-reinforcement-learning-model-of-common-pool-resource-appropriation).

Key ingredients:

- Apples are a renewable common-pool resource.
- Regrowth depends on remaining local stock; local depletion can become self-reinforcing.
- Agents can tag one another with a timeout beam.
- Tagging again carries no direct reward or punishment.
- Exclusion can produce territorial control, inequality, and sometimes higher sustainability by reducing effective population pressure.

Takeaway for us: the important ecology is not just "low initial apples"; it is renewable-resource pressure plus exclusion. A candidate that produces violence by reward shaping or victim penalties is not a clean Stage 0 match.

### Hughes et al. 2018: inequity aversion in intertemporal social dilemmas

Relevant source: [Hughes et al., NeurIPS 2018 proceedings](https://papers.nips.cc/paper/7593-inequity-aversion-improves-cooperation-in-intertemporal-social-dilemmas).

Use this mainly as adjacent intervention literature, not as the Stage 0 selector. It reinforces that Harvest/Cleanup-style SSDs are temporally extended, spatial, mixed-motive ecologies. But Stage 0 should calibrate the baseline pathology before adding social preferences, reward modifications, KARMA, or any other mechanism.

---

## 3. Mapping the Literature to This Codebase

Code anchors:

- Environment: `karmic_rl/envs/harvest_dual.py`
- Training entry point: `train_karma.py`
- Shared M1 defaults: `configs/m1_base.yaml`
- Stage 0 scout configs: `configs/stage0_env_A_*.yaml`

Ecology mapping:

| Literature lever | Meaning | Code knob | Stage 0 guidance |
|---|---|---|---|
| `Napple` | Resource abundance / respawn delay | mostly `regrowth_speed`; secondarily `apple_density` | Vary `regrowth_speed`; keep `apple_density` fixed at `0.30` initially. |
| `Ntagged` | Length/value of exclusion | `zap_timeout` | Compare `25` vs `50`. |
| Competition/contact pressure | How often agents contest the same apples | `num_agents`, `grid_size`, patch geometry | Vary `num_agents` on fixed `grid_size: 15`. |
| No direct tagging payoff | Aggression must be instrumental | `zap_agent_reward`, `victim_penalty`, `zap_cost` | Keep all at `0.0`. |
| No cleanup channel | Pure harm-only Env A | waste/cleanup knobs | Keep waste disabled and `zap_waste_reward: 0.0`. |
| Planning horizon | Delayed benefit of exclusion | `gamma` | Hold fixed at `0.99` during Stage 0. |

Important local nuance: in `HarvestDualEnv`, `apple_density` controls the initial central patch density. It is not the cleanest proxy for ongoing abundance. The ongoing flow pressure is better captured by `regrowth_speed`, which scales neighbor-dependent apple regrowth rates.

---

## 4. Scout Grid

All candidates are Env A, baseline-only, no cleanup, no direct zap reward, no victim penalty.

| Candidate | Config | `num_agents` | `apple_density` | `regrowth_speed` | `zap_timeout` | Purpose |
|---|---|---:|---:|---:|---:|---|
| A | `configs/stage0_env_A_A_n4_ad030_rg050_zt25.yaml` | 4 | 0.30 | 0.50 | 25 | Current-ish behavioral anchor |
| B | `configs/stage0_env_A_B_n6_ad030_rg050_zt25.yaml` | 6 | 0.30 | 0.50 | 25 | More contact pressure |
| C | `configs/stage0_env_A_C_n8_ad030_rg050_zt25.yaml` | 8 | 0.30 | 0.50 | 25 | High contact pressure |
| D | `configs/stage0_env_A_D_n6_ad030_rg025_zt25.yaml` | 6 | 0.30 | 0.25 | 25 | Scarce-flow ecology |
| E | `configs/stage0_env_A_E_n6_ad030_rg100_zt25.yaml` | 6 | 0.30 | 1.00 | 25 | Abundance / low-violence control |
| F | `configs/stage0_env_A_F_n6_ad030_rg050_zt50.yaml` | 6 | 0.30 | 0.50 | 50 | Stronger exclusion / timeout |

Initial budget:

- 2 seeds per candidate.
- 2000 episodes per seed.
- Extend promising or ambiguous candidates to 4000 episodes before freeze.

Suggested seed pair:

```bash
42
123
```

---

## 5. Running a Scout Cell

Single run:

```bash
python train_karma.py \
  --config configs/stage0_env_A_B_n6_ad030_rg050_zt25.yaml \
  --mode baseline \
  --seed 42
```

Two-seed sweep for one candidate:

```bash
for SEED in 42 123; do
  python train_karma.py \
    --config configs/stage0_env_A_B_n6_ad030_rg050_zt25.yaml \
    --mode baseline \
    --seed "$SEED"
done
```

Full six-candidate scout:

```bash
for CFG in \
  configs/stage0_env_A_A_n4_ad030_rg050_zt25.yaml \
  configs/stage0_env_A_B_n6_ad030_rg050_zt25.yaml \
  configs/stage0_env_A_C_n8_ad030_rg050_zt25.yaml \
  configs/stage0_env_A_D_n6_ad030_rg025_zt25.yaml \
  configs/stage0_env_A_E_n6_ad030_rg100_zt25.yaml \
  configs/stage0_env_A_F_n6_ad030_rg050_zt50.yaml
do
  for SEED in 42 123; do
    python train_karma.py --config "$CFG" --mode baseline --seed "$SEED"
  done
done
```

Artifacts:

- Training CSV: `results/<stage0-cell>/<config-stem>_baseline_seed<seed>.csv`
- Summary JSON: `results/<stage0-cell>/<config-stem>_baseline_seed<seed>.json`
- Checkpoints: `results/<stage0-cell>/checkpoints/`

---

## 6. Behavioral Metrics to Inspect

Use only these training CSV columns during Stage 0 selection:

- `ViolenceRate_per_agent_step`
- `BeingZappedRate_per_agent_step`
- `AppleRate_per_agent_step`
- `BeamUseRate_per_agent_step`
- `AvgReturn_per_agent`
- `SkippedMinibatches`

Optional late-window behavior summaries may use checkpoint rollouts to count role events, but only for counts/rates:

- `ZAP_AGENT`
- `BEING_ZAPPED`
- `APPLE_EATEN` / apple-rate proxy

Do not use these for Stage 0 selection:

- `measurement_1_probes.probe_agg_vs_vic_auroc`
- `measurement_1b_lstm_probes.*`
- `measurement_2_cka.*`
- `measurement_3_rsa.*`
- `measurement_4_gradient_transfer.*`
- Any KARMA or Broken Mirror comparison

---

## 7. Pass/Fail Gate

A candidate can be frozen for M1 only if it satisfies all of the following:

1. Sustained non-zero violence late in training.
2. Sustained non-zero being-zapped rate late in training.
3. Enough aggressor/victim event mass for later H1/H2 late-window analysis.
4. No complete harvest collapse; apple rate and return must remain interpretable.
5. Ecological direction is sensible: the abundance control should not be more violent than the scarcity/contact/timeout conditions in a way that undermines the interpretation.
6. The behavior is generated under instrumental-zap settings: no direct zap reward, no victim penalty, no cleanup reward.

Failure modes:

- **All candidates peaceful:** increase contact pressure or timeout; consider lower regrowth only if apple-rate does not collapse.
- **All candidates collapse harvest:** increase regrowth or reduce `num_agents`.
- **Violence appears only with degenerate returns:** do not freeze; the empathy-gap test needs enough social events without destroying the task.
- **Only shaped configs produce violence:** reject for Stage 0; that would not replicate the Leibo/commons mechanism.

---

## 8. Freeze Record Template

When a cell passes, add a dated freeze entry to `docs/M1_2_related/design_decisions.md`:

```markdown
## YYYY-MM-DD Addendum: Stage 0 Ecology Freeze

Frozen Env A cell(s):

- Config(s):
- Seeds:
- Episodes:
- Checkpoint cadence:
- Late window:
- Behavioral evidence:
- Exclusions:
- Confirmatory M1 metrics now allowed:

Decision:
This freezes the ecology for M1 baseline-only empathy-gap testing. M2/KARMA remains blocked until M1 is evaluated under these same settings.
```

---

## 9. Recommended Next Step

Run A, B, D, and E first. This gives a compact contrast:

- A: current-ish anchor
- B: contact-pressure increase
- D: scarce-flow increase
- E: abundance control

If B or D shows sustained violence without harvest collapse, run C and F to test whether additional crowding or longer exclusion improves the behavioral gate. If none of A/B/D produces violence, C and F become the next diagnostic pair.
\««««««\
