# M1 KT Handoff: Ecology Calibration vs Mechanism Testing Refactor

**Date:** 2026-05-29
**Audience:** Collaborator implementing M1/M0 code changes
**Scope:** Analysis and plotting refactor only. This document does not change the scientific model, training code, or stored results.

---

## 1. One-Sentence Context

The current code still treats M1 as a three-`apple_density` scarcity campaign, but the corrected scientific design separates **M0 behavioral ecology calibration** from **M1 representational mechanism testing**.

Short form:

```text
M0: replicate Leibo-like aggression -> freeze ecology
M1: under frozen ecology, test baseline representational separability and gradient disconnect
M2: only after M1, test KARMA intervention
```

The immediate code task is to stop letting "scarcity on the x-axis" carry the main M1 conclusion.

---

## 2. Why This Refactor Is Needed

### Independent scientific reason

Leibo et al. 2017 did not merely vary "scarcity" in the abstract. Their Gathering result links aggressiveness to resource abundance and exclusion value:

- `Napple`: apple respawn / resource abundance
- `Ntagged`: timeout / value of excluding a competitor
- DV: aggressiveness / beam-use rate

In this codebase, the closest proxy for ongoing resource pressure is **`regrowth_speed`**, not `apple_density`. `apple_density` mostly controls initial central patch density.

Therefore:

- It is appropriate for **M0** to plot aggression against resource-pressure knobs.
- It is not appropriate for **M1's main claim** to be "representational separability vs apple-density scarcity."

### T1 / KARMA reason

T1 motivates KARMA through a two-layer model:

- Tragedy layer: scarcity/exclusion can make aggression instrumentally useful.
- Shield/mechanism layer: the agent and user may fail to transfer victim-side negative value into aggression restraint.

M1 should test the mechanism layer in a behaviorally valid ecology. It should not use scarcity moderation as the main proof of the empathy-gap claim.

---

## 3. Current State Of The Codebase

### Branch / repo state observed

Current working branch when audited:

```text
local M1 preregistration/refactor branch
```

There are many unrelated dirty/deleted/untracked files in the working tree. Do not revert or clean them during this refactor unless explicitly asked.

### Current M1 configs

The current M1 configs vary `apple_density` while keeping `regrowth_speed` fixed:

```text
configs/m1_env_A_sc015.yaml
configs/m1_env_A_sc030.yaml
configs/m1_env_A_sc050.yaml
```

Example:

```yaml
env:
  num_agents: 4
  apple_density: 0.30
  regrowth_speed: 1.0
  zap_timeout: 25
```

This encodes the older assumption that `sc###` is the confirmatory scarcity axis.

### Current Stage 0 configs

Stage 0 configs already encode the better Leibo-like ecology search:

```text
configs/stage0_env_A_A_n4_ad030_rg050_zt25.yaml
configs/stage0_env_A_B_n6_ad030_rg050_zt25.yaml
configs/stage0_env_A_D_n6_ad030_rg025_zt25.yaml
configs/stage0_env_A_H_n6_ad030_rg075_zt25.yaml
configs/stage0_env_A_I_n6_ad030_rg100_zt50.yaml
```

These vary:

- `num_agents`
- `regrowth_speed`
- `zap_timeout`

while holding:

- `apple_density: 0.30`
- no direct zap reward
- no victim penalty
- no cleanup channel

This is closer to the corrected M0 design.

### Current aggregation code

`scripts/aggregate_m1.py` currently parses ecology metadata only from filenames like:

```text
m1_env_A_sc030
```

Current output column:

```text
scarcity = 0.30
```

Problem: this means "apple_density from filename," not calibrated resource pressure. Stage 0 names like `rg075_zt25` are not parsed into structured variables.

### Current confirmatory plotting code

`scripts/plot_m1_confirmatory_figures.py` currently hard-codes:

```python
CONFIRMATORY_SCARCITIES = (0.15, 0.30, 0.50)
```

It then:

- groups runs by `scarcity`
- treats the three `apple_density` cells as confirmatory cells
- confirms H1/H2 by counting how many scarcity cells pass
- plots several x-axes as `apple_density`
- includes exploratory "role separability vs scarcity" ANOVA/Kruskal logic

This is the main source of possible wrong inference.

### Current separability measurement

`scripts/analyze_checkpoint.py` already contains the correct core measurement:

```text
embedding -> binary role label
ZAP_AGENT vs BEING_ZAPPED
metric: probe_agg_vs_vic_auroc
```

This should remain. The problem is not the probe itself. The problem is the campaign-level interpretation around it.

### Current gradient-transfer measurement

`scripts/analyze_checkpoint.py` also computes:

```text
cosine(grad log pi(ZAP_AGENT | o_agg), grad V(o_vic))
```

This is the right conceptual companion to separability, because separability alone does not prove an empathy gap. A competent agent can distinguish "I fired" from "I was hit"; the stronger claim is that victim-side negative value fails to transfer into aggression restraint.

---

## 4. Desired Scientific State

### M0: behavioral ecology calibration

Purpose:

```text
Find a Leibo-like Env A ecology that produces sustained instrumental aggression.
```

Unit of analysis:

```text
candidate ecology x seed
```

Allowed metrics:

- `ViolenceRate_per_agent_step`
- `BeingZappedRate_per_agent_step`
- `BeamUseRate_per_agent_step`
- `AppleRate_per_agent_step`
- `AvgReturn_per_agent`
- role-event counts only, if needed for later feasibility

Forbidden for M0 selection:

- linear probe AUROC
- CKA
- RSA
- gradient transfer
- KARMA / Broken Mirror comparisons

Correct x-axis or grouping variables:

- `resource_pressure = 1 / regrowth_speed`
- `regrowth_speed`
- `zap_timeout`
- `num_agents`

Use `apple_density` only as a fixed context variable unless a later experiment explicitly manipulates it.

### M1: baseline mechanism test

Purpose:

```text
Under a frozen aggression-producing ecology, test whether baseline agents show
aggressor-victim representational separability and cross-role gradient disconnect.
```

Unit of analysis:

```text
seed x checkpoint
```

Main design:

```text
1 frozen ecology x 5 seeds x checkpoints
```

Primary metrics:

- H1: `measurement_1_probes.probe_agg_vs_vic_auroc`
- H2: `measurement_4_gradient_transfer.gradient_transfer_cos_mean`

Main x-axis for trajectory figures:

```text
training checkpoint / episode
```

Main summary dimension:

```text
seed-level late-window means
```

Scarcity/resource-pressure moderation is allowed only as robustness or exploratory work after M0/M1 are clean.

---

## 5. Desired Code Features

### Feature A: structured ecology metadata

Aggregation should output explicit ecology columns:

```text
env
apple_density
regrowth_speed
resource_pressure
zap_timeout
num_agents
ecology_id
legacy_sc_from_filename
```

Recommended `ecology_id` format:

```text
envA_n6_ad030_rg075_zt25
```

The aggregator should prefer reading YAML config metadata from each analysis JSON's `config_path`. Filename parsing can remain as fallback only.

### Feature B: separate M0 plotting script

Add or refactor into a behavior-only M0 script, e.g.

```text
scripts/plot_m0_ecology_calibration.py
```

Inputs:

- Stage 0 training CSVs
- optionally aggregate CSVs if role-event counts are needed

Outputs:

- behavior by ecology candidate
- behavior by `resource_pressure`
- behavior by `zap_timeout`
- apple/return sanity panels
- freeze-candidate summary CSV/JSON

This script must not read or plot representational metrics.

### Feature C: M1 mechanism plotting should not require multiple scarcity cells

Refactor `scripts/plot_m1_confirmatory_figures.py` or create a new script:

```text
scripts/plot_m1_mechanism_frozen_ecology.py
```

Inputs:

- one or more aggregate CSVs from the same frozen ecology

Outputs:

- `m1_behavior_trajectory_by_seed.png`
- `m1_h1_binary_probe_trajectory_by_seed.png`
- `m1_h1_late_window_seed_forest.png`
- `m1_h2_gradient_transfer_trajectory_by_seed.png`
- `m1_h2_late_window_seed_forest.png`
- `m1_nmin_attrition_by_seed.png`
- `summary_m1_mechanism.json`

Main logic:

- group by `seed`, not by `scarcity`
- average over late-window eligible checkpoints per seed
- bootstrap CI over seed-level means
- report attrition per seed

### Feature D: keep ecology moderation as a separate exploratory script

If desired, keep a separate exploratory script:

```text
scripts/plot_m1_exploratory_ecology_moderation.py
```

It should:

- use `resource_pressure`, not `apple_density`, as the primary x-axis
- label all outputs `exploratory`
- avoid campaign-level confirmatory pass/fail language

### Feature E: clearer result summaries

Every summary JSON should explicitly state:

```json
{
  "analysis_layer": "M0_behavioral_ecology" | "M1_frozen_ecology_mechanism" | "exploratory_ecology_moderation",
  "confirmatory_scope": "...",
  "x_axis_semantics": "...",
  "primary_unit": "...",
  "uses_representation_metrics_for_selection": false
}
```

This prevents later accidental overclaiming.

---

## 6. Concrete File-Level Change Checklist

### `scripts/aggregate_m1.py`

Current:

- Parses `env` and `scarcity` only from `env_A_sc030` style names.
- Sorts by `env`, `scarcity`, `seed`, `episode`.

Desired:

- Read `config_path` from flattened analysis JSON when present.
- Parse YAML `env` fields into metadata columns.
- Preserve legacy filename parsing as fallback.
- Rename or duplicate current `scarcity` as `legacy_apple_density_scarcity`.
- Add `resource_pressure = 1.0 / regrowth_speed`.
- Sort by `ecology_id`, `seed`, `episode`.

### `scripts/plot_m1_confirmatory_figures.py`

Current:

- Hard-codes confirmatory scarcities.
- Groups by scarcity cells.
- H1/H2 campaign confirmation is based on number of scarcity cells passing.
- Several plots use `apple_density` as x-axis.
- Scarcity moderation logic lives in the same script as confirmatory H1/H2.

Desired:

- Either refactor this script to frozen-ecology mode or leave it as historical and create `plot_m1_mechanism_frozen_ecology.py`.
- H1/H2 should aggregate over seeds inside one frozen ecology.
- The main H1/H2 plots should use training episode or seed-level forest plots, not apple density.
- Remove campaign pass/fail rules based on scarcity cells.
- Move scarcity/resource-pressure moderation to an exploratory script.

### `configs/m1_env_A_sc*.yaml`

Current:

- Main campaign configs vary `apple_density` and fix `regrowth_speed: 1.0`.

Desired:

- Do not use these as the next confirmatory M1 configs.
- After M0 freeze, create one canonical frozen-ecology config with the selected Stage 0 settings.
- Suggested filename pattern:

```text
configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml
```

or, if candidate B wins:

```text
configs/m1_env_A_frozen_n6_ad030_rg050_zt25.yaml
```

Set:

```yaml
training:
  episodes: 4000
logging:
  checkpoint_interval: 200
```

Run five seeds under the same config.

### `scripts/analyze_checkpoint.py`

Current:

- Binary agg-vic probe is conceptually correct.
- Gradient transfer is conceptually relevant.

Desired near-term:

- No urgent change required for the refactor.
- Ensure downstream scripts treat H1 as separability and H2 as transfer/disconnect.

Desired later robustness:

- Add raw-observation and/or CNN-feature probe controls to address the reviewer objection: "of course I fired and I was hit are visually different."
- Consider reporting whether separability remains meaningful after controlling for immediate action/event confounds.

---

## 7. Recommended New Analysis Semantics

### M0 pass/fail

A candidate ecology can be frozen only if:

- late-window violence is sustained and non-zero
- being-zapped rate is sustained and non-zero
- apple rate and return are not degenerate
- aggression is produced without direct zap reward, victim penalty, or cleanup reward
- role-event counts are sufficient for M1 feasibility

No representational metric participates in this decision.

### M1 H1

Claim:

```text
Baseline agents encode aggressor and victim social roles as separable in learned latent space.
```

Metric:

```text
late-window binary AUROC for ZAP_AGENT vs BEING_ZAPPED
```

Aggregation:

```text
per checkpoint -> per seed late-window mean -> bootstrap CI over seeds
```

Interpretation:

- High AUROC supports role separability.
- It does not by itself prove an empathy gap.

### M1 H2

Claim:

```text
Victim-side value gradients do not meaningfully align with aggressor-side restraint.
```

Metric:

```text
late-window gradient-transfer cosine
```

Aggregation:

```text
per checkpoint -> per seed late-window mean -> bootstrap CI or margin rule over seeds
```

Interpretation:

- Near-zero or weak alignment supports the KARMA-relevant gap.
- Strong positive alignment means victim feedback may already transfer, weakening the KARMA motivation.

---

## 8. Acceptance Criteria For The Refactor

A collaborator is done when:

1. M0 ecology plots can be generated without reading any representation metrics.
2. Aggregated CSVs contain explicit ecology columns from YAML, not only filename-derived `scarcity`.
3. M1 H1/H2 can be run on a single frozen ecology with multiple seeds.
4. No main M1 output file or title implies that `apple_density` scarcity moderation is the confirmatory result.
5. Any resource-pressure or scarcity moderation plot is clearly named exploratory.
6. Summary JSON states the analysis layer and x-axis semantics.
7. Existing single-run trajectory plots still work for debugging.

---

## 9. What Not To Claim After The Refactor

Do not claim:

```text
M1 proves aggression increases with apple_density scarcity.
```

Do claim:

```text
M0 calibrates an aggression-producing commons ecology.
```

Do not claim:

```text
Separability alone proves an empathy gap.
```

Do claim:

```text
Separability plus weak cross-role transfer motivates the KARMA intervention.
```

Do not claim:

```text
KARMA preserves Sense of Agency.
```

Do claim:

```text
KARMA is designed to test whether representational intervention can suppress aggression while preserving task performance; SoA preservation requires later human-facing H/I studies.
```

---

## 10. Suggested Implementation Order

1. Update `aggregate_m1.py` metadata extraction.
2. Add M0 behavior-only plotting script.
3. Add M1 frozen-ecology mechanism plotting script.
4. Relegate current `plot_m1_confirmatory_figures.py` to historical/legacy or update its labels.
5. Create frozen M1 config after M0 freeze.
6. Run one smoke aggregate through the new scripts.
7. Only then launch or re-launch full five-seed frozen-ecology M1.

2
