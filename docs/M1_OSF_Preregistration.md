# OSF Preregistration: M1 Baseline Study
## *Following the OSF Standard Pre-Data Collection Template*

---

# Study Information

## 1. Title
**M1: The Empathy Gap in Baseline MARL — Role-Disjoint Representations in Dual-Use Harvest**

---

## 2. Authors
*(Fill before submission)*
- Principal Investigator: [Your Name], IIT Kanpur
- Institution: [Department], IIT Kanpur
- Collaborators: [If any]

---

## 3. Description

This study investigates whether a baseline recurrent PPO agent (PPO+LSTM) trained in the Dual-Use Harvest multi-agent sequential social dilemma (SSD) environment develops geometrically separable internal representations for aggressor-view (`ZAP_AGENT`) and victim-view (`BEING_ZAPPED`) timesteps — and whether this representational separation temporally precedes and statistically predicts the emergence of aggressive behavior during training.

The Dual-Use Harvest environment is a commons-dilemma grid world where agents can harvest apples (cooperative) or zap other agents (aggressive). A key unanswered question is whether standard MARL encoders naturally encode "who is suffering" as a latent signal, or whether there is a systematic "empathy gap" — a geometric orthogonality between aggressor-view and victim-view latent states that prevents negative feedback from victimization from influencing aggressive policy updates. This M1 study is purely diagnostic (baseline pipeline only, no intervention). It is the motivating study for M2 (KARMA intervention), which is the subject of a separate preregistration.

Codebase: `situated-agency-alignment`. Pipeline scripts: `train_karma.py`, `scripts/rollout_from_checkpoint.py`, `scripts/analyze_checkpoint.py`, `scripts/aggregate_m1.py`, `scripts/plot_m1_trajectory.py`. Full protocol: `docs/m1_experimental_guideline.md` and `docs/m1_reproducibility.md`.

---

## 4. Hypotheses

All hypotheses are **directional** unless noted.

**H1 — Existence / Decodability** *(directional)*
Linear probes trained on frozen encoder embeddings will achieve above-chance role classification by mid-training. Specifically, `measurement_1_probes.probe_5way_auroc_mean > 0.80` at or before episode 2000 in at least one scarcity condition.
*Null: AUROC ≤ 0.50 (chance); the empathy-gap framing is incorrect.*

**H2 — Role Asymmetry / Non-Trivial Separability** *(directional)*
Aggressor-view and victim-view embedding distributions are more dissimilar to each other than either is to a neutral baseline. Specifically: `CKA(ZAP_AGENT, BEING_ZAPPED) < CKA(ZAP_AGENT, NEUTRAL)` and `CKA(ZAP_AGENT, BEING_ZAPPED) < CKA(BEING_ZAPPED, NEUTRAL)`.
Primary metric: `measurement_2_cka.cka_agg_vs_vic` (eligible checkpoints only; see Data Exclusion).
*Null: CKA is symmetric across all role-pairs, no special orthogonality between aggressor and victim.*

**H3 — Behavior-Representation Coupling** *(directional)*
Role-separability trajectory is positively associated with the aggression trajectory over training, and representational separability **precedes** behavioral aggression (positive cross-correlation lag). Primary behavioral series: `ViolenceRate_per_agent_step`, `BeingZappedRate_per_agent_step`. Primary representation series: H1 and H2 metrics.
*Null: no lag or reversed lag — representational change is a consequence, not a precursor, of aggression.*

**H4 — Gradient Disconnect** *(directional)*
Cross-role gradient transfer is weak or negative on average: the cosine similarity between the policy gradient at aggressor-view observations and the value gradient at victim-view observations is approximately zero or negative.
Primary metric: `measurement_4_gradient_transfer.gradient_transfer_cos_mean`.
*Null: gradients are positively aligned — negative feedback from victimization is already flowing into aggressor policy updates.*

**H3-mod — Scarcity Moderation** *(directional)*
Role separability (probe AUROC, 1 − CKA, RSA cosine distance) is greater under higher-scarcity training conditions (lower `apple_density`), paralleling the known scarcity-dependence of aggression in SSD environments.
*Null: no effect of scarcity on representational separability.*

---

# Design Plan

## 5. Study Type

**Observational Study** — Data are collected from agents that are not randomly assigned to a treatment in the classical sense. Scarcity levels and environment conditions are systematically varied (quasi-experimental), but this is a simulation-based observational study of emergent representations and behavior. There are no human subjects.

---

## 6. Blinding

**6.1.1. No blinding is involved in this study.**
This is a simulation study with no human subjects and no experimental manipulation of participants. All analyses are fully automated via the registered pipeline scripts. The analyst (researcher) has access to the training configuration but **does not inspect checkpoint-level outcomes prior to locking the analysis scripts**; all scripts are version-controlled and frozen at the registered Git commit SHA.

---

## 7. Additional Blinding

No additional blinding procedures apply. However, to preserve the spirit of confirmatory analysis, the distinction between **primary (confirmatory)** and **exploratory** metrics is frozen in this preregistration before full-scale data collection (i.e., before running the **1 x 5 x 3 = 15-run** main campaign on **Env A only**). A single pilot cell (`m1_env_A_sc030`, baseline, seed 42) was run to validate the pipeline and calibrate the `n_min` threshold; this is documented as the pilot and does not count as confirmatory data.

---

## 8. Study Design

**Quasi-experimental factorial observational study.**
- Design: fully crossed **1 (Environment: Env A tag-only)** x **5 (Scarcity / `apple_density`)** x **3 (Random seed)** factorial = **15** training runs.
- **Exploratory (non-confirmatory):** Dual-use Harvest (`Env B`) may be run for programme or figure supplements; it is **not** part of the confirmatory N or primary hypothesis tests (see `docs/design_decisions_30Apr2026.md`).
- Unit of analysis: **checkpoint within run** (time-series panel design), aggregated across seeds per **scarcity** cell.
- Mode: **baseline only** (`--mode baseline`). KARMA and Broken Mirror intervention conditions are out of scope for M1.
- The design is **between-conditions** across **scarcity** cells and **within-run** across training checkpoints.
- **Exact confirmatory config files frozen at registration:** `configs/m1_env_A_sc005.yaml`, `configs/m1_env_A_sc015.yaml`, `configs/m1_env_A_sc030.yaml`, `configs/m1_env_A_sc050.yaml`, `configs/m1_env_A_sc070.yaml`, with shared defaults in `configs/m1_base.yaml`. **Pre-campaign scouts:** 500 episodes, seed 42, on `sc005` (training stability) and `sc070` (violence near-zero check); bounds may be amended per `docs/design_decisions_30Apr2026.md` §7 if scouts fail.

**Scarcity factor (`apple_density`):**

| Level | `apple_density` | Role |
|-------|-----------------|------|
| sc005 | 0.05 | Extreme scarcity; scout before full 4k |
| sc015 | 0.15 | High scarcity |
| sc030 | 0.30 | Medium; pilot-validated |
| sc050 | 0.50 | Low scarcity |
| sc070 | 0.70 | Near-abundance; scout (violence ~0) |

**Fixed across all runs:** Env A (tag-only): `zap_waste_reward=0.0`, `waste_spawn_rate=0.0`, `zap_agent_reward=0.0`, `victim_penalty=0.0` per YAML. **Seeds:** 3 per scarcity cell, `[42, 123, 456]`.

**Total confirmatory training runs: 15** (5 scarcity levels x 3 seeds).

**Fixed hyperparameters across all conditions:**
- `grid_size: 15`, `num_agents: 4`, `max_steps: 1000`
- `zap_agent_reward: 0.0`, `victim_penalty: 0.0`, `zap_cost: 0.0`
- `episodes: 4000`, `checkpoint_interval: 200` → 20 checkpoints per run
- `lr: 1.5e-4`, `gamma: 0.99`, `gae_lambda: 0.95`, `clip_ratio: 0.2`
- Eval episodes per checkpoint: **20** (see Sample Size Rationale)
- Config files: `configs/m1_env_*_sc*.yaml` with shared defaults in `configs/m1_base.yaml`

---

## 9. Randomization

Each of the 3 seeds per cell is drawn from the pre-specified seed list **`[42, 123, 456]`**. All runs use the same seed list across cells. PyTorch, NumPy, and environment random seeds are set via the `--seed` argument to `train_karma.py`, which seeds all relevant random number generators. No adaptive or stratified randomization is used; seeds are fixed and pre-specified.

---

# Sampling Plan

## 10. Existing Data

**10.1.1 — Registration prior to creation of data (main campaign).**
As of the date of submission of this preregistration, the main **15-run** campaign data on **Env A only** (5 scarcity x 3 seeds) have **not yet been collected**. A single engineering pilot row (`m1_env_A_sc030`, baseline, seed 42, up to ep4000) was completed to validate the pipeline, calibrate `n_min`, and document the power-check decision. This pilot data is used only for protocol calibration and is explicitly labeled non-confirmatory. **Dual-use Env B** runs, if any, are **exploratory** for M1 and are not part of this confirmatory N. Confirmatory inference will be drawn only from the main campaign data collected after this preregistration is timestamped.

---

## 11. Explanation of Existing Pilot Data

A single pilot cell (`m1_env_A_sc030.yaml`, `--mode baseline`, seed 42) was run to:
1. Validate the full pipeline (train → batch rollout → analyze → aggregate → plot).
2. Determine `n_min` — the minimum number of `ZAP_AGENT` and `BEING_ZAPPED` timesteps per checkpoint required for confirmatory use of the agg-vic CKA and binary probe metrics.
3. Conduct a one-time power check (20 vs. 80 eval episodes at ep4000) to decide whether the main campaign should use 20 or 80 eval episodes per checkpoint.

**Power check decision (documented 2026-04-28, Azure VM `tapsvmT4`):**
The 80-episode equivalent was estimated as the mean of four independent 20-episode rollouts (single-process 80-ep rollout was OOM-killed). Observed deltas vs. the original 20-episode run:
- `probe_5way_auroc_mean`: +0.0115 (0.8135 → 0.8250)
- `probe_agg_vs_vic_auroc`: +0.0511 (0.2571 → 0.3082)
- `cka_agg_vs_vic`: +0.00364 (0.00875 → 0.01239)
- `gradient_transfer_cos_mean`: +0.00625 (−0.10761 → −0.10136)

**Decision: main campaign uses 20 eval episodes per checkpoint.** These shifts are not large enough to justify global escalation to 80.

**`n_min` calibration:** Pilot row showed min(`n_ZAP_AGENT`, `n_BEING_ZAPPED`) at coverage by threshold:
- `n_min = 50` → 19/20 checkpoints pass; `n_min = 100` → 14/20; `n_min = 200` → 5/20 (too sparse).
**Frozen value: `n_min = 100`** for all cells and checkpoints.

---

## 12. Data Collection Procedures

**Training (simulation data collection):**
For each of the **15** confirmatory training runs (**Env A**: 5 scarcity levels x 3 seeds):
1. Launch `python train_karma.py --config configs/m1_env_<X>_sc<Y>.yaml --mode baseline --seed <S>`.
2. Training runs for **4000 episodes** (~16M agent-steps per run on 4 agents at 1000 steps/ep).
3. Checkpoints are saved every 200 episodes: `{200, 400, ..., 4000}` = **20 checkpoints per run**.
4. Training metrics (ViolenceRate, BeingZappedRate, AvgReturn, AppleRate, etc.) are logged every 20 episodes to a CSV and W&B.

**Rollout and embedding extraction (per checkpoint):**
For each checkpoint, run `scripts/rollout_from_checkpoint.py` for **20 evaluation episodes** with the fixed eval seed base (documented in config). This produces a per-checkpoint `.parquet` file with columns: `episode_id`, `step`, `agent_id`, `embedding` (float32[64]), `cnn_features`, `lstm_hidden`, `value`, `action`, `log_prob`, `reward`, `role` (5-class label), `role_multilabel`, `event_details`.

**Batch orchestration:** `bash scripts/batch_m1_trajectory.sh <config> <results_dir> <seed> 20` orchestrates rollout → analyze for all 20 checkpoints per run. Parquet files are stored to scratch disk (`/mnt/karma_m1_scratch` or `/dev/shm/karma_m1_scratch`) and deleted after each successful per-checkpoint analysis to manage disk space.

**Per-checkpoint analysis:** `scripts/analyze_checkpoint.py` reads the `.parquet` and produces an analysis `.json` with all measurement fields. `scripts/aggregate_m1.py` merges all 20 JSONs + the training CSV into a single aggregated CSV per run. This CSV is the unit committed to version control.

**Compute platform:** Azure NC16as_T4_v3 VM (1× NVIDIA T4, 16 vCPUs, ~110 GB RAM). Estimated cost scales with confirmatory N (**15** runs); prior 18-run estimate ~$800–$1,200 is an **upper bound** for the reduced design. Estimated duration: ~8–12 GPU-hours per run x **15** runs ~= **120–180** GPU-hours for the confirmatory campaign.

**Reproducibility record:** For each run, record: Git commit SHA (`git rev-parse HEAD`), config path, seed, exact commands, and output artifact paths in `docs/m1_reproducibility.md` (append-only dated entries).

---

## 13. Sample Size

- **Training runs:** **15** (Env A only: 5 scarcity x 3 seeds).
- **Checkpoints per run:** 20 (episodes 200 to 4000, every 200).
- **Eval episodes per checkpoint:** 20 (fixed; see §11 for power-check justification).
- **Total analysis units:** **300** checkpoint-level data points (15 runs x 20 checkpoints), yielding approximately **6.0M** labeled (observation, role, embedding) tuples across the full confirmatory campaign (scaled from the prior 18-run ~7.2M estimate).
- **Minimum usable checkpoints per run for role-comparison primaries:** those with `n_ZAP_AGENT ≥ 100` AND `n_BEING_ZAPPED ≥ 100` (`n_min = 100`); pilot shows ~14/20 checkpoints pass per seed in Env A sc030.

---

## 14. Sample Size Rationale

The sample size is **fixed-compute** rather than power-analysis derived, as is standard for simulation studies with known computational constraints:
- **3 seeds per cell** is the minimum confirmatory design adopted for this registration, balancing seed-level variance estimation with execution speed and compute budget.
- **20 eval episodes per checkpoint** was calibrated via the pilot power check (§11): escalating to 80 episodes produced metric shifts well within interpretable noise, so 20 is sufficient for trajectory-level inference with the `n_min = 100` eligibility rule.
- **4000 training episodes** (16M agent-steps per agent) was chosen to see clear aggression plateaus and provide sufficient signal for H3 (temporal precedence): Leibo-family aggression curves need ≥10⁶ agent-steps; 1.6×10⁷ is well above this.
- **5 scarcity levels** (0.05, 0.15, 0.30, 0.50, 0.70) strengthen the H3-mod dose-response figure relative to 3 levels; compute budget was reallocated from removing Env B from the confirmatory factorial (see `docs/design_decisions_30Apr2026.md` §7).
- **20 checkpoints per run** (Δ=200) provides dense temporal sampling for cross-correlation lag estimation (H3/H4).
- Estimated compute: ~$1,300–$1,900 for the full M1 campaign; remaining ~$3,000–$3,700 reserved for M2.

---

## 15. Stopping Rule

The **15-run** confirmatory factorial is a **fixed design** — the exact run count is pre-specified. No early stopping based on observed results is permitted. However, two **pre-specified amendment triggers** exist:

1. **Eval-episode amendment:** If ≥3 consecutive checkpoints in a given cell fail the `n_min = 100` threshold, an amendment escalating from 20 to 80 eval episodes is permitted for that cell (and potentially globally), provided it is filed with a dated, explicit rationale before any confirmatory inference from that cell.
2. **Convergence failure amendment:** If logistic probe convergence failures exceed 10% of fits across a cell, an amendment to solver scaling or `max_iter` is permitted, followed by re-run of the pilot row only before scaling.

These amendments must be documented before confirmatory analysis is run.

---

# Variables

## 16. Manipulated Variables

*(Simulation — quasi-experimental)*

1. **Scarcity** (`apple_density`, treated as categorical, **5** levels on **Env A** confirmatory grid):
   - **0.05** (extreme; scout before full 4k)
   - **0.15** (high)
   - **0.30** (medium)
   - **0.50** (low)
   - **0.70** (near-abundance; scout)

   **Environment type** is **not** manipulated in the confirmatory factorial: all 15 runs use **Env A** (tag-only: `zap_waste_reward=0.0`, `waste_spawn_rate=0.0`). Dual-use **Env B** may be run as **exploratory** or for M2′ prerequisites; it is out of scope for confirmatory M1 hypothesis tests unless a separate amendment registers Env B cells.

2. **Random seed** (3 levels per scarcity cell): `[42, 123, 456]` — controls initialization and stochasticity of training. Seeds are pre-specified, not manipulated for content.

---

## 17. Measured Variables

### Primary (confirmatory):

1. **`measurement_1_probes.probe_5way_auroc_mean`** — Macro-averaged AUROC of a linear logistic regression probe predicting 5-way role label (`ZAP_AGENT`, `BEING_ZAPPED`, `ZAP_WASTE`, `APPLE_EATEN`, `NEUTRAL`) from frozen encoder embeddings (float32[64]). Computed per checkpoint with 5-fold stratified cross-validation. Evaluates H1.

2. **`measurement_2_cka.cka_agg_vs_vic`** — Linear Centered Kernel Alignment (Kornblith et al. 2019) between the `ZAP_AGENT` embedding distribution and the `BEING_ZAPPED` embedding distribution at each checkpoint. Computed only when `n_ZAP_AGENT ≥ 100` AND `n_BEING_ZAPPED ≥ 100`; otherwise tagged `underpowered_at_this_checkpoint` (non-confirmatory). Evaluates H2.

3. **`measurement_4_gradient_transfer.gradient_transfer_cos_mean`** — Mean cosine similarity between (a) `∂V(o_vic)/∂e_vic` — the gradient of the value function with respect to the encoder embedding at victim-view observations — and (b) `∂ log π(ZAP_AGENT|o_agg)/∂e_agg` — the policy gradient at aggressor-view observations. Computed over matched aggressor-victim pairs per checkpoint. Evaluates H4.

4. **`ViolenceRate_per_agent_step`** — Count of `ZAP_AGENT` events per agent per step, from training CSV. Behavioral time series for H3.

5. **`BeingZappedRate_per_agent_step`** — Count of `BEING_ZAPPED` events per agent per step, from training CSV. Paired behavioral time series for H3.

### Secondary / descriptive (non-confirmatory):

- `AvgReturn_per_agent` — Mean episodic return per agent (welfare/convergence check).
- `AppleRate_per_agent_step` — Apple-eating rate (commons health indicator).
- `BeamUseRate_per_agent_step` — Total beam use rate (aggressive + cooperative zap combined).
- `CooperationRate_per_agent_step` — Cooperative zap rate (Env B only; may be near-zero in Env A by design).
- `measurement_1_probes.probe_agg_vs_vic_auroc` — Binary AUROC for the ZAP_AGENT vs. BEING_ZAPPED probe specifically. Reported alongside primary 5-way AUROC; most sampling-sensitive of the primary metrics (see §11).

---

## 18. Indices

1. **5-way probe AUROC** — `probe_5way_auroc_mean` = macro-average of per-class one-vs-rest AUROC scores from a single logistic regression trained on the full 5-class problem. Formula: `mean([AUROC_class_k for k in {ZAP_AGENT, BEING_ZAPPED, ZAP_WASTE, APPLE_EATEN, NEUTRAL}])`.

2. **Linear CKA** — `cka_agg_vs_vic` = `||X_A^T X_V||_F^2 / (||X_A^T X_A||_F × ||X_V^T X_V||_F)` where `X_A` = mean-centered embeddings with role `ZAP_AGENT`, `X_V` = mean-centered embeddings with role `BEING_ZAPPED`, and `||·||_F` is the Frobenius norm. Samples are matched to the smaller class size before computing CKA. (Kornblith et al., ICML 2019.)

3. **Cross-correlation lag (H3)** — Pearson cross-correlation between the `probe_5way_auroc_mean` time series and the `ViolenceRate_per_agent_step` time series, computed per seed per cell. The lag at peak cross-correlation is reported per cell and averaged across seeds. A positive lag indicates representation change precedes behavioral change.

4. **Gradient transfer cosine** — `gradient_transfer_cos_mean` = mean cosine similarity over all matched (victim obs, aggressor obs) pairs in a checkpoint's rollout: `cos(grad_V_vic, grad_π_agg)` where both gradients are in the `embedding` space (dimension 64).

---

# Analysis Plan

## 19. Statistical Models

### H1 (Existence / decodability)
**Model:** 5-fold stratified cross-validated linear logistic regression (`sklearn.linear_model.LogisticRegression`, `max_iter=2000`, `C=1.0`, `solver='lbfgs'`, `multi_class='ovr'`). Input: encoder embeddings (float32[64]). Output: 5-class role label.
**Test:** Report `probe_5way_auroc_mean` as a function of training episode across checkpoints. Confirm H1 if `probe_5way_auroc_mean > 0.80` is achieved at or before episode 2000 in at least **two of the five scarcity** conditions (Env A). No formal significance test for H1 — this is a precision/threshold claim.

### H2 (Role asymmetry)
**Model:** Linear CKA comparison across three role pairs per checkpoint: `CKA(ZAP_AGENT, BEING_ZAPPED)`, `CKA(ZAP_AGENT, NEUTRAL)`, `CKA(BEING_ZAPPED, NEUTRAL)`. Seed-level bootstrap CI (**1000 resamples, percentile interval**) over checkpoint-averaged CKA per cell. Checkpoint eligibility rule (n_min = 100) applied.
**Test:** Confirm H2 if the bootstrap 95% CI of `CKA(ZAP_AGENT, BEING_ZAPPED) − CKA(ZAP_AGENT, NEUTRAL)` is entirely negative in at least 3 of the 6 cells, with the same for the second comparison.

### H3 (Behavior-representation coupling / temporal precedence)
**Primary model/statistic:** Pearson cross-correlation between the **3-checkpoint rolling mean** of `probe_5way_auroc_mean` and the **3-checkpoint rolling mean** of `ViolenceRate_per_agent_step`, computed per seed per cell. Lag range is frozen to **−10 to +10 checkpoints** (−2000 to +2000 episodes). The confirmatory H3 statistic is the **lag at peak cross-correlation**.
**Test:** Confirm H3 if the **mean peak lag** across seeds and cells is positive (representation precedes behavior). Granger causality (statsmodels `grangercausalitytests`, `max_lag=5` checkpoints) is reported only as a **supporting, secondary confirmatory statistic**, not as the primary decision rule for H3.

### H4 (Gradient disconnect)
**Model:** One-sample t-test of `gradient_transfer_cos_mean` values across seeds per cell against `μ₀ = 0` (null = aligned gradients). Two-tailed, `α = 0.05`.
**Test:** Confirm H4 if the mean `gradient_transfer_cos_mean` is not significantly greater than 0 (i.e., fails to reject H₀: cos ≤ 0, or if the t-test against 0 is non-significant for the direction of positive cosine).

### H3-mod (Scarcity moderation)
**Model:** One-way ANOVA (or Kruskal-Wallis if normality violated) on checkpoint-averaged `probe_5way_auroc_mean` across **five scarcity levels** (Env A), across seeds. Post-hoc: pairwise comparisons with Holm correction if ANOVA is significant.
**Test:** Confirm H3-mod if the ANOVA is significant at `α = 0.05` and post-hoc shows higher separability at lower `apple_density` (0.15 > 0.30 > 0.50).

### Multiple comparisons
Primary outcomes are limited to H1–H4 and H3-mod as listed above. All exploratory outputs (§24) are explicitly non-confirmatory and will be labeled as such in the manuscript. No correction for multiple confirmatory hypotheses is applied across H1–H4 (they are conceptually independent), but this is disclosed.

---

## 20. Transformations

- **Class balancing for probes:** The `NEUTRAL` class is downsampled to match the size of the smallest non-neutral role class per checkpoint, to avoid class imbalance inflating AUROC. This subsampling uses a fixed random seed documented in the analysis script.
- **CKA sample matching:** Before computing CKA for any role pair, both embedding matrices are subsampled to `min(n_r1, n_r2)` rows, drawn with a fixed seed per checkpoint.
- **Trajectory smoothing:** A 3-checkpoint (600-episode) rolling mean is applied to behavioral rate time series before cross-correlation computation for H3. The raw (unsmoothed) series are also reported in supplementary figures.
- **No other transformations** are applied. Embeddings are not normalized before probing or CKA (the scripts use the raw float32 outputs of the encoder).

---

## 21. Inference Criteria

- **H1:** Threshold criterion — `probe_5way_auroc_mean > 0.80` in ≥2 of 6 cells by episode 2000 (no p-value).
- **H2:** Bootstrap 95% CI (**1000-resample percentile CI**) for CKA difference (entirely negative = confirmed).
- **H3:** Mean peak lag > 0 checkpoints across seeds/cells; Granger p < 0.05 as supporting evidence only.
- **H4:** One-sample t-test, two-tailed, α = 0.05; H4 confirmed if mean cosine ≤ 0 (or t-test against μ₀=0 is non-significant with a negative trend).
- **H3-mod:** ANOVA/Kruskal-Wallis at α = 0.05 with Holm post-hoc.
- All uncertainty will be reported as **bootstrap 95% percentile CIs across seeds (1000 resamples)** or parametric 95% CIs where explicitly specified above.
- All tests are **pre-specified**; any additional tests in the paper are labeled exploratory.

---

## 22. Data Exclusion

**Checkpoint-level exclusion (role-comparison primaries):**
For confirmatory use of `cka_agg_vs_vic` and `probe_agg_vs_vic_auroc` at a given checkpoint:
- Require `measurement_1_probes.n_aggressor ≥ 100` AND `measurement_1_probes.n_victim ≥ 100`.
- Checkpoints failing this rule are tagged `underpowered_at_this_checkpoint` and excluded from confirmatory analyses of H2. They remain in all other analyses (H1, H3, H4).

**Run-level exclusion:**
- If a run produces fewer than 10 analyzable checkpoints (out of 20) due to persistent failures, that seed is excluded from confirmatory analyses and noted as attrition. The run is **not** replaced with a new seed post hoc.
- If a run produces NaN or constant embeddings across >50% of checkpoints (embedding collapse), the run is excluded and this is reported.

**No outlier removal** beyond the above rules. All valid checkpoints are included. Attrition rates (fraction of checkpoints passing the `n_min` rule, fraction of seeds producing valid runs) are reported per cell.

---

## 23. Missing Data

- **Missing checkpoints:** If a `.pt` checkpoint file is missing, record attrition and omit that checkpoint only (no interpolation). Do not omit the entire seed/run.
- **Rollout/analysis failures:** If `scripts/rollout_from_checkpoint.py` or `scripts/analyze_checkpoint.py` fails for a checkpoint, rerun that checkpoint once with the same config and seed. If it fails again, mark as missing and report.
- **Missing metric fields in JSON:** If a specific measurement field (e.g., `measurement_4_gradient_transfer`) is missing from a checkpoint JSON (e.g., script crashed), that checkpoint is excluded from the corresponding hypothesis test only; other metrics from the same checkpoint are still used.
- The final manuscript will include a **data availability and attrition table** per **scarcity** cell (Env A).

---

## 24. Exploratory Analysis

The following analyses are **pre-labeled exploratory** and will be reported as hypothesis-generating, not confirmatory:

1. **Full 5×5 CKA matrix** across all role pairs at all checkpoints — to examine whether role-pair distances form a structured hierarchy beyond the primary agg-vic comparison.
2. **Prototype cosine-distance matrix** (`measurement_3_rsa.cosdist_agg_vs_vic`) and full RSA analysis — Spearman correlation between embedding distance matrices and a hand-crafted "symmetric role" reference structure.
3. **LSTM hidden state probes** — repeat Measurement 1 using `lstm_hidden` (float32[256]) instead of `embedding`, to test whether role information that is absent from the encoder is present in the recurrent state.
4. **Binary agg-vic probe confusion matrices** — `probe_agg_vs_vic_auroc` analyzed per checkpoint for systematic misclassification patterns (which role is confused as which).
5. **Demographic correlates** — relationships between individual agent behavioral trajectories (within a run) and their representation trajectories.
6. **Env A vs. Env B comparison of role separability** — direct cross-condition comparison of whether the cooperative zap affordance (Env B) masks or reveals the empathy gap.

Results from exploratory analyses may motivate amendments to M2's hypotheses, to be registered separately.

---

# Other

## 25. Other Information

**Scope clarification:** This preregistration covers **M1 only** — baseline pipeline, no intervention. M2 (KARMA intervention) and M3 (environment paper) are separate preregistrations contingent on M1 outcomes.

**Permitted amendments (pre-analysis only):**
1. **Eval-episode amendment:** Escalate from 20 to 80 eval episodes globally if a future power check substantially exceeds the pilot benchmark (see §11). Must be filed with a dated, explicit rationale before confirmatory inference.
2. **Infrastructure amendment:** If persistent runtime instability requires operational changes (scratch path, process management) that do not alter model logic, these may be documented without filing a protocol amendment.
3. No per-checkpoint threshold tuning, no post hoc relabeling of exploratory metrics as primary.

**Related work disclosures:**
- This study uses the `situated-agency-alignment` codebase, which implements the KARMA framework (see `ethical_agentic_AI.pdf`; `karma.pdf`) and the Dual-Use Harvest environment. The M1 baseline study does **not** use KARMA components.
- The theoretical motivation is described in `SoR_reviewed_NS.pdf` (Extended Self / Proxy Agency Moral Shield framework).
- The 6-8 year research program context is described in `research_program_roadmap.md`.

**Null result policy:** If H1 fails (probes cannot decode role from embeddings), the paper will be reframed as a null-result TMLR submission documenting the absence of an empathy gap and its implications for KARMA's motivating assumption.

**OSF Metadata (fill before submission):**
- Contributors: [Names]
- Institution: IIT Kanpur
- Funding: [Grant/source if applicable]
- Ethics/IRB statement: Not applicable (simulation study, no human subjects).
- Preregistration submission date: [DATE]
- Planned data collection start: [DATE]
- Planned analysis completion date: [DATE, approximately 3 months from start]
- Planned submission venue: TMLR / AAMAS 2027 / NeurIPS Workshop
- Git branch at time of registration: `m1-pipeline`
- Git commit SHA at time of registration: [SHA from `git rev-parse HEAD` run immediately before OSF submission]
- W&B project (public): [link]

---

*Document version 1.0. Generated from `m1_experimental_guideline.md`, `m1_preregistration_osf.md`, and `m1_reproducibility.md`, mapped to the OSF Standard Pre-Data Collection Template. Update with actual results, budget burn, and timeline deltas as the campaign progresses.*
