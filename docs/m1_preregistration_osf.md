# OSF Preregistration Draft (M1)

## Title
**M1: The Empathy Gap in Baseline MARL — Role-Disjoint Representations in Dual-Use Harvest**

## Registration Type
Preregistered quantitative study (confirmatory primary analyses + pre-labeled exploratory analyses).

## Project Context
This preregistration covers **M1 only** (baseline pipeline). It does **not** include KARMA or Broken Mirror interventions (those are M2+).

Codebase: `situated-agency-alignment`  
Main trainer: `train_karma.py`  
Pipeline scripts: `scripts/rollout_from_checkpoint.py`, `scripts/analyze_checkpoint.py`, `scripts/aggregate_m1.py`, `scripts/plot_m1_trajectory.py`  
Protocol references: `docs/m1_experimental_guideline.md`, `docs/m1_reproducibility.md`

---

## 1) Research Question
In a baseline recurrent PPO agent trained in Dual-Use Harvest, are aggressor-view (`ZAP_AGENT`) and victim-view (`BEING_ZAPPED`) internal representations geometrically separable, and does this separability track/precede aggressive behavior over training?

---

## 2) Confirmatory Hypotheses

### H1 (Existence / decodability)
Frozen encoder embeddings support above-chance role decoding. Primary operational metric: `measurement_1_probes.probe_5way_auroc_mean` over checkpoints.

### H2 (Role asymmetry)
Aggressor-victim representation similarity is limited (non-trivial separability). Primary operational metric: `measurement_2_cka.cka_agg_vs_vic` (with checkpoint eligibility rule below).

### H3 (Behavior-representation coupling)
Role-separability trajectory is associated with aggression trajectory over training. Primary behavioral series: `ViolenceRate_per_agent_step`, `BeingZappedRate_per_agent_step`.  
Primary representation series: H1/H2 metrics.  

### H4 (Gradient disconnect)
Cross-role gradient transfer is weak/negative on average. Primary metric: `measurement_4_gradient_transfer.gradient_transfer_cos_mean`.

---

## 3) Study Design

### 3.1 Conditions included in this prereg
- **Mode:** baseline only (`--mode baseline`)
- **Grid:** env × scarcity × seed
- Planned minimum seeds per cell: **>=3**

### 3.2 Training setup
- Config files: `configs/m1_env_*_sc*.yaml` (with shared defaults in `configs/m1_base.yaml`)
- Episodes per run: **4000**
- Checkpoint interval: **200**
- Checkpoints analyzed: `{200, 400, ..., 4000}` (20 checkpoints per run)

### 3.3 Evaluation rollout setup
- Default eval episodes per checkpoint: **20**
- One-time power check at ep4000 may trigger protocol amendment to 80 for all checkpoints (if predefined tolerance is exceeded; see Amendments section).

### 3.5 Completed pilot power check (ep4000)
Completed on the pilot row `m1_env_A_sc030`, baseline, seed 42. A direct single-process `--episodes 80` rollout was operationally unstable on the VM, so the power check was completed as **4 independent 20-episode rollouts** with distinct seed bases and then summarized as an **80-eval equivalent** by averaging the four part-level analysis JSONs.

Observed comparison against the original 20-episode ep4000 analysis:
- `probe_5way_auroc_mean`: `0.8135 -> 0.8250` (delta `+0.0115`)
- `probe_agg_vs_vic_auroc`: `0.2571 -> 0.3082` (delta `+0.0511`)
- `cka_agg_vs_vic`: `0.00875 -> 0.01239` (delta `+0.00364`)
- `gradient_transfer_cos_mean`: `-0.10761 -> -0.10136` (delta `+0.00625`)

Interpretation for preregistration: these shifts were **not large enough to justify a global amendment** from 20 to 80 eval episodes for all checkpoints. Therefore the main M1 campaign remains at **20 eval episodes per checkpoint**, with `n_min = 100` and attrition reporting. The binary agg-vic probe appears the most sampling-sensitive of the primary metrics and should be interpreted with that caveat.

### 3.4 Unit of analysis
Primary time-series unit is **checkpoint within run**; inference aggregated across seeds per cell, then across cells as specified below.

---

## 4) Data Inclusion / Exclusion Rules

### 4.1 Checkpoint eligibility for role-comparison primaries
For confirmatory use of agg-vic binary probe and agg-vic CKA at a checkpoint:
- `n_ZAP_AGENT >= 100` and
- `n_BEING_ZAPPED >= 100`

In aggregate files these correspond to:
- `measurement_1_probes.n_aggressor`
- `measurement_1_probes.n_victim`

If threshold is not met: tag as `underpowered_at_this_checkpoint` and treat those checkpoint-level role-comparison values as non-confirmatory.

### 4.2 Missing checkpoints
If a `.pt` checkpoint is missing, record attrition and omit that checkpoint only (no ad hoc interpolation).

### 4.3 Failures
If rollout/analyze fails for a checkpoint, rerun that checkpoint once using the same config/seed; if still failing, mark missing and report.

---

## 5) Outcomes

## 5.1 Primary outcomes (confirmatory)
1. `measurement_1_probes.probe_5way_auroc_mean`
2. `measurement_2_cka.cka_agg_vs_vic` (eligible checkpoints only per Section 4.1)
3. `measurement_4_gradient_transfer.gradient_transfer_cos_mean`
4. Behavioral trajectories from training CSV merge:
   - `ViolenceRate_per_agent_step`
   - `BeingZappedRate_per_agent_step`

## 5.2 Secondary / descriptive
- `AvgReturn_per_agent`
- `AppleRate_per_agent_step`
- `BeamUseRate_per_agent_step`

## 5.3 Exploratory (pre-labeled)
- Full CKA matrix entries
- Prototype cosine-distance matrix and `measurement_3_rsa.cosdist_agg_vs_vic`
- Additional probe slices beyond predefined primaries

---

## 6) Analysis Plan

### 6.1 Pipeline
Per run:
1. Train baseline (`train_karma.py`)
2. Rollout each checkpoint (`scripts/rollout_from_checkpoint.py`, usually via `scripts/batch_m1_trajectory.sh`)
3. Analyze each checkpoint (`scripts/analyze_checkpoint.py`)
4. Aggregate (`scripts/aggregate_m1.py`)
5. Plot (`scripts/plot_m1_trajectory.py`)

### 6.2 Aggregation levels
- Checkpoint-level metrics within run
- Seed-level summaries within each env×scarcity cell
- Cell-level summaries across grid

### 6.3 Statistical summaries
- Report mean and uncertainty across seeds (CI or bootstrap CI, pre-specified in final analysis notebook/script)
- Report checkpoint attrition rates due to `n_min` rule
- For trajectory association (H3), compute pre-specified lag/association summary (e.g., cross-correlation lag estimate) consistently across runs

### 6.4 Multiple comparisons
Primary outcomes limited to Section 5.1. Exploratory outputs are explicitly non-confirmatory.

---

## 7) Power / Precision Notes
This prereg follows a fixed-compute design under checkpointed trajectories. Precision for role-comparison primaries is controlled by:
- seed count per cell (>=3),
- checkpoint eligibility rule (`n_min=100`),
- potential eval-episode amendment (20 -> 80 globally if power check indicates instability).

---

## 8) Deviations and Amendments

Amendments are allowed only with dated, explicit rationale and before confirmatory inference:

1. **Eval episodes amendment:** 20 -> 80 for all checkpoints only if a future documented power check exceeds the pilot benchmark above by a clearly larger margin than observed here.
2. **Infrastructure amendment:** if persistent runtime instability, document operational changes (e.g., scratch path, tmux/nohup) that do not alter model logic.
3. No per-checkpoint threshold tuning, no post hoc relabeling of exploratory metrics as primary.

---

## 9) Reproducibility and Artifacts

Record for each run:
- Git commit SHA
- Config path
- Seed
- Exact commands used
- Output artifact paths

Recommended run log location: `docs/m1_reproducibility.md` (append-only dated entries).

---

## 10) Planned Deliverables
- Aggregated per-run CSVs under `results/.../aggregated_*.csv`
- Plots per run under `results/.../plots*/`
- Final paper figures/tables derived from preregistered primary outcomes

---

## 11) OSF Metadata (to fill before submission)
- Contributors:
- Institution:
- Funding:
- Ethics/IRB statement (if applicable):
- Start date:
- Planned analysis completion date:

