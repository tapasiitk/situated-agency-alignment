# M1 Experimental Guideline
**The Empathy Gap: Role-Disjoint Representations in MARL Agents**

*A concrete, step-by-step protocol for running the M1 diagnostic study on an Azure NC16as_T4_v3 VM within a $5000 compute budget. This guideline takes M1 from "paper idea" in the program roadmap to submission-ready manuscript in approximately 3 months.*

---

## 0. Executive summary

**Paper one-line claim**: In a standard multi-agent PPO+LSTM agent trained on a dual-use commons SSD, aggressor-view (ZAP_AGENT) and victim-view (BEING_ZAPPED) observations are encoded as geometrically separable latent states. This representational orthogonality temporally precedes, and statistically predicts, the emergence of aggressive beam use across training.

**Why this is a self-contained paper**: M1 requires only the baseline training pipeline (already working on your branch `phase-a-stability-logging-m1`). It does not require KARMA, Broken Mirror, or any intervention. If M1 passes, it is the motivation for M2 (KARMA intervention). If M1 fails — i.e., the baseline's embeddings are already role-invariant — you learn this before investing in M2 and pivot accordingly.

**Deliverable in 3 months**:
- A pre-registered TMLR / AAMAS 2027 / NeurIPS Workshop submission with 5 main figures, 2 tables, and a released dataset of trained-baseline checkpoints + rollout embeddings.
- An arXiv pre-print.
- A clean, reproducible repo snapshot tagged `m1-submission`.

**Estimated compute cost**: ~$600–1200 on Azure NC16as_T4_v3 at $1.20/hour. Well within the $5000 budget, leaving headroom for replication, reviewer-requested ablations, and M2 start.

---

## 1. Hardware and budget framing

### 1.1 VM profile (assumed)

- **Azure NC16as_T4_v3**: 1× NVIDIA T4 (16 GB VRAM), 16 vCPUs, ~110 GB RAM, ~$1.20/hour on-demand (India/US regions).
- **Storage**: assume ~256 GB SSD; budget for ~50 GB of checkpoints + rollouts.
- **Network**: enough for git pull/push, wandb, arXiv uploads.

### 1.2 Budget envelope

| Item | Estimate |
|---|---|
| Training campaign (30 runs × ~8–12 h each) | ~$360–430 |
| Rollout + embedding extraction (30 runs × ~1 h each) | ~$35 |
| Analysis + re-runs + debugging (~2× factor) | ~$400–900 |
| Reviewer-requested ablations (reserve) | ~$500 |
| **Subtotal** | **~$1,300–1,900** |
| **Remaining headroom (for M2)** | **~$3,000–3,700** |

Your $5000 is enough for M1 *and* the starting runs of M2. Do not use more than 40 % of the budget on M1.

### 1.3 Cost-control rules

1. **Stop the VM when idle.** Training queue empty = VM down. Use the Azure CLI or a cron to auto-deallocate after 30 min of idleness.
2. **Prefer spot instances** for long campaigns if the region allows (~70 % discount, tolerates checkpointing).
3. **Log everything to a cheap S3/blob bucket**, not to the VM disk, so you can tear the VM down without losing data.
4. **Don't run the analysis on the training VM.** Do analysis locally on your laptop or a free Colab — it's CPU/memory-bound, not GPU-bound.

---

## 2. Research question and falsifiable predictions

### 2.1 Primary research question

*In a baseline recurrent PPO agent trained in Dual-Use Harvest, are internal representations of ZAP_AGENT timesteps and BEING_ZAPPED timesteps geometrically separable — i.e., orthogonal/disjoint — and does the degree of separability (i) persist throughout training, (ii) correlate across seeds with emergent aggression, and (iii) statistically Granger-precede the behavioural rise in aggression?*

### 2.2 Pre-registerable predictions

**P1 (existence)**. Linear probes trained on frozen encoder embeddings achieve high accuracy (> 80 %) on a 5-way role-classification task (ZAP_AGENT, BEING_ZAPPED, ZAP_WASTE, APPLE_EATEN, NEUTRAL) by mid-training. *If this fails, the "empathy gap" framing is wrong and the paper pivots to a null-result paper.*

**P2 (asymmetry)**. CKA(ZAP_AGENT, BEING_ZAPPED) is lower than CKA(ZAP_AGENT, matched-neutral) and lower than CKA(BEING_ZAPPED, matched-neutral) — i.e., the two role-symmetric views are further apart than either is to a neutral baseline. *Null prediction: if CKA is symmetric across role comparisons, the role-orthogonality claim is not well-supported.*

**P3 (scarcity effect)**. The role separability (probe accuracy, 1 − CKA, RSA distance) is greater under higher-scarcity training conditions, paralleling the known scarcity-dependence of aggression emergence. *Null: no effect of scarcity → representations are shaped by the environment structure alone, not the competitive dynamics.*

**P4 (temporal precedence)**. Across training, role separability rises *before* ZAP_AGENT rate rises. Measured by cross-correlation lag between the two time series. *Null: no lag or reverse lag → representation change is a consequence, not a cause, of aggression.*

**P5 (cross-role gradient disconnect)**. The gradient of the value function at victim-view observations, projected into encoder-embedding space, is approximately orthogonal to the policy gradient of ZAP_AGENT at aggressor-view observations. *If the two gradients are aligned, negative feedback from victimization is already flowing into aggression suppression, and KARMA has no mechanistic room to help.*

These five predictions anchor five of the paper's main figures.

### 2.3 Measurement pilot checklist (paste into OSF / AsPredicted)

Lock this protocol **before** scaling to the full 2×3×5 factorial, or register it as the **pilot phase** with a dated amendment for the main campaign.

**Training (single cell, pilot scope)**

- Config: `m1_env_A_sc030.yaml` (or cite the frozen YAML path + git commit).
- Episodes: **4000** (primary pilot). Optional robustness row: **10000** with the same env/scarcity and **≥2** additional seeds.
- Seeds: **≥3** for variance; report all seeds.

**Checkpoints analyzed (representation trajectory)**

- Analyze **every** checkpoint produced by this run: **`{200, 400, 600, …, 4000}`** (i.e. **Δepisode = 200**, **20** checkpoints, matching `checkpoint_interval: 200`).
- Do not subsample checkpoints ad hoc. If a `.pt` file is missing, record **attrition** and omit that point only.

**Eval rollouts per checkpoint**

- **Primary:** **20** eval episodes per checkpoint (fixed eval seed base; document in registry).
- **Power check (one-time, ep4000 only):** repeat with **80** eval episodes. If **primary** metrics (below) move beyond a pre-specified tolerance, adopt **80** eval episodes for **all** checkpoints and file a **protocol amendment**.

**Minimum data for role metrics (pre-specified)**

- After each rollout, record **`n_ZAP_AGENT`** and **`n_BEING_ZAPPED`** (row counts in the rollout table).
- **Primary** CKA(agg↔vic) and **binary** aggressor-vs-victim probe run **only if** `n_ZAP_AGENT ≥ 200` **and** `n_BEING_ZAPPED ≥ 200`. Otherwise tag **`underpowered_at_this_checkpoint`** and do **not** treat those two as confirmatory at that time point.

**Primary vs exploratory (freeze labels)**

- **Primary (pilot):** time series of **ViolenceRate** / **BeingZappedRate** from training CSV; **5-way** probe AUROC; **CKA_agg_vs_vic** (when the **n** rule is met); **mean gradient-transfer cosine** (when the script completes).
- **Exploratory:** full CKA matrix, prototype distances, ad hoc probe cuts — **hypothesis-generating** until the registry is updated.

**Analysis cadence**

- Per checkpoint: `rollout_from_checkpoint.py` → `analyze_checkpoint.py`; retain Parquet + JSON paths. No within-checkpoint row cherry-picking beyond what the scripts document.
- **Batch (existing 4k run, no retrain):** from repo root,  
  `bash scripts/batch_m1_trajectory.sh configs/m1_env_A_sc030.yaml results/m1_env_A_sc030 42 20`  
  writes analysis JSONs under `results/.../analysis/trajectory_*/` (OS disk). **Temporary** **`.parquet`** rollouts use a **scratch** directory (not the project `results/` tree): prefers **`/mnt/karma_m1_scratch`** if that path exists and is writable (on Azure NC VMs run once: `sudo mkdir -p /mnt/karma_m1_scratch && sudo chown "$USER:$USER" /mnt/karma_m1_scratch`), else **`/dev/shm/karma_m1_scratch`** (tmpfs), else falls back to `results/.../rollouts/`. Parquets are **removed** after each successful `analyze_checkpoint.py`. Rerun **one** checkpoint: fifth arg `4000`. Override: `M1_SCRATCH_ROOT=/path bash scripts/batch_m1_trajectory.sh ...`

**Stop / amend rules**

- If **≥3 consecutive** checkpoints fail the **n** threshold, amend eval episodes to **80** (or **100**) for the remainder of the pilot and register the amendment.
- If logistic / probe **convergence failures** exceed **10%** of fits, amend scaling / solver / `max_iter`, then re-run **only** the pilot row.

**One-line summary for OSF abstract**

*Pilot: Env A sc0.30; 4000 ep; checkpoints **200:200:4000**; **20** eval eps per checkpoint (→ **80** if power check fails); primary CKA_agg↔vic and binary agg–vic probe only if `n_zap ≥ 200` and `n_victim ≥ 200`; **≥3** seeds.*

---

## 3. Experimental design

### 3.1 Environment configuration

Use `HarvestDualEnv` from your existing branch. Two env conditions to cleanly isolate the "dual-use" effect:

| Env condition | Cleanup (ZAP_WASTE) | Rationale |
|---|---|---|
| **A — Tag-only** | `zap_waste_reward: 0.0`, `waste_spawn_rate: 0.0` | Pure competitive commons, closest to Leibo-family Gathering semantics. This is the cleanest test of the empathy gap. |
| **B — Dual-use** | `zap_waste_reward: 0.3`, `waste_spawn_rate: 0.10` | The full program env. Tests whether the cooperative affordance masks or reveals the role-orthogonality pattern. |

Pin the following across both conditions to avoid confounds:

```yaml
env:
  grid_size: 15
  num_agents: 4          # compromise: larger than Leibo's 2, smaller than program's 6
  max_steps: 1000
  apple_spawn_mode: central_patch
  regrowth_speed: 1.0
  zap_agent_reward: 0.0  # aggression must be instrumental, not shaped
  victim_penalty: 0.0    # Leibo-faithful: being hit is an opportunity cost, not a shaped loss
  zap_cost: 0.0
  zap_timeout: 25
  dynamic_waste_enabled: false

training:
  episodes: 4000
  update_every: 10
  ppo_epochs: 4
  batch_size: 64
  lr: 1.5e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  use_amp: true

logging:
  log_interval: 20
  checkpoint_interval: 200   # critical: need dense temporal sampling
```

**Important change from current defaults**: `victim_penalty: 0.0`. Your current env defaults it to `0.5`, which shapes the aggression pathway and makes the empathy-gap story harder to interpret. Override it explicitly in every M1 config.

### 3.2 Factorial design

| Factor | Levels | Rationale |
|---|---|---|
| Env condition | A (Tag-only), B (Dual-use) | Isolates dual-use effect |
| Scarcity (`apple_density`) | 0.15, 0.30, 0.50 | P3 prediction requires variance |
| Seed | 5 seeds per cell | Adequate for seed-level statistics |

**Total runs**: 2 × 3 × 5 = **30 training runs**.

### 3.3 Training length

**4000 episodes per run** at 1000 steps per episode = 4M environment steps per agent per run × 4 agents = 16M agent-steps per run. On a T4 with your current AMP-enabled pipeline, this should take ~8–12 hours per run based on your existing throughput.

Rationale: your smoke-config runs are 40 episodes (~4 × 10⁴ agent-steps). Leibo / Hughes aggression curves need ≥ 10⁶ agent-steps; I'm specifying 1.6 × 10⁷ to see clear plateaus and to give P4 (temporal precedence) enough signal.

### 3.4 Checkpointing and rollout

- **Checkpoint every 200 episodes** (→ 20 checkpoints per run × 30 runs = 600 checkpoints total).
- At each checkpoint, after the training process exits, run a separate **rollout job** that loads the checkpoint and executes 20 evaluation episodes with policy rolled out in `eval` mode (no exploration noise reduction beyond what your current code does; keep the LSTM hidden state reset per episode).
- During rollout, log per-timestep: observation, action, reward, social-event list, encoder embedding (from `out["embedding"]`), CNN feature (from `out["features"]`), LSTM hidden, value.

That gives you a ~20,000-timestep-per-checkpoint dataset per seed per condition = ~12M labeled (observation, role, embedding) tuples in total. Well within disk budget.

### 3.5 Role labels from social events

For each timestep `t`, derive a role label from the env's `social_events[agent_id]`:

- `ZAP_AGENT` — the agent's beam hit another agent this step (aggressor view).
- `BEING_ZAPPED` — the agent was hit by another's beam (victim view).
- `ZAP_WASTE` — the agent zapped waste this step (cleaner view, env B only).
- `APPLE_EATEN` — the agent ate an apple this step.
- `NEUTRAL` — none of the above. (Subsample this class to avoid class imbalance.)

Your existing `_infer_role` in `karma_agent.py` already does this; extract it to a standalone function to ensure consistency between training-time contrastive labels (when you get to M2) and analysis-time labels.

**Co-occurrence resolution rule**: If an agent both zaps and is zapped in the same step, label both roles (multi-label), but for single-label probes use the priority order BEING_ZAPPED > ZAP_AGENT > ZAP_WASTE > APPLE_EATEN > NEUTRAL (victim salience highest, matching your existing `_infer_role`).

---

## 4. What to log during training

Beyond what `train_karma.py` already logs (`ViolenceRate`, `CooperationRate`, `AppleRate`, `EthicalSelectivity`, etc.), add:

- `ep_being_zapped` — count of BEING_ZAPPED events per episode, rate per agent-step. Derive from env's victim-view events in `infos`. This is needed as a paired metric to `ep_zap_agent`.
- `latent_drift` — L2 norm difference between current encoder weights and checkpoint-0 encoder weights. Cheap sanity check for "are representations moving?".
- `embedding_variance` — variance of `out["embedding"]` on a fixed batch of sampled observations, tracked each log interval. Detects representation collapse.
- Standard PPO metrics: entropy, KL, value loss, actor loss, gradient norm.

Use wandb for the full run-level dashboard; mirror the minimum needed into the crash-safe CSV that your pipeline already writes.

---

## 5. What to log during rollout

The rollout is a separate program from training. It runs on the same VM, loads a checkpoint, and produces a per-checkpoint HDF5 or Parquet file with columns:

| Column | Shape | Notes |
|---|---|---|
| `episode_id` | int | 0..19 |
| `step` | int | 0..999 |
| `agent_id` | str | agent_0..agent_3 |
| `obs_hash` | str | MD5 of obs; dedup key for memoization |
| `embedding` | float32 [64] | `out["embedding"]` |
| `cnn_features` | float32 [F] | `out["features"]`, flattened |
| `lstm_hidden` | float32 [256] | `out["new_hidden"][0]` |
| `value` | float32 | `out["value"]` |
| `action` | int | sampled action |
| `log_prob` | float32 | log π(a|s) |
| `reward` | float32 | reward_t |
| `role` | str | 5-class label |
| `role_multilabel` | int8 [5] | one-hot per class, for multi-label |
| `event_details` | JSON | full social event list |

About 20 MB per checkpoint uncompressed, ~600 MB per run × 30 runs = 18 GB total. Comfortable.

---

## 6. Analysis pipeline — five measurements

Each measurement produces one main-paper figure.

### 6.1 Measurement 1 — Linear probes (P1 test)

For each `(env, scarcity, seed, checkpoint)`, train a **linear logistic regression probe** on (`embedding` → `role`). Stratified train/test split, 5-fold cross-validation. Metric: macro-averaged AUROC and per-class F1.

```python
# pseudo-code
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = rollout_df[['embedding']].to_numpy()    # (N, 64)
y = rollout_df['role'].to_numpy()           # (N,)
probe = LogisticRegression(max_iter=2000, C=1.0)
score = cross_val_score(probe, X, y, cv=5, scoring='roc_auc_ovr_weighted')
```

**Reported result**: probe AUROC as a function of training episode, separately per scarcity level. Panel per env condition. Expected pattern: rising AUROC tracking aggression.

Also: separate binary probe for the **only** pair that matters for the KARMA claim — ZAP_AGENT vs BEING_ZAPPED. Report its AUROC and confusion matrix.

### 6.2 Measurement 2 — CKA between role-conditioned embedding distributions (P2 test)

Centered Kernel Alignment between embedding distributions conditional on role. Use the linear CKA estimator of Kornblith et al. 2019:

```python
# X: (n, d) embeddings with role = r1
# Y: (n, d) embeddings with role = r2
def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, 'fro') ** 2
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return num / den
```

For each checkpoint, compute the 5×5 CKA matrix across all role pairs. Track the diagonal entries ZAP_AGENT↔BEING_ZAPPED and the off-diagonal ZAP_AGENT↔ZAP_AGENT (which is 1.0 but useful for sanity).

Match sample sizes across cells (subsample to min class size) to avoid CKA's sample-size sensitivity.

**Reported result**: a panel of 5×5 CKA heatmaps at checkpoints `{0, 5, 10, 15, 20}` × 2 env conditions. Central story: the ZAP_AGENT × BEING_ZAPPED off-diagonal cell darkens (lower CKA) as training progresses.

### 6.3 Measurement 3 — Representational similarity analysis / RSA (complement to CKA)

For robustness, also compute cosine-distance matrices between all role-pair prototypes:

```python
# prototype = mean embedding over all timesteps of that role
protos = {r: X[role == r].mean(axis=0) for r in ROLES}
for (r1, r2) in pairs:
    dist[r1, r2] = 1 - (protos[r1] @ protos[r2]) / (norm(protos[r1]) * norm(protos[r2]))
```

Alternative: RSA using full pairwise distance matrices rather than prototypes, with Spearman correlation of distance matrices between "target" (aggressor-view embedding distances) and "reference" (hand-crafted "symmetric role" structure).

### 6.4 Measurement 4 — Cross-role gradient transfer (P5 test)

This is the paper's most mechanistic measurement and the one that directly motivates KARMA. Conceptual:

- At a victim-view observation `o_vic`, compute `∂V(o_vic)/∂e_vic` — the direction in encoder-embedding space in which the victim's predicted value decreases most.
- At a symmetric aggressor-view observation `o_agg`, compute `∂ log π(ZAP_AGENT | o_agg) / ∂e_agg` — the direction in encoder-embedding space that increases the probability of zapping.
- If there were "empathy," these two gradients — measured in the same embedding space — would be *anti-aligned*: decreasing V at victim states and increasing ZAP_AGENT probability would point in opposite directions.
- The empathy gap prediction: cosine(grad_V_vic, grad_π_agg) ≈ 0, i.e., the two gradients are orthogonal.

Implementation outline:

```python
# Identify paired (o_vic, o_agg) pairs in rollouts (aggressor-victim pairs at same timestep).
# Compute gradients with autograd w.r.t. embedding.
o_vic = torch.tensor(vic_obs, requires_grad=False)
out = model(o_vic.unsqueeze(0))
e_vic = out["embedding"]                     # (1, 64)
v_vic = out["value"]                         # (1, 1)
grad_V_vic = torch.autograd.grad(v_vic.sum(), e_vic)[0].detach()  # (1, 64)

o_agg = torch.tensor(agg_obs)
out = model(o_agg.unsqueeze(0))
e_agg = out["embedding"]
logits_agg = out["policy"]
log_pi_zap_agg = F.log_softmax(logits_agg, dim=-1)[:, 7]         # action 7 = ZAP
grad_pi_agg = torch.autograd.grad(log_pi_zap_agg.sum(), e_agg)[0].detach()

cos = F.cosine_similarity(grad_V_vic, grad_pi_agg, dim=-1).item()
```

Aggregate over ~1000 (vic, agg) pairs per checkpoint. Report distribution of cosines; the key statistic is whether the distribution is centered near 0 (empathy gap) or shifted negative (empathy).

**Why this is a strong measurement**: it directly tests the mechanism KARMA is supposed to intervene on. A tight null result here would be the single most informative negative-result paper you could write in this program. A positive result (mean cosine ≈ 0) is the paper's headline figure.

**Caveat**: the LSTM hidden state will normally affect value and policy. For this measurement, use the *non-recurrent* path — call the model with `hidden=None` so the gradient is computed only through CNN + projector — which is what `karma_agent.py`'s forward already supports.

### 6.5 Measurement 5 — Temporal precedence (P4 test)

For each run, you have two time series across the 20 checkpoints:

- `aggression(t)` = ZAP_AGENT rate at checkpoint t
- `orthogonality(t)` = 1 − CKA(ZAP_AGENT, BEING_ZAPPED) at checkpoint t, or linear-probe AUROC at checkpoint t

Compute cross-correlation at lags `k = -5, -4, ..., +5` (checkpoints). Average across seeds. The prediction: peak cross-correlation at `k > 0`, meaning orthogonality at time `t` predicts aggression at time `t + k`.

For stronger causal framing (though not strict causality), run a Granger-causality test. Report per scarcity × env cell.

**Reported result**: a figure with two curves per panel (orthogonality and aggression rate over training) with a clear visual lead-lag relationship, plus a cross-correlation plot.

---

## 7. Compute budget estimate

| Phase | Runs / jobs | Hours each | Total hours | Cost (@ $1.20/h) |
|---|---|---|---|---|
| Pilot (1 env, 1 scarcity, 3 seeds) | 3 | 8 | 24 | $29 |
| Main training campaign | 30 | 10 | 300 | $360 |
| Rollout generation | 600 checkpoints | 0.1 | 60 | $72 |
| Analysis (GPU-optional) | — | ~5 | 5 | $6 |
| Re-runs / debugging | ~6 | 10 | 60 | $72 |
| Reviewer ablations (reserve) | ~10 | 10 | 100 | $120 |
| **Total M1 compute** | | | **~550 h** | **~$660** |

Leaves ~$4300 of the $5000 for M2 and beyond.

Parallelization note: your T4 has 16 GB VRAM; current code uses ~2 GB. You can run **2–3 training jobs concurrently** on the same GPU if you're comfortable with that (set `CUDA_VISIBLE_DEVICES=0` for all, each in its own process). This halves wall-clock time but does not reduce total compute cost. Recommend running serial for the pilot (debug easier) then parallel for the main campaign.

---

## 8. Timeline (12 weeks)

### Weeks 1–2: Pre-flight and pilot

- [ ] Clean up `HarvestDualEnv` override path so `victim_penalty=0.0` is honored from config. Audit every config file in `configs/`.
- [ ] Add `ep_being_zapped` to metrics in `train_karma.py`.
- [ ] Write `scripts/rollout_from_checkpoint.py` that loads a `.pt` checkpoint, runs 20 episodes, and dumps Parquet with the schema in §5.
- [ ] Write `scripts/analyze_checkpoint.py` implementing measurements 1–4 for a single checkpoint file.
- [ ] Run pilot: env A, scarcity=0.3, seeds 1–3, 1000 episodes each. Confirm pipeline end-to-end.
- [ ] Pre-register predictions P1–P5 on OSF or AsPredicted.

### Weeks 3–6: Main training campaign

- [ ] Launch 30 runs (2 env × 3 scarcity × 5 seeds). If running 2-in-parallel, budget ~7–8 days of wall clock. If serial, ~12–14 days.
- [ ] Monitor via wandb for silent failures (NaN losses, reward collapse).
- [ ] Re-run any failed seeds.
- [ ] Generate rollouts for all 600 checkpoints.

### Weeks 7–8: Analysis

- [ ] Run the five measurements across all 600 checkpoints locally (laptop-class work).
- [ ] Produce per-measurement summary DataFrames.
- [ ] Generate all main figures.
- [ ] Run exploratory analyses: do ZAP_WASTE embeddings cluster with ZAP_AGENT (aggression) or with APPLE_EATEN (reward)? This is a nice bonus result.

### Weeks 9–10: Write

- [ ] Full first draft from the paper outline in §11.
- [ ] Internal red-teaming: have an ML collaborator try to poke holes in measurement 4 (the mechanistic claim).
- [ ] Pre-print to arXiv.

### Weeks 11–12: Iterate and submit

- [ ] Submit to TMLR (no deadline; rolling). Alternative: AAMAS 2027 (deadline ~October 2026) or NeurIPS 2026 (May 2026, tight).
- [ ] Tag repo as `m1-submission`.
- [ ] Release checkpoint + rollout data on Zenodo.

---

## 9. Code changes needed

These are the minimum code changes for M1. Each is local and reversible.

### 9.1 In `karmic_rl/envs/harvest_dual.py`

- Default `victim_penalty` to `0.0` (current default is `0.5` which shapes aggression). Override is already configurable; just change the default.
- Ensure `social_events` reliably logs BEING_ZAPPED events in `infos[victim_id]` after truncation (`self.agents = []` edge case).
- Optional: add a `get_role_label(agent_id, events)` method that returns the 5-class label, so training and analysis use the same function (DRY).

### 9.2 In `train_karma.py`

- Add `ep_being_zapped` counter alongside `ep_zap_agent`.
- Add `latent_drift` and `embedding_variance` to the log payload (sample a fixed minibatch of 64 observations at the start of training and compute on every log interval).
- Make `checkpoint_interval` more aggressive when the config specifies it: save full state_dict + optimizer + rng state. Keep `scaler_state_dict` saving.

### 9.3 New file: `scripts/rollout_from_checkpoint.py`

```python
# Signature
# python scripts/rollout_from_checkpoint.py \
#   --config configs/m1_env_A_scarcity_030.yaml \
#   --checkpoint results/m1/checkpoints/run_seed1_ep2000.pt \
#   --episodes 20 \
#   --output results/m1/rollouts/run_seed1_ep2000.parquet
```

Should load the env from the same config as the training run, load the checkpoint, reset the LSTM per episode, and dump the schema in §5 to Parquet. Keep it under 200 LOC.

### 9.4 New file: `scripts/analyze_checkpoint.py`

Takes a Parquet rollout file and produces:
- Linear-probe AUROC (Meas. 1)
- CKA matrix (Meas. 2)
- Cosine-distance prototype matrix (Meas. 3)
- Gradient-transfer cosine distribution (Meas. 4, needs checkpoint + Parquet together)

Outputs a small JSON summary per checkpoint. ~300 LOC.

### 9.5 New file: `scripts/aggregate_m1.py`

Walks all per-checkpoint JSON summaries, joins with training-time metrics (from CSV/wandb export), and produces a long-format DataFrame indexed by `(env, scarcity, seed, checkpoint_episode)`. This is the master dataset for all figures.

### 9.6 New configs

- `configs/m1_env_A_sc015.yaml`, `m1_env_A_sc030.yaml`, `m1_env_A_sc050.yaml`
- `configs/m1_env_B_sc015.yaml`, `m1_env_B_sc030.yaml`, `m1_env_B_sc050.yaml`

All inherit from a shared `configs/m1_base.yaml` (if your YAML loader supports `!include`, use it; otherwise just duplicate).

---

## 10. Pre-flight checklist

Before launching the 30-run campaign, verify on a single pilot run (1000 episodes, ~2.5 h):

- [ ] Training reaches ≥ 1000 episodes without any NaN or non-finite loss skipped_minibatches.
- [ ] `ViolenceRate` and `BeingZappedRate` are **paired** in the logs: every ZAP_AGENT event produces exactly one BEING_ZAPPED event on some other agent. Sanity-check with a small script.
- [ ] Checkpoint at episode 200 can be loaded by `rollout_from_checkpoint.py` and produces non-degenerate rollouts (agents do at least something).
- [ ] One checkpoint's rollout Parquet file can be analyzed by `analyze_checkpoint.py` end-to-end, producing all four measurements' outputs.
- [ ] CKA code has been tested on a toy (random gaussian) pair — should be near 0.
- [ ] Probe AUROC on randomly-shuffled labels is ~0.5 (no information leakage).
- [ ] Gradient-transfer cosine on random-vector pairs is centered at 0 (sanity).

Only after all boxes are checked, launch the campaign.

---

## 11. Paper outline (pre-write this before the campaign)

Writing the outline before the results forces you to think about what each figure must show. Fill numbers in later.

```
Title: Role-Disjoint Representations in Multi-Agent Reinforcement Learning:
       A Diagnostic Study of the Empathy Gap in Commons Dilemmas

Abstract (200 words)
- Phenomenon: aggression emerges in commons SSDs under scarcity (cite
  Leibo 2017; Hughes 2018).
- Conjecture: representations encode aggressor-view and victim-view
  observations as geometrically separable states, preventing negative
  feedback from generalizing across roles.
- Contribution: first direct measurement of this representational
  "empathy gap" in trained baseline agents.
- Methods: PPO+LSTM on Dual-Use Harvest; 5 probes — linear role
  classification, CKA, RSA, cross-role gradient transfer, temporal
  precedence.
- Results: [summary of P1–P5 outcomes].
- Implication: motivates role-invariant representation interventions
  as a principled alignment approach.

1. Introduction
   - Sequential social dilemmas and emergent aggression (1 para).
   - The empathy-gap conjecture (1 para).
   - Representational interpretability in MARL is under-developed (1 para).
   - Contributions and roadmap (1 para).

2. Background and Related Work
   - SSDs (Leibo 2017; Hughes 2018; Perolat 2017; Koster 2022).
   - MARL interpretability and representation analysis.
   - Contrastive representation learning (SimCLR, MoCo).
   - Mirror neurons and role-invariance in biological agents (brief).

3. Environment and Agent
   - Dual-Use Harvest: grid, dynamics, dual-use beam, social events.
   - Recurrent PPO agent: CNN → projector → LSTM → actor/critic.
   - Training protocol: 4000 episodes × 4 agents × 2 env × 3 scarcity × 5 seeds.

4. Measurements
   4.1 Linear probes (P1)
   4.2 CKA between role-conditioned distributions (P2)
   4.3 Representational similarity analysis (P2 robustness)
   4.4 Cross-role gradient transfer (P5)
   4.5 Temporal precedence (P4)

5. Results
   5.1 The empathy gap is present at convergence.
   5.2 It is scarcity-dependent (P3).
   5.3 It temporally precedes aggression (P4).
   5.4 Cross-role gradients are approximately orthogonal (P5).
   5.5 Dual-use vs tag-only comparison (env A vs B).

6. Discussion
   - Interpretation: role-separability is a necessary condition for
     scarcity-driven aggression.
   - Relation to alignment: architectural interventions targeting
     representation should be effective (foreshadowing M2).
   - Relation to mirror-neuron literature and embodied cognition.
   - Limits of the diagnostic: does not prove causality.

7. Limitations
   - Single env family; single architecture; single algorithm.
   - Observational, not interventional — M2 will intervene.
   - No human baseline for "what a non-empathy-gapped agent looks like."

8. Conclusion
   - The empathy gap is measurable, predictable, and temporally
     upstream of emergent aggression.
   - It is a suitable target for architectural interventions.
```

**Main figures (exactly 5)**:
1. Behavioural curves: ViolenceRate vs training episode, by env × scarcity.
2. Linear-probe AUROC vs training episode (Meas. 1).
3. 5×5 CKA heatmap panels at early/mid/late training × 2 env conditions (Meas. 2).
4. Gradient-transfer cosine distributions at early/mid/late training (Meas. 4).
5. Temporal precedence: orthogonality and aggression overlay, with cross-correlation inset (Meas. 5).

**Main tables (2)**:
1. Summary statistics per `(env, scarcity)` cell — final aggression rate, final probe AUROC, mean gradient cosine, 95 % CIs over seeds.
2. Cross-correlation peak lag per cell, with Granger-test p-values.

---

## 12. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Baseline embeddings turn out to already be role-invariant (P1 fails) | Low | High | Paper pivots to a null-result TMLR paper. Also: means KARMA has no mechanistic room → pre-register this as the honest science. |
| Training instability (NaN losses, reward collapse) at 4000 episodes | Medium | Medium | Your `stability-m1` branch already addresses this. Pre-flight catches it. If a seed fails, re-run with different seed; don't cherry-pick. |
| T4 is too slow, 30 runs take > 6 weeks | Medium | Low | Parallelize 2 runs per GPU (total wall-clock halves). Reduce to 3 seeds per cell for a pilot-submission; add 2 more for revision. |
| Gradient-transfer measurement (Meas. 4) is unstable or noisy | Medium | Medium | Use 1000+ pairs per checkpoint, bootstrap CIs. If still noisy, demote to supporting evidence and make Meas. 1–3 the headline. |
| Reviewers ask "but what about off-policy Q-learning agents? A3C?" | Medium | Medium | Add 2–3 A3C seeds in revision (budget reserve covers it). Or frame the paper as specifically about the PPO+LSTM family. |
| Scarcity levels don't produce variance in aggression | Low | Medium | Re-run with wider spread (`apple_density` ∈ {0.10, 0.30, 0.60}) or tune `regrowth_speed` instead. |
| Out-of-distribution rollouts: the evaluated policy behaves differently from training dynamics | Low | Low | Use on-policy rollouts (same exploration as training); your existing pipeline already does this during training — mirror the same. |
| Being scooped by a concurrent paper | Low | Medium | Pre-print within 1 week of result lock. The precise combination of (Dual-Use Harvest + empathy-gap framing + 5 measurements + temporal precedence) is distinctive enough to be defensible even if a related paper appears. |

---

## 13. What to do if results point the "wrong" way

### Scenario A: P1 passes but P4 (temporal precedence) fails

*Orthogonality and aggression co-occur but orthogonality does not lead.* This is still publishable — the empathy gap is real, just not causally upstream. Reframe as "concomitant representational signature of aggression" rather than "cause." KARMA still plausibly helps because intervening on the representation is still intervening on the signature.

### Scenario B: P1 fails (probes can't recover role from embedding)

*Role is not linearly decodable.* Either (a) the encoder has already discovered role invariance without supervision (representationally), or (b) role information is in the LSTM hidden state, not the CNN/projector output. Test (b) by running the same probes on `lstm_hidden` instead of `embedding`. If (a), pivot the paper to "standard MARL encoders show surprising role invariance; why does aggression still emerge?" — which is a different but equally interesting paper.

### Scenario C: P5 (gradient transfer) shows alignment, not orthogonality

*Negative feedback from victim observations already influences aggressor policy.* Means empathy gap isn't about gradient flow but something subtler — perhaps the behavioural payoff of aggression outweighs the propagated aversion. Reframe around reward landscape, not representation.

### Scenario D: Results support everything

Write the paper; submit; move to M2.

---

## 14. Deliverables checklist

At submission time, these artifacts should exist:

- [ ] Manuscript PDF (25–30 pages for TMLR; 8–10 for AAMAS).
- [ ] arXiv pre-print URL.
- [ ] Git tag `m1-submission` on `phase-a-stability-logging-m1` or a dedicated branch.
- [ ] Zenodo DOI for data release (checkpoints + rollouts + per-checkpoint analysis JSONs).
- [ ] OSF / AsPredicted pre-registration link.
- [ ] W&B project public link with training curves.
- [ ] Companion Colab / Jupyter notebook reproducing main figures from released data.
- [ ] 2-page executive summary (for sharing with collaborators and in job applications).

---

## 15. After M1 — what immediately follows

If M1 passes:

- **M2 (KARMA intervention)** is the natural next paper, co-submitted to the same venue or back-to-back. You already have the infrastructure.
- **M3 (env paper)** essentially writes itself from the M1 methods section.

If M1 partially passes (some predictions hold, others don't):

- Pivot M2's motivation to the predictions that held. KARMA intervenes on what's broken, not on what's fine.
- Adjust M2's primary metric accordingly.

If M1 fails cleanly (null result):

- Submit it anyway as a null-result TMLR paper. Null results in MARL interpretability are valuable and publishable.
- Reassess the full program: the KARMA intervention may need a different theoretical grounding (e.g., behavioural-level rather than representational-level). Consult the program roadmap §9 risk register.

---

*Document version 1.0. Update with actual results, budget burn, and timeline deltas as the campaign progresses.*
