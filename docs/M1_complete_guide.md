# M1 — Complete Self-Contained Guide

This is a single-file guide for the M1 baseline study (the "Empathy Gap" study) in `situated-agency-alignment`. It includes the plain-language explanation, the full repo structure, all relevant code references, and the exact commands to run on the user's specific Azure VM (`tapsvmT4`). No external doc is required to follow this guide.

> **Scope.** M1 is **baseline-only**. It does **not** include the KARMA intervention or the Broken Mirror control. Those are M2.

---

## Table of Contents
1. [What M1 is, in plain words](#1-what-m1-is-in-plain-words)
2. [Two environments and three scarcity levels](#2-two-environments-and-three-scarcity-levels)
3. [Repo structure](#3-repo-structure)
4. [Pipeline overview](#4-pipeline-overview)
5. [Frozen preregistration design](#5-frozen-preregistration-design)
6. [Mac (local) setup](#6-mac-local-setup)
7. [VM setup — `tapsvmT4` specific](#7-vm-setup--tapsvmt4-specific)
8. [Running M1 end-to-end](#8-running-m1-end-to-end)
9. [Power-check decision (already completed)](#9-power-check-decision-already-completed)
10. [Env B feasibility check before freezing prereg](#10-env-b-feasibility-check-before-freezing-prereg)
11. [Plain-language interpretation guide](#11-plain-language-interpretation-guide)
12. [What to commit and push](#12-what-to-commit-and-push)
13. [Run log template](#13-run-log-template)
14. [Glossary](#14-glossary)

---

## 1. What M1 is, in plain words

**TL;DR.** M1 tests whether a standard multi-agent RL agent (no intervention) develops two **separate internal pictures** of the same event — `"I hit someone"` (aggressor view) vs `"I was hit"` (victim view) — during training. If those two pictures stay geometrically separate and the agent's gradients do not transfer between them, that is evidence for an **empathy gap** that motivates M2 (KARMA).

> **Big analogy.** Imagine a kid on a playground.
> - If `me bumping someone` and `someone bumping me` go into **two unrelated boxes in the kid's head**, the kid never combines the lessons -> aggression keeps happening.
> - If both go into **one shared box** ("bumping hurts, regardless of who"), the kid learns to avoid it.
>
> M1 is the careful measurement of **what the kid's boxes look like** when we only let them play.

---

## 2. Two environments and three scarcity levels

We sweep two environment types and three scarcity levels.

### Env A — tag-only (harm-only zap)
- Zap targets agents only.
- No cooperative / waste-cleaning channel.
- Config files:
  - `configs/m1_env_A_sc015.yaml`
  - `configs/m1_env_A_sc030.yaml`
  - `configs/m1_env_A_sc050.yaml`
- Key env values:
  - `waste_spawn_rate: 0.0`
  - `dynamic_waste_enabled: false`
  - `zap_waste_reward: 0.0`

### Env B — dual-use zap (harm + cooperative)
- Zap can hit other agents *or* clear waste, depending on context.
- Config files:
  - `configs/m1_env_B_sc015.yaml`
  - `configs/m1_env_B_sc030.yaml`
  - `configs/m1_env_B_sc050.yaml`
- Key env values:
  - `waste_spawn_rate: 0.10`
  - `dynamic_waste_enabled: true`
  - `dynamic_waste_prob: 0.02`
  - `zap_waste_reward: 0.3`

### Scarcity (`apple_density`)
- `sc015` = high scarcity (0.15)
- `sc030` = medium scarcity (0.30)
- `sc050` = low scarcity (0.50)

### Why two envs?
- **Env A** = clean diagnostic: "even in a harm-only world, does the encoder split aggressor and victim?"
- **Env B** = ecological / generalisation check: "when zap is dual-use, does that change role separability?"

Together, these let M1 distinguish a pure baseline empathy gap from one driven by action ambiguity.

### 2.1 Symmetric Env B (instrumental cleanup)

Env A is Leibo-faithful in the strict sense: aggression is **instrumental**, not rewarded by a shaping bonus (`zap_agent_reward = 0.0`). The original Env B (`configs/m1_env_B_sc030.yaml`) breaks that symmetry: it adds a **shaping reward** for cleanup (`zap_waste_reward = 0.3`), so cooperative beam use is at least partly a reward-hacked behaviour rather than an emergent strategy.

**Symmetric Env B** restores the symmetry by realising cleanup as **purely instrumental**: waste tiles in the 3x3 neighborhood of an empty cell **suppress that cell's apple regrowth probability**, so clearing waste is profitable only because it speeds up the apple supply downstream. The shaping reward is removed.

- Config: `configs/m1_env_B_sc030_sym.yaml`.
- New env knob: `waste_regrowth_suppression: 0.05` (alpha; each WASTE neighbor multiplies the per-cell regrowth rate by `max(0, 1 - alpha * waste_neighbor_count)`).
- Kept setting: `zap_waste_reward: 0.0` (symmetric with `zap_agent_reward: 0.0`).
- Default `waste_regrowth_suppression: 0.0` is set in `configs/m1_base.yaml`; with this default the env is **bit-identical** to the unsuppressed implementation, so all M1 confirmatory results remain reproducible.

Quick ablation (no agents; uniform random actions, fixed seed):

```bash
python scripts/ablate_waste_regrowth.py --steps 1000
```

This prints mean / final APPLE and WASTE counts under both configs and confirms that mean apple count is lower under the `sym` config when `alpha > 0`.

**Canonical Cleanup-style waste spread (`waste_spread_prob`).** The base waste mechanic in `karmic_rl/envs/harvest_dual.py` is *local-cell-blocking only*: a WASTE tile occupies its own cell and (via `waste_regrowth_suppression`) linearly damps regrowth in its 3x3 neighborhood, but it never propagates. The new `waste_spread_prob` knob adds a canonical-Cleanup-style stochastic spread step: each existing WASTE cell, each step, has probability `waste_spread_prob` of converting one random EMPTY 4-neighbor into WASTE. Combined with a non-zero `waste_regrowth_suppression`, unchecked waste grows non-linearly and creates real apple suppression downstream, so cleanup is genuinely instrumental even when there is no shaping reward. Symmetric Env B uses `waste_spread_prob > 0` together with `waste_regrowth_suppression > 0` and `zap_waste_reward = 0.0` so that cleanup is *purely* instrumental (no shaping). With `waste_spread_prob = 0.0` (the default, set in `configs/m1_base.yaml`) the propagation step short-circuits before any RNG draw and the env is bit-identical to the pre-spread implementation, so all M1 confirmatory results remain reproducible. Ablation entry point: `python scripts/ablate_waste_regrowth.py --steps 1000 --seed 0`.

---

## 3. Repo structure

```
situated-agency-alignment/
|-- train_karma.py                          # main training entrypoint (M1: --mode baseline)
|-- requirements.txt                        # pip deps (torch, wandb, sklearn, pyarrow, ...)
|-- karmic_rl/
|   |-- agents/karma_agent.py               # PPO+LSTM agent + projector heads
|   |-- envs/harvest_dual.py                # Dual-Use Harvest environment
|   `-- utils/roles.py                      # role IDs and names (NEUTRAL / AGGRESSOR / VICTIM / ...)
|-- configs/
|   |-- m1_base.yaml                        # shared M1 defaults
|   |-- m1_env_A_sc015.yaml                 # Env A x scarcity 0.15
|   |-- m1_env_A_sc030.yaml                 # Env A x scarcity 0.30  <- pilot cell
|   |-- m1_env_A_sc050.yaml                 # Env A x scarcity 0.50
|   |-- m1_env_B_sc015.yaml                 # Env B x scarcity 0.15
|   |-- m1_env_B_sc030.yaml                 # Env B x scarcity 0.30
|   |-- m1_env_B_sc050.yaml                 # Env B x scarcity 0.50
|   `-- m1_smoke40.yaml                     # 40-episode smoke config (if present)
|-- scripts/
|   |-- m1_smoke.sh                         # 40-episode smoke run
|   |-- batch_m1_trajectory.sh              # rollout + analyze for all checkpoints
|   |-- rollout_from_checkpoint.py          # produces per-checkpoint rollout parquet
|   |-- analyze_checkpoint.py               # 4 measurements -> JSON
|   |-- aggregate_m1.py                     # per-checkpoint JSONs + training CSV -> aggregated CSV
|   `-- plot_m1_trajectory.py               # PNGs from aggregated CSV
|-- results/
|   `-- m1_env_*_sc***/                     # per-cell run outputs (created by training/analysis)
`-- docs/
    `-- M1_complete_guide.md                # this file
```

---

## 4. Pipeline overview

For each `(env, scarcity, seed)` cell:

```
train_karma.py
    -> writes 20 checkpoints + training CSV in results/<cell>/
        -> for each checkpoint:
            scripts/rollout_from_checkpoint.py
                -> writes rollout parquet
            scripts/analyze_checkpoint.py
                -> writes analysis JSON (4 measurements)
        -> scripts/aggregate_m1.py
            -> writes one aggregated CSV per run
        -> scripts/plot_m1_trajectory.py
            -> writes PNGs
```

The orchestrator for the rollout/analyze sweep is `scripts/batch_m1_trajectory.sh`.

### Four measurements (`scripts/analyze_checkpoint.py`)
1. **Linear probes** -> `measurement_1_probes.probe_5way_auroc_mean`, `probe_agg_vs_vic_auroc`, `n_aggressor`, `n_victim`.
2. **CKA** -> `measurement_2_cka.cka_agg_vs_vic`.
3. **Prototype geometry / RSA** -> `measurement_3_rsa.cosdist_agg_vs_vic`.
4. **Gradient transfer** -> `measurement_4_gradient_transfer.gradient_transfer_cos_mean`.

---

## 5. Frozen preregistration design

These are the M1 confirmatory choices, frozen by this guide and the Git commit at submission time.

| Item | Frozen value |
|---|---|
| Mode | `--mode baseline` |
| Episodes per run | 4000 |
| Checkpoint interval | every 200 episodes (20 checkpoints per run) |
| Eval episodes per checkpoint | **20** |
| `n_min` (per-checkpoint role-count threshold) | **100** for both `n_ZAP_AGENT` and `n_BEING_ZAPPED` |
| Env grid | 2 envs (A, B) x 3 scarcities (0.15, 0.30, 0.50) |
| Seeds per cell | **3** |
| Seed list | `[42, 123, 456]` |
| Total confirmatory training runs | **18** (2 x 3 x 3) |

### Confirmatory metrics (primary)
- `measurement_1_probes.probe_5way_auroc_mean`
- `measurement_2_cka.cka_agg_vs_vic` (only for checkpoints passing `n_min=100`)
- `measurement_4_gradient_transfer.gradient_transfer_cos_mean`
- behavioural: `ViolenceRate_per_agent_step`, `BeingZappedRate_per_agent_step`

### Exploratory (non-confirmatory)
- Full CKA matrix entries
- `measurement_3_rsa.cosdist_agg_vs_vic` and full RSA
- Other probe slices (e.g. on LSTM hidden state)

### Uncertainty reporting
- 1000-resample bootstrap percentile 95% CI across seeds.

### H3 (precedence) statistic
- Pearson cross-correlation between 3-checkpoint rolling means of `probe_5way_auroc_mean` and `ViolenceRate_per_agent_step`.
- Lag range frozen to `-10..+10` checkpoints.
- Decision: mean peak lag > 0 = representation precedes behaviour.
- Granger causality reported as supporting only.

---

## 6. Mac (local) setup

These are the steps that work on your Mac for plotting and inspecting CSVs.

```bash
cd ~/postdoc/situated-agency-alignment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Plot from an aggregated CSV:

```bash
python3 scripts/plot_m1_trajectory.py \
  --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv \
  --out results/m1_env_A_sc030/plots
```

To pull a CSV from the VM:

```bash
mkdir -p results/m1_env_A_sc030
scp tapsvmT4:~/situated-agency-alignment/results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv \
  results/m1_env_A_sc030/
```

---

## 7. VM setup — `tapsvmT4` specific

> **VM-specific section.** These commands assume your Azure NC16as_T4_v3 VM is reachable as the SSH alias `tapsvmT4` and the repo is at `~/situated-agency-alignment` with a working venv at `~/situated-agency-alignment/.venv` (managed via `requirements.txt`).

### 7.1 Connect
```bash
ssh tapsvmT4
cd ~/situated-agency-alignment
git checkout m1-pipeline
git pull
source .venv/bin/activate
export PYTHONPATH=.
```

### 7.2 One-time scratch setup on `/mnt`
After every VM deallocate, `/mnt/karma_m1_scratch` may need to be re-created:

```bash
sudo mkdir -p /mnt/karma_m1_scratch
sudo chown "$USER:$USER" /mnt/karma_m1_scratch
```

The batch script `scripts/batch_m1_trajectory.sh` will also try this automatically with non-interactive sudo, and falls back to `/dev/shm/karma_m1_scratch` if `/mnt` cannot be made writable.

### 7.3 Disk hygiene
Do **not** write rollout parquets to the OS disk for long. The pipeline already:
- writes parquet to scratch (`/mnt/karma_m1_scratch` or `/dev/shm/karma_m1_scratch`),
- deletes each parquet **after** successful per-checkpoint analysis,
- keeps only the small per-checkpoint JSONs and the final aggregated CSV under `results/`.

If `/` ever fills up, useful clean-up targets (use with care):
- `~/situated-agency-alignment/results/*/rollouts/` (large parquets, only if not actively needed)
- `~/situated-agency-alignment/results/<old_run>/` (entire old runs you have backed up)
- `~/situated-agency-alignment/wandb/` (regenerable)
- a redundant venv if any (e.g. an older `~/venv` that is not the one the pipeline uses)

### 7.4 W&B (optional)
If you want online sync:

```bash
WANDB_MODE=online python -u train_karma.py --config configs/m1_env_A_sc030.yaml --mode baseline --seed 43 \
  2>&1 | tee run_logs/m1_env_A_sc030_seed43.log
```

If you want offline / disabled:

```bash
WANDB_MODE=disabled python -u train_karma.py --config configs/m1_env_A_sc030.yaml --mode baseline --seed 43
```

### 7.5 tmux pattern (resilient to SSH drops)
```bash
tmux new -d -s m1_run \
  'cd ~/situated-agency-alignment && source .venv/bin/activate && \
   WANDB_MODE=online python -u train_karma.py \
     --config configs/m1_env_A_sc030.yaml --mode baseline --seed 43 \
     2>&1 | tee run_logs/m1_env_A_sc030_seed43.log'
tmux ls
```

Re-attach later:
```bash
tmux attach -t m1_run
```

If `tmux ls` says *no server running*, no session is active. Errors like `error connecting to /tmp/tmux-1000/default` simply mean the tmux server has not been started yet; create a session with `tmux new ...` and the server will start.

### 7.6 Auto-shutdown after a run (`scripts/auto_shutdown_watcher.sh`)

To stop paying for VM time when a long run finishes, use the reusable watcher at `scripts/auto_shutdown_watcher.sh`. It waits for a `pgrep -f` pattern's process to exit, then issues `sudo -n shutdown -h +1`.

One-time check (passwordless sudo for shutdown):
```bash
sudo -n shutdown --help >/dev/null 2>&1 && echo "[ok]" || echo "[needs setup]"
```
If `[needs setup]`, configure once:
```bash
echo "$USER ALL=(ALL) NOPASSWD: /sbin/shutdown" | sudo tee /etc/sudoers.d/auto_shutdown
sudo chmod 440 /etc/sudoers.d/auto_shutdown
```

General usage (after launching your training in tmux):
```bash
nohup bash scripts/auto_shutdown_watcher.sh \
  "<pgrep -f pattern>" \
  "<short_tag>" >/dev/null 2>&1 &
disown
```

Example for an Env B sc030 seed 42 4k run:
```bash
nohup bash scripts/auto_shutdown_watcher.sh \
  "train_karma.py.*m1_env_B_sc030.yaml.*--seed 42" \
  "envB_sc030_s42_4k" >/dev/null 2>&1 &
disown
```

Watcher log goes to `run_logs/auto_shutdown_<tag>.log`. Cancel any pending shutdown with:
```bash
sudo shutdown -c
```

---

## 8. Running M1 end-to-end

Below is the full per-cell, per-seed loop.

> **Always pull first** so the VM has the latest code:
> ```bash
> cd ~/situated-agency-alignment && git checkout m1-pipeline && git pull && source .venv/bin/activate && export PYTHONPATH=.
> ```

### 8.1 Smoke test (40 episodes, sanity check)
This is fast; use after pulls or environment changes.

```bash
bash scripts/m1_smoke.sh 1
```

This script auto-selects an existing 40-episode smoke config (`configs/m1_smoke40.yaml` if present, else a `canonical_baseline_*` smoke variant), trains briefly, runs one rollout, runs one analysis, and exits non-zero if anything fails.

### 8.2 Train (4000 episodes, one seed)
For each seed in `[42, 123, 456]` and each config:

```bash
WANDB_MODE=online python -u train_karma.py \
  --config configs/m1_env_A_sc030.yaml \
  --mode baseline \
  --seed 42 \
  2>&1 | tee run_logs/m1_env_A_sc030_seed42.log
```

Outputs:
- checkpoints: `results/m1_env_A_sc030/checkpoints/m1_env_A_sc030_baseline_seed42_ep<E>.pt`
- training CSV: `results/m1_env_A_sc030/m1_env_A_sc030_baseline_seed42.csv`

### 8.3 Trajectory analysis (rollout + analyze, all 20 checkpoints)
The batch driver writes parquets to scratch and deletes them after each successful analysis. Argument order: config, results dir, seed, eval episodes per checkpoint.

```bash
M1_SCRATCH_ROOT=/mnt/karma_m1_scratch \
bash scripts/batch_m1_trajectory.sh \
  configs/m1_env_A_sc030.yaml \
  results/m1_env_A_sc030 \
  42 \
  20
```

To rerun a single checkpoint, add a 5th argument (its episode), e.g. `4000`.

Outputs (per checkpoint):
- analysis JSON: `results/m1_env_A_sc030/analysis/trajectory_m1_env_A_sc030_baseline_seed42/m1_env_A_sc030_baseline_seed42_ep<E>.json`

### 8.4 Aggregate
```bash
python scripts/aggregate_m1.py \
  --analysis-dir results/m1_env_A_sc030/analysis/trajectory_m1_env_A_sc030_baseline_seed42 \
  --training-dir results/m1_env_A_sc030 \
  --output results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv
```

### 8.5 Plot
On VM (or pull CSV to Mac and run there):

```bash
python scripts/plot_m1_trajectory.py \
  --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv \
  --out results/m1_env_A_sc030/plots
```

Outputs: `01_behaviour_rates.png` ... `07_summary_2x2.png` under `--out`.

### 8.6 Full M1 confirmatory loop
For each `cfg` in:

```
configs/m1_env_A_sc015.yaml
configs/m1_env_A_sc030.yaml
configs/m1_env_A_sc050.yaml
configs/m1_env_B_sc015.yaml
configs/m1_env_B_sc030.yaml
configs/m1_env_B_sc050.yaml
```

run sections **8.2 -> 8.3 -> 8.4 -> 8.5** for each seed in `[42, 123, 456]`.

That gives you **18 runs** total. Each run's results live under `results/<config_stem>/`.

---

## 9. Power-check decision (already completed)

For pilot cell `m1_env_A_sc030`, baseline, seed 42, at episode 4000, we compared the original 20-episode rollout analysis with a 4 x 20 = 80-equivalent (mean of four independent 20-episode rollouts; the single-process 80-episode rollout was OOM-killed).

Observed deltas vs the 20-episode run:

| Metric | 20-ep | mean(4 x 20) | delta |
|---|---:|---:|---:|
| `probe_5way_auroc_mean` | 0.8135 | 0.8250 | +0.0115 |
| `probe_agg_vs_vic_auroc` | 0.2571 | 0.3082 | +0.0511 |
| `cka_agg_vs_vic` | 0.00875 | 0.01239 | +0.00364 |
| `gradient_transfer_cos_mean` | -0.10761 | -0.10136 | +0.00625 |

**Decision.** These shifts are small. The main M1 campaign uses **20 eval episodes per checkpoint**. The binary agg-vs-vic probe is the most sampling-sensitive primary; this is disclosed in the prereg.

Also, on the same pilot row the per-checkpoint `min(n_ZAP_AGENT, n_BEING_ZAPPED)` showed:
- `n_min = 50` -> 19/20 checkpoints pass
- **`n_min = 100` -> 14/20 checkpoints pass**
- `n_min = 200` -> only 5/20 pass

So `n_min = 100` is the frozen threshold.

---

## 10. Env B feasibility check before freezing prereg

We freeze configs only after a feasibility check that **Env B actually produces meaningful aggression** and dual-use behaviour at baseline. Reason: M2 will need at least the harm regime that M1 measures, and may run for 10k+ episodes.

### Recommended minimum check (one VM run)

```bash
WANDB_MODE=online python -u train_karma.py \
  --config configs/m1_env_B_sc030.yaml \
  --mode baseline \
  --seed 42 \
  2>&1 | tee run_logs/m1_env_B_sc030_seed42_4k.log
```

Inspect from the training CSV (`results/m1_env_B_sc030/...csv`):

- `ViolenceRate_per_agent_step` — should become non-zero and sustained.
- `CooperationRate_per_agent_step` — should also become non-zero (Env B is dual-use).
- `BeamUseRate_per_agent_step` — overall beam usage stable / not collapsed.
- `AvgReturn_per_agent` and `AppleRate_per_agent_step` — not catastrophically zero.

### If 4k is ambiguous: longer scout
Make a `configs/m1_env_B_sc030_10k.yaml` copy with `training.episodes: 10000` and run again with the same `--mode baseline --seed 42`. This tests whether Env B becomes meaningful only at longer horizons (relevant since M2 may use 10k+ episodes).

### Decision rule
- If 4k Env B already shows non-trivial `ViolenceRate_per_agent_step` and non-zero `CooperationRate_per_agent_step`: keep both Env A and Env B in the prereg as listed.
- If 4k Env B is unclear: use the 10k scout.
- If even 10k Env B is degenerate: do **not** preregister Env B as confirmatory; either revise `configs/m1_env_B_*.yaml` first, or preregister an Env A-only M1.

---

## 11. Plain-language interpretation guide

For each aggregated CSV / plot, read in this order.

### 11.1 Behaviour first
- `AvgReturn_per_agent`, `AppleRate_per_agent_step` — is the run healthy and harvesting?
- `ViolenceRate_per_agent_step`, `BeingZappedRate_per_agent_step` — is harm actually happening?
- `CooperationRate_per_agent_step` — only meaningful in Env B.

### 11.2 Representation diagnostics next
- `measurement_1_probes.probe_5way_auroc_mean` — can a linear classifier read role from embeddings? Higher = more decodable. (H1)
- `measurement_1_probes.probe_agg_vs_vic_auroc` — narrower aggressor-vs-victim probe; treat as confirmatory only when `n_min=100` is met.
- `measurement_2_cka.cka_agg_vs_vic` — similarity of aggressor and victim representation subspaces. Lower = more separated rooms. (H2)
- `measurement_4_gradient_transfer.gradient_transfer_cos_mean` — do "I was hit" gradients align with "I should hit" gradients? Negative or near-zero supports the empathy gap. (H4)

### 11.3 Cross-cell comparison (M1 only)
- Across env A vs env B and across scarcity levels, **only baseline runs** are compared in M1.
- KARMA / Broken Mirror comparisons are out of scope here; they belong to M2.

### 11.4 Variance
Always show seed-level uncertainty. The frozen rule is **bootstrap percentile 95% CI** over the 3 seeds (1000 resamples).

---

## 12. What to commit and push

### Always commit
- code changes under `train_karma.py`, `scripts/`, `configs/`, `karmic_rl/`
- `requirements.txt`
- this file `docs/M1_complete_guide.md`
- preregistration / guideline / reproducibility doc updates

### Usually commit
- small aggregated CSVs and plot PNGs you want to share

### Do not commit
- large `.pt` checkpoints
- large `.parquet` rollouts
- anything under `wandb/`
- `__pycache__/` and `.DS_Store`
- secrets / `.env` files

### Suggested workflow
```bash
git status
git add <only what you intend to publish>
git commit -m "Concise message about the change"
git push origin m1-pipeline
```

For a paper/registration commit, also record:
```bash
git rev-parse HEAD
```
This is the SHA you put in OSF.

---

## 13. Run log template

Copy-paste into your run log doc (or paste into a section at the bottom of this file as you go).

```
### YYYY-MM-DD — <short title>
- **Host:** Mac | tapsvmT4 (Azure NC16as_T4_v3)
- **Branch:** m1-pipeline
- **Commit:** `<git rev-parse HEAD>`
- **Config:** configs/<...>
- **Seed:** <N>
- **Commands:** see §8 above
- **Artifacts:** results/<cell>/...
- **Notes:** wandb mode, scratch path used, failures, decisions.
```

Example entry already documented:

```
### 2026-04-28 — ep4000 power check (20 vs 4x20 ~ 80-eval equivalent)
- Host: tapsvmT4
- Branch: m1-pipeline
- What: Power check, 20 vs 80-eval equivalent at ep4000 of m1_env_A_sc030 seed 42.
- Result: kept 20 eval episodes per checkpoint for the main campaign (see §9).
```

---

## 14. Glossary

- **Checkpoint.** Saved snapshot of agent weights (`.pt`).
- **Rollout.** Running an agent in the env and recording per-step features (per-row table written to parquet).
- **Embedding.** The agent's internal vector for a situation (the analysis target).
- **Linear probe (AUROC).** Small classifier asking "can I read the role off the embedding?" 0.5 = chance, 1.0 = perfect.
- **CKA.** Centered Kernel Alignment, a similarity score between two sets of vectors. 0 = different shape, 1 = same shape.
- **Prototype distance / RSA.** Distance between mean embeddings of each role.
- **Gradient transfer.** Whether the gradient signal from "being hit" aligns with the gradient signal that would reduce hitting others.
- **`n_min`.** Minimum per-class row count required before a checkpoint contributes a confirmatory CKA / agg-vs-vic probe value. Frozen at **100**.
- **Empathy gap (the hypothesis).** Baseline encoder represents `I hit` and `I was hit` as separate states, so feedback from one does not tame the other.
- **M1 vs M2.** M1 = baseline diagnostic only. M2 = KARMA intervention (separate prereg, scheduled later).

---

## Appendix A - Hypotheses, predictions, and statistical tests

The frozen confirmatory hypotheses for the 18-run campaign. All hypotheses are **directional**. Primary metrics and inference criteria are pre-specified; no post-hoc threshold tuning is permitted.

### A.1 H1 - Existence / decodability
- **Statement.** Linear probes trained on frozen encoder embeddings achieve above-chance role classification by mid-training.
- **Primary metric.** `measurement_1_probes.probe_5way_auroc_mean`.
- **Threshold.** `probe_5way_auroc_mean > 0.80` at or before episode 2000 in at least one scarcity condition.
- **Statistical test.** No p-value: this is a precision/threshold claim. Confirmed if the threshold is met in **>= 2 of the 6** `(env, scarcity)` cells by episode 2000.
- **Null.** AUROC <= 0.50 (chance) - the empathy-gap framing is incorrect, paper pivots to a null-result paper.

### A.2 H2 - Role asymmetry / non-trivial separability
- **Statement.** Aggressor-view and victim-view embedding distributions are more dissimilar to each other than either is to a neutral baseline:
  `CKA(ZAP_AGENT, BEING_ZAPPED) < CKA(ZAP_AGENT, NEUTRAL)` AND `CKA(ZAP_AGENT, BEING_ZAPPED) < CKA(BEING_ZAPPED, NEUTRAL)`.
- **Primary metric.** `measurement_2_cka.cka_agg_vs_vic` (eligible checkpoints only; see Appendix B).
- **Statistical test.** Linear CKA across three role pairs per checkpoint, then **1000-resample bootstrap percentile 95% CI** over checkpoint-averaged CKA per cell.
- **Inference rule.** Confirmed if the bootstrap 95% CI of `CKA(ZAP_AGENT, BEING_ZAPPED) - CKA(ZAP_AGENT, NEUTRAL)` is entirely negative in **>= 3 of the 6 cells**, with the same for the second comparison.
- **Null.** CKA is symmetric across all role pairs.

### A.3 H3 - Behaviour-representation coupling / temporal precedence
- **Statement.** Role-separability trajectory is positively associated with the aggression trajectory over training, and representational separability **precedes** behavioural aggression.
- **Primary representation series.** `probe_5way_auroc_mean` (and H2 metric).
- **Primary behaviour series.** `ViolenceRate_per_agent_step`, `BeingZappedRate_per_agent_step`.
- **Statistical test.** Pearson cross-correlation between the **3-checkpoint rolling mean** of `probe_5way_auroc_mean` and the **3-checkpoint rolling mean** of `ViolenceRate_per_agent_step`, computed per seed per cell. Lag range frozen to **-10 to +10 checkpoints** (-2000 to +2000 episodes).
- **Inference rule.** Confirmed if the **mean peak lag** across seeds and cells is positive (representation precedes behaviour). Granger causality (`statsmodels.grangercausalitytests`, `max_lag=5` checkpoints) is reported as **supporting, secondary** evidence only - not as the primary decision rule.
- **Null.** No lag or reverse lag.

### A.4 H4 - Gradient disconnect
- **Statement.** Cross-role gradient transfer is weak or negative on average.
- **Primary metric.** `measurement_4_gradient_transfer.gradient_transfer_cos_mean`.
- **Statistical test.** One-sample t-test of `gradient_transfer_cos_mean` values across seeds per cell against `mu_0 = 0`, two-tailed, `alpha = 0.05`.
- **Inference rule.** Confirmed if mean cosine is not significantly greater than 0 (i.e., t-test against `mu_0 = 0` is non-significant, or significant with negative trend).
- **Null.** Gradients are positively aligned - negative feedback from victimization is already flowing into aggressor policy updates.

### A.5 H3-mod - Scarcity moderation
- **Statement.** Role separability is greater under higher-scarcity training (lower `apple_density`), paralleling the known scarcity-dependence of aggression in SSDs.
- **Primary metric.** Checkpoint-averaged `probe_5way_auroc_mean` across scarcity levels, within each env condition, across seeds.
- **Statistical test.** One-way ANOVA (or Kruskal-Wallis if normality violated), `alpha = 0.05`. Post-hoc: pairwise comparisons with **Holm correction** if ANOVA is significant.
- **Inference rule.** Confirmed if ANOVA is significant and post-hoc shows higher separability at lower `apple_density` (0.15 > 0.30 > 0.50).
- **Null.** No effect of scarcity on representational separability.

### A.6 Multiple comparisons
Primary outcomes are H1-H4 plus H3-mod. **No correction across H1-H4** is applied because they are conceptually independent; this is disclosed. All exploratory outputs (see section 5 and Appendix B) are explicitly non-confirmatory.

### A.7 Transformations applied before tests
- **Class balancing for probes.** `NEUTRAL` is downsampled per checkpoint to match the smallest non-neutral role class, with a fixed seed in the analysis script.
- **CKA sample matching.** For each role pair, both embedding matrices are subsampled to `min(n_r1, n_r2)` rows with a fixed per-checkpoint seed.
- **Trajectory smoothing for H3.** 3-checkpoint (600-episode) rolling mean on behaviour series before cross-correlation. Raw series also reported in supplementary figures.
- **No embedding normalization** before probing or CKA - raw float32 outputs of the encoder are used.

### A.8 Index definitions (formulas)
- **5-way probe AUROC:** `mean([AUROC_class_k for k in {ZAP_AGENT, BEING_ZAPPED, ZAP_WASTE, APPLE_EATEN, NEUTRAL}])` from a single one-vs-rest logistic regression on the 5-class problem.
- **Linear CKA (Kornblith et al. 2019):** `cka(X_A, X_V) = ||X_A^T X_V||_F^2 / (||X_A^T X_A||_F * ||X_V^T X_V||_F)` on mean-centered embeddings, after sample-size matching.
- **Gradient transfer cosine:** `cos(grad_V_vic, grad_pi_agg)` in the embedding space (dim 64), averaged over matched (victim obs, aggressor obs) pairs per checkpoint.

### A.9 Role-label priority (single-label)
The single-label priority order for probes is **BEING_ZAPPED > ZAP_AGENT > ZAP_WASTE > APPLE_EATEN > NEUTRAL** (victim salience highest). The shared source of truth lives in:

```7:11:karmic_rl/utils/roles.py
Priority (matches KarmaAgent._infer_role):
    BEING_ZAPPED (victim)   > ZAP_AGENT (aggressor)
                            > ZAP_WASTE (cleaner)
                            > APPLE_EATEN (forager)
                            > NEUTRAL
```

Multi-label flags are also produced (one row per role) for analyses that do not collapse to a single label.

---

## Appendix B - Exclusion / inclusion rules and amendment policy

### B.1 Checkpoint-level exclusion (role-comparison primaries)
For confirmatory use of `cka_agg_vs_vic` and `probe_agg_vs_vic_auroc`:
- Require `measurement_1_probes.n_aggressor >= 100` AND `measurement_1_probes.n_victim >= 100` (`n_min = 100`).
- Checkpoints failing the rule are tagged `underpowered_at_this_checkpoint` and excluded from confirmatory H2 analyses. They remain valid for H1, H3, and H4 (which do not require role-pair sample matching).

### B.2 Run-level exclusion
- If a run produces **fewer than 10 analyzable checkpoints** (out of 20) due to persistent failures, that seed is **excluded from confirmatory analyses** and noted as attrition. The run is **not** replaced with a new seed post hoc.
- If a run produces **NaN or constant embeddings across >50% of checkpoints** (embedding collapse), the run is excluded and this is reported.

### B.3 Missing data
- **Missing checkpoint `.pt` file:** record attrition, omit that checkpoint only, no interpolation.
- **Rollout / analysis script failure for a checkpoint:** rerun that checkpoint **once** with the same config and seed. If it fails again, mark missing and report.
- **Missing measurement field within a JSON:** that checkpoint is excluded from the affected hypothesis test only; other metrics from the same checkpoint remain in use.
- The final manuscript will include a **data availability and attrition table** per `(env, scarcity)` cell.

### B.4 Outliers
**No outlier removal** beyond the rules above. All valid checkpoints are included.

### B.5 Stop / amend rules (pre-specified)
- **Eval-episode amendment.** If **>= 3 consecutive checkpoints** in a cell fail the `n_min = 100` threshold, an amendment escalating from **20 to 80 eval episodes** is permitted for that cell (and potentially globally), provided it is filed with a **dated, explicit rationale** before any confirmatory inference from that cell.
- **Convergence-failure amendment.** If logistic probe convergence failures exceed **10% of fits** across a cell, an amendment to solver scaling or `max_iter` is permitted, followed by re-run of the **pilot row only** before scaling.
- **Infrastructure amendment.** Operational changes that do not alter model logic (scratch path, process management, OOM mitigation) may be documented without filing a protocol amendment.
- **No per-checkpoint threshold tuning. No post-hoc relabelling of exploratory metrics as primary.**

### B.6 Where amendments are recorded
Amendments are recorded as dated entries in this file (Appendix D run log) **and** on the OSF page, before any confirmatory inference touches the affected data.

---

## Appendix C - OSF preregistration metadata and deliverables

This appendix folds in the OSF-template fields that are not already covered in section 5.

### C.1 Title
**M1: The Empathy Gap in Baseline MARL - Role-Disjoint Representations in Dual-Use Harvest.**

### C.2 Study type
**Observational simulation study.** Quasi-experimental factorial: 2 (env) x 3 (scarcity) x 3 (seeds). Unit of analysis: **checkpoint within run** (time-series panel). Mode: **baseline only** (`--mode baseline`).

### C.3 Blinding
No blinding (no human subjects). To preserve confirmatory discipline, the **primary vs exploratory** distinction is frozen in this preregistration before the 18-run main campaign begins. The pilot cell (`m1_env_A_sc030`, baseline, seed 42) was used only for pipeline validation and `n_min` calibration and is **non-confirmatory**. Analysis scripts are version-controlled and frozen at the registered Git commit SHA.

### C.4 Randomization
Seeds are pre-specified (`[42, 123, 456]`) and shared across cells. PyTorch / NumPy / env RNGs are seeded from `--seed` in `train_karma.py`. No adaptive or stratified randomization.

### C.5 OSF metadata (fill before submission)
- Contributors: Principal Investigator (IIT Kanpur); collaborators TBD.
- Institution: IIT Kanpur.
- Funding: TBD.
- Ethics / IRB statement: Not applicable (simulation, no human subjects).
- Preregistration submission date: TBD.
- Planned data collection start: TBD.
- Planned analysis completion date: approximately 3 months from start.
- Planned submission venue: TMLR / AAMAS 2027 / NeurIPS Workshop.
- Git branch at registration: `m1-pipeline`.
- Git commit SHA at registration: from `git rev-parse HEAD` at submission.
- W&B project (public): TBD.
- License / embargo: TBD (default: CC-BY 4.0 for the OSF record; data release embargoed until manuscript acceptance).
- Registration type: Pre-Data Collection (OSF Standard Pre-Data Collection Template).

### C.6 Deliverables checklist (at submission)
- Manuscript PDF (25-30 pages for TMLR, 8-10 for AAMAS).
- arXiv pre-print URL.
- Git tag `m1-submission` on the M1 branch.
- Zenodo DOI for data release (checkpoints + rollouts + per-checkpoint analysis JSONs).
- OSF / AsPredicted preregistration link.
- W&B public project link with training curves.
- Companion Colab / Jupyter notebook reproducing main figures from released data.
- 2-page executive summary.

### C.7 Null result policy
If H1 fails (probes cannot decode role from embeddings), the paper is reframed as a **null-result TMLR submission** documenting the absence of an empathy gap and its implications for KARMA's motivating assumption.

### C.8 Related work disclosures
- Codebase: `situated-agency-alignment` (implements the KARMA framework and the Dual-Use Harvest environment).
- Theoretical motivation: the Extended Self / Proxy Agency Moral Shield framework (`SoR_reviewed_NS.pdf`).
- Program context: `research_program_roadmap.md`.
- M1 baseline study does **not** use KARMA components; M2 is a separate preregistration.

---

## Appendix D - Historical run log (verbatim)

These dated entries are copied verbatim from the prior `docs/m1_reproducibility.md` run log. Append new entries above the older ones (newest first).

```
### 2026-04-28 - ep4000 power check (20 vs 4x20 ~ 80-eval equivalent)
- **Host:** Azure VM (`tapsvmT4`)
- **What:** Power check for whether M1 should escalate from 20 to 80 eval episodes per checkpoint.
- **Commands:** Four independent `scripts/rollout_from_checkpoint.py` runs at `--episodes 20` for `ep4000`, followed by `scripts/analyze_checkpoint.py` on each part JSON. The single-process `--episodes 80` rollout and merged-parquet analysis were both operationally unstable (killed with exit 137), so the comparison used **mean(4x20)** as an 80-eval equivalent estimate.
- **Artifacts:** `results/m1_env_A_sc030/analysis/power_check/ep4000_eval20_part{1,2,3,4}.json`
- **Notes:** Final comparison vs the original 20-episode ep4000 JSON: `probe_5way_auroc_mean 0.8135 -> 0.8250`, `probe_agg_vs_vic_auroc 0.2571 -> 0.3082`, `cka_agg_vs_vic 0.00875 -> 0.01239`, `gradient_transfer_cos_mean -0.10761 -> -0.10136`. Decision: **keep 20 eval episodes** for the main campaign; no global amendment to 80. The binary agg-vic probe was the most sampling-sensitive metric.
```

```
### 2026-04-24 - Plots from synced aggregate (local Mac)
- **Host:** Mac (repo: `situated-agency-alignment`)
- **What:** Regenerated figure PNGs from VM-produced aggregate CSV.
- **Commands:** `python3 scripts/plot_m1_trajectory.py --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv --out results/m1_env_A_sc030/plots`
- **Artifacts:** `results/m1_env_A_sc030/plots/01_behaviour_rates.png` ... `07_summary_2x2.png`
- **Notes:** Training + batch analysis were run elsewhere; CSV was synced into `results/m1_env_A_sc030/` before plotting.
```

---

## Appendix E - Extra plain-words material

### E.1 Step-to-file cheat sheet
A compact mapping from each pipeline step to the file that runs it, written in plain words. (Same files as section 3, organized by intent.)

| Step | What it does | File(s) |
|------|--------------|---------|
| 0 | Fix env + agent recipe | `configs/m1_env_*_sc*.yaml`, `configs/m1_base.yaml`, `train_karma.py` |
| 1 | Train and save checkpoints | `train_karma.py` |
| 2 | Rollouts per checkpoint | `scripts/rollout_from_checkpoint.py` (driven by `scripts/batch_m1_trajectory.sh`) |
| 3 | Four brain-shape measurements | `scripts/analyze_checkpoint.py` (driven by `scripts/batch_m1_trajectory.sh`) |
| 4 | Aggregate to one CSV | `scripts/aggregate_m1.py` |
| 5 | Plots | `scripts/plot_m1_trajectory.py` |
| 6 | Rules and bookkeeping | this guide (`docs/M1_complete_guide.md`) |

### E.2 Per-step plain-words gloss
- **Step 1 analogy:** the school year of the kid; we take a school photo every Friday (a checkpoint every 200 episodes) so we can compare how the kid thinks at different ages.
- **Step 2 analogy:** we ask the kid from each school photo to play a few practice games while a silent observer writes a play-by-play in a notebook (the rollout parquet).
- **Step 3 analogy:** four different referees watch the same play-by-play and each writes one sentence about how separate the two experiences look in the kid's head.
- **Step 4 analogy:** we take the 20 referee cards and staple them into one report card for the whole school year.
- **Step 5 analogy:** take the report card and draw growth charts: when does the kid get taller, quieter, more separate-boxed?

### E.3 What M1 does NOT do (anti-confusion list)
- **No KARMA and no broken-mirror variant.** Those are M2 (intervention) and M2b (control). M1 watches the untouched baseline.
- **No claims from one seed alone.** The Env A `sc030` seed 42 run was a pilot; it validated the pipeline and the `n_min` choice, nothing more.
- **No cherry-picking of checkpoints.** Every saved checkpoint is analysed; if a checkpoint fails the `n_min` rule, it is **tagged**, not hidden.
- **No reading of representational diagnostics before behaviour.** Always read welfare and harm rates first (see section 11.1) and only then read probes / CKA / gradient-transfer.

---

## Appendix F - Pointer / deprecation note

The four older M1 docs in `docs/` are now **superseded** by `docs/M1_complete_guide.md` (this file). Their unique content has been folded into Appendices A-E above. The older files remain in the repo as historical artefacts of the design process and are **not** kept in sync with the main guide going forward.

Older docs (do **not** delete; treat as read-only history):
- `docs/m1_in_plain_words.md`
- `docs/m1_reproducibility.md`
- `docs/m1_experimental_guideline.md`
- `docs/M1_OSF_Preregistration.md`

For any new operational change (config edit, script change, run-log entry, amendment), update **only this file** (`docs/M1_complete_guide.md`).

---

## Appendix G - Env B design history (2026-04-28 / 04-29)

This appendix records the design changes to the Dual-Use Harvest environment used by Env B, why each change was made, and the empirical evidence behind it. It is the audit trail for the symmetric / canonical-Cleanup variant of Env B.

### G.1 Starting point

Original Env B (`configs/m1_env_B_sc030.yaml` before this episode) used:
- `waste_spawn_rate: 0.10`
- `dynamic_waste_enabled: true`
- `dynamic_waste_prob: 0.02`
- `zap_waste_reward: 0.3` (shaped per-action bonus)
- `waste_regrowth_suppression: 0.0` (knob did not exist yet)
- `waste_spread_prob: 0.0` (knob did not exist yet)

Env A (`configs/m1_env_A_sc030.yaml`) used `zap_agent_reward: 0.0` (Leibo-faithful: aggression was *instrumental*, not shaped).

### G.2 Symptom that triggered the redesign

The first 4000-episode Env B feasibility scout (seed 42) and a confirming run (seed 123) both showed clear harvest collapse despite some violence rising:

| metric | seed 42 first10% -> last10% | seed 123 first10% -> last10% |
|---|---|---|
| `ViolenceRate_per_agent_step` | 0.00714 -> 0.00755 (+5.7%) | 0.00466 -> 0.00738 (+58.6%) |
| `CooperationRate_per_agent_step` | 0.00670 -> 0.00753 (+12.4%) | 0.00574 -> 0.00690 (+20.2%) |
| `BeamUseRate_per_agent_step` | 0.123 -> 0.133 (+8.7%) | 0.075 -> 0.125 (+66.4%) |
| `AvgReturn_per_agent` | 175.2 -> 13.7 (**-92%**) | 138.5 -> 22.3 (**-84%**) |
| `AppleRate_per_agent_step` | 0.173 -> 0.011 (**-93%**) | 0.137 -> 0.020 (**-85%**) |

Two seeds with the same direction made this a configuration issue, not seed noise. Agents were chasing the per-action `zap_waste_reward` at the cost of harvesting.

### G.3 Conceptual issue: reward asymmetry between Env A and Env B

Env A had no shaped per-action reward for harm (agents had to learn aggression *instrumentally* via competitor-timeout dynamics). Env B had a shaped per-action reward for cleanup. So differences between A and B at the encoder level could be attributed to *reward shaping*, not to *beam semantics*. That is exactly the confound the M1 hypotheses do not want.

The principled fix: make cleanup *also* instrumental in Env B - rewarded only via downstream apple availability, not via a per-action shaping bonus. Then Env A and Env B differ only in *what the beam can target*.

### G.4 First attempt: local linear regrowth suppression (alpha)

Added a knob `waste_regrowth_suppression` (alpha, default `0.0`). When alpha > 0, apple regrowth rates are scaled by `(1 - alpha * waste_neighbor_count)` in the 3x3 neighborhood. Backward-compatible: alpha = 0 is bit-identical to the prior implementation. Wired through `train_karma.py`, `scripts/rollout_from_checkpoint.py`, and the env's `__init__`. (Commit `57b708b`.)

### G.5 Why alpha alone was not enough

Ablation across 4 seeds at the original waste density (script: `scripts/ablate_waste_regrowth.py`) showed:
- Env A vs base Env B already differed by 18-22% in mean apple count under uniform-random actions, before any suppression. So Env B was already **structurally apple-poor** relative to Env A even without alpha.
- alpha = 0.05 had a near-zero effect on Env B's mean apple count (-0.06% to -5.9%).
- alpha = 0.20+ caused phase-transition collapse on some seeds (apple count crashed to ~0.5).
- Increasing alpha widened the A vs B_sym gap rather than closing it.

Conclusion: the alpha knob alone could not satisfy both *cleanup matters ecologically* and *A and B have comparable apple availability*. Resource parity was already broken at alpha = 0 because waste tiles were blocking too many cells.

### G.6 Second change: lower Env B waste density

Reduced waste pressure across all Env B variants (`configs/m1_env_B_sc015.yaml`, `m1_env_B_sc030.yaml`, `m1_env_B_sc050.yaml`, `m1_env_B_sc030_sym.yaml`):
- `waste_spawn_rate`: 0.10 -> 0.04
- `dynamic_waste_prob`: 0.02 -> 0.005

(Commit `79e2a0b`.) The 8-seed re-ablation showed Env A vs Env B mean-apple gap dropped to roughly +/- 3 to 7% (with one seed at -14.9%), restoring approximate resource parity.

But at this lower waste density, the alpha knob's effect on apple availability also dropped to noise (typically -0.0% to -1.3% per alpha tick). Cleanup had near-zero ecological consequence even at alpha = 0.30. Agents would have no instrumental reason to clean - the symmetric variant would degenerate into "Env A with a few inert blocked cells".

### G.7 The trade-off, made explicit

With the local-only suppression mechanic in `_regrow_apples`, raising waste density makes alpha bite but breaks A vs B parity (Env B becomes apple-poor). Lowering waste density restores parity but makes cleanup ecologically irrelevant. There is no Goldilocks zone with that mechanic alone.

To keep both *cleanup is instrumentally rewarded* and *A and B have comparable apple availability*, we needed a non-linear, propagating waste mechanic - the canonical Cleanup design.

### G.8 Third change: canonical Cleanup-style waste spread

Added a second knob `waste_spread_prob` (default `0.0`). Each step, every existing waste tile attempts, with probability `waste_spread_prob`, to spread to one uniformly random EMPTY 4-neighbor (up/down/left/right). With `waste_spread_prob = 0` the method short-circuits and the env stays bit-identical to before. Combined with `waste_regrowth_suppression`, unchecked waste accumulates non-linearly: more waste -> more spread -> more local suppression -> faster apple decline. This creates a real public-goods incentive to clean even at low base waste density. (Commit `5a7ba55`.)

The two configs that use it:
- `configs/m1_env_B_sc030.yaml`: `zap_waste_reward: 0.3`, `waste_regrowth_suppression: 0.0`, `waste_spread_prob: 0.02` (still shaped, but cleanup is now also ecologically relevant; smaller shaping needed).
- `configs/m1_env_B_sc030_sym.yaml`: `zap_waste_reward: 0.0`, `waste_regrowth_suppression: 0.10`, `waste_spread_prob: 0.02` (truly instrumental; the prereg-target variant if the post-spread ablation confirms).

### G.9 What we have not done yet

The post-spread ablation (alpha x spread cross-product) had not been re-run at the time of this writing. The provisional values (`alpha = 0.10`, `spread = 0.02`) are starting points; the final values for both knobs in `m1_env_B_sc030_sym.yaml` will be selected from that ablation and recorded as a run-log entry in section 13. Likewise the matching values for `m1_env_B_sc015.yaml` and `m1_env_B_sc050.yaml` will be set after the sc030 selection is final.

### G.10 Backward compatibility

All earlier M1 results (Env A pilot, Env B baseline scouts, v1 retune scout, power check) remain reproducible: every newly added env knob has a default of `0.0`, and both `_propagate_waste` and the `_regrow_apples` waste-suppression branch early-return when their respective knobs are `0.0`. Configs that do not set the new keys load with the mechanics disabled.

### G.11 Files involved (audit trail)

Code:
- `karmic_rl/envs/harvest_dual.py` — added `waste_regrowth_suppression`, `waste_spread_prob`, `_propagate_waste`; extended `_regrow_apples` with an alpha-gated branch.
- `train_karma.py`, `scripts/rollout_from_checkpoint.py` — explicit env-kwarg wiring for both new knobs.
- `scripts/analyze_checkpoint.py` — unchanged (uses `build_env`).
- `scripts/ablate_waste_regrowth.py` — Env A baseline + base Env B baseline + symmetric (alpha x spread) cross-product + waste-trajectory summary.

Configs:
- `configs/m1_base.yaml` — both new keys defaulted to `0.0` with explanatory comments.
- `configs/m1_env_B_sc030.yaml`, `configs/m1_env_B_sc015.yaml`, `configs/m1_env_B_sc050.yaml` — lower waste density; `m1_env_B_sc030.yaml` also gets `waste_spread_prob: 0.02`.
- `configs/m1_env_B_sc030_sym.yaml` — symmetric variant with `zap_waste_reward = 0.0`, `waste_regrowth_suppression = 0.10`, `waste_spread_prob = 0.02`.

Docs:
- This file (`docs/M1_complete_guide.md`): section 2.1 (symmetric Env B summary) and Appendix G (this design history).

Commits (in order):
- `b9bea00` - Env B feasibility scouts, M1 self-contained guide.
- `c47b1f9` - reusable VM auto-shutdown watcher.
- `57b708b` - `waste_regrowth_suppression` end to end + symmetric variant.
- `ffa52c3` - .gitignore + cache untrack.
- `f5fc23d` - ablation extended with Env A and alpha sweep.
- `79e2a0b` - lower Env B waste density to restore A vs B parity.
- `5a7ba55` - `waste_spread_prob` canonical Cleanup mechanic.
