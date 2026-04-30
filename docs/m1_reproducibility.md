# M1 trajectory pipeline — reproducibility & run log

This file is the **single place** to record how M1 results were produced so anyone with the repo can replay the same steps. **Commit changes to this document** when you finish a stage (training, batch analysis, aggregation, plotting, or syncing artifacts).

## What is versioned for reproducibility

| Item | Location | Notes |
|------|----------|--------|
| Training & env hyperparameters | `configs/m1_*.yaml`, `configs/m1_base.yaml` | Must match the run you analyze. |
| Training entrypoint | `train_karma.py` | `--config`, `--mode`, `--seed` define the run id. |
| Rollout / analysis / aggregation | `scripts/rollout_from_checkpoint.py`, `scripts/analyze_checkpoint.py`, `scripts/aggregate_m1.py` | Deterministic given checkpoint + seed in rollout script. |
| Batch driver | `scripts/batch_m1_trajectory.sh` | Orchestrates rollout → analyze for checkpoints `200:200:4000`. |
| Smoke test (CI-style) | `scripts/m1_smoke.sh` | Short end-to-end check. |
| Figures from aggregates | `scripts/plot_m1_trajectory.py` | Regenerates PNGs from an aggregated CSV. |
| Methodology detail | `docs/m1_experimental_guideline.md` | Experimental design; this file is the **command cookbook**. |

Large binaries (`.pt` checkpoints, rollout `.parquet`) are usually **not** committed. Reproducibility = rerun training + scripts, **or** restore checkpoints from backup and skip to rollout. Small **aggregated CSVs** and **plot PNGs** may be committed if you want paper figures without rerunning the VM.

## Environment

```bash
cd situated-agency-alignment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: `WANDB_MODE=disabled` or `offline` for training without network (see `scripts/m1_smoke.sh`).

Record the **git commit** you used for a paper run:

```bash
git rev-parse HEAD
```

## 1. Train (produces checkpoints + training CSV)

Example: Env A, scarcity 0.30, baseline, seed 42 (results dir comes from the config’s `logging.local_results_dir`).

```bash
python train_karma.py \
  --config configs/m1_env_A_sc030.yaml \
  --mode baseline \
  --seed 42
```

Checkpoints follow:

`results/m1_env_A_sc030/checkpoints/m1_env_A_sc030_baseline_seed42_ep<E>.pt`

Training metrics CSV is under the same `results/...` tree (used by `aggregate_m1.py`).

## 2. Rollout + per-checkpoint analysis (VM or local)

From repo root. Fourth argument = eval episodes per checkpoint (default `20` in the script). Uses scratch disk for Parquet when available; see script header.

```bash
bash scripts/batch_m1_trajectory.sh \
  configs/m1_env_A_sc030.yaml \
  results/m1_env_A_sc030 \
  42 \
  20
```

Rerun **one** checkpoint (e.g. episode 4000): add a fifth argument `4000`.

Override scratch parent: `M1_SCRATCH_ROOT=/path/to/writable bash scripts/batch_m1_trajectory.sh ...`

Analysis JSONs:

`results/m1_env_A_sc030/analysis/trajectory_m1_env_A_sc030_baseline_seed42/m1_env_A_sc030_baseline_seed42_ep<E>.json`

## 3. Aggregate to one CSV

```bash
python scripts/aggregate_m1.py \
  --analysis-dir results/m1_env_A_sc030/analysis/trajectory_m1_env_A_sc030_baseline_seed42 \
  --training-dir results/m1_env_A_sc030 \
  --output results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv
```

Adjust `--analysis-dir` if your trajectory folder name differs.

## 4. Plots (local or CI)

```bash
python scripts/plot_m1_trajectory.py \
  --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv \
  --out results/m1_env_A_sc030/plots
```

Outputs numbered PNGs `01_*.png` … `07_*.png` under `--out`.

## 5. Optional: copy CSV from a remote machine

Example (replace host and paths):

```bash
mkdir -p results/m1_env_A_sc030
scp user@host:~/situated-agency-alignment/results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv \
  results/m1_env_A_sc030/
```

Then run **§4** only.

## 6. One-shot smoke test (short train + one analyze)

```bash
bash scripts/m1_smoke.sh 1
```

## Interpreting results (checklist)

**M1 scope (see `docs/m1_experimental_guideline.md`):** the registered M1 study uses **only the baseline** pipeline (`--mode baseline`). It does **not** require KARMA, Broken Mirror, or any intervention. Those belong to **M2** (motivated only if M1 supports the empathy-gap story). The checklist below is for **M1 baseline** trajectories; comparing `karma` / `broken` modes is **out of scope** for M1 unless you explicitly extend the protocol.

1. **Match the table** — Same `configs/m1_env_*_sc*.yaml` stem, **`--mode baseline`**, same seed list, same checkpoint interval (200 … 4000). Aggregates must be built with `aggregate_m1.py` from the matching `analysis/trajectory_*` folder.
2. **Behaviour first (training log + rollouts)** — Read in this order:
   - `AvgReturn_per_agent` and `AppleRate_per_agent_step` (welfare / commons health).
   - `ViolenceRate_per_agent_step` and `BeingZappedRate_per_agent_step` (harm rates; often correlated in symmetric multi-agent play).
   - `CooperationRate_per_agent_step` only when the env actually supports the cooperative zap channel (Env A tag-only runs may stay at zero by design).
3. **Then diagnostics (analysis JSON)** — Map to the guideline’s P1–P5 style predictions; treat exploratory columns as **hypothesis-generating** until the registry says otherwise:
   - `measurement_1_probes.probe_5way_auroc_mean` — 5-way decodability (P1-style).
   - `measurement_1_probes.probe_agg_vs_vic_auroc` — binary agg vs vic separability when `n` rules are met (interpret with class balance).
   - `measurement_2_cka.cka_agg_vs_vic` — role subspace similarity (P2-style; watch **single-checkpoint spikes**).
   - `measurement_3_rsa.cosdist_agg_vs_vic` — prototype geometry (exploratory unless promoted to primary).
   - `measurement_4_gradient_transfer.gradient_transfer_cos_mean` — cross-role gradient alignment (P5-style).
4. **Cross-cell comparisons (M1)** — For **confirmatory** claims, compare **seeds** and **scarcity levels on Env A** using the same baseline pipeline — not KARMA vs baseline. **Env B** is **exploratory** under the current registry; do not treat A vs B as a confirmatory factorial cell unless the preregistration is amended. Overlay trajectories only for conditions that the preregistration lists.
5. **Variance** — The pilot checklist asks for **≥3 seeds**. One seed validates the **code path**; confirmatory claims need the registered **N** and aggregation rule.

## Pilot vs confirmatory runs

- **Pilot (engineering):** Goal is to verify the **pipeline** (train → trajectory batch → aggregate → plots) and spot bugs or unstable configs. The guideline’s **single pilot cell** (`m1_env_A_sc030.yaml`, 4000 ep, checkpoints 200:200:4000, 20 eval eps) exists to stress-test analysis and **precedence** claims before scaling the **1×5×3 = 15** confirmatory grid on **Env A**.
- **Confirmatory / paper (M1):** Lock **primary vs exploratory** metrics, **Env A × five scarcities × three seeds** (baseline only), checkpoint cadence, eval episodes, and **`n_min = 100`** for CKA/binary agg–vic (row counts `n_ZAP_AGENT`, `n_BEING_ZAPPED`) — see §2.3 of `docs/m1_experimental_guideline.md` for the pilot table and rationale. If the ep4000 power check escalates to **80** eval episodes, follow the guideline’s amendment rule before considering a stricter `n_min`. **M2** (KARMA intervention) is a **separate** preregistration after M1 motivates it.

If the guideline doc lists M1 hypotheses and metrics, align the prereg with `docs/m1_experimental_guideline.md` so GitHub configs + this checklist match what you register.

## Run log (append new entries at the top)

Use this section like a lab notebook: **date**, **machine**, **git SHA**, **commands or script**, **outputs paths**, **notes**.

<!--
Template:

### YYYY-MM-DD — <short title>
- **Host:** (e.g. MacBook / Azure NC6s)
- **Commit:** `<git rev-parse HEAD>`
- **What:** (e.g. full trajectory for env A sc030 seed 42)
- **Commands:** (or “see §2–4 above” + any extra flags)
- **Artifacts:** `path/to/aggregated.csv`, `path/to/plots/`
- **Notes:** (wandb project, disk/scratch fixes, failures, etc.)
-->

### 2026-04-29 — Env B feasibility scouts and symmetric Env B calibration
- **Host:** Azure VM (`tapsvmT4`) for training; Mac for ablation analysis.
- **Why this happened:** before locking the OSF preregistration we tested whether **Env B (dual-use)** at `sc030` was actually viable as a confirmatory cell, and whether the design is **scientifically symmetric** to Env A (where aggression is purely instrumental, `zap_agent_reward = 0.0`).
- **Stage 1 - Env B baseline 4k scouts (original `m1_env_B_sc030.yaml`).**
  - Seed 42 and seed 123 each trained for 4000 episodes with `WANDB_MODE=online`.
  - Both seeds showed **healthy aggression growth** but **harvest collapse**: `AvgReturn_per_agent` dropped 84-92% from first-10% to last-10% window; `AppleRate_per_agent_step` dropped 85-93%. `BeamUseRate_per_agent_step` rose, indicating agents over-invested in beam use targeting waste (driven by `zap_waste_reward = 0.3`) at the cost of harvesting.
  - Conclusion: original Env B is **not safe to preregister** because the shaping reward dominates ecology.
- **Stage 2 - Env-pressure retune scouts (v1, 2k episodes).**
  - `m1_env_B_sc030_v1`: `waste_spawn_rate 0.10 -> 0.05`, `dynamic_waste_prob 0.02 -> 0.01`, `zap_waste_reward 0.3 -> 0.2`, episodes=2000, seed 42.
  - Result: `AvgReturn_per_agent -43%` (better than -92% but **still collapsing**).
- **Stage 3 - Symmetry argument and `waste_regrowth_suppression`.**
  - Decision: instead of lowering shaping, **remove it** (`zap_waste_reward = 0.0`) and make cleanup **truly instrumental** by letting waste suppress nearby apple regrowth. Implementation: new env knob `waste_regrowth_suppression` (alpha) added to `karmic_rl/envs/harvest_dual.py`, plumbed through `train_karma.py` and `scripts/rollout_from_checkpoint.py`. Defaults to `0.0` so prior runs are bit-identical.
  - New file `configs/m1_env_B_sc030_sym.yaml` with `zap_waste_reward: 0.0` and `waste_regrowth_suppression: 0.05`.
  - New no-agent ablation `scripts/ablate_waste_regrowth.py` (uniform random actions, fixed seed).
- **Stage 4 - Alpha sweep at original waste density (`waste_spawn_rate=0.10`, 4 seeds).**
  - alpha=0.05 effect was noise (-0.06% mean apples on seed 0); alpha=0.30 caused **catastrophic collapse** on seed 2 (mean apples 0.53). High variance across seeds; no Goldilocks zone.
- **Stage 5 - Reduce Env B waste density and re-ablate (8 seeds).**
  - All Env B configs (`m1_env_B_sc015/030/050.yaml` + `m1_env_B_sc030_sym.yaml`) updated: `waste_spawn_rate: 0.10 -> 0.04`, `dynamic_waste_prob: 0.02 -> 0.005`. (Env A unchanged.)
  - Result with reduced waste: A vs B base apple parity is now reasonable (mean delta -3.4%, range -14.9% to +7.2% across seeds). **But** the alpha knob is essentially ineffective at this waste density: even alpha=0.30 changes mean apples by only 0.05-1.3% versus alpha=0. Reason: at ~2-5 mean waste cells, the average empty cell sees almost no waste neighbors, so linear neighborhood-suppression cannot bite.
- **Stage 6 - Add canonical Cleanup-style `waste_spread_prob`.**
  - Hypothesis: linear local suppression alone cannot make cleanup instrumental at safe waste densities. We need a **non-linear** mechanic where unchecked waste **grows**.
  - Implementation: new env knob `waste_spread_prob` and a `_propagate_waste()` step in `karmic_rl/envs/harvest_dual.py`: each existing WASTE cell, every step, with probability `waste_spread_prob`, spreads to one uniformly random EMPTY 4-neighbor. Backward compatible (default 0.0; with `spread=0` and `alpha=0` the env is bit-identical to before).
  - Configs updated: `m1_env_B_sc030.yaml` and `m1_env_B_sc030_sym.yaml` set `waste_spread_prob: 0.02`. (Sc015 / sc050 / Env A untouched until sc030 ablation locks.)
  - `scripts/ablate_waste_regrowth.py` extended to sweep the (alpha, spread) cross-product and report waste counts at t=250/500/1000 to expose non-linear waste growth.
- **Status:** the next ablation across 8 seeds with the new mechanic determines the final (alpha, spread) pair to lock in `m1_env_B_sc030_sym.yaml`.
- **Files touched (representative):** `karmic_rl/envs/harvest_dual.py`, `train_karma.py`, `scripts/rollout_from_checkpoint.py`, `scripts/ablate_waste_regrowth.py`, `configs/m1_base.yaml`, `configs/m1_env_B_sc015.yaml`, `configs/m1_env_B_sc030.yaml`, `configs/m1_env_B_sc030_sym.yaml`, `configs/m1_env_B_sc050.yaml`, `docs/M1_complete_guide.md` (section 2.1).
- **Decision impact on prereg:** Env B remains a candidate confirmatory cell pending the (alpha, spread) lock-in ablation; otherwise fall back to Env A only for M1.

### 2026-04-28 — ep4000 power check (20 vs 4x20 ~ 80-eval equivalent)
- **Host:** Azure VM (`tapsvmT4`)
- **What:** Power check for whether M1 should escalate from 20 to 80 eval episodes per checkpoint.
- **Commands:** Four independent `scripts/rollout_from_checkpoint.py` runs at `--episodes 20` for `ep4000`, followed by `scripts/analyze_checkpoint.py` on each part JSON. The single-process `--episodes 80` rollout and merged-parquet analysis were both operationally unstable (killed with exit 137), so the comparison used **mean(4x20)** as an 80-eval equivalent estimate.
- **Artifacts:** `results/m1_env_A_sc030/analysis/power_check/ep4000_eval20_part{1,2,3,4}.json`
- **Notes:** Final comparison vs the original 20-episode ep4000 JSON: `probe_5way_auroc_mean 0.8135 -> 0.8250`, `probe_agg_vs_vic_auroc 0.2571 -> 0.3082`, `cka_agg_vs_vic 0.00875 -> 0.01239`, `gradient_transfer_cos_mean -0.10761 -> -0.10136`. Decision: **keep 20 eval episodes** for the main campaign; no global amendment to 80. The binary agg-vic probe was the most sampling-sensitive metric.

### 2026-04-24 — Plots from synced aggregate (local Mac)
- **Host:** Mac (repo: `situated-agency-alignment`)
- **What:** Regenerated figure PNGs from VM-produced aggregate CSV.
- **Commands:** `python3 scripts/plot_m1_trajectory.py --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv --out results/m1_env_A_sc030/plots`
- **Artifacts:** `results/m1_env_A_sc030/plots/01_behaviour_rates.png` … `07_summary_2x2.png`
- **Notes:** Training + batch analysis were run elsewhere; CSV was synced into `results/m1_env_A_sc030/` before plotting.
