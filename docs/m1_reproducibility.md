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
4. **Cross-cell comparisons (M1)** — Compare **seeds**, **scarcity levels**, and **env A vs B** using the same baseline pipeline — not KARMA vs baseline. Overlay trajectories only for conditions that the preregistration lists.
5. **Variance** — The pilot checklist asks for **≥3 seeds**. One seed validates the **code path**; confirmatory claims need the registered **N** and aggregation rule.

## Pilot vs confirmatory runs

- **Pilot (engineering):** Goal is to verify the **pipeline** (train → trajectory batch → aggregate → plots) and spot bugs or unstable configs. The guideline’s **single pilot cell** (`m1_env_A_sc030.yaml`, 4000 ep, checkpoints 200:200:4000, 20 eval eps) exists to stress-test analysis and **precedence** claims before scaling the **2×3×5** (or similar) baseline factorial.
- **Confirmatory / paper (M1):** Lock **primary vs exploratory** metrics, **env × scarcity × seeds** (baseline only), checkpoint cadence, eval episodes, and **`n_min = 100`** for CKA/binary agg–vic (row counts `n_ZAP_AGENT`, `n_BEING_ZAPPED`) — see §2.3 of `docs/m1_experimental_guideline.md` for the pilot table and rationale. If the ep4000 power check escalates to **80** eval episodes, follow the guideline’s amendment rule before considering a stricter `n_min`. **M2** (KARMA intervention) is a **separate** preregistration after M1 motivates it.

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

### 2026-04-24 — Plots from synced aggregate (local Mac)
- **Host:** Mac (repo: `situated-agency-alignment`)
- **What:** Regenerated figure PNGs from VM-produced aggregate CSV.
- **Commands:** `python3 scripts/plot_m1_trajectory.py --csv results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv --out results/m1_env_A_sc030/plots`
- **Artifacts:** `results/m1_env_A_sc030/plots/01_behaviour_rates.png` … `07_summary_2x2.png`
- **Notes:** Training + batch analysis were run elsewhere; CSV was synced into `results/m1_env_A_sc030/` before plotting.
