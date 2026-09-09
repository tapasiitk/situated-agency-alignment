# M0, M1, and M2 Replication Runbook

This is the operational guide for reproducing the frozen-ecology program on a
fresh machine or handing it to a new teammate. Follow the stages in order:

```text
M0 behavior-only ecology calibration -> freeze one ecology ->
M1 baseline mechanism test -> M2 matched interventions
```

Do not select an M0 ecology using probes, CKA, RSA, gradient transfer, KARMA,
or Broken Mirror results. Do not start M2 until M1 has been completed and
reviewed under the frozen M0 ecology.

> **Current M2 boundary:** the configured `broken` mode binds `ZAP_AGENT` to
> `ZAP_WASTE`, while the frozen Env A config disables waste events. It is an
> engineering placeholder, not a valid confirmatory scrambled control for Env
> A. Do not report it as a fair M2 comparison until its target event is
> corrected and the design decision is recorded.

## 1. What Is Frozen

The current provisional frozen ecology is Env A candidate H:

```text
envA_n6_ad030_rg075_zt25
```

Its canonical baseline M1 config is:

```text
configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml
```

It uses six agents, `apple_density: 0.30`, `regrowth_speed: 0.75`, and
`zap_timeout: 25`. Env A is harm-only in practice because the config disables
the waste/cleanup channel:

```yaml
waste_spawn_rate: 0.0
dynamic_waste_enabled: false
zap_waste_reward: 0.0
zap_agent_reward: 0.0
victim_penalty: 0.0
```

M2 must retain every environment and ordinary training setting from this M1
config. It may change only the intervention condition (`--mode`) and its
representational intervention weight. See `manifests/m2_intervention.yaml`.

## 2. Repository Map

| Purpose | Canonical location |
|---|---|
| M0 ecology configs | `configs/m0_ecology_calibration/` |
| Frozen M1 config | `configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml` |
| M2 configs | `configs/m2_intervention/` |
| Study specifications | `manifests/m0_ecology_calibration.yaml`, `manifests/m1_frozen_mechanism.yaml`, `manifests/m2_intervention.yaml` |
| Training entry point | `train_karma.py` |
| Rollout generator | `scripts/rollout_from_checkpoint.py` |
| Checkpoint analysis | `scripts/analyze_checkpoint.py` |
| Aggregation | `scripts/aggregate_m1.py` |
| M0 figures | `scripts/plot_m0_ecology_calibration.py` |
| M1 figures | `scripts/plot_m1_mechanism_frozen_ecology.py` |
| Scratch batch helpers | `scripts/batch_m1_trajectory*.sh` |

Root-level `configs/stage0_*.yaml` and `configs/m1_env_A_sc*.yaml` are legacy
compatibility paths. They are not the preferred starting point for a new M1
campaign. Use the frozen M1 config above unless reproducing a historical run.

## 3. VM Setup

The following lines marked **VM-SPECIFIC** describe the current Azure VM
convention. Change only those values when using another machine.

From a Mac, connect to the VM:

```bash
# VM-SPECIFIC: `tapsvmT4` is the SSH alias in the current user's ~/.ssh/config.
ssh tapsvmT4
```

On the VM, prepare the repository:

```bash
# VM-SPECIFIC: repository location on the current VM.
cd ~/situated-agency-alignment

# VM-SPECIFIC: virtual environment location for this project.
source .venv/bin/activate
export PYTHONPATH=.

git fetch origin
git checkout main
git pull --ff-only

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected GPU check output includes `True` and the GPU model. If it does not,
stop before launching training.

### Scratch Space

Large rollout parquet files must go to the VM's ephemeral disk, not the root
disk that contains `~/situated-agency-alignment`.

```bash
# VM-SPECIFIC: `/mnt` is the large ephemeral disk on the current VM.
sudo -n mkdir -p /mnt/karma_m1_archive
sudo -n chown "$USER:$USER" /mnt/karma_m1_archive
df -h / /mnt
```

`/mnt` may be erased when the VM is deallocated. Keep checkpoints, aggregate
CSVs, summary JSONs, and final figures under `results/`; treat parquets on
`/mnt` as reusable but disposable working data.

## 4. M0: Calibrate or Verify the Ecology

### 4.1 Start an M0 Candidate

Use only the behavioral M0 configs. Example for candidate H:

```bash
cd ~/situated-agency-alignment  # VM-SPECIFIC repository path
source .venv/bin/activate       # VM-SPECIFIC virtual environment
export PYTHONPATH=.

WANDB_MODE=online python -u train_karma.py \
  --config configs/m0_ecology_calibration/stage0_env_A_H_n6_ad030_rg075_zt25.yaml \
  --mode baseline \
  --seed 42 \
  2>&1 | tee run_logs/m0_H_seed42.log
```

Run the seed list declared in `manifests/m0_ecology_calibration.yaml`. A
candidate should normally be trained to 2,000 episodes first; extend promising
ones to 4,000 before freezing.

### 4.2 M0 Behavioral Gate

For selection, examine only:

- `ViolenceRate_per_agent_step`
- `BeingZappedRate_per_agent_step`
- `BeamUseRate_per_agent_step`
- `AppleRate_per_agent_step`
- `AvgReturn_per_agent`
- counts of `ZAP_AGENT` and `BEING_ZAPPED`

Generate behavior-only M0 plots:

```bash
python scripts/plot_m0_ecology_calibration.py \
  --results-dir results \
  --out results/m0_ecology_calibration/figures \
  --late-start-episode 1000
```

An ecology can be frozen only when its late training window has:

1. Sustained, non-zero violence.
2. Sustained, non-zero being-zapped rate.
3. Non-degenerate apple rate and return.
4. Enough aggressor and victim events for later M1 analysis.

For item 4, make short rollouts from late checkpoints and count the role names.
This uses rollouts only; do **not** inspect representation outputs to select
the ecology.

```bash
# VM-SPECIFIC: keep this large rollout file on the ephemeral disk.
M0_SCRATCH=/mnt/karma_m1_archive/m0_event_mass_seed42_ep4000.parquet

python scripts/rollout_from_checkpoint.py \
  --config configs/m0_ecology_calibration/stage0_env_A_H_n6_ad030_rg075_zt25.yaml \
  --checkpoint results/stage0_env_A_H_n6_ad030_rg075_zt25/checkpoints/stage0_env_A_H_n6_ad030_rg075_zt25_baseline_seed42_ep4000.pt \
  --episodes 20 \
  --output "$M0_SCRATCH" \
  --device cuda

python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("/mnt/karma_m1_archive/m0_event_mass_seed42_ep4000.parquet")
df = pd.read_parquet(path)
print("ZAP_AGENT:", int((df["role_name"] == "ZAP_AGENT").sum()))
print("BEING_ZAPPED:", int((df["role_name"] == "BEING_ZAPPED").sum()))
PY
```

Record the decision, seeds, episode count, checkpoint cadence, late window,
and event-mass evidence in `docs/M1_2_related/design_decisions.md` before M1.

## 5. M1: Frozen-Ecology Baseline Mechanism Test

### 5.1 Train Matched Baseline Seeds

The recommended confirmatory seeds are declared in
`manifests/m1_frozen_mechanism.yaml`:

```text
42, 123, 202, 303, 404
```

Run all seeds with the **same** config, mode, episode budget, and checkpoint
cadence. The following starts one seed; use separate tmux sessions or run
sequentially if the VM has one GPU.

```bash
# VM-SPECIFIC: tmux preserves the run after SSH disconnects.
tmux new -d -s m1_baseline_42 \
  'cd ~/situated-agency-alignment && source .venv/bin/activate && export PYTHONPATH=. && \
   WANDB_MODE=online python -u train_karma.py \
     --config configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml \
     --mode baseline \
     --seed 42 \
     2>&1 | tee run_logs/m1_baseline_seed42.log'
```

Monitor:

```bash
tmux ls
tail -f run_logs/m1_baseline_seed42.log
pgrep -af train_karma.py
```

Do not launch multiple GPU training runs concurrently on the single-GPU VM.

### 5.2 Roll Out and Analyze Checkpoints on Scratch

Use the kept-parquet batch helper when there is enough `/mnt` space. It creates
parquets on scratch, creates durable analysis JSONs under `results/`, and
skips an analysis JSON that already exists.

```bash
# VM-SPECIFIC: this location is on the VM's ephemeral scratch disk.
export M1_SCRATCH_ROOT=/mnt/karma_m1_archive/m1_baseline_seed42

M1_POSTPROCESS_MODE=baseline \
WANDB_MODE=online \
bash scripts/batch_m1_trajectory_keep_parquet.sh \
  configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml \
  results/m1_env_A_frozen_n6_ad030_rg075_zt25 \
  42 \
  20
```

The final `20` means 20 evaluation episodes per checkpoint. Without a fifth
argument, the helper processes all scheduled checkpoints from episode 200 to
4000. To process exactly one checkpoint, append its episode number:

```bash
M1_SCRATCH_ROOT=/mnt/karma_m1_archive/m1_baseline_seed42 \
M1_POSTPROCESS_MODE=baseline \
bash scripts/batch_m1_trajectory_keep_parquet.sh \
  configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml \
  results/m1_env_A_frozen_n6_ad030_rg075_zt25 \
  42 \
  20 \
  4000
```

Use `scripts/batch_m1_trajectory.sh` instead when parquets do not need to be
kept. It deletes a parquet only after successful analysis, conserving scratch
space while retaining the durable JSON analysis output.

### 5.3 Aggregate and Plot M1

After every desired seed has analysis JSONs, aggregate all of them together:

```bash
python scripts/aggregate_m1.py \
  --analysis-dir results/m1_env_A_frozen_n6_ad030_rg075_zt25/analysis \
  --training-dir results/m1_env_A_frozen_n6_ad030_rg075_zt25 \
  --output results/m1_env_A_frozen_n6_ad030_rg075_zt25/aggregated_baseline.csv

python scripts/plot_m1_mechanism_frozen_ecology.py \
  --csv results/m1_env_A_frozen_n6_ad030_rg075_zt25/aggregated_baseline.csv \
  --out results/m1_env_A_frozen_n6_ad030_rg075_zt25/m1_mechanism_figures \
  --late-start-episode 2000 \
  --n-min 100
```

Read `summary_m1_mechanism.json` before making any M2 decision. H1 measures
role separability; H2 measures whether victim-side value gradients align with
aggressor-side action gradients. Both are M1 mechanism outcomes, not M0
selection criteria.

## 6. M2: Matched Intervention Workflow

Only start M2 after the M0 freeze is recorded and the M1 outcome has been
reviewed. Use exactly the same seed list, episode budget, evaluation episodes,
checkpoint cadence, and frozen ecology as M1. The KARMA configuration is ready
for development and preregistration work; a valid Env A scrambled-control
configuration must be defined before a confirmatory multi-arm M2 comparison.

### 6.1 Train KARMA

```bash
WANDB_MODE=online python -u train_karma.py \
  --config configs/m2_intervention/m2_env_A_frozen_n6_ad030_rg075_zt25_karma.yaml \
  --mode karma \
  --seed 42 \
  2>&1 | tee run_logs/m2_karma_seed42.log
```

### 6.2 Broken Mirror Placeholder (Do Not Use Confirmatorily in Frozen Env A)

The command below is provided only to reproduce the current engineering
placeholder or to smoke-test code paths. It must not be presented as an Env A
scrambled-control result because `ZAP_WASTE` event rows are absent by design.

```bash
WANDB_MODE=online python -u train_karma.py \
  --config configs/m2_intervention/m2_env_A_frozen_n6_ad030_rg075_zt25_broken.yaml \
  --mode broken \
  --seed 42 \
  2>&1 | tee run_logs/m2_broken_seed42.log
```

### 6.3 Post-process M2 on Scratch

Set the matching mode. This is essential because checkpoint filenames contain
the condition name.

```bash
# KARMA condition
M1_SCRATCH_ROOT=/mnt/karma_m1_archive/m2_karma_seed42 \
M1_POSTPROCESS_MODE=karma \
bash scripts/batch_m1_trajectory_keep_parquet.sh \
  configs/m2_intervention/m2_env_A_frozen_n6_ad030_rg075_zt25_karma.yaml \
  results/m2_intervention/env_A_frozen_n6_ad030_rg075_zt25/karma \
  42 \
  20

# Broken Mirror engineering placeholder only; not a confirmatory Env A arm.
M1_SCRATCH_ROOT=/mnt/karma_m1_archive/m2_broken_seed42 \
M1_POSTPROCESS_MODE=broken \
bash scripts/batch_m1_trajectory_keep_parquet.sh \
  configs/m2_intervention/m2_env_A_frozen_n6_ad030_rg075_zt25_broken.yaml \
  results/m2_intervention/env_A_frozen_n6_ad030_rg075_zt25/broken \
  42 \
  20
```

Aggregate each condition separately. Preserve the condition name in every
output filename and never merge baseline, KARMA, and control rows into a
single analysis CSV before calculating condition comparisons.

## 7. Preserve Small Results and Manage Disk Space

Keep these durable artifacts under `results/` and sync them off the VM:

- training CSVs and training summary JSONs;
- aggregate CSVs;
- `summary_m1_mechanism.json` and equivalent M2 summaries;
- final PNG figures;
- the M0 gate report and freeze record.

Do not copy raw rollout parquets unless a later robustness analysis requires
them. They are large and belong on `/mnt` during active work.

From a Mac, copy lightweight results:

```bash
# VM-SPECIFIC: `tapsvmT4` is the current SSH alias and the destination is this
# developer's local checkout. Change both for another team member.
mkdir -p ~/cgsExperiments/situated-agency-alignment/results/synced_external
scp -r tapsvmT4:~/situated-agency-alignment/results/m1_env_A_frozen_n6_ad030_rg075_zt25/m1_mechanism_figures \
  ~/cgsExperiments/situated-agency-alignment/results/synced_external/
```

Check disk state before a long rollout batch:

```bash
df -h / /mnt
du -sh /mnt/karma_m1_archive/* 2>/dev/null
```

When a retained-parquet batch is no longer needed, delete only its explicit
scratch subdirectory after confirming durable aggregate/figure outputs exist.
Never delete a broad `results/` directory to reclaim VM disk.

## 8. End-of-Run Checklist

Before ending a stage, confirm:

- `git status --short` shows only expected work;
- all target seeds reached 4000 episodes;
- expected checkpoint files exist;
- analysis JSON count matches the intended seeds x checkpoints;
- aggregate CSV and summary JSON were generated;
- lightweight results have been synced off the VM;
- no GPU training or post-process job is active (`pgrep -af train_karma.py`);
- `/mnt` space and root-disk space are acceptable.

The current VM may remain running while active work continues. When finished,
stop/deallocate it using your institution's approved VM command to avoid GPU
charges.

## 9. Troubleshooting

| Symptom | What to do |
|---|---|
| `scratch root is not writable` | Run the `/mnt` `sudo mkdir` and `chown` setup in Section 3, then retry. |
| Root disk fills during rollouts | Stop the job, verify `M1_SCRATCH_ROOT` starts with `/mnt/`, and use a batch helper rather than `--output results/...`. |
| Batch helper skips a checkpoint | It either cannot find the expected checkpoint or its analysis JSON already exists. Read the printed path and verify the training mode/seed. |
| M2 helper cannot find checkpoints | Set `M1_POSTPROCESS_MODE=karma` or `M1_POSTPROCESS_MODE=broken`; the default is `baseline`. |
| H1/H2 have no eligible late rows | Check `n_aggressor` and `n_victim`; the default `n_min` is 100. This is a data-feasibility issue, not a reason to alter the frozen ecology after inspecting representation metrics. |
| VM disconnects | Reconnect and use `tmux ls`; detached training and post-processing sessions continue running. |
