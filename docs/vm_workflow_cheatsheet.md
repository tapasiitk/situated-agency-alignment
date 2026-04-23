# VM + Repo Workflow Cheat Sheet

Quick reference for running experiments on the Azure T4 VM and the private GitHub repo inside it.

---

## 1. VM lifecycle (from Mac)

| Action            | Command     |
|-------------------|-------------|
| Start VM          | `vmstart`   |
| Stop VM (billing) | `vmstop`    |
| SSH into VM       | `vmssh`     |
| Check status      | `vmstatus`  |

VM details:
- Name: `tapsvmT4`
- Size: `Standard NC16as T4 v3` (1x Tesla T4 GPU)
- Resource group: `tapsvmT4_group_04220752`
- Region: `westus2`
- Cost: ~$1.20/hr when running

Always `vmstop` after experiments to save cost.

---

## 2. Repo location on VM

```
~/situated-agency-alignment
```

- Python venv: `~/situated-agency-alignment/.venv`
- Deploy key configured (so `git pull` / `checkout` works for private repo)
- W&B configured (`~/.netrc`) for user `ratht-iitk`

---

## 3. Useful VM commands

Activate env + set paths:
```bash
cd ~/situated-agency-alignment
source .venv/bin/activate
export PYTHONPATH=.
```

GPU sanity check:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Git on VM

Sync branch to latest:
```bash
cd ~/situated-agency-alignment
git fetch origin
git checkout <branch-name>
git pull --ff-only
```

Useful branches (examples):
- `phase-a-canonical-replication`
- `phase-a-speed-best-current`
- `phase-a-stability-logging-m1`

---

## 5. Running training

Basic run:
```bash
python train_karma.py --config <config.yaml> --mode baseline --seed 42
```

With W&B online + both VM log file and W&B console capture:
```bash
WANDB_MODE=online \
python -u train_karma.py --config <config.yaml> --mode baseline --seed 42 \
  2>&1 | tee run_logs/<run>.log
```

Offline W&B (no cloud sync):
```bash
WANDB_MODE=offline python -u train_karma.py ...
```

No W&B at all:
```bash
WANDB_MODE=disabled python -u train_karma.py ...
```

---

## 6. Tmux (for long/overnight runs)

Create session and run:
```bash
tmux new -d -s run1 \
'cd ~/situated-agency-alignment && source .venv/bin/activate && export PYTHONPATH=. && \
 WANDB_MODE=online python -u train_karma.py --config <config.yaml> --mode baseline --seed 42 \
 2>&1 | tee run_logs/run1.log'
```

Session controls:
- List sessions: `tmux ls`
- Attach: `tmux attach -t run1`
- Detach (inside session): `Ctrl-b` then `d`
- Kill session: `tmux kill-session -t run1`

---

## 7. Monitoring a run

Live log:
```bash
tail -f ~/situated-agency-alignment/run_logs/<run>.log
```

Latest episode checkpoint:
```bash
awk '/^Ep /{last=$0} END{print last}' ~/situated-agency-alignment/run_logs/<run>.log
```

Active training process:
```bash
pgrep -af train_karma.py
```

---

## 8. Auto-shutdown after run (overnight pattern)

VM shutdown from inside VM requires `sudo -n`:
```bash
sudo -n shutdown -h now
```

Minimal watcher (runs in its own tmux session):
```bash
#!/usr/bin/env bash
set -e
LOG=~/situated-agency-alignment/run_logs/<run>.log
while pgrep -f "train_karma.py --config <config.yaml>" >/dev/null 2>&1; do
  sleep 60
done
echo "Job done. Shutting down VM in 60s..." | tee -a "$LOG"
sleep 60
sudo -n shutdown -h now >> "$LOG" 2>&1 || true
```

Verify from Mac afterwards:
```bash
vmstatus
```
If status is not `stopped`, run `vmstop`.

---

## 9. Output locations on VM

- Training logs: `~/situated-agency-alignment/run_logs/`
- Local metrics (CSV + JSON): `~/situated-agency-alignment/results/<run-name>/`
- Checkpoints: `~/situated-agency-alignment/results/<run-name>/checkpoints/`
- W&B local data: `~/situated-agency-alignment/wandb/`

Sync offline W&B runs later:
```bash
wandb sync ~/situated-agency-alignment/wandb/offline-run-*
```

---

## 10. End-of-session hygiene

1. Confirm training completed (log + W&B).
2. `git add -A && git commit -m "..." && git push` from **local Mac**, not VM, unless required.
3. `vmstop` to stop billing.

---

## 11. M1 pipeline (empathy-gap diagnostics)

One-time deps in the VM venv (analysis uses scikit-learn; Parquet needs pyarrow):

```bash
cd ~/situated-agency-alignment
source .venv/bin/activate
export PYTHONPATH=.
pip install -U scikit-learn pyarrow
```

Sync the `m1-pipeline` branch (or whatever branch carries the M1 scripts), then end-to-end smoke (40 train episodes, small rollout, analysis JSON):

```bash
git fetch origin
git checkout m1-pipeline
git pull --ff-only
bash scripts/m1_smoke.sh 1
```

Artifacts land under `results/m1_smoke40/` (checkpoints, rollouts, analysis). Use `WANDB_MODE=disabled` for smoke; the smoke script already sets it for training.

Manual steps (same pattern as §5):

```bash
WANDB_MODE=disabled python -u train_karma.py --config configs/m1_smoke40.yaml --mode baseline --seed 1 \
  2>&1 | tee run_logs/m1_smoke40.log

python scripts/rollout_from_checkpoint.py \
  --config configs/m1_smoke40.yaml \
  --checkpoint results/m1_smoke40/checkpoints/m1_smoke40_baseline_seed1_ep40.pt \
  --episodes 4 \
  --output results/m1_smoke40/rollouts/m1_smoke40_baseline_seed1_ep40.parquet

python scripts/analyze_checkpoint.py \
  --rollout results/m1_smoke40/rollouts/m1_smoke40_baseline_seed1_ep40.parquet \
  --checkpoint results/m1_smoke40/checkpoints/m1_smoke40_baseline_seed1_ep40.pt \
  --config configs/m1_smoke40.yaml \
  --output results/m1_smoke40/analysis/m1_smoke40_baseline_seed1_ep40.json
```

If Parquet is unavailable, rollout falls back to `.jsonl` in the same directory; pass that path to `analyze_checkpoint.py` instead.
