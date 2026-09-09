# Canonical Baseline Replication on Azure VM

This guide is only for **Phase A** (canonical baseline replication).
Do not enable dual-use cleanup dynamics or KARMA mode yet.

## 1) Clone private GitHub repo on VM

Use an SSH deploy key (recommended for a single private repo):

```bash
ssh-keygen -t ed25519 -C "azure-vm-replication" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add the printed public key to your private GitHub repo:
- Repo -> Settings -> Deploy keys -> Add deploy key
- Enable read access (write only if you need to push from VM)

Then verify and clone:

```bash
ssh -T git@github.com
git clone git@github.com:<OWNER>/<REPO>.git
cd <REPO>
```

## 2) Python environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=.
```

Optional GPU check:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 3) Login to Weights & Biases (optional but recommended)

```bash
wandb login
```

If you want local-only runs, set `use_wandb: false` in the config.

## 4) Run canonical baseline (single seed)

```bash
python train_karma.py \
  --config configs/canonical_baseline.yaml \
  --mode baseline \
  --seed 42
```

Artifacts are saved locally under:
- `results/canonical_baseline/*.csv`
- `results/canonical_baseline/*.json`

## 5) Run small multi-seed sweep (recommended)

```bash
for SEED in 11 22 33; do
  python train_karma.py \
    --config configs/canonical_baseline.yaml \
    --mode baseline \
    --seed ${SEED}
done
```

## 6) What to inspect after runs

- In logs/W&B:
  - `ViolenceRate_per_agent_step`
  - `CooperationRate_per_agent_step`
  - `AppleRate_per_agent_step`
  - `AvgReturn_per_agent`
  - `EthicalSelectivity`
- In local artifact JSON:
  - `resolved_env` block to verify runtime config parity
  - `final_metrics` for quick seed-to-seed comparison

## 7) Expected Phase A outcome

Before moving to Phase B, confirm:
- Baseline mode runs with no KARMA contrastive objective.
- Canonical config is faithfully reflected in `resolved_env`.
- Multi-seed trends are coherent enough to establish a replication anchor.
