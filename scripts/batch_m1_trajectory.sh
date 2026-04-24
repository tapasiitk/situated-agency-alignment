#!/usr/bin/env bash
# Batch rollout + analyze for every checkpoint from a finished M1 training run.
# No retraining — only loads existing .pt files.
#
# Large rollout tables (.parquet) are written to Azure ephemeral disk (/mnt) when
# available so the OS root volume does not fill. Analysis JSONs stay under
# results/ on the OS disk (durable path). Each temp parquet is removed after
# analyze_checkpoint.py succeeds.
#
# Usage (from repo root, VM or local):
#   bash scripts/batch_m1_trajectory.sh \
#       configs/m1_env_A_sc030.yaml \
#       results/m1_env_A_sc030 \
#       42 \
#       20
#
# Optional 5th arg: single checkpoint episode only (e.g. 4000) to rerun one step.
#
# Args:
#   $1  path to config yaml (must match training)
#   $2  results directory for that run (contains checkpoints/)
#   $3  training seed
#   $4  eval episodes per checkpoint (default 20)
#   $5  optional: only this episode (e.g. 4000)
#
# Env:
#   M1_SCRATCH_ROOT  override scratch parent (default: /mnt/karma_m1_scratch if /mnt exists)
#
# Output naming matches scripts/aggregate_m1.py:
#   <config_stem>_<mode>_seed<seed>_ep<ep>.json
#
set -euo pipefail

CFG="${1:?config yaml path}"
RESULTS_DIR="${2:?results dir}"
SEED="${3:?training seed}"
EVAL_EPISODES="${4:-20}"
SINGLE_EP="${5:-}"

CONFIG_STEM=$(basename "$CFG" .yaml)
MODE=baseline
RUN_PREFIX="${CONFIG_STEM}_${MODE}_seed${SEED}"

CKPT_DIR="${RESULTS_DIR}/checkpoints"
ANALYSIS_DIR="${RESULTS_DIR}/analysis/trajectory_${RUN_PREFIX}"

# Parquets: use a writable scratch path so the OS root volume is not exhausted.
# - Azure often mounts a large ephemeral disk at /mnt but leaves it root-owned; create a
#   writable subdir once:  sudo mkdir -p /mnt/karma_m1_scratch && sudo chown "$USER:$USER" /mnt/karma_m1_scratch
# - If /mnt is not writable, we fall back to /dev/shm (tmpfs, typically ~50G+ on NC VMs).
# - Override entirely: M1_SCRATCH_ROOT=/path
pick_scratch_parent() {
  if [[ -n "${M1_SCRATCH_ROOT:-}" ]]; then
    echo "${M1_SCRATCH_ROOT}"
    return
  fi
  if [[ -d /mnt/karma_m1_scratch && -w /mnt/karma_m1_scratch ]]; then
    echo "/mnt/karma_m1_scratch"
    return
  fi
  if [[ -d /mnt && -w /mnt ]]; then
    echo "/mnt/karma_m1_scratch"
    return
  fi
  local shm="/dev/shm/karma_m1_scratch"
  mkdir -p "$shm" 2>/dev/null || true
  if [[ -d "$shm" && -w "$shm" ]]; then
    echo "$shm"
    return
  fi
  echo ""
}

SCRATCH_PARENT="$(pick_scratch_parent)"

if [[ -n "$SCRATCH_PARENT" ]]; then
  ROLLOUT_DIR="${SCRATCH_PARENT}/${RUN_PREFIX}"
  mkdir -p "$ROLLOUT_DIR"
  echo "[batch] temp parquets -> ${ROLLOUT_DIR} (scratch; cleared after each analyze — not for long-term storage)"
else
  ROLLOUT_DIR="${RESULTS_DIR}/rollouts/trajectory_${RUN_PREFIX}"
  mkdir -p "$ROLLOUT_DIR"
  echo "[batch] temp parquets -> ${ROLLOUT_DIR} (fallback: results dir — watch OS disk usage)"
fi

mkdir -p "$ANALYSIS_DIR"

if [[ -n "$SINGLE_EP" ]]; then
  EP_SEQ=("$SINGLE_EP")
else
  EP_SEQ=($(seq 200 200 4000))
fi

for EP in "${EP_SEQ[@]}"; do
  CKPT="${CKPT_DIR}/${RUN_PREFIX}_ep${EP}.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] missing checkpoint: $CKPT"
    continue
  fi
  ROLLOUT="${ROLLOUT_DIR}/${RUN_PREFIX}_ep${EP}.parquet"
  ANALYSIS="${ANALYSIS_DIR}/${RUN_PREFIX}_ep${EP}.json"
  if [[ -f "$ANALYSIS" ]]; then
    echo "[skip] already analyzed: $ANALYSIS"
    continue
  fi
  echo "=== ep $EP ==="
  python scripts/rollout_from_checkpoint.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --episodes "$EVAL_EPISODES" \
    --output "$ROLLOUT"
  python scripts/analyze_checkpoint.py \
    --rollout "$ROLLOUT" \
    --checkpoint "$CKPT" \
    --config "$CFG" \
    --output "$ANALYSIS"
  rm -f "$ROLLOUT"
  echo "[scratch] removed temp parquet after successful analyze: ${ROLLOUT}"
done

echo "Done. Aggregate with:"
echo "  python scripts/aggregate_m1.py \\"
echo "    --analysis-dir ${ANALYSIS_DIR} \\"
echo "    --training-dir ${RESULTS_DIR} \\"
echo "    --output ${RESULTS_DIR}/aggregated_${RUN_PREFIX}.csv"
