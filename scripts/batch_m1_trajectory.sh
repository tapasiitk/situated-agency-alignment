#!/usr/bin/env bash
# Batch rollout + analyze for every checkpoint from a finished M1 training run.
# No retraining — only loads existing .pt files.
#
# Usage (from repo root, VM or local):
#   bash scripts/batch_m1_trajectory.sh \
#       configs/m1_env_A_sc030.yaml \
#       results/m1_env_A_sc030 \
#       42 \
#       20
#
# Args:
#   $1  path to config yaml (must match training)
#   $2  results directory for that run (contains checkpoints/)
#   $3  training seed
#   $4  eval episodes per checkpoint (default 20)
#
# Output naming matches scripts/aggregate_m1.py:
#   <config_stem>_<mode>_seed<seed>_ep<ep>.json
#
set -euo pipefail

CFG="${1:?config yaml path}"
RESULTS_DIR="${2:?results dir}"
SEED="${3:?training seed}"
EVAL_EPISODES="${4:-20}"

CONFIG_STEM=$(basename "$CFG" .yaml)
MODE=baseline
RUN_PREFIX="${CONFIG_STEM}_${MODE}_seed${SEED}"

CKPT_DIR="${RESULTS_DIR}/checkpoints"
ROLLOUT_DIR="${RESULTS_DIR}/rollouts/trajectory_${RUN_PREFIX}"
ANALYSIS_DIR="${RESULTS_DIR}/analysis/trajectory_${RUN_PREFIX}"

mkdir -p "$ROLLOUT_DIR" "$ANALYSIS_DIR"

for EP in $(seq 200 200 4000); do
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
done

echo "Done. Aggregate with:"
echo "  python scripts/aggregate_m1.py \\"
echo "    --analysis-dir ${ANALYSIS_DIR} \\"
echo "    --training-dir ${RESULTS_DIR} \\"
echo "    --output ${RESULTS_DIR}/aggregated_${RUN_PREFIX}.csv"
