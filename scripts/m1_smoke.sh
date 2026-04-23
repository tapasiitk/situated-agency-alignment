#!/usr/bin/env bash
# End-to-end M1 smoke: 40-episode train, rollout the last checkpoint,
# run all four measurements on it.
#
# Usage: bash scripts/m1_smoke.sh [seed]
#
# Exits non-zero if any step fails, so it can gate `git push`.
set -euo pipefail

SEED="${1:-1}"
CONFIG="configs/m1_smoke40.yaml"
RESULTS_DIR="results/m1_smoke40"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
ROLLOUT_DIR="${RESULTS_DIR}/rollouts"
ANALYSIS_DIR="${RESULTS_DIR}/analysis"

RUN_STEM="m1_smoke40_baseline_seed${SEED}"
CKPT_EP=40   # with checkpoint_interval=20 and 40 episodes, last ckpt is ep40

mkdir -p "${CKPT_DIR}" "${ROLLOUT_DIR}" "${ANALYSIS_DIR}"

echo "=== [1/3] train_karma.py --config ${CONFIG} --mode baseline --seed ${SEED} ==="
WANDB_MODE=disabled python train_karma.py \
    --config "${CONFIG}" \
    --mode baseline \
    --seed "${SEED}"

CKPT_PATH="${CKPT_DIR}/${RUN_STEM}_ep${CKPT_EP}.pt"
echo "=== checkpoint expected at ${CKPT_PATH} ==="
ls -lh "${CKPT_PATH}"

ROLLOUT_PATH="${ROLLOUT_DIR}/${RUN_STEM}_ep${CKPT_EP}.parquet"
echo "=== [2/3] rollout ==="
python scripts/rollout_from_checkpoint.py \
    --config "${CONFIG}" \
    --checkpoint "${CKPT_PATH}" \
    --episodes 4 \
    --output "${ROLLOUT_PATH}"

ANALYSIS_PATH="${ANALYSIS_DIR}/${RUN_STEM}_ep${CKPT_EP}.json"
# Prefer .jsonl fallback if pandas/parquet not available.
if [[ ! -f "${ROLLOUT_PATH}" ]] && [[ -f "${ROLLOUT_PATH%.parquet}.jsonl" ]]; then
    ROLLOUT_PATH="${ROLLOUT_PATH%.parquet}.jsonl"
fi

echo "=== [3/3] analyze ==="
python scripts/analyze_checkpoint.py \
    --rollout "${ROLLOUT_PATH}" \
    --checkpoint "${CKPT_PATH}" \
    --config "${CONFIG}" \
    --output "${ANALYSIS_PATH}"

echo "=== SMOKE PASS ==="
echo "Analysis JSON: ${ANALYSIS_PATH}"
