#!/usr/bin/env bash
# End-to-end M1 smoke: 40-episode train, rollout the last checkpoint,
# run all core M1 measurements on it.
#
# Usage: bash scripts/m1_smoke.sh [seed]
#
# Exits non-zero if any step fails, so it can gate `git push`.
set -euo pipefail

SEED="${1:-1}"

# Prefer modern M1 smoke config; fall back to older phase-a branch config.
if [[ -f "configs/m1_smoke40.yaml" ]]; then
    CONFIG="configs/m1_smoke40.yaml"
elif [[ -f "configs/canonical_baseline_stability_m1_smoke.yaml" ]]; then
    CONFIG="configs/canonical_baseline_stability_m1_smoke.yaml"
else
    echo "Could not find a smoke config. Expected one of:"
    echo "  - configs/m1_smoke40.yaml"
    echo "  - configs/canonical_baseline_stability_m1_smoke.yaml"
    exit 1
fi

CONFIG_STEM=$(basename "${CONFIG}" .yaml)
RESULTS_DIR=$(python - <<'PY' "${CONFIG}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg.get("logging", {}).get("local_results_dir") or f"results/{sys.argv[1].split('/')[-1].replace('.yaml','')}")
PY
)
CKPT_DIR="${RESULTS_DIR}/checkpoints"
ROLLOUT_DIR="${RESULTS_DIR}/rollouts"
ANALYSIS_DIR="${RESULTS_DIR}/analysis"

RUN_STEM="${CONFIG_STEM}_baseline_seed${SEED}"
CKPT_EP=40   # smoke configs are 40 episodes

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
