#!/usr/bin/env bash
# Rollout + analyze (all checkpoints) → aggregate → plot for one or more training
# seeds, then schedule VM shutdown ~1 min after the *last* plot process exits
# (via scripts/auto_shutdown_watcher.sh).
#
# Usage (on VM, from repo root):
#   nohup bash scripts/vm_m1_postprocess_shutdown.sh \
#     configs/m1_env_B_sc030_sym.yaml results/m1_env_B_sc030_sym 42 \
#     > run_logs/m1_post_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Multi-seed example:
#   nohup bash scripts/vm_m1_postprocess_shutdown.sh \
#     configs/m1_env_B_sc030.yaml results/m1_env_B_sc030 42 123 \
#     > run_logs/....log 2>&1 &
#
# Requires: passwordless sudo for shutdown (see scripts/auto_shutdown_watcher.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <config.yaml> <results_dir> <seed> [seed ...]" >&2
  exit 2
fi

CFG="${1:?}"
RES="${2:?}"
shift 2
SEEDS=("$@")

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export M1_SCRATCH_ROOT="${M1_SCRATCH_ROOT:-/mnt/karma_m1_scratch}"
CONFIG_STEM=$(basename "$CFG" .yaml)
RUN_TAG="${CONFIG_STEM}_post"
mkdir -p run_logs

for SEED in "${SEEDS[@]}"; do
  echo "[m1-post] $(date) batch trajectory $CONFIG_STEM seed $SEED"
  bash scripts/batch_m1_trajectory.sh "$CFG" "$RES" "$SEED" 20

  ADIR="${RES}/analysis/trajectory_${CONFIG_STEM}_baseline_seed${SEED}"
  AGG="${RES}/aggregated_${CONFIG_STEM}_baseline_seed${SEED}.csv"
  echo "[m1-post] $(date) aggregate seed $SEED -> $AGG"
  python scripts/aggregate_m1.py \
    --analysis-dir "$ADIR" \
    --training-dir "$RES" \
    --output "$AGG"

  PDIR="${RES}/plots/seed${SEED}"
  mkdir -p "$PDIR"
 done

N="${#SEEDS[@]}"
for i in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$i]}"
  AGG="${RES}/aggregated_${CONFIG_STEM}_baseline_seed${SEED}.csv"
  PDIR="${RES}/plots/seed${SEED}"
  if [[ $i -eq $((N - 1)) ]]; then
    echo "[m1-post] $(date) plot seed $SEED (last) + auto_shutdown_watcher tag=$RUN_TAG"
    python scripts/plot_m1_trajectory.py --csv "$AGG" --out "$PDIR" &
    PLOT_PID=$!
    sleep 4
    if ! kill -0 "$PLOT_PID" 2>/dev/null; then
      echo "[m1-post] ERROR last plot exited before watcher could attach" >&2
      exit 1
    fi
    nohup bash scripts/auto_shutdown_watcher.sh '^python.*plot_m1_trajectory\.py' "$RUN_TAG" \
      >> run_logs/auto_shutdown_${RUN_TAG}.log 2>&1 &
    wait "$PLOT_PID"
    echo "[m1-post] $(date) last plot finished; watcher will shutdown VM if configured"
  else
    echo "[m1-post] $(date) plot seed $SEED"
    python scripts/plot_m1_trajectory.py --csv "$AGG" --out "$PDIR"
  fi
 done
