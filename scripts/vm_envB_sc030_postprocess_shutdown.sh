#!/usr/bin/env bash
# Run after a finished 4k Env B (m1_env_B_sc030) training job on the VM:
# rollout+analyze all checkpoints (seeds 42 and 123), aggregate, plot, then
# auto-shutdown the VM ~1 min after the last plot process exits.
#
# Usage (on VM, from repo root):
#   nohup bash scripts/vm_envB_sc030_postprocess_shutdown.sh \
#     > run_logs/envB_sc030_postprocess_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# Requires: passwordless sudo for shutdown (see scripts/auto_shutdown_watcher.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

CFG="configs/m1_env_B_sc030.yaml"
RES="results/m1_env_B_sc030"
export M1_SCRATCH_ROOT="${M1_SCRATCH_ROOT:-/mnt/karma_m1_scratch}"

mkdir -p run_logs "${RES}/plots/seed42" "${RES}/plots/seed123"

echo "[envB-post] $(date) starting batch trajectory seed 42"
bash scripts/batch_m1_trajectory.sh "$CFG" "$RES" 42 20

echo "[envB-post] $(date) starting batch trajectory seed 123"
bash scripts/batch_m1_trajectory.sh "$CFG" "$RES" 123 20

echo "[envB-post] $(date) aggregate seed 42"
python scripts/aggregate_m1.py \
  --analysis-dir "${RES}/analysis/trajectory_m1_env_B_sc030_baseline_seed42" \
  --training-dir "$RES" \
  --output "${RES}/aggregated_m1_env_B_sc030_baseline_seed42.csv"

echo "[envB-post] $(date) aggregate seed 123"
python scripts/aggregate_m1.py \
  --analysis-dir "${RES}/analysis/trajectory_m1_env_B_sc030_baseline_seed123" \
  --training-dir "$RES" \
  --output "${RES}/aggregated_m1_env_B_sc030_baseline_seed123.csv"

echo "[envB-post] $(date) plot seed 42"
python scripts/plot_m1_trajectory.py \
  --csv "${RES}/aggregated_m1_env_B_sc030_baseline_seed42.csv" \
  --out "${RES}/plots/seed42"

echo "[envB-post] $(date) plot seed 123 + auto_shutdown_watcher"
# Watcher must start after the plot process exists (pgrep finds python plot_m1).
python scripts/plot_m1_trajectory.py \
  --csv "${RES}/aggregated_m1_env_B_sc030_baseline_seed123.csv" \
  --out "${RES}/plots/seed123" &
PLOT_PID=$!
sleep 4
if ! kill -0 "$PLOT_PID" 2>/dev/null; then
  echo "[envB-post] ERROR plot seed123 exited before watcher could attach" >&2
  exit 1
fi
nohup bash scripts/auto_shutdown_watcher.sh '^python.*plot_m1_trajectory\.py' envB_sc030_final_plot \
  >> run_logs/auto_shutdown_envB_sc030_postprocess.log 2>&1 &
wait "$PLOT_PID"

echo "[envB-post] $(date) plot seed123 finished; watcher will shutdown VM if configured"
