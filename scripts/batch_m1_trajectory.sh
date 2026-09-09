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
#   M1_SCRATCH_ROOT  override scratch parent (must be under /mnt unless
#                    M1_ALLOW_NON_MNT_SCRATCH=1)
#   M1_ALLOW_NON_MNT_SCRATCH=1  allow non-/mnt scratch paths (off by default)
#   M1_POSTPROCESS_MODE baseline (default), karma, or broken; must match the
#                    training mode used to name checkpoints.
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
MODE="${M1_POSTPROCESS_MODE:-baseline}"
case "$MODE" in
  baseline|karma|broken) ;;
  *)
    echo "[batch] invalid M1_POSTPROCESS_MODE=$MODE (expected baseline, karma, or broken)" >&2
    exit 2
    ;;
esac
RUN_PREFIX="${CONFIG_STEM}_${MODE}_seed${SEED}"

CKPT_DIR="${RESULTS_DIR}/checkpoints"
ANALYSIS_DIR="${RESULTS_DIR}/analysis/trajectory_${RUN_PREFIX}"

# Optional lightweight W&B logging for long postprocess jobs.
# Enable by default when config logging.use_wandb is true, override with:
#   POSTPROCESS_USE_WANDB=0|1
read -r CFG_USE_WANDB CFG_WANDB_PROJECT < <(
  python3 - <<'PY' "$CFG"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
log = cfg.get("logging", {})
print(int(bool(log.get("use_wandb", False))), log.get("project_name", "karma-m1-empathy-gap"), sep="\t")
PY
)

POSTPROCESS_USE_WANDB="${POSTPROCESS_USE_WANDB:-$CFG_USE_WANDB}"
WB_ENABLED=0
if [[ "${POSTPROCESS_USE_WANDB}" == "1" ]]; then
  WB_ENABLED=1
fi

WB_PROJECT="${POSTPROCESS_WANDB_PROJECT:-$CFG_WANDB_PROJECT}"
WB_ENTITY="${POSTPROCESS_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
WB_RUN_NAME="${POSTPROCESS_WANDB_RUN_NAME:-${RUN_PREFIX}_postprocess}"
WB_RUN_ID="${POSTPROCESS_WANDB_RUN_ID:-post_${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
WB_RUN_ID="$(echo "$WB_RUN_ID" | tr -cs '[:alnum:]_-' '_')"
WB_MODE="${WANDB_MODE:-online}"

if [[ "$WB_ENABLED" == "1" ]]; then
  export WANDB_MODE="$WB_MODE"
  export WB_PROJECT WB_ENTITY WB_RUN_NAME WB_RUN_ID CFG RESULTS_DIR SEED
  if ! python3 - <<'PY' >/dev/null 2>&1
import wandb
PY
  then
    echo "[wandb-post] wandb SDK unavailable; continuing without postprocess logging"
    WB_ENABLED=0
  fi
fi

WB_PIPE=""
WB_LOGGER_PID=""
WB_PIPE_WRITER_OPEN=0

wb_start_logger() {
  if [[ "$WB_ENABLED" != "1" ]]; then
    return
  fi
  WB_PIPE="$(mktemp -u /tmp/wb_post_pipe.XXXXXX)"
  mkfifo "$WB_PIPE"
  export WB_PIPE
  python3 -u - "$WB_PIPE" <<'PY' &
import os
import sys
import time

pipe_path = sys.argv[1]
import wandb

kwargs = dict(
    project=os.environ["WB_PROJECT"],
    id=os.environ["WB_RUN_ID"],
    resume="allow",
    name=os.environ["WB_RUN_NAME"],
    job_type="m1-postprocess",
)
entity = os.environ.get("WB_ENTITY", "").strip()
if entity:
    kwargs["entity"] = entity
wandb.init(**kwargs)

with open(pipe_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line == "__END__":
            break
        parts = line.split("\t")
        if len(parts) < 18:
            continue
        (
            event,
            status,
            ep,
            message,
            rollout_path,
            analysis_path,
            total,
            done,
            skipped,
            processed,
            left,
            progress_pct,
            elapsed_sec,
            elapsed_hms,
            eta_sec,
            eta_hms,
            phase,
            checkpoint_index,
        ) = parts[:18]
        payload = {
            "post/event": event,
            "post/status": status,
            "post/message": message,
            "post/timestamp": time.time(),
            "post/config": os.environ.get("CFG", ""),
            "post/results_dir": os.environ.get("RESULTS_DIR", ""),
            "post/seed": int(os.environ["SEED"]),
            "post/checkpoints_total": int(total),
            "post/checkpoints_done": int(done),
            "post/checkpoints_skipped": int(skipped),
            "post/checkpoints_processed": int(processed),
            "post/checkpoints_left": int(left),
            "post/progress_pct": float(progress_pct),
            "post/elapsed_sec": int(elapsed_sec),
            "post/elapsed_hms": elapsed_hms,
            "post/eta_sec": int(eta_sec),
            "post/eta_hms": eta_hms,
            "post/phase": phase,
            "post/checkpoint_index": int(checkpoint_index),
        }
        if ep.strip():
            try:
                payload["post/episode"] = int(ep)
            except ValueError:
                pass
        if rollout_path.strip():
            payload["post/rollout_path"] = rollout_path
        if analysis_path.strip():
            payload["post/analysis_path"] = analysis_path
        wandb.log(payload)

wandb.finish()
PY
  WB_LOGGER_PID=$!
  sleep 0.2
  if ! kill -0 "$WB_LOGGER_PID" 2>/dev/null; then
    echo "[wandb-post] logger failed to start; continuing without postprocess logging"
    WB_ENABLED=0
    rm -f "$WB_PIPE" 2>/dev/null || true
    WB_PIPE=""
    WB_LOGGER_PID=""
    WB_PIPE_WRITER_OPEN=0
    return
  fi
  exec 3>"$WB_PIPE"
  WB_PIPE_WRITER_OPEN=1
}

wb_emit() {
  local event="${1:-}"
  local status="${2:-}"
  local ep="${3:-}"
  local message="${4:-}"
  local rollout_path="${5:-}"
  local analysis_path="${6:-}"
  local total="${7:-0}"
  local done="${8:-0}"
  local skipped="${9:-0}"
  local processed="${10:-0}"
  local left="${11:-0}"
  local progress_pct="${12:-0}"
  local elapsed_sec="${13:-0}"
  local elapsed_hms="${14:-00:00:00}"
  local eta_sec="${15:-0}"
  local eta_hms="${16:-00:00:00}"
  local phase="${17:-unknown}"
  local checkpoint_index="${18:-0}"
  if [[ "$WB_ENABLED" != "1" ]]; then
    return
  fi
  event="${event//$'\t'/ }"; event="${event//$'\n'/ }"
  status="${status//$'\t'/ }"; status="${status//$'\n'/ }"
  ep="${ep//$'\t'/ }"; ep="${ep//$'\n'/ }"
  message="${message//$'\t'/ }"; message="${message//$'\n'/ }"
  rollout_path="${rollout_path//$'\t'/ }"; rollout_path="${rollout_path//$'\n'/ }"
  analysis_path="${analysis_path//$'\t'/ }"; analysis_path="${analysis_path//$'\n'/ }"
  elapsed_hms="${elapsed_hms//$'\t'/ }"; elapsed_hms="${elapsed_hms//$'\n'/ }"
  eta_hms="${eta_hms//$'\t'/ }"; eta_hms="${eta_hms//$'\n'/ }"
  phase="${phase//$'\t'/ }"; phase="${phase//$'\n'/ }"
  if [[ "$WB_PIPE_WRITER_OPEN" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$event" "$status" "$ep" "$message" "$rollout_path" "$analysis_path" \
      "$total" "$done" "$skipped" "$processed" "$left" "$progress_pct" \
      "$elapsed_sec" "$elapsed_hms" "$eta_sec" "$eta_hms" "$phase" "$checkpoint_index" >&3 || true
  fi
}

wb_stop_logger() {
  if [[ "$WB_ENABLED" != "1" ]]; then
    return
  fi
  if [[ "$WB_PIPE_WRITER_OPEN" == "1" ]]; then
    printf '__END__\n' >&3 || true
    exec 3>&- || true
    WB_PIPE_WRITER_OPEN=0
  fi
  if [[ -n "$WB_LOGGER_PID" ]]; then
    wait "$WB_LOGGER_PID" 2>/dev/null || true
  fi
  if [[ -n "$WB_PIPE" ]]; then
    rm -f "$WB_PIPE" 2>/dev/null || true
  fi
  WB_PIPE=""
  WB_LOGGER_PID=""
  WB_PIPE_WRITER_OPEN=0
}

CURRENT_EP=""
CURRENT_ROLLOUT=""
CURRENT_ANALYSIS=""

fmt_hms() {
  local sec="${1:-0}"
  if [[ "$sec" -lt 0 ]]; then sec=0; fi
  local h=$((sec / 3600))
  local m=$(((sec % 3600) / 60))
  local s=$((sec % 60))
  printf '%02d:%02d:%02d' "$h" "$m" "$s"
}

emit_progress() {
  local event="${1:-}"
  local status="${2:-}"
  local ep="${3:-}"
  local message="${4:-}"
  local rollout_path="${5:-}"
  local analysis_path="${6:-}"
  local phase="${7:-unknown}"
  local processed=$((DONE_COUNT + SKIP_COUNT))
  local left=$((TOTAL_PLANNED - processed))
  if [[ "$left" -lt 0 ]]; then left=0; fi
  local elapsed_sec=$(( $(date +%s) - RUN_START_TS ))
  if [[ "$elapsed_sec" -lt 0 ]]; then elapsed_sec=0; fi
  local avg_sec=0
  local eta_sec=0
  local eta_hms="--:--:--"
  if [[ "$processed" -gt 0 ]]; then
    avg_sec=$((elapsed_sec / processed))
    eta_sec=$((avg_sec * left))
    eta_hms="$(fmt_hms "$eta_sec")"
  fi
  local progress_pct
  progress_pct="$(awk -v p="$processed" -v t="$TOTAL_PLANNED" 'BEGIN { if (t>0) printf "%.2f", (100.0*p)/t; else printf "0.00" }')"
  local checkpoint_index="$processed"
  if [[ -n "${ep}" ]]; then
    checkpoint_index=$((processed + 1))
  fi
  wb_emit "$event" "$status" "$ep" "$message" "$rollout_path" "$analysis_path" \
    "$TOTAL_PLANNED" "$DONE_COUNT" "$SKIP_COUNT" "$processed" "$left" "$progress_pct" \
    "$elapsed_sec" "$(fmt_hms "$elapsed_sec")" "$eta_sec" "$eta_hms" "$phase" "$checkpoint_index"
}

run_with_heartbeat() {
  local phase="${1:?phase}"
  shift
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    emit_progress "checkpoint_${phase}_heartbeat" "running" "$CURRENT_EP" "phase=${phase}" "$CURRENT_ROLLOUT" "$CURRENT_ANALYSIS" "$phase"
  done
  wait "$pid"
}

on_err() {
  local line_no="$1"
  emit_progress "run_error" "failed" "${CURRENT_EP}" "line=${line_no}" "${CURRENT_ROLLOUT}" "${CURRENT_ANALYSIS}" "error"
}
trap 'on_err $LINENO' ERR
trap 'wb_stop_logger' EXIT

wb_start_logger

# Parquets must be written to scratch storage (default /mnt). No fallback to
# results/ on root disk is allowed.
resolve_scratch_parent() {
  local root="${M1_SCRATCH_ROOT:-/mnt/karma_m1_scratch}"

  if [[ "${root}" != /mnt/* ]] && [[ "${M1_ALLOW_NON_MNT_SCRATCH:-0}" != "1" ]]; then
    echo "[error] scratch root must be under /mnt by default: ${root}" >&2
    echo "        Set M1_SCRATCH_ROOT=/mnt/... or M1_ALLOW_NON_MNT_SCRATCH=1 to override." >&2
    exit 2
  fi

  mkdir -p "${root}" 2>/dev/null || true
  if [[ ! -d "${root}" || ! -w "${root}" ]]; then
    if [[ "${root}" == /mnt/* ]]; then
      sudo -n mkdir -p "${root}" 2>/dev/null || true
      sudo -n chown "$USER:$USER" "${root}" 2>/dev/null || true
    fi
  fi
  if [[ ! -d "${root}" || ! -w "${root}" ]]; then
    echo "[error] scratch root is not writable: ${root}" >&2
    echo "        Refusing to write rollout parquets outside scratch storage." >&2
    exit 2
  fi
  echo "${root}"
}

SCRATCH_PARENT="$(resolve_scratch_parent)"
ROLLOUT_DIR="${SCRATCH_PARENT}/${RUN_PREFIX}"
mkdir -p "$ROLLOUT_DIR"
echo "[batch] temp parquets -> ${ROLLOUT_DIR} (scratch; cleared after each analyze — not for long-term storage)"

mkdir -p "$ANALYSIS_DIR"

if [[ -n "$SINGLE_EP" ]]; then
  EP_SEQ=("$SINGLE_EP")
else
  EP_SEQ=($(seq 200 200 4000))
fi

TOTAL_PLANNED="${#EP_SEQ[@]}"
DONE_COUNT=0
SKIP_COUNT=0
RUN_START_TS="$(date +%s)"
emit_progress "run_start" "running" "" "planned_checkpoints=${TOTAL_PLANNED}" "" "" "run"

for EP in "${EP_SEQ[@]}"; do
  CURRENT_EP="$EP"
  CKPT="${CKPT_DIR}/${RUN_PREFIX}_ep${EP}.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] missing checkpoint: $CKPT"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    emit_progress "checkpoint_skip_missing_ckpt" "skipped" "$EP" "missing checkpoint" "" "" "skip_missing_ckpt"
    continue
  fi
  ROLLOUT="${ROLLOUT_DIR}/${RUN_PREFIX}_ep${EP}.parquet"
  ANALYSIS="${ANALYSIS_DIR}/${RUN_PREFIX}_ep${EP}.json"
  CURRENT_ROLLOUT="$ROLLOUT"
  CURRENT_ANALYSIS="$ANALYSIS"
  if [[ -f "$ANALYSIS" ]]; then
    echo "[skip] already analyzed: $ANALYSIS"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    emit_progress "checkpoint_skip_existing" "skipped" "$EP" "analysis already exists" "$ROLLOUT" "$ANALYSIS" "skip_existing"
    continue
  fi
  echo "=== ep $EP ==="
  emit_progress "checkpoint_start" "running" "$EP" "" "$ROLLOUT" "$ANALYSIS" "start"
  run_with_heartbeat rollout python scripts/rollout_from_checkpoint.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --episodes "$EVAL_EPISODES" \
    --output "$ROLLOUT"
  run_with_heartbeat analyze python scripts/analyze_checkpoint.py \
    --rollout "$ROLLOUT" \
    --checkpoint "$CKPT" \
    --config "$CFG" \
    --output "$ANALYSIS"
  rm -f "$ROLLOUT"
  echo "[scratch] removed temp parquet after successful analyze: ${ROLLOUT}"
  DONE_COUNT=$((DONE_COUNT + 1))
  emit_progress "checkpoint_done" "completed" "$EP" "parquet_removed_after_analyze" "$ROLLOUT" "$ANALYSIS" "done"
done

CURRENT_EP=""
CURRENT_ROLLOUT=""
CURRENT_ANALYSIS=""
emit_progress "run_complete" "completed" "" "done=${DONE_COUNT} skipped=${SKIP_COUNT}" "${ROLLOUT_DIR}" "${ANALYSIS_DIR}" "run"

echo "Done. Aggregate with:"
echo "  python scripts/aggregate_m1.py \\"
echo "    --analysis-dir ${ANALYSIS_DIR} \\"
echo "    --training-dir ${RESULTS_DIR} \\"
echo "    --output ${RESULTS_DIR}/aggregated_${RUN_PREFIX}.csv"
