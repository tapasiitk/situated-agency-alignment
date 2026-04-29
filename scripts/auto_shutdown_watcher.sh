#!/usr/bin/env bash
# Reusable VM auto-shutdown watcher.
# Waits for a process matching the pgrep -f pattern to exit, then runs
# `sudo -n shutdown -h +1` so the VM stops automatically.
#
# Usage:
#   nohup bash scripts/auto_shutdown_watcher.sh "<pattern>" "<tag>" >/dev/null 2>&1 &
#   disown
#
# Example:
#   nohup bash scripts/auto_shutdown_watcher.sh \
#     "train_karma.py.*m1_env_B_sc030.yaml.*--seed 42" \
#     "envB_sc030_s42" >/dev/null 2>&1 &
#   disown
#
# If you launched training inside tmux (or bash -c ... | tee), the command line of
# the tmux/bash parent ALSO contains the substring "python ... train_karma.py", so
# pgrep -f would match the wrong PID first. Anchor the regex to the real process:
#   '^python.*train_karma\.py.*<config_stem>\.yaml.*--seed <N>'
# Example:
#   nohup bash scripts/auto_shutdown_watcher.sh \
#     '^python.*train_karma\.py.*m1_env_B_sc030_sym\.yaml.*--seed 42' \
#     "envB_sym_s42" >/dev/null 2>&1 &
#
# Notes:
# - Requires passwordless sudo for shutdown. Verify once with:
#       sudo -n shutdown --help
#   If it prompts, configure once:
#       echo "$USER ALL=(ALL) NOPASSWD: /sbin/shutdown" | sudo tee /etc/sudoers.d/auto_shutdown
#       sudo chmod 440 /etc/sudoers.d/auto_shutdown
# - Cancel a pending shutdown with:
#       sudo shutdown -c

set -u

PATTERN="${1:-}"
TAG="${2:-run}"

if [ -z "$PATTERN" ]; then
  echo "usage: $0 \"<pgrep -f pattern>\" [tag]" >&2
  exit 2
fi

mkdir -p run_logs
LOG="run_logs/auto_shutdown_${TAG}.log"
exec >>"$LOG" 2>&1

echo "[watcher] start at $(date)  tag=$TAG  pattern=$PATTERN"

PID=$(pgrep -f "$PATTERN" | head -n 1)
echo "[watcher] initial PID=$PID"

if [ -z "$PID" ]; then
  echo "[watcher] no matching PID; exiting"
  exit 1
fi

while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done

echo "[watcher] PID $PID ended at $(date)"
sleep 60
echo "[watcher] calling sudo shutdown -h +1 at $(date)"
sudo -n shutdown -h +1 "auto-shutdown after $TAG"
echo "[watcher] shutdown command issued"
