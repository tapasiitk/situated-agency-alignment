#!/usr/bin/env bash
# Standard dual-use Env B sc030 — two seeds (42, 123). Delegates to vm_m1_postprocess_shutdown.sh.
exec "$(cd "$(dirname "$0")" && pwd)/vm_m1_postprocess_shutdown.sh" \
  configs/m1_env_B_sc030.yaml results/m1_env_B_sc030 42 123
