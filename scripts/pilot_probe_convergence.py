#!/usr/bin/env python3
"""Repeatable pilot: run full analyze_checkpoint on 1+ rollouts; tally probe convergence warnings.

Uses **fixed** (--probe-max-iter, --probe-solver). For a **grid search** and programmatic
pick before prereg, use scripts/pilot_probe_hyperparam_sweep.py (probes only, faster).

Same analysis path as analyze_checkpoint.py (imports run_all).

Usage (probe + CKA/RSA only; skip slow gradient transfer):
  python scripts/pilot_probe_convergence.py \\
    --rollout path/ep2000.parquet \\
    --rollout path/ep4000.parquet \\
    --probe-max-iter 5000 \\
    --probe-solver lbfgs

Full measurements (including M4 if checkpoint+config provided):
  python scripts/pilot_probe_convergence.py \\
    --rollout path/ep2000.parquet \\
    --checkpoint path/to.ckpt.pt \\
    --config configs/m1_env_A_sc030.yaml \\
    --device cuda

Writes:
  --out-dir/pilot_summary.json  (one row per rollout + warning counts + key probe metrics)
  --out-dir/analysis_<stem>.json per rollout (same shape as analyze_checkpoint output)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.exceptions import ConvergenceWarning

from scripts.analyze_checkpoint import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot sklearn probe convergence on rollout parquet(s)")
    parser.add_argument(
        "--rollout",
        action="append",
        required=True,
        dest="rollouts",
        help="Path to .parquet (repeat for multiple checkpoints)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--probe-max-iter", type=int, default=2000)
    parser.add_argument("--probe-solver", type=str, default="lbfgs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/pilot_probe_convergence"),
    )
    args = parser.parse_args()

    cp = Path(args.checkpoint) if args.checkpoint else None
    cf = Path(args.config) if args.config else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for rp in args.rollouts:
        rollout_path = Path(rp)
        if not rollout_path.is_file():
            raise SystemExit(f"Missing rollout: {rollout_path}")

        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always", ConvergenceWarning)
            summary = run_all(
                rollout_path,
                cp,
                cf,
                args.device,
                probe_max_iter=args.probe_max_iter,
                probe_solver=args.probe_solver,
            )

        conv = [w for w in wrec if issubclass(w.category, ConvergenceWarning)]
        stem = rollout_path.stem
        out_json = args.out_dir / f"analysis_{stem}.json"
        with out_json.open("w") as f:
            json.dump(summary, f, indent=2, default=float)

        m1 = summary.get("measurement_1_probes") or {}
        rows.append(
            {
                "rollout": str(rollout_path),
                "analysis_json": str(out_json.resolve()),
                "probe_max_iter": args.probe_max_iter,
                "probe_solver": args.probe_solver,
                "n_convergence_warnings": len(conv),
                "convergence_messages": [str(w.message) for w in conv],
                "probe_5way_auroc_mean": m1.get("probe_5way_auroc_mean"),
                "probe_agg_vs_vic_auroc": m1.get("probe_agg_vs_vic_auroc"),
            }
        )
        print(
            f"[pilot] {rollout_path.name}: ConvergenceWarning x{len(conv)} | "
            f"5-way AUROC={m1.get('probe_5way_auroc_mean')!s} "
            f"agg-vic AUROC={m1.get('probe_agg_vs_vic_auroc')!s}"
        )

    summary_path = args.out_dir / "pilot_summary.json"
    bundle = {
        "settings": {
            "probe_max_iter": args.probe_max_iter,
            "probe_solver": args.probe_solver,
            "checkpoint": str(cp) if cp else None,
            "config": str(cf) if cf else None,
        },
        "runs": rows,
    }
    with summary_path.open("w") as f:
        json.dump(bundle, f, indent=2, default=float)
    print(f"[pilot] wrote {summary_path}")


if __name__ == "__main__":
    main()
