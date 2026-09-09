#!/usr/bin/env python3
"""Grid-search sklearn probe settings (solver × max_iter) before freezing prereg.

Calls the same measure_linear_probes() as analyze_checkpoint.py on each rollout
parquet (**probes only** — no CKA/RSA/M4) so sweeps stay cheap.

Selection rule (programmatic, see --prefer-solver):
  1) Minimize total ConvergenceWarning count summed across rollouts.
  2) Minimize max_iter (faster fits).
  3) Prefer solver order from --prefer-solver (default: lbfgs before saga).

If a cell raises (e.g. invalid solver), combinations with any exception rank last (n_exceptions > 0).

Usage:
  python scripts/pilot_probe_hyperparam_sweep.py \\
    --rollout results/.../ep2000.parquet \\
    --rollout results/.../ep4000.parquet \\
    --max-iters 2000,5000,10000 \\
    --solvers lbfgs,saga \\
    --out-dir results/pilot_probe_sweep

Then lock prereg / defaults to recommended.probe_solver and recommended.probe_max_iter.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.exceptions import ConvergenceWarning

from scripts.analyze_checkpoint import load_rollout, measure_linear_probes, stack_embeddings


@dataclass
class SweepCell:
    rollout: str
    solver: str
    max_iter: int
    n_convergence_warnings: int
    had_exception: bool
    exception: Optional[str]
    probe_5way_auroc_mean: Optional[float]
    probe_agg_vs_vic_auroc: Optional[float]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _solver_rank(solver: str, prefer_order: Sequence[str]) -> int:
    try:
        return prefer_order.index(solver.lower())
    except ValueError:
        return len(prefer_order)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep for probe max_iter / solver")
    parser.add_argument("--rollout", action="append", required=True, dest="rollouts")
    parser.add_argument(
        "--max-iters",
        type=str,
        default="2000,5000,10000",
        help="Comma-separated max_iter values",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="lbfgs,saga",
        help="Comma-separated solvers to try (sklearn LogisticRegression)",
    )
    parser.add_argument(
        "--prefer-solver",
        type=str,
        default="lbfgs,saga",
        help="Tie-break order (earlier = better if warnings and max_iter tie)",
    )
    parser.add_argument("--multi-class", type=str, default="ovr")
    parser.add_argument("--out-dir", type=Path, default=Path("results/pilot_probe_sweep"))
    args = parser.parse_args()

    max_iters = _parse_csv_ints(args.max_iters)
    solvers = _parse_csv_strs(args.solvers)
    prefer_order = tuple(s.lower() for s in _parse_csv_strs(args.prefer_solver))

    rollouts = [Path(p) for p in args.rollouts]
    for p in rollouts:
        if not p.is_file():
            raise SystemExit(f"Missing rollout: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells: List[SweepCell] = []
    cached: List[Tuple[Path, Any, Any]] = []
    for p in rollouts:
        df = load_rollout(p)
        emb = stack_embeddings(df)
        cached.append((p, df, emb))

    for rollout_path, df, emb in cached:
        for solver in solvers:
            for max_iter in max_iters:
                n_w = 0
                exc: Optional[str] = None
                results: Dict[str, Any] = {}
                try:
                    with warnings.catch_warnings(record=True) as wrec:
                        warnings.simplefilter("always", ConvergenceWarning)
                        results = measure_linear_probes(
                            df,
                            emb,
                            max_iter=max_iter,
                            solver=solver,
                            multi_class=args.multi_class,
                        )
                    n_w = sum(
                        1 for w in wrec if issubclass(w.category, ConvergenceWarning)
                    )
                except Exception as e:
                    exc = repr(e)
                    n_w = 0
                    results = {}

                cells.append(
                    SweepCell(
                        rollout=str(rollout_path),
                        solver=solver,
                        max_iter=max_iter,
                        n_convergence_warnings=n_w,
                        had_exception=exc is not None,
                        exception=exc,
                        probe_5way_auroc_mean=results.get("probe_5way_auroc_mean"),
                        probe_agg_vs_vic_auroc=results.get("probe_agg_vs_vic_auroc"),
                    )
                )

    # Aggregate by (solver, max_iter)
    combos: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for c in cells:
        key = (c.solver, c.max_iter)
        if key not in combos:
            combos[key] = {
                "solver": c.solver,
                "max_iter": c.max_iter,
                "total_convergence_warnings": 0,
                "n_exceptions": 0,
                "rollouts": [],
            }
        combos[key]["total_convergence_warnings"] += c.n_convergence_warnings
        if c.had_exception:
            combos[key]["n_exceptions"] += 1
        combos[key]["rollouts"].append(asdict(c))

    combo_list = list(combos.values())

    def sort_key(d: Dict[str, Any]) -> Tuple[int, int, int, int]:
        return (
            d["n_exceptions"] > 0,
            d["total_convergence_warnings"],
            d["max_iter"],
            _solver_rank(str(d["solver"]), prefer_order),
        )

    combo_list.sort(key=sort_key)
    best = combo_list[0] if combo_list else None

    out = {
        "ranking_policy": (
            "Sort keys: (1) zero exceptions first (2) minimize total ConvergenceWarning "
            "count across rollouts (3) minimize max_iter (4) --prefer-solver order"
        ),
        "prefer_solver_order": list(prefer_order),
        "multi_class": args.multi_class,
        "rollouts": [str(p) for p in rollouts],
        "combinations_evaluated": combo_list,
        "recommended": (
            {
                "probe_solver": best["solver"],
                "probe_max_iter": best["max_iter"],
                "total_convergence_warnings": best["total_convergence_warnings"],
                "n_exceptions": best["n_exceptions"],
                "analyze_checkpoint_flags": (
                    f"--probe-solver {best['solver']} --probe-max-iter {best['max_iter']}"
                ),
            }
            if best
            else None
        ),
        "raw_cells": [asdict(c) for c in cells],
    }

    out_path = args.out_dir / "pilot_probe_hyperparam_sweep.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=float)

    print(json.dumps(out["recommended"], indent=2))
    print(f"[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
