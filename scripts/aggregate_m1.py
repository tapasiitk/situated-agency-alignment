"""Aggregate M1 per-checkpoint analysis JSONs into a long-format table.

Walks the analysis directory, joins each per-checkpoint JSON with the
training-time CSV rows at matched episodes, and writes one CSV with
columns: env, scarcity, seed, episode, plus every scalar measurement.

Usage:
    python scripts/aggregate_m1.py \
        --analysis-dir results/m1 \
        --training-dir results/m1 \
        --output       results/m1/aggregated.csv

The convention used to infer (env, scarcity, seed) from filenames:
    <config_stem>_<mode>_seed<N>_ep<E>.json
e.g.  m1_env_A_sc030_baseline_seed3_ep1200.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


NAME_RE = re.compile(
    r"^(?P<config>.+)_(?P<mode>baseline|broken|karma)_seed(?P<seed>\d+)_ep(?P<ep>\d+)\.json$"
)


def flatten(prefix: str, obj: Any, out: Dict[str, Any]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        out[prefix] = obj
    else:
        out[prefix] = json.dumps(obj)


def parse_config_name(config_stem: str) -> Dict[str, Optional[str]]:
    # e.g. m1_env_A_sc030 -> env=A, scarcity=0.30
    m = re.search(r"env_(?P<env>[AB])_sc(?P<sc>\d{3})", config_stem)
    if m:
        env = m.group("env")
        sc = int(m.group("sc")) / 100.0
        return {"env": env, "scarcity": sc}
    return {"env": None, "scarcity": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=str, required=True)
    parser.add_argument("--training-dir", type=str, default=None,
                        help="Directory containing <run>.csv training metrics.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for json_path in sorted(analysis_dir.rglob("*.json")):
        m = NAME_RE.match(json_path.name)
        if not m:
            continue
        config_stem = m.group("config")
        mode = m.group("mode")
        seed = int(m.group("seed"))
        episode = int(m.group("ep"))

        with json_path.open("r") as f:
            summary = json.load(f)

        flat: Dict[str, Any] = {}
        flatten("", summary, flat)

        row: Dict[str, Any] = {
            "config": config_stem,
            "mode": mode,
            "seed": seed,
            "episode": episode,
            **parse_config_name(config_stem),
        }
        row.update(flat)
        rows.append(row)

    if not rows:
        print(f"[aggregate] no analysis JSONs found under {analysis_dir}")
        return

    df = pd.DataFrame.from_records(rows)

    # Optional: join with training-time CSV.
    if args.training_dir:
        training_dir = Path(args.training_dir)
        training_frames: List[pd.DataFrame] = []
        for csv_path in training_dir.rglob("*.csv"):
            stem = csv_path.stem
            m = re.match(r"^(?P<config>.+)_(?P<mode>baseline|broken|karma)_seed(?P<seed>\d+)$", stem)
            if not m:
                continue
            tdf = pd.read_csv(csv_path)
            tdf["config"] = m.group("config")
            tdf["mode"] = m.group("mode")
            tdf["seed"] = int(m.group("seed"))
            tdf["episode"] = tdf["Episode"].astype(int)
            training_frames.append(tdf)
        if training_frames:
            tall = pd.concat(training_frames, ignore_index=True)
            df = df.merge(tall, how="left", on=["config", "mode", "seed", "episode"])

    df = df.sort_values(["env", "scarcity", "seed", "episode"], kind="mergesort")
    df.to_csv(out_path, index=False)
    print(f"[aggregate] wrote {out_path} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
