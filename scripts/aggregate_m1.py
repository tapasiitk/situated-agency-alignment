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
import yaml


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


def parse_config_name(config_stem: str) -> Dict[str, Optional[float]]:
    # e.g. m1_env_A_sc030 -> env=A, legacy scarcity=0.30
    env_m = re.search(r"env_(?P<env>[A-Za-z])", config_stem)
    sc_m = re.search(r"_sc(?P<sc>\d{3})", config_stem)
    out: Dict[str, Optional[float]] = {
        "env": None,
        "legacy_sc_from_filename": None,
        "legacy_apple_density_scarcity": None,
        "apple_density": None,
        "regrowth_speed": None,
        "zap_timeout": None,
        "num_agents": None,
    }
    if env_m:
        out["env"] = env_m.group("env")
    if sc_m:
        sc = int(sc_m.group("sc")) / 100.0
        out["legacy_sc_from_filename"] = sc
        out["legacy_apple_density_scarcity"] = sc
        out["apple_density"] = sc
    rg_m = re.search(r"_rg(?P<rg>\d{3})", config_stem)
    zt_m = re.search(r"_zt(?P<zt>\d{2,3})", config_stem)
    n_m = re.search(r"_n(?P<n>\d+)", config_stem)
    ad_m = re.search(r"_ad(?P<ad>\d{3})", config_stem)
    if rg_m:
        out["regrowth_speed"] = int(rg_m.group("rg")) / 100.0
    if zt_m:
        out["zap_timeout"] = float(int(zt_m.group("zt")))
    if n_m:
        out["num_agents"] = float(int(n_m.group("n")))
    if ad_m:
        out["apple_density"] = int(ad_m.group("ad")) / 100.0
    return out


def parse_env_from_config_path(config_path: Optional[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "env": None,
        "apple_density": None,
        "regrowth_speed": None,
        "zap_timeout": None,
        "num_agents": None,
    }
    if not config_path:
        return out
    p = Path(config_path)
    if not p.exists():
        p = Path.cwd() / config_path
    if not p.exists():
        return out
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return out
    env_cfg = data.get("env", {}) if isinstance(data, dict) else {}
    out["env"] = str(config_path)
    if isinstance(env_cfg, dict):
        env_match = re.search(r"env_([A-Za-z])", p.stem)
        out["env"] = env_match.group(1) if env_match else None
        out["apple_density"] = env_cfg.get("apple_density")
        out["regrowth_speed"] = env_cfg.get("regrowth_speed")
        out["zap_timeout"] = env_cfg.get("zap_timeout")
        out["num_agents"] = env_cfg.get("num_agents")
    return out


def build_ecology_id(
    env: Optional[str],
    num_agents: Optional[float],
    apple_density: Optional[float],
    regrowth_speed: Optional[float],
    zap_timeout: Optional[float],
) -> Optional[str]:
    if None in (env, num_agents, apple_density, regrowth_speed, zap_timeout):
        return None
    return (
        f"env{str(env)}_n{int(num_agents)}_"
        f"ad{int(round(float(apple_density) * 100)):03d}_"
        f"rg{int(round(float(regrowth_speed) * 100)):03d}_"
        f"zt{int(round(float(zap_timeout))):02d}"
    )


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

        cfg_path = flat.get("config_path")
        file_meta = parse_config_name(config_stem)
        yaml_meta = parse_env_from_config_path(cfg_path if isinstance(cfg_path, str) else None)
        env = yaml_meta.get("env") if yaml_meta.get("env") is not None else file_meta.get("env")
        apple_density = (
            yaml_meta.get("apple_density")
            if yaml_meta.get("apple_density") is not None
            else file_meta.get("apple_density")
        )
        regrowth_speed = (
            yaml_meta.get("regrowth_speed")
            if yaml_meta.get("regrowth_speed") is not None
            else file_meta.get("regrowth_speed")
        )
        zap_timeout = (
            yaml_meta.get("zap_timeout")
            if yaml_meta.get("zap_timeout") is not None
            else file_meta.get("zap_timeout")
        )
        num_agents = (
            yaml_meta.get("num_agents")
            if yaml_meta.get("num_agents") is not None
            else file_meta.get("num_agents")
        )
        row: Dict[str, Any] = {
            "config": config_stem,
            "mode": mode,
            "seed": seed,
            "episode": episode,
        }
        row.update(flat)
        row["env"] = env
        row["apple_density"] = apple_density
        row["regrowth_speed"] = regrowth_speed
        row["zap_timeout"] = zap_timeout
        row["num_agents"] = num_agents
        row["legacy_sc_from_filename"] = file_meta.get("legacy_sc_from_filename")
        row["legacy_apple_density_scarcity"] = file_meta.get("legacy_apple_density_scarcity")
        row["ecology_id"] = build_ecology_id(env, num_agents, apple_density, regrowth_speed, zap_timeout)
        row["resource_pressure"] = (
            1.0 / float(regrowth_speed)
            if regrowth_speed is not None and float(regrowth_speed) != 0.0
            else None
        )
        # Backward-compatible alias retained for existing consumers.
        row["scarcity"] = row["legacy_apple_density_scarcity"]
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
            if csv_path.name.startswith("aggregated"):
                continue
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

    df = df.sort_values(["ecology_id", "seed", "episode"], kind="mergesort")
    df.to_csv(out_path, index=False)
    print(f"[aggregate] wrote {out_path} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
