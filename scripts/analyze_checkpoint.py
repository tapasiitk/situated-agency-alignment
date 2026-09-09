"""Compute the M1 representational-analysis measurements on one checkpoint.

Input:
    --rollout      Parquet (or JSONL) file produced by rollout_from_checkpoint.py
    --checkpoint   (optional) Required for Measurement 4 (gradient transfer).
    --config       (optional) Required for Measurement 4 to rebuild the env/agent.
    --output       Path to JSON summary.

Produces a single JSON summary dict with all five measurements, keyed by
measurement name. Missing inputs (e.g. no checkpoint) degrade to partial
outputs rather than crashing.

Usage:
    python scripts/analyze_checkpoint.py \
        --rollout   results/m1_smoke40/rollouts/seed1_ep20.parquet \
        --checkpoint results/m1_smoke40/checkpoints/m1_smoke40_baseline_seed1_ep20.pt \
        --config    configs/m1_smoke40.yaml \
        --output    results/m1_smoke40/analysis/seed1_ep20.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from karmic_rl.utils.roles import (
    ROLE_AGGRESSOR,
    ROLE_CLEANER,
    ROLE_FORAGER,
    ROLE_NAMES,
    ROLE_NEUTRAL,
    ROLE_VICTIM,
)


# --- IO ----------------------------------------------------------------------


def load_rollout(path: Path):
    """Load rollout records into a pandas DataFrame."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        records = [json.loads(line) for line in path.open("r")]
        return pd.DataFrame.from_records(records)
    raise ValueError(f"Unknown rollout format: {path}")


def stack_embeddings(df) -> np.ndarray:
    """Convert the list-of-lists embedding column to a (N, D) float32 array."""
    col = df["embedding"]
    return np.vstack([np.asarray(v, dtype=np.float32) for v in col])


# --- Measurement 1 -- linear probes -----------------------------------------


def measure_linear_probes(df, embeddings: np.ndarray) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score

    results: Dict[str, Any] = {}
    roles = df["role"].to_numpy()

    # --- 5-way probe (may be imbalanced; use balanced CV)
    classes, counts = np.unique(roles, return_counts=True)
    class_counts = dict(zip(classes.tolist(), counts.tolist()))
    results["class_counts"] = {ROLE_NAMES[int(c)]: int(n) for c, n in class_counts.items()}

    if len(classes) >= 2 and counts.min() >= 5:
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            cv = StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True, random_state=0)
            from sklearn.model_selection import cross_val_score

            auc_scores = cross_val_score(
                clf, embeddings, roles, cv=cv, scoring="roc_auc_ovr_weighted"
            )
            f1_scores = cross_val_score(
                clf, embeddings, roles, cv=cv, scoring="f1_macro"
            )
            results["probe_5way_auroc_mean"] = float(auc_scores.mean())
            results["probe_5way_auroc_std"] = float(auc_scores.std())
            results["probe_5way_f1_mean"] = float(f1_scores.mean())
        except Exception as exc:
            results["probe_5way_error"] = str(exc)
    else:
        results["probe_5way_skipped"] = "insufficient class support"

    # --- Binary probe: ZAP_AGENT vs BEING_ZAPPED (the key comparison)
    mask_a = roles == ROLE_AGGRESSOR
    mask_v = roles == ROLE_VICTIM
    n_a, n_v = int(mask_a.sum()), int(mask_v.sum())
    results["n_aggressor"] = n_a
    results["n_victim"] = n_v

    if n_a >= 5 and n_v >= 5:
        X = np.vstack([embeddings[mask_a], embeddings[mask_v]])
        y = np.concatenate([np.zeros(n_a), np.ones(n_v)])
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            cv = StratifiedKFold(n_splits=min(5, n_a, n_v), shuffle=True, random_state=0)
            from sklearn.model_selection import cross_val_score

            auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
            results["probe_agg_vs_vic_auroc"] = float(auc)
        except Exception as exc:
            results["probe_agg_vs_vic_error"] = str(exc)
    else:
        results["probe_agg_vs_vic_skipped"] = f"n_agg={n_a} n_vic={n_v}"

    return results


# --- Measurement 2 -- CKA ----------------------------------------------------


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019). X, Y are (n_samples, n_features)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    den = (
        np.linalg.norm(Xc.T @ Xc, ord="fro")
        * np.linalg.norm(Yc.T @ Yc, ord="fro")
    )
    if den < 1e-12:
        return 0.0
    return float(num / den)


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= n:
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def measure_cka_matrix(df, embeddings: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    roles = df["role"].to_numpy()
    rng = np.random.default_rng(seed)
    role_ids = [ROLE_NEUTRAL, ROLE_AGGRESSOR, ROLE_VICTIM, ROLE_CLEANER, ROLE_FORAGER]
    per_role = {r: embeddings[roles == r] for r in role_ids}
    counts = {r: int(len(per_role[r])) for r in role_ids}

    # Match sample sizes across non-empty classes.
    present = [r for r, n in counts.items() if n > 0]
    if not present:
        return {"cka_skipped": "no role data"}
    n_match = min(counts[r] for r in present)
    n_match = min(n_match, 1000)
    if n_match < 5:
        return {"cka_skipped": f"min class size {n_match} < 5", "class_counts": counts}

    subsampled = {r: _subsample(per_role[r], n_match, rng) for r in present}

    mat: Dict[str, Dict[str, float]] = {}
    for r1 in present:
        row: Dict[str, float] = {}
        for r2 in present:
            if r1 == r2:
                row[ROLE_NAMES[r2]] = 1.0
            else:
                row[ROLE_NAMES[r2]] = linear_cka(subsampled[r1], subsampled[r2])
        mat[ROLE_NAMES[r1]] = row

    out: Dict[str, Any] = {"cka_matrix": mat, "n_per_role_cka": n_match}
    out["class_counts"] = {ROLE_NAMES[r]: n for r, n in counts.items()}

    # Headline scalar: CKA(ZAP_AGENT, BEING_ZAPPED)
    name_a, name_v = ROLE_NAMES[ROLE_AGGRESSOR], ROLE_NAMES[ROLE_VICTIM]
    if name_a in mat and name_v in mat.get(name_a, {}):
        out["cka_agg_vs_vic"] = float(mat[name_a][name_v])
    return out


# --- Measurement 3 -- RSA (prototype cosine distances) ----------------------


def measure_prototype_distances(df, embeddings: np.ndarray) -> Dict[str, Any]:
    roles = df["role"].to_numpy()
    role_ids = [ROLE_NEUTRAL, ROLE_AGGRESSOR, ROLE_VICTIM, ROLE_CLEANER, ROLE_FORAGER]
    protos: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for r in role_ids:
        mask = roles == r
        counts[r] = int(mask.sum())
        if mask.sum() >= 3:
            protos[r] = embeddings[mask].mean(axis=0)

    if len(protos) < 2:
        return {"rsa_skipped": "too few roles present", "class_counts": {
            ROLE_NAMES[r]: n for r, n in counts.items()
        }}

    def cosdist(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(1.0 - (a @ b) / (na * nb))

    mat: Dict[str, Dict[str, float]] = {}
    for r1 in protos:
        row: Dict[str, float] = {}
        for r2 in protos:
            row[ROLE_NAMES[r2]] = cosdist(protos[r1], protos[r2])
        mat[ROLE_NAMES[r1]] = row

    out = {
        "prototype_cosdist_matrix": mat,
        "class_counts": {ROLE_NAMES[r]: n for r, n in counts.items()},
    }
    name_a, name_v = ROLE_NAMES[ROLE_AGGRESSOR], ROLE_NAMES[ROLE_VICTIM]
    if name_a in mat and name_v in mat.get(name_a, {}):
        out["cosdist_agg_vs_vic"] = float(mat[name_a][name_v])
    return out


# --- Measurement 4 -- gradient transfer --------------------------------------


def measure_gradient_transfer(
    df,
    checkpoint_path: Optional[Path],
    config_path: Optional[Path],
    n_pairs: int = 512,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute cosine between ∂V/∂e at victim states and ∂logπ(ZAP)/∂e at
    aggressor states. Uses aggressor/victim observations from the rollout —
    it does NOT pair events by timestep symmetry (that would require env
    replay with two different roles). Instead we take a fresh random matching
    of aggressor and victim observations; under the null H0 (gradients
    uncorrelated), this is the correct comparison.
    """
    if checkpoint_path is None or config_path is None:
        return {"gradient_transfer_skipped": "no checkpoint/config provided"}

    import torch
    import torch.nn.functional as F
    import yaml

    from karmic_rl.agents.karma_agent import KarmaAgent
    from karmic_rl.envs.harvest_dual import HarvestDualEnv

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    grid_sz = cfg["env"]["grid_size"]
    device = torch.device(
        device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "baseline")
    agent = KarmaAgent(obs_shape=(3, grid_sz, grid_sz), mode=mode, contrastive_weight=0.0).to(device)
    state = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in ckpt["model_state_dict"].items()
    }
    agent.load_state_dict(state, strict=False)
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)

    # We need raw observations, which we did not store in the rollout (only
    # embeddings / features). Regenerate a small batch of (aggressor, victim)
    # observations by running a quick rollout here. To avoid a dependency
    # cycle, import the rollout helper directly.
    from scripts.rollout_from_checkpoint import build_env, run_rollout

    env = build_env(cfg["env"])
    mini_records = run_rollout(
        agent=agent,
        env=env,
        num_episodes=2,
        device=device,
        deterministic=False,
        seed_base=99_997,
    )

    # Re-generate observation tensors for aggressor and victim events by
    # replaying: we need to call agent(obs) with grad enabled, so we cannot
    # use the no-grad rollout. Instead, collect obs arrays per role.
    # Strategy: step the env again under a fixed seed, capture (obs, role),
    # subsample into aggressor vs victim buckets.
    env = build_env(cfg["env"])
    agg_obs: List[np.ndarray] = []
    vic_obs: List[np.ndarray] = []
    target_each = max(16, n_pairs // 4)

    seed_iter = 77_777
    while (len(agg_obs) < target_each or len(vic_obs) < target_each) and seed_iter < 77_777 + 50:
        obs_dict, _ = env.reset(seed=seed_iter)
        seed_iter += 1
        agent_ids = env.possible_agents
        n_a = len(agent_ids)
        h_state = torch.zeros(n_a, agent.hidden_dim, device=device)
        c_state = torch.zeros(n_a, agent.hidden_dim, device=device)
        aid_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

        for _ in range(env.max_steps):
            active = [aid for aid in agent_ids if aid in obs_dict]
            if not active:
                break
            active_idx = [aid_to_idx[aid] for aid in active]
            obs_np = (
                np.stack([obs_dict[aid].transpose(2, 0, 1) for aid in active]).astype(np.float32)
                / 255.0
            )
            obs_tensor = torch.from_numpy(obs_np).to(device)
            with torch.inference_mode():
                out = agent(obs_tensor, (h_state[active_idx], c_state[active_idx]))
                h_state[active_idx] = out["new_hidden"][0]
                c_state[active_idx] = out["new_hidden"][1]
                logits = torch.nan_to_num(out["policy"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
            action_dict = {aid: int(a.item()) for aid, a in zip(active, actions)}
            next_obs, _, terms, truncs, infos = env.step(action_dict)
            for i, aid in enumerate(active):
                events = infos[aid]["social_events"]
                for e in events:
                    if e.get("type") == "ZAP_AGENT" and e.get("attacker") == aid:
                        if len(agg_obs) < target_each:
                            agg_obs.append(obs_np[i].copy())
                    elif e.get("type") == "BEING_ZAPPED" and e.get("victim") == aid:
                        if len(vic_obs) < target_each:
                            vic_obs.append(obs_np[i].copy())
            obs_dict = next_obs
            if len(agg_obs) >= target_each and len(vic_obs) >= target_each:
                break
            if all(terms.values()) or all(truncs.values()):
                break

    if len(agg_obs) < 8 or len(vic_obs) < 8:
        return {
            "gradient_transfer_skipped": "insufficient aggressor/victim samples",
            "n_aggressor": len(agg_obs),
            "n_victim": len(vic_obs),
        }

    rng = np.random.default_rng(0)
    n_pair = min(n_pairs, len(agg_obs), len(vic_obs))
    agg_arr = np.stack(agg_obs[:n_pair])
    vic_arr = np.stack(vic_obs[:n_pair])
    # Random matching (permutation of one side).
    perm = rng.permutation(n_pair)
    vic_arr = vic_arr[perm]

    cosines: List[float] = []
    for i in range(n_pair):
        o_a = torch.tensor(agg_arr[i], device=device).unsqueeze(0)
        o_v = torch.tensor(vic_arr[i], device=device).unsqueeze(0)

        # Aggressor: grad of log π(ZAP=action 7) w.r.t. embedding
        features_a = agent.cnn(o_a)
        e_a = agent.projector(features_a)
        e_a_req = e_a.detach().requires_grad_(True)
        h0 = torch.zeros(1, agent.hidden_dim, device=device)
        c0 = torch.zeros(1, agent.hidden_dim, device=device)
        h_a, _ = agent.lstm(e_a_req, (h0, c0))
        logits_a = agent.actor(h_a)
        logp_zap = F.log_softmax(logits_a, dim=-1)[0, 7]
        grad_pi = torch.autograd.grad(logp_zap, e_a_req, retain_graph=False)[0].detach()

        # Victim: grad of V w.r.t. embedding (note sign — "pain reduces V")
        features_v = agent.cnn(o_v)
        e_v = agent.projector(features_v)
        e_v_req = e_v.detach().requires_grad_(True)
        h_v, _ = agent.lstm(e_v_req, (h0, c0))
        v_vic = agent.critic(h_v).sum()
        grad_v = torch.autograd.grad(v_vic, e_v_req, retain_graph=False)[0].detach()

        # Cosine between the two gradients (same embedding-space basis).
        a_vec = grad_pi.view(-1)
        b_vec = grad_v.view(-1)
        na = torch.linalg.norm(a_vec) + 1e-12
        nb = torch.linalg.norm(b_vec) + 1e-12
        cos = float((a_vec @ b_vec) / (na * nb))
        cosines.append(cos)

    cosines = np.array(cosines, dtype=np.float32)
    return {
        "gradient_transfer_n_pairs": int(n_pair),
        "gradient_transfer_cos_mean": float(cosines.mean()),
        "gradient_transfer_cos_std": float(cosines.std()),
        "gradient_transfer_cos_median": float(np.median(cosines)),
        "gradient_transfer_cos_abs_mean": float(np.abs(cosines).mean()),
    }


# --- Driver ------------------------------------------------------------------


def run_all(
    rollout_path: Path,
    checkpoint_path: Optional[Path],
    config_path: Optional[Path],
    device_str: Optional[str],
) -> Dict[str, Any]:
    t0 = time.time()
    df = load_rollout(rollout_path)
    embeddings = stack_embeddings(df)

    summary: Dict[str, Any] = {
        "rollout_path": str(rollout_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "config_path": str(config_path) if config_path else None,
        "n_rows": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
    }

    try:
        summary["measurement_1_probes"] = measure_linear_probes(df, embeddings)
    except Exception as exc:
        summary["measurement_1_probes"] = {"error": str(exc)}

    try:
        summary["measurement_2_cka"] = measure_cka_matrix(df, embeddings)
    except Exception as exc:
        summary["measurement_2_cka"] = {"error": str(exc)}

    try:
        summary["measurement_3_rsa"] = measure_prototype_distances(df, embeddings)
    except Exception as exc:
        summary["measurement_3_rsa"] = {"error": str(exc)}

    try:
        summary["measurement_4_gradient_transfer"] = measure_gradient_transfer(
            df, checkpoint_path, config_path, device_str=device_str
        )
    except Exception as exc:
        summary["measurement_4_gradient_transfer"] = {"error": str(exc)}

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    rollout_path = Path(args.rollout)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    config_path = Path(args.config) if args.config else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = run_all(rollout_path, checkpoint_path, config_path, args.device)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[analyze] wrote {output_path}")
    print(json.dumps(
        {
            k: v for k, v in summary.items()
            if not k.startswith("measurement_")
        },
        indent=2,
    ))
    for name in ("measurement_1_probes", "measurement_2_cka",
                 "measurement_3_rsa", "measurement_4_gradient_transfer"):
        block = summary.get(name, {})
        print(f"[analyze] {name}:")
        if isinstance(block, dict):
            for k, v in block.items():
                if isinstance(v, (int, float, str)):
                    print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
