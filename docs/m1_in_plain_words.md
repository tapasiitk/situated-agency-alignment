# M1 in plain words

A friendly, step-by-step walkthrough of what M1 is doing, in simple language with analogies. Each step points at the **exact code file** that runs it, and the **technical term** is in brackets so you can find it in papers and configs.

**TL;DR** — M1 tests whether a standard multi-agent RL agent, **when it has no intervention like KARMA**, develops **two separate internal pictures** of the same event: “I hit someone” vs “I was hit.” If these two pictures stay geometrically separate, that is evidence for an **empathy gap** that **motivates** M2 (the KARMA intervention). M1 does **not** run KARMA.

> **Big analogy.** Imagine a kid learning about bumping into people on a playground.
> - If the kid files “me bumping someone” and “someone bumping me” into **two unrelated boxes in their head**, they never combine the lessons → aggression keeps happening.
> - If both go into **one shared box** (“bumping hurts, regardless of who”), the kid learns to avoid it.
> M1 is the careful measurement of **what the kid’s boxes look like** when we only let them play (no extra teaching).

---

## Step 0 — Fix the playground (environment) and the agent recipe

We freeze **where** the agents play and **how** they are trained.

- Playground file: `configs/m1_env_A_sc030.yaml` and siblings `configs/m1_env_*_sc*.yaml`, with shared defaults in `configs/m1_base.yaml`.
  - **Env A** = “tag only” (agents can only zap each other).
  - **Env B** = adds a waste/cooperation channel.
  - **`sc030`** = **scarcity 0.30** (apples are moderately scarce). Different cells test **easy**, **medium**, **hard** scarcity.
- Agent recipe (trainer): `train_karma.py`. For M1 we always pass **`--mode baseline`** — no KARMA and no broken-mirror control. That is the whole point: **baseline is the patient under study.**

Analogy: We are **not changing the brain** of the kid. We are only **watching** them grow up in the same park.

Terms in brackets (for reading papers later): **environment config**, **baseline PPO+LSTM agent**, **scarcity**, **sequential social dilemma / SSD**.

---

## Step 1 — Train once; save a photo album of the brain over time

We train the baseline agent for **4000 episodes** per seed. Every **200 episodes** we save the agent’s network weights to disk.

- Command: see `scripts/m1_smoke.sh` for the shape, and the **real training** line documented in `docs/m1_reproducibility.md` §1 (`python train_karma.py --config ... --mode baseline --seed 42`).
- Output folder: `results/m1_env_A_sc030/checkpoints/m1_env_A_sc030_baseline_seed42_ep<E>.pt`. Think of each `.pt` file as a **dated photo of the brain**. We will later peek inside each photo.
- Training log CSV is saved in the same `results/...` tree; it stores per-episode behaviour numbers like **violence rate** and **return**.

Analogy: It is the school year of the kid. We take a **school photo every Friday** so we can compare how the kid thinks at different ages.

Terms in brackets: **checkpoint (`.pt`)**, **training episodes**, **checkpoint interval**.

---

## Step 2 — For each photo, make the brain play a few games and record everything

For each of the **20 checkpoints** (episodes 200, 400, …, 4000), we replay the agent in the environment for a handful of **evaluation episodes** and write out **every step** it took and **every thought** (activations) it had.

- Code: `scripts/rollout_from_checkpoint.py` (writes one row per `(episode, step, agent)` with fields like `role`, `action`, `reward`, `embedding[64]`, `cnn_features`, `lstm_h/c`).
- Output: large tables, one file per checkpoint, e.g.  
  `results/m1_env_A_sc030/rollouts/trajectory_m1_env_A_sc030_baseline_seed42/m1_env_A_sc030_baseline_seed42_ep<E>.parquet`.
- Eval episodes per checkpoint: **20** for the main protocol (might go to **80** only if a one-time power check forces it).

Analogy: We ask the kid from each school photo to **play five practice games** while a silent observer **writes a play-by-play** in a notebook.

Terms in brackets: **rollout**, **evaluation episode**, **trajectory dataframe**, **embeddings / latent representations**.

---

## Step 3 — Run four “brain-shape” measurements on each notebook

For each rollout table, compute the **four representational measurements**. We want to see whether **aggressor-view** (“I zapped someone”) and **victim-view** (“I got zapped”) sit in **different rooms** of the agent’s brain.

- Code: `scripts/analyze_checkpoint.py`. Writes one JSON per checkpoint under `results/.../analysis/trajectory_*/...ep<E>.json`.

The measurements (what they feel like in plain words):

1. **Linear probes** (“Can a lazy reader guess the role just by looking at the brain vector?”)  
   - Output columns: `measurement_1_probes.probe_5way_auroc_mean`, `probe_agg_vs_vic_auroc`, plus the minimum class counts `n_aggressor` and `n_victim` that decide whether a checkpoint has **enough data** to trust the numbers.
2. **CKA** (“How similar are two clouds of brain vectors?”)  
   - Focus column: `measurement_2_cka.cka_agg_vs_vic`. Low CKA between **aggressor** and **victim** views = **they live in different rooms**.
3. **Prototype geometry / RSA** (“Where is the typical ‘aggressor’ vector vs the typical ‘victim’ vector?”)  
   - Focus column: `measurement_3_rsa.cosdist_agg_vs_vic`.
4. **Gradient transfer** (“When the agent feels ‘I was hit,’ does that feedback push the same buttons it uses to decide to hit?”)  
   - Focus column: `measurement_4_gradient_transfer.gradient_transfer_cos_mean`. If it is near zero or negative on average, the two lessons **do not teach each other**.

Analogy: Four different referees watch the same playbook and each writes one sentence about **how separate** the two experiences look in the kid’s head.

Terms in brackets: **linear probe AUROC**, **Centered Kernel Alignment (CKA)**, **prototype distance / representational similarity analysis (RSA)**, **gradient cosine / Fisher-style transfer**.

---

## Step 4 — Collapse everything into one spreadsheet

All those per-checkpoint JSONs are joined with the training CSV into **one long-format table**.

- Code: `scripts/aggregate_m1.py`.
- Output: `results/m1_env_A_sc030/aggregated_m1_env_A_sc030_baseline_seed42.csv`. Each row = one **checkpoint**; each column = **one metric** (behaviour or brain).

Analogy: We take the 20 referee cards and staple them into **one report card** for the whole school year.

Terms in brackets: **aggregated trajectory CSV**, **long-format table**.

---

## Step 5 — Draw the growth curves

We plot the metrics **over training episodes** to see the story.

- Code: `scripts/plot_m1_trajectory.py`.
- Output PNGs under `results/m1_env_A_sc030/plots/01_*.png` … `07_summary_2x2.png`. Figure 7 is a one-page dashboard.

Analogy: Take the report card and draw **growth charts**: when does the kid get taller, quieter, more separate-boxed?

Terms in brackets: **trajectory plots**, **dashboard figure**.

---

## Step 6 — Decide what “M1 passed” means (preregistration)

Before the main runs, we **write down the rules** so we cannot move the goalposts. Those rules are in `docs/m1_experimental_guideline.md` §2.3 and summarised in `docs/m1_reproducibility.md`:

- **Baseline only** (`--mode baseline`), no KARMA, no broken-mirror control.
- **Grid:** env × scarcity × **3 fixed seeds** (`42`, `123`, `456`) for the preregistered main campaign.
- **Checkpoints:** 200 : 200 : 4000 (20 per run).
- **Eval episodes:** 20 per checkpoint, with a **one-time ep4000 power check** that can escalate to 80.
- **`n_min = 100`** for confirmatory CKA and binary agg–vic probe (per-checkpoint row counts ≥ 100). Frozen from the Env A sc030 pilot evidence.
- **Attrition report:** say how many checkpoints actually met the rule.
- **Primary metrics vs exploratory metrics** are **labelled in advance**; everything else is hypothesis-generating.

Analogy: Before watching the kid all year, we write **what counts as evidence** on the fridge. After the year we can only read the fridge — not invent new rules.

Terms in brackets: **preregistration**, **primary vs exploratory endpoints**, **stopping / amendment rules**, **pre-specified `n_min`**.

---

## What M1 does NOT do (to avoid confusion)

- **No KARMA and no “broken mirror” variant.** Those are the **intervention (M2)** and the **control (M2b)**. M1 only watches the **untouched baseline**.
- **No claims from one seed alone.** The Env A sc030 seed 42 run was a **pilot**: it validated the pipeline and the `n_min` choice. Confirmatory claims need the full grid.
- **No cherry-picking of checkpoints.** We analyze **every** saved checkpoint; if one fails the `n_min` rule, we **tag** it, not hide it.

---

## Cheat-sheet of file names

| Step | What it does | File |
|------|--------------|------|
| 0 | Fix env + agent recipe | `configs/m1_env_A_sc030.yaml`, `configs/m1_base.yaml`, `train_karma.py` |
| 1 | Train & save checkpoints | `train_karma.py` |
| 2 | Rollouts per checkpoint | `scripts/rollout_from_checkpoint.py` (called by `scripts/batch_m1_trajectory.sh`) |
| 3 | 4 brain-shape measurements | `scripts/analyze_checkpoint.py` (called by `scripts/batch_m1_trajectory.sh`) |
| 4 | Aggregate to one CSV | `scripts/aggregate_m1.py` |
| 5 | Plots | `scripts/plot_m1_trajectory.py` |
| 6 | Rules & bookkeeping | `docs/m1_experimental_guideline.md`, `docs/m1_reproducibility.md` |

## Glossary (one-liners)

- **Checkpoint.** A saved snapshot of the agent’s weights (`.pt` file).
- **Rollout.** Running an agent in the env and writing a per-step table.
- **Embedding.** The agent’s internal vector for a situation.
- **Linear probe (AUROC).** A small classifier asking “can I read the role off the embedding?” AUROC near 0.5 = can’t; near 1.0 = easy.
- **CKA.** A similarity score between two sets of embeddings (0 = different, 1 = same shape).
- **Prototype distance (RSA).** Distance between the average embedding of each role.
- **Gradient transfer.** Whether the learning signal from “being hit” points in the same direction as the learning signal that would reduce hitting others.
- **`n_min`.** Minimum per-class rows we require in a rollout before we trust the role-comparison numbers (frozen at **100** for the main campaign).
- **Empathy gap (the hypothesis).** The baseline encodes “I hit” and “I was hit” as **separate** internal states, so feedback from one does not tame the other.
