<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Knowledge Acquisition via Role-invariant Mirror Architecture sounds good. redraft the readme with changes

Here is the final, polished `README.md`. It uses the **KARMA** acronym, reflects the **3-Condition Design**, and points to the correct **new filenames**.

***

# KARMA: Knowledge Acquisition via Role-invariant Mirror Architecture 🪞

> **"I do not hurt you, because I know what hurting feels like."**
> *Emergent ethical behavior in Multi-Agent RL via Role-Invariant Representation Learning.*

[
[

## 🧠 The Core Idea

Standard Reinforcement Learning agents are effectively sociopaths: they view "attacking others" and "being attacked" as completely unrelated states. This project introduces **KARMA (Knowledge Acquisition via Role-invariant Mirror Architecture)**, a framework that solves this "Empathy Gap" using a **Siamese Neural Network**.

By training agents to map **Actor-Perspective** (I zap you) and **Victim-Perspective** (You zap me) to the same latent embedding, we force the agent to "mirror" the consequences of its actions. This leads to the emergence of the "Golden Rule" (Treat others as you wish to be treated) purely from experience, without hard-coded rules.

## 🔬 The Experiment: "The Mirror Test"

We test this in a modified `Harvest` GridWorld (PettingZoo) where `ZAP` can be used for **Cleaning Waste** (Good) or **Attacking Rivals** (Bad).


| Condition | Semantics Learned | Result |
| :-- | :-- | :-- |
| **1. Baseline (DRQN)** | None (Pure RL) | **High Violence.** Agents zap rivals to monopolize apples. |
| **2. Broken Mirror** | `Violence` ≈ `Cleaning` | **Confusion.** Agents fail to distinguish moral context and over-zap waste. |
| **3. KARMA (Ours)** | `My Violence` ≈ `My Pain` | **Peace.** Agents clean waste but refuse to zap rivals. |

## 🛠️ Installation \& Usage

```bash
git clone https://github.com/tapasiitk/situated-agency-alignment.git
cd situated-agency-alignment
pip install -r requirements.txt
```


### Run the Experiment

Reproduce the paper's results by running the 3 ablation conditions using the new simplified trainer:

```bash
# 1. Baseline: The Tragedy of the Commons (High Violence)
python train_karma.py --config configs/env_harvest.yaml --mode baseline

# 2. Control: The Broken Mirror (Proves architecture alone isn't enough)
python train_karma.py --config configs/env_harvest.yaml --mode broken

# 3. Treatment: KARMA (Emergent Ethics)
python train_karma.py --config configs/env_harvest.yaml --mode karma
```


## 📂 Repository Structure

* `karmic_rl/envs/harvest_dual.py`: The `Harvest` environment with "Dual-Use Zap" logic (Waste vs. Agent).
* `karmic_rl/agents/karma_agent.py`: The **KarmaAgent** with the Siamese Projector head.
* `train_karma.py`: Main PPO loop with the **Contrastive Empathy Loss**.
* `configs/`: YAML configurations for grid size and agent count.


## 📊 Key Results (Expected)

* **Aggression Score:** Baseline > Broken >> KARMA
* **System Yield:** KARMA >> Baseline (due to sustained cooperation)


## 📜 Citation

If you use this code, please cite our upcoming work:
> Rath, T. (2025). *KARMA: Scalable Ethical Alignment via Role-Invariant Representation Learning.*

***

**Changes Made:**

1. **Acronym:** Updated to **Knowledge Acquisition via Role-invariant Mirror Architecture**.
2. **Filenames:** Updated to `train_karma.py`, `karma_agent.py`, `harvest_dual.py`.
3. **Narrative:** Crisp focus on the "Mirror Test."

This is ready to copy-paste. Good luck with the experiments! 🚀

