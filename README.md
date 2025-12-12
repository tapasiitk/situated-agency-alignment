# Karmic-RL: Scalable Ethical Alignment via Retroactive Credit Assignment 🚀



**Karmic-RL** solves the **Temporal Credit Assignment problem** in Sequential Social Dilemmas. Standard RL agents defect in resource scarcity because they cannot link short-term gains to long-term social costs. Our solution combines **Temporal Value Transport (TVT)** with a **Conspiring Matchmaker** to create emergent cooperation without explicit rules.

## 🎯 The Problem: Tragedy of the Commons in RL

In Leibo et al. (2017)'s **Harvest** game:
```
Standard RL (DRQN/PPO):     Karmic-RL (Ours):
Aggression ↑ with scarcity  Cooperation despite scarcity
```
```
           Standard RL              Karmic-RL
Apples:    ████████ → ░░░░░░     ██████ → ██████
Zaps:      ░░░░░░░ → ████████    ░░░░░░░ → ░░░░░░
```

**Why standard RL fails:** Agent A zaps B to monopolize apples. B retaliates 500 steps later in a different episode. LSTM gradients vanish. Agent A never learns the causal link.

## 🧠 The Innovation: Two Coupled Mechanisms

### 1. **Agent-Level: Semantic TVT (Temporal Value Transport)**
```
When punished → Query: "When was I aggressor?"
Memory snaps to past crime → Value transported back → Policy updated
```

**Architecture:**
```
Input: Grid (20x20x3) → CNN → LSTM → [Policy, Value]
                           ↓ (if TVT enabled)
                       External Memory (1000 slots)
                       Read Head: Semantic Role Matching
```

### 2. **Environment-Level: Conspiring Matchmaker**
```
Debt Ledger: AgentA → AgentB: 3.2 zaps
Reset(): Rig spawns → A & B start adjacent (100% interaction)
Retribution happens → TVT can learn the link
```

## 🧪 The 4 Ablation Conditions (Scientific Rigor)

| Condition | Agent | Environment | Prediction |
|-----------|-------|-------------|------------|
| **1. Baseline** | DRQN | Random Spawn | 🟥 High Aggression |
| **2. TVT-Only** | TVT | Random Spawn | 🟨 Still Aggressive* |
| **3. Matchmaker** | DRQN | Rigged Spawn | 🟨 Still Aggressive |
| **4. Karmic-RL** | TVT | Rigged Spawn | 🟩 Cooperation! |

\*In small grids (10x10). Fails completely in large grids (30x30).

## 📊 Results (Small World 10x10 vs Big City 30x30)



```
Small World (10x10): TVT alone suffices (density effect)
Big City (30x30):   Matchmaker REQUIRED (real-world scale)
```

## 🏗️ Code Structure

```
karmic_rl/
├── envs/
│   ├── harvest_parallel.py     # PettingZoo Harvest env
│   ├── matchmaker_wrapper.py   # God Ledger + Rigged Spawn
│   └── __init__.py            # gym.make("KarmicHarvest-v0")
├── agents/
│   └── ktvt_agent.py          # TVT Agent (use_tvt flag)
└── experiments/
    └── train.py               # 4 ablation conditions
```

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run ALL 4 conditions (2-3 hours)
python experiments/train.py --mode all --grid_size 20

# Run Full Karmic-RL only
python experiments/train.py --mode full --grid_size 30

# Small world test (faster)
python experiments/train.py --mode all --grid_size 10
```

**WandB Dashboard:** [wandb.ai/tapasiitk/karmic-rl](https://wandb.ai/tapasiitk/karmic-rl)

## 🔧 Usage API

```python
import gymnasium as gym

# 1. Baseline
env = gym.make("KarmicHarvest-v0")
agent = KarmicAgent(obs_shape, use_tvt=False)  # DRQN

# 2. Full Karmic-RL
env = gym.make("KarmicHarvest-v0")
env = KarmicMatchmaker(env)  # Rigged spawning
agent = KarmicAgent(obs_shape, use_tvt=True)  # TVT enabled

# Training loop (see train.py)
obs, infos = env.reset()
```

## 📈 Expected Plots (Auto-generated)

1. **Aggression Curves** (Money Plot):
```
Predatory Zaps/Episode vs Training Steps (4 curves overlaid)
```

2. **Debt Dynamics**:
```
Total Ledger Debt vs Steps (should decrease in Karmic-RL)
```

3. **Retribution Ratio**:
```
Good Zaps / Total Zaps (should increase → norm enforcement)
```

4. **Scale Comparison**:
```
10x10 vs 30x30 grids (Matchmaker becomes essential at scale)
```

## 🎓 Theoretical Foundation

> *"By combining Temporal Value Transport (Hung et al., 2019) with history-dependent environmental matching, we enable agents to internalize delayed social externalities without immediate feedback."*

**Key Insight:**
```
Standard RL:    Action(t) → Reward(t+1)     [Short horizon]
Karmic-RL:     Action(t) → Reward(t+500)   [Long horizon via TVT + Matchmaker]
```

## 🔬 Reproducibility

```bash
# Fixed seed for paper results
python experiments/train.py --mode all --grid_size 20 --seed 42

# Hyperparameters (train.py defaults)
lr=3e-4, gamma=0.99, clip=0.2, epochs=4, hidden=256
```

## 🧪 Environment Details

**Harvest Dynamics:**
- Grid: Configurable (10x10 small, 30x30 realistic)
- Apples regrow based on neighbor density
- Zap removes target for 25 steps (-0.5 reward)
- 8 actions: Move×4, Turn×2, Zap, No-op

**Social Events Logged:**
```python
{
  "event_type": "ZAP_HIT",
  "attacker": "agent_0", 
  "victim": "agent_2",
  "apple_context": true,  # Resource dispute?
  "timestamp": 127
}
```

## 📝 Citation

```bibtex
@misc{karmicrl2025,
  author = {Rath, Tapas Ranjan},
  title = {Karmic-RL: Scalable Ethical Alignment via Retroactive Credit Assignment},
  year = {2025},
  publisher = {GitHub: tapasiitk/situated-agency-alignment},
  note = {Code \& Experiments}
}
```

## 🤝 Acknowledgments

Built on:
- **PettingZoo** (MARL standard)
- **Temporal Value Transport** [Hung et al., 2019][2]
- **Leibo et al. SSDs**[1]


## 📄 Paper

**LaTeX version in development.** This README serves as the living paper.

***

⭐ **Star if emergent cooperation excites you!**  
🐛 **Issues/PRs welcome**  
💬 **Discussions for extensions** (human-in-loop, larger scales)

***

<div align="center">
  <img src="https://i.imgur.com/karmic-diagram.png" width="600">
  <p><i>The Karmic Loop: Crime → Matchmaking → Retribution → TVT Learning</i></p>
</div>

***

**Made with ❤️ for ethical AI**[4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_5dbe93eb-a7b5-4e81-a6a7-9b0031f5392c/4b748254-a341-4545-9429-5bcdfc5ccd98/formal_proposal_scalable_ethicalAI.pdf)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_5dbe93eb-a7b5-4e81-a6a7-9b0031f5392c/69394059-eec9-42d6-916d-d2b13dea5302/Hung-et-al.-2019-Optimizing-agent-behavior-over-long-time-scales-by-transporting-value.pdf)
[3](https://github.com/ml-jku/baselines-rudder)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_5dbe93eb-a7b5-4e81-a6a7-9b0031f5392c/072a59d7-41d3-4363-aa32-d3ced0e63cc9/presentation_scalable_ethical_AI.pdf)
