# KARMA: Knowledge Acquisition via Role-Invariant Mirror Architecture

> **Emergent Ethical Alignment in Multi-Agent Reinforcement Learning via Role-Invariant Representation Learning**

> *"I do not harm you, because I know what harm feels like."*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](./ethical_agentic_AI.pdf)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-Multi%20Agent-blueviolet)](https://pettingzoo.farama.org/)
[![WandB](https://img.shields.io/badge/WandB-Dashboard-gold.svg?logo=weightsandbiases)](https://api.wandb.ai/links/ratht-iitk/q7ethgj4)



## 🧠 The Core Problem

Standard Deep Reinforcement Learning (DRL) agents in multi-agent environments systematically converge to **aggressive Nash equilibria** when resources are scarce. In the canonical Harvest commons game, agents learn to use their primary tool (a beam weapon) not only for cooperative resource management but also for competitive exclusion, leading to mutual destruction and commons collapse.

**Root Cause:** Standard convolutional encoders treat logically symmetric social interactions as pixel-wise disjoint states:
- **"I attack a rival"** → latent vector z_agg
- **"I am attacked"** → latent vector z_vic

These representations are **orthogonal** (unrelated). Negative feedback from victimization does not modulate the policy for aggression because the agent perceives itself as existing in two separate "predator" and "prey" state spaces. We call this the **Empathy Gap**.

## 🪞 Our Solution: KARMA

KARMA (**K**nowledge **A**cquisition via **R**ole-invariant **M**irror **A**rchitecture) augments recurrent DRL agents with a **Siamese projector head** trained via contrastive loss to align structurally symmetric interactions:

```
L_KARMA = E[(f_θ(o_agg) - f_θ(o_vic))²]
```

By forcing the aggressor and victim views into the same latent embedding, the value function naturally learns that **harm is bad regardless of role**. Ethical behavior emerges from representational geometry, not explicit rules.

## 🔬 The Mirror Test: Three Conditions

We evaluate KARMA in a novel **Dual-Use Zap** variant of Harvest where the primary action serves both:
- **Cooperative:** ZAP → Waste (removes obstacles, enables apple regrowth)
- **Competitive:** ZAP → Agent (temporarily freezes rivals, monopolizes resources)

| Condition | Architecture | Contrastive Pairing | Expected Outcome |
|-----------|--------------|-------------------|------------------|
| **Baseline** | Standard DRQN | None | High Violence + High Cooperation → Low Yield (Tragedy) |
| **Broken Mirror** | DRQN + Siamese | ZAP_AGENT ≈ ZAP_WASTE | Violence ≈ Cooperation → Semantic Confusion → Lower Yield |
| **KARMA** | DRQN + Siamese | ZAP_AGENT ≈ BEING_ZAPPED | Violence ↓ Cooperation ↑ → Selective Ethics |

## 🛠️ Installation & Quick Start

```bash
# Clone repo
git clone https://github.com/tapasiitk/situated-agency-alignment.git
cd situated-agency-alignment

# Install dependencies
pip install -r requirements.txt
```

### Run the Mirror Test

```bash
# Terminal 1: Baseline (standard DRQN)
python train_karma.py --config configs/env_harvest.yaml --mode baseline

# Terminal 2: Broken Mirror (semantic ablation control)
python train_karma.py --config configs/env_harvest.yaml --mode broken

# Terminal 3: KARMA (ethical role-invariance)
python train_karma.py --config configs/env_harvest.yaml --mode karma
```

Monitor results in **Weights & Biases**: [wandb.ai](https://wandb.ai)

Expected runtime: ~8 hours per condition on NVIDIA RTX 3090.

## Phase A: Canonical Baseline Replication

Use this first before any environment extension or KARMA analysis.

```bash
python train_karma.py --config configs/canonical_baseline.yaml --mode baseline --seed 42
```

- VM/private-repo setup + sweep commands: [`docs/canonical_baseline_vm_guide.md`](docs/canonical_baseline_vm_guide.md)
- Local artifacts are saved as CSV/JSON under `results/canonical_baseline/`

## M1 trajectory (train → rollout → analyze → aggregate → plots)

- **Plain-language walkthrough** (analogies + files): [`docs/m1_in_plain_words.md`](docs/m1_in_plain_words.md).
- End-to-end commands, artifact layout, and a **run log** to append after each stage: [`docs/m1_reproducibility.md`](docs/m1_reproducibility.md).
- Protocol & preregistration detail (P1–P5, `n_min`, amendments): [`docs/m1_experimental_guideline.md`](docs/m1_experimental_guideline.md).

## 📂 Project Structure

```
situated-agency-alignment/
├── karmic_rl/
│   ├── envs/
│   │   └── harvest_dual.py          # Dual-Use Zap environment
│   └── agents/
│       └── karma_agent.py           # Siamese projector + PPO agent
├── train_karma.py                   # Main training script (all 3 conditions)
├── configs/
│   └── env_harvest.yaml             # Hyperparameters (grid, agents, rewards)
├── ethical_agentic_AI.pdf           # Full academic paper (PDF)
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🔧 Configuration

Edit `configs/env_harvest.yaml`:

```yaml
env:
  grid_size: 15                      # 15×15 grid
  num_agents: 6                      # N=6 agents
  max_steps: 1000                    # Episode length
  apple_density: 0.65                # Scarcity forces dilemma
  zap_waste_reward: 0.3              # Cooperative incentive (> violence 0.1)
  
training:
  episodes: 10000                    # 10k episodes per condition
  ppo_epochs: 4                      # PPO update epochs
  batch_size: 64                     # Contrastive batch size
  lr: 3e-4                           # Learning rate
  gamma: 0.99                        # Discount factor
  contrastive_weight: 0.1            # KARMA loss weight λ
```

## 📈 Metrics

We track four metrics per episode:

- **Violence Rate:** ZAP_AGENT events (lower is better)
- **Cooperation Rate:** ZAP_WASTE events (higher is better)  
- **System Yield:** Total apples consumed per agent (reflects sustainability)
- **Ethical Selectivity:** Ratio of cooperation to violence (KARMA achieves 8.3× baseline)

## 🧬 Architecture Details

```
CNN (32-64-128 filters)
    ↓
Siamese Projector f_θ (MLP: 256→128→64)
    ↓
LSTM (hidden_dim=256)
    ↓
Actor Head (Policy π) + Value Head (V)
```

**Training Loss:**
```
L_total = L_PPO + λ L_KARMA + β L_role
```

where λ=0.1 (contrastive weight), β=0.1 (auxiliary role loss).

## 📖 Citation

If you use this code or paper, please cite:

```bibtex
@article{rath2025karma,
  title={Emergent Ethical Alignment in Multi-Agent Reinforcement Learning 
         via Role-Invariant Representation Learning},
  author={Rath, Tapas Ranjan},
  journal={in preparation},
  year={2025},
  url={https://github.com/tapasiitk/situated-agency-alignment}
}
```

## 💡 Key Insights

1. **The Empathy Gap is representational:** Agents fail to generalize "harm is bad" across roles because standard encoders keep aggressor/victim views disjoint.

2. **Semantic correctness matters:** Adding architectural capacity (Broken Mirror) is insufficient. Contrastive pairs must be semantically grounded (role-symmetric).

3. **Ethics emerges from geometry:** By aligning representations, the value function naturally propagates aversion to harm-infliction, regardless of role.

4. **No explicit rules needed:** KARMA uses no reward shaping, hard constraints, or dense human supervision—only the environment's causal structure.

## 🌍 Broader Implications

- **Human-AI Collaboration:** KARMA agents may generalize better to multi-stakeholder settings where perspective-taking is valuable.
- **Adversarial Robustness:** Symmetric representations provide intrinsic regularization (attackers cannot exploit privileges one role lacks).
- **Transfer Learning:** Role-invariant embeddings may transfer to new domains where symmetry exists.

## 📚 Related Work

- **Sequential Social Dilemmas:** Leibo et al. (2017) - foundational Harvest/Cleanup games
- **Contrastive Learning:** Chen et al. (2020) SimCLR, He et al. (2020) MoCo
- **Moral AI:** Neufeld (2022), Hadfield-Menell et al. (2016), ethical RL frameworks

## ⚠️ Limitations & Future Work

### Current Limitations
- Assumes symmetric harm (A ↔ B reciprocal)
- Tested on closed-world Harvest domain only
- Requires manual social event logs for pair specification

### Future Directions
- Asymmetric power dynamics and role hierarchies
- Open-ended environments with novel harm types
- Autonomous pair inference from pixels
- Human user studies on "agency"
- Scaling to 10+ agents and larger grids

## 🤝 Contributing

Contributions welcome! Please open issues or PRs for:
- Bug fixes
- Experimental improvements
- Scalability enhancements
- New dual-use domains
- Visualization tools

## 📜 License

MIT License — See `LICENSE` file

---

## Quick Experiment Guide

### 1. Start Training

```bash
# Run all 3 conditions in parallel
python train_karma.py --config configs/env_harvest.yaml --mode baseline &
python train_karma.py --config configs/env_harvest.yaml --mode broken &
python train_karma.py --config configs/env_harvest.yaml --mode karma &
```

### 2. Monitor Progress

Open wandb dashboard:
```
wandb login
# Follow URL in browser
```

### 3. Expected Timeline

- **Episodes 0-1,000:** All conditions show exploration
- **Episodes 1,000-2,000:** Baseline violence increases; Broken Mirror shows confusion
- **Episodes 2,000-5,000:** KARMA violence collapses, cooperation stabilizes
- **Episodes 5,000-10,000:** KARMA maintains stable ethical equilibrium

### 4. Analyze Results

```bash
# After training completes
python scripts/plot_results.py --wandb-entity <your-entity> --wandb-project karma-mirror-test
```

---

**Made with ❤️ for aligned AI systems**

*For questions, open an issue or contact: [github.com/tapasiitk](https://github.com/tapasiitk)*

**Status:** Pre-release (experiments in progress)
**Last Updated:** December 2025
