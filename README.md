# Deep Hedging — When AI Learns to Outsmart Black-Scholes

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-green" />
  <img src="https://img.shields.io/badge/Gymnasium-1.0%2B-0081A5" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> **Can a neural network learn to hedge derivatives better than the Nobel Prize-winning Black-Scholes formula?**
>
> This project pits a Reinforcement Learning agent against the most famous equation in finance — and the results might surprise you.

---

## The Problem

You sold a European call option. The market moves. You need to **dynamically adjust your stock holdings** to protect yourself from catastrophic losses. For 50 years, traders have relied on the **Black-Scholes delta** — an elegant closed-form solution that tells you exactly how much stock to hold at every instant.

But Black-Scholes makes assumptions that reality doesn't respect:
- It assumes **continuous** trading (you can't trade every nanosecond)
- It ignores **transaction costs** (every trade costs money)
- It assumes **constant volatility** (it never is)

**Deep Hedging** throws away the formula and lets an RL agent learn the optimal hedging strategy directly from simulated experience — transaction costs, discrete time steps, and all.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  DerivativeHedgingEnv                    │
│                                                         │
│   State: [Stock Price, Hedge Ratio, Time Left,          │
│           Option Value, Wealth]                         │
│                                                         │
│   Action: New hedge ratio ∈ [0, 1]                      │
│                                                         │
│   Dynamics: Geometric Brownian Motion (GBM)             │
│   Costs: Proportional transaction costs (1 bp/trade)    │
│   Reward: −λ(ΔWealth)² (penalise P&L variance)         │
│                                                         │
│   Wealth = Cash + Stock Holdings − Option Liability     │
└─────────────────────────────────────────────────────────┘
            │                           │
            ▼                           ▼
    ┌───────────────┐          ┌────────────────┐
    │   PPO Agent   │          │  Black-Scholes  │
    │  (64×64 MLP)  │          │   Δ = N(d₁)    │
    └───────────────┘          └────────────────┘
            │                           │
            └───────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │   Compare P&L    │
              │   Distributions  │
              └──────────────────┘
```

## Project Structure

```
deepHedging/
├── env.py                          # Gymnasium environment (GBM, transaction costs, reward)
├── train_ppo.py                    # CLI script to train the PPO agent
├── bs_benchmark.py                 # CLI script to run the BS delta baseline
├── deep_hedging_comparison.ipynb   # All-in-one notebook: train, evaluate, visualise
├── ppo_hedging.zip                 # Saved PPO model weights
├── ppo_hedging_vecnormalize.pkl    # Observation normalisation statistics
├── bs_benchmark.png                # BS baseline plots (auto-generated)
└── README.md
```

| File | What it does |
|---|---|
| **`env.py`** | The heart of the project. A Gymnasium-compliant environment that simulates discrete-time delta hedging of a European call under GBM with proportional transaction costs. All other files import from here. |
| **`train_ppo.py`** | Standalone training script. Wraps the env in `DummyVecEnv` + `VecNormalize`, trains a PPO agent, and saves the model. Supports command-line arguments. |
| **`bs_benchmark.py`** | Runs the textbook Black-Scholes delta strategy over thousands of Monte-Carlo paths and reports P&L statistics. The "opponent" our RL agent is trying to beat. |
| **`deep_hedging_comparison.ipynb`** | The main deliverable. Trains the agent, runs both strategies on identical price paths, and produces comparison plots. Also demonstrates usage of all standalone scripts. |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/prithvishahh/deepHedging.git
cd deepHedging
python -m venv .venv && source .venv/bin/activate
pip install gymnasium stable-baselines3 torch numpy matplotlib
```

### 2. Train the RL Agent

**Option A — Notebook** (recommended):
Open `deep_hedging_comparison.ipynb` and run all cells.

**Option B — Command line:**
```bash
# Train with defaults (200k timesteps)
python train_ppo.py

# Train longer for better results
python train_ppo.py --timesteps 1000000 --save-path ppo_hedging_1M
```

### 3. Run the Black-Scholes Baseline

```bash
python bs_benchmark.py
```

### 4. Compare Results

The notebook evaluates both strategies on **1,000 identical test paths** and produces:
- **P&L histogram** — overlapping distributions showing how each strategy performs
- **Path comparison** — hedge ratio trajectories on a single path (RL vs. BS delta)

## Environment Details

| Parameter | Default | Description |
|---|---|---|
| `S0` | 100.0 | Initial stock price |
| `K` | 100.0 | Strike price |
| `T` | 30/365 | Time to maturity (≈ 1 month) |
| `sigma` (σ) | 0.2 | Annualised volatility |
| `r` | 0.05 | Risk-free interest rate |
| `n_steps` | 30 | Number of hedging intervals |
| `transaction_cost_bps` | 1.0 | Proportional cost per trade (basis points) |
| `risk_penalty` (λ) | 1.0 | Reward scaling: $r_t = -\lambda (\Delta W_t)^2$ |

## The Math

**Stock dynamics** (Geometric Brownian Motion):

$$S_{t+1} = S_t \cdot \exp\left[\left(r - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\;Z_t\right], \quad Z_t \sim \mathcal{N}(0,1)$$

**Black-Scholes delta** (the baseline):

$$\Delta_{BS} = N(d_1), \quad d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$$

**Reward function** (risk-averse quadratic penalty):

$$r_t = -\lambda \cdot (\Delta W_t)^2$$

where $W_t = \text{Cash}_t + \delta_t \cdot S_t - V_t$ is the net wealth of the hedging portfolio.

## Tech Stack

- **[Gymnasium](https://gymnasium.farama.org/)** — RL environment interface
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — PPO implementation
- **[PyTorch](https://pytorch.org/)** — neural network backend
- **NumPy / Matplotlib** — numerical computing & visualisation

## Key Insights

- With **zero transaction costs**, the BS delta is near-optimal — the RL agent learns to approximately replicate it.
- With **non-zero transaction costs**, the RL agent learns to trade *less frequently* than BS delta, resulting in lower cumulative costs and potentially tighter P&L distributions.
- The quadratic reward naturally makes the agent **risk-averse** — it prefers consistent small errors over occasional large ones.
- Increasing training timesteps (500k–1M+) significantly improves the RL agent's performance.

## License

MIT — do whatever you want with it.

---

<p align="center">
  <i>Built with curiosity about whether AI can improve upon 50 years of quantitative finance.</i>
</p>
