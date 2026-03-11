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
- It assumes **constant volatility** (it never is — volatility itself is random)

**Deep Hedging** throws away the formula and lets an RL agent learn the optimal hedging strategy directly from simulated experience — stochastic volatility, transaction costs, discrete time steps, and all.

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
│   Dynamics: Heston Stochastic Volatility              │
│   Costs: Proportional transaction costs (0.1%/trade)   │
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
├── env.py                          # Gymnasium environment (Heston SV, transaction costs, reward)
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
| **`env.py`** | The heart of the project. A Gymnasium-compliant environment that simulates discrete-time delta hedging of a European call under **Heston stochastic volatility** with proportional transaction costs. All other files import from here. |
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
# Train with defaults (2M timesteps)
python train_ppo.py

# Train with custom settings
python train_ppo.py --timesteps 500000 --save-path ppo_hedging_500k
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
| `sigma` (σ) | 0.2 | Annualised volatility (used for BS pricing) |
| `r` | 0.05 | Risk-free interest rate |
| `n_steps` | 30 | Number of hedging intervals |
| `transaction_cost_rate` | 0.001 | Proportional cost rate per trade (0.001 = 0.1%) |
| `risk_penalty` (λ) | 0.1 | Variance-penalty coefficient |
| `v0` | σ² | Initial instantaneous variance |
| `kappa` (κ) | 2.0 | Variance mean-reversion speed |
| `theta` (θ) | σ² | Long-run variance level |
| `sigma_v` (σ_v) | 0.3 | Vol-of-vol |
| `rho` (ρ) | −0.7 | Correlation between price and variance shocks |

## The Math

**Stock dynamics** (Heston Stochastic Volatility):

$$dS_t = \mu\, S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^{(1)}$$

$$dv_t = \kappa(\theta - v_t)\, dt + \sigma_v \sqrt{v_t}\, dW_t^{(2)}$$

$$\text{corr}(dW^{(1)}, dW^{(2)}) = \rho$$

Discretised via log-Euler for S and Euler–Maruyama for v, with variance floored at zero.

**Black-Scholes delta** (the baseline — assumes constant vol):

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

- With **zero transaction costs and constant vol**, the BS delta is near-optimal — the RL agent learns to approximately replicate it.
- Under **Heston dynamics** the true volatility is stochastic, so BS delta (which assumes constant σ) is systematically wrong — the RL agent can exploit the realised-vol signal in its observation to outperform.
- With **non-zero transaction costs**, the RL agent learns to trade *less frequently* than BS delta, resulting in lower cumulative costs and potentially tighter P&L distributions.
- The **negative correlation** (ρ = −0.7) produces a realistic volatility skew: vol rises when markets fall, exactly as seen in equity markets.
- The variance-penalty reward naturally makes the agent **risk-averse** — it prefers consistent small errors over occasional large ones.
- Default training is 2M timesteps — sufficient for policy convergence on this task.

## License

MIT — do whatever you want with it.

---

<p align="center">
  <i>Built with curiosity about whether AI can improve upon 50 years of quantitative finance.</i>
</p>
