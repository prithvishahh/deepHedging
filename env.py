"""
Gymnasium-compliant Deep Hedging Environment.

Simulates delta-hedging a European call option under Heston
stochastic volatility dynamics with proportional transaction
costs. The agent learns a hedging policy that minimises P&L
variance (risk-averse reward).

Heston dynamics:
    dS = μ·S·dt + √v·S·dW₁
    dv = κ(θ − v)·dt + σ_v·√v·dW₂
    corr(dW₁, dW₂) = ρ
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DerivativeHedgingEnv(gym.Env):
    """
    Observation (6-dim, continuous — all normalised to ~[0, 1] range):
        0  moneyness            S_t / K          (1.0 = at-the-money)
        1  time to maturity     tau / T           (1.0 → 0.0 over episode)
        2  current hedge ratio  delta_t           (agent's position)
        3  Black-Scholes delta  N(d1)             (analytical benchmark)
        4  realised volatility  σ_realised / σ    (ratio vs assumed vol)
        5  current wealth       W_t               (hedging P&L)

    Action (1-dim, continuous):
        new hedge ratio in [0, 1]

    Reward:
        At each step:   r_t = dW - λ·(dW²) - α·(Δδ)²
        At terminal step: r_T = W_T - λ·(W_T²) - α·(Δδ_T)²
        λ (risk_penalty)       controls risk-aversion strength.
        α (smoothness_penalty) penalises large changes in hedge position.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 30 / 365,
        sigma: float = 0.2,
        r: float = 0.05,
        n_steps: int = 30,
        transaction_cost_rate: float = 0.001,
        risk_penalty: float = 0.1,
        smoothness_penalty: float = 0.05,

        # --- Heston stochastic volatility parameters ---
        v0: float | None = None,
        kappa: float = 2.0,
        theta: float | None = None,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        seed: int | None = None,
    ):
        super().__init__()

        # --- market / contract parameters ---
        self.S0 = S0
        self.K = K
        self.T = T
        # implied vol -> BS pricing
        self.sigma = sigma
        self.r = r
        self.n_steps = n_steps
        self.dt = T / n_steps
        # proportional transaction cost rate (e.g. 0.001 = 0.1%)
        self.tc = transaction_cost_rate
        self.risk_penalty = risk_penalty
        # penalise large hedge jumps
        self.smoothness_penalty = smoothness_penalty

        # --- Heston parameters ---
        self.v0 = v0 if v0 is not None else sigma ** 2
        self.kappa = kappa
        #long-run variance
        self.theta = theta if theta is not None else sigma ** 2
        self.sigma_v = sigma_v
        #correlation of price and variance shocks
        self.rho = rho

        # --- spaces ---
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # obs: [moneyness, tau/T, hedge_pos, bs_delta, real_vol_ratio, wealth]
        obs_low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf], dtype=np.float32)
        obs_high = np.array([np.inf, 1.0, 1.0, 1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # will be set in reset()
        self._rng: np.random.Generator | None = None
        self._step_idx: int = 0
        self.S: float = S0
        self.v: float = self.v0
        self.delta: float = 0.0
        #cash account from trading (starts at option premium received)
        self._cash: float = 0.0
        #net wealth = cash + delta*S − V(S, tau)
        self.wealth: float = 0.0
        self._log_returns: list[float] = []

    # --- Black-Scholes helpers ---
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF via the error function (no scipy needed)."""
        from math import erf, sqrt
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _bs_call(self, S: float, tau: float) -> float:
        """Black-Scholes European call price."""
        if tau <= 1e-12:
            return max(S - self.K, 0.0)
        d1 = (
            np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau
        ) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)
        return float(
            S * self._norm_cdf(d1)
            - self.K * np.exp(-self.r * tau) * self._norm_cdf(d2)
        )

    def _bs_delta(self, S: float, tau: float) -> float:
        """Black-Scholes delta of a European call."""
        if tau <= 1e-12:
            return 1.0 if S > self.K else 0.0
        d1 = (
            np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau
        ) / (self.sigma * np.sqrt(tau))
        return float(self._norm_cdf(d1))

    def _realised_vol(self) -> float:
        """Annualised realised volatility from log-returns so far."""
        if len(self._log_returns) < 2:
            return self.sigma
        return float(np.std(self._log_returns, ddof=1) * np.sqrt(1.0 / self.dt))

    # --- Gymnasium API ---
    def _get_obs(self) -> np.ndarray:
        tau = self.T - self._step_idx * self.dt
        bs_delta = self._bs_delta(self.S, tau)
        real_vol = self._realised_vol()
        return np.array([
            #moneyness and time to maturity are normalised to help learning generalisation across different contract params
            self.S / self.K,
            tau / self.T if self.T > 0 else 0.0,
            #current hedge position
            self.delta,
            bs_delta,
            real_vol / self.sigma,
            self.wealth,
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed if seed is not None else None)

        self.S = self.S0
        self.v = self.v0
        self.delta = 0.0
        self._step_idx = 0
        self._log_returns = []

        # writer receives option premium as cash
        V0 = self._bs_call(self.S0, self.T)
        self._cash = V0
        # net wealth = cash + stock − option = V0 + 0 − V0 = 0
        self.wealth = 0.0

        obs = self._get_obs()
        info = {"bs_delta": self._bs_delta(self.S, self.T)}
        return obs, info

    def step(self, action: np.ndarray):
        new_delta = float(np.clip(action[0], 0.0, 1.0))
        old_delta = self.delta
        old_S = self.S

        # 1. rebalance: buy/sell stock, pay transaction costs
        trade_size = new_delta - old_delta
        trade_cost = self.tc * abs(trade_size) * old_S
        self._cash -= trade_size * old_S + trade_cost
        self.delta = new_delta

        # 2. evolve the stock price (Heston SV)
        z1 = self._rng.standard_normal()
        z2 = self._rng.standard_normal()
        # correlated Brownians: W1 = z1, W2 = ρ·z1 + √(1−ρ²)·z2
        w1 = z1
        w2 = self.rho * z1 + np.sqrt(1.0 - self.rho**2) * z2

        v_curr = max(self.v, 0.0)
        sqrt_v = np.sqrt(v_curr)

        # Euler–Maruyama for variance: dv = κ(θ−v)dt + σ_v·√v·dW₂
        dv = (self.kappa * (self.theta - v_curr) * self.dt
              + self.sigma_v * sqrt_v * np.sqrt(self.dt) * w2)
        self.v = max(v_curr + dv, 0.0)  # absorb at zero

        # Log-Euler for stock price: dS = μ·S·dt + √v·S·dW₁
        new_S = old_S * np.exp(
            (self.r - 0.5 * v_curr) * self.dt
            + sqrt_v * np.sqrt(self.dt) * w1
        )
        self._log_returns.append(float(np.log(new_S / old_S)))
        self.S = new_S

        # 3. accrue interest on cash
        self._cash *= 1.0 + self.r * self.dt

        # 4. update net wealth = cash + stock − option
        self._step_idx += 1
        tau_new = self.T - self._step_idx * self.dt
        V_new = self._bs_call(self.S, tau_new)

        old_wealth = self.wealth
        self.wealth = self._cash + self.delta * self.S - V_new
        dW = self.wealth - old_wealth

        # 5. reward: variance-penalty + smoothness-penalty
        #   trade_size = new_delta - old_delta is already computed in step 1.
        #   Smoothness penalty: α·(Δδ)²  — applied every step so the agent
        #   learns to prefer gradual rebalancing over large discrete jumps.
        #   Intermediate steps: r_t = dW - λ·dW² - α·(Δδ)²
        #   Terminal step:      r_T = W_T - λ·W_T² - α·(Δδ)²
        smoothness_pen = self.smoothness_penalty * trade_size ** 2
        terminated = self._step_idx >= self.n_steps
        if terminated:
            reward = float(self.wealth - self.risk_penalty * self.wealth**2 - smoothness_pen)
        else:
            reward = float(dW - self.risk_penalty * dW**2 - smoothness_pen)
        truncated = False

        obs = self._get_obs()
        info = {
            "bs_delta": self._bs_delta(self.S, tau_new),
            "dW": dW,
            "stock_price": self.S,
            "wealth": self.wealth,
            "trade_cost": trade_cost,
            "smoothness_pen": smoothness_pen,
        }

        return obs, reward, terminated, truncated, info
