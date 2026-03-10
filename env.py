"""
Gymnasium-compliant Deep Hedging Environment.

Simulates delta-hedging a European call option under GBM dynamics
with proportional transaction costs. The agent learns a hedging
policy that minimises P&L variance (risk-averse reward).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DerivativeHedgingEnv(gym.Env):
    """
    Observation (5-dim, continuous):
        0  current stock price  S_t
        1  current hedge ratio   delta_t
        2  time remaining        tau  (T - t)
        3  current option value  V_t  (Black-Scholes)
        4  current wealth        W_t

    Action (1-dim, continuous):
        new hedge ratio in [0, 1]

    Reward:
        -( dW^2 )  — quadratic penalty on the change in total wealth,
        encouraging the agent to keep wealth changes (hedging errors)
        close to zero.
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
        transaction_cost_bps: float = 1.0,
        risk_penalty: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()

        # --- market / contract parameters ---
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.tc = transaction_cost_bps * 1e-4  # proportional cost
        self.risk_penalty = risk_penalty

        # --- spaces ---
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        obs_low = np.array([0.0, 0.0, 0.0, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([np.inf, 1.0, float(T), np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # will be set in reset()
        self._rng: np.random.Generator | None = None
        self._step_idx: int = 0
        self.S: float = S0
        self.delta: float = 0.0
        self._cash: float = 0.0   # cash account
        self.wealth: float = 0.0  # net wealth = cash + delta*S − V

    # ------------------------------------------------------------------
    # Black-Scholes helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        tau = self.T - self._step_idx * self.dt
        V = self._bs_call(self.S, tau)
        return np.array(
            [self.S, self.delta, tau, V, self.wealth], dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed if seed is not None else None)

        self.S = self.S0
        self.delta = 0.0
        self._step_idx = 0

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

        # --- 1. rebalance: buy/sell stock, pay transaction costs ---
        trade_size = new_delta - old_delta
        trade_cost = self.tc * abs(trade_size) * old_S
        self._cash -= trade_size * old_S + trade_cost
        self.delta = new_delta

        # --- 2. evolve the stock price (GBM) ---
        z = self._rng.standard_normal()
        self.S = old_S * np.exp(
            (self.r - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )

        # --- 3. accrue interest on cash ---
        self._cash *= 1.0 + self.r * self.dt

        # --- 4. update net wealth = cash + stock − option ---
        self._step_idx += 1
        tau_new = self.T - self._step_idx * self.dt
        V_new = self._bs_call(self.S, tau_new)

        old_wealth = self.wealth
        self.wealth = self._cash + self.delta * self.S - V_new
        dW = self.wealth - old_wealth

        # --- 5. reward: penalise squared hedging error ---
        reward = float(-self.risk_penalty * dW**2)

        # --- 6. termination ---
        terminated = self._step_idx >= self.n_steps
        truncated = False

        obs = self._get_obs()
        info = {
            "bs_delta": self._bs_delta(self.S, tau_new),
            "dW": dW,
            "stock_price": self.S,
            "wealth": self.wealth,
            "trade_cost": trade_cost,
        }

        return obs, reward, terminated, truncated, info
