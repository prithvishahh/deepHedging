"""
Black-Scholes Delta Hedging benchmark.

Runs the DerivativeHedgingEnv using the analytical BS delta as the action
at every step.  Returns per-episode cumulative P&L curves and the final
P&L distribution so it can be compared against an RL agent.
"""

import numpy as np
from env import DerivativeHedgingEnv


def run_bs_hedge(
    n_episodes: int = 10_000,
    seed: int = 42,
    **env_kwargs,
) -> dict:
    """Run Black-Scholes delta hedging over *n_episodes* independent paths.

    Parameters
    ----------
    n_episodes : int
        Number of Monte-Carlo episodes to simulate.
    seed : int
        Base random seed (each episode uses seed + i).
    **env_kwargs
        Forwarded to ``DerivativeHedgingEnv`` (S0, K, sigma, …).

    Returns
    -------
    dict with keys
        "final_pnl"        : np.ndarray (n_episodes,)  — terminal wealth change
        "cumulative_pnl"   : np.ndarray (n_episodes, n_steps+1) — wealth at each step
        "mean_pnl"         : float
        "std_pnl"          : float
        "mean_reward"      : float  — average total (sum of step rewards)
        "total_costs"      : np.ndarray (n_episodes,) — cumulative transaction costs
    """
    env = DerivativeHedgingEnv(**env_kwargs)
    n_steps = env.n_steps

    final_pnl = np.empty(n_episodes)
    cumulative_pnl = np.empty((n_episodes, n_steps + 1))
    total_costs = np.empty(n_episodes)
    total_rewards = np.empty(n_episodes)

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        cumulative_pnl[i, 0] = obs[5]  # wealth starts at 0

        ep_reward = 0.0
        ep_cost = 0.0

        for t in range(n_steps):
            # use BS delta from the info dict as the action
            bs_delta = info["bs_delta"]
            action = np.array([bs_delta], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_cost += info["trade_cost"]
            # net weatth
            cumulative_pnl[i, t + 1] = obs[5]

        final_pnl[i] = obs[5]
        total_costs[i] = ep_cost
        total_rewards[i] = ep_reward

    return {
        "final_pnl": final_pnl,
        "cumulative_pnl": cumulative_pnl,
        "mean_pnl": float(np.mean(final_pnl)),
        "std_pnl": float(np.std(final_pnl)),
        "mean_reward": float(np.mean(total_rewards)),
        "total_costs": total_costs,
    }


# Quick stand-alone run with summary statistics
if __name__ == "__main__":
    results = run_bs_hedge(n_episodes=10_000, seed=0)

    print("=== Black-Scholes Delta Hedge Benchmark ===")
    print(f"Episodes       : {len(results['final_pnl'])}")
    print(f"Mean final P&L : {results['mean_pnl']:+.4f}")
    print(f"Std  final P&L : {results['std_pnl']:.4f}")
    print(f"Mean reward    : {results['mean_reward']:.4f}")
    print(f"Mean tx cost   : {np.mean(results['total_costs']):.4f}")

    # histogram
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(results["final_pnl"], bins=80, edgecolor="k", alpha=0.7)
        axes[0].axvline(0, color="r", ls="--")
        axes[0].set_title("Final P&L Distribution (BS Delta Hedge)")
        axes[0].set_xlabel("P&L")
        axes[0].set_ylabel("Count")

        # plot a handful of cumulative wealth paths
        for j in range(min(50, len(results["cumulative_pnl"]))):
            axes[1].plot(results["cumulative_pnl"][j], alpha=0.3, lw=0.8)
        axes[1].set_title("Cumulative Wealth Paths (sample)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Wealth")

        plt.tight_layout()
        plt.savefig("bs_benchmark.png", dpi=150)
        print("Plot saved to bs_benchmark.png")
    except ImportError:
        print("matplotlib not found — skipping plots.")
