"""
Train a PPO agent on DerivativeHedgingEnv using Stable-Baselines3.

Usage:
    python train_ppo.py              # train with defaults
    python train_ppo.py --timesteps 500000 --save-path ppo_hedging
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import DerivativeHedgingEnv


def make_env(**kwargs):
    def _init():
        return DerivativeHedgingEnv(**kwargs)
    return _init


def train(
    total_timesteps: int = 2_000_000,
    save_path: str = "ppo_hedging",
    n_envs: int = 8,
    **env_kwargs,
) -> PPO:
    """Train a PPO agent and save the model + normalisation stats."""
    vec_env = DummyVecEnv([make_env(**env_kwargs) for _ in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(save_path)
    vec_env.save(f"{save_path}_vecnormalize.pkl")
    print(f"Model saved to {save_path}.zip")
    print(f"VecNormalize stats saved to {save_path}_vecnormalize.pkl")

    vec_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--save-path", type=str, default="ppo_hedging")
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, save_path=args.save_path)
