"""
SAC (Soft Actor-Critic) training script for FinRL trading competition.

Why SAC over PPO?
- SAC is designed for continuous action spaces (buying/selling amounts)
- Automatically balances exploration vs exploitation via entropy tuning
- More sample-efficient than PPO
- Generally achieves better final performance in financial environments

Run from project root:
  python -m student_submission.train_sac --start_date 2013-01-01 --end_date 2018-12-31 --total_timesteps 300000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from test import load_market_dataframe, make_official_finrl_env


class ProgressCallback(BaseCallback):
    def __init__(self, print_every: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            print(f"  [Step {self.n_calls}] Training in progress...")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train a SAC trading agent.")

    parser.add_argument("--start_date", type=str, default="2013-01-01")
    parser.add_argument("--end_date", type=str, default="2018-12-31")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--finrl_repo_path", type=str, default=None)
    parser.add_argument("--data_file", type=str, default="train_data_2013_2018.csv")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--local_csv", type=str, default=None)

    # SAC hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent_coef", type=str, default="auto")
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--continue_from", type=str, default=None,
                         help="Path to an existing saved_agent.zip to continue training from.")

    args = parser.parse_args()

    # ── 1. Load data ───────────────────────────────────────────────────────
    print(f"\n[1/3] Loading market data: {args.start_date} ~ {args.end_date}")
    df = load_market_dataframe(
        split=args.split,
        data_file=args.data_file,
        local_csv=args.local_csv,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"      Loaded {len(df)} rows | {df['tic'].nunique()} tickers")

    # ── 2. Build environment ───────────────────────────────────────────────
    print("\n[2/3] Building trading environment...")
    env = make_official_finrl_env(df=df, finrl_repo_path=args.finrl_repo_path)

    # ── 3. Train SAC ───────────────────────────────────────────────────────
    print(f"\n[3/3] Training SAC for {args.total_timesteps:,} timesteps...")
    print(f"      Policy net: [256, 256] | lr={args.learning_rate} | "
          f"batch={args.batch_size} | ent_coef={args.ent_coef} | " 
          f"gamma={args.gamma} | seed={args.seed} | buffer={args.buffer_size}")

    # 256x256 network for both actor and critic
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
    )

    if args.continue_from:
        print(f"      Continuing training from: {args.continue_from}")
        model = SAC.load(args.continue_from, env=env, device="cpu")
        # Explicitly override gamma in case it differs from the saved model
        print(f"      Previous gamma: {model.gamma} -> New gamma: {args.gamma}")
        model.gamma = args.gamma
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            ent_coef=args.ent_coef,   # "auto" = automatic entropy tuning
            learning_starts=args.learning_starts,
            buffer_size=args.buffer_size,
            seed=args.seed,
            device="cpu",
        )

    callback = ProgressCallback(print_every=10000)
    model.learn(total_timesteps=args.total_timesteps, callback=callback,
                 reset_num_timesteps=(args.continue_from is None))

    # ── Save ───────────────────────────────────────────────────────────────
    save_path = Path("student_submission") / "saved_agent"
    model.save(save_path)
    print(f"\nSaved SAC agent to: {save_path}.zip")


if __name__ == "__main__":
    main()
