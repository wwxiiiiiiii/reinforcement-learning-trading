# reinforcement-learning-trading
A reinforcement learning project for portfolio management using Soft Actor-Critic and Stable-Baselines3

---

## Project Overview

This project develops an automated stock trading agent using reinforcement learning.

Instead of maximizing performance on a single evaluation period, the primary objective was to improve the model's generalization ability so that it performs consistently across different market environments.

The final model was implemented using the Soft Actor-Critic (SAC) algorithm with Stable-Baselines3.

---

## Problem Definition

The initial trading model showed a tendency to overfit the training period. Although it achieved good performance under certain market conditions, its returns varied significantly when evaluated on unseen periods.

The objective of this project was to improve the robustness of the trading policy through systematic model selection and hyperparameter optimization.

---

## Methodology

### Algorithm Comparison

Several reinforcement learning algorithms were compared.

- PPO (Proximal Policy Optimization)
- TQC (Truncated Quantile Critics)
- SAC (Soft Actor-Critic)

After multiple experiments, SAC was selected because it provided more stable learning and superior performance in continuous portfolio allocation tasks.

---

### Model Optimization

The following hyperparameters were systematically tuned.

- Policy network architecture
- Training timesteps
- Discount factor (gamma)
- Batch size
- Replay buffer size
- Random seeds

To increase the model capacity, the policy network was expanded to:

```
256 × 256
```

Training was continued from saved checkpoints to efficiently perform long-running experiments.

---

## Implementation

The project was implemented in Python using:

- Stable-Baselines3
- PyTorch
- NumPy
- Pandas
- FinRL

The training pipeline supports checkpoint-based continual learning, allowing interrupted training sessions to resume without losing previous progress.

---

## Evaluation

Performance was evaluated on multiple unseen market periods using:

- Cumulative Return
- Sharpe Ratio
- Maximum Drawdown

Instead of optimizing for a single test period, the model was validated across various historical market environments to assess its robustness.

---

## Results

Compared with the initial PPO baseline, the final SAC agent demonstrated:

- Higher cumulative returns
- Improved risk-adjusted performance (Sharpe Ratio)
- More stable performance across multiple unseen market periods
- Better generalization through systematic hyperparameter optimization

---

## Technologies

- Python
- Stable-Baselines3
- PyTorch
- FinRL
- NumPy
- Pandas

---

## Repository Structure

```
reinforcement-learning-trading/
│
├── README.md
├── train_sac.py
├── submission.py
├── metadata.json
├── requirements.txt
└── LICENSE
```

---

## Author

**Mijin Son**

Department of Statistics

This repository was created as part of a reinforcement learning project focused on portfolio optimization and financial AI.
