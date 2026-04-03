# Christian Ishimwe – Medical Triage RL Summative

**Course**: Machine Learning tech II | African Leadership University  
**Author**: Christian Ishimwe  
**GitHub Repo**: `christian_ishimwe_rl_summative`

---

## Mission

Train an RL agent to act as an emergency triage coordinator, allocating limited hospital resources (beds, staff, equipment) across multiple simultaneous patients with varying severity levels (ESI 1–5), while patients deteriorate over time if left untreated.

---

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   ├── rendering.py         # Panda3D 3D visualization
│   └── __init__.py
├── training/
│   ├── dqn_training.py      # DQN – 10 hyperparameter experiments
│   └── pg_training.py       # REINFORCE, PPO, A2C – 10 experiments each
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved policy gradient models
├── results/                 # Training curves, plots, JSON results
├── main.py                  # Entry point – runs best model in simulation
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone repo
git clone https://github.com/<your-username>/christian_ishimwe_rl_summative
cd christian_ishimwe_rl_summative

# Create virtual environment (Python 3.10 recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running

### Static demo (random agent, no training required)
```bash
python main.py --random
```

### Train all algorithms (full 10-experiment grids)
```bash
python training/dqn_training.py          # DQN
python training/pg_training.py --algo reinforce
python training/pg_training.py --algo ppo
python training/pg_training.py --algo a2c
```

### Train best config (500k steps)
```bash
python training/dqn_training.py --best
python training/pg_training.py --best ppo
```

### Run best trained agent
```bash
python main.py                           # auto-selects best model
python main.py --algo ppo --episodes 10 # specific algorithm
python main.py --no-render               # terminal output only
```

---

## Environment Design

| Component | Details |
|---|---|
| **Observation space** | Box(43,) — 8 patients × 5 features + 3 resource scalars |
| **Action space** | Discrete(9) — hold or treat one of 8 patients |
| **Severity levels** | ESI 1 (Critical) → ESI 5 (Non-urgent) |
| **Resources** | 6 beds, 4 staff, 3 equipment units |
| **Deterioration** | Untreated patients worsen probabilistically each step |
| **Termination** | 3 deaths OR all patients cleared OR 200 steps |

### Reward Structure

| Event | Reward |
|---|---|
| Treat critical patient (ESI 1) | +10 |
| Treat moderate patient (ESI 2-3) | +5 |
| Treat minor patient (ESI 4-5) | +2 |
| Correct priority order | +2 bonus |
| Patient discharge | +3 |
| Patient death | -20 |
| Wrong priority / dead patient action | -3 |
| Idle with critical untreated patients | -1 per critical |

---

## Algorithms

| Algorithm | Type | Key Strength |
|---|---|---|
| DQN | Value-based | Experience replay, stable off-policy learning |
| REINFORCE | Policy gradient | Monte Carlo returns, simple baseline |
| PPO | Policy gradient | Clipped objective, sample efficient |
| A2C | Actor-Critic | Synchronous advantage estimation, fast updates |

---

## Results

After training, plots are saved to `results/`:
- `dqn_hyperparameter_analysis.png` — DQN experiment grid
- `pg_hyperparameter_analysis.png` — REINFORCE/PPO/A2C grids  
- `algorithm_comparison.png` — Cross-algorithm comparison

---

## Video Demo

See `demo_video.mp4` for the full screen recording with:
- Problem statement
- Agent behavior explanation
- Reward structure walkthrough
- Live Panda3D simulation with GUI + terminal verbose output
