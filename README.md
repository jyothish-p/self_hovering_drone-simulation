# RLT Project — Reinforcement Learning Quadcopter

A staged autonomous flight system using Deep Q-Networks (DQN) for quadcopter hover training.

## Architecture

```
RLT Project/
├── rl-training.html      ← Main training UI  (open this to train)
├── drone-demo.html       ← Interactive drone demo (manual control)
│
├── agent/
│   └── dqn-agent.js      ← DQNAgent (256-256-128-64 network, 16-state, 17-action)
│                            RLEnvironment (orientation-first rewards, 35° reset)
│
├── drone/
│   ├── drone.js          ← DroneModel — Cannon.js physics, motor dynamics
│   └── drone-scene.js    ← DroneScene — Three.js 3D visualization
│
├── training/
│   ├── rl-main.js        ← RL training loop (Stage 1 PD takeoff → Stage 2 DQN)
│   └── drone-main.js     ← Manual drone demo bootstrap
│
├── visualization/
│   └── rl-dashboard.html ← Real-time analytics dashboard (open separately)
│
├── models/               ← Saved model weights (exported via S key in training)
│   └── weights_export.js ← Utility to export/import IndexedDB weights
│
├── logs/                 ← Training telemetry and episode logs
│   └── .gitkeep
│
├── environment/          ← Environment configuration notes
│   └── CONFIG.md
│
└── config/
    └── cannon.js         ← Cannon.js physics engine (local copy)
```

## Quick Start

```bash
# Start local HTTP server from project root
cd "RLT Project"
python3 -m http.server 7892

# Open training:  http://localhost:7892/rl-training.html
# Open dashboard: http://localhost:7892/visualization/rl-dashboard.html
```

## Key Controls (Training UI)
| Key | Action |
|-----|--------|
| `T` | Toggle training on/off |
| `R` | Reset episode |
| `S` | Save model weights to browser IndexedDB |
| `L` | Load saved model weights |
| `C` | Cycle camera mode (Follow / Orbit / Top) |
| Scroll | Zoom |

## Staged Flight System
- **Stage 1** — Autonomous PD takeoff: drone climbs from ground to 10 m keeping level orientation
- **Stage 2** — RL hover training: DQN agent learns to maintain stable hover indefinitely

## Network Architecture
- **Input**: 16 states (position, velocity, orientation, angular velocity, motor thrusts)
- **Hidden**: 256 → 256 → 128 → 64 (ReLU)
- **Output**: 17 Q-values (symmetry-aware actions)
- **Algorithm**: Double DQN, γ=0.995, lr=0.0005, buffer=200K, batch=128

## Reward Design (Orientation-First)
- +50 orientation primary (≤2°), +20 good (≤5°)
- −100 severe tilt (>15°), −200 extreme (>25°)
- +30 altitude precise (±0.2m), +10 moderate (±0.5m)
- +20 symmetry balanced, −25 asymmetric torque
- −300 crash terminal penalty
