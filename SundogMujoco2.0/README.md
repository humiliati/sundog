# Sundog Leisure: Embodied Resonance Through Tinkering

An extension of the Sundog Alignment Theorem for agents that experience **leisure** rather than struggle. This environment solves the classic problem of embodied agents spending all resources on balance/standing by making stability trivially cheap.

## Core Insight

The original Sundog theorem demonstrated that alignment emerges from **indirect feedback** through the halo signature:

$$H(x) = \frac{\partial S}{\partial \tau}$$

Where $S$ is shadow geometry and $\tau$ is torque. This environment extends that insight to a leisure context where:

1. **Stability is free**: The agent lies recumbent (on its back), eliminating balance costs
2. **Interaction is rich**: Pushable blocks create complex shadow patterns on the hexagonal ceiling
3. **Resonance emerges from play**: The agent discovers alignment through tinkering, not optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HEXAGONAL LATTICE CEILING                │
│   ○ ○ ○ ○ ○     Overlapping spheres create bloom collapse   │
│    ○ ○ ○ ○ ○    points where light converges naturally      │
│   ○ ○ ○ ○ ○                                                 │
└─────────────────────────────────────────────────────────────┘
                           ↑ light/shadow
                           
     ┌─────┐  ┌─────┐  ┌─────┐     PLAY BLOCKS
     │     │  │     │  │     │     Cast shadows, pushable
     └─────┘  └─────┘  └─────┘     
                           
    ╭───────╮                      RECUMBENT AGENT  
   ─┤ HEAD  ├─    ←sensor tip      Lying on back, no balance
    ╰───┬───╯                      needed, arms free to tinker
        │
   ╭────┴────╮
  ─┤  TORSO  ├─
   ╰─┬─────┬─╯
     │     │
   ╭─┴─╮ ╭─┴─╮    ARMS
   │   │ │   │    For pushing blocks
   ╰─┬─╯ ╰─┬─╯    Torque feedback for H(x)
     │     │
   ╭─┴─╮ ╭─┴─╮    PADDLES/HANDS
   │   │ │   │    Contact surfaces
   ╰───╯ ╰───╯
```

## Key Components

### Environment: `LeisureEnv`

- Hexagonal lattice ceiling (~2900 overlapping spheres)
- 8-10 pushable blocks of varying sizes
- Upward light source casting shadows on ceiling
- Recumbent agent with 7 DOF (head + 2 arms)

### Agents

| Agent | Strategy | Use Case |
|-------|----------|----------|
| `LeisureAgent` | Mode-switching (explore/curious/rest) | Main agent - learns through indirect feedback |
| `TorqueShadowLeisureAgent` | Uses H(x) halo signature | Tests torque-shadow coupling |
| `RandomLeisureAgent` | Pure noise | Baseline |
| `ZenLeisureAgent` | Zero action | Tests natural convergence |

### Sensors

- `BloomTracker`: Tracks shadow spread and collapse signatures
- `ResonanceDetector`: Identifies sustained resonance states
- `HaloSignatureTracker`: Computes $H(x) = \partial S / \partial \tau$
- `ProprioceptiveState`: Lightweight joint state tracking

## Installation

```bash
# Basic (minimal environment, no rendering)
pip install -e .

# With MuJoCo support
pip install -e ".[mujoco]"

# With visualization
pip install -e ".[mujoco,viz]"
```

## Usage

### Quick Test
```python
from sundog.env import LeisureEnvMinimal
from sundog.agents import LeisureAgent

env = LeisureEnvMinimal(num_blocks=8)
agent = LeisureAgent(action_dim=7)

obs = env.reset()
for _ in range(100):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    print(f"Resonance: {info['resonance']:.3f}")
```

### Training
```bash
# Single agent
python scripts/train_leisure.py --agent leisure --episodes 50

# Compare all agents
python scripts/train_leisure.py --agent compare --episodes 20

# With MuJoCo rendering
python scripts/train_leisure.py --agent leisure --use-mujoco --episodes 10
```

### Generate Custom Environment
```python
from env.generate_leisure_env import generate_leisure_env_xml

generate_leisure_env_xml(
    ceiling_radius=2.0,      # Larger = more ceiling spheres
    ceiling_height=1.2,      # Lower = more shadow detail
    sphere_spacing=0.06,     # Smaller = more overlap (denser lattice)
    sphere_radius=0.05,
    num_blocks=12,           # More blocks = richer interaction
    output_path='custom_env.xml'
)
```

## The Leisure Principle

Traditional embodied AI puts agents in environments where they must:
1. Fight gravity constantly (standing)
2. Spend energy on stability before doing anything useful
3. Optimize directly for task rewards

The Leisure Environment inverts this:
1. **Gravity is a toy, not a constraint** - blocks can be pushed, physics can be played with
2. **Stability is free** - lying down requires no effort
3. **Resonance emerges from structured play** - the agent discovers alignment patterns through tinkering

This mirrors the Sundog insight: alignment is not about direct optimization, but about resonance with structured environmental constraints.

## Theoretical Connection

From the original Sundog theorem:

> "Alignment occurs when a system listens to itself through its interactions with the structured chaos of its environment."

In the Leisure context:
- **Listening** = observing bloom patterns from block shadows
- **Structured chaos** = hexagonal ceiling creates natural attractor points
- **Interaction** = pushing blocks modulates the shadow field

When $H(x) \neq 0$ (halo signature is non-zero), the system has information about how its actions affect the shadow geometry. This is the basis for indirect alignment.

## Project Structure

```
sundog_leisure/
├── env/
│   ├── generate_leisure_env.py   # XML generator
│   └── sundog_leisure.xml        # Generated environment
├── sundog/
│   ├── env.py                    # Environment classes
│   ├── agents/
│   │   └── leisure_agent.py      # Agent implementations
│   └── utils/
│       └── sensors.py            # Bloom tracking, resonance detection
├── scripts/
│   └── train_leisure.py          # Training entrypoint
└── results/                      # Output logs
```

## Future Directions

1. **Multi-agent leisure**: Multiple agents tinkering together
2. **Learned halo estimation**: Neural network to predict $H(x)$
3. **Curriculum from chaos to order**: Gradually introduce structure
4. **Transfer to standing agents**: Can leisure-learned patterns help?

## Citation

If you use this work, please cite:

```bibtex
@article{sundog2024,
  title={The Sundog Alignment Theorem: Embodied Resonance and Indirect Inference},
  author={...},
  year={2024}
}
```

## Rights

Copyright (c) 2026 Stellar Aqua LLC. All rights reserved.

This historical prototype is included for repository inspection only unless a
file or signed agreement states otherwise. Public availability does not grant a
new MIT license, patent license, trademark license, commercial license, hosting
license, training-data license, or derivative-work license.
