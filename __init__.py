"""
sundog

Core package for the Sundog Theorem simulation environment and agents.

This package provides:
- `SundogEnv` for simulating torque-shadow based alignment tasks
- Agent implementations (DirectObservationAgent, TorqueShadowAgent)
- Utility functions for shadow estimation and torque stability

Usage:
    from sundog.env import SundogEnv
    from sundog.agents.doa import DirectObservationAgent
    from sundog.agents.tsa import TorqueShadowAgent
"""

# You can optionally expose high-level accessors
from .env import SundogEnv
from .agents.doa import DirectObservationAgent
from .agents.tsa import TorqueShadowAgent
