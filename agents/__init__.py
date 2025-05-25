"""
sundog.agents

Contains agent policy implementations for the Sundog Theorem environment:
- DirectObservationAgent: aligns to the target using direct tip-laser position.
- TorqueShadowAgent: aligns using shadow projection and torque feedback.
"""

from .doa import DirectObservationAgent
from .tsa import TorqueShadowAgent
