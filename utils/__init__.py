"""
sundog.utils

Utility functions for the Sundog simulation:
- Shadow estimation from lateral light sources
- Torque signal smoothing and stability evaluation
"""

from .sensors import estimate_shadow, compute_torque_stability
