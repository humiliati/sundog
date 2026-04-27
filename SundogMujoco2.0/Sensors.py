"""
Sensors and Trackers for Sundog Leisure Environment

Implements the bloom tracking and resonance detection from the
original Sundog theorem, adapted for the leisure/tinkering context.
"""

import numpy as np
from collections import deque


class BloomTracker:
    """
    Tracks the "bloom" - the spread of light/shadow patterns.
    
    In the original Sundog, bloom collapse indicates alignment.
    Here, we track bloom from block shadows on the hexagonal ceiling.
    """
    
    def __init__(self, max_len=50):
        self.positions = deque(maxlen=max_len)
        self.timestamps = deque(maxlen=max_len)
        
    def update(self, position, timestamp=None):
        """Add new position to tracker."""
        self.positions.append(np.array(position))
        self.timestamps.append(timestamp if timestamp else len(self.positions))
        
    def get_center(self):
        """Get centroid of tracked positions."""
        if len(self.positions) == 0:
            return np.zeros(3)
        return np.mean(self.positions, axis=0)
    
    def get_spread(self):
        """Get spread (std of distances from center)."""
        if len(self.positions) < 2:
            return 0.0
        center = self.get_center()
        distances = [np.linalg.norm(p - center) for p in self.positions]
        return np.std(distances)
    
    def get_velocity(self):
        """Get average velocity of bloom center."""
        if len(self.positions) < 2:
            return np.zeros(3)
        positions = np.array(self.positions)
        velocities = np.diff(positions, axis=0)
        return np.mean(velocities, axis=0)
    
    def get_collapse_signature(self):
        """
        Compute the collapse signature - rate of spread decrease.
        
        Returns:
            signature: Positive = collapsing (good), Negative = expanding
        """
        if len(self.positions) < 10:
            return 0.0
        
        positions = np.array(self.positions)
        center = np.mean(positions, axis=0)
        
        # Compute spread over time
        spreads = []
        window = 5
        for i in range(len(positions) - window):
            subset = positions[i:i+window]
            local_center = np.mean(subset, axis=0)
            dists = np.linalg.norm(subset - local_center, axis=1)
            spreads.append(np.std(dists))
        
        if len(spreads) < 2:
            return 0.0
        
        # Rate of change of spread (negative = collapsing)
        spread_velocity = np.diff(spreads)
        return -np.mean(spread_velocity)  # Return positive for collapse
    
    def clear(self):
        """Reset tracker."""
        self.positions.clear()
        self.timestamps.clear()


class ResonanceDetector:
    """
    Detects resonance states in the environment.
    
    Resonance occurs when:
    1. Bloom is collapsed (low spread)
    2. System is stable (low velocity)
    3. Pattern is coherent (high spatial autocorrelation)
    """
    
    def __init__(self, threshold=0.5, window=20):
        self.threshold = threshold
        self.window = window
        self.history = deque(maxlen=window)
        
    def update(self, obs):
        """
        Update detector with new observation.
        
        Returns:
            resonance_score: 0 to 1, where 1 = perfect resonance
        """
        # Extract relevant features
        block_positions = obs.get('block_positions', np.zeros((1, 3)))
        resonance_point = obs.get('resonance_point', np.zeros(3))
        
        # Compute bloom metrics
        if len(block_positions) > 0:
            centroid = np.mean(block_positions[:, :2], axis=0)
            distances = np.linalg.norm(block_positions[:, :2] - centroid, axis=1)
            spread = np.std(distances)
            center_error = np.linalg.norm(centroid - resonance_point[:2])
        else:
            spread = 1.0
            center_error = 1.0
        
        # Resonance score (inverse of spread + error)
        score = 1.0 / (1.0 + spread + 0.5 * center_error)
        
        self.history.append(score)
        
        return score
    
    def is_resonant(self):
        """Check if system is in sustained resonance."""
        if len(self.history) < self.window // 2:
            return False
        
        recent = list(self.history)[-self.window//2:]
        return np.mean(recent) > self.threshold and np.std(recent) < 0.1
    
    def get_resonance_stats(self):
        """Get statistics about resonance history."""
        if len(self.history) == 0:
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
        
        h = np.array(self.history)
        return {
            'mean': np.mean(h),
            'std': np.std(h),
            'max': np.max(h),
            'min': np.min(h),
            'is_resonant': self.is_resonant()
        }


class HaloSignatureTracker:
    """
    Tracks the H(x) = ∂S/∂τ halo signature.
    
    This is the key insight from Sundog: the derivative of shadow
    with respect to torque indicates alignment potential.
    """
    
    def __init__(self, window=30):
        self.window = window
        self.shadow_history = deque(maxlen=window)  # Bloom spread
        self.torque_history = deque(maxlen=window)
        self.halo_history = deque(maxlen=window)
        
    def update(self, shadow_spread, torque):
        """
        Update with new shadow and torque measurements.
        
        Args:
            shadow_spread: Current bloom spread
            torque: Current joint torques (can be scalar or vector)
        """
        torque_mag = np.linalg.norm(torque) if hasattr(torque, '__len__') else abs(torque)
        
        self.shadow_history.append(shadow_spread)
        self.torque_history.append(torque_mag)
        
        # Compute H(x) if we have history
        if len(self.shadow_history) >= 2:
            ds = self.shadow_history[-1] - self.shadow_history[-2]
            dt = self.torque_history[-1] - self.torque_history[-2]
            
            if abs(dt) > 1e-6:
                halo = ds / dt
            else:
                halo = 0.0
                
            self.halo_history.append(halo)
        
    def get_halo(self):
        """Get current halo signature."""
        if len(self.halo_history) == 0:
            return 0.0
        return self.halo_history[-1]
    
    def get_mean_halo(self):
        """Get mean halo over recent history."""
        if len(self.halo_history) == 0:
            return 0.0
        return np.mean(self.halo_history)
    
    def is_aligned(self, threshold=0.1):
        """
        Check if halo signature indicates alignment.
        
        When H(x) ≠ 0, alignment is possible (from theorem).
        """
        recent = list(self.halo_history)[-10:] if len(self.halo_history) >= 10 else list(self.halo_history)
        if len(recent) == 0:
            return False
        return np.mean(np.abs(recent)) > threshold


class ProprioceptiveState:
    """
    Lightweight proprioceptive state tracker.
    
    Keeps track of minimal body state without expensive computation.
    """
    
    def __init__(self, joint_names=None):
        self.joint_names = joint_names or [
            'head_pitch', 
            'left_shoulder_yaw', 'left_elbow_pitch', 'left_wrist',
            'right_shoulder_yaw', 'right_elbow_pitch', 'right_wrist'
        ]
        self.positions = np.zeros(len(self.joint_names))
        self.velocities = np.zeros(len(self.joint_names))
        self.last_positions = None
        
    def update(self, joint_positions, dt=0.01):
        """Update state from sensor readings."""
        self.positions = np.array(joint_positions)
        
        if self.last_positions is not None:
            self.velocities = (self.positions - self.last_positions) / dt
        
        self.last_positions = self.positions.copy()
        
    def get_energy(self):
        """Get kinetic energy estimate."""
        return 0.5 * np.sum(self.velocities ** 2)
    
    def is_still(self, threshold=0.1):
        """Check if agent is approximately still."""
        return self.get_energy() < threshold
    
    def get_state_vector(self):
        """Get flattened state for learning."""
        return np.concatenate([self.positions, self.velocities])