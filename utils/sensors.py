import numpy as np

def estimate_shadow(tip_pos, light_angle_deg=45):
    angle_rad = np.radians(light_angle_deg)
    offset = np.array([np.cos(angle_rad), np.sin(angle_rad), 0]) * 0.1
    return tip_pos + offset

def compute_torque_stability(torque_vector, window=5):
    recent = torque_vector[-window:]
    return -np.var(recent) if len(recent) >= window else 0

class BloomTracker:
    def __init__(self, max_len=50):
        self.positions = []
        self.max_len = max_len

    def update(self, tip_pos):
        self.positions.append(np.array(tip_pos))
        if len(self.positions) > self.max_len:
            self.positions.pop(0)

    def get_center(self):
        if not self.positions:
            return np.zeros(3)
        return np.mean(self.positions, axis=0)

    def get_spread(self):
        if len(self.positions) < 2:
            return 0
        return np.linalg.norm(np.std(self.positions, axis=0))
