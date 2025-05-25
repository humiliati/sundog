import numpy as np

def estimate_shadow(tip_pos, light_angle_deg=45):
    angle_rad = np.radians(light_angle_deg)
    offset = np.array([np.cos(angle_rad), np.sin(angle_rad), 0]) * 0.1
    return tip_pos + offset

def compute_torque_stability(torque_vector, window=5):
    recent = torque_vector[-window:]
    return -np.var(recent) if len(recent) >= window else 0
