import numpy as np

class DirectObservationAgent:
    def __init__(self):
        self.step_count = 0

    def act(self, obs):
        self.step_count += 1

        tip = obs['tip_pos']
        target = obs['laser_pos']
        direction = target - tip
        distance = np.linalg.norm(direction)

        # Normalize direction for gentle forward movement
        if distance > 0:
            forward = (direction / distance)[:3] * 0.2
        else:
            forward = np.zeros(3)

        # Yawing motion on base (oscillates between -0.3 and 0.3)
        yaw_oscillation = 0.3 * np.sin(self.step_count * 0.1)

        # Blend: [yaw, pitch1, pitch2]
        action = np.array([yaw_oscillation, forward[1], forward[2]])
        return np.clip(action, -1, 1)
