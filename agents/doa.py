import numpy as np

class DirectObservationAgent:
    def __init__(self):
        self.step_count = 0
        self.phase = 'navigate'

    def act(self, obs):
        self.step_count += 1
        tip = obs['tip_pos']
        target = obs['laser_pos']

        # Control vector: [x, y, z, rx, ry, rz, pitch1, pitch2]
        action = np.zeros(8)

        # PHASE 1: Move base toward the laser
        if self.phase == 'navigate':
            direction = target[:2] - tip[:2]
            if np.linalg.norm(direction) > 0.05:
                direction = direction / np.linalg.norm(direction) * 0.1
            action[0] = direction[0]
            action[1] = direction[1]
            if np.linalg.norm(direction) < 0.05:
                self.phase = 'align'

        # PHASE 2: Align tip to the laser dot directly
        elif self.phase == 'align':
            offset = target - tip
            if np.linalg.norm(offset) > 0:
                offset = offset / np.linalg.norm(offset)
            action[6] = offset[1] * 0.3  # pitch1
            action[7] = offset[2] * 0.3  # pitch2

        return np.clip(action, -1, 1)
