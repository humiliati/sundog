import numpy as np

class DirectObservationAgent:
    def act(self, obs):
        tip = obs['tip_pos']
        target = obs['laser_pos']
        direction = target - tip
        return np.clip(direction[:3], -1, 1)
