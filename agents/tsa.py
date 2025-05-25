import numpy as np

class TorqueShadowAgent:
    def act(self, obs):
        shadow = obs.get('shadow_pos', np.zeros(3))
        target = obs['laser_pos']
        torque = obs.get('torque', np.zeros(3))
        direction = target - shadow
        torque_factor = 0.05 * np.tanh(torque)
        return np.clip(direction + torque_factor, -1, 1)
