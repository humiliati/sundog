import numpy as np

class TorqueShadowAgent:
    def __init__(self):
        self.step_count = 0
        self.phase = 'navigate'

    def act(self, obs):
        self.step_count += 1
        tip = obs['tip_pos']
        target = obs['laser_pos']
        torque = obs.get('torque', np.zeros(3))
        shadow = obs.get('shadow_pos', np.zeros(3))
        spread = obs.get('bloom_spread', 1.0)

        action = np.zeros(8)  # [x, y, z, rx, ry, rz, pitch1, pitch2]

        # Phase transition if bloom spread collapses
        if self.phase == 'navigate' and spread < 0.05:
            self.phase = 'align'

        if self.phase == 'navigate':
            direction = target[:2] - tip[:2]
            if np.linalg.norm(direction) > 0.05:
                direction = direction / np.linalg.norm(direction) * 0.1
            action[0] = direction[0]
            action[1] = direction[1]

        elif self.phase == 'align':
            # Oscillate yaw for triangulation
            spin_strength = max(0.1, spread)  # reduce spin when stable
            action[5] = spin_strength * np.sin(self.step_count * 0.1)

            # Horizontal alignment
            flat_align = target - shadow
            if np.linalg.norm(flat_align) > 0:
                flat_align = flat_align / np.linalg.norm(flat_align)
            action[6] = flat_align[1] * 0.3  # pitch1

            # Vertical z alignment (laser above tip)
            z_offset = target[2] - tip[2]
            action[7] = np.clip(z_offset * 0.3, -1, 1)  # pitch2

        return np.clip(action, -1, 1)
