import numpy as np

class TorqueShadowAgent:
    def __init__(self):
        self.step_count = 0
        self.phase = 'navigate'  # phases: navigate -> align

    def act(self, obs):
        self.step_count += 1
        tip = obs['tip_pos']
        target = obs['laser_pos']
        torque = obs.get('torque', np.zeros(3))
        shadow = obs.get('shadow_pos', np.zeros(3))

        # Base and hinge control
        action = np.zeros(8)  # [x, y, z, rx, ry, rz, pitch1, pitch2]

        # PHASE 1: Navigate to center
        if self.phase == 'navigate':
            direction = target[:2] - tip[:2]
            if np.linalg.norm(direction) > 0.05:
                direction = direction / np.linalg.norm(direction) * 0.1
            action[0] = direction[0]
            action[1] = direction[1]
            if np.linalg.norm(direction) < 0.05:
                self.phase = 'align'

        # PHASE 2: Align for sundog triangulation
        elif self.phase == 'align':
            # Spin to induce bloom arc
            action[5] = 0.2 * np.sin(self.step_count * 0.1)
            # Pitching alignment to modulate bloom reflection
            alignment = target - shadow
            if np.linalg.norm(alignment) > 0:
                alignment = alignment / np.linalg.norm(alignment)
            action[6] = alignment[1] * 0.3  # pitch1
            action[7] = alignment[2] * 0.3  # pitch2

        return np.clip(action, -1, 1)
