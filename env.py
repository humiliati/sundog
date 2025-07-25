import mujoco
import mujoco.viewer
import numpy as np
from sundog.utils.sensors import estimate_shadow

class SundogEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.shadow_angle = 45  # degrees

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] += np.random.uniform(-0.01, 0.01, size=self.data.qpos.shape)
        mujoco.mj_forward(self.model, self.data)
        target = self.data.site("laser_dot").xpos
        tip = self.data.site("tip").xpos
        offset = target - tip
        self.data.qpos[6] += offset[1] * 0.05
        self.data.qpos[7] += offset[2] * 0.05
        self.data.qpos[6] += 0.02
        return self._get_obs()

    def step(self, ctrl):
        ctrl = np.clip(ctrl, -1, 1)
        self.data.ctrl[:] = ctrl[-2:]  # pitch1 and pitch2
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), self._compute_reward(), False, {}

    def _get_obs(self):
        tip_pos = self.data.site("tip").xpos.copy()
        laser_pos = self.data.site("laser_dot").xpos.copy()
        torque = self.data.qfrc_actuator.copy()
        shadow_pos = estimate_shadow(tip_pos, self.shadow_angle)
        spread = np.linalg.norm(tip_pos[:2] - laser_pos[:2])
        return {
            "tip_pos": tip_pos,
            "laser_pos": laser_pos,
            "torque": torque,
            "shadow_pos": shadow_pos,
            "bloom_spread": spread
        }

    def _compute_reward(self):
        tip_pos = self.data.site("tip").xpos
        laser_pos = self.data.site("laser_dot").xpos
        torque = self.data.qfrc_actuator
        alignment_error = np.linalg.norm(tip_pos - laser_pos)
        torque_magnitude = np.linalg.norm(torque)
        return -alignment_error + 0.1 * torque_magnitude

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
