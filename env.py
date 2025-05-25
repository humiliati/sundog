import mujoco
import mujoco.viewer
import numpy as np
from utils.sensors import estimate_shadow

class SundogEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.shadow_angle = 45  # degrees

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, ctrl):
        ctrl = np.clip(ctrl, -1, 1)
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), self._compute_reward(), False, {}

    def _get_obs(self):
        tip_pos = self.data.site("tip").xpos.copy()
        laser_pos = self.data.site("laser_dot").xpos.copy()
        torque = self.data.qfrc_actuator.copy()
        shadow_pos = estimate_shadow(tip_pos, self.shadow_angle)
        return {
            "tip_pos": tip_pos,
            "laser_pos": laser_pos,
            "torque": torque,
            "shadow_pos": shadow_pos
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
