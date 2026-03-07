"""
Low-dimensional observation wrapper for the Box Touch environment.

Observations are state vectors containing:
  - Robot joint positions (7)
  - Robot joint velocities (7)
  - End-effector position (3)
  - Box position (3)
  - Box size (3)
  Total: 23 dimensions

Following the pattern of kitchen_lowdim_wrapper.py.
"""

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from diffusion_policy.env.box_touch.box_touch_env import BoxTouchEnv


class BoxTouchLowdimWrapper(gym.Env):
    """
    Wrapper that provides low-dimensional (state) observations for the
    Box Touch environment.
    
    Observation: np.array of shape (23,) containing:
        [joint_pos(7), joint_vel(7), ee_pos(3), box_pos(3), box_size(3)]
    
    Action: np.array of shape (7,) — target joint positions.
    """
    
    def __init__(
        self,
        env: Optional[BoxTouchEnv] = None,
        render_hw: tuple = (240, 360),
        **kwargs,
    ):
        """
        Args:
            env: An existing BoxTouchEnv instance. If None, creates a new one.
            render_hw: (height, width) for rendering.
            **kwargs: Additional arguments passed to BoxTouchEnv if env is None.
        """
        if env is None:
            env = BoxTouchEnv(render_hw=render_hw, **kwargs)
        self.env = env
        self.render_hw = render_hw
        
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def reset(self):
        """Reset and return low-dimensional observation."""
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        """Step and return low-dimensional observation."""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        """Render the environment (for visualization/video recording)."""
        h, w = self.render_hw
        self.env.render_hw = (h, w)
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()
    
    def get_state(self):
        return self.env.get_state()
    
    def set_state(self, state):
        self.env.set_state(state)
