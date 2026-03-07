"""
Image observation wrapper for the Box Touch environment.

Observations are dictionaries containing:
  - 'image': RGB image from camera (H, W, 3) float32 in [0, 1]
  - 'agent_pos': Robot joint positions (7,) float32

Following the pattern of robomimic_image_wrapper.py.
"""

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from diffusion_policy.env.box_touch.box_touch_env import BoxTouchEnv, N_ARM_JOINTS


class BoxTouchImageWrapper(gym.Env):
    """
    Wrapper that provides image-based observations for the Box Touch environment.
    
    Observation: dict with keys:
        'image': np.array of shape (H, W, 3) float32 in [0, 1]
        'agent_pos': np.array of shape (7,) float32 — joint positions
    
    Action: np.array of shape (7,) — target joint positions.
    """
    
    def __init__(
        self,
        env: Optional[BoxTouchEnv] = None,
        shape_meta: Optional[dict] = None,
        render_hw: tuple = (240, 360),
        camera_name: str = 'front_camera',
        render_obs_key: str = 'image',
        **kwargs,
    ):
        """
        Args:
            env: An existing BoxTouchEnv instance. If None, creates a new one.
            shape_meta: Optional dict describing observation/action shapes.
                If provided, used to define observation_space.
            render_hw: (height, width) for rendering.
            camera_name: Name of the MuJoCo camera for image observations.
            render_obs_key: Key in observation dict used for rendering.
            **kwargs: Additional arguments passed to BoxTouchEnv if env is None.
        """
        if env is None:
            env = BoxTouchEnv(render_hw=render_hw, **kwargs)
        self.env = env
        self.render_hw = render_hw
        self.camera_name = camera_name
        self.render_obs_key = render_obs_key
        self.render_cache = None
        
        h, w = render_hw
        
        # Setup spaces
        if shape_meta is not None:
            # Use shape_meta to define spaces (for compatibility with diffusion_policy)
            action_shape = shape_meta['action']['shape']
            self.action_space = spaces.Box(
                low=-1, high=1, shape=action_shape, dtype=np.float32
            )
            
            observation_space = spaces.Dict()
            for key, value in shape_meta['obs'].items():
                shape = value['shape']
                if 'image' in key:
                    min_val, max_val = 0, 1
                else:
                    min_val, max_val = -np.inf, np.inf
                observation_space[key] = spaces.Box(
                    low=min_val, high=max_val, shape=shape, dtype=np.float32
                )
            self._observation_space = observation_space
        else:
            # Default spaces
            self.action_space = self.env.action_space
            
            observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0, high=1, shape=(h, w, 3), dtype=np.float32
                ),
                'agent_pos': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(N_ARM_JOINTS,), dtype=np.float32
                ),
            })
            self._observation_space = observation_space
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def _get_image_obs(self):
        """Render camera image and return as float32 in [0, 1]."""
        h, w = self.render_hw
        self.env.render_hw = (h, w)
        img = self.env.render(mode='rgb_array', camera_name=self.camera_name)
        # Convert to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        return img
    
    def _get_obs(self):
        """Get image-based observation dict."""
        img = self._get_image_obs()
        self.render_cache = img
        
        joint_pos = self.env.physics.data.qpos[:N_ARM_JOINTS].copy().astype(np.float32)
        
        obs = {
            'image': img,
            'agent_pos': joint_pos,
        }
        return obs
    
    def reset(self):
        """Reset and return image-based observation."""
        self.env.reset()
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        """Step and return image-based observation."""
        _, reward, done, info = self.env.step(action)
        obs = self._get_obs()
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        """
        Render the environment.
        
        If render_cache is available (from last step/reset), returns that.
        Otherwise renders fresh.
        """
        if mode == 'rgb_array':
            if self.render_cache is not None:
                # Convert back to uint8 for display
                img = (self.render_cache * 255).astype(np.uint8)
                return img
            else:
                return self.env.render(mode=mode)
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported.")
    
    def close(self):
        self.env.close()
    
    def get_state(self):
        return self.env.get_state()
    
    def set_state(self, state):
        self.env.set_state(state)
