"""
Box Touch Environment: Franka Panda robot with point end-effector must touch
the top surface of a randomly-sized box on a table.

Uses dm_control as the MuJoCo backend (same as the kitchen environment).
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import mujoco as dm_mujoco


# Path to the MuJoCo XML scene
SCENE_XML_PATH = os.path.join(
    os.path.dirname(__file__), 'assets', 'box_touch_scene.xml')

# Robot constants
N_ARM_JOINTS = 7
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Table geometry
TABLE_POS = np.array([0.5, 0.0, 0.4])
TABLE_HALF_SIZE = np.array([0.4, 0.4])  # x, y half-extents of table top
TABLE_TOP_Z = TABLE_POS[2] + 0.02  # table top surface z

# Box randomization ranges
BOX_SIZE_MIN = 0.02  # minimum half-extent per dimension
BOX_SIZE_MAX = 0.08  # maximum half-extent per dimension
BOX_POS_MARGIN = 0.05  # margin from table edge for box placement

# Default fixed box half-extents (used when randomize_box_size=False)
DEFAULT_BOX_HALF_SIZE = np.array([0.04, 0.04, 0.04])

# Reward parameters
TOUCH_BONUS = 10.0       # NDD/RHJIANG: WHY IS THIS NEEDED. 
DISTANCE_SCALE = 1.0
SUCCESS_THRESHOLD = 0.03  # distance threshold for "touching"

# Simulation parameters
DEFAULT_FRAME_SKIP = 20  # number of sim steps per env step
DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_RENDER_SIZE = (240, 360)  # (height, width)


class BoxTouchEnv(gym.Env):
    """
    MuJoCo environment where a Franka Panda robot with a point end-effector
    must touch the top surface of a randomly-sized box on a table.
    
    The robot is position-controlled (7 DOF arm joints).
    
    This is the core environment. Use BoxTouchLowdimWrapper or 
    BoxTouchImageWrapper for the appropriate observation format.
    """
    
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 25,
    }
    
    def __init__(
        self,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        render_hw: tuple = DEFAULT_RENDER_SIZE,
        box_size_range: tuple = (BOX_SIZE_MIN, BOX_SIZE_MAX),
        reward_type: str = 'dense',  # 'dense' or 'sparse'
        randomize_box_size: bool = True,
        fixed_box_half_size: np.ndarray = None,
    ):
        """
        Args:
            frame_skip: Number of simulation steps per environment step.
            max_episode_steps: Maximum number of steps per episode.
            render_hw: (height, width) for rendering.
            box_size_range: (min, max) half-extent for box size randomization.
            reward_type: 'dense' for distance-based reward, 'sparse' for binary.
            randomize_box_size: If True, randomize box size each reset. 
                If False, use fixed_box_half_size.
            fixed_box_half_size: Fixed box half-extents when randomize_box_size=False.
                Defaults to DEFAULT_BOX_HALF_SIZE.
        """
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_hw = render_hw
        self.box_size_range = box_size_range
        self.reward_type = reward_type
        self.randomize_box_size = randomize_box_size
        self.fixed_box_half_size = (
            fixed_box_half_size if fixed_box_half_size is not None 
            else DEFAULT_BOX_HALF_SIZE.copy()
        )
        
        # Load the MuJoCo model
        self.physics = dm_mujoco.Physics.from_xml_path(SCENE_XML_PATH)
        self.model = self.physics.model
        self.data = self.physics.data
        
        # Cache body/geom/site IDs
        self._ee_site_id = self.physics.model.name2id('end_effector', 'site')
        self._box_top_site_id = self.physics.model.name2id('box_top', 'site')
        self._box_body_id = self.physics.model.name2id('target_box', 'body')
        self._box_geom_id = self.physics.model.name2id('target_box_geom', 'geom')
        self._ee_geom_id = self.physics.model.name2id('ee_sphere', 'geom')
        
        # Define action space: 7 joint position targets
        ctrl_range = self.physics.model.actuator_ctrlrange.copy()
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        
        # Define observation space (full state):
        # joint_pos (7) + joint_vel (7) + ee_pos (3) + box_pos (3) + box_size (3)
        obs_dim = 7 + 7 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        ) # NDD/RHJIANG: MAYBE WE NEED TO ADD BOUNDS
        
        # Episode state
        self._step_count = 0
        self._current_box_size = np.array([0.04, 0.04, 0.04])
        self._seed = None
        self.np_random = np.random.default_rng()
        
    def seed(self, seed=None):
        """Set the random seed."""
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]
    
    def reset(self, init_qpos=None):
        """
        Reset the environment with a new random box configuration.
        
        Args:
            init_qpos: Optional (7,) array of initial joint positions.
                If None, uses HOME_QPOS.
        """
        # Reset physics
        self.physics.reset()
        
        # Set robot to initial configuration
        qpos = self.physics.data.qpos.copy()
        if init_qpos is not None:
            qpos[:N_ARM_JOINTS] = np.array(init_qpos)
        else:
            qpos[:N_ARM_JOINTS] = HOME_QPOS
        
        # Box size: either randomize or use fixed
        if self.randomize_box_size:
            box_half_size = self.np_random.uniform(
                self.box_size_range[0], self.box_size_range[1], size=3)
        else:
            box_half_size = self.fixed_box_half_size.copy()
        self._current_box_size = box_half_size.copy()
        
        # Randomize box position on table
        x_range = TABLE_HALF_SIZE[0] - BOX_POS_MARGIN - box_half_size[0]
        y_range = TABLE_HALF_SIZE[1] - BOX_POS_MARGIN - box_half_size[1]
        box_x = TABLE_POS[0] + self.np_random.uniform(-x_range, x_range)
        box_y = TABLE_POS[1] + self.np_random.uniform(-y_range, y_range)
        box_z = TABLE_TOP_Z + box_half_size[2]  # box sits on table
        
        # Update box geom size in the model
        self.physics.model.geom_size[self._box_geom_id] = box_half_size
        
        # Update box body position
        self.physics.model.body_pos[self._box_body_id] = [box_x, box_y, box_z]
        
        # Update box_top site position (relative to box body, at the top)
        self.physics.model.site_pos[self._box_top_site_id] = [0, 0, box_half_size[2]]
        
        # Set the qpos
        self.physics.data.qpos[:] = qpos
        self.physics.data.qvel[:] = 0
        
        # Set control to initial joint position
        init_ctrl = qpos[:N_ARM_JOINTS].copy()
        self.physics.data.ctrl[:N_ARM_JOINTS] = init_ctrl
        
        # Forward to update derived quantities
        self.physics.forward()
        
        self._step_count = 0
        
        return self._get_obs()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: 7D array of target joint positions.
            
        Returns:
            obs, reward, done, info
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Set control targets
        self.physics.data.ctrl[:N_ARM_JOINTS] = action
        
        # Step simulation
        for _ in range(self.frame_skip):
            self.physics.step()
        
        self._step_count += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        ee_pos = self._get_ee_pos()
        box_top_pos = self._get_box_top_pos()
        distance = np.linalg.norm(ee_pos - box_top_pos)
        
        is_touching = self._check_contact()
        is_close = distance < SUCCESS_THRESHOLD
        success = is_touching or is_close
        
        if self.reward_type == 'dense':
            reward = -DISTANCE_SCALE * distance
            if success:
                reward += TOUCH_BONUS
        else:  # sparse
            reward = TOUCH_BONUS if success else 0.0
        
        # Check termination
        done = success or (self._step_count >= self.max_episode_steps)
        
        info = {
            'ee_pos': ee_pos.copy(),
            'box_top_pos': box_top_pos.copy(),
            'distance': distance,
            'is_touching': is_touching,
            'is_close': is_close,
            'success': success,
            'step_count': self._step_count,
            'box_size': self._current_box_size.copy(),
        }
        
        return obs, reward, done, info
    
    def render(self, mode='rgb_array', camera_name='front_camera'):
        """
        Render the environment.
        
        Args:
            mode: 'rgb_array' returns an image array, 'human' not supported.
            camera_name: Name of the camera to render from.
            
        Returns:
            RGB image as numpy array (H, W, 3) uint8.
        """
        h, w = self.render_hw
        if mode == 'rgb_array':
            camera_id = self.physics.model.name2id(camera_name, 'camera')
            img = self.physics.render(
                height=h, width=w, camera_id=camera_id)
            return img
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported.")
    
    def close(self):
        """Clean up resources."""
        pass
    
    # ---- Internal methods ----
    
    def _get_obs(self):
        """
        Get the full state observation.
        
        Returns:
            np.array of shape (23,):
                joint_pos (7) + joint_vel (7) + ee_pos (3) + box_pos (3) + box_size (3)
        """
        joint_pos = self.physics.data.qpos[:N_ARM_JOINTS].copy()
        joint_vel = self.physics.data.qvel[:N_ARM_JOINTS].copy()
        ee_pos = self._get_ee_pos()
        box_pos = self.physics.data.xpos[self._box_body_id].copy()
        box_size = self._current_box_size.copy()
        
        return np.concatenate([joint_pos, joint_vel, ee_pos, box_pos, box_size])
    
    def _get_ee_pos(self):
        """Get end-effector position in world frame."""
        return self.physics.data.site_xpos[self._ee_site_id].copy()
    
    def _get_box_top_pos(self):
        """Get the position of the top center of the box."""
        return self.physics.data.site_xpos[self._box_top_site_id].copy()
    
    def _check_contact(self):
        """Check if the end-effector is in contact with the box."""
        for i in range(self.physics.data.ncon):
            contact = self.physics.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if ((geom1 == self._ee_geom_id and geom2 == self._box_geom_id) or
                (geom1 == self._box_geom_id and geom2 == self._ee_geom_id)):
                return True
        return False
    
    def get_state(self):
        """Get the full simulation state for saving/restoring."""
        return {
            'qpos': self.physics.data.qpos.copy(),
            'qvel': self.physics.data.qvel.copy(),
            'ctrl': self.physics.data.ctrl.copy(),
            'box_size': self._current_box_size.copy(),
            'box_body_pos': self.physics.model.body_pos[self._box_body_id].copy(),
            'box_geom_size': self.physics.model.geom_size[self._box_geom_id].copy(),
            'box_top_site_pos': self.physics.model.site_pos[self._box_top_site_id].copy(),
            'step_count': self._step_count,
        }
    
    def set_state(self, state):
        """Restore the full simulation state."""
        self.physics.data.qpos[:] = state['qpos']
        self.physics.data.qvel[:] = state['qvel']
        self.physics.data.ctrl[:] = state['ctrl']
        self._current_box_size = state['box_size'].copy()
        self.physics.model.body_pos[self._box_body_id] = state['box_body_pos']
        self.physics.model.geom_size[self._box_geom_id] = state['box_geom_size']
        self.physics.model.site_pos[self._box_top_site_id] = state['box_top_site_pos']
        self._step_count = state['step_count']
        self.physics.forward()
