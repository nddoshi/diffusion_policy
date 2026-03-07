"""
Box Touch Environment: Franka Panda robot with point end-effector must touch
the top surface of a randomly-sized box on a table.

Two environment variants:
  - BoxTouchLowdimWrapper: State-based observations (joint pos/vel, EE pos, box pos/size)
  - BoxTouchImageWrapper: Image-based observations (camera image + joint positions)
"""

from diffusion_policy.env.box_touch.box_touch_env import BoxTouchEnv
from diffusion_policy.env.box_touch.box_touch_lowdim_wrapper import BoxTouchLowdimWrapper
from diffusion_policy.env.box_touch.box_touch_image_wrapper import BoxTouchImageWrapper
