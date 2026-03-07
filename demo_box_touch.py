"""
Demo script for the Box Touch environment.

Runs an episode with a Jacobian IK controller that drives the Franka's
end-effector to the top of the randomly-placed box, and saves a video.

Usage:
    python demo_box_touch.py

This will create:
    - demo_box_touch.mp4: Video of the episode from the front camera
    - demo_box_touch_multicam.png: Side-by-side view from 3 cameras
"""

import os
import time
import numpy as np
import imageio

# Use osmesa for headless rendering (no display needed)
os.environ['MUJOCO_GL'] = 'osmesa'

from diffusion_policy.env.box_touch.box_touch_env import BoxTouchEnv, N_ARM_JOINTS


def jacobian_ik_action(env, target_pos, step_size=0.3):
    """
    Compute a joint position target that moves the end-effector toward
    target_pos using the MuJoCo Jacobian (resolved-rate IK).
    
    Args:
        env: BoxTouchEnv instance
        target_pos: (3,) desired EE position in world frame
        step_size: Gain for the IK step (higher = faster but less stable)
    
    Returns:
        action: (7,) joint position targets
    """
    physics = env.physics
    
    # Current EE position and joint positions
    ee_pos = env._get_ee_pos()
    q_current = physics.data.qpos[:N_ARM_JOINTS].copy()
    
    # Position error
    dx = target_pos - ee_pos  # (3,)
    
    # Get the translational Jacobian for the EE site
    # jacp is (3, nv) — translational Jacobian
    nv = physics.model.nv
    jacp = np.zeros((3, nv))
    physics.data.site_xpos  # ensure forward kinematics are up to date
    
    # Use mujoco's Jacobian computation
    import mujoco
    mujoco.mj_jacSite(
        physics.model.ptr, physics.data.ptr,
        jacp, None,  # jacp, jacr (we only need translational)
        env._ee_site_id
    )
    
    # Extract only the arm joint columns (first 7)
    J = jacp[:, :N_ARM_JOINTS]  # (3, 7)
    
    # Pseudoinverse IK: dq = J_pinv @ dx
    # Add damping for numerical stability (damped least squares)
    damping = 1e-4
    JtJ = J.T @ J + damping * np.eye(N_ARM_JOINTS)
    dq = np.linalg.solve(JtJ, J.T @ dx)  # (7,)
    
    # Compute target joint positions
    q_target = q_current + step_size * dq
    
    # Clip to joint limits
    q_target = np.clip(q_target, env.action_space.low, env.action_space.high)
    
    return q_target.astype(np.float32)


def run_demo(
    num_steps=100,
    seed=42,
    video_path='demo_box_touch.mp4',
    render_size=(256, 368),
):
    """
    Run a demo episode with the Jacobian IK controller and save a video.
    """
    t_start = time.time()
    
    print("Creating BoxTouchEnv...")
    env = BoxTouchEnv(render_hw=render_size)
    env.seed(seed)
    
    print("Resetting environment...")
    obs = env.reset()
    
    # Get initial info
    ee_pos = env._get_ee_pos()
    box_top_pos = env._get_box_top_pos()
    box_size = env._current_box_size
    print(f"  End-effector position: {ee_pos}")
    print(f"  Box top position:      {box_top_pos}")
    print(f"  Box half-size:         {box_size}")
    print(f"  Initial distance:      {np.linalg.norm(ee_pos - box_top_pos):.4f}")
    
    frames = []
    
    # Render initial frame
    frame = env.render(mode='rgb_array', camera_name='front_camera')
    frames.append(frame)
    
    print(f"\nRunning IK controller for up to {num_steps} steps...")
    
    for step in range(num_steps):
        # Compute IK action toward box top
        target = env._get_box_top_pos()
        action = jacobian_ik_action(env, target, step_size=0.3)
        
        obs, reward, done, info = env.step(action)
        
        # Render
        frame = env.render(mode='rgb_array', camera_name='front_camera')
        frames.append(frame)
        
        if step % 5 == 0 or done:
            print(f"  Step {step:3d}: reward={reward:8.4f}, "
                  f"distance={info['distance']:.4f}, "
                  f"touching={info['is_touching']}, "
                  f"success={info['success']}")
        
        if done:
            print(f"\n  Episode ended at step {step}!")
            if info['success']:
                print("  ✓ SUCCESS: Robot touched the box top!")
            break
    
    if not done:
        print(f"\n  Episode reached max steps ({num_steps})")
    
    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.2f}s ({len(frames)} frames, "
          f"{len(frames)/elapsed:.1f} fps)")
    
    # Save video
    print(f"\nSaving video to {video_path}...")
    writer = imageio.get_writer(video_path, fps=25)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Video saved: {video_path} ({len(frames)} frames)")
    
    env.close()
    print("\nDone!")


def run_multi_camera_demo(
    seed=42,
    render_size=(240, 360),
):
    """
    Render a single reset from 3 different cameras side by side.
    """
    print("Creating multi-camera demo...")
    env = BoxTouchEnv(render_hw=render_size)
    env.seed(seed)
    env.reset()
    
    cameras = ['front_camera', 'side_camera', 'top_camera']
    images = []
    for cam in cameras:
        img = env.render(mode='rgb_array', camera_name=cam)
        images.append(img)
    
    # Stack horizontally
    combined = np.concatenate(images, axis=1)
    
    imageio.imwrite('demo_box_touch_multicam.png', combined)
    print(f"  Multi-camera image saved: demo_box_touch_multicam.png")
    print(f"  Image shape: {combined.shape}")
    
    env.close()


if __name__ == '__main__':
    run_demo(num_steps=100)
    print()
    run_multi_camera_demo()
