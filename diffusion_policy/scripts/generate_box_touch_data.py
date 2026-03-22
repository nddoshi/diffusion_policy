"""
Generate expert demonstration data for the Box Touch environment.

Uses a Jacobian IK oracle controller to drive the Franka's end-effector
to the top center of a randomly-placed box on a table.

Each episode has:
  - Randomized box position on the table (fixed box size)
  - Randomized initial arm configuration (EE starts anywhere above the table)
  - IK oracle drives EE to box top center

Saves both low-dimensional state and camera images to a zarr ReplayBuffer.

Usage:
    python diffusion_policy/scripts/generate_box_touch_data.py \
        -o data/box_touch/box_touch.zarr -n 1000
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import numpy as np
import mujoco
from scipy.optimize import minimize as scipy_minimize
from tqdm import tqdm

# Use EGL for GPU-accelerated headless rendering (~130x faster than osmesa).
# Requires NVIDIA EGL vendor JSON. If EGL fails, fall back to osmesa.
_EGL_VENDOR_DIR = os.path.expanduser("~/.local/share/glvnd/egl_vendor.d")
if os.path.isdir(_EGL_VENDOR_DIR):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", _EGL_VENDOR_DIR)
os.environ.setdefault("MUJOCO_GL", "egl")

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.box_touch.box_touch_env import (
    BoxTouchEnv,
    N_ARM_JOINTS,
    HOME_QPOS,
    TABLE_POS,
    TABLE_HALF_SIZE,
    TABLE_TOP_Z,
    DEFAULT_BOX_HALF_SIZE,
)


# ---- IK Oracle ----


def jacobian_ik_action(env, target_pos, max_cart_step=0.10, damping=1e-4,
                       null_space_gain=0.1):
    """
    Compute a joint position target that moves the end-effector toward
    target_pos using the MuJoCo Jacobian with:
      - Capped Cartesian displacement (no overshoot)
      - Null-space joint-limit avoidance (keeps joints centered)

    Args:
        env: BoxTouchEnv instance
        target_pos: (3,) desired EE position in world frame
        max_cart_step: Maximum Cartesian displacement per step (meters)
        damping: Damping factor for numerical stability
        null_space_gain: Gain for null-space joint-centering

    Returns:
        action: (7,) joint position targets
    """
    physics = env.physics
    ee_pos = env._get_ee_pos()
    q_current = physics.data.qpos[:N_ARM_JOINTS].copy()

    dx = target_pos - ee_pos  # (3,)
    dist = np.linalg.norm(dx)

    # Cap Cartesian step to prevent overshoot
    if dist > max_cart_step:
        dx = dx / dist * max_cart_step

    # Get Jacobian
    nv = physics.model.nv
    jacp = np.zeros((3, nv))
    mujoco.mj_jacSite(physics.model.ptr, physics.data.ptr, jacp, None, env._ee_site_id)
    J = jacp[:, :N_ARM_JOINTS]  # (3, 7)

    # Damped pseudo-inverse: J_pinv = (J^T J + λI)^{-1} J^T
    JtJ = J.T @ J + damping * np.eye(N_ARM_JOINTS)
    J_pinv = np.linalg.solve(JtJ, J.T)  # (7, 3)

    # Primary task: move EE toward target
    dq_task = J_pinv @ dx  # (7,)

    # Null-space: push joints toward center of their range (away from limits)
    q_low = env.action_space.low
    q_high = env.action_space.high
    q_mid = 0.5 * (q_low + q_high)
    q_range = q_high - q_low
    q_null = -null_space_gain * (q_current - q_mid) / q_range

    # Null-space projector: N = I - J_pinv @ J
    N = np.eye(N_ARM_JOINTS) - J_pinv @ J
    dq = dq_task + N @ q_null

    q_target = q_current + dq
    q_target = np.clip(q_target, q_low, q_high)

    return q_target.astype(np.float32)


def solve_ik_for_position(env, target_pos, q_init=None, tol=0.02, max_iter=100):
    """
    Solve IK as a one-shot nonlinear program using L-BFGS-B (scipy).

    Minimizes ||FK(q) - target_pos||^2 subject to joint limit bounds,
    using the analytical MuJoCo Jacobian as the gradient.

    Args:
        env: BoxTouchEnv instance (will be modified in place via qpos/forward)
        target_pos: (3,) desired EE position in world frame
        q_init: (7,) initial joint guess. If None, uses current qpos.
        tol: Position tolerance (meters) for success
        max_iter: Maximum L-BFGS-B iterations

    Returns:
        q_solution: (7,) joint positions, or None if IK failed
        residual: Final EE position error (meters)
    """
    physics = env.physics
    ctrl_range = physics.model.actuator_ctrlrange
    bounds = [(ctrl_range[i, 0], ctrl_range[i, 1]) for i in range(N_ARM_JOINTS)]

    if q_init is None:
        q_init = physics.data.qpos[:N_ARM_JOINTS].copy()

    target = np.asarray(target_pos, dtype=np.float64)

    def objective(q):
        physics.data.qpos[:N_ARM_JOINTS] = q
        physics.data.qvel[:] = 0
        physics.forward()
        ee_pos = physics.data.site_xpos[env._ee_site_id].copy()
        return float(np.sum((ee_pos - target) ** 2))

    def gradient(q):
        physics.data.qpos[:N_ARM_JOINTS] = q
        physics.data.qvel[:] = 0
        physics.forward()
        ee_pos = physics.data.site_xpos[env._ee_site_id].copy()

        nv = physics.model.nv
        jacp = np.zeros((3, nv))
        mujoco.mj_jacSite(
            physics.model.ptr, physics.data.ptr, jacp, None, env._ee_site_id
        )
        J = jacp[:, :N_ARM_JOINTS]  # (3, 7)

        # grad of ||FK(q) - target||^2 = 2 * J^T @ (FK(q) - target)
        return (2.0 * J.T @ (ee_pos - target)).astype(np.float64)

    result = scipy_minimize(
        objective,
        x0=q_init.astype(np.float64),
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
    )

    # Set final state
    physics.data.qpos[:N_ARM_JOINTS] = result.x
    physics.data.qvel[:] = 0
    physics.forward()

    residual = np.sqrt(result.fun)
    if residual < tol:
        return result.x.copy(), residual
    else:
        return None, residual


def sample_random_ee_above_table(
    rng, box_top_z, ee_z_min_offset=0.1, ee_z_max_offset=0.3
):
    """
    Sample a random EE target position anywhere above the table.

    Args:
        rng: numpy random generator
        box_top_z: z-coordinate of the box top
        ee_z_min_offset: Minimum height above box top
        ee_z_max_offset: Maximum height above box top

    Returns:
        target_pos: (3,) position above the table
    """
    # Table XY bounds
    x_min = TABLE_POS[0] - TABLE_HALF_SIZE[0] + 0.05
    x_max = TABLE_POS[0] + TABLE_HALF_SIZE[0] - 0.05
    y_min = TABLE_POS[1] - TABLE_HALF_SIZE[1] + 0.05
    y_max = TABLE_POS[1] + TABLE_HALF_SIZE[1] - 0.05

    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    z = box_top_z + rng.uniform(ee_z_min_offset, ee_z_max_offset)

    return np.array([x, y, z])


def find_random_init_qpos(
    env, rng, box_top_z, max_attempts=50, ee_z_min_offset=0.1, ee_z_max_offset=0.3
):
    """
    Find a valid initial joint configuration with the EE at a random position
    above the table, using rejection sampling.

    Args:
        env: BoxTouchEnv instance
        rng: numpy random generator
        box_top_z: z-coordinate of the box top
        max_attempts: Maximum IK attempts before giving up
        ee_z_min_offset: Min height above box top for initial EE
        ee_z_max_offset: Max height above box top for initial EE

    Returns:
        q_init: (7,) joint positions
        target_pos: (3,) the EE target that was achieved
        n_attempts: Number of IK attempts made
        failures: List of (target_pos, reason) for failed attempts
    """
    failures = []

    for attempt in range(max_attempts):
        # Sample random target above table
        target_pos = sample_random_ee_above_table(
            rng,
            box_top_z,
            ee_z_min_offset=ee_z_min_offset,
            ee_z_max_offset=ee_z_max_offset,
        )

        # Start IK from home position
        env.physics.data.qpos[:N_ARM_JOINTS] = HOME_QPOS
        env.physics.data.qvel[:] = 0
        env.physics.forward()

        q_solution, residual = solve_ik_for_position(
            env, target_pos, q_init=HOME_QPOS, tol=0.02, max_iter=100
        )

        if q_solution is not None:
            return q_solution, target_pos, attempt + 1, failures
        else:
            reason = f"residual={residual:.4f}"
            failures.append((target_pos.copy(), reason))
            print(
                f"  IK attempt {attempt+1}/{max_attempts} failed: "
                f"target_pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}], "
                f"{reason}"
            )

    # All attempts failed — fall back to home position
    print(f"  WARNING: All {max_attempts} IK attempts failed, using HOME_QPOS")
    return HOME_QPOS.copy(), None, max_attempts, failures


# ---- Main Generation ----


@click.command()
@click.option("-o", "--output", required=True, help="Output zarr path")
@click.option("-n", "--n_episodes", default=1000, help="Number of episodes")
@click.option("--max_steps", default=200, help="Max steps per episode")
@click.option("--render_h", default=240, help="Render height for images")
@click.option("--render_w", default=360, help="Render width for images")
@click.option("--camera", default="front_camera", help="Camera name for images")
@click.option("--max_cart_step", default=0.10, help="Max Cartesian step per timestep (meters)")
@click.option("--ee_z_min", default=0.1, help="Min EE height above box top for init")
@click.option("--ee_z_max", default=0.3, help="Max EE height above box top for init")
@click.option("--chunk_length", default=-1, help="Zarr chunk length (-1 for auto)")
@click.option("--seed_offset", default=0, help="Starting seed offset")
def main(
    output,
    n_episodes,
    max_steps,
    render_h,
    render_w,
    camera,
    max_cart_step,
    ee_z_min,
    ee_z_max,
    chunk_length,
    seed_offset,
):
    """Generate expert demonstrations for the Box Touch environment."""

    print(f"Generating {n_episodes} episodes → {output}")
    print(f"  Max steps: {max_steps}")
    print(f"  Image size: ({render_h}, {render_w}), camera: {camera}")
    print(f"  EE init height range: [{ee_z_min}, {ee_z_max}] above box top")
    print()

    buffer = ReplayBuffer.create_empty_numpy()

    env = BoxTouchEnv(
        render_hw=(render_h, render_w),
        max_episode_steps=max_steps,
        randomize_box_size=False,  # Fixed box size
        reward_type="dense",
    )

    # IK statistics
    total_ik_attempts = 0
    total_ik_failures = 0
    max_ik_attempts = 0
    successful_episodes = 0
    failed_episodes = 0

    for ep_idx in tqdm(range(n_episodes), desc="Generating episodes"):
        seed = seed_offset + ep_idx
        env.seed(seed)
        rng = env.np_random

        # Step 1: Reset env (places box randomly, arm at home)
        env.reset()

        # Get box top position after reset
        box_top_z = env._get_box_top_pos()[2]

        # Step 2: Find random initial arm configuration via IK
        q_init, init_target, n_attempts, failures = find_random_init_qpos(
            env,
            rng,
            box_top_z,
            max_attempts=50,
            ee_z_min_offset=ee_z_min,
            ee_z_max_offset=ee_z_max,
        )

        total_ik_attempts += n_attempts
        total_ik_failures += len(failures)
        max_ik_attempts = max(max_ik_attempts, n_attempts)

        # Step 3: Reset env again with the found initial qpos
        # (re-seed to get the same box position)
        env.seed(seed)
        obs = env.reset(init_qpos=q_init)

        # Collect episode data
        obs_history = []
        action_history = []
        img_history = []

        # Record initial observation and image
        img = env.render(mode="rgb_array", camera_name=camera)

        # Step 4: Run IK oracle toward box top center
        done = False
        episode_success = False
        action = None

        for step in range(max_steps):
            # Get target: center of box top face
            target = env._get_box_top_pos()

            # Compute IK action
            action = jacobian_ik_action(env, target, max_cart_step=max_cart_step)

            # Record current state before stepping
            obs_history.append(obs.astype(np.float32))
            action_history.append(action.astype(np.float32))
            img_history.append(img)

            # Step environment
            obs, reward, done, info = env.step(action)
            img = env.render(mode="rgb_array", camera_name=camera)

            if done:
                episode_success = info["success"]
                break

        # Record final observation (after last step)
        # We want obs and action arrays to have the same length,
        # so we record the action that would be taken at the final state too
        if done and episode_success:
            # At success, record the final obs with a "hold" action
            obs_history.append(obs.astype(np.float32))
            action_history.append(action.astype(np.float32))  # repeat last action
            img_history.append(img)

        if episode_success:
            successful_episodes += 1
        else:
            failed_episodes += 1

        # Convert to arrays
        obs_array = np.stack(obs_history)  # (T, 23)
        action_array = np.stack(action_history)  # (T, 7)
        img_array = np.stack(img_history)  # (T, H, W, 3) uint8

        episode_data = {
            "obs": obs_array,
            "action": action_array,
            "img": img_array,
        }

        buffer.add_episode(episode_data)

        if (ep_idx + 1) % 100 == 0:
            print(
                f"\n  Progress: {ep_idx+1}/{n_episodes} episodes, "
                f"{successful_episodes} successful, {failed_episodes} failed"
            )

    # Save
    print(f"\nSaving to {output}...")
    if chunk_length > 0:
        buffer.save_to_path(output, chunk_length=chunk_length)
    else:
        buffer.save_to_path(output)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"{'='*60}")
    print(f"  Output: {output}")
    print(f"  Episodes: {n_episodes}")
    print(
        f"  Successful: {successful_episodes} ({100*successful_episodes/n_episodes:.1f}%)"
    )
    print(
        f"  Failed (max steps): {failed_episodes} ({100*failed_episodes/n_episodes:.1f}%)"
    )
    print(f"  Total steps: {buffer.n_steps}")
    print(f"  Avg episode length: {buffer.n_steps / n_episodes:.1f}")
    print(f"\n  IK init stats:")
    print(f"    Total IK attempts: {total_ik_attempts}")
    print(f"    Total IK failures: {total_ik_failures}")
    print(f"    Avg attempts/episode: {total_ik_attempts/n_episodes:.2f}")
    print(f"    Max attempts for one episode: {max_ik_attempts}")
    print(f"\n  Data shapes:")
    for key in buffer.keys():
        arr = buffer[key]
        print(f"    {key}: {arr.shape} ({arr.dtype})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
