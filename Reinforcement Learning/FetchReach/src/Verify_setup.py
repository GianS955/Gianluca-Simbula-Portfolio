"""
Phase 1 — MuJoCo + Gymnasium setup verification
Run with: python src/verify_setup.py
"""
import sys
import numpy as np


def check_imports():
    print("=" * 60)
    print("1. Checking imports")
    print("=" * 60)
    try:
        import gymnasium
        print(f"  [OK] gymnasium       {gymnasium.__version__}")
    except ImportError as e:
        print(f"  [FAIL] gymnasium: {e}")
        sys.exit(1)

    try:
        import gymnasium_robotics
        print(f"  [OK] gymnasium-robotics  {gymnasium_robotics.__version__}")
    except ImportError:
        # In recent versions the package registers envs automatically.
        # Check that FetchReach exists in the registry instead.
        import gymnasium as gym
        all_envs = gym.envs.registry.keys()
        if any("Fetch" in e for e in all_envs):
            print(f"  [OK] gymnasium-robotics  (Fetch envs registered correctly)")
        else:
            print(f"  [FAIL] Fetch envs not found — try: pip install gymnasium-robotics")

    try:
        import mujoco
        print(f"  [OK] mujoco          {mujoco.__version__}")
    except ImportError as e:
        print(f"  [FAIL] mujoco: {e}")
        sys.exit(1)

    try:
        import torch
        print(f"  [OK] torch           {torch.__version__}")
        print(f"       CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("  [WARN] torch not installed (required for Phase 2)")

    try:
        import matplotlib
        print(f"  [OK] matplotlib      {matplotlib.__version__}")
    except ImportError:
        print("  [WARN] matplotlib not installed")


def check_env_creation():
    print("\n" + "=" * 60)
    print("2. Environment creation")
    print("=" * 60)
    import gymnasium as gym

    envs = ["FetchReach-v4", "FetchPush-v3"]
    for env_id in envs:
        try:
            env = gym.make(env_id)
            obs, info = env.reset(seed=42)
            print(f"  [OK] {env_id}")
            env.close()
        except Exception as e:
            print(f"  [FAIL] {env_id}: {e}")


def inspect_env(env_id: str = "FetchReach-v4"):
    """
    Full inspection of the observation and action spaces.
    FetchReach uses a dict observation — essential for understanding how HER works.
    """
    import gymnasium as gym

    print("\n" + "=" * 60)
    print(f"3. Space inspection — {env_id}")
    print("=" * 60)

    env = gym.make(env_id)
    obs, info = env.reset(seed=0)

    print("\n  --- Observation space ---")
    print(f"  Type: {type(env.observation_space)}")
    for key, space in env.observation_space.spaces.items():
        print(f"  '{key}': shape={space.shape}, dtype={space.dtype}")
        if key == "observation":
            print(f"          (gripper position + velocity + gripper state)")
        elif key == "achieved_goal":
            print(f"          (current end-effector position in 3D space)")
        elif key == "desired_goal":
            print(f"          (target position — changes at every reset)")

    print("\n  --- Action space ---")
    print(f"  Type:  {type(env.action_space)}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low:   {env.action_space.low}")
    print(f"  High:  {env.action_space.high}")
    print(f"  (4D continuous: dx, dy, dz gripper + gripper opening)")

    print("\n  --- Obs at reset ---")
    for key, val in obs.items():
        print(f"  obs['{key}'] = {np.round(val, 4)}")

    print("\n  --- Reward structure (sparse) ---")
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"  reward after random step: {reward}")
    print(f"  (0.0 = success, -1.0 = failure — sparse!)")
    print(f"  is_success in info: {info2.get('is_success', 'n/a')}")

    env.close()
    return env


def run_random_episode(env_id: str = "FetchReach-v4", n_steps: int = 50):
    """
    Random episode to verify the environment loop works.
    Collects data for the diagnostic plot.
    """
    import gymnasium as gym

    print("\n" + "=" * 60)
    print(f"4. Random episode — {env_id} ({n_steps} steps)")
    print("=" * 60)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=1)

    rewards = []
    distances = []  # distance from achieved_goal to desired_goal

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        distances.append(dist)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    total_success = sum(1 for r in rewards if r == 0.0)
    print(f"  Total steps:          {n_steps}")
    print(f"  Successes (reward=0): {total_success} ({100*total_success/n_steps:.1f}%)")
    print(f"  Mean distance:        {np.mean(distances):.4f} m")
    print(f"  Min distance:         {np.min(distances):.4f} m")

    return rewards, distances


def plot_results(rewards, distances):
    """Save a diagnostic plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Phase 1 — Random Policy on FetchReach-v4", fontsize=13)

    steps = range(len(rewards))
    ax1.step(steps, rewards, color="#2a9d8f", linewidth=1.2, where="post")
    ax1.set_ylabel("Reward (0=success, -1=failure)")
    ax1.set_ylim(-1.2, 0.2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_title("Reward per step — random policy")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, distances, color="#e76f51", linewidth=1.2)
    ax2.set_ylabel("Goal distance (m)")
    ax2.set_xlabel("Step")
    ax2.set_title("Distance from achieved_goal to desired_goal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase1_diagnostic.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved: phase1_diagnostic.png")
    plt.show()


def check_compute_reward():
    """
    Verify that compute_reward() works — required for HER.
    HER calls it directly with virtual goals outside the env loop.
    """
    import gymnasium as gym

    print("\n" + "=" * 60)
    print("5. Testing compute_reward() — critical for HER")
    print("=" * 60)

    env = gym.make("FetchReach-v4")  # raw env, no wrapper
    obs, _ = env.reset(seed=0)

    # Simulate achieved_goal == desired_goal (success)
    achieved = obs["desired_goal"].copy()
    desired = obs["desired_goal"].copy()
    info = {}

    reward = env.unwrapped.compute_reward(achieved, desired, info)
    print(f"  compute_reward(achieved==desired):  {reward}  (expected: 0.0)")
    assert reward == 0.0, "Error: success reward should be 0.0"

    # Simulate failure
    achieved_fail = obs["desired_goal"] + np.array([1.0, 0.0, 0.0])
    reward_fail = env.unwrapped.compute_reward(achieved_fail, desired, info)
    print(f"  compute_reward(achieved!=desired):  {reward_fail}  (expected: -1.0)")
    assert reward_fail == -1.0, "Error: failure reward should be -1.0"

    print("  [OK] compute_reward() works — HER can use it for virtual transitions")
    env.close()


if __name__ == "__main__":
    check_imports()
    check_env_creation()
    inspect_env("FetchReach-v4")
    rewards, distances = run_random_episode("FetchReach-v4", n_steps=100)
    plot_results(rewards, distances)
    check_compute_reward()

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE — Environment verified and operational")
    print("Next step: implement Replay Buffer + SAC (Phase 2)")
    print("=" * 60)
