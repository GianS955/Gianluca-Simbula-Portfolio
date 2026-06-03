# FetchPush ROS2

![Demo](assets/demo.gif)

Deployment of a SAC+HER agent trained on [FetchPush-v4](https://robotics.farama.org/envs/fetch/push/) using ROS2 as middleware. The trained policy runs inside a C++ ROS2 action server via LibTorch, while a Python node handles the Gymnasium simulation environment.

The agent and training code live in a [separate repository](https://github.com/GianS955/Gianluca-Simbula-Portfolio/tree/main/Reinforcement%20Learning/FetchPush). Model weights are available on [HuggingFace](#).

---

## Architecture

The system is split across 4 ROS2 packages:

| Package                      | Language | Role                                               |
| ---------------------------- | -------- | -------------------------------------------------- |
| `fetchpush_action_interface` | —        | Custom action definition (goal/feedback/result)    |
| `fetchpush_msgs`             | —        | Custom messages (Observation, Action, DesiredGoal) |
| `fetchpush_action`           | C++      | Action server + SAC policy inference via LibTorch  |
| `fetchpush_env`              | Python   | Gymnasium environment management                   |

### Communication flow

```
Client
  │
  │  send_goal (desired_goal)
  ▼
fetchpush_action (C++)
  │  publishes DesiredGoal
  ▼
fetchpush_env (Python)
  │  resets env, publishes Observation
  ▼
fetchpush_action (C++)
  │  runs policy → publishes Action + Feedback
  ▼
fetchpush_env (Python)
  │  steps env, publishes next Observation
  └─ loop until is_success or truncated
```

The action server uses `rclcpp_action` and runs the execution loop in a detached thread, while observation and episode state arrive asynchronously through callbacks.

---

## Requirements

- ROS2 Jazzy
- [LibTorch CPU](https://pytorch.org/get-started/locally/) (C++/Java, Linux, CPU, cxx11 ABI) — extract to `~/libtorch`
- Python 3.12
- `gymnasium==1.3.0`
- `gymnasium-robotics`
- `imageio[ffmpeg]`

> **Note:** do not activate a Python virtual environment when building with colcon. ROS2 uses the system Python and the two will conflict.

---

## Installation

```bash
# Clone the repository
git clone <repo-url> ~/FetchPush
cd ~/FetchPush

# Download model weights from HuggingFace
# Place actor_traced.pt somewhere accessible, e.g. ~/projects/fetchpush_sac/

# Install Python dependencies (system Python, not venv)
pip install gymnasium gymnasium-robotics imageio[ffmpeg] --break-system-packages

# Add LibTorch to LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/home/$USER/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Build
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

---

## Usage

Launch both nodes:

```bash
ros2 launch fetchpush_action fetchpush.launch.xml \
  model_path:=/path/to/actor_traced.pt \
  output_path:=/path/to/output/folder
```

Send a goal from a separate terminal:

```bash
source install/setup.bash
ros2 action send_goal --feedback /fetchpush/FetchPush \
  fetchpush_action_interface/action/FetchPush \
  "{desired_goal: [1.38, 0.79, 0.42]}"
```

A GIF of the episode is saved to `output_path`, named after the goal UUID.

### Launch arguments

| Argument      | Default                     | Description                            |
| ------------- | --------------------------- | -------------------------------------- |
| `model_path`  | `/home/.../actor_traced.pt` | Path to the TorchScript actor model    |
| `output_path` | `/home/.../fetchpush_sac`   | Directory where episode GIFs are saved |

---

## Model export

The action server loads a TorchScript model exported with `torch.jit.trace`. To re-export from the training repository:

```python
agent.actor.cpu()
input_example = torch.zeros(1, input_shape).float()
traced = torch.jit.trace(agent.actor, input_example)
traced.save("actor_traced.pt")
```

The model must be on CPU before tracing if you are using the CPU version of LibTorch.

---

## Notes on WSL2

- Keep the ROS2 workspace on the native WSL filesystem (`~/`), not on `/mnt/c/` or `/mnt/d/`. NTFS mounts cause colcon to fail finding `catkin_pkg`.
- If the build fails after moving the workspace, delete `build/`, `install/`, `log/` and rebuild — CMake caches absolute paths.
- MuJoCo rendering uses `render_mode="rgb_array"` to avoid OpenGL issues on WSL2. Episodes are saved as GIFs instead of rendering live.
