import os
import gymnasium as gym
import imageio
import numpy as np

def record_agent(agent, output_path="/videos/test-video.gif"):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, info = env.reset()
    agent.start(state)
    done = False
    frames = []
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    
    print(f"\Frames collected: {len(frames)}")
    
    frames = [f for f in frames if f is not None]
    frames = [np.array(f, dtype=np.uint8) for f in frames]
    
    if len(frames) == 0:
        print("ERROR: No valid frames")
        return
    
    imageio.mimsave(output_path, frames, fps=30)

def moving_average(data, window_size = 50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')