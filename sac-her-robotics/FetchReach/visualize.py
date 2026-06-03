import imageio
import gymnasium as gym
import gymnasium_robotics
from sac_agent import Agent
import numpy as np
import os

def visualize(path: str) -> None:
    """Generate a GIF of the trained agent playing 5 episodes of FetchReach-v4.

    Loads actor and critic weights from the given results folder, runs 5 episodes
    using the stochastic policy, and saves the rendered frames as a GIF.
    A black frame is inserted between episodes as a visual separator.

    Args:
        path: Directory containing actor.pt, critic.pt, and log_alpha.pt checkpoints.
            The output GIF is saved to the same directory as fetchreach.gif.
    """
    gym.register_envs(gymnasium_robotics)
    env = gym.make('FetchReach-v4', render_mode = 'rgb_array')
    frames = []

    obs, _  = env.reset()
    dummy_action = env.action_space.sample()
    agent_dict = {
            'actor':{
                'input_shape': obs['observation'].shape[0] + obs['desired_goal'].shape[0],
                'hidden_sizes': [64,64,64],
                'output_shape': dummy_action.shape[0],
                'learning_rate': 1e-4
            },

            'critic':{
                'input_shape': obs['observation'].shape[0] + obs['desired_goal'].shape[0]+dummy_action.shape[0],
                'hidden_sizes': [64,64,64],
                'output_shape': 1,
                'learning_rate': 1e-4
            },

            'alpha':{
                'learning_rate': 1e-4
            },
            
            'target_entropy': dummy_action.shape[0]
        }
    
    agent = Agent(agent_dict)
    agent.load(path)

    for _ in range(5):
        done = False
        frames.append(env.render())
        while not done:
            action = agent.select_action(np.concatenate([obs['observation'],obs['desired_goal']]))
            obs, _, terminal, truncated, info = env.step(action)
            frames.append(env.render())
            if terminal or truncated:
                break
        obs, _ = env.reset()
        frames.append(np.zeros_like(frames[0]))
    
    gif_path = os.path.join(path, 'fetchreach.gif')
    imageio.mimsave(gif_path, frames, fps = 30)
    print(f'File {gif_path} generated.')
    
if __name__ == '__main__':
    import sys
    visualize(sys.argv[1])