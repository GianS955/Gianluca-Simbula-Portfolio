import gymnasium as gym
import gymnasium_robotics
from sac_agent import Agent
from replay_buffer import ReplayBuffer
import numpy as np
import copy
from tqdm import tqdm
import csv
import os 
from torch.utils.tensorboard import SummaryWriter
import random
from datetime import datetime

def train():
    """Train a SAC agent on FetchReach-v4 using Hindsight Experience Replay.

    Runs an episode-based training loop with HER augmentation, logging losses
    and success rates to both TensorBoard and CSV files. Saves the best model
    checkpoint and stops early if success rate does not improve for 10 consecutive
    evaluations.
    """
    formatted_daytime = datetime.now().strftime("%d%m%y_%H%M")
    result_folder = os.path.join('results', formatted_daytime)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    writer = SummaryWriter(log_dir = result_folder)
    
    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchReach-v4')
    obs, info = env.reset()
    dummy_action = env.action_space.sample()

    buffer = ReplayBuffer(100_000, obs, dummy_action)

    agent_dict = {
        'actor':{
            'input_shape': obs['observation'].shape[0] + obs['desired_goal'].shape[0],
            'hidden_sizes': [64,64,64],
            'output_shape': dummy_action.shape[0],
            'learning_rate': 1e-4
        },

        'critic':{
            'input_shape': obs['observation'].shape[0] + obs['desired_goal'].shape[0] + dummy_action.shape[0],
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

    max_steps = 50 # max number of steps for FetchReach
    batch_size = 256
    learning_start = 10_000
    gamma = 0.99
    tau = 5e-2
    eval_freq = 5000
    eval_env = gym.make('FetchReach-v4')
    eval_step = 0
    total_episodes = 100_000
    k = 4 # HER ratio
    global_step = 0
    best_success_ratio = 0
    patience = 10
    patience_counter = 0

    progress_bar = tqdm(range(total_episodes))
    for episode in progress_bar:
        progress_bar.set_description(f'Episode {episode+1}')
        
        current_obs, _ = env.reset()
        trajectory = []
        for step in range(max_steps):
            action = agent.select_action(np.concatenate([current_obs['observation'],current_obs['desired_goal']])) 
            next_obs, reward, terminal, truncated, info = env.step(action)
            buffer.store(current_obs, next_obs, action, reward, terminal, truncated)
            trajectory.append({
                'current_obs': current_obs,
                'next_obs' : next_obs,
                'action' : action,
                'reward' : reward,
                'terminal' : terminal,
                'truncated' : truncated,
                'info' : info
            })
            if terminal or truncated:
                break
            else:
                current_obs = next_obs
        
        global_step += len(trajectory)

        HER(buffer, env, trajectory, k)    

        # with HER, the number of update steps it is equal to the length of the episode
        for _ in range(len(trajectory)):
            if buffer.size() >= learning_start: # it's time to update the agent    
                    
                batch = buffer.sample(batch_size)
                critic_loss = agent.update_critic(batch['current_states'],
                                    batch['next_states'],
                                    batch['current_actions'],
                                    batch['rewards'],
                                    batch['terminals'],
                                    gamma)
                actor_loss = agent.update_actor(batch['current_states'])
                alpha_loss = agent.update_alpha(batch['current_states'])
                agent.soft_update(tau)

                writer.add_scalar('Loss/critic', critic_loss, global_step)
                writer.add_scalar('Loss/actor', actor_loss, global_step)
                writer.add_scalar('Loss/alpha', alpha_loss, global_step)

                write_csv(f'{result_folder}/losses.csv',{'Critic Loss': critic_loss,
                                                        'Actor Loss': actor_loss,
                                                        'Alpha Loss': alpha_loss})
                
        if (global_step % eval_freq) == 0 and global_step > 0:
            success_ratio = evaluate(agent, eval_env, 20)
            writer.add_scalar('Success_Ratio', success_ratio, eval_step)
            write_csv(f'{result_folder}/success_rates.csv',{'Success Ratio': success_ratio})
            eval_step +=1

            if success_ratio > best_success_ratio:
                best_success_ratio = success_ratio
                agent.save(result_folder)
                patience_counter = 0
            else:
                patience_counter +=1

            if patience_counter >= patience:
                break

    writer.close()    

def evaluate(agent: Agent, env: gym.Env, n_episodes: int) -> float:
    """Evaluate the agent over a fixed number of episodes and return the success rate.

    Args:
        agent: Trained SAC agent.
        env: Evaluation environment (separate instance from training env).
        n_episodes: Number of full episodes to run.

    Returns:
        Fraction of episodes in which the agent reached the goal (0.0 to 1.0).
    """
    outcomes = np.zeros(n_episodes)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        while (terminated is False):
            action = agent.select_action(np.concatenate([obs['observation'],obs['desired_goal']]))
            obs, _, terminal, truncated, info = env.step(action)
            terminated = terminal or truncated
            outcomes[ep] = info['is_success']

    return outcomes.mean()

def write_csv(file: str, row: dict) -> None:
    """Append a single row to a CSV file, writing the header on the first call.

    Args:
        file: Path to the CSV file. Created if it does not exist.
        row: Dictionary mapping column names to values for this row.
    """
    file_exists = os.path.exists(file)
    with open(file,'a', newline='') as csv_file:
        wrt = csv.DictWriter(csv_file, fieldnames=row.keys())
        if not file_exists:
            wrt.writeheader()
        wrt.writerow(row)

def HER(buffer: ReplayBuffer, env: gym.Env, trajectory: list, k: int) -> None:
    """Augment the replay buffer with Hindsight Experience Replay (future strategy).

    For k randomly selected steps in the episode, replaces the desired goal with
    a goal achieved at a future step, recomputes the reward, and stores the
    relabelled transition in the buffer. This provides positive reward signal
    even from failed episodes.

    Args:
        buffer: Replay buffer to augment in-place.
        env: Training environment, used to recompute the reward after relabelling.
        trajectory: List of transition dicts from the current episode, each containing
            'current_obs', 'next_obs', 'action', 'reward', 'terminal', 'truncated', 'info'.
        k: Number of relabelled transitions to generate per episode.
    """
    indices = random.sample(range(len(trajectory) -2), k)
    for index in indices:
        sample = copy.deepcopy(trajectory[index])
        future_index = random.randint(index+1, len(trajectory)-1)
        sample['current_obs']['desired_goal'] = trajectory[future_index]['current_obs']['achieved_goal']
        sample['next_obs']['desired_goal'] = trajectory[future_index]['current_obs']['achieved_goal']
        future_reward = env.unwrapped.compute_reward(sample['current_obs']['achieved_goal'], 
                                           sample['current_obs']['desired_goal'],
                                           sample['info'])
        
        sample['reward'] = future_reward
        buffer.store(sample['current_obs'], sample['next_obs'],sample['action'],sample['reward'],sample['terminal'], sample['truncated'])