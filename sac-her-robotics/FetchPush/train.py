from replay_buffer import ReplayBuffer, ImprovementBuffer
from sac_agent import Agent
import gymnasium as gym
import gymnasium_robotics
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
import random
import copy
from tqdm import tqdm
import csv
import json
import os
from datetime import datetime

def train(result_folder = None):
    """Train a SAC agent on FetchPush-v4 using Hindsight Experience Replay.

    Runs an episode-based training loop with HER augmentation, logging losses
    and success rates to both TensorBoard and CSV files. Saves the best model
    checkpoint and stops early if the average on the last 20 values of success
    rate does not improve for 40 consecutive evaluations.

    Args:
        result_folder: the path of results of the training to resume
    """

    if result_folder is None:
        time_stamp = datetime.now().strftime("%d%m%y_%H%M")
        result_folder = os.path.join('results',time_stamp)
        os.makedirs(result_folder, exist_ok=True)
        new_run = True
    else:
        new_run = False

    gym.register_envs(gymnasium_robotics)
    env = gym.make('FetchPush-v4')

    obs, _ = env.reset()
    dummy_action = env.action_space.sample()

    if new_run is True:
        infos = {
            'actor':{
                'input_shape': obs['observation'].shape[0] + obs['desired_goal'].shape[0],
                'hidden_shapes': [256, 256, 256],
                'output_shape': dummy_action.shape[0],
                'learning_rate': 3e-5
            },
            'critic':
            {
                'input_shape': obs['observation'].shape[0] + obs['desired_goal'].shape[0] + dummy_action.shape[0],
                'hidden_shapes': [256, 256, 256],
                'output_shape': 1,
                'learning_rate': 3e-5,
                'tau': 1e-3
            },
            'log_alpha':{
                'learning_rate': 3e-5,
                'target_entropy': - dummy_action.shape[0]
            },
            'training':{
                'gamma' : 0.99,
                'learning_starts':10_000,
                'her_ratio' : 4,
                'improvement_ratio' : 0.05,
                'patience' : 40,
                'batch_size' : 1024,
                'replay_buffer_size' : 1_000_000,
                'improvement_buffer_size' : 20,
                'evaluation_steps': 50
            }
        }

        with open(os.path.join(result_folder,'parameters.json'), 'w') as fp:
            json.dump(infos, fp)

        best_success_ratio = -np.inf

    else:
        with open(os.path.join(result_folder,'parameters.json')) as json_file:
            infos = json.load(json_file)

        with open(os.path.join(result_folder,'success_ratio.csv'), 'r') as f:
            reader = csv.DictReader(f)
            best_success_ratio = max(float(row['Success Ratio']) for row in reader)


    buffer = ReplayBuffer(infos['training']['replay_buffer_size'], obs, dummy_action.shape[0])
    improvement_buffer = ImprovementBuffer(infos['training']['improvement_buffer_size'])

    agent = Agent(infos)
    if new_run is False:
        agent.load(result_folder)
        with open(os.path.join(result_folder,'losses.csv'), 'r') as f:
            old_run_time_step = sum(1 for line in f) - 1
        with open(os.path.join(result_folder,'success_ratio.csv'), 'r') as f:
            old_run_evaluation_step = sum(1 for line in f) - 1
    else:
        old_run_time_step = 0
        old_run_evaluation_step = 0

    max_episodes = 500_000
    progress_bar = tqdm(range(max_episodes))
    improvement_counter = 0    
    evaluation_freq = 5_000
    evaluation_env = gym.make('FetchPush-v4')
    evaluation_step = 0
    writer = SummaryWriter(log_dir = result_folder)
    time_step = 0

    for episode in progress_bar:
        progress_bar.set_description(f'Episode {episode}')
        state, _ = env.reset()
        trajectory = []
        done = False
        while not done:
            action = agent.select_action(np.concatenate([state['observation'],state['desired_goal']]))
            next_state, reward, terminal, truncated, env_info = env.step(action) 
            buffer.store(state, next_state, action, reward, terminal, truncated)
            trajectory.append({
                'current_observation': state,
                'next_observation': next_state,
                'action' : action,
                'reward': reward,
                'terminal': terminal,
                'truncated': truncated,
                'info' : env_info
            })
            state = next_state
            time_step += 1
            if terminal or truncated:
                done = True

        her(trajectory, buffer, env, infos['training']['her_ratio'])

        if buffer.get_size() >= infos['training']['learning_starts']:
            for _ in range(len(trajectory)):
                states, next_states, actions, rewards, terminals, truncated = buffer.sample(infos['training']['batch_size'])
                critic_loss = agent.update_online_critic(states, next_states, actions, rewards, terminals, infos['training']['gamma'])
                actor_loss = agent.update_actor(states)
                log_alpha_loss = agent.update_log_alpha(states)
                agent.soft_update()
                writer.add_scalar('Loss/actor', actor_loss, time_step + old_run_time_step)
                writer.add_scalar('Loss/critic', critic_loss, time_step + old_run_time_step)
                writer.add_scalar('Loss/log_alpha', log_alpha_loss, time_step + old_run_time_step)

                write_csv(os.path.join(result_folder,'losses.csv'),{'Critic Loss': critic_loss,
                                                            'Actor Loss': actor_loss,
                                                            'Log_Alpha Loss': log_alpha_loss})

        if time_step > 0 and (time_step%evaluation_freq)==0:
            success_ratio = evauate_agent(agent, evaluation_env, infos['training']['evaluation_steps'])
            improvement_buffer.store(success_ratio)
            current_mean = improvement_buffer.mean()
            writer.add_scalar('Success Ratio', success_ratio, evaluation_step + old_run_evaluation_step)
            write_csv(os.path.join(result_folder,'success_ratio.csv'),{'Success Ratio': success_ratio})
            evaluation_step += 1

            if success_ratio >= best_success_ratio:
                best_success_ratio = success_ratio
                agent.save(result_folder)
            
            if improvement_buffer.full and best_success_ratio > 0.8:
                if abs(best_success_ratio - current_mean) <= infos['training']['improvement_ratio']:
                    improvement_counter += 1
                else:
                    improvement_counter = 0           
            
            if improvement_counter > infos['training']['patience']:
                print('Training completed.')
                break
                    
    writer.close() 

def her(trajectory: list, buffer: ReplayBuffer, env: gym.Env, her_ratio: int):
    """Augment the replay buffer with Hindsight Experience Replay (future strategy).

    For k randomly selected steps in the episode, replaces the desired goal with
    a goal achieved at a future step, recomputes the reward, and stores the
    relabelled transition in the buffer. This provides positive reward signal
    even from failed episodes.

    Args:
        buffer: Replay buffer to augment in-place.
        env: Training environment, used to recompute the reward after relabelling.
        trajectory: List of transition dicts from the current episode, each containing
            'current_observation', 'next_observation', 'action', 'reward', 'terminal', 'truncated', 'info'.
        her_ratio: Number of relabelled transitions to generate per episode.
    """
    indices = random.sample(range(len(trajectory)-1), her_ratio)
    for index in indices:
        sample = copy.deepcopy(trajectory[index])
        future_index = random.randint(index+1, len(trajectory)-1)
        sample['current_observation']['desired_goal'] = trajectory[future_index]['current_observation']['achieved_goal']
        sample['next_observation']['desired_goal'] =  trajectory[future_index]['current_observation']['achieved_goal']
        reward = env.unwrapped.compute_reward(sample['current_observation']['achieved_goal'],
                                              trajectory[future_index]['current_observation']['achieved_goal'],
                                              sample['info'])
        buffer.store(sample['current_observation'], sample['next_observation'],sample['action'], reward, sample['terminal'], sample['truncated'])

def evauate_agent(agent: Agent, env: gym.Env, n_episodes: int):
    """Evaluate the agent over a fixed number of episodes and return the success rate.

    Args:
        agent: Trained SAC agent.
        env: Evaluation environment (separate instance from training env).
        n_episodes: Number of full episodes to run.

    Returns:
        Fraction of episodes in which the agent reached the goal (0.0 to 1.0).
    """
    successes = np.zeros(n_episodes)
    for i in range(n_episodes):
        state, _ = env.reset()
        while True:
            action = agent.select_action_deterministic(np.concatenate([state['observation'],state['desired_goal']]))
            next_state, _, _, truncated, info = env.step(action) 
            state = next_state
            if info['is_success']:
                successes[i] = 1
                break
            if truncated:
                break
    return successes.mean()

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

train()
