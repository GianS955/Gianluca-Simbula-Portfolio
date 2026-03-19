import math
from cartpole_ppo.agent import A2CAgent
import numpy as np
import gymnasium as gym
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from cartpole_ppo.utils import record_agent, moving_average
import random

def run_episode(agent_params, actor_network_info, critic_network_info, optimizer_params, experiment_params, max_episodes, seed, environment = 'CartPole-v1'):

    env = gym.make(environment)
    state, _ = env.reset(seed=seed)

    agent = A2CAgent(actor_network_info, critic_network_info,agent_params,optimizer_params, experiment_params)
    
    history = {}
    episode_rewards = []    
    episode_policy_losses = []
    episode_value_losses = []
    episode_entropies = []
    episode_ratios = []
    episode_advantages = []
    episode_total_losses = []  
    episode_clip_fractions = [] 

    agent.start(state)
    episode = 0
    for episode in tqdm(range(max_episodes)):    
        
        old_agent = copy.deepcopy(agent)

        rewards = []
        cumulative_reward = 0
        for _ in range(agent.rollout_steps):
            is_terminal, reward = old_agent.step(env)
            cumulative_reward += reward
            if is_terminal:
                state, _ = env.reset(seed = seed)
                old_agent.start(state)
                rewards.append(cumulative_reward)
                cumulative_reward = 0
        policy_losses, value_losses, entropies, ratios, advantages = agent.update_networks(old_agent)

        if (math.isnan(np.mean(rewards))):
            episode_rewards.append(0.)
        else:
            episode_rewards.append(np.mean(rewards))
            
        episode_policy_losses.append(np.mean(policy_losses))
        episode_value_losses.append(np.mean(value_losses))
        episode_entropies.append(np.mean(entropies))
        episode_ratios.append(np.mean(ratios))
        episode_advantages.append(np.mean(advantages))
        episode_total_losses.append(np.mean(policy_losses + value_losses + entropies))
        episode_clip_fractions.append(len(ratios[(ratios > 1 + agent.epsilon) | (ratios < 1 - agent.epsilon)]) / len(ratios))        
        
        episode += 1
        
    history['Reward']= episode_rewards
    history['Policy Loss']= episode_policy_losses
    history['Value Loss']= episode_value_losses
    history['Entropy']= episode_entropies
    history['Total Loss']= episode_total_losses
    history['Clip Fraction']= episode_clip_fractions
    history['Ratio']=episode_ratios
    history['Advantage'] = episode_advantages
    env.close()
    return history, agent

def plot_episode_history(history):
    
    for k in history.keys():
        avg_metric = moving_average(history[k],50)
        std_reward = np.std(avg_metric)
        fig,ax = plt.subplots()
        ax.plot(avg_metric, label = k)
        ax.fill_between(range(len(avg_metric)), avg_metric - std_reward, avg_metric + std_reward, alpha = 0.2)
        ax.set_title('Episode ' + k)
        plt.show()

def plot_comparison(results, parameter):

    plots = {}
    for i in results.keys():
        if results[i]['parameter'] == parameter:
            for metric in results[i]['history'].keys():
                if plots.get(metric,None) is None:
                    fig, ax = plt.subplots()
                    plots[metric] = {}
                    plots[metric]['figure'] = fig
                    plots[metric]['ax'] = ax
                    plots[metric]['ax'].set_title(metric)
                    plots[metric]['ax'].set_xlabel('Episode')
                    plots[metric]['ax'].set_ylabel(metric)
                    
                avg_metric = moving_average(results[i]['history'][metric],50)
                std_metric = np.std(avg_metric)

                plots[metric]['ax'].plot(avg_metric, label = f'{parameter} = {results[i]['value']}')                
                plots[metric]['ax'].fill_between(range(len(avg_metric)), avg_metric - std_metric, avg_metric + std_metric, alpha = 0.2)

        else:
            continue

    for metric in plots.keys():
        plots[metric]['ax'].legend()
        plots[metric]['figure'].show()
        plots[metric]['figure'].savefig(f'results\\{parameter}_{metric}.png')
        plt.close(plots[metric]['figure'])

def average_histories(histories):
    history = {}
    for metric in histories[0].keys():
        min_len = min(len(histories[i][metric]) for i in histories)
        matrix = np.array([histories[i][metric][:min_len] for i in histories])
        history[metric] = np.mean(matrix, axis=0)
    return history

def run_experiments():
    experiment_parameters = {
        'learning_rate': [1e-4, 3e-4, 1e-3], # curves: reward, entropy, value loss
        'clip_coefficient': [0.1, 0.2, 0.3], #curves: reward, clip fraction, ratio
        'entropy_coefficient': [0, 0.01, 0.05], # curves: reward, entropy, clip loss
        'lambda': [0.9, 0.95, 0.99], # curves: reward, advantage variance
        'value_loss_coefficient': [0.1, 0.5, 1.], # curves: reward, value loss
        'rollout_steps': [512, 2048, 4096], # curves: reward, ratio variance
        'update_epochs': [3, 4, 5] # curves: reward, clip fraction, ratio
    }

    parameter_baseline = {
        'learning_rate': 3e-4, 
        'discount_factor': 0.99,
        'clip_coefficient': 0.2, 
        'entropy_coefficient': 0.01,
        'lambda': 0.95, 
        'value_loss_coefficient': 0.5, 
        'batch_size': 64,
        'rollout_steps' : 2048,
        'update_epochs' : 4
    }

    seeds = [0, 1, 2, 4, 5]

    results = {}
    experiment_counter = 0
    for param in experiment_parameters.keys():
        for value in experiment_parameters[param]:
            print(f'Running experiment with {param} = {value}')
            env = gym.make('CartPole-v1')

            actor_network_info ={'state_dimensions': env.observation_space.shape[0],
                            'hidden_layer_dimensions':[64],
                            'action_dimensions':env.action_space.n,
                            'activation':'tanh',
                            }
                
            critic_network_info ={'state_dimensions': env.observation_space.shape[0],
                                'hidden_layer_dimensions':[64],
                                'activation':'tanh'}

            agent_params = {'batch_size' : parameter_baseline['batch_size'],
                            'rollout_steps': value if param == 'rollout_steps' else parameter_baseline['rollout_steps'],
                            'value_loss_coefficient': value if param == 'value_loss_coefficient' else parameter_baseline['value_loss_coefficient'],
                            'entropy_coefficient': value if param == 'entropy_coefficient' else parameter_baseline['entropy_coefficient'],
                            'clip_coefficient': value if param == 'clip_coefficient' else parameter_baseline['clip_coefficient'],
                            'lambda': value if param == 'lambda' else parameter_baseline['lambda'],
                            'update_epochs': value if param == 'update_epochs' else parameter_baseline['update_epochs']}

            optimizer_params = {'learning_rate': value if param == 'learning_rate' else parameter_baseline['learning_rate'], 
                                'beta_1': 0.9, 
                                'beta_2': 0.999}
            experiment_params = {'discount_factor':parameter_baseline['discount_factor']}

            env.close()
            histories = {}
            for current_seed in seeds:
                print(f'Running seed: {current_seed}')
                np.random.seed(current_seed)
                random.seed(current_seed)
                
                histories[current_seed], _ = run_episode(agent_params,actor_network_info, critic_network_info, optimizer_params, experiment_params,1000,current_seed)
            
            results[experiment_counter] = {
                'parameter' : param,
                'value' : value,
                'history' : average_histories(histories)
            }

            experiment_counter += 1

        plot_comparison(results, param)

def play_episode():
    env = gym.make('CartPole-v1')
    parameter_baseline = {
        'learning_rate': 3e-4, 
        'discount_factor': 0.99,
        'clip_coefficient': 0.2, 
        'entropy_coefficient': 0.01,
        'lambda': 0.99, 
        'value_loss_coefficient': 0.5, 
        'batch_size': 64,
        'rollout_steps' : 2048,
        'update_epochs' : 4
    }
    actor_network_info ={'state_dimensions': env.observation_space.shape[0],
                    'hidden_layer_dimensions':[64],
                    'action_dimensions':env.action_space.n,
                    'activation':'tanh',
                    }
                
    critic_network_info ={'state_dimensions': env.observation_space.shape[0],
        'hidden_layer_dimensions':[64],
        'activation':'tanh'}

    agent_params = {'batch_size' : parameter_baseline['batch_size'],
                    'rollout_steps': parameter_baseline['rollout_steps'],
                    'value_loss_coefficient': parameter_baseline['value_loss_coefficient'],
                    'entropy_coefficient': parameter_baseline['entropy_coefficient'],
                    'clip_coefficient': parameter_baseline['clip_coefficient'],
                    'lambda': parameter_baseline['lambda'],
                    'update_epochs': parameter_baseline['update_epochs']}

    optimizer_params = {'learning_rate': parameter_baseline['learning_rate'], 
                        'beta_1': 0.9, 
                        'beta_2': 0.999}
    experiment_params = {'discount_factor':parameter_baseline['discount_factor']}

    _, agent = run_episode(agent_params,actor_network_info, critic_network_info, optimizer_params, experiment_params,5000,5)
            
    record_agent(agent)