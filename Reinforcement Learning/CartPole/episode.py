import math
from agent.agent import A2CAgent
import numpy as np
import gymnasium as gym
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import record_agent, moving_average
import random

def run_episode(agent_params, actor_network_info, critic_network_info, optimizer_params, experiment_params, max_episodes, seed, environment = 'CartPole-v1'):

    env = gym.make(environment)
    state, _ = env.reset(seed)

    agent = A2CAgent(actor_network_info, critic_network_info,agent_params,optimizer_params, experiment_params)
    history = {}

    agent.start(state)
    episode = 0
    for episode in tqdm(range(max_episodes)):    
        
        old_agent = copy.deepcopy(agent)
        rewards = []
        for _ in range(agent.rollout_steps):
            is_terminal, reward = old_agent.step(env)
            rewards.append(reward)
            if is_terminal:
                break
        policy_loss, value_loss, entropy = agent.update_networks(old_agent)
        
        if episode % 100 == 0:
            history[episode] = {'Total Reward':sum(rewards),
                                'Average Reward':np.mean(rewards),
                                'Reward Variance':np.std(rewards),
                                'Policy Loss': policy_loss,
                                'Value Loss': value_loss,
                                'Entropy': entropy,
                                'Total Loss': policy_loss + value_loss - entropy,
                                'Clip Fraction' : np.mean(agent.buffer.ratios > agent.epsilon +1 | agent.buffer.ratios < agent.epsilon +1),
                                'Ratio' : agent.buffer.ratios,
                                'Advantage Variance': np.std(old_agent.buffer.advantages),
                                'Ratio Variance' : np.std(agent.buffer.ratios),
                                'Mean Ratio' : np.mean(agent.buffer.ratios)
                                }
        state, _ = env.reset(seed)
        agent.start(state)
        episode += 1
        
    env.close()
    return history

def plot_history(history, ):
    fig, axs = plt.subplot(2,3, figsize = (16,8))
    axs[0,0].plot(history['total_reward'], alpha = 0.3)
    axs[0,0].plot(moving_average(history['total_reward'],50))
    axs[0,0].set_title('Episode Reward')

epxeriment_parameters = {
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
    'gamma': 0.99,
    'clip_coefficient': 0.2, 
    'entropy_coefficient': 0.01,
    'lambda': 0.95, 
    'value_loss_coefficient': 0.5, 
    'batch_size': 64,
    'rollout_steps' : 2048,
    'update_epochs' : 4
}

seeds = [0, 1, 2, 3, 4]

for current_seed in seeds:
    np.random.seed(current_seed)
    random.seed(current_seed)


buffer_size = 100
env = gym.make('CartPole-v1')
state, info = env.reset()

state[2] = math.pi
actor_network_info ={'state_dimensions': env.observation_space.shape[0],
                     'hidden_layer_dimensions':[64],
                     'action_dimensions':env.action_space.n,
                     'activation':'tanh',
                     }

critic_network_info ={'state_dimensions': env.observation_space.shape[0],
                     'hidden_layer_dimensions':[64],
                     'activation':'tanh'}

agent_params = {'buffer_size' : 100,
                'rollout_steps': 1,
                'value_loss_coefficient': 0.5,
                'entropy_coefficient':0.01,
                'clip_coefficient':0.2,
                'lambda':0.95,
                'update_epochs': 4}

optimizer_params = {'learning_rate': 1e-4, 
                    'beta_1': 0.9, 
                    'beta_2': 0.999}
experiment_params = {'discount_factor':0.8}

agent = A2CAgent(actor_network_info, critic_network_info,agent_params,optimizer_params, experiment_params)
results = {}

agent.start(state)
episode = 0
max_episodes = 10000
for episode in tqdm(range(max_episodes)):    
    
    old_agent = copy.deepcopy(agent)
    rewards = []
    for _ in range(buffer_size):
        is_terminal, reward = old_agent.step(env)
        rewards.append(reward)
        if is_terminal:
            break
    clip_loss, value_loss, entropy_loss = agent.update_networks(old_agent)
    
    if episode % 100 == 0:
        results[episode] = {'total_reward':sum(rewards),
                            'average_reward':np.mean(rewards),
                            'reward_std':np.std(rewards),
                            'clip_loss': clip_loss,
                            'value_loss': value_loss,
                            'entropy_loss': entropy_loss,
                            'total_loss': clip_loss + value_loss - entropy_loss }
    state, _ = env.reset()
    agent.start(state)
    episode += 1
    
env.close()

# Recording the agent's performance after training

state, info = env.reset()
record_agent(agent)


plt.figure(figsize=(12,8))
plt.plot([agent.history['actor_gradients']['w'][i] for i in range(len(agent.history['actor_gradients']['w']))], label='Actor w gradients')
plt.show()
plt.figure(figsize=(12,8))
plt.plot([agent.history['critic_gradients']['w'][i] for i in range(len(agent.history['critic_gradients']['w']))], label='Critic w gradients')
plt.show()
plt.figure(figsize=(12,8))
plt.plot([results[ep]['clip_loss'] for ep in results.keys()])
plt.show()
plt.figure(figsize=(12,8))
plt.plot([results[ep]['value_loss'] for ep in results.keys()])
plt.show()
plt.figure(figsize=(12,8))
plt.plot([results[ep]['entropy_loss'] for ep in results.keys()])
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2)


ax1.plot([results[ep]['total_reward'] for ep in results.keys()])
fig.suptitle('Total Reward per Episode')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax2.plot([results[ep]['total_loss'] for ep in results.keys()])
fig.suptitle('Total Loss per Episode')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Total Loss')
plt.show()
a=0