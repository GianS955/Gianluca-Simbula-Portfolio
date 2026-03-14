from csv import writer
from functools import cache
import math
import os
import gymnasium as gym
import imageio
import numpy as np

class RolloutBuffer:
    def __init__ (self, num_states, num_actions,layers):
        self.num_states = num_states
        self.num_actions = num_actions
        self.buffer = {}
        # self.layers = layers    
        # self.states = np.zeros(shape = (1, num_states))
        # self.one_hot = np.zeros(shape = (1, num_actions))
        # self.rewards = np.zeros(shape = (1, 1))
        # self.state_values = np.zeros(shape = (1, 1))
        # self.log_probs = np.zeros(shape = (1, 1))
        # self.action_probs = np.zeros(shape = (1, num_actions))
        # self.is_terminal = np.zeros(shape = (1, 1))
        # self.advantages = np.zeros(shape = (1, 1))
        # self.returns = np.zeros(shape = (1, 1))
        # self.ratios = np.zeros(shape = (1, 1))
        # self.critic_returns = np.zeros(shape = (1, 1))
        # self.actor_cache = {}
        # for i in range(len(self.layers)):
        #     self.actor_cache[i] = {"input": np.zeros(shape = (1, self.layers[i])), "output": np.zeros(shape = (1, self.layers[i]))}  
        # self.actor_cache = {}
        # for i in range(len(self.layers)):
        #     self.actor_cache[i] = {"input": np.zeros(shape = (1, self.layers[i])), "output": np.zeros(shape = (1, self.layers[i]))}  
        # self.critic_cache = {}
        # for i in range(len(self.layers)):
        #     self.critic_cache[i] = {"input": np.zeros(shape = (1, self.layers[i])), "output": np.zeros(shape = (1, self.layers[i]))}  
        # self.current_index = 0
    
    def __len__(self):
        return len(self.buffer[0]['state'])
    
    def append(self, state, action, reward, state_value, log_prob, action_probs, is_terminal, actor_cache, critic_cache):
        self.buffer[self.current_index] = {}
        self.buffer[self.current_index]['state'] = state.reshape(1,-1)
        self.buffer[self.current_index]['one_hot'] = action.reshape(1,-1)
        self.buffer[self.current_index]['rewards'] = np.array([[reward]])
        self.buffer[self.current_index]['state_values'] = np.array([[float(state_value)]])
        self.buffer[self.current_index]['log_probs'] = np.array([[log_prob]])
        self.buffer[self.current_index]['action_probs'] = action_probs.reshape(1,-1)
        self.buffer[self.current_index]['is_terminal'] = np.array([[is_terminal]])
        self.buffer[self.current_index]['actor_cache'] = {}
        for i in list(actor_cache.keys()):
            self.buffer[self.current_index]['actor_cache'][i] = {}
            self.buffer[self.current_index]['actor_cache'][i]['input'] = np.reshape(actor_cache[i]['input'], (1, -1))
            self.buffer[self.current_index]['actor_cache'][i]['output'] = np.reshape(actor_cache[i]['output'], (1, -1))
        self.buffer[self.current_index]['critic_cache'] = {}
        for i in list(critic_cache.keys()):
            self.buffer[self.current_index]['critic_cache'][i] = {}
            self.buffer[self.current_index]['critic_cache'][i]['input'] = np.reshape(critic_cache[i]['input'], (1, -1))
            self.buffer[self.current_index]['critic_cache'][i]['output'] = np.reshape(critic_cache[i]['output'], (1, -1))
        # if self.current_index == 0:
        #     self.states = state.reshape(1,-1)
        #     self.one_hot = action.reshape(1,-1)
        #     self.rewards = np.array([[reward]])
        #     self.state_values = np.array([[float(state_value)]])
        #     self.log_probs = np.array([[log_prob]])
        #     self.action_probs = action_probs.reshape(1,-1)
        #     self.is_terminal = np.array([[is_terminal]])
        #     for i in list(actor_cache.keys()):
        #         self.actor_cache[i] = {}
        #         self.actor_cache[i]['input'] = np.reshape(actor_cache[i]['input'], (1, -1))
        #         self.actor_cache[i]['output'] = np.reshape(actor_cache[i]['output'], (1, -1))
        #     for i in list(critic_cache.keys()):
        #         self.critic_cache[i] = {}
        #         self.critic_cache[i]['input'] = np.reshape(critic_cache[i]['input'], (1, -1))
        #         self.critic_cache[i]['output'] = np.reshape(critic_cache[i]['output'], (1, -1))
        # else:
        #     self.states = np.vstack([self.states, state.reshape(1,-1)])
        #     self.one_hot = np.vstack([self.one_hot, action.reshape(1,-1)])
        #     self.rewards = np.vstack([self.rewards, [[reward]]])
        #     self.state_values = np.vstack([self.state_values, [float(state_value)]])
        #     self.log_probs = np.vstack([self.log_probs, [[log_prob]]])
        #     self.action_probs = np.vstack([self.action_probs, action_probs.reshape(1,-1)])
        #     self.is_terminal = np.vstack([self.is_terminal, [[is_terminal]]])
        #     for i in list(actor_cache.keys()):
        #         self.actor_cache[i]['input'] = np.vstack([self.actor_cache[i]['input'], np.reshape(actor_cache[i]['input'], (1, -1))])
        #         self.actor_cache[i]['output'] = np.vstack([self.actor_cache[i]['output'], np.reshape(actor_cache[i]['output'], (1, -1))])
        #     for i in list(critic_cache.keys()):
        #         self.critic_cache[i]['input'] = np.vstack([self.critic_cache[i]['input'], np.reshape(critic_cache[i]['input'], (1, -1))])
        #         self.critic_cache[i]['output'] = np.vstack([self.critic_cache[i]['output'], np.reshape(critic_cache[i]['output'], (1, -1))])
        self.current_index += 1

    def get(self, key):
        return np.vstack([self.buffer[i][key] for i in range(len(self))])
    
    def empty_buffer(self):
        # self.states = np.zeros(shape = (1, self.num_states))
        # self.one_hot = np.zeros(shape = (1, self.num_actions))
        # self.rewards = np.zeros(shape = (1, 1))
        # self.state_values = np.zeros(shape = (1, 1))
        # self.log_probs = np.zeros(shape = (1, 1))
        # self.action_probs = np.zeros(shape = (1, self.num_actions))
        # self.is_terminal = np.zeros(shape = (1, 1))
        # self.advantages = np.zeros(shape = (1, 1))
        # self.returns = np.zeros(shape = (1, 1))
        # self.ratios = np.zeros(shape = (1, 1))
        # self.critic_returns = np.zeros(shape = (1,1))
        # self.actor_cache = {}
        # for i in range(len(self.layers)):
        #     self.actor_cache[i] = {"input": np.zeros(shape = (1, self.layers[i])), "output": np.zeros(shape = (1, self.layers[i]))}  
        # self.actor_cache = {}
        # for i in range(len(self.layers)):
        #     self.actor_cache[i] = {"input": np.zeros(shape = (1, self.layers[i])), "output": np.zeros(shape = (1, self.layers[i]))}  
        # self.critic_cache = {}
        # for i in range(len(self.layers)):
        #     self.critic_cache[i] = {"input": np.zeros(shape = (1, self.layers[i])), "output": np.zeros(shape = (1, self.layers[i]))}  
        self.current_index = 0
        self.buffer = {}

    def get_indexes(self):
        return [i for i in range(len(self))]
    
    def get_actor_cache(self, layer, cache_type):
        return self.actor_cache[layer][cache_type]
    def get_critic_cache(self, layer, cache_type):
        return self.critic_cache[layer][cache_type]
    
    def get_minibatch(self, batch_size):
        N = len(self)
        indices = np.random.permutation(N)

        return {k: self.buffer[k] for k in indices}
        for start in range(0,N,batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield{
                self.get('state')[batch_idx],
                self.get('one_hot')[batch_idx],
                self.get('rewards')[batch_idx],
                self.get('state_values')[batch_idx],
                self.get('log_probs')[batch_idx],
                self.get('action_probs')[batch_idx],
                self.get('is_terminal')[batch_idx],
                self.get('advantages')[batch_idx],
                self.get('returns')[batch_idx],
                self.get('ratios')[batch_idx],
            }

class Optimizer:
    def __init__(self, optimization_info):
        self.learning_rate = optimization_info.get('learning_rate', 0.001)
        self.beta_1 = optimization_info.get('beta_1', 0.9)
        self.beta_2 = optimization_info.get('beta_2', 0.999)
        self.m = {}
        self.v = {}

    def optimize_gradient(self, gradient, layer, parameter_type):
        if layer not in self.m:
            self.m[layer] = {}
            
        if layer not in self.v:
            self.v[layer] = {}

        if parameter_type not in self.m[layer]:
            self.m[layer][parameter_type] = np.zeros_like(gradient)
        if parameter_type not in self.v[layer]:
            self.v[layer][parameter_type] = np.zeros_like(gradient)
            
        self.m[layer][parameter_type] = self.beta_1 * self.m[layer][parameter_type] + (1 - self.beta_1) * gradient
        self.v[layer][parameter_type] = self.beta_2 * self.v[layer][parameter_type] + (1 - self.beta_2) * (gradient ** 2)
        m_hat = self.m[layer][parameter_type] / (1 - self.beta_1)
        v_hat = self.v[layer][parameter_type] / (1 - self.beta_2)
        return - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    

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
    
    print(f"\nTotale frames raccolti: {len(frames)}")
    print(f"Frames None: {sum(1 for f in frames if f is None)}")
    
    # Filtra eventuali None
    frames = [f for f in frames if f is not None]
    frames = [np.array(f, dtype=np.uint8) for f in frames]
    
    if len(frames) == 0:
        print("ERRORE: nessun frame valido!")
        return
    
    print(f"Shape primo frame: {frames[0].shape}")
    

    # Prova 2: mp4 con fps bassi per allungare la durata visiva
    imageio.mimsave(output_path, frames, fps=30)

def moving_average(data, window_size = 50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')