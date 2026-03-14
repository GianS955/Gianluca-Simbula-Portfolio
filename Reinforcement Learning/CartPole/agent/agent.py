from utils.utils import RolloutBuffer
from network.network import *
import numpy as np
import copy

class A2CAgent:
    def __init__(self, actor_network_info, critic_network_info, agent_parameters, optimizer_parameters, experiment_coefficient, temperature = 1):
        self.actor_network = NeuralNetwork(actor_network_info, optimizer_parameters, name='actor')
        self.actor_network.initialize()
        self.critic_network = NeuralNetwork(critic_network_info, optimizer_parameters, name='critic')
        self.critic_network.initialize()
        self.value_loss_coefficient = agent_parameters['value_loss_coefficient']
        self.entropy_coefficient = agent_parameters['entropy_coefficient']
        self.epsilon = agent_parameters['clip_coefficient']
        self.discount = experiment_coefficient['discount_factor']
        self.lambda_coefficient = agent_parameters['lambda']
        self.buffer_size = agent_parameters['buffer_size']
        self.rollout_steps = agent_parameters['rollout_steps']
        self.temperature = temperature
        self.buffer = RolloutBuffer(actor_network_info['state_dimensions'], actor_network_info['action_dimensions'], actor_network_info['hidden_layer_dimensions'])
        self.history = {'actor_gradients': { 'w':[], 'b':[]}, 'critic_gradients': { 'w':[], 'b':[]}}
    
    def softmax(self,logits, temperature=1):
        shifted = logits -np.max(logits, axis = 1 , keepdims=True)
        probabilities = np.exp(shifted/temperature)/np.sum(np.exp(shifted/temperature), axis=1, keepdims=True)
        return probabilities
    
    def start(self, state):
        self.initial_state = state
        self.last_state = state

    def step(self,environment):
        state = self.last_state
        logits, actor_cache = self.actor_network.forward_pass(self.last_state)
        action_probs = self.softmax(logits)

        action = np.random.choice(len(action_probs[0]), p=action_probs[0])
        log_prob = np.log(action_probs[0,action])

        state_value, critic_cache = self.critic_network.forward_pass(self.last_state)
        next_state, reward, is_terminal, _, _ = environment.step(action)
        one_hot = np.zeros(logits.shape[1])
        one_hot[action] = 1
        self.buffer.append(state, one_hot, reward, state_value, log_prob, action_probs, is_terminal, actor_cache, critic_cache)
        self.last_state = next_state
        return is_terminal, reward
    
    def replay_buffer(self, buffer):
        for i in range(len(buffer)):
            logits, actor_cache =  self.actor_network.forward_pass(buffer.states[i])
            action_probs = self.softmax(logits)
            action = buffer.one_hot[i].argmax()
            log_prob = np.log(action_probs[0,action])
            state_value, critic_cache = self.critic_network.forward_pass(buffer.states[i])
            self.buffer.append(buffer.states[i], buffer.one_hot[i], buffer.rewards[i], state_value, log_prob, action_probs, buffer.is_terminal[i], actor_cache, critic_cache)


    def compute_returns_and_advantage(self):
        size = len(self.buffer)
        deltas = np.zeros(shape=(size, 1))
        last_state_value ,_ = self.critic_network.forward_pass(self.last_state)
        for t in range(size):
            if t == size -1 and self.buffer.is_terminal[t] == False:
                next_state_value = last_state_value
            elif t != size-1:
                next_state_value = self.buffer.state_values[t+1]
            
            mask = 1 - self.buffer.is_terminal[t]
            deltas[t] = self.buffer.rewards[t] + self.discount * mask * next_state_value - self.buffer.state_values[t] 

        # Calculation of advantages
        advantages = np.zeros(shape = (size,1))
        advantage = 0
        for t in reversed(range(size)):
            mask = 1 - self.buffer.is_terminal[t]
            advantage = deltas[t] + self.discount * self.lambda_coefficient * mask * advantage
            advantages[t] = advantage

        # normalization of advantages:
        advantages = (advantages - np.mean(advantages))/(np.std(advantages)+ 1e-8)

        # Calculation of returns:
        actor_returns = advantages + self.buffer.state_values

        # Update RolloutBuffer
        self.buffer.advantages = advantages
        self.buffer.returns = actor_returns

        # # calculation of critic returns:
        value_returns = np.zeros(shape = (size,1))
        temp = 0
        for t in reversed(range(size)):
            temp = self.buffer.rewards[t] + self.discount * temp 
            value_returns[t] = temp
        
        self.buffer.critic_returns = value_returns

    def update_networks(self, old_policy):
        old_policy.compute_returns_and_advantage() 

        self.replay_buffer(old_policy.buffer)

        policy_loss, value_loss, entropy = self.compute_actor_losses(old_policy)
        actor_grad_w, actor_grad_b = self.compute_actor_gradients(old_policy)
        self.actor_network.update_parameters(actor_grad_w, actor_grad_b)

        critic_grad_w, critic_grad_b = self.compute_critic_gradients(old_policy)
        self.critic_network.update_parameters(critic_grad_w, critic_grad_b)
        self.buffer.empty_buffer()
        return policy_loss, value_loss, entropy

    def compute_actor_gradients(self, old_policy):
        
        # derivative of loss on logits
        dLclip_dz = - old_policy.buffer.advantages * self.buffer.ratios * (self.buffer.one_hot - self.buffer.action_probs)
        # clipped1 = (self.buffer.ratios <1-self.epsilon) | (self.buffer.ratios > 1+self.epsilon)
        clipped1  = (old_policy.buffer.advantages > 0) & (self.buffer.ratios > 1 + self.epsilon)
        clipped2 = (old_policy.buffer.advantages <0) & (self.buffer.ratios < 1 - self.epsilon)
        dLclip_dz[clipped1[:,0]] =0 
        dLclip_dz[clipped2[:,0]] =0 
        dH_dz = - self.entropy_coefficient * self.buffer.action_probs * (np.log(self.buffer.action_probs + 1e-8) + 1) # derivative of entropy loss on logits
        dH_dz = - self.buffer.action_probs*(self.buffer.log_probs +1) - np.sum(self.buffer.action_probs * (np.log(self.buffer.action_probs + 1)*self.buffer.action_probs), axis=1, keepdims=True) # derivative of entropy loss on logits

        delta = dLclip_dz + self.entropy_coefficient *dH_dz
        w_gradients = {}
        b_gradients = {}
        for layer in reversed(self.actor_network.w.keys()):
            delta = delta * self.actor_network.activation_derivative(self.buffer.get_actor_cache(layer, 'output'),layer)
            w_gradients[layer] = self.buffer.get_actor_cache(layer, 'input').T @ delta
            b_gradients[layer] = np.sum(delta , axis=0, keepdims=True)

            delta = delta @ self.actor_network.w[layer].T

        self.history['actor_gradients']['w'].append(np.mean(w_gradients[layer]))
        self.history['actor_gradients']['b'].append(np.mean(b_gradients[layer]))
        return w_gradients, b_gradients
    
    def compute_critic_gradients(self, old_policy):
        delta = 2*(self.buffer.state_values - old_policy.buffer.returns)/len(self.buffer)
        w_gradients = {}
        b_gradients = {}
        for layer in reversed(self.critic_network.w.keys()):
            delta = delta * self.critic_network.activation_derivative(self.buffer.get_critic_cache(layer, 'output'),layer)
            w_gradients[layer] = self.buffer.get_critic_cache(layer,'input').T @ delta
            b_gradients[layer] = np.sum(delta , axis=0, keepdims=True)
            delta = delta @ self.critic_network.w[layer].T
        self.history['critic_gradients']['w'].append(np.average(list(w_gradients[layer])))
        self.history['critic_gradients']['b'].append(np.average(list(b_gradients[layer])))
        return w_gradients, b_gradients

    def compute_actor_losses(self, old_policy):      

        self.buffer.ratios = np.exp(self.buffer.log_probs - old_policy.buffer.log_probs)
        clip_loss = -np.minimum(self.buffer.ratios*old_policy.buffer.advantages,np.clip(self.buffer.ratios,1-self.epsilon, 1+self.epsilon)*old_policy.buffer.advantages)
        
        value_loss =(self.buffer.state_values - old_policy.buffer.returns)**2
        entropy_loss = np.sum(self.buffer.action_probs * np.log(self.buffer.action_probs+1e-8), axis = 1)        

        return np.mean(clip_loss), self.value_loss_coefficient * np.mean(value_loss) , -self.entropy_coefficient * np.mean(entropy_loss)
            
    def compute_critic_loss(self, old_policy):
        return np.mean((self.buffer.state_values - old_policy.buffer.returns)**2)

    def get_policy(self):
        return copy.deepcopy(self.actor_network)
    
    def set_policy(self,policy):
        self.actor_network = copy.deepcopy(policy)

    def select_action(self, state):
        logits, _ = self.actor_network.forward_pass(state)
        action_probs = self.softmax(logits)
        action = np.argmax(action_probs)
        return action