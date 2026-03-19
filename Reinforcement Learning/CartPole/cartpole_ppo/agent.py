from buffer import RolloutBuffer
from network import NeuralNetwork
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
        self.batch_size = agent_parameters['batch_size']
        self.rollout_steps = agent_parameters['rollout_steps']
        self.update_epochs = agent_parameters['update_epochs']
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
    
    def replay_buffer(self, buffer, indices):
        for index in indices:
            state = buffer.get('state',index)
            logits, actor_cache =  self.actor_network.forward_pass(state)
            action_probs = self.softmax(logits)
            action = buffer.get('one_hot',index).argmax()
            log_prob = np.log(action_probs[0,action])
            state_value, critic_cache = self.critic_network.forward_pass(state)
            self.buffer.append(state, 
                               buffer.get('one_hot',index), 
                               buffer.get('reward',index), 
                               state_value, 
                               log_prob, 
                               action_probs, 
                               buffer.get('is_terminal',index), 
                               actor_cache, 
                               critic_cache,
                               index)

    def compute_returns_and_advantage(self):
        size = len(self.buffer)
        deltas = np.zeros(shape=(size, 1))
        last_state_value ,_ = self.critic_network.forward_pass(self.last_state)
        for t in range(size):
            # if t == size -1 and self.buffer.get('is_terminal',t) == False:
            if t == size -1:
                next_state_value = last_state_value
            else:
                next_state_value = self.buffer.get('state_value',t+1)
            
            mask = 1 - self.buffer.get('is_terminal',t)
            deltas[t] = self.buffer.get('reward',t) + self.discount * mask * next_state_value - self.buffer.get('state_value',t)

        # Calculation of advantages
        advantages = np.zeros(shape = (size,1))
        advantage = 0
        for t in reversed(range(size)):
            mask = 1 - self.buffer.get('is_terminal',t)
            advantage = deltas[t] + self.discount * self.lambda_coefficient * mask * advantage
            advantages[t] = advantage

        # normalization of advantages:
        advantages = (advantages - np.mean(advantages))/(np.std(advantages)+ 1e-8)

        # Calculation of returns:
        actor_returns = advantages + self.buffer.get('state_value')

        # Update RolloutBuffer
        self.buffer.set(advantages,'advantage')
        self.buffer.set(actor_returns,'return')

        # Calculation of critic returns:
        value_returns = np.zeros(shape = (size,1))
        temp = 0
        for t in reversed(range(size)):
            temp = self.buffer.get('reward',t) + self.discount * temp 
            value_returns[t] = temp
        
        self.buffer.set(value_returns,'critic_return')

    def update_networks(self, old_policy):

        policy_losses = np.zeros(shape = (self.update_epochs * self.batch_size,1))
        value_losses = np.zeros(shape = (self.update_epochs * self.batch_size,1))
        entropies = np.zeros(shape = (self.update_epochs * self.batch_size,1))
        ratios = np.zeros(shape = (self.update_epochs * self.batch_size,1))
        advantages =  [np.zeros(shape = (self.update_epochs * self.batch_size,1))]
        old_policy.compute_returns_and_advantage()
        for i in range(self.update_epochs):
            indices = old_policy.buffer.get_minibatch(self.batch_size)

            self.replay_buffer(old_policy.buffer, indices)
            policy_loss, value_loss, entropy = self.compute_actor_losses(old_policy.buffer,indices)
            actor_grad_w, actor_grad_b = self.compute_actor_gradients(old_policy.buffer, indices)
            self.actor_network.update_parameters(actor_grad_w, actor_grad_b)

            critic_grad_w, critic_grad_b = self.compute_critic_gradients(old_policy.buffer, indices)
            self.critic_network.update_parameters(critic_grad_w, critic_grad_b)

            index = i*self.batch_size
            policy_losses[index:index + self.batch_size] = policy_loss.reshape(self.batch_size,1)
            value_losses[index:index + self.batch_size] = value_loss.reshape(self.batch_size,1)
            entropies[index:index + self.batch_size] = entropy.reshape(self.batch_size,1)
            ratios[index:index + self.batch_size] = self.buffer.get('ratio',indices).reshape(self.batch_size,1)
            advantages[index:index + self.batch_size] = old_policy.buffer.get('advantage',indices).reshape(self.batch_size,1)
        
            self.buffer.empty_buffer()
        
        return policy_losses, value_losses, entropies, ratios, advantages

    def compute_actor_gradients(self, old_buffer,indices = None):
        
        # derivative of loss on logits
        advantages = old_buffer.get('advantage',indices)
        ratios = self.buffer.get('ratio',indices)
        one_hot = self.buffer.get('one_hot',indices)
        action_probs = self.buffer.get('action_prob',indices)
        log_probs = self.buffer.get('log_prob',indices)

        dLclip_dz = - advantages * ratios * (one_hot - action_probs)
        # clipped1 = (self.buffer.ratios <1-self.epsilon) | (self.buffer.ratios > 1+self.epsilon)
        clipped1  = (advantages > 0) & (ratios > 1 + self.epsilon)
        clipped2 = (advantages <0) & (ratios < 1 - self.epsilon)
        dLclip_dz[clipped1[:,0]] =0 
        dLclip_dz[clipped2[:,0]] =0 
        # dH_dz = - self.entropy_coefficient * action_probs * (np.log(action_probs + 1e-8) + 1) # derivative of entropy loss on logits
        dH_dz = - action_probs*(log_probs +1) - np.sum(action_probs * (np.log(action_probs + 1)*action_probs), axis=1, keepdims=True) # derivative of entropy loss on logits

        log_action_probs = np.log(action_probs + 1e-8)
        entropy_vec = -np.sum(action_probs * log_action_probs, axis=1, keepdims=True)
        dH_dz = action_probs * (entropy_vec - log_action_probs - 1)

        delta = dLclip_dz + self.entropy_coefficient *dH_dz
        w_gradients = {}
        b_gradients = {}
        for layer in reversed(self.actor_network.w.keys()):
            delta = delta * self.actor_network.activation_derivative(self.buffer.get_actor_cache(layer, 'output',indices),layer)
            w_gradients[layer] = self.buffer.get_actor_cache(layer, 'input', indices).T @ delta
            b_gradients[layer] = np.sum(delta , axis=0, keepdims=True)

            delta = delta @ self.actor_network.w[layer].T

        self.history['actor_gradients']['w'].append(np.mean(w_gradients[layer]))
        self.history['actor_gradients']['b'].append(np.mean(b_gradients[layer]))
        return w_gradients, b_gradients
    
    def compute_critic_gradients(self, old_buffer, indices = None):
        returns = old_buffer.get('return',indices)
        delta = 2*(self.buffer.get('state_value',indices) - returns)/len(indices)
        w_gradients = {}
        b_gradients = {}
        for layer in reversed(self.critic_network.w.keys()):
            delta = delta * self.critic_network.activation_derivative(self.buffer.get_critic_cache(layer, 'output',indices),layer)
            w_gradients[layer] = self.buffer.get_critic_cache(layer,'input', indices).T @ delta
            b_gradients[layer] = np.sum(delta , axis=0, keepdims=True)
            delta = delta @ self.critic_network.w[layer].T
        self.history['critic_gradients']['w'].append(np.average(list(w_gradients[layer])))
        self.history['critic_gradients']['b'].append(np.average(list(b_gradients[layer])))
        return w_gradients, b_gradients

    def compute_actor_losses(self, old_buffer, indices):      

        old_log_probs = old_buffer.get('log_prob',indices)
        new_log_probs = self.buffer.get('log_prob',indices)
        ratios = np.exp(new_log_probs - old_log_probs)
        self.buffer.set(ratios,'ratio',indices)
        advantages = old_buffer.get('advantage',indices)
        clip_loss = -np.minimum(ratios*advantages,np.clip(ratios,1-self.epsilon, 1+self.epsilon)*advantages)
        
        returns = old_buffer.get('return',indices)
        value_loss =(self.buffer.get('state_value',indices) - returns)**2
        action_probs = self.buffer.get('action_prob',indices)
        entropy_loss = np.sum(action_probs * np.log(action_probs+1e-8), axis = 1)        

        return clip_loss, self.value_loss_coefficient * value_loss , -self.entropy_coefficient * entropy_loss
            
    def compute_critic_loss(self, old_policy,indices):
        return np.mean((self.buffer.state_values - old_policy.buffer.get('critic_returns', indices))**2)

    def get_policy(self):
        return copy.deepcopy(self.actor_network)
    
    def set_policy(self,policy):
        self.actor_network = copy.deepcopy(policy)

    def select_action(self, state):
        logits, _ = self.actor_network.forward_pass(state)
        action_probs = self.softmax(logits)
        action = np.argmax(action_probs)
        return action