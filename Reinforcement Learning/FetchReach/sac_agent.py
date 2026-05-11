from network import Actor, TwinCritic
import torch
import torch.nn as nn
import copy
import numpy as np
import os

class Agent:
    """SAC Agent with twin critic, target network, and automatic entropy tuning.

    Attributes:
        actor: Gaussian actor network with reparameterisation trick and tanh squashing.
        critic_online: TwinCritic online network used for gradient updates.
        critic_target: TwinCritic target network updated via soft update (no gradients).
        log_alpha: Learnable log-scale entropy coefficient.
        alpha: Entropy coefficient (exp of log_alpha), used in actor and critic updates.
        actor_optimizer: Adam optimizer for the actor network.
        critic_online_optimizer: Adam optimizer for the online critic network.
        log_alpha_optimizer: Adam optimizer for log_alpha.
        target_entropy: Target entropy for automatic temperature tuning, set to -action_dims.
        loss_MSE: MSE loss function used in critic updates.
    """
    def __init__(self, info: dict):
        self.actor = Actor(info['actor'])

        self.critic_online = TwinCritic(info['critic'])
        self.critic_target = copy.deepcopy(self.critic_online) # target critic is initialized as the same as the online one

        # the critic_target is updated without propagation, see further method for its update
        # Therefore it is necessary to disable its gradients:
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.log_alpha = torch.zeros(1, requires_grad=True) # it is a scalar that needs to be updated throughout the training. Log is used to have always positive values
        self.alpha = self.log_alpha.exp().item()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = info['actor']['learning_rate']) 
        self.critic_online_optimizer = torch.optim.Adam(self.critic_online.parameters(), lr = info['critic']['learning_rate']) 
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = info['alpha']['learning_rate']) 

        self.target_entropy = -info['target_entropy'] # Euristic standard sets it to -action_dims

        self.loss_MSE = nn.MSELoss() 

    def update_critic(self, current_states: torch.Tensor, 
                      next_states: torch.Tensor, 
                      current_actions: torch.Tensor, 
                      rewards: torch.Tensor, 
                      terminals: torch.Tensor, 
                      gamma: float):
        """Update the TwinCritic online network using clipped double Q-learning.

        Args:
            current_states: Batch of current states, shape (batch_size, obs_dim + goal_dim).
            next_states: Batch of next states, same shape as current_states.
            current_actions: Batch of actions taken at current_states, shape (batch_size, action_dim).
            rewards: Batch of received rewards, shape (batch_size, 1).
            terminals: Batch of terminal flags, shape (batch_size, 1).
            gamma: Discount factor.

        Returns:
            Combined MSE loss of both critics as a float.
        """
        with torch.no_grad():
            next_actions, logs_action = self.actor.forward(next_states)
            # It is necessary to compute the target:
            target_input = torch.cat([next_states, next_actions], dim = 1)
            q1_target, q2_target = self.critic_target.forward(target_input)
            q_minimum = torch.minimum(q1_target,q2_target)
            y = rewards + gamma * (1 - terminals) * (q_minimum - self.alpha * logs_action)

        online_input = torch.cat([current_states,current_actions], dim = 1)
        q1_online, q2_online = self.critic_online.forward(online_input)
        
        loss_q1 = self.loss_MSE(q1_online, y)
        loss_q2 = self.loss_MSE(q2_online, y)

        loss = loss_q1 + loss_q2
        self.critic_online_optimizer.zero_grad()
        loss.backward()
        self.critic_online_optimizer.step()
        return loss.item()

    def update_actor(self, current_states: torch.Tensor):
        """Update the actor network by maximising expected Q-value minus entropy cost.

        Args:
            current_states: Batch of current states, shape (batch_size, obs_dim + goal_dim).

        Returns:
            Mean actor loss as a float.
        """
        current_actions, current_log_probs = self.actor.forward(current_states)
        
        critic_input = torch.cat([current_states, current_actions],dim = 1)
        q1, q2 = self.critic_online.forward(critic_input)
            
        loss = self.alpha * current_log_probs - torch.minimum(q1, q2)
        self.actor_optimizer.zero_grad()
        loss.mean().backward()
        self.actor_optimizer.step()
        return loss.mean().item()

    def update_alpha(self, current_states: torch.Tensor):
        """Update the entropy coefficient alpha via dual gradient descent.

        Args:
            current_states: Batch of current states, shape (batch_size, obs_dim + goal_dim).

        Returns:
            Mean alpha loss as a float.
        """
        _, current_log_probs = self.actor.forward(current_states)

        loss_alpha = - self.log_alpha * (current_log_probs + self.target_entropy)
        self.log_alpha_optimizer.zero_grad()
        loss_alpha.mean().backward()
        self.log_alpha_optimizer.step() 

        # Synchronization of alpha:
        self.alpha = self.log_alpha.exp().item()
        return loss_alpha.mean().item()

    def soft_update(self, tau: float):
        """Soft-update the target critic towards the online critic.

        Applies: theta_target = tau * theta_online + (1 - tau) * theta_target.

        Args:
            tau: Interpolation coefficient in [0, 1]. Smaller values give more stable targets.
        """
        for param_online, param_target in zip(self.critic_online.parameters(), self.critic_target.parameters()):
            param_target.data = tau * param_online.data + (1-tau) * param_target.data

    def select_action(self, current_state: np.ndarray):
        """Sample an action from the actor without computing gradients.

        Args:
            current_state: Flattened observation array (observation + desired_goal).

        Returns:
            Action array of shape (action_dim,).
        """
        with torch.no_grad():
            tensor = torch.from_numpy(current_state).float()
            action, _ = self.actor.forward(tensor.unsqueeze(0))
        return action.cpu().numpy().squeeze(0)
    
    def save(self, path: str):
        """Save actor, online critic, and log_alpha weights to disk.

        Args:
            path: Directory where actor.pt, critic.pt, and log_alpha.pt will be written.
        """
        torch.save(self.actor.state_dict(), os.path.join(path,'actor.pt'))
        torch.save(self.critic_online.state_dict(), os.path.join(path,'critic.pt'))
        torch.save(self.log_alpha, os.path.join(path,'log_alpha.pt'))

    def load(self, path: str):
        """Load actor, online critic, and log_alpha weights from disk.

        Args:
            path: Directory containing actor.pt, critic.pt, and log_alpha.pt.
        """
        self.actor.load_state_dict(torch.load(os.path.join(path,'actor.pt'), weights_only=True))
        self.critic_online.load_state_dict(torch.load(os.path.join(path,'critic.pt'), weights_only=True))
        self.log_alpha.data.copy_(torch.load(os.path.join(path,'log_alpha.pt'), weights_only=True))
        self.alpha = self.log_alpha.exp().item()