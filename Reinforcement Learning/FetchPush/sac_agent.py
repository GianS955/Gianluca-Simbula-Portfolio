from network import Actor, TwinCritic
import torch
import torch.nn as nn
import numpy as np
import copy
import os

class Agent():
    """SAC Agent with twin critic, target network, and automatic entropy tuning.

    Attributes:
        actor: Gaussian actor network with reparameterisation trick and tanh squashing.
        critic_online: TwinCritic online network used for gradient updates.
        critic_target: TwinCritic target network updated via soft update (no gradients).
        log_alpha: Learnable log-scale entropy coefficient.
        alpha: Entropy coefficient (exp of log_alpha), used in actor and critic updates.
        actor_optimizer: Adam optimizer for the actor network.
        critic_optimizer: Adam optimizer for the online critic network.
        log_alpha_optimizer: Adam optimizer for log_alpha.
        target_entropy: Target entropy for automatic temperature tuning, set to -action_dims.
        mse: MSE loss function used in critic updates.
        tau: Interpolation coefficient in [0, 1] used in target critic updates.
    """
    def __init__(self, info: dict):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actor = Actor(info['actor']).to(self.device)
        self.critic_online = TwinCritic(info['critic']).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_online)

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()

        self.mse = nn.MSELoss()

        # Optimizers:
        self.critic_optimizer = torch.optim.Adam(self.critic_online.parameters(),
                                                 lr = info['critic']['learning_rate'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr = info['actor']['learning_rate'])
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr = info['log_alpha']['learning_rate'])
        
        # Soft update parameter:
        self.tau = info['critic']['tau']

        # Alpha update parameter:
        self.target_entropy = info['log_alpha']['target_entropy'] 


    def update_online_critic(self, 
                            current_states: torch.Tensor,
                            next_states: torch.Tensor, 
                            current_actions: torch.Tensor,
                            rewards: torch.Tensor,
                            terminal: torch.Tensor,
                            gamma: float):
        
        """Update the TwinCritic online network using clipped double Q-learning.

        Args:
            current_states: Batch of current states, shape (batch_size, obs_dim + goal_dim).
            next_states: Batch of next states, same shape as current_states.
            current_actions: Batch of actions taken at current_states, shape (batch_size, action_dim).
            rewards: Batch of received rewards, shape (batch_size, 1).
            terminal: Batch of terminal flags, shape (batch_size, 1).
            gamma: Discount factor.

        Returns:
            Combined MSE loss of both critics as a float.
        """

        with torch.no_grad():
            future_actions, log_pi_future = self.actor.forward(next_states)
            q1_target, q2_target = self.critic_target.forward(torch.concatenate((next_states, future_actions),dim= 1))
            y = rewards + gamma * (1 - terminal) * (torch.minimum(q1_target,q2_target) - self.alpha * log_pi_future)
            y = torch.clamp(y, -1 / (1 - gamma), 0)
        
        q1_online, q2_online = self.critic_online.forward(torch.concatenate((current_states, current_actions), dim = 1))

        loss1 = self.mse(q1_online, y)
        loss2 = self.mse(q2_online, y)

        loss = loss1 + loss2
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_online.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        return loss.item()
    
    def update_actor(self, current_states: torch.Tensor):
        """Update the actor network by maximising expected Q-value minus entropy cost.

        Args:
            current_states: Batch of current states, shape (batch_size, obs_dim + goal_dim).

        Returns:
            Mean actor loss as a float.
        """
        
        actions, log_pi_actions = self.actor.forward(current_states)
        q1, q2 = self.critic_online.forward(torch.concatenate((current_states, actions), dim = 1))

        loss = self.alpha * log_pi_actions - torch.minimum(q1, q2)
        self.actor_optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        return loss.mean().item()
        
    def update_log_alpha(self, current_states: torch.Tensor):
        """Update the entropy coefficient alpha via dual gradient descent.

        Args:
            current_states: Batch of current states, shape (batch_size, obs_dim + goal_dim).

        Returns:
            Mean alpha loss as a float.
        """
        _, log_pi = self.actor.forward(current_states)
        loss = - self.log_alpha * (log_pi + self.target_entropy)
        self.log_alpha_optimizer.zero_grad()
        loss.mean().backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        return loss.mean().item()

    def soft_update(self):
        """Soft-update the target critic towards the online critic.

        Applies: theta_target = tau * theta_online + (1 - tau) * theta_target.
        """
        for online_param, target_param in zip(self.critic_online.parameters(), self.critic_target.parameters()):
            target_param.data = self.tau * online_param.data + (1 - self.tau) * target_param.data

    def select_action(self, current_state: np.ndarray): 
        """Sample an action from the actor without computing gradients.

        Args:
            current_state: Flattened observation array (observation + desired_goal).

        Returns:
            Action array of shape (action_dim,).
        """
        with torch.no_grad():
            input_tensor = torch.from_numpy(current_state).float().to(self.device)
            action, _ = self.actor.forward(input_tensor.unsqueeze(0))
        return action.cpu().numpy().squeeze()
    
    def select_action_deterministic(self, current_state: np.ndarray):
        """Return the deterministic mean action without computing gradients.

        Used during evaluation to remove stochasticity from the policy.

        Args:
            current_state: Flattened observation array (observation + desired_goal).

        Returns:
            Action array of shape (action_dim,).
        """
        with torch.no_grad():
            input_tensor = torch.from_numpy(current_state).float().to(self.device)
            action = self.actor.forward_deterministic(input_tensor.unsqueeze(0))
        return action.cpu().numpy().squeeze()
    
    def save(self, folder: str):
        """Save actor, online critic, and log_alpha weights to disk.

        Args:
            folder: Directory where actor.pt, critic.pt, and log_alpha.pt will be written.
        """
        torch.save(self.actor.state_dict(), os.path.join(folder, 'agent.pt'))
        torch.save(self.critic_online.state_dict(), os.path.join(folder, 'critic.pt'))
        torch.save(self.log_alpha, os.path.join(folder, 'log_alpha.pt'))

    def load(self, folder: str):
        """Load actor, online critic, and log_alpha weights from disk.

        Args:
            folder: Directory containing actor.pt, critic.pt, and log_alpha.pt.
        """
        self.actor.load_state_dict(torch.load(os.path.join(folder,'agent.pt'), weights_only=True))
        self.critic_online.load_state_dict(torch.load(os.path.join(folder,'critic.pt'), weights_only=True))
        self.log_alpha.data.copy_(torch.load(os.path.join(folder,'log_alpha.pt'), weights_only=True))
        self.alpha = self.log_alpha.exp().item()