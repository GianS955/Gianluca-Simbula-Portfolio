import torch
import numpy as np

class ReplayBuffer():
    """Circular replay buffer for goal-conditioned environments.

    Stores transitions from Gymnasium Robotics dict-observation environments
    (with 'observation', 'desired_goal', 'achieved_goal' keys) and returns
    batches of goal-conditioned state tensors ready for network input.

    Attributes:
        current_states: Array of observations, shape (max_size, obs_dim).
        next_states: Array of next observations, shape (max_size, obs_dim).
        current_desired_goal: Array of desired goals, shape (max_size, goal_dim).
        next_desired_goal: Array of next desired goals, shape (max_size, goal_dim).
        current_achieved_goal: Array of achieved goals, shape (max_size, goal_dim).
        next_achieved_goal: Array of next achieved goals, shape (max_size, goal_dim).
        actions: Array of actions, shape (max_size, action_dim).
        rewards: Array of rewards, shape (max_size, 1).
        terminals: Array of terminal flags, shape (max_size, 1).
        truncated: Array of truncated flags, shape (max_size, 1).
        size: Maximum capacity of the buffer.
        index: Current write position.
        full: True once the buffer has wrapped around at least once.
    """
    def __init__(self, size: int, observation: dict, action_shape: int):
        """Allocate buffer arrays sized from a sample observation and action.

        Args:
            size: Maximum number of transitions to store.
            observation: A single environment observation dict used to infer dimensions.
            action_shape: Action dimension.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_states = np.zeros((size, observation['observation'].shape[0]))
        self.next_states = np.zeros((size, observation['observation'].shape[0]))
        self.rewards = np.zeros((size, 1))
        self.actions = np.zeros((size, action_shape))
        self.terminals = np.zeros((size,1))
        self.truncated = np.zeros((size, 1))
        self.current_desired_goal = np.zeros((size, observation['desired_goal'].shape[0]))
        self.current_achieved_goal = np.zeros((size, observation['achieved_goal'].shape[0]))
        self.next_desired_goal = np.zeros((size, observation['desired_goal'].shape[0]))
        self.next_achieved_goal = np.zeros((size, observation['achieved_goal'].shape[0]))

        self.index = 0
        self.size = size
        self.full = False

    def store(self, 
              current_observation: dict,
              next_observation: dict,            
              action: np.ndarray, 
              reward: float, 
              terminal: bool, 
              truncated: bool):
        """Write a single transition into the buffer, overwriting the oldest entry when full.

        Args:
            current_observation: Current observation dict with 'observation', 'desired_goal', 'achieved_goal'.
            next_observation: Next observation dict, same structure.
            action: Action taken, shape (action_dim,).
            reward: Scalar reward received.
            terminal: True if the episode ended due to a success or failure condition.
            truncated: True if the episode ended due to a step limit.
        """
        
        if self.index >= self.size:
            self.index = 0
            self.full = True
        
        self.current_states[self.index] = current_observation['observation']
        self.current_desired_goal[self.index] = current_observation['desired_goal']
        self.current_achieved_goal[self.index] = current_observation['achieved_goal']
        self.next_states[self.index] = next_observation['observation']
        self.next_desired_goal[self.index] = next_observation['desired_goal']
        self.next_achieved_goal[self.index] = next_observation['achieved_goal']
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal
        self.truncated[self.index] = truncated

        self.index += 1

    def sample(self, batch_size: int):
        """Sample a random mini-batch of transitions without replacement.

        Observations and goals are concatenated along the feature axis before
        being returned, producing goal-conditioned state tensors ready for
        direct input to the actor and critic networks.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys:
                - 'current_states': FloatTensor of shape (batch_size, obs_dim + goal_dim).
                - 'next_states': FloatTensor of shape (batch_size, obs_dim + goal_dim).
                - 'actions': FloatTensor of shape (batch_size, action_dim).
                - 'rewards': FloatTensor of shape (batch_size, 1).
                - 'terminals': FloatTensor of shape (batch_size, 1).
                - 'truncated': FloatTensor of shape (batch_size, 1).
        """
        indices = np.random.choice(self.get_size(), size = batch_size, replace=False)
        current_states = torch.from_numpy(np.concatenate([self.current_states[indices],self.current_desired_goal[indices]], 
                                                         axis = 1)).float().to(self.device)
        
        next_states = torch.from_numpy(np.concatenate([self.next_states[indices], self.next_desired_goal[indices]],
                                                      axis = 1)).float().to(self.device)
        actions = torch.from_numpy(self.actions[indices]).float().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device) 
        terminals = torch.from_numpy(self.terminals[indices]).float().to(self.device) 
        truncated = torch.from_numpy(self.truncated[indices]).float().to(self.device)

        return current_states, next_states, actions, rewards, terminals, truncated

    def get_size(self):
        """Return the number of transitions currently stored in the buffer.

        Returns:
            size if the buffer has wrapped around, otherwise the current write index.
        """
        if self.full:
            return self.size
        return self.index
    
class ImprovementBuffer():
    """Circular buffer that tracks recent success ratios for early stopping.

    Attributes:
        buffer: Array storing the last N success ratio values.
        index: Current write position.
        full: True once the buffer has been filled at least once.
    """
    def __init__(self, size: int):
        self.buffer = np.zeros(size)
        self.index = 0
        self.full = False

    def store(self, value):
        """Write a value into the buffer, overwriting the oldest entry when full."""
        self.buffer[self.index] = value
        self.index += 1
        if self.index >= len(self.buffer):
            self.index = 0
            self.full = True
    
    def mean(self):
        """Return the mean of all values currently in the buffer."""
        return np.mean(self.buffer)
