import numpy as np
import torch

class ReplayBuffer:
    """Circular replay buffer for goal-conditioned environments.

    Stores transitions from Gymnasium Robotics dict-observation environments
    (with 'observation', 'desired_goal', 'achieved_goal' keys) and returns
    batches of goal-conditioned state tensors ready for network input.

    Attributes:
        obs: Array of observations, shape (max_size, obs_dim).
        next_obs: Array of next observations, shape (max_size, obs_dim).
        desired_goal: Array of desired goals, shape (max_size, goal_dim).
        next_desired_goal: Array of next desired goals, shape (max_size, goal_dim).
        achieved_goal: Array of achieved goals, shape (max_size, goal_dim).
        next_achieved_goal: Array of next achieved goals, shape (max_size, goal_dim).
        action: Array of actions, shape (max_size, action_dim).
        reward: Array of rewards, shape (max_size, 1).
        terminal: Array of terminal flags, shape (max_size, 1).
        truncated: Array of truncated flags, shape (max_size, 1).
        max_size: Maximum capacity of the buffer.
        index: Current write position.
        full: True once the buffer has wrapped around at least once.
    """

    def __init__(self, max_size: int, observation: dict, action: np.ndarray):
        """Allocate buffer arrays sized from a sample observation and action.

        Args:
            max_size: Maximum number of transitions to store.
            observation: A single environment observation dict used to infer dimensions.
            action: A single action array used to infer action dimension.
        """

        self.obs = np.zeros((max_size, observation['observation'].shape[0]))
        self.next_obs = np.zeros((max_size, observation['observation'].shape[0]))
        self.desired_goal = np.zeros((max_size, observation['desired_goal'].shape[0]))
        self.next_desired_goal = np.zeros((max_size, observation['desired_goal'].shape[0]))
        self.achieved_goal = np.zeros((max_size, observation['achieved_goal'].shape[0]))
        self.next_achieved_goal = np.zeros((max_size, observation['achieved_goal'].shape[0]))
        self.action = np.zeros((max_size, action.shape[0]))
        self.reward = np.zeros((max_size, 1))
        self.terminal = np.zeros((max_size, 1))
        self.truncated = np.zeros((max_size, 1))
        self.max_size = max_size
        self.index = 0
        self.full = False


    def store(self, observation: dict, next_observation: dict, action: np.ndarray, reward: np.float32, terminal: bool, truncated: bool):
        """Write a single transition into the buffer, overwriting the oldest entry when full.

        Args:
            observation: Current observation dict with 'observation', 'desired_goal', 'achieved_goal'.
            next_observation: Next observation dict, same structure.
            action: Action taken, shape (action_dim,).
            reward: Scalar reward received.
            terminal: True if the episode ended due to a success or failure condition.
            truncated: True if the episode ended due to a step limit.
        """
        if self.index >= self.max_size:
            self.index = 0
            self.full = True
        
        self.obs[self.index] = observation['observation']
        self.next_obs[self.index] = next_observation['observation']
        self.desired_goal[self.index] = observation['desired_goal']
        self.next_desired_goal[self.index] = next_observation['desired_goal']
        self.achieved_goal[self.index] = observation['achieved_goal']
        self.next_achieved_goal[self.index] = next_observation['achieved_goal']
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.terminal[self.index] = terminal
        self.truncated[self.index] = truncated
        self.index += 1


    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
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
                - 'current_actions': FloatTensor of shape (batch_size, action_dim).
                - 'rewards': FloatTensor of shape (batch_size, 1).
                - 'terminals': FloatTensor of shape (batch_size, 1).
                - 'truncated': FloatTensor of shape (batch_size, 1).
        """
        
        indices = np.random.choice(self.size(), size = batch_size, replace=False)
        
        actions = torch.from_numpy(self.action[indices]).float()
        rewards = torch.from_numpy(self.reward[indices]).float()
        obs = torch.from_numpy(np.concatenate((self.obs[indices], self.desired_goal[indices]), axis=1)).float()
        next_obs = torch.from_numpy(np.concatenate((self.next_obs[indices],self.next_desired_goal[indices]), axis=1)).float()
        terminal = torch.from_numpy(self.terminal[indices]).float()
        truncated = torch.from_numpy(self.truncated[indices]).float()
        return {
            "current_states": obs,
            "next_states": next_obs,
            "current_actions": actions,
            "rewards": rewards,
            "terminals": terminal,
            'truncated': truncated
        }

    def size(self) -> int:
        """Return the number of transitions currently stored in the buffer.

        Returns:
            max_size if the buffer has wrapped around, otherwise the current write index.
        """
        if self.full:
            return self.max_size
        return self.index
    