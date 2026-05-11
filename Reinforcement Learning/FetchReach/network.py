import torch
import torch.nn as nn

class Critic(nn.Module):
    """MLP Q-network that maps (state, action) pairs to scalar Q-values.

    Attributes:
        network: Sequential MLP with ReLU activations and a linear output head.
    """

    def __init__(self, info: dict):
        """Build the MLP from a configuration dictionary.

        Args:
            info: Dictionary with keys:
                - input_shape (int): Input dimension (obs + goal + action).
                - hidden_sizes (list[int]): Width of each hidden layer.
                - output_shape (int): Output dimension (1 for a Q-value).
        """
        super().__init__()
        layers = []
        dims = [info['input_shape']] + info['hidden_sizes']
        for i in range(len(info['hidden_sizes'])):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(info['hidden_sizes'][-1],info['output_shape']))

        self.network = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the Q-value for a batch of (state, action) pairs.

        Args:
            input: Concatenated (state, action) tensor, shape (batch_size, input_shape).

        Returns:
            Q-value tensor of shape (batch_size, 1).
        """
        return self.network(input)

class TwinCritic(nn.Module):
    """Pair of independent Critic networks for clipped double Q-learning.

    Attributes:
        critic_1: First Q-network.
        critic_2: Second Q-network.
    """

    def __init__(self, info: dict):
        """Instantiate two independent Critic networks with the same configuration.

        Args:
            info: Configuration dictionary passed to each Critic (see Critic.__init__).
        """
        super().__init__()
        self.critic_1 = Critic(info)
        self.critic_2 = Critic(info)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values from both critics independently.

        Args:
            input: Concatenated (state, action) tensor, shape (batch_size, input_shape).

        Returns:
            Tuple (q1, q2) where each element has shape (batch_size, 1).
        """
        output_1 = self.critic_1(input)
        output_2 = self.critic_2(input)
        return output_1, output_2

class Actor(nn.Module):
    """Gaussian policy network with reparameterisation trick and tanh squashing.

    Outputs a tanh-squashed action sampled from a state-conditioned Gaussian,
    along with the corresponding log-probability corrected for the tanh transformation.

    Attributes:
        network: Shared MLP trunk that processes the input state.
        mu_head: Linear head that outputs the mean of the Gaussian.
        log_std_head: Linear head that outputs the log standard deviation.
        LOG_STD_MIN: Lower bound for log_std clamping (-20).
        LOG_STD_MAX: Upper bound for log_std clamping (2).
    """

    def __init__(self, info: dict):
        """Build the actor MLP trunk and the two output heads.

        Args:
            info: Dictionary with keys:
                - input_shape (int): Input dimension (obs + goal).
                - hidden_sizes (list[int]): Width of each hidden layer.
                - output_shape (int): Action dimension.
                - learning_rate (float): Not used here; consumed by the agent optimizer.
        """
        super().__init__()
        layers = []
        dims = [info['input_shape']] + info['hidden_sizes']
        for i in range(len(info['hidden_sizes'])):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        # For actor network it is needed to have two separate heads: one for \mu and one for \sigma
        self.mu_head = nn.Linear(info['hidden_sizes'][-1],info['output_shape'])
        self.log_std_head = nn.Linear(info['hidden_sizes'][-1],info['output_shape'])
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-probability.

        Applies the reparameterisation trick to sample from N(mu, std),
        squashes the result with tanh, and corrects the log-probability
        for the change of variables induced by tanh.

        Args:
            input: State tensor of shape (batch_size, input_shape).

        Returns:
            Tuple (action, log_prob) where:
                - action has shape (batch_size, action_dim) and values in (-1, 1).
                - log_prob has shape (batch_size, 1).
        """
        temp = self.network(input)
        mu = self.mu_head(temp)
        log_std = self.log_std_head(temp)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX) # Clamping for stability
        std = torch.exp(log_std) # Transformation in log to keep std > 0
        epsilon = torch.randn_like(mu)

        # parametrization trick
        action_raw = mu + std * epsilon
        # tanh squashing:
        action = torch.tanh(action_raw)

        # Jacobian correction:
        log_prob = self.jacobian_correction(mu, std, action_raw, action)
        return action, log_prob

    @staticmethod
    def jacobian_correction(mu: torch.Tensor, std: torch.Tensor, action_raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability of action_raw and correct for the tanh transformation.

        The log-prob of the squashed action is:
            log pi(a|s) = log N(a_raw | mu, std) - sum log(1 - tanh(a_raw)^2 + eps)

        Args:
            mu: Mean of the Gaussian, shape (batch_size, action_dim).
            std: Standard deviation of the Gaussian, shape (batch_size, action_dim).
            action_raw: Pre-squashing sample (mu + std * eps), shape (batch_size, action_dim).
            action: Tanh-squashed action, shape (batch_size, action_dim).

        Returns:
            Log-probability tensor of shape (batch_size, 1).
        """
        distribution = torch.distributions.Normal(mu, std)
        log_prob = distribution.log_prob(action_raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob
