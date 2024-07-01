"""
Agent and network configuration 
"""

import configparser
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.linalg as la


class DeepQNetwork(nn.Module):
    """
    Deep Q-Network class for reinforcement learning.

    Args:
        config_file (str): Path to the configuration file. Contains the parameters for the network.

    Attributes:
        n2 (int): Number of features. Two times the length of the chain, since coefficients have real and imaginary parts.
        fc1_dims (int): Number of units in the first fully connected layer.
        fc2_dims (int): Number of units in the second fully connected layer.
        n_actions (int): Number of possible actions.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer.
        dropout (nn.Dropout): Dropout layer.
        optimizer (optim.RMSprop): Optimizer for model parameters.
        loss (nn.MSELoss): Loss function.
        device (torch.device): Device to run the model on.

    Methods:
        forward(state): Performs forward pass through the network.
        _initialize_weights_normal(): Initializes the weights of the linear layers.

    """

    def __init__(self, config_file):

        super(DeepQNetwork, self).__init__()

        # access configuration file
        config = configparser.ConfigParser()
        config.read(config_file)

        # import values for net parameters
        self.n2 = config.getint("learning_parameters", "number_of_features")
        self.fc1_dims = config.getint("learning_parameters", "fc1_dims")
        self.fc2_dims = config.getint("learning_parameters", "fc2_dims")
        self.n_actions = config.getint("system_parameters", "n_actions")
        self.dropout = nn.Dropout(p=config.getfloat("learning_parameters", "dropout"))
        lr = config.getfloat("learning_parameters", "learning_rate")

        # define layers
        self.fc1 = nn.Linear(self.n2, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # define optimizer and loss
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # use gpu if available
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        # initialize weights with normal distribution
        self._initialize_weights_normal()

    def forward(self, state):
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Output actions.

        """
        x = F.relu(self.fc1(state.float()))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        actions = self.fc3(x)

        return actions

    def _initialize_weights_normal(self):
        """
        Initializes the weights of the linear layers using a normal distribution.

        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class Agent(object):

    def __init__(self, config_file):
        """
        Initialize the Agent object.

        Parameters:
        - config_file (str): Path to the configuration file.

        Attributes:
        - gamma (float): Discount factor for future rewards.
        - epsilon (float): Exploration rate for epsilon-greedy action selection.
        - eps_min (float): Minimum value for epsilon.
        - eps_dec (float): Decay rate for epsilon.
        - lr (float): Learning rate for the neural networks.
        - mem_size (int): Size of the replay memory.
        - batch_size (int): Number of samples to train on in each learning iteration.
        - replace_target (int): Number of iterations to update the target network.
        - action_space (list): List of possible actions.
        - mem_cntr (int): Counter for the replay memory.
        - iter_cntr (int): Counter for the number of iterations.
        - state_memory (ndarray): Memory for storing states.
        - new_state_memory (ndarray): Memory for storing new states.
        - action_memory (ndarray): Memory for storing actions.
        - reward_memory (ndarray): Memory for storing rewards.
        - terminal_memory (ndarray): Memory for storing terminal states.
        - Q_eval (DeepQNetwork): Neural network for Q-value estimation.
        - Q_target (DeepQNetwork): Target neural network for Q-value estimation.
        """
        config_agent = configparser.ConfigParser()
        config_agent.read(config_file)

        # number of features, 2*n where n is the length of the chain
        n2 = config_agent.getint("learning_parameters", "number_of_features")

        # learning parameters
        self.gamma = config_agent.getfloat("learning_parameters", "gamma")
        self.epsilon = config_agent.getfloat("learning_parameters", "epsilon")
        self.eps_min = config_agent.getfloat("learning_parameters", "epsilon_minimum")
        self.eps_dec = config_agent.getfloat("learning_parameters", "epsilon_decay")
        self.lr = config_agent.getfloat("learning_parameters", "learning_rate")
        self.mem_size = config_agent.getint("learning_parameters", "memory_size")
        self.batch_size = config_agent.getint("learning_parameters", "batch_size")
        self.replace_target = config_agent.getint("learning_parameters", "replace_target_iter")

        # action space: 16 actions
        self.action_space = [
            i for i in range(config_agent.getint("system_parameters", "n_actions"))
        ]

        # initialize iteration counters
        self.mem_cntr = 0
        self.iter_cntr = 0

        # initialize memories
        self.state_memory = np.zeros((self.mem_size, n2), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, n2), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        # initialize neural networks
        self.Q_eval = DeepQNetwork(config_file)
        self.Q_target = DeepQNetwork(config_file)

    def store_transition(self, state, action, reward, state_, terminal):
        """
        Stores a transition in the agent's memory.

        Args:
            state (object): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received from the environment.
            state_ (object): The next state of the environment.
            terminal (bool): Indicates whether the episode terminated after this transition.

        Returns:
            None
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1

    def choose_action(self, observation):
            """
            Selects an action based on the given observation and the current epsilon value.

            Parameters:
                observation (array-like): The current observation/state.

            Returns:
                int: The selected action.
            """
            state = T.tensor([observation]).to(self.Q_eval.device)

            if np.random.uniform() > self.epsilon:
                actions = self.Q_eval.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
            return action

    def update_target_network(self):
        """
        Updates the target network by loading the state dictionary of the evaluation network.
        """
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def learn(self):
        """
        Learning method for the agent. Samples a batch from the replay memory and performs a Q-value update.

        Attributes:
            max_mem (int): The maximum number of samples in the memory.
            batch (ndarray): A random sample of indices from the memory.
            batch_index (ndarray): An array of indices from 0 to the batch size.
            state_batch (Tensor): The batch of states.
            new_state_batch (Tensor): The batch of new states.
            action_batch (ndarray): The batch of actions.
            reward_batch (Tensor): The batch of rewards.
            terminal_batch (Tensor): The batch of terminal states.
            q_eval (Tensor): The Q-values of the evaluation network for the given states and actions.
            q_next (Tensor): The Q-values of the target network for the new states.
            q_target (Tensor): The target Q-values for the Q-value update.
            loss (Tensor): The loss value for the Q-value update.
        """
        
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()

        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        if self.iter_cntr % self.replace_target == 0:
            self.update_target_network()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )
