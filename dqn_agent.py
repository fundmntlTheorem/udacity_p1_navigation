import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork, DuelingQNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_network(is_dueling):
    return DuelingQNetwork if is_dueling else QNetwork

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        print("Using device {}".format(device))
        network_class = get_network(config['is_dueling'])
        # Q-Network
        self.qnetwork_local = network_class(state_size, action_size).to(device)
        self.qnetwork_target = network_class(state_size, action_size).to(device)
        self.little_bit = torch.tensor(1e-10).to(device)
        self.weight_normalizer = torch.tensor(1./config['replay_buffer_size']).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['learning_rate'])

        # Replay memory
        self.memory = ReplayBuffer(action_size, config['replay_buffer_size'], config['batch_size'], seed)
        # Initialize time step (for updating every learn_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every learn_every time steps.
        self.t_step = (self.t_step + 1) % self.config['learn_every']
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config['batch_size']:
                experiences, random_indices = self.memory.sample()
                self.learn(experiences, random_indices, self.config['gamma'])

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, random_indices, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.config['double_dqn']:
            # https://arxiv.org/pdf/1509.06461.pdf Deep Reinforcement Learning with Double Q-learning
            # "For each update, one set of weights is used to determine the greedy policy and the other to determine its value."
            # the next line evaluates the q for the next states and gets the indices of the best actions with the max(1)[1] call.
            # unsqueeze makes them a batch_size x 1 tensor to be used with gather
            local_net_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            # determine the value of these actions
            Q_targets_next = self.qnetwork_target(next_states).gather(1, local_net_actions)
        else:
            # use the standard dqn algorithm, where the same network is used to determine the greedy policy AND determine its value
            # Get max predicted Q values (for next states) from target model
            # the network outputs a tensor BATCH_SIZE x ACTION_SIZE.  Using max(1) finds the 
            # maximum value along the column dimension, returning the BATCH_SIZE maximum values and
            # their indices.  The [0] takes the maximum values (and not their indices), and unsqueeze 
            # makes that vector into a tensor of BATCH_SIZE x 1
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.config['prioritized_replay']:
            # the first step in the loss is the difference
            diffs = Q_expected - Q_targets

            # use the abs difference plus a little bit
            # compute sampling priority (|di|+e)**alpha
            priority = torch.pow(torch.add(self.little_bit, 
                    torch.abs(diffs.clone().detach())), 
                    self.config['alpha'])

            # the new sampling probability for each of the sampled experiences
            probs = priority / torch.sum(priority)

            # finish computing the loss.  first compute the weights
            weights = torch.pow((self.weight_normalizer * 1./probs), self.config['beta'])
            max_weight = torch.max(weights)
            # normalize to the maximum for stability
            weights /= max_weight
            # finish the loss
            loss = torch.mean(torch.square(weights * diffs))

            # update the sampling weights with these probabilities
            for index, prob in zip(random_indices, probs):
                self.memory.sampling_weights[index] = prob
        else:
            # Compute regular loss
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config['tau'])                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        buffer_size = int(buffer_size)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.sampling_weights = np.ones(buffer_size) / float(buffer_size)
        # set of indices representing each value in the buffer
        self.indices = np.arange(buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
            Randomly sample a batch of experiences from memory.
            return the batched experiences and the list of indices that were chosen,
                in case we are using prioritized replay
        """
        # get a set of random indices, weighted by the sampling weights
        current_size = len(self.memory)
        random_indices = random.choices(self.indices[:current_size], weights=self.sampling_weights[:current_size], k=self.batch_size)
        # get the experiences corresponding to these indices
        experiences = []
        for idx in random_indices:
            experiences.append(self.memory[idx])

        #experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones), random_indices

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)