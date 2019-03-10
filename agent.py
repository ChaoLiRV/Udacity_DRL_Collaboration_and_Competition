import random
import numpy as np
import copy
from collections import deque, namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
## Param choice refer to https://medium.com/@amitpatel.gt/maddpg-91caa221d75e
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4 # faster learning for critic than actor
WEIGHT_DECAY = 1.e-5
UPDATE_FREQUENCY = 2 #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.state_size = state_size   # 24
        self.action_size = action_size # 2
        self.num_agents = num_agents   # 2

        #Actor Network: State -> Action
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)

        #Critic Network: State1 x State2 x Action1 x Action2 ... -> Qvalue
        self.critic_local = Critic(state_size*num_agents, action_size*num_agents, seed).to(device)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

    def act(self, state, add_noise = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval() # set module to evaluation mode
        with torch.no_grad():
            action = self.actor_local.forward(state).cpu().data.numpy()
        self.actor_local.train() # reset it back to training mode

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1) # restrict the output boundary -1, 1

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done, timestep, agent_index):
        """Save experience in replay memory, and use random sample from buffer to updateWeight_local."""
        # for i in range(self.num_agents):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE and timestep%UPDATE_FREQUENCY==0:
            self.updateWeight_local(agent_index, self.memory.sample(), GAMMA)

    def updateWeight_local(self, agent_index, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
           actor_target(state) -> action
           critic_target(state, action) -> Q-value

        Params
        ======
           experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
           gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # states: (batchsize, 24x2)
        # actions: (batchsize, 2x2)
        # rewards: (batchsize, 1x2)
        # next_states: (batchsize, 24x2)
        # dones: (batchsize, 1x2)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        self_next_actions = self.actor_target(next_states[:, self.state_size*agent_index:
                                                        self.state_size*(agent_index+1)]) # actor by self obser
        notSelf_actions = actions[:, self.action_size*(1-agent_index):
                                     self.action_size*(2-agent_index)] # competitor's actions
        if agent_index==0: # concat order by agent index
            next_acitons = torch.cat((self_next_actions, notSelf_actions), dim = 1).to(device) # index0-> self:first
        else:
            next_acitons = torch.cat((notSelf_actions, self_next_actions), dim = 1).to(device) # index1 -> self:second


        Q_target_next = self.critic_target.forward(next_states, next_acitons) # critic by both agent's obs and actions
        Q_target = rewards + gamma * Q_target_next * (1-dones)
        Q_local = self.critic_local.forward(states, actions)
        # critic_loss = F.mse_loss(Q_local, Q_target)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(Q_local, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self_actions_pred = self.actor_local.forward(states[:, self.state_size*agent_index:
                                                        self.state_size*(agent_index+1)])#actor by self agent's obser
        notSelf_actions = actions[:, self.action_size*(1-agent_index):
                             self.action_size*(2-agent_index)] # competitor's actions
        if agent_index==0:
            actions_pred = torch.cat((self_actions_pred, notSelf_actions), dim=1).to(device)
        else:
            actions_pred = torch.cat((notSelf_actions, self_actions_pred), dim=1).to(device)

        actor_loss = -self.critic_local(states, actions_pred).mean() # '-' for Reward Maxim, gradient ascent
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.updateWeight_target(self.critic_local, self.critic_target, TAU)
        self.updateWeight_target(self.actor_local, self.actor_target, TAU)



    def updateWeight_target(self, local_model, target_model, tau):
        """Soft update TARGET model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, seed):
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience',
                                     field_names=['state','action','reward','next_state','done'])

    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def __len__(self):
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu = 0., theta = 0.15, sigma = 0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state