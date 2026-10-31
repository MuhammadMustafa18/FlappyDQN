import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    """Deep Q-Network: input features -> Q-values for [idle, flap]"""
    def __init__(self, state_dim, action_dim=2):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience Replay buffer storing (s, a, r, s', done) transitions"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim=19,
        action_dim=2,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_capacity=60000,
        batch_size=64,
        target_update_freq=500,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Exploration epsilon parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.steps_done = 0

    def select_action(self, state, evaluate=False):
        """Epsilon-greedy action selection"""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q estimates: Q(s, a)
        q_current = self.policy_net(states).gather(1, actions)

        # Target Q computation (Double DQN or Standard Target DQN)
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (1 - dones) * self.gamma * q_next_max

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filepath="model.pth"):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath="model.pth"):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
