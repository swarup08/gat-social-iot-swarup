"""
====================================================
Deep Q-Network (DQN) Agent
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """
    Deep Q-Network for Dynamic Graph Pruning.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim)

        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:

    def __init__(
            self,
            state_dim,
            action_dim,
            lr=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Main Network
        self.model = DQN(
            state_dim,
            action_dim
        ).to(self.device)

        # Target Network
        self.target_model = DQN(
            state_dim,
            action_dim
        ).to(self.device)

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        # Loss
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        """
        Epsilon-Greedy Action Selection.
        """

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Exploitation
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.model(state)

        return int(torch.argmax(q_values).item())

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    def update_target(self):
        """
        Synchronize target network with the main network.
        """

        self.target_model.load_state_dict(
            self.model.state_dict()
        )
    def learn(self, replay_buffer, batch_size):
        """
        Train the DQN using a mini-batch from replay buffer.
        """

        # Enough samples?
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.model(states).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Target Q values
        with torch.no_grad():

            next_q = self.target_model(next_states).max(1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return float(loss.item())
    def save_model(self, path):

        torch.save(
            self.model.state_dict(),
            path
        )


    def load_model(self, path):

        self.model.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.update_target()