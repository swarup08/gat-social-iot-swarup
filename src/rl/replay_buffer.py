"""
====================================================
Experience Replay Buffer
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import random
from collections import deque

import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    """

    def __init__(self, capacity=10000):

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store one experience.
        """

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch.
        """

        batch = random.sample(
            self.buffer,
            batch_size
        )

        state, action, reward, next_state, done = zip(*batch)

        return (

            torch.FloatTensor(state),

            torch.LongTensor(action),

            torch.FloatTensor(reward),

            torch.FloatTensor(next_state),

            torch.FloatTensor(done)

        )

    def __len__(self):

        return len(self.buffer)