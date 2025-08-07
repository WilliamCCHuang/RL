import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 400


class DDPG(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        self.critic = QNet(obs_size, act_size)
        self.actor = Actor(obs_size, act_size)

    def forward(self, obs):
        actions = None  # TODO
        q_values = self.critic()


class QNet(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU()
        )
        self.out_net = nn.Sequential(
            nn.Linear(HID_SIZE + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

    def forward(self, obs, act):
        x = self.obs_net(obs)
        x = torch.cat([x, act], dim=1)
        return self.out_net(x)


class Actor(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)
