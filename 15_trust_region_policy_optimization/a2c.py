import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 64


class A2C(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        self.critic = Critic(obs_size)
        self.actor = Actor(obs_size, act_size)

    def predict_values(self, obs):
        return self.critic(obs)
    
    @torch.no_grad()
    def sample_actions(self, obs, add_noise=True):
        # obs: (bs, obs_size)

        return self.actor.sample_actions(obs, add_noise=True)
    

class Actor(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, obs):
        return self.mu(obs)
    
    def sample_actions(self, obs, add_noise=True):
        mu = self.mu(obs)
        mu = mu.detach().cpu().numpy()

        sigma = np.zeros_like(mu)
        if add_noise:
            logstd = self.logstd.detach().cpu().numpy()
            sigma = np.exp(logstd)

        actions = mu + sigma * np.random.normal(size=mu.shape)
        actions = np.clip(actions, -1, 1)

        return actions
    

class Critic(nn.Module):

    def __init__(self, obs_size):
        super().__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

    def forward(self, obs):
        return self.value(obs)


def cal_log_policy(mu_tensor, var_tensor, action_tensor):
    # mu_tensor: (bs, 8)
    # var_tensor: (bs, 8)
    # action_tensor: (bs, 8)
    
    res = - (action_tensor - mu_tensor)**2 / (2 * var_tensor.clamp(min=1e-3))
    res -= 0.5 * torch.log(2 * torch.pi * var_tensor)
    return res


def cal_entropy(var_tensor):
    return 0.5 + 0.5 * torch.log(2 * torch.pi * var_tensor).mean()


def train_a2c(model, obs_history, value_history, action_history, critic_optimizer, actor_optimizer, writer, step_idx, args, device):
    obs_history = obs_history.to(device)  # (bs, 28)
    value_history = value_history.unsqueeze(-1).to(device)  # (bs, 1)
    action_history = action_history.to(device)  # (bs, 8)

    critic_optimizer.zero_grad()
    pred_values = model.predict_values(obs_history)  # (bs, 8), (bs, 8), (bs, 1)
    loss_value = F.mse_loss(pred_values, value_history)
    loss_value.backward()
    critic_optimizer.step()

    actor_optimizer.zero_grad()
    mus = model.actor(obs_history)  # (32, 8)
    adv_values = value_history - pred_values.detach()  # (32, 1)
    log_pred_action_prob = adv_values * cal_log_policy(mus, model.actor.logstd, action_history)  # (32, 8)
    loss_policy = - log_pred_action_prob.mean()
    
    loss_entropy = - cal_entropy(torch.exp(model.actor.logstd))

    loss_actor = loss_policy + args.loss_entropy_coef * loss_entropy
    loss_actor.backward()
    actor_optimizer.step()

    writer.add_scalar('loss_value', loss_value, step_idx)
    writer.add_scalar('loss_policy', loss_policy, step_idx)
    writer.add_scalar('loss_entropy', loss_entropy, step_idx)
    writer.add_scalar('loss_actor', loss_actor, step_idx)


@torch.no_grad()
def test_a2c(model, env, num_rounds=10, device='cpu'):
    rewards = 0
    steps = 0 

    for _ in range(num_rounds):
        obs = env.reset()

        while True:
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)  # (1, 28)
            action = model.sample_actions(obs_tensor, add_noise=False)[0]  # (8,)

            obs, reward, done, info = env.step(action)

            rewards += reward
            steps += 1

            if done:
                break

    return rewards / num_rounds, steps / num_rounds