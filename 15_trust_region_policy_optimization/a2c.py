from math import isnan
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
        if np.isnan(actions).any():
            breakpoint()
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


def cal_log_policy(mu_tensor, logstd_tensor, action_tensor):
    # mu_tensor: (bs, 8)
    # var_tensor: (bs, 8)
    # action_tensor: (bs, 8)
    
    vars = torch.exp(logstd_tensor)

    res = - (action_tensor - mu_tensor)**2 / (2 * vars.clamp(min=1e-3))
    res -= 0.5 * torch.log(2 * torch.pi * vars)
    return res


def cal_entropy(logstd_tensor):
    vars = torch.exp(logstd_tensor)
    return 0.5 + 0.5 * torch.log(2 * torch.pi * vars).mean()


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
    
    loss_entropy = - cal_entropy(model.actor.logstd)

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


def train_ppo(model, trajectory, critic_optimizer, actor_optimizer, writer, step_idx, args, device):
    states = torch.tensor([e[0].state for e in trajectory]).to(device)
    actions = torch.tensor([e[0].action for e in trajectory]).to(device)
    # obs_history = obs_history.to(device)  # (bs, 28)
    # value_history = value_history.unsqueeze(-1).to(device)  # (bs, 1)
    # action_history = action_history.to(device)  # (bs, 8)


    return
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
    
    loss_entropy = - cal_entropy(model.actor.logstd)

    loss_actor = loss_policy + args.loss_entropy_coef * loss_entropy
    loss_actor.backward()
    actor_optimizer.step()

    writer.add_scalar('loss_value', loss_value, step_idx)
    writer.add_scalar('loss_policy', loss_policy, step_idx)
    writer.add_scalar('loss_entropy', loss_entropy, step_idx)
    writer.add_scalar('loss_actor', loss_actor, step_idx)


def cal_gae(trajectory, model, gamma, gae_lambda, device):
    states = torch.tensor([e[0].state for e in trajectory]).to(device)

    with torch.no_grad():
        values = model.predict_values(states)
    values = values.detach().cpu().numpy()
    
    gae_values = []
    for (exp,), value, next_value in zip(reversed(trajectory[:-1]), reversed(values[:-1]), reversed(values[1:])):
        if exp.done:
            delta = exp.reward - value
            gae_value = delta
        else:
            delta = exp.reward + gamma * next_value - value
            gae_value = delta + gamma * gae_lambda * gae_value
        
        gae_values.append(gae_value)
    
    gae_values.reverse()
    gae_tensor = torch.tensor(gae_values).to(device)
    return gae_tensor