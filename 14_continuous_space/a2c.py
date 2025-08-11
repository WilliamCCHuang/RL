import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 128


class A2C(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus()
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, obs):
        x = self.backbone(obs)
        mu = self.mu(x)
        var = self.var(x)
        value = self.value(x)

        return mu, var, value
    
    @torch.no_grad()
    def sample_actions(self, obs):
        """Sample actions from normal distributions predicted by the network.
        """
        # obs: (bs, obs_size)

        mu, var, _ = self.forward(obs)
        mu = mu.detach().cpu().numpy()
        sigma = torch.sqrt(var).detach().cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)

        return actions


def cal_log_policy(mu_tensor, var_tensor, action_tensor):
    # mu_tensor: (bs, 8)
    # var_tensor: (bs, 8)
    # action_tensor: (bs, 8)
    
    res = - (action_tensor - mu_tensor)**2 / (2 * var_tensor.clamp(min=1e-3))
    res -= 0.5 * torch.log(2 * torch.pi * var_tensor)
    return res


def cal_entropy(var_tensor):
    return 0.5 + 0.5 * torch.log(2 * torch.pi * var_tensor).mean()


def train_a2c(model, obs_history, value_history, action_history, optimizer, writer, step_idx, args, device):
    optimizer.zero_grad()

    obs_history = obs_history.to(device)  # (bs, 28)
    value_history = value_history.unsqueeze(-1).to(device)  # (bs, 1)
    action_history = action_history.to(device)  # (bs, 8)

    mus, vars, pred_values = model(obs_history)  # (bs, 8), (bs, 8), (bs, 1)

    loss_value = F.mse_loss(pred_values, value_history)

    adv_values = value_history - pred_values.detach()  # (32, 1)
    log_pred_action_prob = adv_values * cal_log_policy(mus, vars, action_history)  # (32, 8)
    loss_policy = - log_pred_action_prob.mean()
    
    loss_entropy = - cal_entropy(vars)

    loss_total = loss_policy + loss_value + args.loss_entropy_coef * loss_entropy
    loss_total.backward()
    optimizer.step()

    writer.add_scalar('loss_value', loss_value, step_idx)
    writer.add_scalar('loss_policy', loss_policy, step_idx)
    writer.add_scalar('loss_entropy', loss_entropy, step_idx)
    writer.add_scalar('loss_total', loss_total, step_idx)


def test_a2c(model, env, num_rounds=10, device='cpu'):
    rewards = 0
    steps = 0 

    for _ in range(num_rounds):
        obs = env.reset()

        while True:
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)  # (1, 28)
            mu = model(obs_tensor)[0]  # (1, 28)
            action = mu.detach().cpu().numpy()[0]  # (28,)
            action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(action)

            rewards += reward
            steps += 1

            if done:
                break

    return rewards / num_rounds, steps / num_rounds