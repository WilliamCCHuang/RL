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
    
    # FIXME:
    vars = torch.exp(logstd_tensor)

    res = - (action_tensor - mu_tensor)**2 / (2 * vars.clamp(min=1e-3))
    res -= 0.5 * (logstd_tensor + np.log(2 * np.pi))
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


def train_ppo(model, trajectory, optimizer, writer, step_idx, args, device):
    # calculate old policy
    gae_values, tgt_values, states, actions = cal_gae_and_tgt_values(trajectory, model, args.discount_rate, args.ppo_gae_lambda, device)

    with torch.no_grad():
        old_mus = model.actor(states)
    old_log_policy = cal_log_policy(old_mus, model.actor.logstd, actions)
    if (old_log_policy < -20).any():
        # FIXME:
        print('Loss policy would explode as `old_log_policy` is too negative large')
        breakpoint()

    if args.ppo_normalize_gae:
        gae_values = (gae_values - gae_values.mean()) / (gae_values.std() + 1e-8)

    trajectory = trajectory[:-1]
    old_log_policy = old_log_policy[:-1].detach()

    steps = 0
    mean_loss_value = 0
    mean_loss_policy = 0
    for _ in range(args.ppo_epochs):
        for start_idx in range(0, len(trajectory), args.ppo_bs):
            end_idx = start_idx + args.ppo_bs
            if end_idx > len(trajectory):
                end_idx = len(trajectory)

            batch_states = states[start_idx:end_idx]
            batch_actions = actions[start_idx:end_idx]
            batch_gae_values = gae_values[start_idx:end_idx].unsqueeze(-1)
            batch_tgt_values = tgt_values[start_idx:end_idx].unsqueeze(-1)
            batch_old_log_policy = old_log_policy[start_idx:end_idx]

            optimizer.zero_grad()

            # train critic
            pred_values = model.predict_values(batch_states)
            loss_value = F.mse_loss(pred_values, batch_tgt_values)

            # train actor
            mus = model.actor(batch_states)
            batch_log_policy = cal_log_policy(mus, model.actor.logstd, batch_actions)
            
            ratio = torch.exp(batch_log_policy - batch_old_log_policy)
            clipped_ratio = torch.clamp(ratio, 1 - args.ppo_eps, 1 + args.ppo_eps)
            
            surr_obj = ratio * batch_gae_values
            clipped_surr_obj = clipped_ratio * batch_gae_values
            loss_policy = - torch.min(surr_obj, clipped_surr_obj).mean()
            
            loss = loss_value + loss_policy
            loss.backward()
            
            # nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)  # sometimes the loss would blow up
            optimizer.step()

            model.actor.logstd.data.clamp_(min=-1)  # FIXME: <---
            
            mean_loss_value += loss_value.item()
            mean_loss_policy += loss_policy.item()
            steps += 1

    mean_loss_value /= steps
    mean_loss_policy /= steps

    writer.add_scalar('loss_value', mean_loss_value, step_idx)
    writer.add_scalar('loss_policy', mean_loss_policy, step_idx)


def cal_gae_and_tgt_values(trajectory, model, gamma, gae_lambda, device):
    states = torch.tensor([e[0].obs for e in trajectory]).to(device)  # (trajectory length, 17)
    actions = torch.tensor([e[0].action for e in trajectory]).to(device)  # (trajectory length, 6)

    with torch.no_grad():
        values = model.predict_values(states)
    values = values.detach().cpu().numpy()  # (trajectory length, 1)
    
    tgt_value, tgt_values = 0, []
    gae_value, gae_values = 0, []
    for exp, value, next_value in zip(reversed(trajectory[:-1]), reversed(values[:-1]), reversed(values[1:])):
        assert len(exp) == 1, f'Need to be 1-step experience but got {len(exp)}'
        exp = exp[0]

        if exp.done:
            tgt_value = exp.reward

            delta = exp.reward - value.item()
            gae_value = delta
        else:
            tgt_value = exp.reward + gamma * next_value.item()

            delta = exp.reward + gamma * next_value.item() - value.item()
            gae_value = delta + gamma * gae_lambda * gae_value
        
        tgt_values.append(tgt_value)
        gae_values.append(gae_value)

    tgt_values.reverse()
    gae_values.reverse()

    tgt_values = torch.tensor(tgt_values).float().to(device)
    gae_values = torch.tensor(gae_values).float().to(device)

    return gae_values, tgt_values, states, actions