import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class A2C(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(A2C, self).__init__()
        # input_shape: (w, h, #frame)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_channel = self._get_conv_out(input_shape)  # 3136
        self.policy = nn.Sequential(
            nn.Linear(conv_out_channel, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_channel, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        return torch.numel(conv_out)

    def forward(self, stack_obs):
        # stack_obs: (#envs, #frame, w, h)
        x = stack_obs.float() / 256

        conv_out = self.conv(x).flatten(1)
        pred_values = self.value(conv_out).flatten()  # (#envs,)
        action_logits = self.policy(conv_out)  # (#envs, n_actions)

        return pred_values, action_logits


def train_a2c(model, obs_history, value_history, action_history, dis_reward_history, optimizer, writer, step_idx, args, device):
    bs = len(obs_history)
    optimizer.zero_grad()

    obs_history = obs_history.to(device)
    value_history = value_history.to(device)
    action_history = action_history.to(device)
    dis_reward_history = dis_reward_history.to(device)
    
    pred_values, pred_action_logits = model(obs_history)

    # loss for value
    loss_value = F.mse_loss(pred_values, dis_reward_history)  # (1,)

    # loss for policy
    advance_history = dis_reward_history - value_history  # (#envs * reward_steps,)
    log_pred_action_prob = F.log_softmax(pred_action_logits, dim=-1)  # (#envs * reward_steps, n_actions)
    objective_policy = advance_history * log_pred_action_prob[range(bs), action_history]  # (#envs * reward_steps,)
    loss_policy = - objective_policy.mean()  # (1,) # another negative sign for maximum the actor's objective

    # loss for entropy
    pred_action_prob = F.softmax(pred_action_logits, dim=-1)  # (#envs * reward_steps, n_actions)
    entropy = - (pred_action_prob * log_pred_action_prob).sum(dim=-1).mean()
    loss_entropy = - entropy  # maximizing the entropy is equivalent to minimizing this loss

    # total loss
    loss_total = loss_policy + args.loss_value_coef * loss_value + args.loss_entropy_coef * loss_entropy
    loss_total.backward()

    # clip the gradient and then update weights
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optimizer.step()
    #####################################

    writer.add_scalar('loss_value', loss_value, step_idx)
    writer.add_scalar('loss_policy', loss_policy, step_idx)
    writer.add_scalar('loss_entropy', loss_entropy, step_idx)
    writer.add_scalar('loss_total', loss_total, step_idx)


def test_a2c(model, env, num_rounds, device):
    total_steps = 0
    total_reward = 0

    for _ in range(num_rounds):
        obs = env.reset()  # (2, 84, 84)
        
        while True:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)  # (1, 2, 84, 84)
                _, action_logits = model(obs_tensor)
                action_probs = F.softmax(action_logits, dim=-1).cpu().numpy()
            actions = [np.random.choice(len(p), p=p) for p in action_probs]
            action = actions[0]

            obs, reward, done, truncated, info = env.step(action)

            total_steps += 1
            total_reward += reward

            if done:
                break

    return total_steps / num_rounds, total_reward / num_rounds

