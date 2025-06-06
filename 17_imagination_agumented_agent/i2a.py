import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvironmentModel(nn.Module):

    def __init__(self, input_shape, n_actions):
        # input_shape: (#frame, h, w)
        # n_actions: #actions

        super(EnvironmentModel, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions

        # conv backbone
        n_planes = input_shape[0] + n_actions
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_planes, 64, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # for predicting next image
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4, padding=0)

        # for predict reward
        self.reward_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        reward_conv_out_dim = self._cal_reward_conv_out_dim(shape=(n_planes, *input_shape[1:]))  # input shape: (#frame+#actions, h, w)
        self.reward_fc = nn.Sequential(
            nn.Linear(reward_conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _cal_reward_conv_out_dim(self, shape):
        dummy_input = torch.zeros(1, *shape)
        out = self.conv2(self.conv1(dummy_input))
        r = self.reward_conv(out)
        return torch.numel(r)

    def _init_action_plane(self, actions):
        # actions: (bs,)
        bs = actions.shape[0]
        one_hot_action_planes = torch.zeros(bs, self.n_actions, *self.input_shape[1:], dtype=torch.float32).to(actions.device)  # (bs, #actions, h, w)
        one_hot_action_planes[torch.arange(bs), actions] = 1.0

        return one_hot_action_planes

    def forward(self, obs, actions):
        # obs: (bs, #frames, h, w)
        # actions: (bs,)

        one_hot_action_planes = self._init_action_plane(actions)  # (bs, #actions, h, w)
        inputs = torch.cat([obs, one_hot_action_planes], dim=1)  # (bs, #frames+#actions, h, w)

        out = self.conv1(inputs)
        out = self.conv2(out) + out

        pred_img_diff = self.deconv(out)  # (bs, 1, h, w)
        pred_reward = self.reward_fc(self.reward_conv(out).flatten(start_dim=1))  # (bs, 1)
        pred_reward = pred_reward.squeeze(dim=-1)  # (bs,)

        return pred_img_diff, pred_reward


def train_env_model(env_model, obs_history, next_obs_history, action_history, reward_history, optimizer, writer, step_idx, args, device):
    optimizer.zero_grad()

    obs_history = obs_history.to(device)  # (bs, #freme, 84, 84)
    next_obs_history = next_obs_history.to(device)  # (bs, #frame, 84, 84)
    action_history = action_history.to(device)
    reward_history = reward_history.to(device)  # values of {-1, 0, 1} as the rewards are clipped

    pred_obs_diff, pred_reward = env_model(obs_history / 255, action_history)
    
    # loss for observation
    tgt_obs_diff = (next_obs_history[:, -1:, ...] - obs_history[:, -1:, ...]) / 255  # -1 represents the last frame
    loss_obs = F.mse_loss(pred_obs_diff, tgt_obs_diff)

    # loss for reward
    loss_reward = F.mse_loss(pred_reward, reward_history)

    # total loss
    loss_total = args.obs_weight * loss_obs + args.reward_weight * loss_reward
    loss_total.backward()

    optimizer.step()

    writer.add_scalar('loss/obs', loss_obs, step_idx)
    writer.add_scalar('loss/reward', loss_reward, step_idx)
    writer.add_scalar('loss/total', loss_total, step_idx)

    return loss_total.item(), loss_obs.item(), loss_reward.item()
    

class I2A(nn.Module):

    def __init__(self, input_shape, n_actions, env_model, rollout_policy, rollout_steps, rollout_hidden_dim):
        super(I2A, self).__init__()
        # input_shape: (#frame, h, w)
        # n_actions: #actions

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.rollout_steps = rollout_steps
        self.rollout_hidden_dim = rollout_hidden_dim
        
        self.model_free_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_channel = self._cal_conv_out_dim(input_shape)
        fc_input_dim = conv_out_channel + n_actions * rollout_hidden_dim
        self.fc_head = nn.Sequential(
            nn.Linear(fc_input_dim, 512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

        env_model_output_shape = (1,) + input_shape[1:]  # (1, h, w)
        self.rollout_encoder = RolloutEncoder(env_model_output_shape, rollout_hidden_dim)
        self.env_model = env_model
        [p.requires_grad_(False) for p in env_model.parameters()]
        self.rollout_policy = rollout_policy

    def _cal_conv_out_dim(self, shape):
        dummy_input = torch.zeros(1, *shape)
        out = self.model_free_conv(dummy_input)
        return torch.numel(out)
    
    def rollout(self, obs):
        # obs: (bs, #frame, h, w)
        bs = obs.shape[0]

        # we explore every possible action in the first, then rollout by the policy predicted by the `rollout_policy`
        # `obs` should be repeated `n_actions`
        obs = obs.unsqueeze(1).expand(bs, self.n_actions, *self.input_shape)  # (bs, #frame, h, w) -> (bs, 1, #frame, h, w) -> (bs, n_action, #frame, h, w)
        obs = obs.reshape(bs * self.n_actions, *self.input_shape)  # (bs * #action, #frame, h, w)
        actions = torch.arange(self.n_actions).tile(bs).to(obs.device)  # (bs * #action,)

        # start rollout
        rollout_obs_diff_list = []
        rollout_reward_list = []
        for step_idx in range(self.rollout_steps):
            pred_obs_diffs, pred_rewards = self.env_model(obs, actions)
            # pred_obs_diffs: # (bs * #action, 1, h, w)
            # pred_rewards: # (bs * #action,)

            rollout_obs_diff_list.append(pred_obs_diffs.detach())
            rollout_reward_list.append(pred_rewards.detach())

            # don't need actions for the last step
            if step_idx == self.rollout_steps - 1:
                break
            
            # recover the next obs
            cur_obs = obs[:, -1:, ...]  # (bs * #action, 1, h, w), `-1` represents the last frame
            next_obs = cur_obs + pred_obs_diffs  # (bs * #action, 1, h, w)
            obs = torch.cat([obs[:, 1:, ...], next_obs], dim=1)  # (bs * #action, #frame, h, w)

            # select actions by `rollout_policy`
            pred_values, action_logits = self.rollout_policy(obs)
            # pred_values: (bs * #action,), it is useless here
            # action_logits: (bs * #action, n_actions)
            action_probs = torch.softmax(action_logits, dim=-1).detach().cpu().numpy()
            actions = torch.tensor([np.random.choice(self.n_actions, p=prob) for prob in action_probs]).to(obs.device)  # (bs * #action,)

        rollout_obs_diffs = torch.stack(rollout_obs_diff_list, dim=0)  # (rollout_steps, bs * #action, 1, h, w)
        rollout_rewards = torch.stack(rollout_reward_list, dim=0)  # (rollout_steps, bs * #action)

        rollout_out = self.rollout_encoder(rollout_obs_diffs, rollout_rewards)  # ()

        return rollout_out.reshape(bs, -1)
    
    def forward(self, obs):
        # obs: (bs, #frame, h, w). Unnormalized
        
        obs = obs / 255
        model_free_out = self.model_free_conv(obs).flatten(start_dim=1)  # (bs, conv_out_dim=3136)
        rollout_out = self.rollout(obs)  # (bs, n_actions * hidden_dim)
        
        fc_in = torch.cat([model_free_out, rollout_out], dim=-1)  # (bs, conv_out_dim + n_actions * rollout])
        fc_out = self.fc_head(fc_in)
        
        pred_values = self.value(fc_out).squeeze()  # (bs,)
        pred_action_logits = self.policy(fc_out)  # (bs, n_actions)

        return pred_values, pred_action_logits


class RolloutEncoder(nn.Module):

    def __init__(self, input_shape, hidden_dim):
        super(RolloutEncoder, self).__init__()
        # input_shape: (1, h, w)

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_channel = self._cal_conv_out_dim(input_shape)

        self.rnn = nn.LSTM(conv_out_channel + 1, hidden_dim, batch_first=False)
        
    def _cal_conv_out_dim(self, shape):
        dummy_input = torch.zeros(1, *shape)
        out = self.conv(dummy_input)
        return torch.numel(out)

    def forward(self, rollout_obs_diffs, rollout_rewards):
        # rollout_obs_diffs: (rollout_steps, bs * #action, 1, h, w)
        # rollout_rewards: (rollout_steps, bs * #action,)

        num_steps, bs, _, _, _ = rollout_obs_diffs.shape
        flatten_obs_diff = rollout_obs_diffs.reshape(num_steps * bs, *self.input_shape)  # (rollout_steps * bs * #action, 1, h, w)
        conv_out = self.conv(flatten_obs_diff)
        conv_out = conv_out.reshape(num_steps, bs, -1)  # (rollout_steps, bs * #action, conv_out_channel)

        rnn_in = torch.cat([conv_out, rollout_rewards.unsqueeze(dim=-1)], dim=-1)  # (rollout_steps, bs * #action, conv_out_channel + 1])
        _, (rnn_hidden, _)= self.rnn(rnn_in)  # (1, bs, hidden_dim)
        out = rnn_hidden.squeeze()  # (bs, hidden_dim)
        
        return out
    

def train_rollout_policy(rollout_policy, obs_history, action_logit_history, rollout_policy_optimizer, writer, step_idx, args, device):
    rollout_policy_optimizer.zero_grad()

    obs_history = obs_history.to(device)
    action_logit_history = action_logit_history.to(device)
    action_prob_history = F.softmax(action_logit_history, dim=-1)

    _, pred_action_logits = rollout_policy(obs_history)
    dist_loss = F.cross_entropy(pred_action_logits, action_prob_history)
    dist_loss.backward()

    rollout_policy_optimizer.step()

    writer.add_scalar('loss/dist', dist_loss, step_idx)