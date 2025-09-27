import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 400


class DDPG(nn.Module):

    def __init__(self, obs_size, act_size, update_weight, epsilon=0.1):
        super().__init__()

        self.update_weight = update_weight
        self.epsilon = epsilon

        self.critic = DDPGCritic(obs_size, act_size)
        self.actor = DDPGActor(obs_size, act_size)

    def pred_q_value(self, obs, actions):
        return self.critic(obs, actions)
    
    def pred_action(self, obs):
        return self.actor(obs)
    
    @torch.no_grad()
    def sample_actions(self, obs):
        """Sample actions from normal distributions predicted by the network.
        """
        # obs: (1, obs_size)

        actions = self.pred_action(obs)  # (bs, act_size)
        actions = actions.detach().cpu().numpy()
        
        actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, -1, 1)

        return actions

    def update(self, other_model):
        state_dicts = self.state_dict()
        other_state_dicts = other_model.state_dict()

        for (k, v), (k_, v_) in zip(state_dicts.items(), other_state_dicts.items()):
            assert k == k_
            state_dicts[k] = (1 - self.update_weight) * v + self.update_weight * v_
        self.load_state_dict(state_dicts)


class DDPGCritic(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
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


class DDPGActor(nn.Module):

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


def train_ddpg(model, tgt_model, obs, values, actions, dones, last_obs, critic_optimizer, actor_optimizer, writer, step_idx, args):
    # obs: (bs, 28)
    # values: (bs,)
    # actions: (bs, 8)
    # dones: (bs)
    # last_obs: (bs)

    values = values.unsqueeze(-1)
    dones = dones.unsqueeze(-1)

    # critic
    # loss_critic: |r + γ*(1 - done)*Q'(s', a'|θ') - Q(s, a|θ)|^2
    critic_optimizer.zero_grad()
    pred_q_values = model.pred_q_value(obs, actions)  # Q(s, a|θ)
    last_actions = model.pred_action(last_obs)  # a'|θ'
    tgt_pred_q_values = tgt_model.pred_q_value(last_obs, last_actions)  # Q'(s', a'|θ')
    tgt_q_values =  + args.discount_rate * (1 - dones) * tgt_pred_q_values
    loss_critic = F.mse_loss(pred_q_values, tgt_q_values)
    loss_critic.backward()
    critic_optimizer.step()

    # actor
    # loss_actor: - Q(s, a|θ)
    actor_optimizer.zero_grad()
    pred_actions = model.pred_action(obs)  # a|θ
    loss_actor = - model.pred_q_value(obs, pred_actions).mean()
    loss_actor.backward()
    actor_optimizer.step()

    tgt_model.update(model)

    writer.add_scalar('loss_critic', loss_critic, step_idx)
    writer.add_scalar('loss_actor', loss_actor, step_idx)


@torch.no_grad()
def test_ddpg(model, env, num_rounds=10, device='cpu'):
    rewards = 0
    steps = 0 

    for _ in range(num_rounds):
        obs = env.reset()

        while True:
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)  # (1, 28)
            action = model.pred_action(obs_tensor)  # (1, 28)
            action = action.detach().cpu().numpy()[0]  # (28,)
            action = np.clip(action, -1, 1)
            
            obs, reward, done, info = env.step(action)

            rewards += reward
            steps += 1

            if done:
                break

    return rewards / num_rounds, steps / num_rounds