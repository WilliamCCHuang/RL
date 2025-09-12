import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gym


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['episode_reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])


def collect_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    state = env.reset()
    softmax_fn = nn.Softmax(dim=1)

    while True:
        state_tensor = torch.FloatTensor([state])  # (1, obs_size)
        action_probs_tensor = softmax_fn(net(state_tensor))  # (1, n_actions)
        action_probs = action_probs_tensor.detach().numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)

        next_state, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(state=state, action=action))
        
        if is_done:
            batch.append(Episode(episode_reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_state = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        state = next_state


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.episode_reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_states = []
    train_actions = []
    for episode_example in batch:
        if episode_example.episode_reward < reward_bound:
            continue
        train_states.extend(map(lambda step: step.state, episode_example.steps))
        train_actions.extend(map(lambda step: step.action, episode_example.steps))

    train_states_tensor = torch.FloatTensor(train_states)
    train_actions_tensor = torch.LongTensor(train_actions)
    return train_states_tensor, train_actions_tensor, reward_bound, reward_mean


def main():
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for i, batch in enumerate(collect_batches(env, net, BATCH_SIZE)):
        states, actions_gt, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_logits = net(states)
        loss = loss_fn(action_logits, actions_gt)
        loss.backward()
        optimizer.step()

        print(f"{i}: loss={loss.item():.3f}, reward_mean={reward_mean:.1f}, reward_bound={reward_bound:.1f}")
        writer.add_scalar("loss", loss.item(), i)
        writer.add_scalar("reward_mean", reward_mean, i)
        writer.add_scalar("reward_bound", reward_bound, i)
        if reward_mean > 199:
            print("Solved!")
            break

    writer.close()


if __name__ == "__main__":
    main()