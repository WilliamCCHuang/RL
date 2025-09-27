from tqdm import tqdm
from multiprocessing import Lock
tqdm.set_lock(Lock())

import argparse
import numpy as np
from pathlib import Path

import gym
import torch

from a2c import A2C
from atari_env import make_atari_env

ENV_NAME = 'BreakoutNoFrameskip-v4'


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = Path(args.resume_path).parent

    env = make_atari_env(ENV_NAME, args.frame_count, test_mode=True, video=True)
    env = gym.wrappers.RecordVideo(env, video_folder=str(exp_dir))

    # envs[0].observation_space.shape: (#frame, 84, 84)
    # envs[0].action_space.n: 4
    model = A2C(env.observation_space.shape, env.action_space.n)
    model_state_dict = torch.load(args.resume_path, map_location='cpu')['model']
    model.load_state_dict(model_state_dict)

    obs = env.reset()
    total_steps = 0
    total_reward = 0.0
    while True:
        obs_tensor = torch.tensor(obs).unsqueeze(0)  # (1, 2, 84, 84)
        pred_values, action_logits = model(obs_tensor)  # (1,), (1, 4)
        action = action_logits.argmax(dim=-1).item()

        obs, reward, done, trunc, info = env.step(action)

        total_steps += 1
        total_reward += reward

        if done:
            break

    print(f'Steps: {total_steps}, Reward: {total_reward:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-path', type=str)
    parser.add_argument('--frame-count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    main(args)