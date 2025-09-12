import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from atari_env import make_atari_env
from a2c import A2C, train_a2c, test_a2c
from experience import FirstLastExperienceMemory


def pack_exp_into_batch(exp_batch, model, last_gamma, device):
    obs_history = []
    value_history = []
    action_history = []
    not_done_idx = []
    last_obs = []
    for idx, exp in enumerate(exp_batch):
        obs_history.append(exp.obs)
        value_history.append(exp.dis_reward)
        action_history.append(exp.action)

        # this exp is not done yet
        if exp.last_obs is not None:
            not_done_idx.append(idx)
            last_obs.append(exp.last_obs)

    obs_history = torch.tensor(np.array(obs_history)).to(device)
    action_history = torch.tensor(np.array(action_history)).to(device)
    value_history = torch.tensor(np.array(value_history)).float().to(device)

    if not not_done_idx:
        last_obs = torch.tensor(last_obs).to(device)

        with torch.no_grad():
            _, _, last_obs_values = model(last_obs)
        value_history[not_done_idx] += last_gamma * last_obs_values.detach()

    return obs_history, value_history, action_history


def main(args):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    print('Use device:', device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = Path(f'./{args.exp_name}')
    exp_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(exp_dir))

    env = make_atari_env(args.env_name)
    test_env = make_atari_env(args.env_name)

    model = A2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    exp_memory = FirstLastExperienceMemory(env, model, args.discount_rate, args.reward_steps)

    print()
    print('Start training...')
    step_idx = 0
    best_test_episode_reward = -np.inf

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        step_idx = state_dict['step']
        best_test_episode_reward = state_dict['best_test_episode_reward']
        print(f'Loaded resume. Continue training from the step {step_idx}. Best test episode reward: {best_test_episode_reward:.2f}')

    exp_batch = []
    for first_last_exp in exp_memory:
        episode_reward_pairs = exp_memory.pop_episode_reward_step_pairs()
        if episode_reward_pairs:
            rewards, steps = list(zip(*episode_reward_pairs))

            writer.add_scalar('episode/reward', rewards[0], step_idx)
            writer.add_scalar('episode/steps', steps[0], step_idx)
        
        # collect exps until they are enough to pack a batch
        exp_batch.append(first_last_exp)
        if len(exp_batch) < args.bs:
            continue
        
        last_gamma = args.discount_rate ** args.reward_steps
        obs_history, value_history, action_history = pack_exp_into_batch(exp_batch, model, last_gamma, device)
        exp_batch = []

        train_a2c(model, obs_history, value_history, action_history, optimizer, writer, step_idx, args, device)

        step_idx += 1

        # if training is done
        if args.training_steps is not None and step_idx >= args.training_steps:
            break

        # do testing
        if step_idx % args.test_steps == 0:
            test_avg_reward, test_avg_steps = test_a2c(model, test_env, num_rounds=10, device=device)

            if best_test_episode_reward < test_avg_reward:
                best_test_episode_reward = test_avg_reward

                file_name = exp_dir / f'best_a2c_{step_idx}_{best_test_episode_reward:.2f}.pt'
                state_dict = {
                    'step': step_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_test_episode_reward': best_test_episode_reward,
                }
                torch.save(state_dict, file_name)

            file_name = exp_dir / f'latest_a2c.pt'
            state_dict = {
                'step': step_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_test_episode_reward': best_test_episode_reward,
            }
            torch.save(state_dict, file_name)
            print(f'Step: {step_idx} | test steps: {test_avg_steps:.2f} | test reward: {test_avg_reward:.2f} | best test reward: {best_test_episode_reward:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp-name', type=str, default='exp_a2c')
    parser.add_argument('--env-name', type=str, default='MinitaurBulletEnv-v0')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--training-steps', type=int, default=1000000)
    parser.add_argument('--reward-steps', type=int, default=2)
    parser.add_argument('--discount-rate', type=float, default=0.99)
    parser.add_argument('--bs', type=float, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--loss-entropy-coef', type=float, default=1e-4)
    parser.add_argument('--test-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    main(args)
