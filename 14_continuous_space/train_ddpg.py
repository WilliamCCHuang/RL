import copy
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from atari_env import make_atari_env
from ddpg import DDPG, train_ddpg, test_ddpg
from experience import FirstLastExperienceMemory, ExperienceMemoryReplayBuffer


def pack_exp_into_batch(exp_batch, device):
    obs = [exp.obs for exp in exp_batch]
    values = [exp.dis_reward for exp in exp_batch]
    actions = [exp.action for exp in exp_batch]
    dones = [exp.last_obs is None for exp in exp_batch]
    last_obs = [exp.obs if exp.last_obs is None else exp.last_obs for exp in exp_batch]

    # # this exp is done
    # if exp.last_obs is None:
    #     last_obs.append(exp.obs)

    # # this exp is not done yet
    # else:
    #     last_obs.append(exp.last_obs)

    obs = torch.tensor(np.array(obs)).to(device)
    values = torch.tensor(np.array(values)).float().to(device)
    actions = torch.tensor(np.array(actions)).to(device)
    dones = torch.ByteTensor(dones).to(device)
    last_obs = torch.tensor(np.array(last_obs)).to(device)
    
    return obs, values, actions, dones, last_obs


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

    model = DDPG(env.observation_space.shape[0], env.action_space.shape[0], args.update_weight).to(device)
    tgt_model = copy.deepcopy(model).to(device)
    print(model)

    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=args.lr)
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr)

    exp_memory = FirstLastExperienceMemory(env, model, args.discount_rate, args.reward_steps)
    exp_buffer = ExperienceMemoryReplayBuffer(exp_memory, buffer_size=args.buffer_size)

    print()
    print('Start training...')
    step_idx = 0
    best_test_episode_reward = -np.inf

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        step_idx = state_dict['step']
        best_test_episode_reward = state_dict['best_test_episode_reward']
        print(f'Loaded resume. Continue training from the step {step_idx}. Best test episode reward: {best_test_episode_reward:.2f}')

    while True:
        exp_buffer.populate(1)
        if len(exp_buffer) < args.buffer_init_size:
            continue

        episode_reward_pairs = exp_memory.pop_episode_reward_step_pairs()
        if episode_reward_pairs:
            rewards, steps = list(zip(*episode_reward_pairs))

            writer.add_scalar('episode/reward', rewards[0], step_idx)
            writer.add_scalar('episode/steps', steps[0], step_idx)

        exp_batch = exp_buffer.sample(args.bs)
        obs, values, actions, dones, last_obs = pack_exp_into_batch(exp_batch, device)

        train_ddpg(model, tgt_model, obs, values, actions, dones, last_obs, critic_optimizer, actor_optimizer, writer, step_idx, args)
        step_idx += 1

        # if training is done
        if args.training_steps is not None and step_idx >= args.training_steps:
            break

        # do testing
        if step_idx % args.test_steps == 0:
            test_avg_reward, test_avg_steps = test_ddpg(model, test_env, num_rounds=10, device=device)

            if best_test_episode_reward < test_avg_reward:
                best_test_episode_reward = test_avg_reward

                file_name = exp_dir / f'best_ddpg_{step_idx}_{best_test_episode_reward:.2f}.pt'
                state_dict = {
                    'step': step_idx,
                    'model': model.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'best_test_episode_reward': best_test_episode_reward,
                }
                torch.save(state_dict, file_name)

            file_name = exp_dir / f'latest_ddpg.pt'
            state_dict = {
                'step': step_idx,
                'model': model.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                'best_test_episode_reward': best_test_episode_reward,
            }
            torch.save(state_dict, file_name)
            print(f'Step: {step_idx} | test steps: {test_avg_steps:.2f} | test reward: {test_avg_reward:.2f} | best test reward: {best_test_episode_reward:.2f}')   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp-name', type=str, default='exp_ddpg')
    parser.add_argument('--env-name', type=str, default='MinitaurBulletEnv-v0')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--training-steps', type=int, default=1000000)
    parser.add_argument('--reward-steps', type=int, default=2)
    parser.add_argument('--discount-rate', type=float, default=0.99)
    parser.add_argument('--update-weight', type=int, default=1e-3)
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--buffer-init-size', type=int, default=100)
    parser.add_argument('--bs', type=float, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--loss-entropy-coef', type=float, default=1e-4)
    parser.add_argument('--test-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    main(args)
