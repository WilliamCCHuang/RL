import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from atari_env import make_atari_env
from a2c import A2C, train_ppo, test_a2c
from experience import NStepExperienceMemory


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
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), args.lr_critic)
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), args.lr_actor)
    print(model)

    exp_memory = NStepExperienceMemory(env, model, n_steps=1)

    print()
    print('Start training...')
    step_idx = 0
    best_test_episode_reward = -np.inf

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict['model'])
        critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        step_idx = state_dict['step']
        best_test_episode_reward = state_dict['best_test_episode_reward']
        print(f'Loaded resume. Continue training from the step {step_idx}. Best test episode reward: {best_test_episode_reward:.2f}')
    
    trajectory = []
    for exps in exp_memory:
        episode_reward_pairs = exp_memory.pop_episode_reward_step_pairs()
        if episode_reward_pairs:
            rewards, steps = list(zip(*episode_reward_pairs))

            writer.add_scalar('episode/reward', rewards[0], step_idx)
            writer.add_scalar('episode/steps', steps[0], step_idx)
        
        # collect exps until they are enough
        trajectory.append(exps)
        if len(trajectory) < args.ppo_trajectory_size:
            continue
        
        train_ppo(model, trajectory, critic_optimizer, actor_optimizer, writer, step_idx, args, device)
        step_idx += 1

        # if training is done
        if args.training_steps is not None and step_idx >= args.training_steps:
            break

        # do testing
        if step_idx % args.test_steps == 0:
            test_avg_reward, test_avg_steps = test_a2c(model, test_env, num_rounds=10, device=device)

            if best_test_episode_reward < test_avg_reward:
                best_test_episode_reward = test_avg_reward

                file_name = exp_dir / f'best_ppo_{step_idx}_{best_test_episode_reward:.2f}.pt'
                state_dict = {
                    'step': step_idx,
                    'model': model.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'best_test_episode_reward': best_test_episode_reward,
                }
                torch.save(state_dict, file_name)

            file_name = exp_dir / f'latest_ppo.pt'
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
    parser.add_argument('--exp-name', type=str, default='exp_ppo')
    parser.add_argument('--env-name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--num-env', type=int, default=16)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--training-steps', type=int, default=1000000)
    parser.add_argument('--discount-rate', type=float, default=0.99)
    parser.add_argument('--ppo-trajectory-size', type=int, default=2049)
    parser.add_argument('--ppo-bs', type=int, default=64)
    parser.add_argument('--ppo-epochs', type=float, default=10)
    parser.add_argument('--ppo-gae-lambda', type=float, default=0.95)
    parser.add_argument('--ppo-normalize-gae', action='store_true')
    parser.add_argument('--ppo-eps', type=float, default=0.2)
    parser.add_argument('--lr-actor', type=float, default=1e-4)
    parser.add_argument('--lr-critic', type=float, default=1e-3)
    parser.add_argument('--loss-entropy-coef', type=float, default=1e-3)
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0)
    parser.add_argument('--test-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    main(args)
