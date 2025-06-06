import argparse
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from atari_env import make_atari_env
from a2c import A2C, train_a2c, test_a2c
from train_a2c import iterative_training_steps

from i2a import I2A, EnvironmentModel, train_rollout_policy


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

    envs = [make_atari_env(args.env_name, args.frame_count) for _ in range(args.num_envs)]
    test_env = make_atari_env(args.env_name, args.frame_count, test_mode=True)

    obs_shape = envs[0].observation_space.shape  # (#frame, 84, 84)
    n_actions = envs[0].action_space.n  # 4

    env_model = EnvironmentModel(obs_shape, n_actions)
    env_model.load_state_dict(torch.load(args.env_model_path)['env_model'])
    env_model.eval()  # Set to evaluation mode for pre-trained model

    rollout_policy = A2C(obs_shape, n_actions)
    i2a_model = I2A(obs_shape, n_actions, env_model, rollout_policy, args.i2a_model_rollout_steps, args.rollout_hidden_dim).to(device)
    print(i2a_model)
    
    # Ensure i2a_optimizer only updates I2A's own parameters,
    # excluding env_model (frozen) and rollout_policy (trained separately)
    i2a_params = [
        p for n, p in i2a_model.named_parameters()
        if p.requires_grad and 'env_model' not in n and 'rollout_policy' not in n
    ]
    rollout_policy_params = list(i2a_model.rollout_policy.parameters())
    i2a_optimizer = optim.RMSprop(i2a_params, lr=args.lr, eps=1e-5)
    rollout_policy_optimizer = optim.Adam(rollout_policy_params, lr=args.lr_policy)

    print()
    print('Start training...')
    step_idx = 0
    best_test_episode_reward = None

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path)
        rollout_policy.load_state_dict(state_dict['rollout_policy'])
        i2a_model.load_state_dict(state_dict['i2a_model'])
        i2a_optimizer.load_state_dict(state_dict['i2a_optimizer'])
        rollout_policy_optimizer.load_state_dict(state_dict['rollout_policy_optimizer'])
        step_idx = state_dict['step']
        best_test_episode_reward = state_dict['best_test_episode_reward']
        print(f'Loaded resume. Continue training from the step {step_idx}. Best test episode reward: {best_test_episode_reward}')
    
    for obs_history, value_history, action_history, action_logit_history, raw_reward_history, dis_reward_history in iterative_training_steps(envs, i2a_model, args.rollout_steps, args.discount_rate, device):
        # obs_history: (#envs * rollout_steps, *input_shape)
        # value_history: (#envs * rollout_steps,)
        # action_history: (#envs * rollout_steps,)
        # action_logit_history: (#envs * rollout_steps, n_actions)
        # raw_reward_history: (#envs * rollout_steps,)
        # dis_reward_history: (#envs * rollout_steps,)
        # episode_steps: [] * #episode_steps * #envs
        # episode_reward: [] * #episode_steps * #envs

        train_a2c(i2a_model, obs_history, value_history, action_history, dis_reward_history, i2a_optimizer, writer, step_idx, args, device)
        train_rollout_policy(rollout_policy, obs_history, action_logit_history, rollout_policy_optimizer, writer, step_idx, args, device)

        step_idx += 1

        # if training is done
        if args.training_steps is not None and step_idx >= args.training_steps:
            break

        # do testing
        if step_idx % args.test_steps == 0:
            test_avg_steps, test_avg_reward = test_a2c(i2a_model, test_env, num_rounds=args.test_num_rounds, device=device)

            writer.add_scalar('test/steps', test_avg_steps, step_idx)
            writer.add_scalar('test/reward', test_avg_reward, step_idx)

            if best_test_episode_reward is None or best_test_episode_reward < test_avg_reward:
                best_test_episode_reward = test_avg_reward

                file_name = exp_dir / f'best_i2a_{step_idx}_{best_test_episode_reward:.2f}.pt'
                state_dict = {
                    'step': step_idx,
                    'rollout_policy': rollout_policy.state_dict(),
                    'i2a_model': i2a_model.state_dict(),
                    'i2a_optimizer': i2a_optimizer.state_dict(),
                    'rollout_policy_optimizer': rollout_policy_optimizer.state_dict(),
                    'best_episode_reward': best_test_episode_reward,
                    'best_test_episode_reward': best_test_episode_reward,
                }
                torch.save(state_dict, file_name)
            
            print(f'Step: {step_idx} | test steps: {test_avg_steps:.2f} | test reward: {test_avg_reward:.2f} | best test episode reward: {best_test_episode_reward:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp-name', type=str, default='exp_i2a')
    parser.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--frame-count', type=int, default=2)
    parser.add_argument('--env-model-path', type=str, default='./exp_env_model/best_env_model.pt')
    parser.add_argument('--i2a-model-rollout-steps', type=int, default=3)
    parser.add_argument('--rollout-hidden-dim', type=int, default=256)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--training-steps', type=int, default=1000000)
    parser.add_argument('--rollout-steps', type=int, default=5)
    parser.add_argument('--discount-rate', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-policy', type=float, default=1e-4)
    parser.add_argument('--loss-value-coef', type=float, default=0.5)
    parser.add_argument('--loss-entropy-coef', type=float, default=0.01)
    parser.add_argument('--clip-grad-norm', type=float, default=0.5)
    parser.add_argument('--test-steps', type=int, default=100)
    parser.add_argument('--test-num-rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    main(args)
