import argparse
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from atari_env import make_atari_env
from a2c import A2C, train_a2c, test_a2c
    

def discount_reward(rewards, dones, discount_rate):
    # if `dones[-1]` is `True`, then `rewards[-1]` should be a reward
    # if `dones[-1]` is `False`, then `rewards[-1]` should be a predicted value

    rev_dis_rewards = []
    dis_r = 0
    for raw_r, done in zip(rewards[::-1], dones[::-1]):
        dis_r = dis_r + discount_rate * raw_r * (1 - done)
        rev_dis_rewards.append(dis_r)

    return rev_dis_rewards[::-1]


def iterative_training_steps(envs, model, rollout_steps, discount_rate, device):
    input_shape = envs[0].observation_space.shape  # (#frame, 84, 84)
    n_actions = envs[0].action_space.n  # 4

    cur_obs = np.array([env.reset() for env in envs])  # (#envs, #frame, 84, 84)
    done_records = [[False] for _ in range(len(envs))]

    obs_history = torch.zeros((len(envs), rollout_steps, *input_shape), dtype=torch.uint8)  # (#envs, rollout_steps, #frame, 84, 84)
    value_history = torch.zeros((len(envs), rollout_steps), dtype=torch.float32)
    action_history = torch.zeros((len(envs), rollout_steps), dtype=torch.int32)
    action_logit_history = torch.zeros((len(envs), rollout_steps, n_actions), dtype=torch.float32)
    raw_reward_history = torch.zeros((len(envs), rollout_steps), dtype=torch.float32)
    dis_reward_history = torch.zeros((len(envs), rollout_steps), dtype=torch.float32)

    while True:
        done_records = [[dones[-1]] for dones in done_records]  # start from scratch or the interrupted states
        
        # rollout and collect data
        for step_idx in range(rollout_steps):
            cur_obs_tensor = torch.tensor(cur_obs).to(device)  # (#envs, #frame, 84, 84))

            with torch.no_grad():
                pred_values, action_logits = model(cur_obs_tensor)
                # pred_values: (#envs,)
                # action_logits: (#envs, n_actions)
                action_probs = torch.softmax(action_logits, dim=-1).detach().cpu().numpy()
            actions = torch.tensor([np.random.choice(n_actions, p=prob) for prob in action_probs])  # (#envs,)

            # record current obs, selected action, action prob.
            obs_history[:, step_idx, ...] = cur_obs_tensor.detach().cpu()
            value_history[:, step_idx] = pred_values.detach().cpu()
            action_history[:, step_idx] = actions
            action_logit_history[:, step_idx, :] = action_logits.detach().cpu()

            for env_idx, env in enumerate(envs):
                _obs, _reward, _done, _truncated, _info = env.step(actions[env_idx])
                
                if _done:
                    _obs = env.reset()

                cur_obs[env_idx] = _obs
                raw_reward_history[env_idx, step_idx] = _reward
                done_records[env_idx].append(_done)

        # esitmate the values of the last state for boostrap if needed
        with torch.no_grad():
            last_obs_tensor = torch.tensor(cur_obs).to(device)
            pred_last_values, _ = model(last_obs_tensor)
            pred_last_values = pred_last_values.detach().cpu()  # (#envs,)
        
        # comput the discounted reward
        for env_idx, (pred_last_value, rewards, dones) in enumerate(zip(pred_last_values, raw_reward_history, done_records)):
            rewards = list(rewards)

            # dones[1:] record the done flag of the future env while dones[0] is set to `False` by default
            if dones[-1]:
                # a completed episode
                dis_rewards = discount_reward(rewards, dones[1:], discount_rate)
            else:
                # this episode is not ended, and then interrupt it
                dis_rewards = discount_reward(rewards + [pred_last_value], dones[1:] + [False], discount_rate)  # (rollout_steps + 1,)
                dis_rewards = dis_rewards[:-1]  # (rollout_steps,) # discard the last item as it is the predicted value

            dis_reward_history[env_idx] = torch.tensor(dis_rewards)  # (rollout_steps,)
        
        obs_history_output = obs_history.reshape(-1, *input_shape)  # (#envs * rollout_steps, #frame, 84, 84)
        value_history_output = value_history.flatten()  # (#envs * rollout_steps,)
        action_history_output = action_history.flatten()  # (#envs * rollout_steps,)
        action_logit_history_output = action_logit_history.reshape(-1, n_actions)  # (#envs * rollout_steps, n_actions)
        raw_reward_history_output = raw_reward_history.flatten()  # (#envs * rollout_steps,)
        dis_reward_history_output = dis_reward_history.flatten()  # (#envs * rollout_steps,)

        yield obs_history_output, value_history_output, action_history_output, action_logit_history_output, raw_reward_history_output, dis_reward_history_output


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

    # envs[0].observation_space.shape: (#frame, 84, 84)
    # envs[0].action_space.n: 4
    model = A2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-5)
    print(model)

    print()
    print('Start training...')
    step_idx = 0
    total_rollout_steps = 0
    best_episode_reward = None
    best_test_episode_reward = None

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        step_idx = state_dict['step']
        best_test_episode_reward = state_dict['best_test_episode_reward']
        print(f'Loaded resume. Continue training from the step {step_idx}. Best test episode reward: {best_test_episode_reward}')
    
    for obs_history, value_history, action_history, action_logit_history, raw_reward_history, dis_reward_history in iterative_training_steps(envs, model, args.rollout_steps, args.discount_rate, device):
        # obs_history: (#envs * rollout_steps, *input_shape)
        # value_history: (#envs * rollout_steps,)
        # action_history: (#envs * rollout_steps,)
        # action_logit_history: (#envs * rollout_steps, n_actions)
        # raw_reward_history: (#envs * rollout_steps,)
        # dis_reward_history: (#envs * rollout_steps,)
        # episode_steps: [] * #episode_steps * #envs
        # episode_reward: [] * #episode_steps * #envs

        train_a2c(model, obs_history, value_history, action_history, dis_reward_history, optimizer, writer, step_idx, args, device)

        step_idx += 1

        # if training is done
        if args.training_steps is not None and step_idx >= args.training_steps:
            break

        # do testing
        if step_idx % args.test_steps == 0:
            test_avg_steps, test_avg_reward = test_a2c(model, test_env, num_rounds=args.test_num_rounds, device=device)

            writer.add_scalar('test/steps', test_avg_steps, step_idx)
            writer.add_scalar('test/reward', test_avg_reward, step_idx)

            if best_test_episode_reward is None or best_test_episode_reward < test_avg_reward:
                best_test_episode_reward = test_avg_reward

                file_name = exp_dir / f'best_a2c_{step_idx}_{best_test_episode_reward:.2f}.pt'
                state_dict = {
                    'step': step_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_episode_reward': best_test_episode_reward,
                    'best_test_episode_reward': best_test_episode_reward,
                }
                torch.save(state_dict, file_name)
            
            print(f'Step: {step_idx} | test steps: {test_avg_steps:.2f} | test reward: {test_avg_reward:.2f} | best test episode reward: {best_test_episode_reward:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp-name', type=str, default='exp_a2c')
    parser.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--frame-count', type=int, default=2)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--training-steps', type=int, default=1000000)
    parser.add_argument('--rollout-steps', type=int, default=5)
    parser.add_argument('--discount-rate', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss-value-coef', type=float, default=0.5)
    parser.add_argument('--loss-entropy-coef', type=float, default=0.01)
    parser.add_argument('--clip-grad-norm', type=float, default=0.5)
    parser.add_argument('--test-steps', type=int, default=100)
    parser.add_argument('--test-num-rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    main(args)
