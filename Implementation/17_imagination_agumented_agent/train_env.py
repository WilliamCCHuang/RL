import argparse
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from atari_env import make_atari_env
from a2c import A2C
from i2a import EnvironmentModel, train_env_model


def iterative_training_steps(envs, a2c_model, args, device):
    input_shape = envs[0].observation_space.shape  # (#frame, 84, 84)
    n_actions = envs[0].action_space.n  # 4

    cur_obs = np.array([env.reset() for env in envs])  # (#envs, #frame, 84, 84)
    
    # necessary information for training an Env mdoel
    # note that the 1st channel represents the batch idx
    batch_idx = 0
    obs_history = torch.zeros((args.bs, *input_shape), dtype=torch.uint8)  # (bs, #freme, 84, 84)
    next_obs_history = torch.zeros((args.bs, *input_shape), dtype=torch.float32)  # (bs, #freme, 84, 84)
    action_history = torch.zeros(args.bs, dtype=torch.int32)
    reward_history = torch.zeros(args.bs, dtype=torch.float32)

    # for checking the quality of the A2C model
    cur_steps = [0] * len(envs)
    cur_reward = [0] * len(envs)
    episode_steps = [] 
    episode_reward = []

    while True:
        cur_obs_tensor = torch.tensor(cur_obs).to(device)  # (#envs, #frame, 84, 84))

        # sample action according to the A2C model
        with torch.no_grad():
            pred_values, pred_action_logits = a2c_model(cur_obs_tensor)
            # pred_values: (#envs,)
            # pred_action_logits: (#envs, n_actions)
            pred_action_probs = torch.softmax(pred_action_logits, dim=-1).detach().cpu().numpy()
        actions = torch.tensor([np.random.choice(n_actions, p=prob) for prob in pred_action_probs])  # (#envs,)

        for env_idx, (env, action) in enumerate(zip(envs, actions)):
            _obs, _reward, _done, _truncated, _info = env.step(action)

            # collect training data
            obs_history[batch_idx] = cur_obs_tensor[env_idx].detach().cpu()
            next_obs_history[batch_idx] = torch.tensor(_obs)  # (#frame, 84, 84)
            action_history[batch_idx] = action
            reward_history[batch_idx] = _reward

            cur_steps[env_idx] += 1
            cur_reward[env_idx] += _reward

            batch_idx += 1

            # a batch of data is collected, and then return it
            if batch_idx == args.bs:
                yield obs_history, next_obs_history, action_history, reward_history, episode_steps, episode_reward

                # reset
                batch_idx = 0
                episode_steps = []
                episode_reward = []

            if _done:
                _obs = env.reset()

                episode_steps.append(cur_steps[env_idx])
                episode_reward.append(cur_reward[env_idx])
                cur_steps[env_idx] = 0
                cur_reward[env_idx] = 0

            cur_obs[env_idx] = _obs
        

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

    a2c_model = A2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    state_dict = torch.load(args.a2c_weight_path, map_location=device)
    if 'model' not in state_dict:
        raise KeyError(f'The checkpoint {args.a2c_weight_path} does not contain the key `model`')
    a2c_model.load_state_dict(state_dict['model'])
    print(f'Loaded A2C model from {args.a2c_weight_path}')

    env_model = EnvironmentModel(input_shape=envs[0].observation_space.shape, n_actions=envs[0].action_space.n).to(device)
    optimizer = optim.Adam(env_model.parameters(), lr=args.lr)
    print(env_model)

    step_idx = 0
    best_loss = np.inf
    for obs_history, next_obs_history, action_history, reward_history, episode_steps, episode_reward in iterative_training_steps(envs, a2c_model, args, device):
        loss_total, loss_obs, loss_reward = train_env_model(env_model, obs_history, next_obs_history, action_history, reward_history, optimizer, writer, step_idx, args, device)

        step_idx += 1

        msg = f'Step: {step_idx} | loss: {loss_total:.4f} | loss obs: {loss_obs:.4f} | loss reward: {loss_reward:.4f}'
        if len(episode_steps) > 0:  # one episode is done
            avg_steps = np.mean(episode_steps)
            avg_reward = np.mean(episode_reward)

            msg += f' | avg episode steps: {avg_steps:.2f} | avg episode reward: {avg_reward:.2f}'
            
            writer.add_scalar('a2c/avg_steps', avg_steps, step_idx)
            writer.add_scalar('a2c/avg_reward', avg_reward, step_idx)

        if loss_total < best_loss:
            msg += f' | New best loss: {best_loss:.4f} -> {loss_total:.4f}'
            best_loss = loss_total

            file_name = exp_dir / f'best_em.pt'
            state_dict = {
                'step': step_idx,
                'env_model': env_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(state_dict, file_name)

        print(msg)

        if step_idx == args.training_steps:
            print('Training is done.')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp-name', type=str, default='exp_env_model')
    parser.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--frame-count', type=int, default=2)
    parser.add_argument('--a2c-weight-path', type=str)
    parser.add_argument('--training-steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--obs-weight', type=float, default=1.0)
    parser.add_argument('--reward-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20)

    args = parser.parse_args()

    main(args)

