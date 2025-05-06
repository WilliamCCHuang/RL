import warnings
import numpy as np

import gymnasium as gym

import wandb

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN


warnings.filterwarnings("ignore")

# Set hyper params (configurations) for training
my_config = {
    "run_id": "cartpole_a2c_1",  # TODO: change it

    "algorithm": A2C,
    "policy_network": "MlpPolicy",  # TODO: change it
    "save_path": "models/cartpole_a2c",  # TODO: change it

    "epoch_num": 100,  # TODO: change it
    "timesteps_per_epoch": 200,  # TODO: change it
    "eval_episode_num": 1,  # TODO: change it
}


def make_env():
    env = gym.make('CartPole-v1', render_mode="human")
    return env


def train(env, model, config):
    current_best = -np.inf
    for epoch in range(config["epoch_num"]):
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False
        )

        # Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_reward = 0
        for seed in range(config["eval_episode_num"]):
            done = False

            # Set seed using old Gym API
            env.seed(seed)
            obs = env.reset()

            # Interact with env using old Gym API
            total_reward = 0
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            
            avg_reward += total_reward / config["eval_episode_num"]
        
        print("Avg reward:", avg_reward)
        print()
        wandb.log(
            {"avg_reward": avg_reward}
        )
        
        # Save best model
        if current_best < avg_reward:
            print("Saving Model")
            current_best = avg_reward
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")


if __name__ == "__main__":
    # Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"]
    )

    env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        env,
        verbose=1,
        tensorboard_log=my_config["run_id"]
    )
    train(env, model, my_config)
