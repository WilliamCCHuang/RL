import cv2
import numpy as np
from collections import deque

import gymnasium as gym
from stable_baselines3.common import atari_wrappers


class AtariWrapper(gym.Wrapper):

    def __init__(self, env, frame_stack=4):
        super().__init__(env)
        self.env = env  # `env` should be an object of `atari_wrappers.AtariWrapper`
        self.frame_stack = frame_stack
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(frame_stack, 84, 84), dtype=np.uint8)
        self.buffer = deque(maxlen=frame_stack)

    def reset(self):
        low_obs = self.env.observation_space.low  # zero tensor of the shape (84, 84, 1)
        low_obs = self.preprocess_obs(low_obs)
        
        # reset buffer with all zero tensors
        for _ in range(self.buffer.maxlen):
            self.buffer.append(low_obs)

        obs, extra = self.env.reset()  # obs: (84, 84, 1), extra: {}
        obs = self.preprocess_obs(obs)
        self.buffer.append(obs)
        stack_obs = np.concatenate(self.buffer, axis=0)  # (#frame, 84, 84)

        return stack_obs
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # `obs`: (84, 84, 1)
        # `reward`: float
        # `done`: float
        # `truncated`: bool. Whether the truncation condition outside the scope of the MDP is satisfied.
        # Typically, this is a timelimit, but could also be used to indicate an agent physically
        # going out of bounds. Can be used to end the episode prematurely before a terminal state
        # is reached. If true, the user needs to call reset().

        obs = self.preprocess_obs(obs)
        self.buffer.append(obs)

        stack_obs = np.concatenate(self.buffer, axis=0)
        return stack_obs, reward, done, truncated, info

    def preprocess_obs(self, obs):
        # obs: (84, 84, 1)
        obs = np.transpose(obs, (2, 0, 1))  # (frame_stack, 84, 84)
        return obs


def make_atari_env(env_name, frame_count, test_mode=False):
    env = gym.make(env_name)
    
    # env should be `<OrderEnforcing<PassiveEnvChecker<AtariEnv<BreakoutNoFrameskip-v4>>>>`
    # we need to extract the core env object as `AtariWrapper` only accept `gymnasium.Env` objects
    while True:
        if not isinstance(env, gym.Wrapper):
            break
        env = env.env

    env = atari_wrappers.AtariWrapper(
        env=env,
        clip_reward=not test_mode,
        noop_max=0,
        terminal_on_life_loss=not test_mode
    )
    env = AtariWrapper(env, frame_stack=frame_count)

    return env