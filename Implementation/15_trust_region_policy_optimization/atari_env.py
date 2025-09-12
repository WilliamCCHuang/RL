import numpy as np
from collections import UserDict

import gym

import gym.envs.registration

# Solution: https://github.com/bulletphysics/bullet3/issues/4368
# Do this before importing pybullet_envs (adds an extra property env_specs as a property to the registry, so it looks like the <0.26 envspec version)
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry

import pybullet_envs


class AtariWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs = obs.astype(np.float32)
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.astype(np.float32)
        return obs, reward, done, info
    

def make_atari_env(env_name):
    env = gym.make(env_name)

    # The env outputs an `obs` which is in the type of `np.float32`
    # We need to cast `obs` into `np.float32`
    env = AtariWrapper(env)

    return env