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
    
    # env should be `<TimeLimit<OrderEnforcing<PassiveEnvChecker<MinitaurBulletEnv<MinitaurBulletEnv-v0>>>>>`
    # we need to extract the core env object otherwise the returned value of `env.step()` would not be 
    while True:
        if 'env' not in dir(env):
            break
        env = env.env

    # env: <pybullet_envs.bullet.minitaur_gym_env.MinitaurBulletEnv object at 0x1412b4bb0>
    # The env `MinitaurBulletEnv-v0` which provided by `pybullet_envs` outputs a `obs` which is in the type of `np.float32`
    # We need to cast `obs` into `np.float32`
    env = AtariWrapper(env)

    return env