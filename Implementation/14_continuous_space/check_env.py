from collections import UserDict

import gym

import gym.envs.registration

# Solution: https://github.com/bulletphysics/bullet3/issues/4368
# Do this before importing pybullet_envs (adds an extra property env_specs as a property to the registry, so it looks like the <0.26 envspec version)
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry

import pybullet_envs

ENV_ID = 'MinitaurBulletEnv-v0'


if __name__ == '__main__':
    env = gym.make(ENV_ID)

    print('Observation space:')
    print(env.observation_space)  # (28,)
    print('Action space:')
    print(env.action_space)  # (8,)
    print('Sampled action')
    print(env.action_space.sample())  # (8,)
    print('Env:')
    print(env)

    obs = env.reset()  # not returned in the form `(obs, info)`
    print('Observation')
    print(obs)  # (28,)
    print(obs.dtype)  # float64
    
    input('Press any key to exit\n')
    env.close()