import gym
import pybullet_envs

ENV_ID = 'MinitaurBulletEnv-v0'
RENDER = True


if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV_ID)
    spec._kwargs['render'] = RENDER
    env = gym.make(ENV_ID)

    print('Observation space:')
    print(env.observation_space)  # (28,)
    print('Action space:')
    print(env.action_space)  # (8,)
    print('Sampled action')
    print(env.action_space.sample())  # (8,)
    print('Env:')
    print(env)

    obs = env.reset()
    print('Observation')
    print(obs)  # (28,)
    input('Press any key to exit\n')
    env.close()