import numpy as np
from collections import deque

import torch


class Experience:

    def __init__(self, obs, action, reward, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done


class FirstLastExperience:

    def __init__(self, obs, action, dis_reward, last_obs):
        self.obs = obs
        self.action = action
        self.dis_reward = dis_reward
        self.last_obs = last_obs


class NStepExperienceMemory:

    def __init__(
        self,
        env,
        agent,
        n_steps=2
    ):
        self.env_pool = [env]
        if isinstance(env, (list, tuple)):
            self.pool = env
        self.agent = agent
        self.n_steps = n_steps
        self.episode_reward_list = []
        self.episode_step_list = []
        self.device = next(agent.parameters()).device

    def __iter__(self):
        """Collect experience and store it into memory deques.
        Once a memory deque is full, output it to train model.
        """
        # initialize
        obs_list = []
        memory_deque_list = []
        cur_reward_list = []
        cur_step_list = []

        for env in self.env_pool:
            obs = env.reset()
            
            obs_list.append(obs)
            memory_deque_list.append(deque(maxlen=self.n_steps))
            cur_reward_list.append(0)
            cur_step_list.append(0)

        # collect experience based on the current agent
        while True:
            obs_tensor = torch.tensor(np.array(obs_list)).to(self.device)
            actions = self.agent.sample_actions(obs_tensor)

            for env_idx, (env, obs, memory_deque) in enumerate(zip(self.env_pool, obs_list, memory_deque_list)):
                action = actions[env_idx]
                next_obs, reward, done, info = env.step(action)
                cur_reward_list[env_idx] += reward
                cur_step_list[env_idx] += 1

                # store the current transition
                exp = Experience(obs, action, reward, done)
                memory_deque.append(exp)

                # if enough transitions are collected, output them to train the model
                if len(memory_deque) == self.n_steps:
                    yield list(memory_deque)  # deque would not be cleared

                obs_list[env_idx] = next_obs


                if done:
                    # output the memory deque even it is not enough as the env is done
                    if 0 < len(memory_deque) < self.n_steps:
                        yield list(memory_deque)
                    # remove the oldest transition from the memory deque and output the memory deque until it is cleared
                    while len(memory_deque) > 1:
                        memory_deque.popleft()  # len(memory_deque) >= 1
                        yield list(memory_deque)

                    self.episode_reward_list.append(cur_reward_list[env_idx])
                    self.episode_step_list.append(cur_step_list[env_idx]) 
                    cur_reward_list[env_idx] = 0
                    cur_step_list[env_idx] = 0

                    # reset env
                    obs = env.reset()
                    obs_list[env_idx] = obs
                    memory_deque.clear()

    def pop_episode_rewards(self):
        res = self.episode_reward_list
        self.episode_reward_list = []
        self.episode_step_list = []
        return res
    
    def pop_episode_reward_step_pairs(self):
        res = list(zip(self.episode_reward_list, self.episode_step_list))
        self.episode_reward_list = []
        self.episode_step_list = []
        return res


class FirstLastExperienceMemory(NStepExperienceMemory):

    def __init__(
        self,
        env,
        agent,
        gamma,
        n_steps=1
    ):
        super().__init__(env, agent, n_steps+1)

        self.gamma = gamma
        self.n_steps = n_steps

    def __iter__(self):
        for n_step_exp_list in super().__iter__():
            first_exp = n_step_exp_list[0]
            last_exp = n_step_exp_list[-1]

            last_obs = None
            # if the exp is stopped earlily
            if last_exp.done and len(n_step_exp_list) <= self.n_steps:
                pass
            else:
                last_obs = last_exp.obs
                n_step_exp_list = n_step_exp_list[:-1]

            dis_reward = 0
            for exp in n_step_exp_list[::-1]:
                dis_reward = self.gamma * dis_reward + exp.reward
            

            yield FirstLastExperience(
                obs=first_exp.obs,
                action=first_exp.action,
                dis_reward=dis_reward,
                last_obs=last_obs
            )