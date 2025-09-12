import numpy as np
from collections import deque
from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(
            self,
            grid_world: GridWorld,
            discount_factor: float = 1.0,
            record_episodic_reward: bool = False,
            record_episodic_loss: bool = False,
            visualize: bool = False
        ):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space = grid_world.get_state_space()
        self.q_values = np.zeros((self.state_space, self.action_space))
        self.policy = np.ones((self.state_space, self.action_space)) / self.action_space
        self.policy_index = np.zeros(self.state_space, dtype=int)

        self.record_episodic_reward = record_episodic_reward
        self.record_episodic_loss = record_episodic_loss
        self.episodic_rewards = [[]]
        self.episodic_losses = [[]]
        self.visualize = visualize

    def choose_action(self, s):
        p = self.policy[s]
        num_a = self.grid_world.get_action_space()
        return np.random.choice(num_a, p=p)

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values


class MonteCarloPolicyIteration(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        
        G = 0
        for t in np.arange(len(state_trace))[::-1]:
            s = state_trace[t]
            a = action_trace[t]
            r = reward_trace[t]
            G = self.discount_factor * G + r
            error = G - self.q_values[s, a]
            self.q_values[s, a] += self.lr * error

            if self.record_episodic_loss:
                self.episodic_losses[-1].append(abs(error))

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy

        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                self.policy[s, a] = self.epsilon / self.grid_world.get_action_space()
            best_a = np.argmax(self.q_values[s])
            self.policy[s, best_a] += 1 - self.epsilon

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace = []
        action_trace = []
        reward_trace = []
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            action = self.choose_action(current_state)
            next_state, reward, done = self.grid_world.step(action)

            state_trace.append(current_state)
            action_trace.append(action)
            reward_trace.append(reward)

            if self.record_episodic_reward:
                self.episodic_rewards[-1].append(reward)

            if done:
                iter_episode += 1
                self.policy_evaluation(state_trace, action_trace, reward_trace)
                self.policy_improvement()

                if self.record_episodic_reward:
                    self.episodic_rewards[-1] = np.mean(self.episodic_rewards[-1])
                    self.episodic_rewards.append([])
                
                if self.record_episodic_loss:
                    self.episodic_losses[-1] = np.mean(self.episodic_losses[-1])
                    self.episodic_losses.append([])
            
                state_trace = []
                action_trace = []
                reward_trace = []

                if self.visualize and iter_episode % 10000 == 0:
                    self.grid_world.visualize(
                        self.get_max_state_values(),
                        self.get_policy_index(),
                        title=f'MC Policy Iteration ({iter_episode})',
                        show=False,
                        filename=f'MC_policy_iteration.png',
                    )

            current_state = next_state

        if len(self.episodic_rewards[-1]) == 0:
            self.episodic_rewards = self.episodic_rewards[:-1]
        if len(self.episodic_losses[-1]) == 0:
            self.episodic_losses = self.episodic_losses[:-1]


class SARSA(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
    
    def cal_error(self, s, a, r, s2, a2, done):
        return r + self.discount_factor * self.q_values[s2, a2] * (1 - done) - self.q_values[s, a]
    
    def policy_eval_improve(self, s, a, r, s2, a2, done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        
        error = self.cal_error(s, a, r, s2, a2, done)
        self.q_values[s, a] += self.lr * error

        for a in range(self.grid_world.get_action_space()):
            self.policy[s, a] = self.epsilon / self.grid_world.get_action_space()
        best_a = np.argmax(self.q_values[s])
        self.policy[s, best_a] += 1 - self.epsilon

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        curr_s = self.grid_world.reset()
        curr_a = self.choose_action(curr_s)
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            next_s, r, done = self.grid_world.step(curr_a)
            next_a = self.choose_action(next_s)

            if self.record_episodic_reward:
                self.episodic_rewards[-1].append(r)
            if self.record_episodic_loss:
                error = self.cal_error(curr_s, curr_a, r, next_s, next_a, done)
                self.episodic_losses[-1].append(abs(error))

            self.policy_eval_improve(curr_s, curr_a, r, next_s, next_a, done)

            curr_s = next_s
            curr_a = next_a
            
            if done:
                iter_episode += 1

                if self.record_episodic_reward:
                    self.episodic_rewards[-1] = np.mean(self.episodic_rewards[-1])
                    self.episodic_rewards.append([])
                
                if self.record_episodic_loss:
                    self.episodic_losses[-1] = np.mean(self.episodic_losses[-1])
                    self.episodic_losses.append([])

                if self.visualize and iter_episode % 10000 == 0:
                    self.grid_world.visualize(
                        self.get_max_state_values(),
                        self.get_policy_index(),
                        title=f'SARSA ({iter_episode})',
                        show=False,
                        filename=f'SARSA_iteration.png',
                    )

        if len(self.episodic_rewards[-1]) == 0:
            self.episodic_rewards = self.episodic_rewards[:-1]
        if len(self.episodic_losses[-1]) == 0:
            self.episodic_losses = self.episodic_losses[:-1]


class Q_Learning(DynamicProgramming):
    def __init__(
            self,
            grid_world: GridWorld,
            discount_factor: float,
            learning_rate: float,
            epsilon: float,
            buffer_size: int,
            update_frequency: int,
            sample_batch_size: int,
        ):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.buffer = deque(maxlen=buffer_size)
        self.update_frequency = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        return [self.buffer[i] for i in np.random.choice(len(self.buffer), size=self.sample_batch_size)]

    def cal_error(self, s, a, r, s2, done):
        return r + self.discount_factor * np.max(self.q_values[s2]) * (1 - done) - self.q_values[s, a]
    
    def policy_eval_improve(self, s, a, r, s2, done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        error = self.cal_error(s, a, r, s2, done)
        self.q_values[s, a] += self.lr * error

        for a in range(self.grid_world.get_action_space()):
            self.policy[s, a] = self.epsilon / self.grid_world.get_action_space()
        best_a = np.argmax(self.q_values[s])
        self.policy[s, best_a] += 1 - self.epsilon

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        transition_count = 0
        curr_s = self.grid_world.reset()

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            curr_a = self.choose_action(curr_s)
            next_s, r, done = self.grid_world.step(curr_a)
            self.add_buffer(curr_s, curr_a, r, next_s, done)
            transition_count += 1

            if self.record_episodic_reward:
                self.episodic_rewards[-1].append(r)
            if self.record_episodic_loss:
                error = self.cal_error(curr_s, curr_a, r, next_s, done)
                self.episodic_losses[-1].append(abs(error))

            B = []
            if transition_count % self.update_frequency == 0:
                B = self.sample_batch()
            for s, a, r, s2, d in B:
                self.policy_eval_improve(s, a, r, s2, d)

            if done:
                if self.record_episodic_reward:
                    self.episodic_rewards[-1] = np.mean(self.episodic_rewards[-1])
                    self.episodic_rewards.append([])
                
                if self.record_episodic_loss:
                    self.episodic_losses[-1] = np.mean(self.episodic_losses[-1])
                    self.episodic_losses.append([])
                
                iter_episode += 1

                if self.visualize and iter_episode % 10000 == 0:
                    self.grid_world.visualize(
                        self.get_max_state_values(),
                        self.get_policy_index(),
                        title=f'Q_Learning ({iter_episode})',
                        show=False,
                        filename=f'Q_Learning.png',
                    )
            
            curr_s = next_s

        if len(self.episodic_rewards[-1]) == 0:
            self.episodic_rewards = self.episodic_rewards[:-1]
        if len(self.episodic_losses[-1]) == 0:
            self.episodic_losses = self.episodic_losses[:-1]