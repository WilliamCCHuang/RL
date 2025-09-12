import numpy as np

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space = grid_world.get_state_space()
        self.values = np.zeros(self.state_space)

    def get_all_state_values(self) -> np.array:
        return self.values


class MonteCarloPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float):
        """Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        
        episode_states = []
        episode_rewards = []
        returned_rewards = [[] for _ in range(self.grid_world.get_state_space())]

        current_state = self.grid_world.reset()
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            
            episode_states.append(current_state)
            episode_rewards.append(reward)
            current_state = next_state

            if done:
                G = 0
                for t in np.arange(len(episode_states))[::-1]:
                    s = episode_states[t]
                    r = episode_rewards[t]
                    G = self.discount_factor * G + r

                    if s not in episode_states[:t]:  # first-visit
                        returned_rewards[s].append(G)
                    
                episode_states = []
                episode_rewards = []

        for s in range(self.grid_world.get_state_space()):
            self.values[s] = np.mean(returned_rewards[s])


class TDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            td_error = reward + self.discount_factor * self.values[next_state] * (1 - done) - self.values[current_state]
            self.values[current_state] += self.lr * td_error

            current_state = next_state


class NstepTDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, num_step: int):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.n = num_step

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with N-step TD Algorithm

        t = 0
        T = np.inf
        observations = []

        current_state = self.grid_world.reset()
        observations.append((current_state, 0, False))

        while self.grid_world.check():
            if t < T:
                next_state, reward, done = self.grid_world.step()
                observations.append((next_state, reward, done))
                if done:
                    T = t + 1
            
            tau = t - self.n + 1
            if tau >= 0:  # then len(observations) >= self.n
                start = tau + 1
                end = min(tau + self.n, T) + 1

                G = 0
                for _, r, _ in observations[start:end][::-1]:
                    G = self.discount_factor * G + r

                if tau + self.n < T:
                    s, _, _ = observations[tau + self.n]
                    G += self.discount_factor**self.n * self.values[s]
                
                s = observations[tau][0]
                self.values[s] += self.lr * (G - self.values[s])

            if tau == T - 1:
                t = 0
                T = np.inf
                observations = []

                current_state = self.grid_world.reset()
                observations.append((current_state, 0, False))
            else:
                t += 1
