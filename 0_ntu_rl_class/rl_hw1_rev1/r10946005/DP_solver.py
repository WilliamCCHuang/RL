import heapq
import random
import numpy as np
from typing import Any

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0, save_value_history=False):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

        self.save_value_history = save_value_history
        self.value_history = {}  # {step_count: values}

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        # q_pi(s, a) = \sum_{s', r} p(s', r| s, a) [r + gamma * v_pi(s')]
        
        next_state, reward, done = self.grid_world.step(state, action)
        return reward + self.discount_factor * self.values[next_state] * (1 - done)
    
    def get_best_q_value_action_pair(self, state):
        q_values = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]

        return max(q_values), np.argmax(q_values)


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        # v_pi(s) = \sum_{a \in A} pi(a|s) * q_pi(s, a)
        #         = \sum_{a \in A} pi(a|s) \sum_{s', r} p(s', r| s, a) [r + gamma * v_pi(s')]

        action_probs = self.policy[state]  # (num_action,)

        state_value = 0
        for action in range(self.grid_world.get_action_space()):
            action_prob = action_probs[action]
            state_value += action_prob * self.get_q_value(state, action)

        return state_value

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        new_values = np.copy(self.values)
        
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)

        delta = np.max(np.abs(self.values - new_values))
        self.values = new_values

        return delta

    def run(self):
        while True:
            delta = self.evaluate()

            if delta < self.threshold:
                break


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action = self.policy[state]
        state_value = self.get_q_value(state, action)

        return state_value
    
    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            new_values = self.values.copy()
            for state in range(self.grid_world.get_state_space()):
                new_values[state] = self.get_state_value(state)

            delta = np.max(np.abs(self.values - new_values))
            self.values = new_values

            if self.save_value_history:
                self.value_history[self.grid_world.get_step_count()] = self.values.copy()

            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        
        is_policy_stable = True
        for state in range(self.grid_world.get_state_space()):
            old_a = self.policy[state]
            _, best_a = self.get_best_q_value_action_pair(state)

            self.policy[state] = best_a

            if self.save_value_history:
                self.value_history[self.grid_world.get_step_count()] = self.values.copy()

            if old_a != best_a:
                is_policy_stable = False

        return is_policy_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            self.policy_evaluation()
            is_policy_stable = self.policy_improvement()

            if is_policy_stable:
                break


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        best_q, _ = self.get_best_q_value_action_pair(state)

        return best_q

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            new_values = self.values.copy()
            for state in range(self.grid_world.get_state_space()):
                new_values[state] = self.get_state_value(state)

            delta = np.max(np.abs(self.values - new_values))
            self.values = new_values

            if self.save_value_history:
                self.value_history[self.grid_world.get_step_count()] = new_values.copy()

            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            _, best_a = self.get_best_q_value_action_pair(state)
            self.policy[state] = best_a

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        self.policy_evaluation()
        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0, learning_rate: float = 0.9, epsilon: float = 0.1, n_planning=4):
        super().__init__(grid_world, discount_factor)

        self.q_values = np.zeros((self.grid_world.get_state_space(), self.grid_world.get_action_space()))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.memory = {}

    def take_action(self, state):
        best_action = np.argmax(self.q_values[state])

        if np.random.random() < self.epsilon:
            all_actions = list(range(self.grid_world.get_action_space()))
            all_actions.remove(best_action)
            return np.random.choice(all_actions)
        
        return best_action
    
    def cal_values(self):
        return self.q_values.max(axis=-1).copy()
    
    def cal_policy(self):
        return self.q_values.argmax(axis=-1).copy()
    
    def update_values_and_policy(self):
        self.values = self.cal_values()
        self.policy = self.cal_policy()

    def run(self):
        while True:
            old_q_values = self.q_values.copy()

            for state in range(self.grid_world.get_state_space()):
                action = self.take_action(state)
                next_state, reward, done = self.grid_world.step(state, action)
                td_error = reward + self.discount_factor * self.q_values[next_state].max() * (1 - done) - self.q_values[state, action]
                self.q_values[state, action] += self.learning_rate * td_error
                self.memory[(state, action)] = next_state, reward, done

                for _ in range(self.n_planning):
                    for (state, action), (next_state, reward, done) in self.memory.items():
                        td_error = reward + self.discount_factor * self.q_values[next_state].max() * (1 - done) - self.q_values[state, action]
                        self.q_values[state, action] += self.learning_rate * td_error

                if self.save_value_history:
                    self.value_history[self.grid_world.get_step_count()] = self.cal_values()

            delta = np.max(np.abs(old_q_values - self.q_values))
            if delta < self.threshold:
                self.update_values_and_policy()
                break


class InPlacePolicyIteration(AsyncDynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action = self.policy[state]
        state_value = self.get_q_value(state, action)

        return state_value
    
    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            old_values = self.values.copy()
            for state in range(self.grid_world.get_state_space()):
                self.values[state] = self.get_state_value(state)

                if self.save_value_history:
                    self.value_history[self.grid_world.get_step_count()] = self.values.copy()

            delta = np.max(np.abs(self.values - old_values))

            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        
        is_policy_stable = True
        for state in range(self.grid_world.get_state_space()):
            old_a = self.policy[state]
            _, best_a = self.get_best_q_value_action_pair(state)

            self.policy[state] = best_a

            if self.save_value_history:
                self.value_history[self.grid_world.get_step_count()] = self.values.copy()

            if old_a != best_a:
                is_policy_stable = False

        return is_policy_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            self.policy_evaluation()
            is_policy_stable = self.policy_improvement()

            if is_policy_stable:
                break


class InPlaceValueIteration(AsyncDynamicProgramming):

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        while True:
            old_values = self.values.copy()
            for state in range(self.grid_world.get_state_space()):
                max_q, best_a = self.get_best_q_value_action_pair(state)

                self.values[state] = max_q
                self.policy[state] = best_a

                if self.save_value_history:
                    self.value_history[self.grid_world.get_step_count()] = self.values.copy()

            delta = np.max(np.abs(self.values - old_values))

            if delta < self.threshold:
                break


class PrioritizedSweepingValueIteration(AsyncDynamicProgramming):

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        super().__init__(grid_world, discount_factor)

        self._history = {}
        self._predecessors = {}  # {state: {precedessor_1, precedessor_2, ...}, ...}
        self._bellman_errors = {}  # {state: error, ...}
        self._pq = PriorityQueue()

    def _collect_predecessors(self):
        for state in range(self.grid_world.get_state_space()):
            q_values = []
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, done = self.grid_world.step(state, action)
                q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
                q_values.append(q_value)
                
                self._history[(state, action)] = (next_state, reward, done)

                if next_state not in self._predecessors:
                    self._predecessors[next_state] = {state}
                else:
                    self._predecessors[next_state].add(state)

            self.values[state] = np.max(q_values)
            self.policy[state] = np.argmax(q_values)

            if self.save_value_history:
                self.value_history[self.grid_world.get_step_count()] = self.values.copy()
    
    def _update_pq_with_history(self):
        for state in range(self.grid_world.get_state_space()):
            q_values = []
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, done = self._history[(state, action)]
                q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
                q_values.append(q_value)

            error = abs(max(q_values) - self.values[state])
            self._pq.update(item=state, priority=-error)  # max priority queue

    def run(self) -> None:
        self._collect_predecessors()
        self._update_pq_with_history()

        while not self._pq.is_empty:
            state = self._pq.pop()

            q_values = []
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, done = self.grid_world.step(state, action)
                q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
                q_values.append(q_value)

                # update env infos as step reward may not be constant
                self._history[(state, action)] = next_state, reward, done

            self.values[state] = max(q_values)
            self.policy[state] = np.argmax(q_values)

            if self.save_value_history:
                self.value_history[self.grid_world.get_step_count()] = self.values.copy()

            for p in self._predecessors[state]:
                q_values = []
                for action in range(self.grid_world.get_action_space()):
                    next_state, reward, done = self._history[(p, action)]
                    q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
                    q_values.append(q_value)
                error = abs(max(q_values) - self.values[p])
                if error > self.threshold:
                    self._pq.update(item=p, priority=-error)


class PriorityQueue:

    def __init__(self):
        self._elements = []
        self.count = 0

    def push(self, item: Any, priority: float):
        element = [priority, self.count, item]
        self._elements.append(element)
        heapq.heapify(self._elements)
        self.count += 1

    def pop(self):
        _, _, item = heapq.heappop(self._elements)
        return item
    
    @property
    def is_empty(self):
        return len(self._elements) == 0
    
    def update(self, item, priority):
        for idx, (p, _, i) in enumerate(self._elements):
            # item is already in queue
            if i == item:
                # the orginial priority is than the new priority, update the priority and sort the queue
                # otherwise, do nothing
                if p > priority:
                    self._elements[idx][0] = p  # update priority
                    heapq.heapify(self._elements)  # sort the queue
                break
        else:
            # item is not in queue, just add it into queue
            self.push(item, priority)


class RealTimeValueIteration(AsyncDynamicProgramming):

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        super().__init__(grid_world, discount_factor)

    def run(self):
        while True:
            unvisited_states = list(range(self.grid_world.get_state_space()))
            old_values = self.values.copy()

            while unvisited_states:
                state = np.random.choice(unvisited_states)

                while True:
                    best_action = None
                    best_q_value = -np.inf
                    done = False
                    next_state = None
                    for action in range(self.grid_world.get_action_space()):
                        _next_state, _reward, _done = self.grid_world.step(state, action)
                        q_value = _reward + self.discount_factor * self.values[_next_state] * (1 - _done)

                        if q_value > best_q_value:
                            best_action = action
                            best_q_value = q_value
                            done = _done
                            next_state = _next_state

                    self.policy[state] = best_action
                    self.values[state] = best_q_value

                    if self.save_value_history:
                        self.value_history[self.grid_world.get_step_count()] = self.values.copy()

                    if state in unvisited_states:
                        unvisited_states.remove(state)

                    if done:
                        break
                    
                    state = next_state

            delta = np.max(np.abs(old_values - self.values))
            if delta < self.threshold:
                break


class QLearning(AsyncDynamicProgramming):

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1, learning_rate: float = 0.9, epsilon: float = 0.1):
        super().__init__(grid_world, discount_factor)

        self.q_values = np.zeros((self.grid_world.get_state_space(), self.grid_world.get_action_space()))
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def take_action(self, state):
        best_action = np.argmax(self.q_values[state])

        if np.random.random() < self.epsilon:
            all_actions = list(range(self.grid_world.get_action_space()))
            all_actions.remove(best_action)
            return np.random.choice(all_actions)
        
        return best_action
    
    def cal_values(self):
        return self.q_values.max(axis=-1).copy()
    
    def cal_policy(self):
        return self.q_values.argmax(axis=-1).copy()
    
    def update_values_and_policy(self):
        self.values = self.cal_values()
        self.policy = self.cal_policy()
    
    def run(self):
        while True:
            old_q_values = self.q_values.copy()

            for state in range(self.grid_world.get_state_space()):
                action = self.take_action(state)
                next_state, reward, done = self.grid_world.step(state, action)
                td_error = reward + self.discount_factor * self.q_values[next_state].max() * (1 - done) - self.q_values[state, action]
                self.q_values[state, action] += self.learning_rate * td_error

                if self.save_value_history:
                    self.value_history[self.grid_world.get_step_count()] = self.cal_values()
            
            delta = np.max(np.abs(old_q_values - self.q_values))
            if delta < self.threshold:
                self.update_values_and_policy()
                break


class DynaQLearning(QLearning):

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1, learning_rate: float = 0.9, epsilon: float = 0.1, n_planning=4):
        super().__init__(grid_world, discount_factor, learning_rate, epsilon)

        self.n_planning = n_planning
        self.memory = {}

    def run(self):
        while True:
            old_q_values = self.q_values.copy()

            for state in range(self.grid_world.get_state_space()):
                action = self.take_action(state)
                next_state, reward, done = self.grid_world.step(state, action)
                td_error = reward + self.discount_factor * self.q_values[next_state].max() * (1 - done) - self.q_values[state, action]
                self.q_values[state, action] += self.learning_rate * td_error
                self.memory[(state, action)] = next_state, reward, done

                for _ in range(self.n_planning):
                    for (state, action), (next_state, reward, done) in self.memory.items():
                        td_error = reward + self.discount_factor * self.q_values[next_state].max() * (1 - done) - self.q_values[state, action]
                        self.q_values[state, action] += self.learning_rate * td_error

                if self.save_value_history:
                    self.value_history[self.grid_world.get_step_count()] = self.cal_values()

            delta = np.max(np.abs(old_q_values - self.q_values))
            if delta < self.threshold:
                self.update_values_and_policy()
                break


class DoubleQLearning(QLearning):

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1, learning_rate: float = 0.9, epsilon: float = 0.1):
        super().__init__(grid_world, discount_factor, learning_rate, epsilon)
        
        self.q_values_1 = self.q_values
        self.q_values_2 = np.zeros_like(self.q_values)

    @property
    def avg_q_values(self):
        return (self.q_values_1 + self.q_values_2) / 2
        
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            all_actions = list(range(self.grid_world.get_action_space()))
            return np.random.choice(all_actions)
        
        return np.argmax(self.avg_q_values[state])
    
    def update_values_and_policy(self):
        self.values = self.avg_q_values.max(axis=-1)
        self.policy = self.avg_q_values.argmax(axis=-1)
    
    def run(self):
        while True:
            old_values = self.avg_q_values.max(axis=-1).copy()

            for state in range(self.grid_world.get_state_space()):
                action = self.take_action(state)
                next_state, reward, done = self.grid_world.step(state, action)

                if np.random.random() < 0.5:
                    best_action_1 = np.argmax(self.q_values_1[next_state])
                    td_error = reward + self.discount_factor * self.q_values_2[next_state, best_action_1] * (1 - done) - self.q_values_1[state, action]
                    self.q_values_1[state, action] += self.learning_rate * td_error
                else:
                    best_action_2 = np.argmax(self.q_values_2[next_state])
                    td_error = reward + self.discount_factor * self.q_values_1[next_state, best_action_2] * (1 - done) - self.q_values_2[state, action]
                    self.q_values_2[state, action] += self.learning_rate * td_error

                if self.save_value_history:
                    self.value_history[self.grid_world.get_step_count()] = self.avg_q_values.max(axis=-1).copy()

            delta = np.max(np.abs(old_values - self.avg_q_values.max(axis=-1)))
            if delta < self.threshold:
                self.update_values_and_policy()
                break


class SARSA(QLearning):

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1, learning_rate: float = 0.9, epsilon: float = 0.1):
        super().__init__(grid_world, discount_factor, learning_rate, epsilon)

    def run(self):
        while True:
            prob = np.ones(self.grid_world.get_state_space()) / self.grid_world.get_state_space()
            old_values = self.q_values.copy()

            done = False
            state = np.random.choice(self.grid_world.get_state_space(), p=prob)
            action = self.take_action(state)
            while not done:
                next_state, reward, done = self.grid_world.step(state, action)
                next_action = self.take_action(next_state)
                td_error = reward + self.discount_factor * self.q_values[next_state, next_action] * (1 - done) - self.q_values[state, action]
                self.q_values[state, action] += self.learning_rate * td_error

                if self.save_value_history:
                    self.value_history[self.grid_world.get_step_count()] = self.q_values.max(axis=-1).copy()

                state = next_state
                action = next_action

            diff = np.abs(old_values - self.q_values)
            delta = np.max(diff)
            if delta < self.threshold:
                self.update_values_and_policy()
                break
            
            prob = diff / diff.sum()
