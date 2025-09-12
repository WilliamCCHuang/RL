import numpy as np

from DP_solver import (
    IterativePolicyEvaluation,
    PolicyIteration,
    ValueIteration,
    AsyncDynamicProgramming,
    InPlacePolicyIteration,
    InPlaceValueIteration,
    RealTimeValueIteration,
    PrioritizedSweepingValueIteration,
    QLearning,
    DoubleQLearning,
    DynaQLearning,
    SARSA
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

STEP_REWARD = -1.0
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
DISCOUNT_FACTOR = 0.9


def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt"):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_policy_evaluation(grid_world: GridWorld):
    print(bold(underline("Iterative Policy Evaluation")))
    policy = np.ones((grid_world.get_state_space(), 4)) / 4

    iterative_policy_evaluation = IterativePolicyEvaluation(
        grid_world, policy, discount_factor=DISCOUNT_FACTOR
    )
    iterative_policy_evaluation.run()

    grid_world.visualize(
        iterative_policy_evaluation.values,
        title="Iterative Policy Evaluation",
        show=False,
        filename="iterative_policy_evaluation.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    grid_world.reset()
    print()


def run_policy_iteration(grid_world: GridWorld):
    print(bold(underline("Policy Iteration")))
    policy_iteration = PolicyIteration(grid_world, discount_factor=DISCOUNT_FACTOR)
    policy_iteration.run()
    grid_world.visualize(
        policy_iteration.values,
        policy_iteration.policy,
        title="Policy Iteration",
        show=False,
        filename="policy_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(policy_iteration.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_value_iteration(grid_world: GridWorld):
    print(bold(underline("Value Iteration")))
    value_iteration = ValueIteration(grid_world, discount_factor=DISCOUNT_FACTOR)
    value_iteration.run()
    grid_world.visualize(
        value_iteration.values,
        value_iteration.policy,
        title="Value Iteration",
        show=False,
        filename="value_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(value_iteration.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_async_dynamic_programming(grid_world: GridWorld):
    print(bold(underline("Async Dynamic Programming")))
    async_dynamic_programming = AsyncDynamicProgramming(
        grid_world, discount_factor=DISCOUNT_FACTOR
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Async Dynamic Programming",
        show=False,
        filename="async_dynamic_programming.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_in_place_policy_iteration(grid_world: GridWorld):
    print(bold(underline("In-place Policy Iteration")))
    async_dynamic_programming = InPlacePolicyIteration(
        grid_world, discount_factor=DISCOUNT_FACTOR
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Async Value Iteration",
        show=False,
        filename="in_place_policy_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_in_place_value_iteration(grid_world: GridWorld):
    print(bold(underline("In-place Value Iteration")))
    async_dynamic_programming = InPlaceValueIteration(
        grid_world, discount_factor=DISCOUNT_FACTOR
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Async Value Iteration",
        show=False,
        filename="in_place_value_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_prioritised_sweeping_value_iteration(grid_world: GridWorld):
    print(bold(underline("Prioritised Sweeping Value Iteration")))
    async_dynamic_programming = PrioritizedSweepingValueIteration(
        grid_world, discount_factor=DISCOUNT_FACTOR
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Prioritised Sweeping Value Iteration",
        show=False,
        filename="prioritised_sweeping_value_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_real_time_value_iteration(grid_world: GridWorld):
    print(bold(underline("Real-time Value Iteration")))
    async_dynamic_programming = RealTimeValueIteration(
        grid_world, discount_factor=DISCOUNT_FACTOR,
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Real-time Value Iteration",
        show=False,
        filename="real_time_value_iteration.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_q_learning(grid_world: GridWorld):
    print(bold(underline("Q Learning")))
    async_dynamic_programming = QLearning(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=0.9,
        epsilon=0.1
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Q-Learning",
        show=False,
        filename="q_learning.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_double_q_learning(grid_world: GridWorld):
    print(bold(underline("Double Q Learning")))
    async_dynamic_programming = DoubleQLearning(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=0.9,
        epsilon=0.1
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Double Q-Learning",
        show=False,
        filename="double_q_learning.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_dyna_q_learning(grid_world: GridWorld):
    print(bold(underline("Dyna-Q Learning")))
    async_dynamic_programming = DynaQLearning(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=0.9,
        epsilon=0.1,
        n_planning=4
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="Dyna-Q Learning",
        show=False,
        filename="dyna_q_learning.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_sarsa(grid_world: GridWorld):
    print(bold(underline("SARSA")))
    async_dynamic_programming = SARSA(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=0.5,
        epsilon=0.1
    )
    async_dynamic_programming.run()
    grid_world.visualize(
        async_dynamic_programming.values,
        async_dynamic_programming.policy,
        title="SARSA",
        show=False,
        filename="sarsa.png",
    )
    print(f"Solved in {bold(green(grid_world.get_step_count()))} steps")
    history = grid_world.run_policy(async_dynamic_programming.policy, 0)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


if __name__ == "__main__":
    grid_world = init_grid_world(maze_file='maze.txt')
    # run_policy_evaluation(grid_world)
    # run_policy_iteration(grid_world)  # step count: 3256
    # run_value_iteration(grid_world)  # step count: 1144
    run_async_dynamic_programming(grid_world)
    # run_in_place_policy_iteration(grid_world)  # step count: 3146
    # run_in_place_value_iteration(grid_world)  # step count: 968
    # run_real_time_value_iteration(grid_world)  # step count: 1888
    # run_prioritised_sweeping_value_iteration(grid_world)  # step count: 444
    # run_q_learning(grid_world)  # 1012
    # run_double_q_learning(grid_world)
    run_dyna_q_learning(grid_world)  # 154
    # run_prioritized_sweeping_dyna_q_learning(grid_world)
    # run_sarsa(grid_world)  # 1088
