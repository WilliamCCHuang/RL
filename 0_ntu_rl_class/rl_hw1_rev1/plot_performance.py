import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from DP_solver import (
    PolicyIteration,
    ValueIteration,
    InPlacePolicyIteration,
    InPlaceValueIteration,
    RealTimeValueIteration,
    PrioritizedSweepingValueIteration,
    QLearning,
    DynaQLearning
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

STEP_REWARD = -1.0
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0


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
    # grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def get_gt_values(grid_world, discount_factor):
    print(bold(underline("Get Ground True Value")))
    value_iteration = ValueIteration(grid_world, discount_factor=discount_factor)
    value_iteration.run()
    grid_world.reset()

    return value_iteration.values


def compute_value_error(gt_values, value_history):
    error_history = {}
    for step_count, est_values in value_history.items():
        error = np.max(np.abs(gt_values - est_values))
        # error = np.sqrt(((gt_values - est_values)**2).mean())
        error_history[step_count] = error

    return error_history


def evaluate_dp(dp, gt_values):
    dp.save_value_history = True
    dp.run()
    print(f"Solved in {bold(green(dp.grid_world.get_step_count()))} steps")
    error_history = compute_value_error(gt_values, dp.value_history)
    return error_history


def evaluate_policy_iteration(grid_world, discount_factor, gt_values):
    print(bold(underline("Policy Iteration")))
    dp = PolicyIteration(grid_world, discount_factor=discount_factor)
    error_history = evaluate_dp(dp, gt_values)
    grid_world.reset()

    return error_history


def evaluate_value_iteration(grid_world, discount_factor, gt_values):
    print(bold(underline("Value Iteration")))
    dp = ValueIteration(grid_world, discount_factor=discount_factor)
    error_history = evaluate_dp(dp, gt_values)
    grid_world.reset()
    
    return error_history


def evaluate_in_place_policy_iteration(grid_world, discount_factor, gt_values):
    print(bold(underline("In-place Policy Iteration")))
    async_dp = InPlacePolicyIteration(grid_world, discount_factor=discount_factor)
    error_history = evaluate_dp(async_dp, gt_values)
    grid_world.reset()
    
    return error_history


def evaluate_in_place_value_iteration(grid_world, discount_factor, gt_values):
    print(bold(underline("In-place Value Iteration")))
    async_dp = InPlaceValueIteration(grid_world, discount_factor=discount_factor)
    error_history = evaluate_dp(async_dp, gt_values)
    grid_world.reset()
    
    return error_history


def evaluate_prioritised_sweeping_value_iteration(grid_world, discount_factor, gt_values):
    print(bold(underline("Prioritised Sweeping Value Iteration")))
    async_dp = PrioritizedSweepingValueIteration(grid_world, discount_factor=discount_factor)
    error_history = evaluate_dp(async_dp, gt_values)
    grid_world.reset()
    
    return error_history


def evaluate_real_time_value_iteration(grid_world, discount_factor, gt_values):
    print(bold(underline("Real-time Value Iteration")))
    async_dp = RealTimeValueIteration(
        grid_world, discount_factor=discount_factor,
    )
    error_history = evaluate_dp(async_dp, gt_values)
    grid_world.reset()
    
    return error_history


def evaluate_q_learning(grid_world, discount_factor, gt_values):
    print(bold(underline("Q Learning")))
    async_dp = QLearning(grid_world, discount_factor=discount_factor, learning_rate=0.9, epsilon=0.1)
    error_history = evaluate_dp(async_dp, gt_values)
    grid_world.reset()
    
    return error_history


def evaluate_dyna_q_learning(grid_world, discount_factor, gt_values):
    print(bold(underline("Dyna Q Learning")))
    async_dp = DynaQLearning(grid_world, discount_factor=discount_factor, learning_rate=0.9, epsilon=0.1, n_planning=4)
    error_history = evaluate_dp(async_dp, gt_values)
    grid_world.reset()
    
    return error_history


if __name__ == "__main__":
    save_dir = Path('performance_results')
    save_dir.mkdir(parents=True, exist_ok=True)

    grid_world = init_grid_world()

    discount_factors = [0.1, 0.5, 0.9]
    value_erros_dict = {gamma: {'policy_iteration': {}, 'value_iteration': {}} for gamma in discount_factors}
    dp_methods = {
        'policy_iteration': evaluate_policy_iteration,
        'value_iteration': evaluate_value_iteration,
        'in_place_policy_iteration': evaluate_in_place_policy_iteration,
        'in_place_value_iteration': evaluate_in_place_value_iteration,
        'prioritized_sweeping_value_iteration': evaluate_prioritised_sweeping_value_iteration,
        'real_time_value_iteration': evaluate_real_time_value_iteration,
        'q_learning': evaluate_q_learning,
        'dyna_q_learning': evaluate_dyna_q_learning

    }
    for discount_factor in discount_factors:
        print(f'=== GAMMA ({discount_factor:.1f}) ===')
        gt_values = get_gt_values(grid_world, discount_factor)

        for dp_name, eval_dp_func in dp_methods.items():
            value_erros = eval_dp_func(grid_world, discount_factor, gt_values)
            value_erros_dict[discount_factor][dp_name] = value_erros

    non_dp_methods = ['policy_iteration', 'value_iteration']
    for gamma, method_error_dict in value_erros_dict.items():
        plt.close()
        fig = plt.figure(figsize=(12, 8))
        for dp_method, error_dict in method_error_dict.items():
            step_counts = list(error_dict.keys())
            error_per_step_count = list(error_dict.values())
            
            line_style = '--' if dp_method in non_dp_methods else '-'
            plt.plot(step_counts, error_per_step_count, line_style, label=dp_method)
        
        plt.legend()
        plt.ylim()
        plt.title(f'$\gamma={gamma}$')
        plt.xlabel('Step Count')
        plt.ylabel('$\Delta$')
        plt.tight_layout()

        save_path = str(save_dir / f'gamma={gamma}.png')
        plt.savefig(save_path)
