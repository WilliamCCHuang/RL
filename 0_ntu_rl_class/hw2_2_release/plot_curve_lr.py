import wandb
import numpy as np

from DP_solver_2_2 import Q_Learning
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

STEP_REWARD = -0.1
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
DISCOUNT_FACTOR = 0.99
# LEARNING_RATE = 0.01
EPSILON = 0.2
BUFFER_SIZE = 10000
UPDATE_FREQUENCY = 200
SAMPLE_BATCH_SIZE = 500


def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt"):
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
    )
    return grid_world


def run_Q_Learning(grid_world: GridWorld, iter_num: int, lr: float):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=lr,
        epsilon=EPSILON,
        buffer_size=BUFFER_SIZE,
        update_frequency=UPDATE_FREQUENCY,
        sample_batch_size=SAMPLE_BATCH_SIZE,
    )
    policy_iteration.record_episodic_reward = True
    policy_iteration.record_episodic_loss = True
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title="Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_lr={lr}.png",
    )
    
    wandb.init(
        project="RL HW2 Reward Loss Curve of Q Learning in Different LR",
        name=f'lr={lr}',
        config={
            "lr": lr,
        }
    )
    for i in range(len(policy_iteration.episodic_rewards)):
        start, end = max(0, i - 9), i + 1
        last_ten_avg_rewards = policy_iteration.episodic_rewards[start:end]
        wandb.log({'Average Non-Discounted Episodic Reward': np.sum(last_ten_avg_rewards) / 10})
    for i in range(len(policy_iteration.episodic_losses)):
        start, end = max(0, i - 9), i + 1
        last_ten_avg_losses = policy_iteration.episodic_losses[start:end]
        wandb.log({'Average Absolute Estimation Loss': np.sum(last_ten_avg_losses) / 10})
    wandb.finish()

if __name__ == '__main__':
    for lr in [0.01, 0.1, 0.5, 0.9, 0.99]:
        grid_world = init_grid_world()
        run_Q_Learning(grid_world, 50000, lr)
