import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Cart_with_Double_Pendulum",
    entry_point=f"{__name__}.cart_with_double_pendulum_env:CartWithDoublePendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_double_pendulum_env:CartWithDoublePendulumEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)