import gymnasium as gym

from . import agents

gym.register(
    id="Finish-Gate-Drive-Direct-v0",
    entry_point=f"{__name__}.finish_gate_drive_env:FinishGateDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.finish_gate_drive_env:FinishGateDriveEnvfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
