import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

FINISH_GATE_CFG = RigidObjectCfg(
    prim_path="World/envs/env_.*/FinishGate",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/content/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/custom_assets/finish_gate.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg()
)

FINISH_GATE_CFG = sim_utils.UsdFileCfg(
    usd_path="/content/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/custom_assets/finish_gate.usd",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
)

NO_PASS_GATE_CFG = sim_utils.UsdFileCfg(
    usd_path="/content/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/custom_assets/no_pass_gate.usd",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
)