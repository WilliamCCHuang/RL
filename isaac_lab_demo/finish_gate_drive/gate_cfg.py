import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

FINISH_GATE_CFG = RigidObjectCfg(
    # prim_path="/World/envs/env_.*/FinishGate",
    prim_path="/World/Visuals/FinishGate",
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
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)

NO_PASS_GATE_CFG = RigidObjectCfg(
    # prim_path="/World/envs/env_.*/FinishGate",
    prim_path="/World/envs/env_.*/NoPassGate",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/content/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/custom_assets/NoPassGate.usd",
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
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)

# NO_PASS_GATE_CFG = ArticulationCfg(
#     prim_path="/World/Visuals/NoPassGate",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/content/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/custom_assets/no_pass_gate.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.0),
#         joint_pos={}
#         # joint_pos={
#         #     "Wheel__Knuckle__Front_Left": 0.0,
#         #     "Wheel__Knuckle__Front_Right": 0.0,
#         #     "Wheel__Upright__Rear_Right": 0.0,
#         #     "Wheel__Upright__Rear_Left": 0.0,
#         #     "Knuckle__Upright__Front_Right": 0.0,
#         #     "Knuckle__Upright__Front_Left": 0.0,
#         # },
#     ),
#     actuators={}
#     # actuators={
#     #     "throttle": ImplicitActuatorCfg(
#     #         joint_names_expr=["Wheel.*"],
#     #         effort_limit=40000.0,
#     #         velocity_limit=100.0,
#     #         stiffness=0.0,
#     #         damping=100000.0,
#     #     ),
#     #     "steering": ImplicitActuatorCfg(
#     #         joint_names_expr=["Knuckle__Upright__Front.*"],
#     #         effort_limit=40000.0,
#     #         velocity_limit=100.0,
#     #         stiffness=1000.0,
#     #         damping=0.0,
#     #     ),
#     # },
# )

# FINISH_GATE_CFG = sim_utils.UsdFileCfg(
#     usd_path="/content/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/custom_assets/finish_gate.usd",
#     rigid_props=sim_utils.RigidBodyPropertiesCfg(
#         disable_gravity=False,
#         retain_accelerations=False,
#         linear_damping=0.0,
#         angular_damping=0.0,
#         max_linear_velocity=1000.0,
#         max_angular_velocity=1000.0,
#         max_depenetration_velocity=1.0,
#     ),
# )

# NO_PASS_GATE_CFG = sim_utils.UsdFileCfg(
#     usd_path="/content/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/leatherback/custom_assets/no_pass_gate.usd",
#     rigid_props=sim_utils.RigidBodyPropertiesCfg(
#         disable_gravity=False,
#         retain_accelerations=False,
#         linear_damping=0.0,
#         angular_damping=0.0,
#         max_linear_velocity=1000.0,
#         max_angular_velocity=1000.0,
#         max_depenetration_velocity=1.0,
#     ),
# )