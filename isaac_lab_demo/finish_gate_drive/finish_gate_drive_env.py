from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import Camera, CameraCfg

from cfgs.waypoint_cfg import WAYPOINT_CFG
from cfgs.car_cfg import CAR_CFG
from cfgs.gate_cfg import (
    FINISH_GATE_CFG,
    FINISH_GATE_WITH_BASE_CFG,
    NO_PASS_GATE_CFG,
    NO_PASS_GATE_WITH_BASE_CFG
)

import gymnasium as gym
import agents

gym.register(
    id="Finish-Gate-Drive-Direct",
    entry_point=f"finish_gate_drive_env:FinishGateDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "finish_gate_drive_env:FinishGateDriveEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

@configclass
class FinishGateDriveEnvCfg(DirectRLEnvCfg):
    decimation = 4
    env_spacing = 32.0
    episode_length_s = 20.0

    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(-5, 0, 30), lookat=(-5, 0, 0))
    
    # cfgs
    car_cfg: ArticulationCfg = CAR_CFG.replace(prim_path="/World/envs/env_.*/Car")
    car_camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Car/Rigid_Bodies/Chassis/Camera_Right",
        update_period=0.1,
        height=120, # 480,
        width=160, # 640,
        spawn=None,  # set to `None` as the `Camera_Right` is already present. See https://github.com/isaac-sim/IsaacLab/blob/802ec5b7df771be1b91bc6744b2442eaa8f458df/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py#L45
        data_types=["rgb"],
        debug_vis=False
    )
    waypoint_cfg: RigidObjectCfg = WAYPOINT_CFG
    finish_gate_cfg: RigidObjectCfg = FINISH_GATE_CFG
    finish_gate_with_base_cfg: RigidObjectCfg = FINISH_GATE_WITH_BASE_CFG
    no_pass_gate_cfg: RigidObjectCfg = NO_PASS_GATE_CFG
    no_pass_gate_with_base_cfg: RigidObjectCfg = NO_PASS_GATE_WITH_BASE_CFG

    # env space
    action_space = 2  # one for throttle, and another for steering
    observation_space = [car_camera_cfg.height, car_camera_cfg.width, 3]
    state_space = 0

    # joint names
    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]
    
    # task-specific configuration
    num_goals = 5

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    target_y_offset_scale = 1.0
    car_to_gate_center_dis_tolerance = 0.25
    cat_to_gate_center_dis_change_weight = 1.0
    goal_reached_weight = 10.0
    car_heading_to_gate_center_coef = 0.25
    car_heading_to_gate_center_weight = 0.05

    record_obs_view_video = False
    record_back_view_video = True
    if record_obs_view_video:
        viewer: ViewerCfg = ViewerCfg(
            eye=(0.0, 0.0, 0.0),
            lookat=(1.0, 0.0, 0.0),
            env_index=0,
            origin_type="asset_root",
            asset_name="car_camera"
        )
    if record_back_view_video:
        viewer: ViewerCfg = ViewerCfg(
            eye=(-1.5, 0.0, 0.5),
            lookat=(1.0, 0.0, 0.4),
            env_index=0,
            origin_type="asset_root",
            asset_name="car"
        )


class FinishGateDriveEnv(DirectRLEnv):
    cfg: FinishGateDriveEnvCfg

    def __init__(self, cfg: FinishGateDriveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.env_spacing = self.cfg.env_spacing
        self._num_goals = self.cfg.num_goals
        
        self._throttle_dof_idx, _ = self.car.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.car.find_joints(self.cfg.steering_dof_name)
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)

    def _setup_scene(self):
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # Setup rest of the scene
        self.car = Articulation(self.cfg.car_cfg)
        self.car_camera = Camera(self.cfg.car_camera_cfg)
        self.finish_gates = []
        for i in range(self.cfg.num_goals):
            finish_gate_cfg = self.cfg.finish_gate_cfg.replace(
              prim_path=f"/World/envs/env_.*/FinishGate_{i}",
              init_state=RigidObjectCfg.InitialStateCfg(
                pos=(i / 5, 0.0, 0.0)
              )
            )
            finish_gate = RigidObject(finish_gate_cfg)
            self.finish_gates.append(finish_gate)
        self.no_pass_gates = []
        # for i in range(self.cfg.num_goals):
        #     no_pass_gate_cfg = self.cfg.no_pass_gate_cfg.replace(
        #       prim_path=f"/World/envs/env_.*/NoPassGate_{i}",
        #       init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=(i / 5 + 1, 2.0, 0.0)
        #       )
        #     )
        #     no_pass_gate = RigidObject(no_pass_gate_cfg)
        #     self.no_pass_gates.append(no_pass_gate)
        
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["car"] = self.car
        self.scene.sensors["car_camera"] = self.car_camera

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self.cfg.throttle_scale
        self._throttle_action = torch.clamp(self._throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
        self._throttle_state = self._throttle_action
        
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * self.cfg.steering_scale
        self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
        self._steering_state = self._steering_action

    def _apply_action(self) -> None:
        self.car.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.car.set_joint_position_target(self._steering_action, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        # `self.car.data.root_pos_w` is the same as `self.car.data.root_link_pos_w`
        current_gate_center_pos = self._target_positions[self.car._ALL_INDICES, self._target_index]
        self._car_to_gate_center_vec = current_gate_center_pos - self.car.data.root_pos_w[:, :2]
        self._prev_car_to_gate_center_dis = self._car_to_gate_center_dis.clone()
        self._car_to_gate_center_dis = torch.norm(self._car_to_gate_center_vec, dim=-1)

        car_heading_angle = self.car.data.heading_w  # Yaw heading of the base frame (in radians).
        car_to_gate_center_angle = torch.atan2(
            self._target_positions[self.car._ALL_INDICES, self._target_index, 1] - self.car.data.root_pos_w[:, 1],  # Δy between car and gate center
            self._target_positions[self.car._ALL_INDICES, self._target_index, 0] - self.car.data.root_pos_w[:, 0],  # Δx between car and gate center
        )  # angle (in radians) between the car-to-target direction and the +x axis
        self.car_heading_to_gate_center_angle = torch.atan2(
            torch.sin(car_to_gate_center_angle - car_heading_angle),
            torch.cos(car_to_gate_center_angle - car_heading_angle)
        )  # angle (in radians) between the car-to-target direction and car heading direction
        
        camera_data = self.car_camera.data.output['rgb'] / 255  # (num_envs, h, w, c)

        return {"policy": camera_data.clone()}
    
    def _get_rewards(self) -> torch.Tensor:
        car_to_gate_center_dis_change = self._prev_car_to_gate_center_dis - self._car_to_gate_center_dis  # positive if a car is closer to a gate center
        heading_alignment_score = torch.exp(-torch.abs(self.car_heading_to_gate_center_angle) / self.cfg.heading_alignment_coef)
        goal_reached = self._car_to_gate_center_dis < self.cfg.car_to_gate_center_dis_tolerance
        
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

        total_reward = car_to_gate_center_dis_change * self.cfg.cat_to_gate_center_dis_change_weight
        total_reward += heading_alignment_score * self.cfg.car_heading_to_gate_center_weight
        total_reward += goal_reached * self.cfg.goal_reached_weight

        if torch.any(total_reward.isnan()):
            raise ValueError("Rewards cannot be NaN")
        
        self._visualize_waypoints(target_index=self._target_index.long())

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # should return a tuple `(is_terminalted, time_out)`
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.car._ALL_INDICES
        super()._reset_idx(env_ids)

        self._reset_car(env_ids)
        self._reset_target_positions(env_ids)
        self._reset_markers_pos(env_ids)
        self._visualize_waypoints(self._markers_pos)
        self._reset_gates(env_ids, self.finish_gates)
        self._reset_gates(env_ids, self.no_pass_gates, x_offset=0.5)

        self._target_index[env_ids] = 0
        current_gate_center_pos = self._target_positions[self.car._ALL_INDICES, self._target_index]
        self._car_to_gate_center_vec = current_gate_center_pos[:, :2] - self.car.data.root_pos_w[:, :2]
        self._car_to_gate_center_dis = torch.norm(self._car_to_gate_center_vec, dim=-1)
        self._prev_car_to_gate_center_dis = self._car_to_gate_center_dis.clone()

        heading = self.car.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.car.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.car.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()

    def _reset_car(self, env_ids):
        num_envs = len(env_ids)
        default_state = self.car.data.default_root_state[env_ids]

        # default_state: position (3) & quaternion orientation (4) & linear velocities (3) & angular velocities (3)
        # quaternion: (w, x, y, z)
        car_pose = default_state[:, :7]
        car_velocities = default_state[:, 7:]
        joint_positions = self.car.data.default_joint_pos[env_ids]
        joint_velocities = self.car.data.default_joint_vel[env_ids]
        
        car_pose[:, :3] += self.scene.env_origins[env_ids]
        x_offset = - 0.5 * self.env_spacing
        car_pose[:, 0] += x_offset
        
        # If we rotate the car about the z-axis by an angle θ, then the quaternion should be `q = [cos(θ/2), sin(θ/2)k] = [cos(θ/2), 0, 0, sin(θ/2)]`
        angles = torch.pi / 4.0 * self._generate_rand_at_cente((num_envs))
        car_pose[:, 3] = torch.cos(angles / 2)  # divided by 2 as half angle is used in quaternion
        car_pose[:, 6] = torch.sin(angles / 2)  # divided by 2 as half angle is used in quaternion

        self.car.write_root_pose_to_sim(car_pose, env_ids)
        self.car.write_root_velocity_to_sim(car_velocities, env_ids)
        self.car.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

    def _reset_target_positions(self, env_ids):
        num_envs = len(env_ids)

        self._target_positions[env_ids, :, :] = 0.0  # (num_envs, num_goals, 2)

        spacing = 1 / self._num_goals  # 0.1
        target_positions = torch.arange(-0.4, 0.5, spacing, device=self.device) * self.env_spacing
        assert len(target_positions) == self._num_goals
        y_offset = self.cfg.target_y_offset_scale * self._generate_rand_at_cente((num_envs, self._num_goals))
        y_offset[:, 0] = 0  # set the first gate in the front of the car

        self._target_positions[env_ids, :, 0] = target_positions
        self._target_positions[env_ids, :, 1] = y_offset
        self._target_positions[env_ids, :, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

    def _reset_markers_pos(self, env_ids):
        self._markers_pos[env_ids, :, :] = 0.0  # (num_envs, num_goals, 3)
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]

    def _reset_gates(self, env_ids, gates, x_offset=0.0):
        for i, gate in enumerate(gates):
            default_state = gate.data.default_root_state[env_ids]  # (num_envs, 13)
            pos = default_state[:, :7]
            pos[:, :2] = self._markers_pos[env_ids, i, :2]
            pos[:, 0] += x_offset
            gate.write_root_pose_to_sim(pos, env_ids)

    def _visualize_waypoints(self, markers_pos=None, target_index=None):
        visualize_pos = None
        if markers_pos is not None:        
            visualize_pos = markers_pos.view(-1, 3)
            self.waypoints.visualize(translations=visualize_pos)

        marker_indices = None
        if target_index is not None:
            one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
            marker_indices = one_hot_encoded.view(-1).tolist()
            self.waypoints.visualize(marker_indices=marker_indices)

    def _generate_rand_at_cente(self, shape):
        return torch.rand(shape, dtype=torch.float32, device=self.device) - 0.5