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
from isaaclab.sensors import Camera, CameraCfg, save_images_to_file

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
    viewer: ViewerCfg = ViewerCfg(eye=(12, 18, 10), lookat=(12, 18, 0))
    
    # cfgs
    car_cfg: ArticulationCfg = CAR_CFG.replace(prim_path="/World/envs/env_.*/Car")
    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Car/Rigid_Bodies/Chassis/Camera_Right",
        update_period=0.1,
        height=120, # 480,
        width=160, # 640,
        spawn=None,
        data_types=["rgb"],
    )
    waypoint_cfg = WAYPOINT_CFG
    finish_gate_cfg = FINISH_GATE_CFG
    finish_gate_with_base_cfg = FINISH_GATE_WITH_BASE_CFG
    no_pass_gate_cfg = NO_PASS_GATE_CFG
    no_pass_gate_with_base_cfg = NO_PASS_GATE_WITH_BASE_CFG

    # env space
    action_space = 2  # one for throttle, and another for steering
    observation_space = [camera_cfg.height, camera_cfg.width, 3]
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
    num_goals = 10

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    course_length_coefficient = 2.5
    course_width_coefficient = 2.0
    position_tolerance = 0.15
    goal_reached_weight = 10.0
    position_progress_weight = 1.0
    heading_coefficient = 0.25
    heading_progress_weight = 0.05

    save_images = True


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
        # self.camera  # used to record video?
        self._camera = Camera(self.cfg.camera_cfg)
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
        self.scene.articulations["leatherback"] = self.car
        self.scene.sensors["camera"] = self._camera

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
        current_target_positions = self._target_positions[self.car._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.car.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.car.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.car._ALL_INDICES, self._target_index, 1] - self.car.data.root_link_pos_w[:, 1],
            self._target_positions[self.car._ALL_INDICES, self._target_index, 0] - self.car.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # obs = torch.cat(
        #     (
        #         self._position_error.unsqueeze(dim=1),
        #         torch.cos(self.target_heading_error).unsqueeze(dim=1),
        #         torch.sin(self.target_heading_error).unsqueeze(dim=1),
        #         self.car.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
        #         self.car.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
        #         self.car.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
        #         self._throttle_state[:, 0].unsqueeze(dim=1),
        #         self._steering_state[:, 0].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        
        # if torch.any(obs.isnan()):
        #     raise ValueError("Observations cannot be NAN")

        # return {"policy": obs}
        
        camera_data = self._camera.data.output['rgb'] / 255  # (num_envs, h, w, c)

        if self.cfg.save_images:
            save_images_to_file(camera_data, f"camera_data.png")

        return {"policy": camera_data.clone()}
    
    def _get_rewards(self) -> torch.Tensor:
        position_progress_rew = self._previous_position_error - self._position_error
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.cfg.heading_coefficient)
        goal_reached = self._position_error < self.cfg.position_tolerance
        
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

        total_reward = position_progress_rew * self.cfg.position_progress_weight
        total_reward += target_heading_rew * self.cfg.heading_progress_weight
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
        current_target_positions = self._target_positions[self.car._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.car.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

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
        leatherback_pose = default_state[:, :7]
        leatherback_velocities = default_state[:, 7:]
        joint_positions = self.car.data.default_joint_pos[env_ids]
        joint_velocities = self.car.data.default_joint_vel[env_ids]

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_envs), dtype=torch.float32, device=self.device) * self.cfg.course_width_coefficient

        angles = torch.pi / 6.0 * torch.rand((num_envs), dtype=torch.float32, device=self.device)
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.car.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.car.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.car.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

    def _reset_target_positions(self, env_ids):
        num_envs = len(env_ids)

        self._target_positions[env_ids, :, :] = 0.0  # (num_envs, num_goals, 2)

        spacing = 2 / self._num_goals  # 0.2
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.cfg.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_envs, self._num_goals), dtype=torch.float32, device=self.device) + self.cfg.course_length_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

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