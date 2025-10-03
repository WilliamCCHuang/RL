from __future__ import annotations
import math
from collections.abc import Sequence

import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

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
    action_space = 2  # one is for the throttle, and another is for the steering
    # observation_space = [car_camera_cfg.height, car_camera_cfg.width, 3]  # for camera only
    observation_space = {
        "camera_img": [car_camera_cfg.height, car_camera_cfg.width, 3],
        "car_state": 8
    }
    # for multiple obs. One is for the camera, and another is for the state of the car
    # the keys must be matched with those of the returned observation in `self.`
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

    num_curriculum = 3
    gate_center_y_offset_scales = [1.0, 1.0, 2.0]
    show_waypoints = [True, False, False]
    car_to_gate_dis_targets = ['gate_center', 'gate_plane', 'gate_plane']
    car_to_gate_center_dis_tolerance = 0.25
    cat_to_gate_center_dis_change_weight = 1.0
    goal_reached_weight = 10.0
    car_heading_to_gate_center_coef = 0.25
    car_heading_to_gate_center_weight = 0.05

    save_obs_img = False
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
        self._task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._gate_center_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self._goal_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self._curriculum_idx = 0

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
        current_gate_center_pos = self._gate_center_positions[self.car._ALL_INDICES, self._goal_index]
        self._car_to_gate_center_vec = current_gate_center_pos - self.car.data.root_pos_w[:, :2]
        self._prev_car_to_gate_center_dis = self._car_to_gate_center_dis.detach()
        self._car_to_gate_center_dis = torch.norm(self._car_to_gate_center_vec, dim=-1)  # (num_envs,)

        car_heading_angle = self.car.data.heading_w  # the yaw heading of the base frame (in radians).
        car_to_gate_center_angle = torch.atan2(
            self._gate_center_positions[self.car._ALL_INDICES, self._goal_index, 1] - self.car.data.root_pos_w[:, 1],  # Δy between car and gate center
            self._gate_center_positions[self.car._ALL_INDICES, self._goal_index, 0] - self.car.data.root_pos_w[:, 0],  # Δx between car and gate center
        )  # (num_envs,)  # angle (in radians) between the car-to-target direction and the +x axis
        self._car_heading_to_gate_center_angle = torch.atan2(
            torch.sin(car_to_gate_center_angle - car_heading_angle),
            torch.cos(car_to_gate_center_angle - car_heading_angle)
        )  # (num_envs,) # angle (in radians) between the car-to-target direction and the car heading direction
        
        camera_img = self.car_camera.data.output['rgb'] / 255  # (num_envs, h, w, c)

        if self.cfg.save_obs_img:
            camera_img = camera_img.permute(0, 3, 1, 2)
            num_img = camera_img.shape[0]
            nrow = round(math.sqrt(num_img))
            camera_img = make_grid(camera_img, nrow=nrow)
            camera_img = to_pil_image(camera_img)
            camera_img.save('camera_image.png')

        car_state = torch.cat(
            [
                self.car.data.root_link_vel_w,
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ], dim=1
        )

        # Two kinds of observation are provided to the agent:
        # 1: an image recorded by the car camera
        # 2: the car's state:
        #   * root link velocity [lin_vel, ang_vel] in simulation world frame.
        #   * action applied on the throttle
        #   * action applied on the steering
        obs = {
            'policy': {
                'camera_img': camera_img.detach(),
                'car_state': car_state
            }
        }

        return obs
    
    def _get_rewards(self) -> torch.Tensor:
        car_to_gate_center_dis_change = self._prev_car_to_gate_center_dis - self._car_to_gate_center_dis  # positive if a car is closer to a gate center
        heading_alignment_score = torch.exp(-torch.abs(self._car_heading_to_gate_center_angle) / self.cfg.car_heading_to_gate_center_coef)
        goal_reached = self._car_to_gate_center_dis < self.cfg.car_to_gate_center_dis_tolerance
        
        self._goal_index = self._goal_index + goal_reached
        self._task_completed = self._goal_index > (self._num_goals - 1)
        self._goal_index = self._goal_index % self._num_goals

        total_reward = car_to_gate_center_dis_change * self.cfg.cat_to_gate_center_dis_change_weight
        total_reward += heading_alignment_score * self.cfg.car_heading_to_gate_center_weight
        total_reward += goal_reached * self.cfg.goal_reached_weight
        
        self._visualize_waypoints(target_index=self._goal_index.long())

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # should return a tuple `(is_terminalted, time_out)`
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self._task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.car._ALL_INDICES
        super()._reset_idx(env_ids)

        self._pass_curriculum(env_ids)

        self._reset_car(env_ids)
        self._reset_gate_center_positions(env_ids)
        self._reset_markers_pos(env_ids)
        self._visualize_waypoints(self._markers_pos)
        self._reset_gates(env_ids, self.finish_gates)
        self._reset_gates(env_ids, self.no_pass_gates, x_offset=0.5)

        self._goal_index[env_ids] = 0
        current_gate_center_pos = self._gate_center_positions[self.car._ALL_INDICES, self._goal_index]
        self._car_to_gate_center_vec = current_gate_center_pos[:, :2] - self.car.data.root_pos_w[:, :2]
        self._car_to_gate_center_dis = torch.norm(self._car_to_gate_center_vec, dim=-1)
        self._prev_car_to_gate_center_dis = self._car_to_gate_center_dis.clone()

    def _pass_curriculum(self, env_ids):
        if self._curriculum_idx == self.cfg.num_curriculum - 1:
            return
        
        complete_rate = ((self._goal_index[env_ids] + 1) / self._num_goals).mean()

        if complete_rate > 0.8:
            self._curriculum_idx += 1
            print(f'Enter the next lesson {self._curriculum_idx}')

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

    def _reset_gate_center_positions(self, env_ids):
        num_envs = len(env_ids)

        self._gate_center_positions[env_ids, :, :] = 0.0  # (num_envs, num_goals, 2)

        spacing = 1 / self._num_goals  # 0.1
        x_offsets = torch.arange(-0.4, 0.501, spacing, device=self.device) * self.env_spacing
        assert len(x_offsets) == self._num_goals
        gate_center_y_offset_scale = self.cfg.gate_center_y_offset_scales[self._curriculum_idx]
        y_offsets = gate_center_y_offset_scale * self._generate_rand_at_cente((num_envs, self._num_goals))
        y_offsets[:, 0] = 0  # set the first gate in the front of the car

        self._gate_center_positions[env_ids, :, 0] = x_offsets
        self._gate_center_positions[env_ids, :, 1] = y_offsets
        self._gate_center_positions[env_ids, :, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

    def _reset_markers_pos(self, env_ids):
        self._markers_pos[env_ids, :, :] = 0.0  # (num_envs, num_goals, 3)
        self._markers_pos[env_ids, :, :2] = self._gate_center_positions[env_ids]

    def _reset_gates(self, env_ids, gates, x_offset=0.0):
        for i, gate in enumerate(gates):
            default_state = gate.data.default_root_state[env_ids]  # (num_envs, 13)
            pos = default_state[:, :7]
            pos[:, :2] = self._markers_pos[env_ids, i, :2]
            pos[:, 0] += x_offset
            gate.write_root_pose_to_sim(pos, env_ids)

    def _visualize_waypoints(self, markers_pos=None, target_index=None):
        if not self.cfg.show_waypoints[self._curriculum_idx]:
            return 
        
        visualize_pos = None
        if markers_pos is not None:        
            visualize_pos = markers_pos.view(-1, 3)
            self.waypoints.visualize(translations=visualize_pos)

        marker_indices = None
        if target_index is not None:
            one_hot_encoded = torch.nn.functional.one_hot(self._goal_index.long(), num_classes=self._num_goals)
            marker_indices = one_hot_encoded.view(-1).tolist()
            self.waypoints.visualize(marker_indices=marker_indices)

    def _generate_rand_at_cente(self, shape):
        return torch.rand(shape, dtype=torch.float32, device=self.device) - 0.5