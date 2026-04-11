from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

from ..config import resolve_project_path
from ..math_utils import quaternion_xyzw_to_euler
from ..rewards import BipedRewardCalculator, RewardContext
from ..ros2_interface import GugujiRos2Interface, RobotStateSnapshot
from ..urdf_utils import JointLimit, parse_joint_limits


class GazeboBipedEnv(gym.Env):
    """把 Gazebo + ROS 2 机器人封装成 Gymnasium 环境。"""

    metadata = {'render_modes': []}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.robot_config = config['robot']
        self.ros_config = config['ros']
        self.sim_config = config['sim']
        self.task_config = config['task']
        self.training_config = config['training']
        self.configured_target_base_height = self.task_config['target_base_height']

        urdf_path = resolve_project_path(config, self.robot_config['urdf_path'])
        self.joint_limits: list[JointLimit] = parse_joint_limits(
            urdf_path,
            self.robot_config['joint_names'],
        )
        self.joint_names = [joint_limit.name for joint_limit in self.joint_limits]
        self.joint_lower = np.array([joint.lower for joint in self.joint_limits], dtype=np.float32)
        self.joint_upper = np.array([joint.upper for joint in self.joint_limits], dtype=np.float32)
        self.joint_mid = np.array([joint.midpoint for joint in self.joint_limits], dtype=np.float32)
        self.joint_half_range = np.array([joint.half_range for joint in self.joint_limits], dtype=np.float32)

        # 第一版平衡训练不适合直接把策略动作映射到整个关节极限范围，
        # 否则随机初始化策略往往一步就能把机器人打翻。
        # 这里使用“名义站姿 + 小范围残差动作”的方式，让训练从更温和的控制开始。
        nominal_joint_config = self.robot_config.get('nominal_joint_positions', {})
        self.nominal_joint_targets = np.array(
            [float(nominal_joint_config.get(joint_name, 0.0)) for joint_name in self.joint_names],
            dtype=np.float32,
        )
        self.action_scale = float(self.robot_config.get('action_scale', 1.0))
        self.action_smoothing = float(self.robot_config.get('action_smoothing', 0.0))
        self.action_smoothing = float(np.clip(self.action_smoothing, 0.0, 0.99))
        self.reset_hold_steps = int(self.sim_config.get('reset_hold_steps', 1))
        self.termination_grace_steps = int(self.task_config.get('termination_grace_steps', 0))

        self.interface = GugujiRos2Interface(
            joint_names=self.joint_names,
            command_topic_prefix=self.robot_config['command_topic_prefix'],
            joint_state_topic=self.ros_config['joint_state_topic'],
            tf_topic=self.ros_config['tf_topic'],
            clock_topic=self.ros_config['clock_topic'],
            world_control_service=self.ros_config['world_control_service'],
            model_name=self.robot_config['model_name'],
            world_name=self.sim_config['world_name'],
            model_file_path=str(urdf_path),
            spawn_x=float(self.sim_config.get('spawn_x', 0.0)),
            spawn_y=float(self.sim_config.get('spawn_y', 0.0)),
            spawn_z=float(self.sim_config.get('spawn_z', 0.35)),
            spawn_roll=float(self.sim_config.get('spawn_roll', 0.0)),
            spawn_pitch=float(self.sim_config.get('spawn_pitch', 0.0)),
            spawn_yaw=float(self.sim_config.get('spawn_yaw', 0.0)),
        )
        self.interface.wait_for_world_control_service()
        self.reward_calculator = BipedRewardCalculator(config['rewards'])

        # 动作空间采用每个关节一个归一化动作，后面再映射回真实关节角度范围。
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.joint_names),),
            dtype=np.float32,
        )
        # 观测里包含关节位置/速度、上一时刻动作，以及基座姿态和速度信息。
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.joint_names) * 3 + 5,),
            dtype=np.float32,
        )

        self.previous_action = np.zeros(len(self.joint_names), dtype=np.float32)
        self.previous_snapshot: RobotStateSnapshot | None = None
        self.current_snapshot: RobotStateSnapshot | None = None
        self.current_joint_target = self.nominal_joint_targets.copy()
        self.step_count = 0
        self.target_base_height = self.configured_target_base_height

    def _normalized_joint_position(self, joint_position: np.ndarray) -> np.ndarray:
        return (joint_position - self.joint_mid) / self.joint_half_range

    def _action_to_joint_targets(self, action: np.ndarray) -> np.ndarray:
        clipped_action = np.clip(action, -1.0, 1.0)
        # 将 [-1, 1] 的策略输出缩放成“围绕名义站姿的小残差动作”，
        # 这样 PPO 初期随机动作不会一下把关节打到很远的位置。
        desired_targets = (
            self.nominal_joint_targets
            + clipped_action * self.joint_half_range * self.action_scale
        )
        desired_targets = np.clip(desired_targets, self.joint_lower, self.joint_upper)

        if self.action_smoothing > 0.0:
            joint_targets = (
                self.action_smoothing * self.current_joint_target
                + (1.0 - self.action_smoothing) * desired_targets
            )
        else:
            joint_targets = desired_targets

        return np.clip(joint_targets, self.joint_lower, self.joint_upper).astype(np.float32)

    def _get_target_base_height(self, snapshot: RobotStateSnapshot) -> float:
        if self.target_base_height is None:
            self.target_base_height = float(snapshot.base_position[2])
        return float(self.target_base_height)

    def _build_observation(
        self,
        snapshot: RobotStateSnapshot,
        previous_snapshot: RobotStateSnapshot,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        dt = max(snapshot.sim_time - previous_snapshot.sim_time, self.sim_config['control_dt'], 1e-3)
        velocity = (snapshot.base_position - previous_snapshot.base_position) / dt
        roll, pitch, _ = quaternion_xyzw_to_euler(snapshot.base_quaternion)
        observation = np.concatenate(
            [
                self._normalized_joint_position(snapshot.joint_position),
                snapshot.joint_velocity,
                previous_action,
                np.array(
                    [
                        snapshot.base_position[2],
                        roll,
                        pitch,
                        velocity[0],
                        self.task_config['target_forward_velocity'],
                    ],
                    dtype=np.float32,
                ),
            ],
            dtype=np.float32,
        )
        return observation

    def _publish_joint_targets(self, joint_targets: np.ndarray) -> None:
        self.interface.publish_joint_targets(joint_targets)

    def _hold_nominal_pose_after_reset(self) -> None:
        """reset 后先保持几步名义站姿，减少第一拍就摔倒的概率。"""
        if self.sim_config['step_mode'] == 'service_step':
            for _ in range(max(self.reset_hold_steps, 1)):
                self._publish_joint_targets(self.current_joint_target)
                time.sleep(float(self.sim_config['action_publish_delay']))
                self.interface.step_world(int(self.sim_config['service_step_iterations']))
                time.sleep(float(self.sim_config['post_step_wait_seconds']))
        else:
            self._publish_joint_targets(self.current_joint_target)
            time.sleep(float(self.sim_config['reset_settle_seconds']))

    def _advance_simulation(self) -> None:
        if self.sim_config['step_mode'] == 'service_step':
            # service_step 更适合训练：先发动作，再精确推进固定步数。
            time.sleep(float(self.sim_config['action_publish_delay']))
            self.interface.step_world(int(self.sim_config['service_step_iterations']))
            time.sleep(float(self.sim_config['post_step_wait_seconds']))
        else:
            # realtime 更接近交互式调试，但训练时可重复性会差一些。
            time.sleep(float(self.sim_config['control_dt']))

    def _terminated(self, snapshot: RobotStateSnapshot) -> bool:
        # 刚 reset 后的前几步允许一点过渡，避免初始数值抖动把 episode 过早终止。
        if self.step_count <= self.termination_grace_steps:
            return False

        roll, pitch, _ = quaternion_xyzw_to_euler(snapshot.base_quaternion)
        if abs(roll) > float(self.task_config['max_roll_rad']):
            return True
        if abs(pitch) > float(self.task_config['max_pitch_rad']):
            return True
        if float(snapshot.base_position[2]) < float(self.task_config['min_base_height']):
            return True
        return False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        pause_after_reset = self.sim_config['step_mode'] == 'service_step'
        # 这里使用“删掉并重生模型”的硬复位，避免 Gazebo 对动态生成模型 reset 不彻底。
        self.interface.respawn_model(pause_after_reset=pause_after_reset)
        self.current_joint_target = self.nominal_joint_targets.copy()
        self._hold_nominal_pose_after_reset()

        snapshot = self.interface.wait_for_snapshot()
        # 如果 target_base_height 没有在配置里写死，就在每个 episode 开头重新取一次。
        # 这样高度奖励始终参考当前 reset 后的稳定站姿，而不会被上一个 episode 的状态污染。
        self.target_base_height = self.configured_target_base_height
        self._get_target_base_height(snapshot)
        self.current_snapshot = snapshot
        self.previous_snapshot = snapshot
        self.previous_action = np.zeros(len(self.joint_names), dtype=np.float32)
        self.current_joint_target = self.nominal_joint_targets.copy()
        self.step_count = 0

        observation = self._build_observation(snapshot, snapshot, self.previous_action)
        info = {
            'reset': True,
            'target_base_height': float(self.target_base_height),
            'target_forward_velocity': float(self.task_config['target_forward_velocity']),
        }
        return observation, info

    def step(self, action: np.ndarray):
        if self.current_snapshot is None or self.previous_snapshot is None:
            raise RuntimeError('环境尚未 reset，不能直接 step。')

        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        joint_targets = self._action_to_joint_targets(action)
        self.current_joint_target = joint_targets.copy()
        self._publish_joint_targets(joint_targets)
        self._advance_simulation()

        self.previous_snapshot = self.current_snapshot
        self.current_snapshot = self.interface.wait_for_snapshot()
        terminated = self._terminated(self.current_snapshot)
        truncated = self.step_count >= int(self.training_config['max_episode_steps'])

        reward, reward_terms = self.reward_calculator.compute(
            RewardContext(
                current=self.current_snapshot,
                previous=self.previous_snapshot,
                action=action,
                previous_action=self.previous_action,
                joint_limits=self.joint_limits,
                target_forward_velocity=float(self.task_config['target_forward_velocity']),
                target_base_height=float(self.target_base_height),
                control_dt=float(self.sim_config['control_dt']),
                terminated=terminated,
            )
        )
        # 训练时保留奖励分项，方便后面定位“为什么学不会走”。
        observation = self._build_observation(
            self.current_snapshot,
            self.previous_snapshot,
            self.previous_action,
        )
        self.previous_action = action.copy()

        info = {
            'joint_targets': joint_targets.tolist(),
            'reward_terms': reward_terms,
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.interface.close()
