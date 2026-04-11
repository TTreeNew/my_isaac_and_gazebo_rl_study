from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import quaternion_xyzw_to_euler
from .ros2_interface import RobotStateSnapshot
from .urdf_utils import JointLimit


@dataclass
class RewardContext:
    current: RobotStateSnapshot
    previous: RobotStateSnapshot
    action: np.ndarray
    previous_action: np.ndarray
    joint_limits: list[JointLimit]
    target_forward_velocity: float
    target_base_height: float
    control_dt: float
    terminated: bool


class BipedRewardCalculator:
    def __init__(self, reward_config: dict) -> None:
        self.reward_config = reward_config

    def compute(self, context: RewardContext) -> tuple[float, dict[str, float]]:
        dt = max(context.current.sim_time - context.previous.sim_time, context.control_dt, 1e-3)
        delta_position = context.current.base_position - context.previous.base_position
        forward_velocity = float(delta_position[0] / dt)
        lateral_velocity = float(delta_position[1] / dt)

        roll, pitch, _ = quaternion_xyzw_to_euler(context.current.base_quaternion)
        upright_reward = np.exp(-4.0 * (roll * roll + pitch * pitch))
        height_error = context.current.base_position[2] - context.target_base_height
        height_reward = np.exp(-8.0 * height_error * height_error)

        sigma = max(float(self.reward_config['velocity_tracking_sigma']), 1e-6)
        velocity_error = forward_velocity - context.target_forward_velocity
        velocity_tracking = np.exp(-(velocity_error * velocity_error) / (2.0 * sigma * sigma))
        # 仅奖励“真正向前”的速度，避免策略通过左右晃动或后退来钻空子。
        positive_forward_velocity = max(forward_velocity, 0.0)
        backward_velocity = max(-forward_velocity, 0.0)

        # 如果前向速度长期太低，就给一个停滞惩罚，逼着策略去迈开步子。
        stall_velocity_threshold = float(self.reward_config.get('stall_velocity_threshold', 0.0))
        stall_penalty = max(stall_velocity_threshold - positive_forward_velocity, 0.0)

        action_rate_penalty = float(np.mean(np.square(context.action - context.previous_action)))

        joint_limit_penalty = 0.0
        for index, joint_limit in enumerate(context.joint_limits):
            normalized = abs((context.current.joint_position[index] - joint_limit.midpoint) / joint_limit.half_range)
            joint_limit_penalty += max(normalized - 0.9, 0.0)
        joint_limit_penalty /= max(len(context.joint_limits), 1)

        joint_name_to_index = {
            joint_limit.name: index
            for index, joint_limit in enumerate(context.joint_limits)
        }
        hip_alternation_reward = 0.0
        knee_flexion_reward = 0.0

        left_hip_index = joint_name_to_index.get('left_hip_pitch_joint')
        right_hip_index = joint_name_to_index.get('right_hip_pitch_joint')
        if left_hip_index is not None and right_hip_index is not None:
            left_hip = float(context.current.joint_position[left_hip_index])
            right_hip = float(context.current.joint_position[right_hip_index])
            hip_separation = abs(left_hip - right_hip)
            hip_target_separation = max(float(self.reward_config.get('hip_target_separation', 0.3)), 1e-6)
            hip_antiphase_sigma = max(float(self.reward_config.get('hip_antiphase_sigma', 0.2)), 1e-6)
            # 鼓励左右髋关节朝相反方向摆动，并且摆动幅度不要太小。
            hip_separation_reward = min(hip_separation / hip_target_separation, 1.0)
            hip_antiphase_reward = np.exp(-((left_hip + right_hip) ** 2) / (2.0 * hip_antiphase_sigma ** 2))
            hip_alternation_reward = float(hip_separation_reward * hip_antiphase_reward)

        left_knee_index = joint_name_to_index.get('left_knee_pitch_joint')
        right_knee_index = joint_name_to_index.get('right_knee_pitch_joint')
        if left_knee_index is not None and right_knee_index is not None:
            left_knee = abs(float(context.current.joint_position[left_knee_index]))
            right_knee = abs(float(context.current.joint_position[right_knee_index]))
            knee_target = float(self.reward_config.get('knee_target', 0.2))
            knee_sigma = max(float(self.reward_config.get('knee_flexion_sigma', 0.12)), 1e-6)
            average_knee_flexion = 0.5 * (left_knee + right_knee)
            # 鼓励膝关节保持适度弯曲，帮助策略学会抬腿而不是整条腿僵直拖行。
            knee_flexion_reward = float(
                np.exp(-((average_knee_flexion - knee_target) ** 2) / (2.0 * knee_sigma ** 2))
            )

        reward_terms = {
            'alive_bonus': float(self.reward_config['alive_bonus']),
            'velocity_tracking': float(self.reward_config['velocity_tracking_scale']) * float(velocity_tracking),
            'forward_progress': float(self.reward_config.get('forward_progress_scale', 0.0)) * positive_forward_velocity,
            'hip_alternation': float(self.reward_config.get('hip_alternation_scale', 0.0)) * hip_alternation_reward,
            'knee_flexion': float(self.reward_config.get('knee_flexion_scale', 0.0)) * knee_flexion_reward,
            'upright': float(self.reward_config['upright_scale']) * float(upright_reward),
            'height': float(self.reward_config['height_scale']) * float(height_reward),
            'action_rate_penalty': -float(self.reward_config['action_rate_penalty_scale']) * action_rate_penalty,
            'joint_limit_penalty': -float(self.reward_config['joint_limit_penalty_scale']) * joint_limit_penalty,
            'lateral_velocity_penalty': -float(self.reward_config['lateral_velocity_penalty_scale']) * abs(lateral_velocity),
            'backward_velocity_penalty': -float(self.reward_config.get('backward_velocity_penalty_scale', 0.0)) * backward_velocity,
            'stall_penalty': -float(self.reward_config.get('stall_penalty_scale', 0.0)) * stall_penalty,
        }

        total_reward = sum(reward_terms.values())
        if context.terminated:
            reward_terms['fall_penalty'] = float(self.reward_config['fall_penalty'])
            total_reward += reward_terms['fall_penalty']
        else:
            reward_terms['fall_penalty'] = 0.0

        reward_terms['forward_velocity'] = forward_velocity
        reward_terms['roll'] = float(roll)
        reward_terms['pitch'] = float(pitch)
        reward_terms['base_height'] = float(context.current.base_position[2])
        reward_terms['total_reward'] = float(total_reward)
        return float(total_reward), reward_terms
