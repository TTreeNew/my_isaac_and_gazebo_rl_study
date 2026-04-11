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

        action_rate_penalty = float(np.mean(np.square(context.action - context.previous_action)))

        joint_limit_penalty = 0.0
        for index, joint_limit in enumerate(context.joint_limits):
            normalized = abs((context.current.joint_position[index] - joint_limit.midpoint) / joint_limit.half_range)
            joint_limit_penalty += max(normalized - 0.9, 0.0)
        joint_limit_penalty /= max(len(context.joint_limits), 1)

        reward_terms = {
            'alive_bonus': float(self.reward_config['alive_bonus']),
            'velocity_tracking': float(self.reward_config['velocity_tracking_scale']) * float(velocity_tracking),
            'upright': float(self.reward_config['upright_scale']) * float(upright_reward),
            'height': float(self.reward_config['height_scale']) * float(height_reward),
            'action_rate_penalty': -float(self.reward_config['action_rate_penalty_scale']) * action_rate_penalty,
            'joint_limit_penalty': -float(self.reward_config['joint_limit_penalty_scale']) * joint_limit_penalty,
            'lateral_velocity_penalty': -float(self.reward_config['lateral_velocity_penalty_scale']) * abs(lateral_velocity),
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
