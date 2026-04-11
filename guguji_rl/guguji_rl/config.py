from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    'experiment': {
        'name': 'guguji_rl_experiment',
    },
    'robot': {
        'model_name': 'guguji',
        'urdf_path': 'guguji_ros2_ws/src/guguji_ros2/urdf/guguji.urdf',
        'joint_names': [],
        'command_topic_prefix': '/guguji/command',
    },
    'ros': {
        'joint_state_topic': '/joint_states',
        'tf_topic': '/tf',
        'clock_topic': '/clock',
        'world_control_service': '/world/default/control',
    },
    'sim': {
        'world_name': 'default',
        'step_mode': 'realtime',
        'control_dt': 0.05,
        'service_step_iterations': 50,
        'reset_settle_seconds': 1.0,
        'action_publish_delay': 0.01,
        'post_step_wait_seconds': 0.01,
        'spawn_x': 0.0,
        'spawn_y': 0.0,
        'spawn_z': 0.35,
        'spawn_roll': 0.0,
        'spawn_pitch': 0.0,
        'spawn_yaw': 0.0,
    },
    'task': {
        'target_forward_velocity': 0.0,
        'target_base_height': None,
        'max_roll_rad': 0.6,
        'max_pitch_rad': 0.6,
        'min_base_height': 0.12,
    },
    'rewards': {
        'alive_bonus': 1.0,
        'velocity_tracking_scale': 1.0,
        'velocity_tracking_sigma': 0.3,
        'forward_progress_scale': 0.0,
        'hip_alternation_scale': 0.0,
        'hip_target_separation': 0.3,
        'hip_antiphase_sigma': 0.2,
        'knee_flexion_scale': 0.0,
        'knee_target': 0.2,
        'knee_flexion_sigma': 0.12,
        'upright_scale': 1.0,
        'height_scale': 1.0,
        'action_rate_penalty_scale': 0.02,
        'joint_limit_penalty_scale': 0.02,
        'lateral_velocity_penalty_scale': 0.05,
        'backward_velocity_penalty_scale': 0.0,
        'stall_penalty_scale': 0.0,
        'stall_velocity_threshold': 0.0,
        'fall_penalty': -10.0,
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 200000,
        'max_episode_steps': 400,
        'seed': 42,
        'device': 'auto',
        'init_model_path': None,
        'learning_rate': 3e-4,
        'n_steps': 1024,
        'batch_size': 256,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'policy_net_arch': [256, 256],
        'checkpoint_freq': 20000,
        'output_root': 'guguji_rl/outputs',
    },
    'evaluation': {
        'episodes': 3,
        'deterministic': True,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open('r', encoding='utf-8') as file:
        user_config = yaml.safe_load(file) or {}

    config = _deep_update(copy.deepcopy(DEFAULT_CONFIG), user_config)
    config['meta'] = {
        'config_path': str(config_path),
        'project_root': str(config_path.parents[2]),
        'rl_root': str(config_path.parents[1]),
    }
    return config


def resolve_project_path(config: dict[str, Any], relative_path: str | Path) -> Path:
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        return relative_path
    return Path(config['meta']['project_root']) / relative_path


def save_yaml(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
