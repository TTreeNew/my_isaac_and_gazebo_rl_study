from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ForwardEpisodeMetrics:
    episode_index: int
    steps: int
    terminated: bool
    truncated: bool
    total_reward: float
    start_x: float
    end_x: float
    delta_x: float
    mean_vx: float
    start_z: float
    end_z: float


@dataclass
class ForwardProgressSummary:
    model_path: str
    episodes: list[ForwardEpisodeMetrics]

    @property
    def mean_delta_x(self) -> float:
        return float(np.mean([episode.delta_x for episode in self.episodes])) if self.episodes else 0.0

    @property
    def mean_mean_vx(self) -> float:
        return float(np.mean([episode.mean_vx for episode in self.episodes])) if self.episodes else 0.0

    @property
    def mean_total_reward(self) -> float:
        return float(np.mean([episode.total_reward for episode in self.episodes])) if self.episodes else 0.0


def resolve_input_path(script_root: Path, path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return script_root / path


def resolve_sb3_model_path(model_path: str | Path) -> str:
    """兼容 stable-baselines3 对 .zip 扩展名的处理。"""
    path = Path(model_path)
    # SB3 的 PPO.load("foo") 会自动补成 foo.zip。
    # 如果这里传入已经带 .zip 的 Path，某些情况下会再拼一次 .zip，导致找不到文件。
    if path.suffix == '.zip':
        return str(path.with_suffix(''))
    return str(path)


def evaluate_forward_progress(
    *,
    config: dict[str, Any],
    model_path: str | Path,
    episodes: int,
    max_steps: int,
    deterministic: bool,
) -> ForwardProgressSummary:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise RuntimeError('缺少 stable-baselines3，无法执行策略评估。') from error

    from guguji_rl.envs import GazeboBipedEnv

    env = GazeboBipedEnv(config)
    summary_episodes: list[ForwardEpisodeMetrics] = []

    try:
        model = PPO.load(resolve_sb3_model_path(model_path))

        for episode_index in range(max(int(episodes), 1)):
            observation, _ = env.reset()
            start_x = float(env.current_snapshot.base_position[0])
            start_z = float(env.current_snapshot.base_position[2])
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not terminated and not truncated and steps < int(max_steps):
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1

            end_x = float(env.current_snapshot.base_position[0])
            end_z = float(env.current_snapshot.base_position[2])
            delta_x = end_x - start_x
            mean_vx = delta_x / max(steps * float(config['sim']['control_dt']), 1e-6)

            summary_episodes.append(
                ForwardEpisodeMetrics(
                    episode_index=episode_index,
                    steps=steps,
                    terminated=terminated,
                    truncated=truncated,
                    total_reward=total_reward,
                    start_x=start_x,
                    end_x=end_x,
                    delta_x=delta_x,
                    mean_vx=mean_vx,
                    start_z=start_z,
                    end_z=end_z,
                )
            )
    finally:
        env.close()

    return ForwardProgressSummary(
        model_path=str(model_path),
        episodes=summary_episodes,
    )


def print_forward_progress_summary(summary: ForwardProgressSummary) -> None:
    print('前进评估结果:')
    print(f'  模型: {summary.model_path}')
    for episode in summary.episodes:
        print(
            f'  episode={episode.episode_index} '
            f'steps={episode.steps} '
            f'delta_x={episode.delta_x:.3f} '
            f'mean_vx={episode.mean_vx:.3f} '
            f'total_reward={episode.total_reward:.2f} '
            f'terminated={episode.terminated} '
            f'truncated={episode.truncated}'
        )
    print(
        '  汇总: '
        f'mean_delta_x={summary.mean_delta_x:.3f} '
        f'mean_mean_vx={summary.mean_mean_vx:.3f} '
        f'mean_total_reward={summary.mean_total_reward:.2f}'
    )
