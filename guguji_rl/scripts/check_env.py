from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from guguji_rl.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sanity check for the Gazebo RL environment.')
    parser.add_argument('--config', default='configs/balance_ppo.yaml', help='配置文件路径')
    parser.add_argument('--steps', type=int, default=5, help='检查时执行多少个 step')
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return SCRIPT_ROOT / path


def main() -> int:
    args = parse_args()

    try:
        from guguji_rl.envs import GazeboBipedEnv
    except ImportError:
        print('缺少训练环境依赖，请先进入 guguji_rl 目录安装 requirements.txt', file=sys.stderr)
        return 1

    config = load_config(resolve_input_path(args.config))
    env = GazeboBipedEnv(config)

    observation, info = env.reset()
    print('reset ok')
    print(f'observation shape: {observation.shape}')
    print(f'target_base_height: {info["target_base_height"]:.4f}')

    for step_index in range(args.steps):
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_terms = info['reward_terms']
        print(
            f'step={step_index} reward={reward:.3f} '
            f'vx={reward_terms["forward_velocity"]:.3f} '
            f'base_z={reward_terms["base_height"]:.3f}'
        )
        if terminated or truncated:
            print(f'episode finished early: terminated={terminated} truncated={truncated}')
            break

    env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
