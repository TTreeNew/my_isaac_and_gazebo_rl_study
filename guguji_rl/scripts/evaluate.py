from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from guguji_rl.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate trained PPO policy in Gazebo.')
    parser.add_argument('--config', default='configs/walk_ppo.yaml', help='配置文件路径')
    parser.add_argument('--model', required=True, help='训练好的模型路径，例如 outputs/.../final_model.zip')
    parser.add_argument('--episodes', type=int, default=None, help='可选覆盖配置中的评估轮数')
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return SCRIPT_ROOT / path


def main() -> int:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print('缺少 stable-baselines3，请先安装 requirements.txt', file=sys.stderr)
        return 1

    from guguji_rl.envs import GazeboBipedEnv

    config = load_config(resolve_input_path(args.config))
    if args.episodes is not None:
        config['evaluation']['episodes'] = args.episodes

    env = GazeboBipedEnv(config)
    model = PPO.load(resolve_input_path(args.model))

    for episode_index in range(int(config['evaluation']['episodes'])):
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        while not done and not truncated:
            action, _ = model.predict(
                observation,
                deterministic=bool(config['evaluation']['deterministic']),
            )
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
        print(f'episode={episode_index} total_reward={total_reward:.3f}')

    env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
