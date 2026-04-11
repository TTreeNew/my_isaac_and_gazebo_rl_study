from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from guguji_rl.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a trained PPO policy online in Gazebo / ROS 2.')
    parser.add_argument('--config', default='configs/walk_ppo.yaml', help='配置文件路径')
    parser.add_argument('--model', required=True, help='训练好的模型路径，例如 outputs/.../final_model.zip')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='是否强制使用确定性动作，适合策略回放和部署调试',
    )
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=0,
        help='最多运行多少个 episode，0 表示一直循环运行',
    )
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

    env = GazeboBipedEnv(config)
    model = PPO.load(resolve_input_path(args.model))

    episode_index = 0
    try:
        while args.max_episodes <= 0 or episode_index < args.max_episodes:
            # 这里直接复用训练环境，方便策略回放和后续在线部署共用同一套观测/动作逻辑。
            observation, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0

            while not terminated and not truncated:
                action, _ = model.predict(
                    observation,
                    deterministic=args.deterministic or bool(config['evaluation']['deterministic']),
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            print(
                f'episode={episode_index} '
                f'total_reward={total_reward:.3f} '
                f'terminated={terminated} truncated={truncated}'
            )
            episode_index += 1
    except KeyboardInterrupt:
        print('收到 Ctrl+C，停止在线策略运行。')
    finally:
        env.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
