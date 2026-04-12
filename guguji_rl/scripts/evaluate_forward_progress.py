from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from guguji_rl.config import load_config
from guguji_rl.evaluation import (
    evaluate_forward_progress,
    print_forward_progress_summary,
    resolve_input_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate forward progress metrics for a trained PPO policy.')
    parser.add_argument('--config', default='configs/walk_ppo.yaml', help='配置文件路径')
    parser.add_argument('--model', required=True, help='训练好的模型路径，例如 outputs/.../final_model.zip')
    parser.add_argument('--episodes', type=int, default=None, help='评估 episode 数量，默认读取配置文件')
    parser.add_argument('--max-steps', type=int, default=None, help='每个 episode 最多评估多少步，默认读取配置文件')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='是否强制使用确定性动作，默认读取配置文件',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(resolve_input_path(SCRIPT_ROOT, args.config))
    evaluation_config = config['evaluation']
    episodes = args.episodes if args.episodes is not None else int(evaluation_config['forward_progress_episodes'])
    max_steps = args.max_steps if args.max_steps is not None else int(evaluation_config['forward_progress_max_steps'])
    deterministic = args.deterministic or bool(evaluation_config['forward_progress_deterministic'])

    summary = evaluate_forward_progress(
        config=config,
        model_path=resolve_input_path(SCRIPT_ROOT, args.model),
        episodes=episodes,
        max_steps=max_steps,
        deterministic=deterministic,
    )
    print_forward_progress_summary(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
