from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import torch

from guguji_rl.config import load_config, resolve_project_path, save_yaml


def resolve_device(device_name: str) -> str:
    if device_name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('配置要求使用 CUDA，但当前 torch 检测不到可用 GPU。')
    return device_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train PPO policy for guguji biped robot.')
    parser.add_argument(
        '--config',
        default='configs/balance_ppo.yaml',
        help='训练配置文件路径，默认使用 balance_ppo.yaml',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='可选覆盖配置文件中的设备设置，例如 cpu / cuda / auto',
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=None,
        help='可选覆盖配置文件中的 total_timesteps',
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
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print(
            '缺少 stable-baselines3，请先进入 guguji_rl 目录安装依赖: '
            'pip install -r requirements.txt',
            file=sys.stderr,
        )
        return 1

    from guguji_rl.envs import GazeboBipedEnv

    config = load_config(resolve_input_path(args.config))
    if args.device is not None:
        config['training']['device'] = args.device
    if args.total_timesteps is not None:
        config['training']['total_timesteps'] = args.total_timesteps

    # 这里统一解析训练设备，方便你只改 YAML 就切换 CPU / GPU。
    config['training']['device'] = resolve_device(config['training']['device'])

    output_root = resolve_project_path(config, config['training']['output_root'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f"{config['experiment']['name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # 保存一份展开后的配置，便于后面复现实验。
    save_yaml(config, run_dir / 'resolved_config.yaml')

    env = Monitor(GazeboBipedEnv(config))
    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(config['training']['checkpoint_freq']), 1),
        save_path=str(run_dir / 'checkpoints'),
        name_prefix='guguji_ppo',
    )

    policy_kwargs = {
        'net_arch': list(config['training']['policy_net_arch']),
    }

    print(f"训练设备: {config['training']['device']}")
    print(f"输出目录: {run_dir}")

    # 先用 MLP + PPO 跑通训练闭环，后面你可以再逐步增大网络规模。
    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        seed=int(config['training']['seed']),
        learning_rate=float(config['training']['learning_rate']),
        n_steps=int(config['training']['n_steps']),
        batch_size=int(config['training']['batch_size']),
        gamma=float(config['training']['gamma']),
        gae_lambda=float(config['training']['gae_lambda']),
        clip_range=float(config['training']['clip_range']),
        ent_coef=float(config['training']['ent_coef']),
        vf_coef=float(config['training']['vf_coef']),
        device=config['training']['device'],
        tensorboard_log=str(run_dir / 'tensorboard'),
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=int(config['training']['total_timesteps']),
        callback=checkpoint_callback,
        progress_bar=True,
    )
    model.save(run_dir / 'final_model')
    env.close()
    print(f'训练完成，模型已保存到: {run_dir / "final_model.zip"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
