from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import torch

from guguji_rl.config import load_config, resolve_project_path, save_yaml
from guguji_rl.evaluation import evaluate_forward_progress, print_forward_progress_summary


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
    parser.add_argument(
        '--init-model',
        default=None,
        help='可选指定一个已有 PPO 模型，用于继续训练或做课程学习初始化',
    )
    parser.add_argument(
        '--skip-auto-eval',
        action='store_true',
        help='训练完成后跳过自动前进评估',
    )
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return SCRIPT_ROOT / path


def maybe_override_policy_log_std(model: object, initial_log_std: float | None) -> None:
    """可选地缩小 PPO 的初始探索方差，适合课程学习后的精修阶段。"""
    if initial_log_std is None:
        return
    policy = getattr(model, 'policy', None)
    if policy is None or not hasattr(policy, 'log_std'):
        raise RuntimeError('当前策略对象不支持直接设置 log_std。')

    # 这里直接把每个动作维度的对数标准差统一改成同一个值，
    # 方便在“已有步态基础上继续训练”时降低探索噪声，减少无意义的乱踢。
    policy.log_std.data.fill_(float(initial_log_std))
    print(f'已将策略初始 log_std 设为: {float(initial_log_std):.3f}')


def sanitize_stage_name(stage_name: str) -> str:
    sanitized = ''.join(
        character if character.isalnum() or character in {'-', '_'} else '_'
        for character in stage_name.strip()
    )
    return sanitized.strip('_') or 'stage'


def build_curriculum_stage_configs(config: dict[str, Any]) -> list[tuple[str | None, dict[str, Any]]]:
    """把课程学习阶段展开成一组可直接训练的独立配置。"""
    raw_stages = config['training'].get('curriculum_stages') or []
    if not raw_stages:
        single_stage_config = copy.deepcopy(config)
        single_stage_config['training'].pop('curriculum_stages', None)
        return [(None, single_stage_config)]

    stage_configs: list[tuple[str | None, dict[str, Any]]] = []
    for stage_index, raw_stage in enumerate(raw_stages, start=1):
        if not isinstance(raw_stage, dict):
            raise RuntimeError('training.curriculum_stages 里的每个阶段都必须是字典。')

        stage_config = copy.deepcopy(config)
        stage_config['training'].pop('curriculum_stages', None)

        raw_name = str(raw_stage.get('name') or f'stage_{stage_index}')
        stage_name = f'{stage_index:02d}_{sanitize_stage_name(raw_name)}'

        # 课程阶段目前主要控制“目标前进速度 + 本阶段训练步数 + 探索方差”。
        # 这样 walking 阶段就能从慢到快逐段抬升，而不用一次把目标速度顶太高。
        if 'target_forward_velocity' in raw_stage:
            stage_config['task']['target_forward_velocity'] = float(raw_stage['target_forward_velocity'])
        if 'total_timesteps' in raw_stage:
            stage_config['training']['total_timesteps'] = int(raw_stage['total_timesteps'])
        if 'initial_log_std' in raw_stage:
            stage_config['training']['initial_log_std'] = float(raw_stage['initial_log_std'])

        stage_config['experiment']['name'] = f"{config['experiment']['name']}_{stage_name}"
        stage_configs.append((stage_name, stage_config))

    return stage_configs


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
    if args.init_model is not None:
        config['training']['init_model_path'] = str(resolve_input_path(args.init_model))

    # 这里统一解析训练设备，方便你只改 YAML 就切换 CPU / GPU。
    config['training']['device'] = resolve_device(config['training']['device'])

    output_root = resolve_project_path(config, config['training']['output_root'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f"{config['experiment']['name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # 保存一份展开后的配置，便于后面复现实验。
    save_yaml(config, run_dir / 'resolved_config.yaml')
    stage_configs = build_curriculum_stage_configs(config)

    print(f"训练设备: {config['training']['device']}")
    print(f"输出目录: {run_dir}")
    model = None
    final_model_path = run_dir / 'final_model'
    final_stage_config = config

    for stage_index, (stage_name, stage_config) in enumerate(stage_configs, start=1):
        stage_dir = run_dir if stage_name is None else run_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        # 每个阶段都单独保存一份实际生效的配置，后面你回看实验会很方便。
        save_yaml(stage_config, stage_dir / 'resolved_config.yaml')

        if stage_name is not None:
            print(
                f'开始课程阶段 {stage_index}/{len(stage_configs)}: {stage_name} '
                f'(target_forward_velocity={stage_config["task"]["target_forward_velocity"]:.2f}, '
                f'timesteps={int(stage_config["training"]["total_timesteps"])})'
            )

        env = Monitor(GazeboBipedEnv(stage_config))
        checkpoint_callback = CheckpointCallback(
            save_freq=max(int(stage_config['training']['checkpoint_freq']), 1),
            save_path=str(stage_dir / 'checkpoints'),
            name_prefix='guguji_ppo',
        )

        try:
            if model is None:
                policy_kwargs = {
                    'net_arch': list(stage_config['training']['policy_net_arch']),
                }
                # 先用 MLP + PPO 跑通训练闭环，后面你可以再逐步增大网络规模。
                model = PPO(
                    policy='MlpPolicy',
                    env=env,
                    verbose=1,
                    seed=int(stage_config['training']['seed']),
                    learning_rate=float(stage_config['training']['learning_rate']),
                    n_steps=int(stage_config['training']['n_steps']),
                    batch_size=int(stage_config['training']['batch_size']),
                    gamma=float(stage_config['training']['gamma']),
                    gae_lambda=float(stage_config['training']['gae_lambda']),
                    clip_range=float(stage_config['training']['clip_range']),
                    ent_coef=float(stage_config['training']['ent_coef']),
                    vf_coef=float(stage_config['training']['vf_coef']),
                    device=stage_config['training']['device'],
                    tensorboard_log=str(run_dir / 'tensorboard'),
                    policy_kwargs=policy_kwargs,
                )

                init_model_path = stage_config['training'].get('init_model_path')
                if init_model_path:
                    resolved_init_model_path = resolve_input_path(str(init_model_path))
                    # 这里不是直接 load 整个 PPO 对象，而是把旧模型参数灌入新模型。
                    # 好处是：我们仍然使用当前配置文件里的超参数，只复用之前学到的策略权重。
                    model.set_parameters(
                        str(resolved_init_model_path),
                        exact_match=False,
                        device=stage_config['training']['device'],
                    )
                    print(f"已加载课程初始化模型: {resolved_init_model_path}")
            else:
                model.set_env(env)

            maybe_override_policy_log_std(model, stage_config['training'].get('initial_log_std'))
            model.learn(
                total_timesteps=int(stage_config['training']['total_timesteps']),
                callback=checkpoint_callback,
                progress_bar=True,
                reset_num_timesteps=(stage_index == 1),
            )

            stage_model_path = stage_dir / 'final_model'
            model.save(stage_model_path)
            final_model_path = stage_model_path
            final_stage_config = stage_config

            if stage_name is not None:
                print(f'课程阶段完成，模型已保存到: {stage_model_path.with_suffix(".zip")}')
        finally:
            env.close()

    if final_model_path != run_dir / 'final_model' and model is not None:
        # 在课程学习模式下，额外在 run 根目录保存一份最终模型，方便统一引用。
        model.save(run_dir / 'final_model')
        final_model_path = run_dir / 'final_model'

    print(f'训练完成，模型已保存到: {run_dir / "final_model.zip"}')

    evaluation_config = final_stage_config['evaluation']
    if bool(evaluation_config.get('auto_forward_progress', True)) and not args.skip_auto_eval:
        try:
            # 每轮训练结束后自动做一次前进评估，方便你快速看 delta_x / mean_vx。
            summary = evaluate_forward_progress(
                config=final_stage_config,
                model_path=final_model_path,
                episodes=int(evaluation_config['forward_progress_episodes']),
                max_steps=int(evaluation_config['forward_progress_max_steps']),
                deterministic=bool(evaluation_config['forward_progress_deterministic']),
            )
            print_forward_progress_summary(summary)
        except Exception as error:
            print(f'自动前进评估失败: {error}', file=sys.stderr)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
