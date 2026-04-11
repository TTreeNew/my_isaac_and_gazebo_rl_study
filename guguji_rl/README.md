# guguji_rl

`guguji_rl` 是 `guguji_simulation` 仓库里的强化学习训练工程。

## 为什么单独放在仓库根目录

我把训练代码放在 `guguji_ros2_ws` 外面，原因是：

- `guguji_ros2_ws/src/guguji_ros2` 继续专注于机器人描述、Gazebo 启动和 ROS 2 接口
- `guguji_rl` 专注于 Python 强化学习依赖，例如 `stable-baselines3`、`gymnasium`、`torch`
- 这样后续你更容易分别调试“机器人仿真问题”和“训练算法问题”

## 当前实现内容

- `configs/`：训练配置，支持 CPU / GPU 切换
- `guguji_rl/ros2_interface.py`：ROS 2 与 Gazebo 的训练接口
- `guguji_rl/envs/gazebo_biped_env.py`：Gymnasium 环境封装
- `guguji_rl/rewards.py`：奖励函数
- `scripts/train.py`：PPO 训练入口
- `scripts/evaluate.py`：加载训练好的策略并在 Gazebo 中运行
- `scripts/run_policy.py`：把训练好的策略作为在线控制程序持续运行
- `scripts/check_env.py`：训练前自检脚本

## 建议工作流

1. 先启动 Gazebo 仿真
2. 再运行 `check_env.py` 检查 ROS 2 / Gazebo 训练接口
3. 先跑 `balance_ppo.yaml` 训练站立平衡
4. 再跑 `walk_ppo.yaml` 训练前进
5. 训练出模型后，先用 `evaluate.py` 做回放，再用 `run_policy.py` 在 ROS 2 系统中持续推理控制

## 安装依赖

建议在独立虚拟环境中安装：

```bash
cd /home/corvin/Project/guguji_simulation/guguji_rl
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果要用 GPU，请先确认 `torch.cuda.is_available()` 为 `True`，然后在配置文件里把 `training.device` 设为 `cuda` 或 `auto`。

## 训练时推荐的 Gazebo 启动方式

如果你准备正式开始训练，建议这样启动 Gazebo：

```bash
source /opt/ros/humble/setup.bash
cd /home/corvin/Project/guguji_simulation/guguji_ros2_ws
source install/setup.bash
ros2 launch guguji_ros2 gazebo.launch.py gui:=false pause:=true
```

这样训练环境可以通过 `/world/default/control` 精确推进仿真步数，更适合强化学习。

## 在线运行训练好的策略

```bash
cd /home/corvin/Project/guguji_simulation/guguji_rl
source .venv/bin/activate
python3 scripts/run_policy.py \
  --config configs/walk_ppo.yaml \
  --model outputs/<你的实验目录>/final_model.zip \
  --deterministic
```
