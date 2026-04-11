# guguji 双足机器人强化学习训练指南

这份文档面向“第一次做双足机器人强化学习”的场景，目标是帮助你把当前已经能在 ROS 2 Humble + Gazebo Fortress 中运行的机器人，逐步推进到“可以用强化学习训练行走策略，并在 ROS 2 系统中在线运行策略”。

## 1. 总体目标

你的完整目标可以拆成四层：

1. 仿真层：
   机器人能在 Gazebo Fortress 中稳定启动，并提供关节命令、关节状态、TF、世界 reset 接口。
2. 环境层：
   把 Gazebo + ROS 2 封装成 Gymnasium 环境，形成 `reset()` / `step(action)` / `reward` / `done` 这套强化学习接口。
3. 训练层：
   用 PPO 等算法训练策略，让机器人先学会站稳，再学会前进。
4. 部署层：
   训练好的策略作为一个在线推理程序，在 ROS 2 系统里读取状态并发布关节命令。

## 2. 目录应该怎么放

我建议采用你现在仓库里的这种分层方式：

- `guguji_ros2_ws/src/guguji_ros2`
  负责机器人模型、Gazebo 启动、ROS 2 话题和仿真接口。
- `guguji_rl`
  负责强化学习训练代码、算法依赖、配置文件、训练输出。

这样做的好处：

- ROS 2 包保持干净，不会混进大量训练依赖。
- 强化学习代码可以单独建虚拟环境。
- 你以后调试时可以快速区分“仿真问题”和“算法问题”。

## 3. 当前已经帮你准备好的能力

当前代码已经具备下面这些训练所需的基础接口：

- `/joint_states`
  Gazebo 真实关节状态
- `/tf`
  Gazebo 真实模型 / 连杆位姿
- `/clock`
  仿真时钟
- `/guguji/command/<joint_name>`
  每个关节的位置命令接口
- `/world/default/control`
  Gazebo 世界控制服务，可用于 reset / pause / step

这些接口已经足够构成一个最小的强化学习训练环境。

## 4. 训练流程建议

对于双足机器人，不建议一上来就直接训练“向前走”。更稳妥的路线是课程式训练。

### 阶段 A：接口验证

目标：

- 确认环境能 reset
- 确认发关节命令后 `/joint_states` 会变化
- 确认 `/tf` 可以读到机器人底座位姿

你可以先运行：

```bash
source /opt/ros/humble/setup.bash
cd /home/corvin/Project/guguji_simulation/guguji_ros2_ws
source install/setup.bash
ros2 launch guguji_ros2 gazebo.launch.py gui:=false
```

另开一个终端执行：

```bash
cd /home/corvin/Project/guguji_simulation/guguji_rl
source .venv/bin/activate
python3 scripts/check_env.py --config configs/balance_ppo.yaml
```

如果你准备正式训练，建议把 Gazebo 改成下面这种启动方式：

```bash
ros2 launch guguji_ros2 gazebo.launch.py gui:=false pause:=true
```

这样训练程序可以通过世界控制服务精确推进仿真步数，训练会更稳定。

### 阶段 B：站立平衡

目标：

- 让机器人学会不摔倒
- 把姿态稳定在较小的 `roll / pitch`
- 尽量保持基座高度稳定

建议先用：

- `configs/balance_ppo.yaml`
- `task.target_forward_velocity = 0.0`

这个阶段的关键不是“走”，而是“活着且稳定”。

### 阶段 C：原地摆腿 / 重心转移

目标：

- 在不摔倒的前提下学会左右腿配合
- 逐渐形成可重复的步态雏形

这个阶段可以继续沿用 balance 配置，但逐步：

- 增大关节可动范围
- 减小对动作变化的惩罚
- 允许更积极的姿态变化

### 阶段 D：前进训练

目标：

- 跟踪目标前进速度
- 尽量减少横向漂移
- 保持基座稳定

建议从较低目标速度开始，例如：

- `0.10 m/s`
- `0.15 m/s`
- `0.25 m/s`

不要一开始就追求很快，否则双足很容易学成“扑倒式前冲”。

## 5. 奖励函数应该怎么理解

当前代码中已经实现了一套适合入门调试的奖励结构：

- `alive_bonus`
  只要没倒就给基础奖励
- `velocity_tracking`
  鼓励前进速度接近目标值
- `upright`
  鼓励机器人保持正立
- `height`
  鼓励基座高度不要塌陷
- `action_rate_penalty`
  惩罚动作变化过快
- `joint_limit_penalty`
  惩罚关节长期打到极限
- `lateral_velocity_penalty`
  惩罚横向乱漂
- `fall_penalty`
  一旦倒下，给予额外负奖励

对于第一次做双足 RL，建议你先只调这几类奖励，不要一下把奖励做得太复杂。

## 6. 训练算法为什么先选 PPO

我给你准备的是 PPO 路线，原因是：

- 对连续动作控制比较常见
- 文档和教程多
- 在单机器人、低并发、Gazebo 这类较慢环境里更容易先跑通

但你也要注意：Gazebo 单实例训练速度会比较慢，远慢于 Isaac Gym / Mujoco 这类高并发模拟器。

所以更现实的预期是：

- 先用 Gazebo 跑通“训练管线”
- 先学会平衡和低速前进
- 真正想追求高效率训练时，再考虑并行环境或更快的模拟器

## 7. CPU 和 GPU 怎么选

你的机器上已经有 CUDA 驱动，但显卡显存只有 2GB 左右，因此建议这样使用：

- 初期调试：
  用 `training.device: cpu`
- 小网络正式训练：
  用 `training.device: auto` 或 `cuda`
- 如果显存不够：
  降低 `policy_net_arch`、`batch_size`

当前配置文件已经支持：

- `cpu`
- `cuda`
- `auto`

不需要改训练代码，只要改 YAML 或命令行参数。

## 8. 已准备好的代码结构

### 仿真接口

- `guguji_ros2_ws/src/guguji_ros2/launch/gazebo.launch.py`
  Gazebo Fortress 启动与桥接
- `guguji_ros2_ws/src/guguji_ros2/urdf/guguji.urdf`
  Gazebo 插件、关节控制器、关节状态和位姿发布器

### 强化学习工程

- `guguji_rl/guguji_rl/ros2_interface.py`
  ROS 2 / Gazebo 接口封装
- `guguji_rl/guguji_rl/envs/gazebo_biped_env.py`
  Gymnasium 环境
- `guguji_rl/guguji_rl/rewards.py`
  奖励函数
- `guguji_rl/scripts/train.py`
  PPO 训练
- `guguji_rl/scripts/evaluate.py`
  加载策略并运行
- `guguji_rl/scripts/run_policy.py`
  把训练好的策略作为在线控制程序持续运行
- `guguji_rl/scripts/check_env.py`
  环境检查

## 9. 推荐的第一轮实践步骤

1. 启动 Gazebo 仿真，确认机器人正常出现。
2. 运行 `check_env.py`，确认 reset / step / reward 能跑通。
3. 训练 `balance_ppo.yaml`，不要急着训练 walking。
4. 观察 `/joint_states` 和 `/tf`，检查机器人是否会频繁倒地。
5. 调整奖励和关节范围后，再切到 `walk_ppo.yaml`。

## 10. 训练完成后怎么在 ROS 2 里持续控制

训练完成后，建议分成两步：

1. 用 `evaluate.py` 做短时回放，确认模型文件没有问题。
2. 用 `run_policy.py` 持续运行策略，让它在当前 Gazebo / ROS 2 系统里不断读取状态并发布关节命令。

示例命令：

```bash
cd /home/corvin/Project/guguji_simulation/guguji_rl
source .venv/bin/activate
python3 scripts/run_policy.py \
  --config configs/walk_ppo.yaml \
  --model outputs/<你的实验目录>/final_model.zip \
  --deterministic
```

## 11. 后续你最可能会继续改的地方

你后面大概率会集中修改这些点：

- `guguji_rl/guguji_rl/rewards.py`
  调奖励项和权重
- `guguji_rl/guguji_rl/envs/gazebo_biped_env.py`
  调 observation、done 条件、动作映射
- `guguji_rl/configs/*.yaml`
  调训练超参数、设备、目标速度
- `guguji_ros2_ws/src/guguji_ros2/urdf/guguji.urdf`
  调关节范围、PD 增益、Gazebo 控制器参数

## 12. 现实建议

对双足机器人来说，第一次做强化学习最难的通常不是“算法”，而是下面三件事：

- 动作空间是否合理
- 奖励函数是否稳定
- reset 是否干净且可重复

所以你的第一目标不要设成“尽快走得很像人”，而是：

- 先站住
- 再学会规律摆腿
- 最后再追求前进速度和稳定性

这条路线更稳，也更适合第一次上手。
