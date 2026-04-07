# guguji_simulation

guguji 的 ROS 2 Humble 仿真环境。

## 当前结构

当前工作空间位于 `guguji_ros2_ws/`，其中 `src/guguji_ros2` 是机器人描述与仿真包。

`guguji_ros2` 软件包包含：

- `urdf/`：机器人模型
- `meshes/`：网格模型
- `launch/`：ROS 2 启动文件
- `worlds/`：Gazebo world 文件
- `config/`：关节配置

## 依赖安装

在 Ubuntu 22.04 + ROS 2 Humble 环境中，至少需要安装：

```bash
sudo apt update
sudo apt install \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-robot-state-publisher \
  ros-humble-ros-gz \
  ros-humble-ros-gz-bridge \
  ros-humble-ros-gz-sim \
  ros-humble-rviz2 \
  ros-humble-sdformat-urdf \
  ros-humble-xacro \
  ros-humble-tf2-msgs
```

## 构建

在仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
cd guguji_ros2_ws
colcon build --packages-select guguji_ros2
source install/setup.bash
```

## 启动方式

仅查看模型与 TF：

```bash
ros2 launch guguji_ros2 display.launch.py
```

启动 Gazebo Fortress 仿真：

```bash
ros2 launch guguji_ros2 gazebo.launch.py
```

如果你在无图形界面的环境中调试，可以关闭 Gazebo GUI：

```bash
ros2 launch guguji_ros2 gazebo.launch.py gui:=false
```

## Fortress 调试接口

`gazebo.launch.py` 现在已经接通了下面几类接口：

- `/clock`：Gazebo 仿真时钟
- `/joint_states`：Gazebo 真实关节状态
- `/tf`：Gazebo 真实模型/连杆位姿
- `/guguji/command/<joint_name>`：每个关节的位置控制命令，消息类型为 `std_msgs/msg/Float64`

## 控制示例

例如把左膝关节目标角度设置为 `0.8 rad`：

```bash
ros2 topic pub --once /guguji/command/left_knee_pitch_joint std_msgs/msg/Float64 "{data: 0.8}"
```

查看关节状态：

```bash
ros2 topic echo /joint_states
```

查看 TF：

```bash
ros2 topic echo /tf
```
