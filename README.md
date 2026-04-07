# guguji_simulation

guguji 的 ROS 2 Humble 仿真环境。

## 当前结构

`guguji_ros2` 是已经整理好的 ROS 2 软件包，包含：

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
  ros-humble-rviz2 \
  ros-humble-xacro \
  ros-humble-gazebo-ros-pkgs
```

## 构建

在仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select guguji_ros2
source install/setup.bash
```

## 启动方式

仅查看模型与 TF：

```bash
ros2 launch guguji_ros2 display.launch.py
```

启动 Gazebo 仿真：

```bash
ros2 launch guguji_ros2 gazebo.launch.py
```

如果你在无图形界面的环境中调试，可以关闭 Gazebo GUI：

```bash
ros2 launch guguji_ros2 gazebo.launch.py gui:=false
```
