import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


JOINT_NAMES = [
    'left_hip_pitch_joint',
    'left_knee_pitch_joint',
    'left_ankle_pitch_joint',
    'left_ankle_joint',
    'right_hip_pitch_joint',
    'right_knee_pitch_joint',
    'right_ankle_pitch_joint',
    'right_ankle_joint',
]


def _configure_gz_args(context, *args, **kwargs):
    """根据 launch 参数拼接 Fortress 的启动命令参数。"""
    world = LaunchConfiguration('world').perform(context)
    gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
    pause = LaunchConfiguration('pause').perform(context).lower() == 'true'
    verbose = LaunchConfiguration('verbose').perform(context).strip().lower()

    gz_args = []

    # -r 表示启动后立即运行；如果 pause:=true，则故意不加 -r。
    if not pause:
        gz_args.append('-r')

    # Fortress 下 -s 表示仅启动服务端，不弹 GUI。
    if not gui:
        gz_args.append('-s')

    # 兼容两种传参方式：
    # 1. verbose:=true  -> 使用较详细的 4 级日志
    # 2. verbose:=3     -> 直接指定日志等级
    if verbose not in ('', 'false', '0', 'no', 'off'):
        verbose_level = '4' if verbose in ('true', 'yes', 'on') else verbose
        gz_args.extend(['-v', verbose_level])

    gz_args.append(world)

    return [
        SetLaunchConfiguration(
            'gz_args_resolved',
            ' '.join(shlex.quote(arg) for arg in gz_args)
        )
    ]


def _create_bridge_node(context, *args, **kwargs):
    """创建 Fortress <-> ROS 的桥接节点。"""
    world_name = LaunchConfiguration('world_name').perform(context)
    bridge_tf = LaunchConfiguration('bridge_tf').perform(context).lower() == 'true'

    bridge_arguments = [
        # 仿真时钟：Gazebo -> ROS
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        # 关节状态：Gazebo -> ROS
        f'/world/{world_name}/model/guguji/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
    ]

    remappings = [
        (f'/world/{world_name}/model/guguji/joint_state', '/joint_states'),
    ]

    if bridge_tf:
        # 这里把 Gazebo 的 Pose_V 直接桥接到 /tf。
        # 这样 TF 反映的就是仿真里的真实位姿，而不是仅由 URDF 正向解算得到的理想位姿。
        bridge_arguments.append('/model/guguji/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V')
        remappings.append(('/model/guguji/pose', '/tf'))

    # 每个关节都提供一个 ROS -> Gazebo 的位置命令话题。
    # 话题名与 URDF 中控制器插件的 <topic> 配置保持一致。
    for joint_name in JOINT_NAMES:
        bridge_arguments.append(
            f'/guguji/command/{joint_name}@std_msgs/msg/Float64]gz.msgs.Double'
        )

    return [
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='guguji_gz_bridge',
            output='screen',
            arguments=bridge_arguments,
            remappings=remappings,
        )
    ]


def generate_launch_description():
    package_name = 'guguji_ros2'
    package_share = get_package_share_directory(package_name)
    package_share_parent = os.path.dirname(package_share)
    urdf_path = os.path.join(package_share, 'urdf', 'guguji.urdf')
    default_world = os.path.join(package_share, 'worlds', 'empty.world')

    # 这里保留 robot_state_publisher，主要用于提供 /robot_description。
    # 仿真中的真实 joint_states 和 /tf 则由 Gazebo + ros_gz_bridge 负责。
    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    world = LaunchConfiguration('world')
    world_name = LaunchConfiguration('world_name')
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    pause = LaunchConfiguration('pause')
    verbose = LaunchConfiguration('verbose')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    z_pose = LaunchConfiguration('z_pose')
    roll = LaunchConfiguration('roll')
    pitch = LaunchConfiguration('pitch')
    yaw = LaunchConfiguration('yaw')

    # Gazebo Fortress 通过资源搜索路径查找 package:// 包内网格文件。
    # 对于安装后的包，加入 share 和 share 的父目录后，
    # 就能正确解析 package://guguji_ros2/meshes/... 这类路径。
    resource_path_actions = [
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=[
                package_share_parent,
                os.pathsep,
                package_share,
                os.pathsep,
                EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value=''),
            ],
        ),
        SetEnvironmentVariable(
            name='IGN_GAZEBO_RESOURCE_PATH',
            value=[
                package_share_parent,
                os.pathsep,
                package_share,
                os.pathsep,
                EnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', default_value=''),
            ],
        ),
    ]

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            ])
        ),
        launch_arguments={
            'gz_args': LaunchConfiguration('gz_args_resolved'),
            'on_exit_shutdown': 'true',
        }.items(),
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_guguji',
        output='screen',
        arguments=[
            '-world', world_name,
            '-name', 'guguji',
            # 这里直接从安装后的 URDF 文件生成模型，
            # 启动链路更直接，也避免了 spawn 依赖 /robot_description 话题时序。
            '-file', urdf_path,
            '-x', x_pose,
            '-y', y_pose,
            '-z', z_pose,
            '-R', roll,
            '-P', pitch,
            '-Y', yaw,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Gazebo 启动时使用的 world 文件路径'
        ),
        DeclareLaunchArgument(
            'world_name',
            default_value='default',
            description='在 world 文件中用于生成实体的 world 名称'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='是否使用仿真时钟'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='是否启动 Gazebo 图形界面'
        ),
        DeclareLaunchArgument(
            'pause',
            default_value='false',
            description='Gazebo 启动后是否暂停'
        ),
        DeclareLaunchArgument(
            'verbose',
            default_value='false',
            description='Gazebo 日志等级；可填 false/true 或数值 0~4'
        ),
        DeclareLaunchArgument(
            'bridge_tf',
            default_value='true',
            description='是否把 Gazebo 的真实位姿桥接到 ROS /tf'
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='0.0',
            description='模型生成时的 X 坐标'
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='0.0',
            description='模型生成时的 Y 坐标'
        ),
        DeclareLaunchArgument(
            'z_pose',
            default_value='0.35',
            description='模型生成时的 Z 坐标，适当抬高可以避免初始穿地'
        ),
        DeclareLaunchArgument(
            'roll',
            default_value='0.0',
            description='模型生成时的 Roll 角'
        ),
        DeclareLaunchArgument(
            'pitch',
            default_value='0.0',
            description='模型生成时的 Pitch 角'
        ),
        DeclareLaunchArgument(
            'yaw',
            default_value='0.0',
            description='模型生成时的 Yaw 角'
        ),
        # 先根据 launch 参数拼出 Fortress 的 gz_args，再去 include 官方启动文件。
        OpaqueFunction(function=_configure_gz_args),
        *resource_path_actions,
        gazebo_launch,
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            # Gazebo 已经会发布真实 /tf，因此这里把 robot_state_publisher 的 TF 重定向到调试话题，
            # 只保留它的 /robot_description 能力，避免双份 TF 冲突。
            remappings=[
                ('/tf', '/guguji/debug/robot_state_publisher/tf'),
                ('/tf_static', '/guguji/debug/robot_state_publisher/tf_static'),
            ],
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }],
        ),
        # 创建 ROS <-> Gazebo 的桥接节点：
        # /clock、/joint_states、/tf 以及每个关节的位置控制命令都从这里打通。
        OpaqueFunction(function=_create_bridge_node),
        # 稍微等待 Gazebo 世界初始化完成后再创建模型，可减少 Fortress 启动初期的竞争问题。
        TimerAction(
            period=2.0,
            actions=[spawn_entity],
        ),
    ])
