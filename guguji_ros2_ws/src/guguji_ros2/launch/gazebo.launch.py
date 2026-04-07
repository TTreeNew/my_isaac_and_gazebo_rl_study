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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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


def generate_launch_description():
    package_name = 'guguji_ros2'
    package_share = get_package_share_directory(package_name)
    package_share_parent = os.path.dirname(package_share)
    urdf_path = os.path.join(package_share, 'urdf', 'guguji.urdf')
    default_world = os.path.join(package_share, 'worlds', 'empty.world')

    # 这里继续保留 robot_state_publisher + joint_state_publisher 的组合，
    # 这样即使暂时还没有接入 ros2_control，也能先把 TF 树和基础模型显示跑起来。
    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    world = LaunchConfiguration('world')
    world_name = LaunchConfiguration('world_name')
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    pause = LaunchConfiguration('pause')
    verbose = LaunchConfiguration('verbose')
    publish_joint_state = LaunchConfiguration('publish_joint_state')
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
            # 直接从 /robot_description 取模型字符串，
            # 后续如果你把 URDF 改成 xacro，也不需要改这里的加载方式。
            '-topic', '/robot_description',
            '-x', x_pose,
            '-y', y_pose,
            '-z', z_pose,
            '-R', roll,
            '-P', pitch,
            '-Y', yaw,
            '-allow_renaming', 'true',
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
            'publish_joint_state',
            default_value='true',
            description='未接入控制器前，是否用 joint_state_publisher 发布默认关节角'
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
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            condition=IfCondition(publish_joint_state),
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }],
        ),
        # 稍微等待 Gazebo 世界初始化完成后再创建模型，可减少 Fortress 启动初期的竞争问题。
        TimerAction(
            period=2.0,
            actions=[spawn_entity],
        ),
    ])
