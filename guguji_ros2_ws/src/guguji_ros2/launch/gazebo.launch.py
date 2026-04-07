import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'guguji_ros2'
    package_share = get_package_share_directory(package_name)
    urdf_path = os.path.join(package_share, 'urdf', 'guguji.urdf')
    default_world = os.path.join(package_share, 'worlds', 'empty.world')

    # 这里继续保留 robot_state_publisher + joint_state_publisher 的组合，
    # 这样即使暂时还没有接入 ros2_control，也能先把 TF 树和基础模型显示跑起来。
    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    world = LaunchConfiguration('world')
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

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py',
            ])
        ),
        launch_arguments={
            'world': world,
            'gui': gui,
            'pause': pause,
            'verbose': verbose,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Gazebo 启动时使用的 world 文件路径'
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
            description='是否输出 Gazebo 详细日志'
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
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_guguji',
            output='screen',
            arguments=[
                '-entity', 'guguji',
                '-file', urdf_path,
                '-x', x_pose,
                '-y', y_pose,
                '-z', z_pose,
                '-R', roll,
                '-P', pitch,
                '-Y', yaw,
            ],
        ),
    ])
