import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'guguji_ros2'
    package_share = get_package_share_directory(package_name)
    urdf_path = os.path.join(package_share, 'urdf', 'guguji.urdf')
    rviz_config_path = os.path.join(package_share, 'rviz', 'guguji.rviz')

    # 直接读取 URDF 文件内容，并分别注入到状态发布节点中。
    # 在 ROS 2 中参数是节点私有的，不能像 ROS 1 那样依赖全局参数服务器。
    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    use_joint_state_gui = LaunchConfiguration('use_joint_state_gui')
    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_joint_state_gui',
            default_value='true',
            description='是否启动带滑块的 joint_state_publisher_gui'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='是否启动 RViz2'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': False,
            }],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            condition=IfCondition(use_joint_state_gui),
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': False,
            }],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            condition=UnlessCondition(use_joint_state_gui),
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': False,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            condition=IfCondition(use_rviz),
            output='screen',
            arguments=['-d', rviz_config_path],
        ),
    ])
