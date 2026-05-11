cmake_minimum_required(VERSION 3.10)

project(guguji_ros2)

find_package(ament_cmake REQUIRED)

# 这是一个纯描述/仿真资源包，不编译节点源码，
# 只需要把 launch、URDF、mesh、world 等资源安装到 share 目录即可。
install(
  DIRECTORY
    config
    launch
    meshes
    rviz
    # textures #可能是模型的纹理贴图，原git仓库里没有
    urdf
    worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
