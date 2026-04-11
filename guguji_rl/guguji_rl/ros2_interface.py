from __future__ import annotations

import threading
import time
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from ros_gz_interfaces.srv import ControlWorld
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage


@dataclass
class RobotStateSnapshot:
    sim_time: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    base_position: np.ndarray
    base_quaternion: np.ndarray


class _GugujiInterfaceNode(Node):
    def __init__(
        self,
        *,
        context: Context,
        joint_names: list[str],
        command_topic_prefix: str,
        joint_state_topic: str,
        tf_topic: str,
        clock_topic: str,
        world_control_service: str,
        model_name: str,
    ) -> None:
        super().__init__('guguji_rl_interface', context=context)
        self.joint_names = joint_names
        self.model_name = model_name
        self.command_publishers = {
            joint_name: self.create_publisher(
                Float64,
                f'{command_topic_prefix}/{joint_name}',
                10,
            )
            for joint_name in joint_names
        }

        self._lock = threading.Lock()
        self._latest_joint_state: JointState | None = None
        self._latest_tf: TFMessage | None = None
        self._latest_clock: Clock | None = None

        self.create_subscription(JointState, joint_state_topic, self._joint_state_callback, 10)
        self.create_subscription(TFMessage, tf_topic, self._tf_callback, 50)
        self.create_subscription(Clock, clock_topic, self._clock_callback, 10)
        self.world_control_client = self.create_client(ControlWorld, world_control_service)

    def _joint_state_callback(self, message: JointState) -> None:
        with self._lock:
            self._latest_joint_state = message

    def _tf_callback(self, message: TFMessage) -> None:
        with self._lock:
            self._latest_tf = message

    def _clock_callback(self, message: Clock) -> None:
        with self._lock:
            self._latest_clock = message

    def _extract_base_pose(self, tf_message: TFMessage) -> tuple[np.ndarray, np.ndarray] | None:
        """从 TF 消息中提取机器人整体模型在世界坐标系下的位姿。"""
        # Gazebo PosePublisher 转成 /tf 后，整机模型通常对应 child_frame_id == model_name，
        # 例如这里会出现 child_frame_id == "guguji"、parent == "default" 的变换。
        for transform in tf_message.transforms:
            if transform.child_frame_id == self.model_name:
                base_position = np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    dtype=np.float32,
                )
                base_quaternion = np.array(
                    [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ],
                    dtype=np.float32,
                )
                return base_position, base_quaternion

        return None

    def snapshot(self) -> RobotStateSnapshot | None:
        with self._lock:
            joint_state = self._latest_joint_state
            tf_message = self._latest_tf
            clock_message = self._latest_clock

        # 强化学习环境依赖 joint_state、真实位姿和仿真时钟三者同时存在。
        # 如果这里只等到 joint_state 就提前返回，会导致 base pose 仍然是默认全零值，
        # 进而让环境误以为机器人一开始就已经倒地。
        if joint_state is None or tf_message is None or clock_message is None:
            return None

        joint_map = {name: index for index, name in enumerate(joint_state.name)}
        joint_position = np.zeros(len(self.joint_names), dtype=np.float32)
        joint_velocity = np.zeros(len(self.joint_names), dtype=np.float32)

        for output_index, joint_name in enumerate(self.joint_names):
            source_index = joint_map.get(joint_name)
            if source_index is None:
                continue
            if source_index < len(joint_state.position):
                joint_position[output_index] = joint_state.position[source_index]
            if source_index < len(joint_state.velocity):
                joint_velocity[output_index] = joint_state.velocity[source_index]

        base_pose = self._extract_base_pose(tf_message)
        if base_pose is None:
            return None

        base_position, base_quaternion = base_pose

        sim_time = 0.0
        sim_time = clock_message.clock.sec + clock_message.clock.nanosec * 1e-9

        return RobotStateSnapshot(
            sim_time=sim_time,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            base_position=base_position,
            base_quaternion=base_quaternion,
        )


class GugujiRos2Interface:
    """封装 RL 环境需要的 ROS 2 通信接口。"""

    def __init__(
        self,
        *,
        joint_names: list[str],
        command_topic_prefix: str,
        joint_state_topic: str,
        tf_topic: str,
        clock_topic: str,
        world_control_service: str,
        model_name: str,
        world_name: str,
        model_file_path: str,
        spawn_x: float,
        spawn_y: float,
        spawn_z: float,
        spawn_roll: float,
        spawn_pitch: float,
        spawn_yaw: float,
    ) -> None:
        self._context = Context()
        # 训练程序使用独立 context，避免和外部 ROS 2 进程互相干扰。
        rclpy.init(args=None, context=self._context)
        self._world_name = world_name
        self._model_name = model_name
        self._model_file_path = str(Path(model_file_path).resolve())
        self._spawn_x = float(spawn_x)
        self._spawn_y = float(spawn_y)
        self._spawn_z = float(spawn_z)
        self._spawn_roll = float(spawn_roll)
        self._spawn_pitch = float(spawn_pitch)
        self._spawn_yaw = float(spawn_yaw)

        self._node = _GugujiInterfaceNode(
            context=self._context,
            joint_names=joint_names,
            command_topic_prefix=command_topic_prefix,
            joint_state_topic=joint_state_topic,
            tf_topic=tf_topic,
            clock_topic=clock_topic,
            world_control_service=world_control_service,
            model_name=model_name,
        )

        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        # 单独起一个线程持续 spin，这样训练主循环可以专注做 step / reward。
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _run_ign_service(
        self,
        *,
        service_name: str,
        request_type: str,
        response_type: str,
        request_payload: str,
        timeout_ms: int = 5000,
        allow_false_response: bool = False,
    ) -> bool:
        """调用 Gazebo 原生 transport service。"""
        completed = subprocess.run(
            [
                'ign',
                'service',
                '-s',
                service_name,
                '--reqtype',
                request_type,
                '--reptype',
                response_type,
                '--timeout',
                str(timeout_ms),
                '--req',
                request_payload,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f'调用 Gazebo service 失败: {service_name}\n'
                f'stdout: {completed.stdout}\n'
                f'stderr: {completed.stderr}'
            )

        output = completed.stdout.strip().lower()
        if 'data: true' in output:
            return True
        if allow_false_response and 'data: false' in output:
            return False
        raise RuntimeError(
            f'Gazebo service 返回了意外结果: {service_name}\n'
            f'stdout: {completed.stdout}\n'
            f'stderr: {completed.stderr}'
        )

    def _spawn_quaternion_xyzw(self) -> tuple[float, float, float, float]:
        """把配置中的 RPY 初始姿态转换成 Gazebo 需要的四元数。"""
        half_roll = self._spawn_roll * 0.5
        half_pitch = self._spawn_pitch * 0.5
        half_yaw = self._spawn_yaw * 0.5

        cr = math.cos(half_roll)
        sr = math.sin(half_roll)
        cp = math.cos(half_pitch)
        sp = math.sin(half_pitch)
        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)

        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        return x, y, z, w

    def respawn_model(self, *, pause_after_reset: bool) -> None:
        """通过 model_only reset + set_pose 做真正的模型复位。"""
        # Fortress 对动态生成模型执行 world reset 时，往往不会可靠恢复起始位姿；
        # 但如果直接 remove/create，又会触发当前 JointStatePublisher 插件崩溃。
        # 这里改用更稳妥的方案：先 reset 模型状态，再把整机姿态强制放回初始位置。
        self.control_world(
            pause=True,
            reset_model_only=True,
            timeout=5.0,
        )
        time.sleep(0.1)

        qx, qy, qz, qw = self._spawn_quaternion_xyzw()
        set_pose_payload = (
            f'name: "{self._model_name}" '
            'position: { '
            f'x: {self._spawn_x} y: {self._spawn_y} z: {self._spawn_z} '
            '} '
            'orientation: { '
            f'x: {qx} y: {qy} z: {qz} w: {qw} '
            '}'
        )
        self._run_ign_service(
            service_name=f'/world/{self._world_name}/set_pose',
            request_type='ignition.msgs.Pose',
            response_type='ignition.msgs.Boolean',
            request_payload=set_pose_payload,
            timeout_ms=5000,
        )
        time.sleep(0.2)
        self.control_world(pause=pause_after_reset, timeout=5.0)

    def wait_for_world_control_service(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._node.world_control_client.wait_for_service(timeout_sec=0.2):
                return
        raise TimeoutError('等待 Gazebo world control 服务超时。')

    def wait_for_snapshot(self, timeout: float = 5.0) -> RobotStateSnapshot:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = self._node.snapshot()
            if snapshot is not None:
                # snapshot 把训练需要的 joint_states / tf / clock 聚合成了一份结构化状态。
                return snapshot
            time.sleep(0.05)
        raise TimeoutError('等待 joint_states / tf 数据超时。')

    def publish_joint_targets(self, joint_targets: dict[str, float] | Iterable[float]) -> None:
        if isinstance(joint_targets, dict):
            target_map = joint_targets
        else:
            target_map = {
                joint_name: float(value)
                for joint_name, value in zip(self._node.joint_names, joint_targets)
            }

        for joint_name, target_value in target_map.items():
            publisher = self._node.command_publishers[joint_name]
            message = Float64()
            message.data = float(target_value)
            publisher.publish(message)

    def control_world(
        self,
        *,
        pause: bool | None = None,
        step: bool = False,
        multi_step: int = 0,
        reset_all: bool = False,
        reset_model_only: bool = False,
        reset_time_only: bool = False,
        timeout: float = 5.0,
    ) -> bool:
        request = ControlWorld.Request()
        if pause is not None:
            request.world_control.pause = pause
        request.world_control.step = step
        request.world_control.multi_step = int(multi_step)
        request.world_control.reset.all = reset_all
        request.world_control.reset.model_only = reset_model_only
        request.world_control.reset.time_only = reset_time_only

        future = self._node.world_control_client.call_async(request)
        deadline = time.time() + timeout

        while time.time() < deadline:
            if future.done():
                response = future.result()
                return bool(response.success)
            time.sleep(0.01)

        raise TimeoutError('调用 Gazebo world control 服务超时。')

    def reset_world(self, *, pause_after_reset: bool) -> bool:
        return self.control_world(
            pause=pause_after_reset,
            reset_all=True,
        )

    def step_world(self, multi_step: int) -> bool:
        return self.control_world(
            pause=True,
            step=True,
            multi_step=multi_step,
        )

    def close(self) -> None:
        try:
            self._executor.shutdown()
        finally:
            self._node.destroy_node()
            rclpy.shutdown(context=self._context)
            self._spin_thread.join(timeout=1.0)
