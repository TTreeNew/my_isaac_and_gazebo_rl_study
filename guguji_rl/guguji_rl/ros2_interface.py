from __future__ import annotations

import threading
import time
from dataclasses import dataclass
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

    def snapshot(self) -> RobotStateSnapshot | None:
        with self._lock:
            joint_state = self._latest_joint_state
            tf_message = self._latest_tf
            clock_message = self._latest_clock

        if joint_state is None:
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

        base_position = np.zeros(3, dtype=np.float32)
        base_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if tf_message is not None:
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
                    break

        sim_time = 0.0
        if clock_message is not None:
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
    ) -> None:
        self._context = Context()
        # 训练程序使用独立 context，避免和外部 ROS 2 进程互相干扰。
        rclpy.init(args=None, context=self._context)

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
