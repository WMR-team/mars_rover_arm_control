import time
from typing import Iterable, List, Optional


class RosJointStatePublisher:
    """ROS1 JointState publisher for target and actual joint positions."""

    def __init__(
        self,
        node_name: str,
        target_topic: str,
        actual_topic: str,
        joint_names: Optional[List[str]] = None,
        publish_hz: float = 0.0,
        queue_size: int = 10,
    ) -> None:
        try:
            import rospy  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "ROS1 python packages not available. "
                "Please source your ROS1 environment (e.g. `source /opt/ros/noetic/setup.bash`) "
                "and ensure `rospy` + `sensor_msgs` are installed."
            ) from exc

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        self._rospy = rospy
        self._JointState = JointState
        self._target_pub = rospy.Publisher(
            target_topic, JointState, queue_size=queue_size
        )
        self._actual_pub = rospy.Publisher(
            actual_topic, JointState, queue_size=queue_size
        )

        self._joint_names = list(joint_names) if joint_names is not None else None
        self._expected_len = len(self._joint_names) if self._joint_names else None
        self._publish_interval = (
            1.0 / publish_hz if publish_hz and publish_hz > 0.0 else 0.0
        )
        self._last_pub_time = 0.0

    def _make_msg(self, positions: Iterable[float]):
        msg = self._JointState()
        msg.header.stamp = self._rospy.Time.now()
        if self._joint_names is not None:
            msg.name = self._joint_names
        msg.position = list(positions)
        return msg

    def publish(
        self, target_positions: Iterable[float], actual_positions: Iterable[float]
    ) -> None:
        if self._publish_interval > 0.0:
            now = time.time()
            if now - self._last_pub_time < self._publish_interval:
                return
            self._last_pub_time = now

        target_list = list(target_positions)
        actual_list = list(actual_positions)

        if self._expected_len is not None:
            if (
                len(target_list) != self._expected_len
                or len(actual_list) != self._expected_len
            ):
                raise ValueError(
                    "JointState length mismatch: expected "
                    f"{self._expected_len}, got target={len(target_list)}, actual={len(actual_list)}."
                )

        target_msg = self._make_msg(target_list)
        actual_msg = self._make_msg(actual_list)
        self._target_pub.publish(target_msg)
        self._actual_pub.publish(actual_msg)
