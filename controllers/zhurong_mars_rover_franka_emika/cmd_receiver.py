import numpy as np
import threading

class CmdVelReceiver:
    """ROS1 /cmd_vel receiver.

    Maps geometry_msgs/Twist to cmd = [vx, vy, wz] (policy command order).
    """

    def __init__(self, topic: str = "/cmd_vel") -> None:
        self._lock = threading.Lock()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._last_msg_time = None  # type: float | None

        try:
            import rospy  # type: ignore
            from geometry_msgs.msg import Twist  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ROS1 python packages not available. "
                "Please source your ROS1 environment (e.g. `source /opt/ros/noetic/setup.bash`) "
                "and ensure `rospy` + `geometry_msgs` are installed."
            ) from e

        self._rospy = rospy
        self._Twist = Twist

        self._sub = rospy.Subscriber(topic, Twist, self._cb, queue_size=1)

    def _cb(self, msg) -> None:
        cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)
        with self._lock:
            self._cmd[:] = cmd
            try:
                self._last_msg_time = float(self._rospy.get_time())
            except Exception:
                self._last_msg_time = time.time()

    def get_cmd(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()

    @property
    def last_msg_time(self):
        with self._lock:
            return self._last_msg_time