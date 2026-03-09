from math import tau
import threading
import time
import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, SetBoolResponse
import os
import yaml
import mujoco.viewer
import mujoco

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

class ZhurongMarsRoverControl(object):
    def __init__(self, m:mujoco.MjModel, d:mujoco.MjData):

        self.control_msg = Float64()

        self.h = 0.652  # 1/2 width
        self.l = 0.775  # 1/2 length
        self.r = 0.15

        self.cam_pitch = 0
        self.cam_yaw = 0
        self.zhurong_publishers = {}
        self.controller_command = "command"

        self.m = m
        self.d = d

        self.cmd_vel_msg = Twist()
        self.control_msg.data = 0

        rospy.logwarn("ZhurongMarsRoverControl...READY")

    def set_cmd_vel(self, body_velocity_x: float, body_velocity_y: float, body_omega: float):
        # print('received!!!')
        self.body_velocity = body_velocity_x
        self.body_omega = body_omega
        self.move_with_cmd_vel()

    def cam_ctl_callback(self, msg):
        self.cam_yaw = msg.data
        self.set_navcam_angle()

    def set_turning_radius(self, turn_radius):
        for i in range(6):
            self.d.ctrl[i+12] = turn_radius[i]

    def set_wheels_speed(self, turning_speed):
        """
        Sets the turning speed in radians per second
        :param turning_speed: In radians per second
        :return:
        """
        # TODO: turning_speed for each wheel should change based on ackerman.
        for i in range(6):
            self.d.ctrl[i] = turning_speed[i]

    def set_navcam_angle(self):

        self.cam_yaw_msg.data = self.cam_yaw
        self.cam_pitch_msg.data = self.cam_pitch
        self.cam_yaw_ctl_publisher.publish(self.cam_yaw_msg.data)
        self.cam_pitch_ctl_publisher.publish(self.cam_pitch_msg.data)

        self.body_velocity = 0.3
        self.body_omega = -0.08
        self.move_with_cmd_vel()

    def cam_pitch_ctl_1(self):
        self.cam_pitch += 0.1
        self.set_navcam_angle()

    def cam_pitch_ctl_2(self):
        self.cam_pitch -= 0.1
        self.set_navcam_angle()

    def cam_yaw_ctl_1(self):
        self.cam_yaw += 0.1
        self.set_navcam_angle()

    def cam_yaw_ctl_2(self):
        self.cam_yaw -= 0.1
        self.set_navcam_angle()

    def move_with_cmd_vel(self):
        if self.body_omega == 0:
            theta = np.zeros(6)
            self.set_turning_radius(theta)
            vel_arr = np.ones(6) * self.body_velocity / self.r
            self.set_wheels_speed(vel_arr)
        else:
            turning_radius = self.body_velocity / self.body_omega
            r_arr = np.zeros(6)
            r_arr[0] = np.sqrt((turning_radius - self.h) ** 2 + self.l**2)
            r_arr[1] = np.sqrt((turning_radius + self.h) ** 2 + self.l**2)
            r_arr[2] = abs(turning_radius - self.h)
            r_arr[3] = abs(turning_radius + self.h)
            r_arr[4] = r_arr[0]
            r_arr[5] = r_arr[1]
            vel_arr = abs(self.body_omega) * r_arr / self.r
            if self.body_velocity < 0:
                vel_arr = -vel_arr
            elif self.body_velocity == 0 and self.body_omega < 0:
                vel_arr = -vel_arr
            # print(vel_arr)

            theta = np.zeros(6)
            theta[0] = np.arctan(self.l / (turning_radius - self.h))
            theta[1] = np.arctan(self.l / (turning_radius + self.h))
            theta[2] = 0
            theta[3] = 0
            theta[4] = -theta[0]
            theta[5] = -theta[1]

            if turning_radius >= 0 and abs(turning_radius) < self.h:
                vel_arr[0] = -vel_arr[0]
                vel_arr[2] = -vel_arr[2]
                vel_arr[4] = -vel_arr[4]

            elif turning_radius < 0 and abs(turning_radius) < self.h:
                vel_arr[1] = -vel_arr[1]
                vel_arr[3] = -vel_arr[3]
                vel_arr[5] = -vel_arr[5]

            # print(theta)

            self.set_turning_radius(theta)
            self.set_wheels_speed(vel_arr)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configs/config.yaml")
MODEL_DIR = os.path.join(ROOT_DIR, "../../planetary_robot_zoo/zhurong_mars_rover/scene.xml")

if __name__ == "__main__":
    # ROS1 subscriber for /cmd_vel
    counter = 0
    cmd_vel_receiver = None
    try:
        import rospy  # type: ignore
        rospy.init_node("deploy_mujoco", anonymous=True, disable_signals=True)
        cmd_vel_receiver = CmdVelReceiver(topic="/cmd_vel")
        print("[deploy_mujoco_ros1] ROS1 subscribed to: /cmd_vel")
    except Exception as e:
        print(f"[deploy_mujoco_ros1] ROS1 init/subscribe failed: {e}")
        print("[deploy_mujoco_ros1] Continuing without ROS commands (using cmd_init).")
        cmd_vel_receiver = None

    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = MODEL_DIR
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    zhurong_mars_rover_control = ZhurongMarsRoverControl(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Read q/dq in actuator ctrl order (robust to joint ordering differences).

            # PD in Mujoco ctrl order.

            # mj_step can be replaced with code that also evaluates

            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # Update commands from ROS1 /cmd_vel (vx, vy, wz) if available.
                if cmd_vel_receiver is not None:
                    cmd = cmd_vel_receiver.get_cmd()
                    zhurong_mars_rover_control.set_cmd_vel(cmd[0], cmd[1], cmd[2])

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # time.sleep(10)
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)