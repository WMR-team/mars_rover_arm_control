import os
import threading
import time
from dataclasses import dataclass
import sys

# 当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（根据你的层级向上走几级）
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..")
)  # 自行调整级数
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 日志目录
LOG_ROOT = os.path.join(CURRENT_DIR, "logs")

import mujoco
import numpy as np
import mujoco.viewer
import pinocchio
import yaml
from typing import List
from mars_rover_arm_control.utils.time_analysis import timeit, warn_if_overrun
from mars_rover_arm_control.utils.print_control import (
    control_print,
    init_run_logger,
    log_and_print,
)
from mars_rover_arm_control.utils.arm_kinematics_utils import (
    print_pinocchio_info,
    mujoco_q_to_pinocchio_q,
    pinocchio_q_to_mujoco_q,
    inverse_arm_kinematics,
    inverse_arm_kinematics_bounded,
    inverse_arm_kinematics_bounded_retry,
    inverse_kinematics,
)
from mars_rover_arm_control.utils.fps_counter import FPSCounter
from mars_rover_arm_control.utils.ros_joint_publisher import RosJointStatePublisher
from mars_rover_arm_control.utils.thread_pool import ThreadPool
from mars_rover_arm_control.utils.trajectory_utils import trajectory_point
from mars_rover_arm_control.controllers.zhurong_mars_rover_franka_emika.control_api import (
    ArmController,
    ArmKinematics,
    RoverController,
    TaskCoordinator,
)

CONFIG_FILE_PATH = os.path.join(CURRENT_DIR, "configs/config.yaml")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


CONFIG = load_config(CONFIG_FILE_PATH)

MODEL_PATH = CONFIG["pinocchio_model_path"] or os.path.join(
    CURRENT_DIR, "../../planetary_robot_zoo/zhurong_mars_rover_franka_emika/zhurong.xml"
)
SCENE_PATH = CONFIG["mujoco_scene_path"] or os.path.join(
    CURRENT_DIR, "../../planetary_robot_zoo/zhurong_mars_rover_franka_emika/scene.xml"
)


mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
mj_data = mujoco.MjData(mj_model)

mj_model.opt.timestep = float(CONFIG["simulation_dt"])
# 从 URDF 文件构建机器人模型
pinocchio_robot = pinocchio.RobotWrapper.BuildFromMJCF(MODEL_PATH)
# 为模型创建数据对象，用于存储计算过程中的中间结果
ph_model = pinocchio_robot.model
ph_data = pinocchio_robot.data


@dataclass
class SharedState:
    lock: threading.Lock
    current_qpos: np.ndarray
    desired_ctrl: np.ndarray
    desired_ee: np.ndarray
    actual_ee: np.ndarray
    target_pos: np.ndarray
    rover_wheel_cmd: np.ndarray
    rover_steer_cmd: np.ndarray
    base_pos: np.ndarray
    base_yaw: float


def fmt_3dec(x: np.ndarray) -> str:
    return np.array2string(
        x, formatter={"float_kind": lambda v: f"{v: .3f}"}, separator=" "
    )


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.user_scn.ngeom >= scene.user_scn.maxgeom:
        scene.user_scn.ngeom = 0  # user_scn不会自动清除geom
        # return
    scene.user_scn.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.user_scn.geoms[scene.user_scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.user_scn.geoms[scene.user_scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1,
        point2,
    )


class CustomViewer:
    def __init__(self, mj_model, mj_data):
        self.viewer_handle = mujoco.viewer.launch_passive(mj_model, mj_data)
        # self.pos = 0.0001

        # # 找到末端执行器的 body id
        # self.end_effector_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        # print(f"End effector ID: {self.end_effector_id}")
        # if self.end_effector_id == -1:
        #     print("Warning: Could not find the end effector with the given name.")

        # 初始关节角度
        qpos_len = int(CONFIG["qpos_len"])
        self.initial_mj_q = mj_data.qpos[:qpos_len].copy()
        self.cur_mj_q = mj_data.qpos[:qpos_len].copy()
        log_and_print(f"Initial joint positions: {self.initial_mj_q}")
        theta_x = 0.0  # 旋转角度，单位为弧度
        theta_y = np.pi / 2  # 旋转角度，单位为弧度
        self.R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)],
            ]
        )
        self.R_y = np.array(
            [
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)],
            ]
        )
        self.R = self.R_y @ self.R_x

        self.new_mj_q = self.initial_mj_q

    def is_running(self):
        return self.viewer_handle.is_running()

    def sync(self):
        self.viewer_handle.sync()

    def draw_geom(self, type, size, pos, mat, rgba):
        self.viewer_handle.user_scn.ngeom += 1
        geom = self.viewer_handle.user_scn.geoms[self.viewer_handle.user_scn.ngeom - 1]
        mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)

    def draw_line(self, start, end, width, rgba):
        self.viewer_handle.user_scn.ngeom += 1
        geom = self.viewer_handle.user_scn.geoms[self.viewer_handle.user_scn.ngeom - 1]
        size = [0.0, 0.0, 0.0]
        pos = [0, 0, 0]
        mat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)

    def draw_arrow(self, start, end, width, rgba):
        self.viewer_handle.user_scn.ngeom += 1
        geom = self.viewer_handle.user_scn.geoms[self.viewer_handle.user_scn.ngeom - 1]
        size = [0.0, 0.0, 0.0]
        pos = [0, 0, 0]
        mat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)

    @property
    def cam(self):
        return self.viewer_handle.cam

    @property
    def viewport(self):
        return self.viewer_handle.viewport

    def _init_ids(self):
        mujoco.mj_forward(mj_model, mj_data)
        target_geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, CONFIG["target_sphere_geom"]
        )
        base_body_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            str(CONFIG.get("arm_base_body", "base_link")),
        )
        return target_geom_id, base_body_id

    def _init_trail_state(self):
        trail_state = {
            "trail_enabled": bool(CONFIG.get("trail_enabled", True)),
            "trail_stride_steps": max(1, int(CONFIG.get("trail_stride_steps", 5))),
            "trail_max_points": max(0, int(CONFIG.get("trail_max_points", 1500))),
            "trail_radius": float(CONFIG.get("trail_radius", 0.01)),
            "trail_rgba": np.array(
                CONFIG.get("trail_rgba", [0.1, 0.7, 1.0, 1.0]), dtype=float
            ),
            "trail_step_counter": 0,
            "target_traj": [],
            "ee_trail_enabled": bool(CONFIG.get("ee_trail_enabled", True)),
            "ee_trail_stride_steps": max(
                1, int(CONFIG.get("ee_trail_stride_steps", 5))
            ),
            "ee_trail_max_points": max(0, int(CONFIG.get("ee_trail_max_points", 1500))),
            "ee_desired_radius": float(CONFIG.get("ee_desired_trail_radius", 0.01)),
            "ee_actual_radius": float(CONFIG.get("ee_actual_trail_radius", 0.01)),
            "ee_desired_rgba": np.array(
                CONFIG.get("ee_desired_trail_rgba", [0.2, 0.9, 0.2, 1.0]),
                dtype=float,
            ),
            "ee_actual_rgba": np.array(
                CONFIG.get("ee_actual_trail_rgba", [1.0, 0.5, 0.1, 1.0]),
                dtype=float,
            ),
            "ee_step_counter": 0,
            "ee_desired_traj": [],
            "ee_actual_traj": [],
        }
        return trail_state

    def _read_thread_config(self):
        sim_hz = float(CONFIG.get("sim_hz", 1.0 / mj_model.opt.timestep))
        control_hz = float(
            CONFIG.get(
                "control_hz",
                1.0 / (mj_model.opt.timestep * float(CONFIG["control_decimation"])),
            )
        )
        view_hz = float(CONFIG.get("view_hz", 60.0))
        vision_hz = float(CONFIG.get("vision_hz", 0.0))
        stat_print_every = int(CONFIG.get("thread_stat_print_every", 200))
        overrun_warn = bool(CONFIG.get("thread_overrun_warn", True))
        vision_use_process = bool(CONFIG.get("vision_use_process", False))
        vision_camera_name = str(CONFIG.get("vision_camera_name", ""))
        vision_width = int(CONFIG.get("vision_width", 640))
        vision_height = int(CONFIG.get("vision_height", 480))
        vision_save_enable = bool(CONFIG.get("vision_save_enable", False))
        vision_save_every = int(CONFIG.get("vision_save_every", 10))
        vision_save_dir = str(CONFIG.get("vision_save_dir", "logs/vision"))
        if not os.path.isabs(vision_save_dir):
            vision_save_dir = os.path.join(PROJECT_ROOT, vision_save_dir)
        if vision_save_enable:
            os.makedirs(vision_save_dir, exist_ok=True)

        return {
            "sim_hz": sim_hz,
            "control_hz": control_hz,
            "view_hz": view_hz,
            "vision_hz": vision_hz,
            "stat_print_every": stat_print_every,
            "overrun_warn": overrun_warn,
            "vision_use_process": vision_use_process,
            "vision_camera_name": vision_camera_name,
            "vision_width": vision_width,
            "vision_height": vision_height,
            "vision_save_enable": vision_save_enable,
            "vision_save_every": vision_save_every,
            "vision_save_dir": vision_save_dir,
        }

    def _init_controllers(self):
        arm_kin = ArmKinematics(ph_model, ph_data, CONFIG)
        arm_ctrl = ArmController(arm_kin, CONFIG, self.R)
        rover_ctrl = RoverController(CONFIG)
        task = TaskCoordinator(arm_ctrl, rover_ctrl, CONFIG)
        return arm_kin, arm_ctrl, rover_ctrl, task

    def _init_shared_state(self, qpos_len, arm_len, base_body_id, arm_kin):
        if base_body_id >= 0:
            base_pos_init = mj_data.xpos[base_body_id].copy()
            base_rot = mj_data.xmat[base_body_id].reshape(3, 3)
            base_yaw_init = float(np.arctan2(base_rot[1, 0], base_rot[0, 0]))
        else:
            base_pos_init = np.zeros(3, dtype=float)
            base_yaw_init = 0.0

        init_qpos = mj_data.qpos[:qpos_len].copy()
        init_actual_ee = arm_kin.ee_position(arm_kin.mujoco_q_to_pinocchio_q(init_qpos))

        shared = SharedState(
            lock=threading.Lock(),
            current_qpos=init_qpos,
            desired_ctrl=np.zeros(arm_len, dtype=float),
            desired_ee=np.array(CONFIG["target_start"], dtype=float),
            actual_ee=init_actual_ee,
            target_pos=np.array(CONFIG["target_start"], dtype=float),
            rover_wheel_cmd=np.zeros(6, dtype=float),
            rover_steer_cmd=np.zeros(6, dtype=float),
            base_pos=base_pos_init,
            base_yaw=base_yaw_init,
        )
        return shared

    def _init_ros_pub(self):
        if not bool(CONFIG.get("ros_joint_pub_enable", False)):
            return None
        arm_dofs = int(CONFIG["arm_end"]) - int(CONFIG["arm_start"])
        joint_names = CONFIG.get("arm_joint_names")
        if not joint_names:
            joint_names = [f"arm_joint_{i}" for i in range(arm_dofs)]
        return RosJointStatePublisher(
            node_name=str(CONFIG.get("ros_joint_node_name", "arm_joint_state_pub")),
            target_topic=str(CONFIG.get("ros_joint_target_topic", "/arm/joint_target")),
            actual_topic=str(CONFIG.get("ros_joint_actual_topic", "/arm/joint_actual")),
            joint_names=joint_names,
            publish_hz=float(CONFIG.get("ros_joint_pub_hz", 0.0)),
            queue_size=int(CONFIG.get("ros_joint_queue_size", 10)),
        )

    def _build_control_step(
        self,
        shared,
        rover_ctrl,
        task,
        arm_ctrl,
        control_hz,
        control_print_every,
        ros_pub,
        control_state,
    ):
        def control_step() -> None:
            if control_state["stop_event"].is_set():
                return

            t = time.perf_counter() - control_state["start_time"]
            desired_target = trajectory_point(CONFIG, t)
            with shared.lock:
                current_qpos = shared.current_qpos.copy()
                base_pos = shared.base_pos.copy()
                base_yaw = shared.base_yaw

            rover_ctrl.update_base_pose(base_pos, base_yaw)
            with warn_if_overrun(1.0 / max(control_hz, 1e-6), label="control_thread"):
                control_state["fps"].tick()
                task.update(desired_target, current_qpos)
                new_mj_q, desired_ee = arm_ctrl.step(current_qpos)

            desired_ctrl = new_mj_q[CONFIG["arm_start"] : CONFIG["arm_end"]]
            rover_ctrl.step()
            wheel_cmd, steer_cmd = rover_ctrl.get_cmd()

            with shared.lock:
                shared.desired_ctrl[:] = desired_ctrl
                shared.desired_ee[:] = desired_ee
                shared.target_pos[:] = desired_target
                shared.rover_wheel_cmd[:] = wheel_cmd
                shared.rover_steer_cmd[:] = steer_cmd

            cur_arr = current_qpos[CONFIG["arm_start"] : CONFIG["arm_end"]]
            if ros_pub is not None:
                ros_pub.publish(desired_ctrl, cur_arr)

            control_state["counter"] += 1
            if control_state["counter"] % control_print_every == 0:
                diff = desired_ctrl - cur_arr
                log_and_print(f"ctrl[0:7]: {fmt_3dec(desired_ctrl)}")
                log_and_print(f"cur [0:7]: {fmt_3dec(cur_arr)}")
                log_and_print(
                    "diff[0:7]: "
                    f"{fmt_3dec(diff)}, total_diff = {np.linalg.norm(diff):.4f}"
                )

        return control_step

    def _build_sim_step(
        self,
        shared,
        sim_lock,
        target_geom_id,
        base_body_id,
        arm_kin,
        qpos_len,
        ctrl_start,
        ctrl_end,
        delay_s,
        ramp_s,
        max_delta,
        sim_state,
    ):
        def sim_step() -> None:
            if sim_state["stop_event"].is_set():
                return

            elapsed = time.perf_counter() - sim_state["start_time"]
            with shared.lock:
                desired_ctrl = shared.desired_ctrl.copy()
                rover_wheel_cmd = shared.rover_wheel_cmd.copy()
                rover_steer_cmd = shared.rover_steer_cmd.copy()
                target_pos = shared.target_pos.copy()

            with sim_lock:
                mj_model.geom_pos[target_geom_id] = target_pos

                if elapsed < delay_s:
                    mj_data.ctrl[ctrl_start:ctrl_end] = mj_data.qpos[
                        CONFIG["arm_start"] : CONFIG["arm_end"]
                    ]
                else:
                    cur_ctrl = mj_data.ctrl[ctrl_start:ctrl_end].copy()
                    if ramp_s > 0.0:
                        ramp_alpha = min((elapsed - delay_s) / ramp_s, 1.0)
                        desired_ctrl = cur_ctrl + ramp_alpha * (desired_ctrl - cur_ctrl)

                    if max_delta > 0.0:
                        delta = np.clip(desired_ctrl - cur_ctrl, -max_delta, max_delta)
                        mj_data.ctrl[ctrl_start:ctrl_end] = cur_ctrl + delta
                    else:
                        mj_data.ctrl[ctrl_start:ctrl_end] = desired_ctrl

                mj_data.ctrl[
                    int(CONFIG["base_steer_ctrl_start"]) : int(
                        CONFIG["base_steer_ctrl_start"]
                    )
                    + 6
                ] = rover_steer_cmd
                mj_data.ctrl[
                    int(CONFIG["base_wheel_ctrl_start"]) : int(
                        CONFIG["base_wheel_ctrl_start"]
                    )
                    + 6
                ] = rover_wheel_cmd

                mujoco.mj_step(mj_model, mj_data)

                qpos_copy = mj_data.qpos[:qpos_len].copy()
                actual_ph_q = arm_kin.mujoco_q_to_pinocchio_q(qpos_copy)
                actual_ee = arm_kin.ee_position(actual_ph_q)

                if base_body_id >= 0:
                    base_pos = mj_data.xpos[base_body_id].copy()
                    rot = mj_data.xmat[base_body_id].reshape(3, 3)
                    base_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
                else:
                    base_pos = np.zeros(3, dtype=float)
                    base_yaw = 0.0

            with shared.lock:
                shared.current_qpos[:] = qpos_copy
                shared.actual_ee[:] = actual_ee
                shared.base_pos[:] = base_pos
                shared.base_yaw = base_yaw

        return sim_step

    def _build_vision_step(self, sim_lock, vision_state, vision_cfg):
        def vision_step() -> None:
            if vision_state["stop_event"].is_set():
                return
            if vision_state["renderer"] is None:
                vision_state["renderer"] = mujoco.Renderer(
                    mj_model,
                    width=vision_cfg["vision_width"],
                    height=vision_cfg["vision_height"],
                )
            with sim_lock:
                if vision_cfg["vision_camera_name"]:
                    vision_state["renderer"].update_scene(
                        mj_data, camera=vision_cfg["vision_camera_name"]
                    )
                else:
                    vision_state["renderer"].update_scene(mj_data)
                frame = vision_state["renderer"].render()

            vision_state["frame_idx"] += 1
            if (
                vision_cfg["vision_save_enable"]
                and vision_cfg["vision_save_every"] > 0
                and vision_state["frame_idx"] % vision_cfg["vision_save_every"] == 0
            ):
                file_path = os.path.join(
                    vision_cfg["vision_save_dir"],
                    f"frame_{vision_state['frame_idx']:06d}.png",
                )
                try:
                    import imageio.v3 as iio  # type: ignore

                    iio.imwrite(file_path, frame)
                except Exception:
                    try:
                        from PIL import Image  # type: ignore

                        Image.fromarray(frame).save(file_path)
                    except Exception:
                        if not vision_state["warned_no_writer"]:
                            log_and_print(
                                "[vision] image writer unavailable, install imageio or pillow"
                            )
                            vision_state["warned_no_writer"] = True

        return vision_step

    def _run_view_loop(
        self,
        shared,
        sim_lock,
        trail_state,
        view_hz,
        stat_print_every,
        stop_event,
    ):
        view_period = 1.0 / max(view_hz, 1e-6)
        next_view = time.perf_counter()
        view_fps = FPSCounter(print_every=0, label="view")
        view_counter = 0
        view_overruns = 0

        while self.is_running():
            now = time.perf_counter()
            if now < next_view:
                time.sleep(max(0.0, next_view - now))
                continue
            next_view += view_period
            loop_start = time.perf_counter()

            with shared.lock:
                target_pos = shared.target_pos.copy()
                desired_ee = shared.desired_ee.copy()
                actual_ee = shared.actual_ee.copy()

            if trail_state["trail_enabled"] and trail_state["trail_max_points"] > 0:
                trail_state["trail_step_counter"] += 1
                if (
                    trail_state["trail_step_counter"]
                    % trail_state["trail_stride_steps"]
                    == 0
                ):
                    trail_state["target_traj"].append(target_pos.copy())
                    if (
                        len(trail_state["target_traj"])
                        > trail_state["trail_max_points"]
                    ):
                        trail_state["target_traj"] = trail_state["target_traj"][
                            -trail_state["trail_max_points"] :
                        ]

            if (
                trail_state["ee_trail_enabled"]
                and trail_state["ee_trail_max_points"] > 0
            ):
                trail_state["ee_step_counter"] += 1
                if (
                    trail_state["ee_step_counter"]
                    % trail_state["ee_trail_stride_steps"]
                    == 0
                ):
                    trail_state["ee_desired_traj"].append(desired_ee.copy())
                    trail_state["ee_actual_traj"].append(actual_ee.copy())
                    if (
                        len(trail_state["ee_desired_traj"])
                        > trail_state["ee_trail_max_points"]
                    ):
                        trail_state["ee_desired_traj"] = trail_state["ee_desired_traj"][
                            -trail_state["ee_trail_max_points"] :
                        ]
                    if (
                        len(trail_state["ee_actual_traj"])
                        > trail_state["ee_trail_max_points"]
                    ):
                        trail_state["ee_actual_traj"] = trail_state["ee_actual_traj"][
                            -trail_state["ee_trail_max_points"] :
                        ]

            if (
                trail_state["trail_enabled"] and len(trail_state["target_traj"]) > 1
            ) or (
                trail_state["ee_trail_enabled"]
                and (
                    len(trail_state["ee_desired_traj"]) > 1
                    or len(trail_state["ee_actual_traj"]) > 1
                )
            ):
                self.viewer_handle.user_scn.ngeom = 0

            if trail_state["trail_enabled"] and len(trail_state["target_traj"]) > 1:
                for i in range(len(trail_state["target_traj"]) - 1):
                    add_visual_capsule(
                        self.viewer_handle,
                        trail_state["target_traj"][i],
                        trail_state["target_traj"][i + 1],
                        trail_state["trail_radius"],
                        trail_state["trail_rgba"],
                    )

            if (
                trail_state["ee_trail_enabled"]
                and len(trail_state["ee_desired_traj"]) > 1
            ):
                for i in range(len(trail_state["ee_desired_traj"]) - 1):
                    add_visual_capsule(
                        self.viewer_handle,
                        trail_state["ee_desired_traj"][i],
                        trail_state["ee_desired_traj"][i + 1],
                        trail_state["ee_desired_radius"],
                        trail_state["ee_desired_rgba"],
                    )

            if (
                trail_state["ee_trail_enabled"]
                and len(trail_state["ee_actual_traj"]) > 1
            ):
                for i in range(len(trail_state["ee_actual_traj"]) - 1):
                    add_visual_capsule(
                        self.viewer_handle,
                        trail_state["ee_actual_traj"][i],
                        trail_state["ee_actual_traj"][i + 1],
                        trail_state["ee_actual_radius"],
                        trail_state["ee_actual_rgba"],
                    )

            with sim_lock:
                self.sync()

            loop_elapsed = time.perf_counter() - loop_start
            view_fps.tick()
            view_counter += 1
            if view_period > 0 and loop_elapsed > view_period:
                view_overruns += 1
            if stat_print_every > 0 and view_counter % stat_print_every == 0:
                actual = view_fps.fps()
                if actual is None:
                    actual_str = "n/a"
                else:
                    actual_str = f"{actual:.1f}"
                log_and_print(
                    f"[thread_stats] view: actual {actual_str} Hz / "
                    f"target {view_hz:.1f} Hz / overruns {view_overruns}"
                )

        stop_event.set()

    def run_loop(self):
        target_geom_id, base_body_id = self._init_ids()
        qpos_len = int(CONFIG["qpos_len"])
        ctrl_start = int(CONFIG["ctrl_start"])
        ctrl_end = int(CONFIG["ctrl_end"])
        arm_len = int(CONFIG["arm_end"]) - int(CONFIG["arm_start"])

        trail_state = self._init_trail_state()
        thread_cfg = self._read_thread_config()

        delay_s = float(CONFIG["control_start_delay"])
        ramp_s = float(CONFIG["control_ramp_time"])
        max_delta = float(CONFIG["control_max_delta"])
        control_print_every = int(CONFIG["control_print_every"])

        arm_kin, arm_ctrl, rover_ctrl, task = self._init_controllers()
        shared = self._init_shared_state(qpos_len, arm_len, base_body_id, arm_kin)

        sim_lock = threading.Lock()
        stop_event = threading.Event()

        control_state = {
            "counter": 0,
            "fps": FPSCounter(print_every=control_print_every, label="control_thread"),
            "start_time": time.perf_counter(),
            "stop_event": stop_event,
        }
        sim_state = {
            "start_time": time.perf_counter(),
            "stop_event": stop_event,
        }
        vision_state = {
            "renderer": None,
            "frame_idx": 0,
            "warned_no_writer": False,
            "stop_event": stop_event,
        }

        ros_pub = self._init_ros_pub()

        if thread_cfg["vision_use_process"]:
            log_and_print(
                "[vision] MuJoCo camera uses shared mj_data; forcing vision_use_process=False"
            )
            thread_cfg["vision_use_process"] = False

        control_step = self._build_control_step(
            shared,
            rover_ctrl,
            task,
            arm_ctrl,
            thread_cfg["control_hz"],
            control_print_every,
            ros_pub,
            control_state,
        )
        sim_step = self._build_sim_step(
            shared,
            sim_lock,
            target_geom_id,
            base_body_id,
            arm_kin,
            qpos_len,
            ctrl_start,
            ctrl_end,
            delay_s,
            ramp_s,
            max_delta,
            sim_state,
        )
        vision_step = self._build_vision_step(sim_lock, vision_state, thread_cfg)

        pool = ThreadPool()
        pool.add_task(
            "control",
            control_step,
            thread_cfg["control_hz"],
            print_every=thread_cfg["stat_print_every"],
            warn_overrun=thread_cfg["overrun_warn"],
        )
        pool.add_task(
            "sim",
            sim_step,
            thread_cfg["sim_hz"],
            print_every=thread_cfg["stat_print_every"],
            warn_overrun=thread_cfg["overrun_warn"],
        )
        if thread_cfg["vision_use_process"]:
            pool.add_process_task(
                "vision",
                vision_step,
                thread_cfg["vision_hz"],
                print_every=thread_cfg["stat_print_every"],
                warn_overrun=thread_cfg["overrun_warn"],
            )
        else:
            pool.add_task(
                "vision",
                vision_step,
                thread_cfg["vision_hz"],
                print_every=thread_cfg["stat_print_every"],
                warn_overrun=thread_cfg["overrun_warn"],
            )
        pool.start()

        try:
            self._run_view_loop(
                shared,
                sim_lock,
                trail_state,
                thread_cfg["view_hz"],
                thread_cfg["stat_print_every"],
                stop_event,
            )
        finally:
            stop_event.set()
            pool.stop()


if __name__ == "__main__":
    log_path = init_run_logger(LOG_ROOT, "arm_end_control")
    log_and_print(f"[logger] logging to {log_path}")
    # print_pinocchio_info(ph_model, log_and_print)
    viewer = CustomViewer(mj_model, mj_data)
    viewer.cam.distance = float(CONFIG["cam_distance"])
    viewer.cam.azimuth = float(CONFIG["cam_azimuth"])
    viewer.cam.elevation = float(CONFIG["cam_elevation"])
    viewer.run_loop()
