import os
import sys

# 当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（根据你的层级向上走几级）
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..")
)  # 自行调整级数
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import time
import multiprocessing as mp
import ctypes
import mujoco
import numpy as np
import mujoco.viewer
from scipy.optimize import minimize
import os
import numpy as np
import pinocchio
import yaml
from numpy.linalg import norm, solve
from typing import List
from mars_rover_arm_control.utils.time_analysis import timeit
from mars_rover_arm_control.utils.print_control import control_print
from mars_rover_arm_control.utils.fps_counter import FPSCounter
from mars_rover_arm_control.utils.ros_joint_publisher import RosJointStatePublisher

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configs/config.yaml")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


CONFIG = load_config(CONFIG_FILE_PATH)

MODEL_PATH = CONFIG["pinocchio_model_path"] or os.path.join(
    ROOT_DIR, "../../planetary_robot_zoo/zhurong_mars_rover_franka_emika/zhurong.xml"
)
SCENE_PATH = CONFIG["mujoco_scene_path"] or os.path.join(
    ROOT_DIR, "../../planetary_robot_zoo/zhurong_mars_rover_franka_emika/scene.xml"
)


mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
mj_data = mujoco.MjData(mj_model)

mj_model.opt.timestep = float(CONFIG["simulation_dt"])
# 从 URDF 文件构建机器人模型
pinocchio_robot = pinocchio.RobotWrapper.BuildFromMJCF(MODEL_PATH)
# 为模型创建数据对象，用于存储计算过程中的中间结果
ph_model = pinocchio_robot.model
ph_data = pinocchio_robot.data

print("-" * 80)
print("pinocchio joints related info")
for i, jn in enumerate(ph_model.names):
    print(i, jn, ph_model.joints[i].nq, ph_model.joints[i].nv)
print("-" * 80)


def mujoco_q_to_pinocchio_q(mujoco_q):
    # Convert Mujoco joint positions to Pinocchio format
    # This is a placeholder - you need to implement the actual conversion based on your model
    # MJ: [x, y, z, qw, qx, qy, qz]
    # Pin: [x, y, z, qx, qy, qz, qw]
    pinocchio_q = mujoco_q.copy()
    pinocchio_q[3:7] = [
        mujoco_q[4],
        mujoco_q[5],
        mujoco_q[6],
        mujoco_q[3],
    ]  # Convert (qw, qx, qy, qz) to (qx, qy, qz, qw)
    return pinocchio_q


def pinocchio_q_to_mujoco_q(pinocchio_q):
    # Convert Pinocchio joint positions to Mujoco format
    # This is a placeholder - you need to implement the actual conversion based on your model
    # MJ: [x, y, z, qw, qx, qy, qz]
    # Pin: [x, y, z, qx, qy, qz, qw]
    mujoco_q = pinocchio_q.copy()
    mujoco_q[3:7] = [
        pinocchio_q[6],
        pinocchio_q[3],
        pinocchio_q[4],
        pinocchio_q[5],
    ]  # Convert (qx, qy, qz, qw) to (qw, qx, qy, qz)
    return mujoco_q


# @timeit(unit="ms")
def inverse_arm_kinematics(current_q, target_dir, target_pos):
    arm_idx = np.asarray(range(CONFIG["arm_start"], CONFIG["arm_end"], 1))
    # 指定要控制的关节 ID
    JOINT_ID = int(CONFIG["ik_joint_id"])  # TODO: 根据模型中的关节名称获取对应的关节 ID
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = float(CONFIG["ik_eps"])
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = int(CONFIG["ik_max_iters"])
    # 定义积分步长，用于更新关节角度
    DT = float(CONFIG["ik_dt"])
    # 定义阻尼因子，用于避免矩阵奇异
    damp = float(CONFIG["ik_damp"])

    # 初始化迭代次数为 0
    i = 0
    while True:
        # 进行正运动学计算，得到当前关节角度下机器人各关节的位置和姿态
        pinocchio.forwardKinematics(ph_model, ph_data, q)
        # 计算目标位姿到当前位姿之间的变换
        iMd = ph_data.oMi[JOINT_ID].actInv(oMdes)
        # 通过李群对数映射将变换矩阵转换为 6 维误差向量（包含位置误差和方向误差），用于量化当前位姿与目标位姿的差异
        err = pinocchio.log(iMd).vector

        # 判断误差是否小于收敛阈值，如果是则认为算法收敛
        if norm(err) < eps:
            success = True
            break
        # 判断迭代次数是否超过最大迭代次数，如果是则认为算法未收敛
        if i >= IT_MAX:
            success = False
            # print(f"inverse_kinematics failed, err = {norm(err)}")
            break

        # 计算当前关节角度下的雅可比矩阵，关节速度与末端速度的映射关系
        J_full = pinocchio.computeJointJacobian(ph_model, ph_data, q, JOINT_ID)
        # 对雅可比矩阵进行变换，转换到李代数空间，以匹配误差向量的坐标系，同时取反以调整误差方向
        J = -pinocchio.Jlog6(iMd.inverse()) @ J_full

        J_arm = J[:, arm_idx]
        v_arm = -J_arm.T @ solve(J_arm @ J_arm.T + damp * np.eye(6), err)

        v = np.zeros(ph_model.nv)
        v[arm_idx] = v_arm

        # 根据关节速度更新关节角度
        q = pinocchio.integrate(ph_model, q, v * DT)

        # # 每迭代 300 次打印一次当前的误差信息
        # if not i % 1000:
        #     print(f"{i}: error = {err.T}")
        # 迭代次数加 1
        i += 1
    # print(f"IK iters: {i}, success={success}")

    # 根据算法是否收敛输出相应的信息
    # if success:
    #     print("Convergence achieved!")
    #     for name, oMi in zip(ph_model.names, ph_data.oMi):
    #         print(
    #             "{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)
    #         )
    # else:
    #     print(
    #         "\n"
    #         "Warning: the iterative algorithm has not reached convergence "
    #         "to the desired precision"
    #     )
    # if success:
    #     print(f"time: {time.time()} Convergence achieved!")
    # else:
    #     print(f"time: {time.time()} No!!!")

    # 打印最终的关节角度和误差向量
    # print(f"\nresult: {q.flatten().tolist()}")
    # print(f"\nfinal error: {err.T}")
    # 返回最终的关节角度向量（以列表形式）
    return q.flatten().tolist(), success


def inverse_arm_kinematics_bounded(current_q, target_dir, target_pos):
    arm_idx = np.asarray(range(CONFIG["arm_start"], CONFIG["arm_end"], 1))
    joint_id = int(CONFIG["ik_joint_id"])
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    q0 = np.array(current_q, dtype=float)
    x0 = q0[arm_idx].copy()

    lower = ph_model.lowerPositionLimit[arm_idx]
    upper = ph_model.upperPositionLimit[arm_idx]
    bounds = []
    for lo, hi in zip(lower, upper):
        lo_b = None if np.isneginf(lo) else float(lo)
        hi_b = None if np.isposinf(hi) else float(hi)
        bounds.append((lo_b, hi_b))

    def cost(x):
        q = q0.copy()
        q[arm_idx] = x
        pinocchio.forwardKinematics(ph_model, ph_data, q)
        iMd = ph_data.oMi[joint_id].actInv(oMdes)
        err = pinocchio.log(iMd).vector
        return float(err.T @ err)

    res = minimize(
        cost,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": int(CONFIG["ik_bounds_max_iters"])},
    )

    q = q0.copy()
    q[arm_idx] = res.x
    pinocchio.forwardKinematics(ph_model, ph_data, q)
    iMd = ph_data.oMi[joint_id].actInv(oMdes)
    err = pinocchio.log(iMd).vector
    success = bool(res.success) and norm(err) < float(CONFIG["ik_eps"])
    return q.flatten().tolist(), success


def inverse_arm_kinematics_bounded_retry(current_q, target_dir, target_pos):
    arm_idx = np.asarray(range(CONFIG["arm_start"], CONFIG["arm_end"], 1))
    joint_id = int(CONFIG["ik_joint_id"])
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    q0 = np.array(current_q, dtype=float)
    x0 = q0[arm_idx].copy()

    lower = ph_model.lowerPositionLimit[arm_idx]
    upper = ph_model.upperPositionLimit[arm_idx]
    bounds = []
    for lo, hi in zip(lower, upper):
        lo_b = None if np.isneginf(lo) else float(lo)
        hi_b = None if np.isposinf(hi) else float(hi)
        bounds.append((lo_b, hi_b))

    def compute_err(q):
        pinocchio.forwardKinematics(ph_model, ph_data, q)
        iMd = ph_data.oMi[joint_id].actInv(oMdes)
        return pinocchio.log(iMd).vector

    def cost(x):
        q = q0.copy()
        q[arm_idx] = x
        err = compute_err(q)
        return float(err.T @ err)

    def solve_from(x_init):
        res = minimize(
            cost,
            x_init,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(CONFIG["ik_bounds_max_iters"])},
        )
        q = q0.copy()
        q[arm_idx] = res.x
        err = compute_err(q)
        return res, q, err

    base_res, base_q, base_err = solve_from(x0)
    best_q = base_q
    best_err = base_err
    best_success = bool(base_res.success) and norm(base_err) < float(CONFIG["ik_eps"])

    if norm(base_err) <= float(CONFIG["ik_retry_err_thresh"]):
        return best_q.flatten().tolist(), best_success

    rng = np.random.default_rng()
    noise_scale = float(CONFIG["ik_retry_noise_scale"])
    attempts = int(CONFIG["ik_retry_attempts"])

    for _ in range(max(0, attempts)):
        noise = rng.normal(0.0, noise_scale, size=x0.shape)
        x_init = x0 + noise
        # Clamp to bounds when finite.
        for i, (lo_b, hi_b) in enumerate(bounds):
            if lo_b is not None and x_init[i] < lo_b:
                x_init[i] = lo_b
            if hi_b is not None and x_init[i] > hi_b:
                x_init[i] = hi_b

        res, q, err = solve_from(x_init)
        if norm(err) < norm(best_err):
            best_q = q
            best_err = err
            best_success = bool(res.success) and norm(err) < float(CONFIG["ik_eps"])

    return best_q.flatten().tolist(), best_success


@timeit(unit="ms")
def inverse_kinematics(current_q, target_dir, target_pos):

    # 指定要控制的关节 ID
    JOINT_ID = int(CONFIG["ik_joint_id"])  # TODO: 根据模型中的关节名称获取对应的关节 ID
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = float(CONFIG["ik_eps"])
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = int(CONFIG["ik_max_iters"])
    # 定义积分步长，用于更新关节角度
    DT = float(CONFIG["ik_dt"])
    # 定义阻尼因子，用于避免矩阵奇异
    damp = float(CONFIG["ik_damp"])

    # 初始化迭代次数为 0
    i = 0
    while True:
        # 进行正运动学计算，得到当前关节角度下机器人各关节的位置和姿态
        pinocchio.forwardKinematics(ph_model, ph_data, q)
        # 计算目标位姿到当前位姿之间的变换
        iMd = ph_data.oMi[JOINT_ID].actInv(oMdes)
        # 通过李群对数映射将变换矩阵转换为 6 维误差向量（包含位置误差和方向误差），用于量化当前位姿与目标位姿的差异
        err = pinocchio.log(iMd).vector

        # 判断误差是否小于收敛阈值，如果是则认为算法收敛
        if norm(err) < eps:
            success = True
            break
        # 判断迭代次数是否超过最大迭代次数，如果是则认为算法未收敛
        if i >= IT_MAX:
            success = False
            break

        # 计算当前关节角度下的雅可比矩阵，关节速度与末端速度的映射关系
        J = pinocchio.computeJointJacobian(ph_model, ph_data, q, JOINT_ID)
        # 对雅可比矩阵进行变换，转换到李代数空间，以匹配误差向量的坐标系，同时取反以调整误差方向
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        # 使用阻尼最小二乘法求解关节速度
        # v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        JJt = J.dot(J.T) + damp * np.eye(6)
        v = -J.T @ solve(JJt, err)

        # 锁死前 7 个关节：不允许它们动
        v[: int(CONFIG["ik_lock_front_dofs"])] = 0.0
        # 根据关节速度更新关节角度
        q = pinocchio.integrate(ph_model, q, v * DT)

        # # 每迭代 300 次打印一次当前的误差信息
        # if not i % 1000:
        #     print(f"{i}: error = {err.T}")
        # 迭代次数加 1
        i += 1
    print(f"IK iters: {i}, success={success}")

    # 根据算法是否收敛输出相应的信息
    # if success:
    #     print("Convergence achieved!")
    #     for name, oMi in zip(ph_model.names, ph_data.oMi):
    #         print(
    #             "{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)
    #         )
    # else:
    #     print(
    #         "\n"
    #         "Warning: the iterative algorithm has not reached convergence "
    #         "to the desired precision"
    #     )
    # if success:
    #     print(f"time: {time.time()} Convergence achieved!")
    # else:
    #     print(f"time: {time.time()} No!!!")

    # 打印最终的关节角度和误差向量
    # print(f"\nresult: {q.flatten().tolist()}")
    # print(f"\nfinal error: {err.T}")
    # 返回最终的关节角度向量（以列表形式）
    return q.flatten().tolist()


def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def fmt_3dec(x: np.ndarray) -> str:
    return np.array2string(
        x, formatter={"float_kind": lambda v: f"{v: .3f}"}, separator=" "
    )


def _trajectory_point(t: float) -> np.ndarray:
    traj_type = str(CONFIG.get("trajectory_type", "line_y")).lower()
    center = np.array(
        CONFIG.get("trajectory_center", CONFIG["target_start"]), dtype=float
    )
    scale = np.array(CONFIG.get("trajectory_scale", [0.15, 0.25, 0.0]), dtype=float)
    period = float(CONFIG.get("trajectory_period", 8.0))
    w = 2.0 * np.pi / max(period, 1e-6)

    if traj_type == "figure8":
        # Lissajous style figure-eight in XY plane.
        x = scale[0] * np.sin(w * t)
        y = scale[1] * 1.6 * np.cos(w * t)
        z = scale[2] * np.sin(2.0 * w * t)
        return center + np.array([x, y, z], dtype=float)

    if traj_type == "heart":
        # Classic parametric heart curve in XY plane, scaled down.
        s = w * t
        x = scale[0] * np.cos(s)
        y = scale[1] * 2.0 * (np.sin(s) ** 3)
        z = (
            scale[2]
            / 8.0
            * (
                13.0 * np.cos(s)
                - 5.0 * np.cos(2.0 * s)
                - 2.0 * np.cos(3.0 * s)
                - np.cos(4.0 * s)
            )
        )
        return center + np.array([x, y, z], dtype=float)

    if traj_type == "circle":
        s = w * t
        x = scale[0] * np.cos(s)
        y = scale[1] * np.cos(s)
        z = scale[2] * np.sin(s)
        return center + np.array([x, y, z], dtype=float)

    # Default: original line sweep in Y with fixed X/Z from target_start.
    y_min = float(CONFIG["target_y_min"])
    y_max = float(CONFIG["target_y_max"])
    y_speed = float(CONFIG["target_y_speed"])
    y_span = max(y_max - y_min, 1e-6)
    # Triangle wave 0..1..0
    phase = (y_speed * t) / y_span
    tri = 2.0 * np.abs(phase - np.floor(phase + 0.5))
    y = y_min + (y_max - y_min) * tri
    return np.array([center[0], y, center[2]], dtype=float)


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
        print(f"Initial joint positions: {self.initial_mj_q}")
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

    def _control_worker(
        self,
        qpos_shared,
        ctrl_shared,
        target_shared,
        desired_ee_shared,
        actual_ee_shared,
        run_flag,
        control_dt,
        control_print_every,
    ):
        qpos_arr = np.ctypeslib.as_array(qpos_shared)
        ctrl_arr = np.ctypeslib.as_array(ctrl_shared)
        target_arr = np.ctypeslib.as_array(target_shared)
        desired_ee_arr = np.ctypeslib.as_array(desired_ee_shared)
        actual_ee_arr = np.ctypeslib.as_array(actual_ee_shared)

        target_start = CONFIG["target_start"]
        x = float(target_start[0])
        y = float(target_start[1])
        z = float(target_start[2])
        counter = 0

        delay_s = float(CONFIG["control_start_delay"])
        ramp_s = float(CONFIG["control_ramp_time"])
        max_delta = float(CONFIG["control_max_delta"])
        start_time = time.time()

        last_ik_q = None
        ik_fail_count = 0
        ik_fail_reset_count = int(CONFIG["ik_fail_reset_count"])

        fps = FPSCounter(print_every=control_print_every, label="control_worker")

        ros_pub = None
        if bool(CONFIG.get("ros_joint_pub_enable", False)):
            arm_dofs = int(CONFIG["arm_end"]) - int(CONFIG["arm_start"])
            joint_names = CONFIG.get("arm_joint_names")
            if not joint_names:
                joint_names = [f"arm_joint_{i}" for i in range(arm_dofs)]
            ros_pub = RosJointStatePublisher(
                node_name=str(CONFIG.get("ros_joint_node_name", "arm_joint_state_pub")),
                target_topic=str(
                    CONFIG.get("ros_joint_target_topic", "/arm/joint_target")
                ),
                actual_topic=str(
                    CONFIG.get("ros_joint_actual_topic", "/arm/joint_actual")
                ),
                joint_names=joint_names,
                publish_hz=float(CONFIG.get("ros_joint_pub_hz", 0.0)),
                queue_size=int(CONFIG.get("ros_joint_queue_size", 10)),
            )

        while run_flag.value:
            elapsed = time.time() - start_time
            if elapsed < delay_s:
                ctrl_arr[:] = qpos_arr[CONFIG["arm_start"] : CONFIG["arm_end"]]
                time.sleep(min(control_dt, delay_s - elapsed))
                continue
            counter += 1
            fps.tick()

            x = float(target_arr[0])
            y = float(target_arr[1])
            z = float(target_arr[2])

            cur_mj_q = qpos_arr.copy()
            if last_ik_q is None:
                cur_ph_q = mujoco_q_to_pinocchio_q(cur_mj_q)
            else:
                cur_ph_q = last_ik_q

            if bool(CONFIG["ik_use_bounds"]):
                new_ph_q, ik_success = inverse_arm_kinematics_bounded_retry(
                    cur_ph_q, self.R, [x, y, z]
                )
            else:
                new_ph_q, ik_success = inverse_arm_kinematics(
                    cur_ph_q, self.R, [x, y, z]
                )
            if ik_success:
                last_ik_q = np.array(new_ph_q, dtype=float)
                ik_fail_count = 0
            else:
                ik_fail_count += 1
                if ik_fail_count >= ik_fail_reset_count:
                    last_ik_q = mujoco_q_to_pinocchio_q(cur_mj_q)
                    ik_fail_count = 0

            new_mj_q = pinocchio_q_to_mujoco_q(np.array(new_ph_q, dtype=float))
            desired_ctrl = new_mj_q[CONFIG["arm_start"] : CONFIG["arm_end"]]
            cur_ctrl = ctrl_arr.copy()

            # Update desired and actual end-effector positions for visualization.
            joint_id = int(CONFIG["ik_joint_id"])
            pinocchio.forwardKinematics(ph_model, ph_data, np.array(new_ph_q))
            desired_ee_arr[:] = ph_data.oMi[joint_id].translation
            actual_ph_q = mujoco_q_to_pinocchio_q(cur_mj_q)
            pinocchio.forwardKinematics(ph_model, ph_data, actual_ph_q)
            actual_ee_arr[:] = ph_data.oMi[joint_id].translation

            if ramp_s > 0.0:
                ramp_alpha = min((elapsed - delay_s) / ramp_s, 1.0)
                desired_ctrl = cur_ctrl + ramp_alpha * (desired_ctrl - cur_ctrl)

            if max_delta > 0.0:
                delta = np.clip(desired_ctrl - cur_ctrl, -max_delta, max_delta)
                ctrl_arr[:] = cur_ctrl + delta
            else:
                ctrl_arr[:] = desired_ctrl
            cur_arr = cur_mj_q[CONFIG["arm_start"] : CONFIG["arm_end"]]

            if ros_pub is not None:
                ros_pub.publish(desired_ctrl, cur_arr)

            if counter % control_print_every == 0:
                # print(f"ctrl[0:7]: {fmt_3dec(ctrl_arr)}")
                # print(f"cur [0:7]: {fmt_3dec(cur_arr)}")
                # print(f"diff[0:7]: {fmt_3dec(ctrl_arr - cur_arr)}")
                pass

            time.sleep(control_dt)

    def run_loop(self):
        step_start = time.time()
        mujoco.mj_forward(mj_model, mj_data)
        target_geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, CONFIG["target_sphere_geom"]
        )

        qpos_len = int(CONFIG["qpos_len"])
        ctrl_start = int(CONFIG["ctrl_start"])
        ctrl_end = int(CONFIG["ctrl_end"])
        ctrl_len = ctrl_end - ctrl_start

        qpos_shared = mp.Array(ctypes.c_double, qpos_len, lock=False)
        ctrl_shared = mp.Array(ctypes.c_double, ctrl_len, lock=False)
        target_shared = mp.Array(ctypes.c_double, 3, lock=False)
        desired_ee_shared = mp.Array(ctypes.c_double, 3, lock=False)
        actual_ee_shared = mp.Array(ctypes.c_double, 3, lock=False)
        run_flag = mp.Value(ctypes.c_bool, True)

        qpos_arr = np.ctypeslib.as_array(qpos_shared)
        ctrl_arr = np.ctypeslib.as_array(ctrl_shared)
        target_arr = np.ctypeslib.as_array(target_shared)

        qpos_arr[:] = mj_data.qpos[:qpos_len]
        ctrl_arr[:] = mj_data.ctrl[ctrl_start:ctrl_end]
        target_arr[:] = np.array(CONFIG["target_start"], dtype=float)
        desired_ee_arr = np.ctypeslib.as_array(desired_ee_shared)
        actual_ee_arr = np.ctypeslib.as_array(actual_ee_shared)
        desired_ee_arr[:] = target_arr
        actual_ee_arr[:] = target_arr
        traj_start_time = time.time()

        trail_enabled = bool(CONFIG.get("trail_enabled", True))
        trail_stride_steps = max(1, int(CONFIG.get("trail_stride_steps", 5)))
        trail_max_points = max(0, int(CONFIG.get("trail_max_points", 1500)))
        trail_radius = float(CONFIG.get("trail_radius", 0.01))
        trail_rgba = np.array(
            CONFIG.get("trail_rgba", [0.1, 0.7, 1.0, 1.0]), dtype=float
        )
        trail_step_counter = 0
        target_traj = []

        ee_trail_enabled = bool(CONFIG.get("ee_trail_enabled", True))
        ee_trail_stride_steps = max(1, int(CONFIG.get("ee_trail_stride_steps", 5)))
        ee_trail_max_points = max(0, int(CONFIG.get("ee_trail_max_points", 1500)))
        ee_desired_radius = float(CONFIG.get("ee_desired_trail_radius", 0.01))
        ee_actual_radius = float(CONFIG.get("ee_actual_trail_radius", 0.01))
        ee_desired_rgba = np.array(
            CONFIG.get("ee_desired_trail_rgba", [0.2, 0.9, 0.2, 1.0]), dtype=float
        )
        ee_actual_rgba = np.array(
            CONFIG.get("ee_actual_trail_rgba", [1.0, 0.5, 0.1, 1.0]), dtype=float
        )
        ee_step_counter = 0
        ee_desired_traj = []
        ee_actual_traj = []

        control_dt = mj_model.opt.timestep * float(CONFIG["control_decimation"])
        control_print_every = int(CONFIG["control_print_every"])

        control_process = mp.Process(
            target=self._control_worker,
            args=(
                qpos_shared,
                ctrl_shared,
                target_shared,
                desired_ee_shared,
                actual_ee_shared,
                run_flag,
                control_dt,
                control_print_every,
            ),
            daemon=True,
        )
        control_process.start()

        try:
            while self.is_running():
                qpos_arr[:] = mj_data.qpos[:qpos_len]
                mj_data.ctrl[ctrl_start:ctrl_end] = ctrl_arr

                t = time.time() - traj_start_time
                target_arr[:] = _trajectory_point(t)

                mj_model.geom_pos[target_geom_id] = target_arr

                if trail_enabled and trail_max_points > 0:
                    trail_step_counter += 1
                    if trail_step_counter % trail_stride_steps == 0:
                        target_traj.append(target_arr.copy())
                        if len(target_traj) > trail_max_points:
                            target_traj = target_traj[-trail_max_points:]

                if ee_trail_enabled and ee_trail_max_points > 0:
                    ee_step_counter += 1
                    if ee_step_counter % ee_trail_stride_steps == 0:
                        ee_desired_traj.append(desired_ee_arr.copy())
                        ee_actual_traj.append(actual_ee_arr.copy())
                        if len(ee_desired_traj) > ee_trail_max_points:
                            ee_desired_traj = ee_desired_traj[-ee_trail_max_points:]
                        if len(ee_actual_traj) > ee_trail_max_points:
                            ee_actual_traj = ee_actual_traj[-ee_trail_max_points:]

                if (trail_enabled and len(target_traj) > 1) or (
                    ee_trail_enabled
                    and (len(ee_desired_traj) > 1 or len(ee_actual_traj) > 1)
                ):
                    self.viewer_handle.user_scn.ngeom = 0

                if trail_enabled and len(target_traj) > 1:
                    for i in range(len(target_traj) - 1):
                        add_visual_capsule(
                            self.viewer_handle,
                            target_traj[i],
                            target_traj[i + 1],
                            trail_radius,
                            trail_rgba,
                        )

                if ee_trail_enabled and len(ee_desired_traj) > 1:
                    for i in range(len(ee_desired_traj) - 1):
                        add_visual_capsule(
                            self.viewer_handle,
                            ee_desired_traj[i],
                            ee_desired_traj[i + 1],
                            ee_desired_radius,
                            ee_desired_rgba,
                        )

                if ee_trail_enabled and len(ee_actual_traj) > 1:
                    for i in range(len(ee_actual_traj) - 1):
                        add_visual_capsule(
                            self.viewer_handle,
                            ee_actual_traj[i],
                            ee_actual_traj[i + 1],
                            ee_actual_radius,
                            ee_actual_rgba,
                        )

                mujoco.mj_step(mj_model, mj_data)
                self.sync()

                time_until_next_step = mj_model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                step_start = time.time()
        finally:
            run_flag.value = False
            control_process.join(timeout=2.0)

        while True:
            time.sleep(0.01)


if __name__ == "__main__":
    viewer = CustomViewer(mj_model, mj_data)
    viewer.cam.distance = float(CONFIG["cam_distance"])
    viewer.cam.azimuth = float(CONFIG["cam_azimuth"])
    viewer.cam.elevation = float(CONFIG["cam_elevation"])
    viewer.run_loop()
