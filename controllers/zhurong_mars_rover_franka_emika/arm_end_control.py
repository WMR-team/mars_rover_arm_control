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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configs/config.yaml")

DEFAULT_CONFIG = {
    "simulation_dt": 0.002,
    "control_decimation": 10,
    "mujoco_scene_path": "",
    "pinocchio_model_path": "",
    "target_sphere_geom": "target_sphere",
    "target_start": [1.15, 0.0, 0.65],
    "target_y_min": -0.35,
    "target_y_max": 0.35,
    "target_y_step": 0.002,
    "control_print_every": 5,
    "control_start_delay": 0.0,
    "control_ramp_time": 0.0,
    "control_max_delta": 0.0,
    "ik_eps": 0.01,
    "ik_max_iters": 2000,
    "ik_dt": 0.05,
    "ik_damp": 0.0001,
    "ik_lock_front_dofs": 25,
    "qpos_len": 36,
    "ctrl_start": 20,
    "ctrl_end": 27,
    "arm_start": 27,
    "arm_end": 34,
    "ik_joint_id": 28,
    "cam_distance": 3,
    "cam_azimuth": 0,
    "cam_elevation": -30,
}


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: v for k, v in data.items() if v is not None})
    return cfg


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
    return q.flatten().tolist()


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


class CustomViewer:
    def __init__(self, mj_model, mj_data):
        self.handle = mujoco.viewer.launch_passive(mj_model, mj_data)
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
        theta = np.pi
        self.R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

        self.new_mj_q = self.initial_mj_q

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def _control_worker(
        self,
        qpos_shared,
        ctrl_shared,
        target_shared,
        run_flag,
        control_dt,
        control_print_every,
    ):
        qpos_arr = np.ctypeslib.as_array(qpos_shared)
        ctrl_arr = np.ctypeslib.as_array(ctrl_shared)
        target_arr = np.ctypeslib.as_array(target_shared)

        target_start = CONFIG["target_start"]
        x = float(target_start[0])
        y = float(target_start[1])
        z = float(target_start[2])
        y_min = float(CONFIG["target_y_min"])
        y_max = float(CONFIG["target_y_max"])
        y_step = float(CONFIG["target_y_step"])
        op = -1
        counter = 0

        delay_s = float(CONFIG["control_start_delay"])
        ramp_s = float(CONFIG["control_ramp_time"])
        max_delta = float(CONFIG["control_max_delta"])
        start_time = time.time()

        while run_flag.value:
            elapsed = time.time() - start_time
            if elapsed < delay_s:
                ctrl_arr[:] = qpos_arr[CONFIG["arm_start"] : CONFIG["arm_end"]]
                time.sleep(min(control_dt, delay_s - elapsed))
                continue
            counter += 1

            if y < y_max and y > y_min:
                y += y_step * op
            elif y >= y_max:
                y = y_max - 0.01
                op = -1
            elif y <= y_min:
                y = y_min + 0.01
                op = 1

            cur_mj_q = qpos_arr.copy()
            cur_ph_q = mujoco_q_to_pinocchio_q(cur_mj_q)
            new_ph_q = inverse_arm_kinematics(cur_ph_q, self.R_x, [x, y, z])
            new_mj_q = pinocchio_q_to_mujoco_q(np.array(new_ph_q, dtype=float))
            desired_ctrl = new_mj_q[CONFIG["arm_start"] : CONFIG["arm_end"]]
            cur_ctrl = ctrl_arr.copy()

            if ramp_s > 0.0:
                ramp_alpha = min((elapsed - delay_s) / ramp_s, 1.0)
                desired_ctrl = cur_ctrl + ramp_alpha * (desired_ctrl - cur_ctrl)

            if max_delta > 0.0:
                delta = np.clip(desired_ctrl - cur_ctrl, -max_delta, max_delta)
                ctrl_arr[:] = cur_ctrl + delta
            else:
                ctrl_arr[:] = desired_ctrl
            cur_arr = cur_mj_q[CONFIG["arm_start"] : CONFIG["arm_end"]]
            target_arr[:] = np.array([x, y, z], dtype=float)

            if counter % control_print_every == 0:
                # print(f"ctrl[0:7]: {fmt_3dec(ctrl_arr)}")
                # print(f"cur [0:7]: {fmt_3dec(cur_arr)}")
                print(f"diff[0:7]: {fmt_3dec(ctrl_arr - cur_arr)}")

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
        run_flag = mp.Value(ctypes.c_bool, True)

        qpos_arr = np.ctypeslib.as_array(qpos_shared)
        ctrl_arr = np.ctypeslib.as_array(ctrl_shared)
        target_arr = np.ctypeslib.as_array(target_shared)

        qpos_arr[:] = mj_data.qpos[:qpos_len]
        ctrl_arr[:] = mj_data.ctrl[ctrl_start:ctrl_end]
        target_arr[:] = np.array(CONFIG["target_start"], dtype=float)

        control_dt = mj_model.opt.timestep * float(CONFIG["control_decimation"])
        control_print_every = int(CONFIG["control_print_every"])

        control_process = mp.Process(
            target=self._control_worker,
            args=(
                qpos_shared,
                ctrl_shared,
                target_shared,
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
                mj_model.geom_pos[target_geom_id] = target_arr

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
