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
import mujoco
import numpy as np
import mujoco.viewer
from scipy.optimize import minimize
import os
import numpy as np
import pinocchio
from numpy.linalg import norm, solve
from typing import List
from mars_rover_arm_control.utils.time_analysis import timeit
from mars_rover_arm_control.utils.print_control import control_print

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configs/config.yaml")
MODEL_PATH = os.path.join(
    ROOT_DIR, "../../planetary_robot_zoo/zhurong_mars_rover_franka_emika/zhurong.xml"
)
SCENE_PATH = os.path.join(
    ROOT_DIR, "../../planetary_robot_zoo/zhurong_mars_rover_franka_emika/scene.xml"
)


mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
mj_data = mujoco.MjData(mj_model)

mj_model.opt.timestep = 0.002
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

@timeit(unit="ms")
def inverse_arm_kinematics(current_q, target_dir, target_pos):
    arm_idx = np.asarray(range(27, 34, 1))
    # 指定要控制的关节 ID
    JOINT_ID = 28  # TODO: 根据模型中的关节名称获取对应的关节 ID
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = 1e-2
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = 2000
    # 定义积分步长，用于更新关节角度
    DT = 5e-2
    # 定义阻尼因子，用于避免矩阵奇异
    damp = 1e-4

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
            print(f"inverse_kinematics failed, err = {norm(err)}")
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

@timeit(unit="ms")
def inverse_kinematics(current_q, target_dir, target_pos):

    # 指定要控制的关节 ID
    JOINT_ID = 28  # TODO: 根据模型中的关节名称获取对应的关节 ID
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = 1e-2
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = 1000
    # 定义积分步长，用于更新关节角度
    DT = 5e-2
    # 定义阻尼因子，用于避免矩阵奇异
    damp = 1e-4

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
        v[:25] = 0.0
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
        self.initial_mj_q = mj_data.qpos[:36].copy()
        self.cur_mj_q = mj_data.qpos[:36].copy()
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

    def run_loop(self):
        step_start = time.time()
        status = 0
        # while self.is_running():
        mujoco.mj_forward(mj_model, mj_data)
        x = 1.15
        z = 0.65
        y = 0.0
        op = -1
        counter = 0
        control_decimation = 10
        target_geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_sphere"
        )
        while True:

            counter += 1

            if counter % control_decimation == 0:
                if y < 0.35 and y > -0.35:
                    y += 0.002 * op
                elif y >= 0.35:
                    y = 0.34
                    op = -1
                elif y <= -0.35:
                    y = -0.34
                    op = 1

                # for y in np.arange(-0.4, 0.4, 0.01):
                # FIXME: 这里每次都需要再拷贝吗
                self.cur_mj_q = mj_data.qpos[:36].copy()
                self.cur_ph_q = mujoco_q_to_pinocchio_q(self.cur_mj_q)
                new_ph_q = inverse_arm_kinematics(self.cur_ph_q, self.R_x, [x, y, z])
                # new_ph_q = arm_inverse_kinematics(self.cur_ph_q, self.R_x, [x, y, z])
                new_mj_q = pinocchio_q_to_mujoco_q(new_ph_q)

                mj_data.ctrl[20:27] = new_mj_q[27:34]  # 只更新 1DoF joints 部分
                # mj_data.qpos[27:34] = new_mj_q[27:34]  # 只更新 1DoF joints 部分
                if counter % (control_decimation * 5) == 0:  # 每隔一段时间打印一次
                    # print(f"newq[27:34]: {new_mj_q[27:34]}")
                    print(f"ctrl[20:27]: {fmt_3dec(mj_data.ctrl[20:27])}")
                    print(f"qpos[27:34]: {fmt_3dec(mj_data.qpos[27:34])}")
                    print(
                        f"diff: {fmt_3dec(mj_data.ctrl[20:27] - mj_data.qpos[27:34])}",
                    )
                mj_model.geom_pos[target_geom_id] = np.array([x, y, z], dtype=float)
            mujoco.mj_step(mj_model, mj_data)
            self.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            # print("sync!!!!!!!!!!!!!")
            # print(f"expected pos: {[x, float(y), z]}")
            # time.sleep(1.0)

        while True:
            time.sleep(0.01)


viewer = CustomViewer(mj_model, mj_data)
viewer.cam.distance = 3
viewer.cam.azimuth = 0
viewer.cam.elevation = -30
viewer.run_loop()
