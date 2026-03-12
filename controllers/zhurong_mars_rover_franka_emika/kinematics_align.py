# 这个得到了结论, 对于freejoint, MJ: [x, y, z, qw, qx, qy, qz]
# Pin: [x, y, z, qx, qy, qz, qw]


import time

import mujoco
import numpy as np
import glfw
import mujoco.viewer
from scipy.optimize import minimize
import os
import numpy as np
import pinocchio
from numpy.linalg import norm, solve
from typing import List

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
# 从 URDF 文件构建机器人模型
pinocchio_robot = pinocchio.RobotWrapper.BuildFromMJCF(MODEL_PATH)
# 为模型创建数据对象，用于存储计算过程中的中间结果
ph_model = pinocchio_robot.model
ph_collision_model = pinocchio_robot.collision_model
ph_visual_model = pinocchio_robot.visual_model
ph_data = pinocchio_robot.data
ph_collision_data = pinocchio_robot.collision_data
ph_visual_data = pinocchio_robot.visual_data


def print_model_info():
    print("-" * 40)
    print("Pinocchio nq, nv:", ph_model.nq, ph_model.nv)
    print("MuJoCo nq, nv    :", mj_model.nq, mj_model.nv)
    print("Pinocchio joint names:")
    for i, name in enumerate(ph_model.names):
        print(i, name)
    print("MuJoCo joint names:")
    for jid in range(mj_model.njnt):
        print(jid, mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    print("-" * 40)


# print_model_info()


def print_mujoco_q():
    print("-" * 40)
    print("mujoco q")
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)

        print(
            f"jnt {j:2d}: name={name:20s} "
            f"type={mj_model.jnt_type[j]:2d} "
            f"qposadr={mj_model.jnt_qposadr[j]:2d} "
            f"bodyid={mj_model.jnt_bodyid[j]:2d}"
        )
    print("-" * 40)


def print_pinocchio_q():
    print("-" * 40)
    print("pinocchio q")
    for jid, joint in enumerate(ph_model.joints):
        print(
            f"jid = {jid}: "
            f"joint.shortname() = {joint.shortname()}, "
            f"nq = {joint.nq}, "
            f"idx_q = {joint.idx_q}, "
            #   f"parent = {joint.parent}"
        )
    print("-" * 40)


def print_mujoco_actuator_info():
    for i in range(mj_model.nu):  # nu = number of actuators
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = mj_model.actuator_trnid[i]  # [joint_id, ...]
        joint_id = trnid[0]
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        act_type = mj_model.actuator_biastype[i]  # 或 actuator_gainprm,typefrc 等
        print(i, name, "-> joint", joint_id, joint_name)


def print_mujoco_actuator_info2():
    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    arm_joint_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in arm_joint_names
    ]
    arm_qpos_idx = [mj_model.jnt_qposadr[jid] for jid in arm_joint_ids]
    print("arm joint ids:", arm_joint_ids)
    print("arm qpos idx:", arm_qpos_idx)


def print_pinocchio_joint_limit():
    print("Pinocchio limits (lower/upper):")
    for i in range(len(ph_model.lowerPositionLimit)):
        # name = ph_model.names[i]
        lower = ph_model.lowerPositionLimit[i]
        upper = ph_model.upperPositionLimit[i]
        # print(f"{i:2d}: {name:20s} : lower={lower: .3f}, upper={upper: .3f}")
        print(f"{i:2d} : lower={lower: .3f}, upper={upper: .3f}")


def print_mujoco_joint_limit():
    print("MuJoCo jnt_range:")
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lower, upper = mj_model.jnt_range[j]
        print(f"{j:2d}: {name:20s} : lower={lower: .3f}, upper={upper: .3f}")


def align_pinocchio_joint_limits_from_mujoco():
    name_to_mj = {
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j): j
        for j in range(mj_model.njnt)
    }
    updated = 0
    skipped = 0

    for jid, joint in enumerate(ph_model.joints):
        name = ph_model.names[jid]
        if name not in name_to_mj:
            skipped += 1
            continue
        if joint.nq != 1:
            skipped += 1
            continue

        mj_jid = name_to_mj[name]
        lower, upper = mj_model.jnt_range[mj_jid]
        idx_q = joint.idx_q
        ph_model.lowerPositionLimit[idx_q] = lower
        ph_model.upperPositionLimit[idx_q] = upper
        updated += 1

    print(f"Aligned limits: updated={updated}, skipped={skipped}")


# exit the code

# def print_mujoco_joint_info():
#     print("-"*80)
#     for j in range(mj_model.njnt):
#         jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
#         body_id = mj_model.jnt_bodyid[j]       # 这个 joint 所在的 body
#         pos = mj_data.xpos[body_id]            # 该 body frame 原点的 world 位置
#         print(f"joint {j:2d}: {str(jname):20s}:  world pos = {pos}")
#     print("-"*80)


def print_pinocchio_joint_info():
    print("-" * 80)
    for i, (name, oMi) in enumerate(zip(ph_model.names, ph_data.oMi)):
        print(
            "{} : {:<24} : {: .2f} {: .2f} {: .2f}".format(
                i, name, *oMi.translation.T.flat
            )
        )
    print("-" * 80)


# def print_difference_between_mujoco_and_pinocchio():
#     print("-"*80)
#     for j in range(min(mj_model.njnt, ph_model.njoints)):
#         mujoco_pos = mj_data.xpos[mj_model.jnt_bodyid[j]]
#         mujoco_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
#         pinocchio_pos = ph_data.oMi[j+1].translation
#         pinocchio_name = ph_model.names[j+1]
#         print(f"joint {j:2d}:  mujoco pos = {mujoco_pos}, mujoco name = {mujoco_name}, pinocchio name = {pinocchio_name}, pinocchio pos = {pinocchio_pos}, difference = {mujoco_pos - pinocchio_pos}")
#     print("-"*80)


def print_joint_difference_between_mujoco_and_pinocchio():
    print("-" * 80)
    print("Compare JOINT frames (Pinocchio oMi vs MuJoCo xanchor)")
    print("-" * 80)
    for j in range(1, ph_model.njoints):  # 0 是 universe, 从 1 开始
        name = ph_model.names[j]

        # Pinocchio joint frame
        pin_pos = ph_data.oMi[j].translation

        # MuJoCo: joint frame 的位置
        mj_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if mj_jid < 0:
            print(f"joint {j:2d} {name:20s}: not found in MuJoCo")
            continue

        mj_pos = mj_data.xanchor[mj_jid]

        diff = mj_pos - pin_pos
        print(
            f"joint {j:2d}: {name:20s}:  "
            f"mj = {fmt_3dec(mj_pos)}, "
            f"ph = {fmt_3dec(pin_pos)}, "
            f"diff = {fmt_2dec(diff)}"
        )
    print("-" * 80)


def print_mujoco_body_info():
    print("-" * 80)
    for j in range(mj_model.njnt):
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, j)
        body_id = mj_model.jnt_bodyid[j]  # 这个 joint 所在的 body
        pos = mj_data.xpos[body_id]  # 该 body frame 原点的 world 位置
        print(f"joint {j:2d}: {str(jname):20s}:  world pos = {pos}")
    print("-" * 80)


def print_pinocchio_body_info():
    print("-" * 80)
    for i in range(len(ph_model.names)):
        name = ph_model.names[i]
        joint_pos = ph_data.oMi[i].translation
        visual_pos = ph_visual_data.oMg[i].translation
        collision_pos = ph_collision_data.oMg[i].translation
        print(
            f"body {i:2d}: {name:20s}:  joint pos = {joint_pos}, visual pos = {visual_pos}, collision pos = {collision_pos}"
        )

    # for i, (name, oMg) in enumerate(zip(ph_model.names, ph_collision_data.oMg)):
    #     print("{:d} : {:<24} : {: .2f} {: .2f} {: .2f}".format(i, name, *oMg.translation.T.flat))
    # for i, (name, oMg) in enumerate(zip(ph_model.names, ph_visual_data.oMg)):
    #     print("{:d} : {:<24} : {: .2f} {: .2f} {: .2f}".format(i, name, *oMg.translation.T.flat))
    # for i, (name, oMi) in enumerate(zip(ph_model.names, ph_data.oMi)):
    #     print(
    #         "{} : {:<24} : {: .2f} {: .2f} {: .2f}".format(i, name, *oMi.translation.T.flat)
    #     )
    print("-" * 80)


def fmt_3dec(x: np.ndarray) -> str:
    return np.array2string(
        x, formatter={"float_kind": lambda v: f"{v: .3f}"}, separator=" "
    )


def fmt_2dec(x: np.ndarray) -> str:
    return np.array2string(
        x, formatter={"float_kind": lambda v: f"{v: .2f}"}, separator=" "
    )


def print_body_difference_between_mujoco_and_pinocchio():
    print("-" * 80)
    print(f"Comparing body positions between MuJoCo and Pinocchio:")
    print("-" * 80)

    for i in range(len(ph_model.names)):
        name = ph_model.names[i]
        ph_joint_pos = ph_data.oMi[i].translation
        ph_visual_pos = ph_visual_data.oMg[i].translation
        ph_collision_pos = ph_collision_data.oMg[i].translation

        # MuJoCo: body frame 的位置
        mj_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if mj_jid < 0:
            print(f"joint {mj_jid:2d} {name:20s}: not found in MuJoCo")
            continue
        mj_pos = mj_data.xpos[mj_jid]

        diff1 = mj_pos - ph_joint_pos
        diff2 = mj_pos - ph_visual_pos
        diff3 = mj_pos - ph_collision_pos
        print(
            f"body {mj_jid:2d}: {name:20s}:  "
            f"mj = {fmt_3dec(mj_pos)}, "
            f"ph_jnt_pos = {fmt_3dec(ph_joint_pos)}, "
            f"ph_vis_pos = {fmt_3dec(ph_visual_pos)}, "
            f"ph_col_pos = {fmt_3dec(ph_collision_pos)}, "
            f"d1 = {fmt_2dec(diff1)}, "
            f"d2 = {fmt_2dec(diff2)}, "
            f"d3 = {fmt_2dec(diff3)}"
        )
    print("-" * 80)


def pinocchio_forward_kinematics(q):
    pinocchio.forwardKinematics(ph_model, ph_data, q)
    pinocchio.updateGeometryPlacements(
        ph_model, ph_data, ph_collision_model, ph_collision_data
    )
    pinocchio.updateGeometryPlacements(
        ph_model, ph_data, ph_visual_model, ph_visual_data
    )
    # for j in range(ph_model.nq):
    #     print(f"joint {j:2d}:  world pos = {ph_data.oMi[j].translation}")


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


class CustomViewer:
    def __init__(self, mj_model, mj_data):
        self.handle = mujoco.viewer.launch_passive(mj_model, mj_data)

        # 初始关节角度
        self.initial_mj_q = mj_data.qpos[:36].copy()
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
        status = 0
        while self.is_running():
            mujoco.mj_forward(mj_model, mj_data)
            self.initial_mj_q[0] += 0.01
            mj_data.qpos[:36] = self.initial_mj_q[:36]
            pinocchio_forward_kinematics(
                mujoco_q_to_pinocchio_q(self.initial_mj_q[:36])
            )
            mujoco.mj_step(mj_model, mj_data)
            self.sync()
            print_joint_difference_between_mujoco_and_pinocchio()
            time.sleep(0.05)

        # while True:
        #     time.sleep(0.01)


# viewer = CustomViewer(mj_model, mj_data)
# viewer.cam.distance = 3
# viewer.cam.azimuth = 0
# viewer.cam.elevation = -30
# viewer.run_loop()


# print_mujoco_q()
# print_pinocchio_q()
# print_mujoco_actuator_info2()
# align_pinocchio_joint_limits_from_mujoco()
print_pinocchio_joint_limit()
print_mujoco_joint_limit()
