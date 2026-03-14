import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pinocchio

from mars_rover_arm_control.utils.print_control import log_and_print
from mars_rover_arm_control.utils.state_machine import State, StateMachine


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def clamp_vec(vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(vec, lo), hi)


class ArmKinematics:
    def __init__(self, ph_model, ph_data, config: dict) -> None:
        self.ph_model = ph_model
        self.ph_data = ph_data
        self.config = config
        self.arm_idx = np.asarray(
            range(int(config["arm_start"]), int(config["arm_end"]), 1)
        )
        self.joint_id = int(config["ik_joint_id"])

    def mujoco_q_to_pinocchio_q(self, mujoco_q: np.ndarray) -> np.ndarray:
        pinocchio_q = mujoco_q.copy()
        pinocchio_q[3:7] = [
            mujoco_q[4],
            mujoco_q[5],
            mujoco_q[6],
            mujoco_q[3],
        ]
        return pinocchio_q

    def pinocchio_q_to_mujoco_q(self, pinocchio_q: np.ndarray) -> np.ndarray:
        mujoco_q = pinocchio_q.copy()
        mujoco_q[3:7] = [
            pinocchio_q[6],
            pinocchio_q[3],
            pinocchio_q[4],
            pinocchio_q[5],
        ]
        return mujoco_q

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        pinocchio.forwardKinematics(self.ph_model, self.ph_data, q)
        return self.ph_data.oMi[self.joint_id].translation.copy()

    def solve_ik(
        self,
        current_q: np.ndarray,
        target_pose: Pose,
        control_orientation: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        eps = float(self.config["ik_eps"])
        it_max = int(self.config["ik_max_iters"])
        dt = float(self.config["ik_dt"])
        damp = float(self.config["ik_damp"])

        q = current_q.copy()
        i = 0
        while True:
            pinocchio.forwardKinematics(self.ph_model, self.ph_data, q)
            iMd = self.ph_data.oMi[self.joint_id].actInv(
                pinocchio.SE3(target_pose.rotation, target_pose.position)
            )
            err = pinocchio.log(iMd).vector
            if not control_orientation:
                err[3:] = 0.0
            if np.linalg.norm(err) < eps:
                return q.copy(), True
            if i >= it_max:
                return q.copy(), False

            J_full = pinocchio.computeJointJacobian(
                self.ph_model, self.ph_data, q, self.joint_id
            )
            J = -pinocchio.Jlog6(iMd.inverse()) @ J_full
            if not control_orientation:
                J[3:, :] = 0.0

            J_arm = J[:, self.arm_idx]
            v_arm = -J_arm.T @ np.linalg.solve(J_arm @ J_arm.T + damp * np.eye(6), err)

            v = np.zeros(self.ph_model.nv)
            v[self.arm_idx] = v_arm
            q = pinocchio.integrate(self.ph_model, q, v * dt)
            i += 1

        return q.copy(), False

    def solve_ik_bounded(
        self,
        current_q: np.ndarray,
        target_pose: Pose,
        control_orientation: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        if not bool(self.config.get("ik_use_bounds", False)):
            return self.solve_ik(current_q, target_pose, control_orientation)

        from scipy.optimize import minimize

        q0 = np.array(current_q, dtype=float)
        x0 = q0[self.arm_idx].copy()

        lower = self.ph_model.lowerPositionLimit[self.arm_idx]
        upper = self.ph_model.upperPositionLimit[self.arm_idx]
        bounds = []
        for lo, hi in zip(lower, upper):
            lo_b = None if np.isneginf(lo) else float(lo)
            hi_b = None if np.isposinf(hi) else float(hi)
            bounds.append((lo_b, hi_b))

        def cost(x):
            q = q0.copy()
            q[self.arm_idx] = x
            pinocchio.forwardKinematics(self.ph_model, self.ph_data, q)
            iMd = self.ph_data.oMi[self.joint_id].actInv(
                pinocchio.SE3(target_pose.rotation, target_pose.position)
            )
            err = pinocchio.log(iMd).vector
            if not control_orientation:
                err[3:] = 0.0
            return float(err.T @ err)

        res = minimize(
            cost,
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(self.config["ik_bounds_max_iters"])},
        )

        q = q0.copy()
        q[self.arm_idx] = res.x
        pinocchio.forwardKinematics(self.ph_model, self.ph_data, q)
        iMd = self.ph_data.oMi[self.joint_id].actInv(
            pinocchio.SE3(target_pose.rotation, target_pose.position)
        )
        err = pinocchio.log(iMd).vector
        if not control_orientation:
            err[3:] = 0.0
        success = bool(res.success) and np.linalg.norm(err) < float(
            self.config["ik_eps"]
        )
        return q.copy(), success


class ArmController:
    def __init__(
        self,
        kinematics: ArmKinematics,
        config: dict,
        default_rotation: np.ndarray,
    ) -> None:
        self.kin = kinematics
        self.config = config
        self.default_rotation = default_rotation

        init_pos = np.array(config.get("arm_init_pos", [1.2, 0.0, 0.6]), dtype=float)
        init_quat = np.array(
            config.get("arm_init_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float
        )
        init_rot = quat_to_rotmat(init_quat)
        self.init_pose = Pose(init_pos, init_rot)

        self.target_pose: Optional[Pose] = None
        self._traj: List[Pose] = []
        self._traj_index = 0
        self._last_target_pos: Optional[np.ndarray] = None

        self._machine = StateMachine(initial_state="init")
        self._machine.add_state(_ArmInitState(self))
        self._machine.add_state(_ArmIdleState(self))
        self._machine.add_state(_ArmTrackState(self))

    def set_target_pose(self, target_pos: np.ndarray, target_rot: Optional[np.ndarray]):
        rot = target_rot if target_rot is not None else self.default_rotation
        self.target_pose = Pose(np.array(target_pos, dtype=float), np.array(rot))

    def set_target_from_joystick(self, delta_pos: np.ndarray):
        if self.target_pose is None:
            base = self.init_pose.position
        else:
            base = self.target_pose.position
        new_pos = base + np.array(delta_pos, dtype=float)
        self.set_target_pose(new_pos, None)

    def is_pose_feasible(
        self,
        base_pos: np.ndarray,
        current_q: np.ndarray,
        target_pos: np.ndarray,
    ) -> bool:
        lo = np.array(self.config.get("arm_workspace_min", [-0.6, -0.6, 0.0]))
        hi = np.array(self.config.get("arm_workspace_max", [0.8, 0.6, 1.0]))
        rel = target_pos - base_pos
        in_bounds = np.all(rel >= lo) and np.all(rel <= hi)
        if not in_bounds:
            return False

        target_pose = Pose(target_pos, self.default_rotation)
        _, success = self.kin.solve_ik_bounded(
            current_q,
            target_pose,
            bool(self.config.get("ik_control_orientation", True)),
        )
        return bool(success)

    def plan_trajectory(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> List[Pose]:
        step = float(self.config.get("arm_traj_step", 0.02))
        dist = float(np.linalg.norm(target_pos - current_pos))
        steps = max(2, int(dist / max(step, 1e-6)) + 1)
        points = []
        for i in range(steps):
            alpha = i / (steps - 1)
            pos = current_pos + alpha * (target_pos - current_pos)
            points.append(Pose(pos, self.default_rotation))
        return points

    def step(self, current_mj_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._machine.update(current_mj_q=current_mj_q)

        if self._traj:
            pose = self._traj[self._traj_index]
        elif self.target_pose is not None:
            pose = self.target_pose
        else:
            pose = self.init_pose

        current_ph_q = self.kin.mujoco_q_to_pinocchio_q(current_mj_q)
        new_ph_q, _ = self.kin.solve_ik_bounded(
            current_ph_q, pose, bool(self.config.get("ik_control_orientation", True))
        )
        new_mj_q = self.kin.pinocchio_q_to_mujoco_q(new_ph_q)
        return new_mj_q, pose.position

    def _enter_init(self, current_mj_q: np.ndarray) -> None:
        current_ph_q = self.kin.mujoco_q_to_pinocchio_q(current_mj_q)
        current_pos = self.kin.ee_position(current_ph_q)
        self._traj = self.plan_trajectory(current_pos, self.init_pose.position)
        self._traj_index = 0

    def _enter_track(self, current_mj_q: np.ndarray) -> None:
        if self.target_pose is None:
            self._traj = []
            return
        current_ph_q = self.kin.mujoco_q_to_pinocchio_q(current_mj_q)
        current_pos = self.kin.ee_position(current_ph_q)
        self._traj = self.plan_trajectory(current_pos, self.target_pose.position)
        self._traj_index = 0

    def _advance_traj(self) -> None:
        if not self._traj:
            return
        self._traj_index += 1
        if self._traj_index >= len(self._traj):
            self._traj = []
            self._traj_index = 0


class _ArmInitState(State):
    def __init__(self, controller: ArmController) -> None:
        super().__init__("init")
        self.controller = controller

    def on_enter(self, **kwargs) -> None:
        self.controller._enter_init(kwargs["current_mj_q"])
        log_and_print("[arm] enter init")

    def update(self, **kwargs) -> Optional[str]:
        if not self.controller._traj:
            return "idle"
        self.controller._advance_traj()
        return None


class _ArmIdleState(State):
    def __init__(self, controller: ArmController) -> None:
        super().__init__("idle")
        self.controller = controller

    def on_enter(self, **kwargs) -> None:
        log_and_print("[arm] enter idle")

    def update(self, **kwargs) -> Optional[str]:
        target = self.controller.target_pose
        if target is None:
            return None
        if self.controller._last_target_pos is None or not np.allclose(
            self.controller._last_target_pos, target.position, atol=1e-4
        ):
            return "track"
        return None


class _ArmTrackState(State):
    def __init__(self, controller: ArmController) -> None:
        super().__init__("track")
        self.controller = controller

    def on_enter(self, **kwargs) -> None:
        self.controller._enter_track(kwargs["current_mj_q"])
        if self.controller.target_pose is not None:
            self.controller._last_target_pos = (
                self.controller.target_pose.position.copy()
            )
        log_and_print("[arm] enter track")

    def update(self, **kwargs) -> Optional[str]:
        if not self.controller._traj:
            return "idle"
        self.controller._advance_traj()
        return None


class RoverController:
    def __init__(self, config: dict) -> None:
        self.config = config

        self.half_width = float(config.get("base_half_width", 0.652))
        self.half_length = float(config.get("base_half_length", 0.775))
        self.wheel_radius = float(config.get("base_wheel_radius", 0.15))

        self.wheel_start = int(config.get("base_wheel_ctrl_start", 0))
        self.wheel_count = int(config.get("base_wheel_ctrl_count", 6))
        self.steer_start = int(config.get("base_steer_ctrl_start", 12))
        self.steer_count = int(config.get("base_steer_ctrl_count", 6))

        self.max_speed = float(config.get("base_max_speed", 0.3))
        self.max_omega = float(config.get("base_max_omega", 0.6))
        self.kp_lin = float(config.get("base_kp_lin", 1.2))
        self.kp_ang = float(config.get("base_kp_ang", 1.5))
        self.goal_tol = float(config.get("base_goal_tolerance", 0.05))

        self._path: List[np.ndarray] = []
        self._path_index = 0

        self._base_pos = np.zeros(3, dtype=float)
        self._base_yaw = 0.0
        self._wheel_cmd = np.zeros(6, dtype=float)
        self._steer_cmd = np.zeros(6, dtype=float)

        self._machine = StateMachine(initial_state="idle")
        self._machine.add_state(_RoverIdleState(self))
        self._machine.add_state(_RoverMoveState(self))

    def update_base_pose(self, pos: np.ndarray, yaw: float) -> None:
        self._base_pos = np.array(pos, dtype=float)
        self._base_yaw = float(yaw)

    def get_base_pose(self) -> Tuple[np.ndarray, float]:
        return self._base_pos.copy(), self._base_yaw

    def plan_to_reachable(
        self,
        target_pos: np.ndarray,
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
    ) -> None:
        base_pos, _ = self.get_base_pose()
        rel = target_pos - base_pos
        clamped = clamp_vec(rel, workspace_min, workspace_max)
        base_goal = target_pos - clamped

        step = float(self.config.get("base_traj_step", 0.05))
        dist = float(np.linalg.norm(base_goal[:2] - base_pos[:2]))
        steps = max(2, int(dist / max(step, 1e-6)) + 1)

        self._path = []
        for i in range(steps):
            alpha = i / (steps - 1)
            pos = base_pos + alpha * (base_goal - base_pos)
            self._path.append(pos)
        self._path_index = 0

    def step(self) -> None:
        self._machine.update()

    def _advance_path(self) -> None:
        if not self._path:
            return
        self._path_index += 1
        if self._path_index >= len(self._path):
            self._path = []
            self._path_index = 0

    def _apply_cmd_vel(self, vx: float, omega: float) -> None:
        if self.wheel_count < 6 or self.steer_count < 6:
            return

        if abs(omega) < 1e-6:
            theta = np.zeros(6)
            vel_arr = np.ones(6) * vx / self.wheel_radius
        else:
            turning_radius = vx / omega
            r_arr = np.zeros(6)
            r_arr[0] = math.sqrt(
                (turning_radius - self.half_width) ** 2 + self.half_length**2
            )
            r_arr[1] = math.sqrt(
                (turning_radius + self.half_width) ** 2 + self.half_length**2
            )
            r_arr[2] = abs(turning_radius - self.half_width)
            r_arr[3] = abs(turning_radius + self.half_width)
            r_arr[4] = r_arr[0]
            r_arr[5] = r_arr[1]
            vel_arr = abs(omega) * r_arr / self.wheel_radius
            if vx < 0:
                vel_arr = -vel_arr
            elif vx == 0 and omega < 0:
                vel_arr = -vel_arr

            theta = np.zeros(6)
            theta[0] = math.atan(self.half_length / (turning_radius - self.half_width))
            theta[1] = math.atan(self.half_length / (turning_radius + self.half_width))
            theta[2] = 0.0
            theta[3] = 0.0
            theta[4] = -theta[0]
            theta[5] = -theta[1]

            if turning_radius >= 0 and abs(turning_radius) < self.half_width:
                vel_arr[0] = -vel_arr[0]
                vel_arr[2] = -vel_arr[2]
                vel_arr[4] = -vel_arr[4]
            elif turning_radius < 0 and abs(turning_radius) < self.half_width:
                vel_arr[1] = -vel_arr[1]
                vel_arr[3] = -vel_arr[3]
                vel_arr[5] = -vel_arr[5]

        self._steer_cmd = theta.copy()
        self._wheel_cmd = vel_arr.copy()

    def _stop(self) -> None:
        self._wheel_cmd = np.zeros(6, dtype=float)
        self._steer_cmd = np.zeros(6, dtype=float)

    def get_cmd(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._wheel_cmd.copy(), self._steer_cmd.copy()

    def apply_to_mj(self, mj_data) -> None:
        mj_data.ctrl[self.steer_start : self.steer_start + 6] = self._steer_cmd
        mj_data.ctrl[self.wheel_start : self.wheel_start + 6] = self._wheel_cmd


class _RoverIdleState(State):
    def __init__(self, rover: RoverController) -> None:
        super().__init__("idle")
        self.rover = rover

    def on_enter(self, **kwargs) -> None:
        log_and_print("[rover] enter idle")
        self.rover._stop()

    def update(self, **kwargs) -> Optional[str]:
        if self.rover._path:
            return "move"
        return None


class _RoverMoveState(State):
    def __init__(self, rover: RoverController) -> None:
        super().__init__("move")
        self.rover = rover

    def on_enter(self, **kwargs) -> None:
        log_and_print("[rover] enter move")

    def update(self, **kwargs) -> Optional[str]:
        if not self.rover._path:
            return "idle"

        base_pos, base_yaw = self.rover.get_base_pose()
        goal = self.rover._path[self.rover._path_index]
        delta = goal[:2] - base_pos[:2]
        dist = float(np.linalg.norm(delta))
        if dist < self.rover.goal_tol:
            self.rover._advance_path()
            if not self.rover._path:
                return "idle"
            return None

        heading = math.atan2(delta[1], delta[0])
        yaw_err = (heading - base_yaw + math.pi) % (2 * math.pi) - math.pi

        vx = max(
            -self.rover.max_speed, min(self.rover.kp_lin * dist, self.rover.max_speed)
        )
        omega = max(
            -self.rover.max_omega,
            min(self.rover.kp_ang * yaw_err, self.rover.max_omega),
        )
        self.rover._apply_cmd_vel(vx, omega)
        return None


class TaskCoordinator:
    def __init__(
        self,
        arm: ArmController,
        rover: RoverController,
        config: dict,
    ) -> None:
        self.arm = arm
        self.rover = rover
        self.config = config
        self._waiting_for_base = False

    def update(
        self,
        desired_target: np.ndarray,
        current_mj_q: np.ndarray,
    ) -> np.ndarray:
        base_pos, _ = self.rover.get_base_pose()
        workspace_min = np.array(
            self.config.get("arm_workspace_min", [-0.6, -0.6, 0.0])
        )
        workspace_max = np.array(self.config.get("arm_workspace_max", [0.8, 0.6, 1.0]))

        feasible = self.arm.is_pose_feasible(base_pos, current_mj_q, desired_target)
        if feasible:
            if self._waiting_for_base:
                log_and_print("[task] base reached, arm moving to target")
            self._waiting_for_base = False
            self.rover._path = []
            self.arm.set_target_pose(desired_target, None)
            return desired_target

        if not self._waiting_for_base:
            log_and_print("[task] target out of workspace, moving base")
            self.rover.plan_to_reachable(desired_target, workspace_min, workspace_max)
            self._waiting_for_base = True

        # While moving base, keep arm at reachable boundary.
        rel = desired_target - base_pos
        clamped = clamp_vec(rel, workspace_min, workspace_max)
        arm_target = base_pos + clamped
        self.arm.set_target_pose(arm_target, None)
        return arm_target
