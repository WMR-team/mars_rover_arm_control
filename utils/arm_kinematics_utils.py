from typing import Callable, Tuple

import numpy as np
import pinocchio
from numpy.linalg import norm, solve
from scipy.optimize import minimize


def print_pinocchio_info(ph_model, log_fn: Callable[[str], None]) -> None:
    log_fn("-" * 80)
    log_fn("pinocchio joints related info")
    for i, jn in enumerate(ph_model.names):
        log_fn(f"{i} {jn} {ph_model.joints[i].nq} {ph_model.joints[i].nv}")
    log_fn("-" * 80)


def mujoco_q_to_pinocchio_q(mujoco_q: np.ndarray) -> np.ndarray:
    pinocchio_q = mujoco_q.copy()
    pinocchio_q[3:7] = [
        mujoco_q[4],
        mujoco_q[5],
        mujoco_q[6],
        mujoco_q[3],
    ]
    return pinocchio_q


def pinocchio_q_to_mujoco_q(pinocchio_q: np.ndarray) -> np.ndarray:
    mujoco_q = pinocchio_q.copy()
    mujoco_q[3:7] = [
        pinocchio_q[6],
        pinocchio_q[3],
        pinocchio_q[4],
        pinocchio_q[5],
    ]
    return mujoco_q


def inverse_arm_kinematics(
    current_q: np.ndarray,
    target_dir: np.ndarray,
    target_pos,
    config: dict,
    ph_model,
    ph_data,
    control_orientation: bool = True,
) -> Tuple[list, bool]:
    arm_idx = np.asarray(range(config["arm_start"], config["arm_end"], 1))
    joint_id = int(config["ik_joint_id"])
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    q = current_q
    eps = float(config["ik_eps"])
    it_max = int(config["ik_max_iters"])
    dt = float(config["ik_dt"])
    damp = float(config["ik_damp"])

    i = 0
    while True:
        pinocchio.forwardKinematics(ph_model, ph_data, q)
        iMd = ph_data.oMi[joint_id].actInv(oMdes)
        err = pinocchio.log(iMd).vector
        if not control_orientation:
            err[3:] = 0.0

        if norm(err) < eps:
            success = True
            break
        if i >= it_max:
            success = False
            break

        J_full = pinocchio.computeJointJacobian(ph_model, ph_data, q, joint_id)
        J = -pinocchio.Jlog6(iMd.inverse()) @ J_full
        if not control_orientation:
            J[3:, :] = 0.0

        J_arm = J[:, arm_idx]
        v_arm = -J_arm.T @ solve(J_arm @ J_arm.T + damp * np.eye(6), err)

        v = np.zeros(ph_model.nv)
        v[arm_idx] = v_arm
        q = pinocchio.integrate(ph_model, q, v * dt)
        i += 1

    return q.flatten().tolist(), success


def inverse_arm_kinematics_bounded(
    current_q: np.ndarray,
    target_dir: np.ndarray,
    target_pos,
    config: dict,
    ph_model,
    ph_data,
    control_orientation: bool = True,
) -> Tuple[list, bool]:
    arm_idx = np.asarray(range(config["arm_start"], config["arm_end"], 1))
    joint_id = int(config["ik_joint_id"])
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
        if not control_orientation:
            err[3:] = 0.0
        return float(err.T @ err)

    res = minimize(
        cost,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": int(config["ik_bounds_max_iters"])},
    )

    q = q0.copy()
    q[arm_idx] = res.x
    pinocchio.forwardKinematics(ph_model, ph_data, q)
    iMd = ph_data.oMi[joint_id].actInv(oMdes)
    err = pinocchio.log(iMd).vector
    if not control_orientation:
        err[3:] = 0.0
    success = bool(res.success) and norm(err) < float(config["ik_eps"])
    return q.flatten().tolist(), success


def inverse_arm_kinematics_bounded_retry(
    current_q: np.ndarray,
    target_dir: np.ndarray,
    target_pos,
    config: dict,
    ph_model,
    ph_data,
    control_orientation: bool = True,
) -> Tuple[list, bool]:
    arm_idx = np.asarray(range(config["arm_start"], config["arm_end"], 1))
    joint_id = int(config["ik_joint_id"])
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
        err = pinocchio.log(iMd).vector
        if not control_orientation:
            err[3:] = 0.0
        return err

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
            options={"maxiter": int(config["ik_bounds_max_iters"])},
        )
        q = q0.copy()
        q[arm_idx] = res.x
        err = compute_err(q)
        return res, q, err

    base_res, base_q, base_err = solve_from(x0)
    best_q = base_q
    best_err = base_err
    best_success = bool(base_res.success) and norm(base_err) < float(config["ik_eps"])

    if norm(base_err) <= float(config["ik_retry_err_thresh"]):
        return best_q.flatten().tolist(), best_success

    rng = np.random.default_rng()
    noise_scale = float(config["ik_retry_noise_scale"])
    attempts = int(config["ik_retry_attempts"])

    for _ in range(max(0, attempts)):
        noise = rng.normal(0.0, noise_scale, size=x0.shape)
        x_init = x0 + noise
        for i, (lo_b, hi_b) in enumerate(bounds):
            if lo_b is not None and x_init[i] < lo_b:
                x_init[i] = lo_b
            if hi_b is not None and x_init[i] > hi_b:
                x_init[i] = hi_b

        res, q, err = solve_from(x_init)
        if norm(err) < norm(best_err):
            best_q = q
            best_err = err
            best_success = bool(res.success) and norm(err) < float(config["ik_eps"])

    return best_q.flatten().tolist(), best_success


def inverse_kinematics(
    current_q: np.ndarray,
    target_dir: np.ndarray,
    target_pos,
    config: dict,
    ph_model,
    ph_data,
    control_orientation: bool = True,
    log_fn: Callable[[str], None] = None,
) -> list:
    joint_id = int(config["ik_joint_id"])
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    q = current_q
    eps = float(config["ik_eps"])
    it_max = int(config["ik_max_iters"])
    dt = float(config["ik_dt"])
    damp = float(config["ik_damp"])

    i = 0
    while True:
        pinocchio.forwardKinematics(ph_model, ph_data, q)
        iMd = ph_data.oMi[joint_id].actInv(oMdes)
        err = pinocchio.log(iMd).vector
        if not control_orientation:
            err[3:] = 0.0

        if norm(err) < eps:
            success = True
            break
        if i >= it_max:
            success = False
            break

        J = pinocchio.computeJointJacobian(ph_model, ph_data, q, joint_id)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        if not control_orientation:
            J[3:, :] = 0.0
        JJt = J.dot(J.T) + damp * np.eye(6)
        v = -J.T @ solve(JJt, err)

        v[: int(config["ik_lock_front_dofs"])] = 0.0
        q = pinocchio.integrate(ph_model, q, v * dt)
        i += 1

    if log_fn is not None:
        log_fn(f"IK iters: {i}, success={success}")

    return q.flatten().tolist()
