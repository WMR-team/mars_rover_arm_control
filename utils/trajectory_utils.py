import numpy as np


def trajectory_point(config: dict, t: float) -> np.ndarray:
    traj_type = str(config.get("trajectory_type", "line_y")).lower()
    center = np.array(
        config.get("trajectory_center", config["target_start"]), dtype=float
    )
    scale = np.array(config.get("trajectory_scale", [0.15, 0.25, 0.0]), dtype=float)
    period = float(config.get("trajectory_period", 8.0))
    w = 2.0 * np.pi / max(period, 1e-6)

    if traj_type == "figure8":
        x = scale[0] * np.sin(w * t)
        y = scale[1] * 1.6 * np.cos(w * t)
        z = scale[2] * np.sin(2.0 * w * t)
        return center + np.array([x, y, z], dtype=float)

    if traj_type == "heart":
        s = w * t
        x = scale[0] * np.cos(s)
        y = scale[1] * 2.0 / 1.25 * (np.sin(s) ** 3)
        z = (
            scale[2]
            / 8.0
            / 1.25
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

    y_min = float(config["target_y_min"])
    y_max = float(config["target_y_max"])
    y_speed = float(config["target_y_speed"])
    y_span = max(y_max - y_min, 1e-6)
    phase = (y_speed * t) / y_span
    tri = 2.0 * np.abs(phase - np.floor(phase + 0.5))
    y = y_min + (y_max - y_min) * tri
    return np.array([center[0], y, center[2]], dtype=float)
