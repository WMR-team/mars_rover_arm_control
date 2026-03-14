import time
from contextlib import contextmanager
from functools import wraps

from mars_rover_arm_control.utils.print_control import log_and_print


def timeit(unit: str = "ms"):
    """
    一个用于计时的装饰器。

    参数:
        unit: 输出时间单位，支持 "s"（秒），"ms"（毫秒），"us"（微秒）
    """
    unit = unit.lower()
    if unit not in ("s", "ms", "us"):
        raise ValueError("unit must be one of: 's', 'ms', 'us'")

    factor_map = {
        "s": 1,
        "ms": 1_000,
        "us": 1_000_000,
    }
    factor = factor_map[unit]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()  # 高精度计时
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = (end - start) * factor

            log_and_print(f"[timeit] {func.__name__} took {elapsed:.3f} {unit}")
            return result

        return wrapper

    return decorator


@contextmanager
def warn_if_overrun(
    expected_s: float,
    label: str = "control",
    unit: str = "ms",
    color: bool = True,
):
    """
    监控代码块执行时间，超过预计时间则输出警告。

    参数:
        expected_s: 预计上限时间（秒）
        label: 输出标识
        unit: 输出单位（"s" | "ms" | "us"）
        color: 是否使用红色输出
    """
    unit = unit.lower()
    if unit not in ("s", "ms", "us"):
        raise ValueError("unit must be one of: 's', 'ms', 'us'")

    factor_map = {
        "s": 1,
        "ms": 1_000,
        "us": 1_000_000,
    }
    factor = factor_map[unit]

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_s = time.perf_counter() - start
        if expected_s > 0 and elapsed_s > expected_s:
            elapsed = elapsed_s * factor
            budget = expected_s * factor
            msg = (
                f"[control_overrun] {label} took {elapsed:.3f} {unit} "
                f"(budget {budget:.3f} {unit})"
            )
            if color:
                msg = f"\033[31m{msg}\033[0m"
            log_and_print(msg)
