import time
from functools import wraps


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

            print(f"[timeit] {func.__name__} took {elapsed:.3f} {unit}")
            return result

        return wrapper

    return decorator
