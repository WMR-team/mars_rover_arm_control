import time
from collections import deque
from functools import wraps
from typing import Callable, Optional

from mars_rover_arm_control.utils.print_control import log_and_print


class FPSCounter:
    """
    轻量级帧率统计器，适合在 while 循环中使用。

    用法示例
    --------
    fps = FPSCounter(print_every=100, label="control_worker")
    while run_flag.value:
        fps.tick()       # 每次迭代调用一次即可
        ... 控制逻辑 ...

    参数
    ----
    print_every : int
        每隔多少帧打印一次帧率，0 表示不自动打印。
    window : int
        用于计算滑动平均帧率的窗口大小（帧数）。
    label : str
        打印时显示的名称，方便区分多个计数器。
    """

    def __init__(
        self,
        print_every: int = 100,
        window: int = 200,
        label: str = "loop",
    ) -> None:
        self.print_every = print_every
        self.label = label
        self._timestamps: deque = deque(maxlen=window)
        self._count: int = 0

    def tick(self) -> Optional[float]:
        """
        记录一帧并（可选）打印帧率。

        返回值
        ------
        当前滑动平均 FPS（帧/秒），若窗口内帧数不足 2 则返回 None。
        """
        now = time.perf_counter()
        self._timestamps.append(now)
        self._count += 1

        fps = self.fps()

        if self.print_every > 0 and self._count % self.print_every == 0:
            if fps is not None:
                log_and_print(f"[fps_counter] {self.label}: {fps:.1f} Hz")
            else:
                log_and_print(f"[fps_counter] {self.label}: collecting samples...")

        return fps

    def fps(self) -> Optional[float]:
        """返回当前窗口内的滑动平均帧率，不足 2 帧时返回 None。"""
        if len(self._timestamps) < 2:
            return None
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return None
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """重置所有统计数据。"""
        self._timestamps.clear()
        self._count = 0


def fps_logger(
    print_every: int = 100,
    window: int = 200,
    label: Optional[str] = None,
):
    """
    装饰器版本：统计被装饰函数的调用帧率。
    适合将循环体提取为独立函数后使用。

    用法示例
    --------
    @fps_logger(print_every=100, label="control_step")
    def control_step(...):
        ... 单次控制逻辑 ...

    while run_flag.value:
        control_step(...)

    参数
    ----
    print_every : int
        每隔多少次调用打印一次帧率，0 表示不自动打印。
    window : int
        滑动平均窗口大小（调用次数）。
    label : str | None
        打印名称，默认使用被装饰函数的名称。
    """

    def decorator(func: Callable) -> Callable:
        _label = label if label is not None else func.__name__
        counter = FPSCounter(print_every=print_every, window=window, label=_label)

        @wraps(func)
        def wrapper(*args, **kwargs):
            counter.tick()
            return func(*args, **kwargs)

        # 暴露 counter 对象，方便外部查询帧率
        wrapper.fps_counter = counter  # type: ignore[attr-defined]
        return wrapper

    return decorator
