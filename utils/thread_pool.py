import multiprocessing as mp
import threading
import time
from typing import Callable, Dict, Optional

from mars_rover_arm_control.utils.fps_counter import FPSCounter
from mars_rover_arm_control.utils.print_control import log_and_print


class PeriodicTask:
    def __init__(
        self,
        name: str,
        func: Callable[[], None],
        hz: float,
        print_every: int = 200,
        warn_overrun: bool = True,
    ) -> None:
        self.name = name
        self.func = func
        self.hz = float(hz)
        self.print_every = print_every
        self.warn_overrun = warn_overrun
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fps = FPSCounter(print_every=0, label=name)
        self._count = 0
        self._overruns = 0

    def start(self) -> None:
        if self.hz <= 0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        period = 1.0 / self.hz if self.hz > 0 else 0.0
        next_time = time.perf_counter()
        while not self._stop.is_set():
            start = time.perf_counter()
            try:
                self.func()
            except Exception:
                # Do not kill the thread on errors
                pass
            self._fps.tick()
            elapsed = time.perf_counter() - start
            if self.warn_overrun and period > 0 and elapsed > period:
                self._overruns += 1
                log_and_print(
                    f"\033[31m[thread_overrun] {self.name} took {elapsed * 1000:.2f} ms "
                    f"(budget {period * 1000:.2f} ms)\033[0m"
                )
            self._count += 1
            if self.print_every > 0 and self._count % self.print_every == 0:
                actual = self._fps.fps()
                if actual is None:
                    actual_str = "n/a"
                else:
                    actual_str = f"{actual:.1f}"
                log_and_print(
                    f"[thread_stats] {self.name}: actual {actual_str} Hz / "
                    f"target {self.hz:.1f} Hz / overruns {self._overruns}"
                )
            next_time += period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()


class ProcessTask:
    def __init__(
        self,
        name: str,
        func: Callable[[], None],
        hz: float,
        print_every: int = 200,
        warn_overrun: bool = True,
    ) -> None:
        self.name = name
        self.func = func
        self.hz = float(hz)
        self.print_every = print_every
        self.warn_overrun = warn_overrun
        self._process: Optional[mp.Process] = None
        self._stop = mp.Event()

    def start(self) -> None:
        if self.hz <= 0:
            return
        if self._process is not None and self._process.is_alive():
            return
        self._process = mp.Process(target=self._run, name=self.name, daemon=True)
        self._process.start()

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=2.0)

    def _run(self) -> None:
        fps = FPSCounter(print_every=0, label=self.name)
        count = 0
        overruns = 0
        period = 1.0 / self.hz if self.hz > 0 else 0.0
        next_time = time.perf_counter()
        while not self._stop.is_set():
            start = time.perf_counter()
            try:
                self.func()
            except Exception:
                pass
            fps.tick()
            elapsed = time.perf_counter() - start
            if self.warn_overrun and period > 0 and elapsed > period:
                overruns += 1
                log_and_print(
                    f"\033[31m[proc_overrun] {self.name} took {elapsed * 1000:.2f} ms "
                    f"(budget {period * 1000:.2f} ms)\033[0m"
                )
            count += 1
            if self.print_every > 0 and count % self.print_every == 0:
                actual = fps.fps()
                if actual is None:
                    actual_str = "n/a"
                else:
                    actual_str = f"{actual:.1f}"
                log_and_print(
                    f"[proc_stats] {self.name}: actual {actual_str} Hz / "
                    f"target {self.hz:.1f} Hz / overruns {overruns}"
                )
            next_time += period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()


class ThreadPool:
    def __init__(self) -> None:
        self._tasks: Dict[str, PeriodicTask] = {}
        self._proc_tasks: Dict[str, ProcessTask] = {}

    def add_task(
        self,
        name: str,
        func: Callable[[], None],
        hz: float,
        print_every: int = 200,
        warn_overrun: bool = True,
    ) -> None:
        self._tasks[name] = PeriodicTask(
            name, func, hz, print_every=print_every, warn_overrun=warn_overrun
        )

    def add_process_task(
        self,
        name: str,
        func: Callable[[], None],
        hz: float,
        print_every: int = 200,
        warn_overrun: bool = True,
    ) -> None:
        self._proc_tasks[name] = ProcessTask(
            name, func, hz, print_every=print_every, warn_overrun=warn_overrun
        )

    def start(self) -> None:
        for task in self._tasks.values():
            task.start()
        for task in self._proc_tasks.values():
            task.start()

    def stop(self) -> None:
        for task in self._tasks.values():
            task.stop()
        for task in self._proc_tasks.values():
            task.stop()
