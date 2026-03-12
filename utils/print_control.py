import sys
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from typing import Callable, Any, Optional


@contextmanager
def redirect_print(enable: bool = True, stream: Optional[StringIO] = None):
    """
    enable=True  时：正常打印到原来的 stdout
    enable=False 时：把所有 print 重定向到一个 StringIO，不输出到终端   
    """
    if enable:
        # 不做任何事，保持原样
        yield None
        return

    # enable=False 的情况：临时把 sys.stdout 换成 StringIO
    old_stdout = sys.stdout
    buffer = stream if stream is not None else StringIO()
    try:
        sys.stdout = buffer
        yield buffer
    finally:
        sys.stdout = old_stdout


def control_print(enable: bool = True):
    """
    作为装饰器使用：
    @control_print(enable=False)
    def func(...):
        print("这句话不会出现在终端")

    enable=True 表示正常打印
    enable=False 表示函数里的所有 print 都被吃掉（重定向到 buffer）
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 你如果想要拿到函数内部的打印内容，可以保存 buffer
            with redirect_print(enable=enable) as buf:
                result = func(*args, **kwargs)

            # 例如：在这里你可以根据需要处理 buf.getvalue()
            # if buf is not None:
            #     logs = buf.getvalue()
            #     ...  做日志记录/写文件 等操作

            return result

        return wrapper

    return decorator
