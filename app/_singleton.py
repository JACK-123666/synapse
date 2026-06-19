"""模块级单例。"""

from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def singleton(cls: Callable[..., T]) -> Callable[..., T]:
    """将类变为模块级单例，首次调用时初始化，后续复用。"""
    _inst: Optional[T] = None

    def get(*args: Any, **kwargs: Any) -> T:
        nonlocal _inst
        if _inst is None:
            _inst = cls(*args, **kwargs)
        return _inst

    get.__name__ = f"get_{cls.__name__.lower()}"
    return get
