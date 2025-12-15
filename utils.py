# utils.py
import time
import logging
from functools import wraps
from typing import Callable, Any
import asyncio

logger = logging.getLogger(__name__)


def timeit_sync(operation: str = None):
    """Декоратор для синхронных функций"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            op_name = operation or func.__name__
            logger.info(
                "⏱️  %s: %.3f seconds",
                op_name,
                duration
            )
            return result
        return wrapper
    return decorator


def timeit_async(operation: str = None):
    """Декоратор для асинхронных функций"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start

            op_name = operation or func.__name__
            logger.info(
                "⏱️  %s: %.3f seconds",
                op_name,
                duration
            )
            return result
        return wrapper
    return decorator


def auto_timeit(operation: str = None):
    """Автоматически выбирает sync/async декоратор"""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            return timeit_async(operation)(func)
        return timeit_sync(operation)(func)
    return decorator


# Пример использования
if __name__ == "__main__":
    @auto_timeit("test_sync")
    def test_sync():
        time.sleep(0.1)
        return "done"

    @auto_timeit("test_async")
    async def test_async():
        await asyncio.sleep(0.1)
        return "done"

    # Тест
    test_sync()
    asyncio.run(test_async())
