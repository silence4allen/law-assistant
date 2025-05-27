# -*- coding: utf-8 -*-#
import time
from functools import wraps

from common.log import get_logger

logger = get_logger()


def timer(func):
    """
    装饰器：打印函数的执行时间（以秒为单位）。
    """

    @wraps(func)  # 保留原函数的元信息（如函数名、文档字符串等）
    def wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.time()

        # 执行原函数
        result = func(*args, **kwargs)

        # 记录结束时间
        end_time = time.time()

        # 计算并打印耗时
        elapsed_time = end_time - start_time
        logger.info(f"函数 {func.__name__} 执行耗时: {elapsed_time:.6f} 秒")

        # 返回原函数的结果
        return result

    return wrapper


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
