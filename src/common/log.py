# -*- coding: utf-8 -*-
import logging
import colorlog
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import threading
from typing import Optional, Union
import datetime

from common.constants import LOG_ENABLE_FILE_LOG, LOG_CONSOLE_ONLY, LOG_LEVEL


class LogConfig:
    """日志配置中心（可通过环境变量覆盖默认值）"""

    # 路径配置（支持相对/绝对路径）
    DEFAULT_LOG_DIR = (
            Path(__file__).parent.parent.parent / "logs"
    )  # 向上三级到python目录
    LOG_DIR = os.getenv(
        "LOG_DIR", DEFAULT_LOG_DIR
    )  # 通过环境变量LOG_DIR配置，默认logs目录
    DEFAULT_LOG_NAME = "system"  # 默认日志名称
    CUSTOM_DEFAULT_NAME = ""  # 默认自定义日志名称
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # 文件日志格式
    COLOR_FORMAT = "%(log_color)s" + LOG_FORMAT  # 控制台带颜色日志格式

    # 日志分割配置
    ROTATE_WHEN = "midnight"  # 按天分割
    ROTATE_INTERVAL = 1  # 分割间隔（天）
    BACKUP_COUNT = 30  # 保留30天日志
    MAX_BYTES = 10 * 1024 * 1024  # 单个日志最大10MB

    # 功能配置
    CUSTOM_LOG_LEVEL = LOG_LEVEL
    ENABLE_FILE_LOG = LOG_ENABLE_FILE_LOG  # 是否启用文件日志
    CONSOLE_ONLY = LOG_CONSOLE_ONLY  # 是否只打印到控制台（优先于ENABLE_FILE_LOG）

    @classmethod
    def get_log_dir(cls) -> Path:
        """解析日志目录路径（自动处理相对/绝对路径）"""
        path = Path(cls.LOG_DIR)
        if not path.is_absolute():  # 如果是相对路径
            base_dir = os.getenv("LOG_BASE_DIR", os.getcwd())  # 获取基准目录
            path = Path(base_dir) / path  # 拼接绝对路径
        return path.resolve()  # 返回标准化绝对路径

    @classmethod
    def get_current_date_str(cls) -> str:
        """获取当前日期字符串，格式为YYYYMMDD"""
        return datetime.datetime.now().strftime("%Y%m%d")


# 线程安全的日志器缓存
_logger_cache = {}
_lock = threading.Lock()  # 线程锁


def _ensure_log_dir(log_dir: Path) -> Path:
    """
    确保日志目录存在且可写
    如果失败则自动降级到/tmp/fallback_logs目录
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # 测试目录可写性
        test_file = log_dir / ".permission_test"
        test_file.touch()
        test_file.unlink()
        return log_dir
    except (OSError, PermissionError) as e:
        # 降级处理：使用临时目录
        fallback_dir = Path("/tmp") / "fallback_logs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        logging.error(f"无法访问日志目录 {log_dir}: {e}，已降级到 {fallback_dir}")
        return fallback_dir


def _init_console_handler(logger: logging.Logger):
    """初始化带颜色的控制台日志处理器"""
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            LogConfig.COLOR_FORMAT,
            log_colors={  # 颜色配置
                "DEBUG": "cyan",  # 青色
                "INFO": "green",  # 绿色
                "WARNING": "yellow",  # 黄色
                "ERROR": "red",  # 红色
                "CRITICAL": "bold_red",  # 加粗红色
            },
        )
    )
    logger.addHandler(console_handler)


class DateRotatingFileHandler(TimedRotatingFileHandler):
    """自定义日期分割文件处理器"""

    def __init__(self, *args, **kwargs):
        self.base_log_name = kwargs.pop("base_log_name", "system")
        super().__init__(*args, **kwargs)
        self.current_date = LogConfig.get_current_date_str()
        self.suffix = ".log"
        self.namer = self._date_namer

    def _date_namer(self, default_name):
        """自定义日志文件名生成规则"""
        dirname, basename = os.path.split(default_name)
        base, ext = os.path.splitext(basename)

        # 如果是主日志文件
        if not base.endswith(self.current_date):
            return os.path.join(
                dirname, f"{self.base_log_name}_{self.current_date}{self.suffix}"
            )

        # 如果是备份文件
        parts = base.split(".")
        if len(parts) > 1:
            return os.path.join(
                dirname,
                f"{self.base_log_name}_{self.current_date}{self.suffix}.{parts[-1]}",
            )
        return default_name

    def shouldRollover(self, record):
        """检查是否需要滚动日志"""
        new_date = LogConfig.get_current_date_str()
        if new_date != self.current_date:
            self.current_date = new_date
            return 1
        return super().shouldRollover(record)


def _init_file_handler(logger: logging.Logger, log_name: str):
    """初始化文件日志处理器（带自动分割功能）"""
    # 如果配置为仅控制台输出或文件日志被禁用，则不初始化文件处理器
    if LogConfig.CONSOLE_ONLY or not LogConfig.ENABLE_FILE_LOG:
        return

    # 清理现有的文件处理器（防止重复）
    for handler in logger.handlers[:]:  # 遍历副本
        if isinstance(
                handler, (logging.FileHandler, logging.handlers.BaseRotatingHandler)
        ):
            logger.removeHandler(handler)
            handler.close()  # 确保资源释放

    # 准备日志文件路径
    log_dir = _ensure_log_dir(LogConfig.get_log_dir())
    base_log_name = f"{log_name}_{LogConfig.get_current_date_str()}"
    log_file = log_dir / f"{base_log_name}.log"

    # 配置按时间分割的处理器
    file_handler = DateRotatingFileHandler(
        filename=str(log_file),
        when=LogConfig.ROTATE_WHEN,
        interval=LogConfig.ROTATE_INTERVAL,
        backupCount=LogConfig.BACKUP_COUNT,
        encoding="utf-8",
        base_log_name=log_name,
    )
    file_handler.setFormatter(logging.Formatter(LogConfig.LOG_FORMAT))
    logger.addHandler(file_handler)


def set_default_logger_name(name: str):
    """设置默认的logger名称"""
    LogConfig.CUSTOM_DEFAULT_NAME = name


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器实例（线程安全）
    :param name: 日志器名称，None表示使用默认日志器
    :return: 配置好的logging.Logger实例
    """
    if name is None:
        log_name = LogConfig.CUSTOM_DEFAULT_NAME or LogConfig.DEFAULT_LOG_NAME
    else:
        log_name = name

    # 双检锁模式确保线程安全
    if log_name not in _logger_cache:
        with _lock:
            if log_name not in _logger_cache:
                logger = logging.getLogger(log_name)
                logger.setLevel(LogConfig.CUSTOM_LOG_LEVEL)  # 默认日志级别
                logger.propagate = False  # 防止日志向上传播造成重复

                # 初始化处理器
                _init_console_handler(logger)
                _init_file_handler(logger, log_name)

                _logger_cache[log_name] = logger

    return _logger_cache[log_name]


def set_file_logging(enabled: bool):
    """动态开关文件日志功能"""
    LogConfig.ENABLE_FILE_LOG = enabled
    # 重新配置所有已存在的日志器
    for logger in _logger_cache.values():
        _init_file_handler(logger, logger.name)


def set_console_only(enabled: bool):
    """
    设置是否仅输出到控制台
    :param enabled: True表示只输出到控制台，False则根据ENABLE_FILE_LOG决定
    """
    LogConfig.CONSOLE_ONLY = enabled
    # 重新配置所有已存在的日志器
    for logger in _logger_cache.values():
        _init_file_handler(logger, logger.name)


def set_log_dir(new_dir: Union[str, Path]):
    """运行时修改日志目录"""
    LogConfig.LOG_DIR = str(new_dir)
    # 为所有已存在的日志器重新初始化文件处理器
    for logger in _logger_cache.values():
        _init_file_handler(logger, logger.name)
