"""统一日志记录器模块。

本模块提供项目全局的日志记录器（Logger），基于 Python 标准库 logging 实现单例模式：
首次调用时配置 StreamHandler 和日志格式，后续调用返回同一个已配置的实例。
这种设计避免了在多个模块中重复配置日志，也防止了重复的日志输出。

日志格式为：
    YYYY-MM-DD HH:MM:SS,mmm | LEVEL | message

日志级别默认为 INFO，可通过标准 logging API 动态调整。

在整个项目中的角色：
  被训练脚本、评估脚本及 src 内部各模块调用，提供一致的日志输出格式，
  便于在控制台和日志文件中追踪实验进度和排查问题。
"""

import logging


def get_logger(name: str = "soc") -> logging.Logger:
    """获取或创建项目全局日志记录器。

    首次调用时创建并配置一个带 StreamHandler 的 Logger；
    后续调用（同一 name）返回已配置的实例，不会重复添加 handler。

    参数:
        name: 日志记录器的名称，默认为 "soc"。
              通过该名称可在不同模块间共享同一实例。

    返回:
        配置好的 logging.Logger 实例，日志级别为 INFO，
        输出到 stderr，格式包含时间戳、日志级别和消息。

    注意事项:
        - 如果通过其他方式预先对同名 logger 添加了 handler，
          本函数不会重复添加（通过 hasHandlers 判断）
        - 子模块可通过 `logging.getLogger("soc")` 直接获取，
          但建议统一使用本函数以保持一致性
    """
    logger = logging.getLogger(name)
    # 仅在首次调用时配置 handler，避免重复输出
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
