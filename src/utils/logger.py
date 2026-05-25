"""统一日志记录器。"""

import logging


def get_logger(name: str = "soc") -> logging.Logger:
    """获取或创建名为 soc 的日志记录器。

    首次调用时配置 StreamHandler 和格式，后续返回同一实例。

    Args:
        name: 日志记录器名称

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
