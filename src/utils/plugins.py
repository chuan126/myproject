"""插件加载模块。

通过配置声明动态导入外部模块，利用其注册副作用扩展组件。
"""

from importlib import import_module
from typing import Any


def load_plugins(config: dict[str, Any]) -> None:
    """导入配置中声明的插件模块。

    插件模块通过其注册副作用生效（例如注册自定义编码器或池化）。
    重复导入已加载的模块是安全的，因为 Python 会缓存模块导入。
    """
    plugins = config.get("plugins", [])
    if not isinstance(plugins, list) or not all(isinstance(name, str) for name in plugins):
        raise ValueError("plugins must be a list of importable module names.")
    for module_name in plugins:
        import_module(module_name)
