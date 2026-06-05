"""插件加载模块。

本模块实现基于配置声明的插件动态导入机制。
插件是普通的 Python 模块，在被 import 时通过副作用（side-effect）向全局注册表
注册自定义组件（如自定义编码器、池化层、融合策略等）。

设计思想：
  用户在 YAML 配置文件中声明 `plugins` 字段（一个模块名列表），
  程序启动时调用本模块的 load_plugins 函数逐一导入这些模块。
  模块导入触发的副作用自动完成注册，核心代码无需感知插件的具体实现。
  重复导入已加载的模块是安全的，因为 Python 的模块缓存机制（sys.modules）
  确保每个模块只执行一次。

在整个项目中的角色：
  被 scripts/train.py 和 scripts/eval.py 在初始化阶段调用，
  在模型和数据加载之前完成插件注册，确保所有自定义组件可用。
"""

from importlib import import_module
from typing import Any


def load_plugins(config: dict[str, Any]) -> None:
    """导入配置中声明的所有插件模块。

    从配置字典中读取 `plugins` 键（预期为字符串列表），
    依次调用 import_module 导入每个模块。
    插件模块的副作用（如装饰器注册、子类注册）会在导入时自动完成。

    参数:
        config: 完整的配置字典，可选的 `plugins` 键应为字符串列表，
                每个字符串为可导入的模块名（如 "mypackage.myplugin"）。

    异常:
        ValueError: 如果 `plugins` 键存在但格式不正确
                   （非列表或列表元素非字符串）。
        ModuleNotFoundError: 如果指定的模块无法找到（由 import_module 抛出）。

    注意事项:
        - `plugins` 键不存在时静默跳过，不产生错误
        - 模块导入顺序与列表中的声明顺序一致
        - 重复导入同一模块不会重复执行模块代码（Python 模块缓存保障）
    """
    plugins = config.get("plugins", [])
    # 类型校验：确保 plugins 是字符串列表
    if not isinstance(plugins, list) or not all(isinstance(name, str) for name in plugins):
        raise ValueError("plugins must be a list of importable module names.")
    for module_name in plugins:
        # import_module 等价于 `import module_name`，副作用自动生效
        import_module(module_name)
