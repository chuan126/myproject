"""工具子包。

本子包提供项目运行所需的基础设施工具，包括：
- config: 配置文件的加载、合并与写入
- logger: 统一日志记录器
- plugins: 插件动态加载机制
- seed: 随机种子设置（可复现性）

对外暴露的公共 API 通过 __all__ 明确声明，便于外部模块通过
`from src.utils import ...` 的方式导入。

在整个项目中的角色：
  作为 src 包的子包，为训练脚本、评估脚本及 src 内部各模块提供横切关注点
  （cross-cutting concerns）的基础支持。
"""

from .config import load_config, read_yaml, write_yaml
from .logger import get_logger
from .plugins import load_plugins
from .seed import seed_everything

# 显式声明本模块的公共接口
__all__ = ["get_logger", "load_config", "load_plugins", "read_yaml", "seed_everything", "write_yaml"]
