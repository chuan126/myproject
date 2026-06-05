"""src 包——电池 SOC 估计项目的核心源码。

本包包含项目的所有核心模块：
- data: 数据处理、转换和数据集定义
- models: 模型架构（编码器、融合层等）
- evaluation: 评估指标和可视化
- utils: 配置、日志、插件、随机种子等基础设施
- training: 训练循环和训练器
- experiment: 实验编排（设备选择、评估流程）

在整个项目中的角色：
  作为项目的顶层 Python 包，使 scripts/ 目录下的脚本可以通过
  `from src.xxx import ...` 的方式导入项目内部模块。
"""
