"""双流 SOC 估计模型。

本模块定义 DualStreamSOCModel 类，实现双分支架构：将输入特征按列拆分为
主分支（main）和力学分支（mech），两支独立编码和池化后通过融合层合并，
再由共享的预测头输出 SOC 估计值。

在整个项目中的角色：
- 作为双流架构的模型载体，被 registry.py 中的 _build_dual_stream 组装
- 适用于需要显式分离不同性质特征（如电学参数 vs 力学参数）的场景
- 每个分支可以独立选择编码器类型和池化策略
"""

from torch import Tensor, nn


class DualStreamSOCModel(nn.Module):
    """双流 SOC 估计模型。

    架构总览：
    - 输入根据 main_indices 和 mech_indices 被按列切片，分成两路
    - 每路独立经过各自的编码器 → 池化层，提取分支特征
    - 两路特征经融合层合并为统一的特征表示
    - 融合特征送入预测头得到最终 SOC 估计值

    设计意图：
    - 分离不同物理含义的特征，避免编码器混淆不同类型的信号
    - 允许两个分支使用不同的编码器架构（如主分支用 LSTM，力学分支用 TCN）
    - 融合层可学习两路特征的最优组合方式（拼接或门控）

    Args:
        main_indices: 主分支特征在全部特征列中的索引列表
        mech_indices: 力学分支特征在全部特征列中的索引列表
        main_encoder: 主分支的时序编码器
        main_pooling: 主分支的池化层
        mech_encoder: 力学分支的时序编码器
        mech_pooling: 力学分支的池化层
        fusion: 特征融合模块，将两个分支的特征合并
        head: 回归预测头
    """

    def __init__(
        self,
        main_indices: list[int],
        mech_indices: list[int],
        main_encoder: nn.Module,
        main_pooling: nn.Module,
        mech_encoder: nn.Module,
        mech_pooling: nn.Module,
        fusion: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.main_indices = main_indices
        self.mech_indices = mech_indices
        self.main_encoder = main_encoder
        self.main_pooling = main_pooling
        self.mech_encoder = mech_encoder
        self.mech_pooling = mech_pooling
        self.fusion = fusion
        self.head = head

    def forward(self, inputs: Tensor) -> Tensor:
        """前向传播：双流并行处理 → 融合 → 预测。

        数据流：
        1. 按 main_indices/mech_indices 从输入张量中切片
        2. 两个分支并行编码和池化（逻辑上并行，实现上顺序执行）
        3. 融合两路特征
        4. 预测头输出 SOC

        Args:
            inputs: 输入张量，形状 (batch, seq_len, input_dim)，
                    其中 input_dim 为全部特征列的总数

        Returns:
            SOC 预测值，形状 (batch, 1)
        """
        # 按列索引切片，将完整输入拆分为两个分支的输入
        main_inputs = inputs[:, :, self.main_indices]
        mech_inputs = inputs[:, :, self.mech_indices]

        # 各分支独立编码
        main_encoded = self.main_encoder(main_inputs)
        mech_encoded = self.mech_encoder(mech_inputs)

        # 各分支独立池化，压缩时间维度
        main_features = self.main_pooling(main_encoded)
        mech_features = self.mech_pooling(mech_encoded)

        # 融合两个分支的特征
        fused_features = self.fusion(main_features, mech_features)
        return self.head(fused_features)
