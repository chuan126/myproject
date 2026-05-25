"""最后时间步池化。取序列最后一个时间步的输出。"""

from torch import nn


class LastPooling(nn.Module):
    """取序列最后一个时间步：encoded[:, -1, :]。

    对应原项目 LSTM 中 out[:, -1, :] 的逻辑。
    """

    def forward(self, encoded):
        """Args:
            encoded: (batch, seq_len, feature_dim)

        Returns:
            (batch, feature_dim)
        """
        return encoded[:, -1, :]
