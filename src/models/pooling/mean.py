"""均值池化。对序列时间维取平均。"""

from torch import nn


class MeanPooling(nn.Module):
    """沿时间维取均值：encoded.mean(dim=1)。"""

    def forward(self, encoded):
        """Args:
            encoded: (batch, seq_len, feature_dim)

        Returns:
            (batch, feature_dim)
        """
        return encoded.mean(dim=1)
