"""最大值池化。对序列时间维取最大值。"""

from torch import nn


class MaxPooling(nn.Module):
    """沿时间维取最大值：encoded.max(dim=1).values。"""

    def forward(self, encoded):
        """Args:
            encoded: (batch, seq_len, feature_dim)

        Returns:
            (batch, feature_dim)
        """
        return encoded.max(dim=1).values
