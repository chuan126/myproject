"""Informer encoder adapted for window-level SOC regression.

The implementation keeps the encoder-side ideas from Informer2020 while
matching this project's encoder contract: (batch, seq_len, input_dim) in and
(batch, seq_len, hidden_size) out.
"""

import math
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : inputs.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        nn.init.kaiming_normal_(self.projection.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs.transpose(1, 2)).transpose(1, 2)


class InformerEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = TokenEmbedding(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(inputs) + self.position_embedding(inputs))


class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1, output_attention: bool = False):
        super().__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, _, _, width = queries.shape
        scale = 1.0 / sqrt(width)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        return output.contiguous(), attention if self.output_attention else None


class ProbAttention(nn.Module):
    def __init__(self, factor: int = 5, attention_dropout: float = 0.1, output_attention: bool = False):
        super().__init__()
        self.factor = factor
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_qk(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        sample_k: int,
        n_top: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, key_len, width = keys.shape
        _, _, query_len, _ = queries.shape

        key_expand = keys.unsqueeze(-3).expand(batch, heads, query_len, key_len, width)
        index_sample = torch.randint(key_len, (query_len, sample_k), device=queries.device)
        key_sample = key_expand[:, :, torch.arange(query_len, device=queries.device).unsqueeze(1), index_sample, :]
        query_key_sample = torch.matmul(queries.unsqueeze(-2), key_sample.transpose(-2, -1)).squeeze(-2)

        sparsity = query_key_sample.max(-1)[0] - query_key_sample.mean(-1)
        top_indices = sparsity.topk(n_top, sorted=False)[1]
        reduced_queries = queries[
            torch.arange(batch, device=queries.device)[:, None, None],
            torch.arange(heads, device=queries.device)[None, :, None],
            top_indices,
            :,
        ]
        scores = torch.matmul(reduced_queries, keys.transpose(-2, -1))
        return scores, top_indices

    @staticmethod
    def _initial_context(values: torch.Tensor, query_len: int) -> torch.Tensor:
        context = values.mean(dim=-2)
        return context.unsqueeze(-2).expand(*values.shape[:2], query_len, values.shape[-1]).clone()

    def _update_context(
        self,
        context: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, heads, value_len, _ = values.shape
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context[
            torch.arange(batch, device=values.device)[:, None, None],
            torch.arange(heads, device=values.device)[None, :, None],
            indices,
            :,
        ] = torch.matmul(attention, values).type_as(context)
        if not self.output_attention:
            return context, None
        full_attention = torch.full(
            (batch, heads, value_len, value_len),
            1.0 / value_len,
            dtype=attention.dtype,
            device=attention.device,
        )
        full_attention[
            torch.arange(batch, device=values.device)[:, None, None],
            torch.arange(heads, device=values.device)[None, :, None],
            indices,
            :,
        ] = attention
        return context, full_attention

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, query_len, _, width = queries.shape
        _, key_len, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        sample_k = min(key_len, self.factor * math.ceil(math.log(max(key_len, 2))))
        n_top = min(query_len, self.factor * math.ceil(math.log(max(query_len, 2))))
        scores, indices = self._prob_qk(queries, keys, sample_k=sample_k, n_top=n_top)
        scores = scores * (1.0 / sqrt(width))
        context = self._initial_context(values, query_len)
        context, attention = self._update_context(context, values, scores, indices)
        return context.transpose(2, 1).contiguous(), attention


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("Informer hidden_size must be divisible by n_heads.")
        head_dim = d_model // n_heads
        self.query_projection = nn.Linear(d_model, head_dim * n_heads)
        self.key_projection = nn.Linear(d_model, head_dim * n_heads)
        self.value_projection = nn.Linear(d_model, head_dim * n_heads)
        self.out_projection = nn.Linear(head_dim * n_heads, d_model)
        self.inner_attention = attention
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        queries = self.query_projection(queries).view(batch, query_len, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch, key_len, self.n_heads, -1)
        values = self.value_projection(values).view(batch, key_len, self.n_heads, -1)
        output, attention = self.inner_attention(queries, keys, values)
        output = output.view(batch, query_len, -1)
        return self.out_projection(output), attention


class InformerEncoderLayer(nn.Module):
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(inputs, inputs, inputs)
        inputs = inputs + self.dropout(attended)
        values = self.norm1(inputs)
        values = self.dropout(self.activation(self.conv1(values.transpose(1, 2))))
        values = self.dropout(self.conv2(values).transpose(1, 2))
        return self.norm2(inputs + values)


class InformerDistilLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.conv(inputs.transpose(1, 2))
        values = self.norm(values)
        values = self.activation(values)
        values = self.pool(values)
        return values.transpose(1, 2)


class InformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attention: str = "prob",
        factor: int = 5,
        distil: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Informer num_layers must be at least 1.")
        self.output_dim = hidden_size
        self.embedding = InformerEmbedding(input_dim, hidden_size, dropout)
        attention_cls = ProbAttention if attention == "prob" else FullAttention
        self.layers = nn.ModuleList(
            [
                InformerEncoderLayer(
                    AttentionLayer(
                        attention_cls(factor=factor, attention_dropout=dropout)
                        if attention == "prob"
                        else attention_cls(attention_dropout=dropout),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    d_ff,
                    dropout,
                    activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.distil_layers = nn.ModuleList(
            [InformerDistilLayer(hidden_size) for _ in range(num_layers - 1)]
        ) if distil else None
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.embedding(inputs)
        if self.distil_layers is None:
            for layer in self.layers:
                values = layer(values)
        else:
            for layer, distil_layer in zip(self.layers[:-1], self.distil_layers):
                values = distil_layer(layer(values))
            values = self.layers[-1](values)
        return self.norm(values)
