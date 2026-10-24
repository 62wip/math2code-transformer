"""RoPE attention, gated feed-forward layers and pre-norm Transformer blocks."""

import torch
from torch import nn
from torch.nn import functional as F


class RoPE(nn.Module):
    """Apply rotary position embeddings to adjacent pairs of head features."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 1024,
        base: float = 10000.0,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        frequencies = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        angles = positions[:, None] * frequencies[None, :]

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        length = x.size(-2)
        positions = slice(offset, offset + length)
        shape = (1, 1, length, self.head_dim // 2)
        sin = self.sin_cache[positions].reshape(shape).to(dtype=x.dtype)
        cos = self.cos_cache[positions].reshape(shape).to(dtype=x.dtype)

        even = x[..., 0::2]
        odd = x[..., 1::2]

        new_even = even * cos - odd * sin
        new_odd = even * sin + odd * cos

        return torch.stack((new_even, new_odd), dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    """Self/cross-attention with optional grouped KV heads, RoPE and masking."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_rope: bool = True,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )
        assert self.num_kv_heads > 0 and num_heads % self.num_kv_heads == 0, (
            "num_kv_heads must be positive and divide num_heads"
        )

        self.head_dim = embedding_dim // num_heads
        self.qk_norm = qk_norm
        self.qk_scale = (
            nn.Parameter(torch.tensor(self.head_dim**0.5)) if qk_norm else None
        )
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len) if use_rope else None
        self.dropout = dropout

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        kv_dim = self.num_kv_heads * self.head_dim
        self.key_projection = nn.Linear(embedding_dim, kv_dim)
        self.value_projection = nn.Linear(embedding_dim, kv_dim)

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    @staticmethod
    def _normalize_heads(x: torch.Tensor) -> torch.Tensor:
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return F.normalize(values, dim=-1, eps=1e-6).to(dtype=x.dtype)

    def _attend(self, query, key, value, attn_mask=None, is_causal=False):
        if self.num_kv_heads != self.num_heads:
            # Equal head counts allow efficient SDPA on T4. Repeat only the
            # attention inputs, leaving the projections and KV cache compact.
            repeats = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=False,
            scale=1.0 if self.qk_norm else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        source = x if context is None else context

        query = self.query_projection(x)
        key = self.key_projection(source)
        value = self.value_projection(source)

        batch, query_length, _ = x.shape
        source_length = source.size(1)

        query_shape = (batch, query_length, self.num_heads, self.head_dim)
        source_shape = (batch, source_length, self.num_kv_heads, self.head_dim)
        query = query.reshape(query_shape).transpose(1, 2)
        key = key.reshape(source_shape).transpose(1, 2)
        value = value.reshape(source_shape).transpose(1, 2)

        if self.qk_norm:
            query = self._normalize_heads(query) * self.qk_scale
            key = self._normalize_heads(key)

        if self.rope is not None:
            query = self.rope(query)
            key = self.rope(key)

        attn_mask = None
        sdpa_causal = is_causal

        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]

            if is_causal:
                causal = torch.ones(
                    query_length, query_length, dtype=torch.bool, device=query.device
                ).tril()
                attn_mask = attn_mask & causal

            sdpa_causal = False

        attended = self._attend(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=sdpa_causal,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_length, -1)
        return self.output_projection(attended)

    def forward_step(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_length, _ = x.shape
        if query_length != 1:
            raise ValueError("forward_step expects exactly one token.")

        query = self.query_projection(x)

        query_shape = (batch, 1, self.num_heads, self.head_dim)
        query = query.reshape(query_shape).transpose(1, 2)
        if self.qk_norm:
            query = self._normalize_heads(query) * self.qk_scale

        if context is not None and kv_cache is not None:
            key, value = kv_cache
        else:
            source = x if context is None else context

            key = self.key_projection(source)
            value = self.value_projection(source)

            source_shape = (batch, source.size(1), self.num_kv_heads, self.head_dim)
            key = key.reshape(source_shape).transpose(1, 2)
            value = value.reshape(source_shape).transpose(1, 2)
            if self.qk_norm:
                key = self._normalize_heads(key)

            if context is None:
                past_length = 0 if kv_cache is None else kv_cache[0].size(-2)
                if self.rope is not None:
                    query = self.rope(query, offset=past_length)
                    key = self.rope(key, offset=past_length)

                if kv_cache is not None:
                    key_cache, value_cache = kv_cache
                    key = torch.cat((key_cache, key), dim=-2)
                    value = torch.cat((value_cache, value), dim=-2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]

        attended = self._attend(
            query,
            key,
            value,
            attn_mask=attn_mask,
        )
        attended = attended.transpose(1, 2).reshape(batch, 1, -1)
        return self.output_projection(attended), (key, value)


class GLU(nn.Module):
    """Gated feed-forward layer; SiLU activation gives SwiGLU."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.gate_projection = nn.Linear(embedding_dim, hidden_dim)
        self.value_projection = nn.Linear(embedding_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate_projection(x))
        value = self.value_projection(x)

        return self.output_projection(gate * value)


class EncoderBlock(nn.Module):
    """Pre-RMSNorm self-attention and SwiGLU with residual connections."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        qk_norm: bool = False,
    ):
        super().__init__()

        self.attention_norm = nn.RMSNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            qk_norm=qk_norm,
        )

        self.ffn_norm = nn.RMSNorm(embedding_dim)
        self.ffn = GLU(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            activation=nn.SiLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attention_out = self.attention(
            self.attention_norm(x),
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attention_out)

        ffn_out = self.ffn(self.ffn_norm(x))
        return x + self.dropout(ffn_out)


class DecoderBlock(nn.Module):
    """Pre-RMSNorm causal attention, cross-attention and SwiGLU."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()

        self.self_attention_norm = nn.RMSNorm(embedding_dim)
        self.self_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
        )

        self.cross_attention_norm = nn.RMSNorm(embedding_dim)
        self.cross_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rope=False,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
        )

        self.ffn_norm = nn.RMSNorm(embedding_dim)
        self.ffn = GLU(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            activation=nn.SiLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_key_padding_mask: torch.Tensor | None,
        encoder_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attention_out = self.self_attention(
            self.self_attention_norm(x),
            is_causal=True,
        )
        x = x + self.dropout(attention_out)

        attention_out = self.cross_attention(
            self.cross_attention_norm(x),
            context=encoder_output,
            key_padding_mask=encoder_key_padding_mask,
        )
        x = x + self.dropout(attention_out)

        ffn_out = self.ffn(self.ffn_norm(x))
        return x + self.dropout(ffn_out)

    def forward_step(
        self,
        x,
        encoder_output,
        encoder_key_padding_mask=None,
        kv_cache=None,
    ):
        self_kv_cache = None if kv_cache is None else kv_cache["self"]
        cross_kv_cache = None if kv_cache is None else kv_cache["cross"]

        attention_out, self_kv_cache = self.self_attention.forward_step(
            self.self_attention_norm(x),
            kv_cache=self_kv_cache,
        )
        x = x + self.dropout(attention_out)

        attention_out, cross_kv_cache = self.cross_attention.forward_step(
            self.cross_attention_norm(x),
            context=encoder_output,
            key_padding_mask=encoder_key_padding_mask,
            kv_cache=cross_kv_cache,
        )
        x = x + self.dropout(attention_out)

        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)

        cache = {"self": self_kv_cache, "cross": cross_kv_cache}

        return x, cache
