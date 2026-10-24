import torch
from torch import nn
from torch.nn import functional as F

from .blocks import EncoderBlock, DecoderBlock


class BaselinePlusPlusModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        embedding_dim: int = 512,
        pad_id: int = 0,
        num_encoder_layers: int = 5,
        num_decoder_layers: int = 7,
        num_heads: int = 8,
        dim_glu: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim,
                    num_heads,
                    dim_glu,
                    dropout,
                    max_seq_len,
                    qk_norm=qk_norm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = nn.RMSNorm(embedding_dim)

        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim,
                    num_heads,
                    dim_glu,
                    dropout,
                    max_seq_len,
                    num_kv_heads=num_kv_heads,
                    qk_norm=qk_norm,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.RMSNorm(embedding_dim)

        self.linear_head = nn.Linear(embedding_dim, vocab_size)

    def encode(self, input_ids, encoder_key_padding_mask=None):
        curr_embeddings = self.embedding(input_ids)

        for encoder in self.encoders:
            curr_embeddings = encoder(curr_embeddings, encoder_key_padding_mask)

        out = self.encoder_norm(curr_embeddings)
        return out

    def decode(
        self,
        input_ids,
        encoder_outputs,
        decoder_key_padding_mask=None,
        encoder_key_padding_mask=None,
    ):
        curr_embeddings = self.embedding(input_ids)

        for decoder in self.decoders:
            curr_embeddings = decoder(
                curr_embeddings,
                encoder_outputs,
                decoder_key_padding_mask,
                encoder_key_padding_mask,
            )

        out = self.decoder_norm(curr_embeddings)
        return out

    def decode_step(
        self,
        input_ids,
        encoder_outputs,
        encoder_key_padding_mask=None,
        kv_caches=None,
    ):
        x = self.embedding(input_ids)
        new_kv_caches = []

        for i, decoder in enumerate(self.decoders):
            layer_kv_cache = None if kv_caches is None else kv_caches[i]

            x, layer_kv_cache = decoder.forward_step(
                x,
                encoder_outputs,
                encoder_key_padding_mask=encoder_key_padding_mask,
                kv_cache=layer_kv_cache,
            )
            new_kv_caches.append(layer_kv_cache)

        x = self.decoder_norm(x)
        logits = self.linear_head(x[:, 0])

        return logits, new_kv_caches

    def decode_logits(
        self,
        input_ids,
        encoder_outputs,
        decoder_key_padding_mask=None,
        encoder_key_padding_mask=None,
        *,
        last_only: bool = False,
    ):
        outputs = self.decode(
            input_ids,
            encoder_outputs,
            decoder_key_padding_mask,
            encoder_key_padding_mask,
        )
        if last_only:
            outputs = outputs[:, -1]
        return self.linear_head(outputs)

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_key_padding_mask=None,
        decoder_key_padding_mask=None,
    ):
        encoder_outputs = self.encode(encoder_input_ids, encoder_key_padding_mask)
        return self.decode_logits(
            decoder_input_ids,
            encoder_outputs,
            decoder_key_padding_mask,
            encoder_key_padding_mask,
        )
