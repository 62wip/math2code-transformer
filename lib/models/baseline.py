import torch
from torch import nn
from torch.nn import functional as F


class BaselineTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 256,
        pad_id: int = 0,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.encoder_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )
        self.decoder_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                num_heads,
                dim_feedforward,
                dropout,
                batch_first=True,
            ),
            num_encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model,
                num_heads,
                dim_feedforward,
                dropout,
                batch_first=True,
            ),
            num_decoder_layers,
        )

        self.linear_head = nn.Linear(d_model, vocab_size)

    def encode(self, input_ids, encoder_key_padding_mask=None):
        positions = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        )
        embeddings = self.encoder_embedding(input_ids) + self.position_embedding(
            positions
        ).unsqueeze(0)
        outputs = self.encoder(
            embeddings, src_key_padding_mask=encoder_key_padding_mask
        )
        return outputs

    def decode(
        self,
        input_ids,
        encoder_outputs,
        decoder_key_padding_mask=None,
        encoder_key_padding_mask=None,
    ):
        length = input_ids.size(1)

        positions = torch.arange(
            length,
            device=input_ids.device,
        )
        embeddings = self.decoder_embedding(input_ids) + self.position_embedding(
            positions
        ).unsqueeze(0)

        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=input_ids.device
        ).triu(diagonal=1)

        outputs = self.decoder(
            embeddings,
            encoder_outputs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_key_padding_mask,
            memory_key_padding_mask=encoder_key_padding_mask,
        )
        return outputs

    def decode_logits(
        self,
        input_ids,
        encoder_outputs,
        decoder_key_padding_mask=None,
        encoder_key_padding_mask=None,
        *,
        last_only: bool = False,
    ):
        """Decode and apply the output head to all positions or only the last."""
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
