"""Decoder-only model conditioned on the immutable released gMolAI vector."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from study_common import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class ConditionalSmilesTransformer(nn.Module):
    """Autoregressive byte-SMILES decoder with condition at every layer."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.vocab_size = int(config["vocab_size"])
        self.d_model = int(config["d_model"])
        self.maximum_positions = int(config["maximum_positions"])
        self.condition_memory_tokens = int(config["condition_memory_tokens"])
        self.token_embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_TOKEN
        )
        self.position_embedding = nn.Embedding(
            self.maximum_positions, self.d_model
        )
        condition_dimensions = int(config["condition_dimensions"])
        self.condition_memory = nn.Sequential(
            nn.Linear(condition_dimensions, self.d_model * 2),
            nn.SiLU(),
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(
                self.d_model * 2,
                self.condition_memory_tokens * self.d_model,
            ),
        )
        self.condition_bias = nn.Sequential(
            nn.Linear(condition_dimensions, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.memory_position = nn.Parameter(
            torch.empty(self.condition_memory_tokens, self.d_model)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(config["attention_heads"]),
            dim_feedforward=int(config["feedforward_dimensions"]),
            dropout=float(config["dropout"]),
            activation=str(config["activation"]),
            batch_first=True,
            norm_first=bool(config["norm_first"]),
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=int(config["decoder_layers"]),
            norm=nn.LayerNorm(self.d_model),
        )
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        causal = torch.triu(
            torch.ones(
                self.maximum_positions,
                self.maximum_positions,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD_TOKEN].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_position, mean=0.0, std=0.02)

    def encode_condition(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.condition_memory(condition).view(
            len(condition), self.condition_memory_tokens, self.d_model
        )
        memory = memory + self.memory_position.unsqueeze(0)
        bias = self.condition_bias(condition)
        return memory, bias

    def decode_prefix(
        self,
        input_tokens: torch.Tensor,
        memory: torch.Tensor,
        condition_bias: torch.Tensor,
    ) -> torch.Tensor:
        length = int(input_tokens.shape[1])
        if length > self.maximum_positions:
            raise ValueError("Decoder prefix exceeds positional capacity")
        positions = torch.arange(length, device=input_tokens.device)
        hidden = (
            self.token_embedding(input_tokens) * math.sqrt(self.d_model)
            + self.position_embedding(positions).unsqueeze(0)
            + condition_bias.unsqueeze(1)
        )
        decoded = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=self.causal_mask[:length, :length],
            tgt_key_padding_mask=input_tokens.eq(PAD_TOKEN),
        )
        return self.output(decoded)

    def forward(
        self, input_tokens: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        memory, bias = self.encode_condition(condition)
        return self.decode_prefix(input_tokens, memory, bias)

    @torch.inference_mode()
    def generate(
        self,
        condition: torch.Tensor,
        *,
        maximum_steps: int,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        if maximum_steps + 1 > self.maximum_positions:
            raise ValueError("Generation ceiling exceeds positional capacity")
        batch = len(condition)
        tokens = torch.full(
            (batch, maximum_steps + 1),
            PAD_TOKEN,
            dtype=torch.long,
            device=condition.device,
        )
        tokens[:, 0] = BOS_TOKEN
        finished = torch.zeros(batch, dtype=torch.bool, device=condition.device)
        memory, bias = self.encode_condition(condition)
        for step in range(maximum_steps):
            with torch.autocast(
                device_type=condition.device.type,
                dtype=autocast_dtype,
                enabled=condition.device.type == "cuda",
            ):
                logits = self.decode_prefix(
                    tokens[:, : step + 1], memory, bias
                )[:, -1]
            next_token = logits.float().argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, PAD_TOKEN),
                next_token,
            )
            tokens[:, step + 1] = next_token
            finished |= next_token.eq(EOS_TOKEN)
            if bool(finished.all()):
                return tokens[:, 1 : step + 2]
        return tokens[:, 1:]

    @torch.inference_mode()
    def generate_beam(
        self,
        condition: torch.Tensor,
        *,
        maximum_steps: int,
        beam_width: int = 4,
        length_penalty: float = 0.6,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Deterministic batched beam search returning one sequence per vector."""
        if maximum_steps + 1 > self.maximum_positions:
            raise ValueError("Generation ceiling exceeds positional capacity")
        if beam_width < 2:
            raise ValueError("Beam width must be at least two")
        batch = len(condition)
        tokens = torch.full(
            (batch, beam_width, maximum_steps + 1),
            PAD_TOKEN,
            dtype=torch.long,
            device=condition.device,
        )
        tokens[:, :, 0] = BOS_TOKEN
        scores = torch.full(
            (batch, beam_width),
            -torch.inf,
            dtype=torch.float32,
            device=condition.device,
        )
        scores[:, 0] = 0.0
        finished = torch.zeros(
            (batch, beam_width),
            dtype=torch.bool,
            device=condition.device,
        )
        memory, bias = self.encode_condition(condition)
        memory = (
            memory.unsqueeze(1)
            .expand(-1, beam_width, -1, -1)
            .reshape(
                batch * beam_width,
                self.condition_memory_tokens,
                self.d_model,
            )
        )
        bias = (
            bias.unsqueeze(1)
            .expand(-1, beam_width, -1)
            .reshape(batch * beam_width, self.d_model)
        )
        for step in range(maximum_steps):
            prefix = tokens[:, :, : step + 1].reshape(
                batch * beam_width, step + 1
            )
            with torch.autocast(
                device_type=condition.device.type,
                dtype=autocast_dtype,
                enabled=condition.device.type == "cuda",
            ):
                logits = self.decode_prefix(prefix, memory, bias)[:, -1]
            log_probabilities = logits.float().log_softmax(dim=-1).view(
                batch, beam_width, self.vocab_size
            )
            active = ~finished
            log_probabilities[:, :, PAD_TOKEN] = -torch.inf
            log_probabilities[:, :, BOS_TOKEN] = -torch.inf
            if bool(finished.any()):
                log_probabilities = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(log_probabilities, -torch.inf),
                    log_probabilities,
                )
                finished_rows = finished.nonzero(as_tuple=False)
                log_probabilities[
                    finished_rows[:, 0],
                    finished_rows[:, 1],
                    PAD_TOKEN,
                ] = 0.0
            candidates = scores.unsqueeze(-1) + log_probabilities
            next_scores, flat_indices = torch.topk(
                candidates.view(batch, -1),
                k=beam_width,
                dim=1,
                largest=True,
                sorted=True,
            )
            parents = torch.div(
                flat_indices, self.vocab_size, rounding_mode="floor"
            )
            next_tokens = flat_indices.remainder(self.vocab_size)
            gather = parents.unsqueeze(-1).expand(
                -1, -1, maximum_steps + 1
            )
            tokens = tokens.gather(1, gather)
            finished = finished.gather(1, parents)
            tokens[:, :, step + 1] = next_tokens
            finished |= next_tokens.eq(EOS_TOKEN)
            scores = next_scores
            if bool(finished.all()):
                break
        generated = tokens[:, :, 1:]
        eos = generated.eq(EOS_TOKEN)
        positions = torch.arange(
            1, maximum_steps + 1, device=condition.device
        ).view(1, 1, -1)
        lengths = torch.where(
            eos,
            positions,
            torch.full_like(positions, maximum_steps + 1),
        ).amin(dim=-1).clamp_max(maximum_steps)
        normalizer = torch.pow(
            (5.0 + lengths.float()) / 6.0,
            float(length_penalty),
        )
        selected = (scores / normalizer).argmax(dim=1)
        rows = torch.arange(batch, device=condition.device)
        return generated[rows, selected]


def decoder_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
