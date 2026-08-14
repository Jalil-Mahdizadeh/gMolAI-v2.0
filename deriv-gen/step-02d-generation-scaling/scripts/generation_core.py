"""Frozen-model candidate-stream generation primitives for Step 2d."""

from __future__ import annotations

from typing import Sequence

import torch

from candidate_model import generate_beam_pool
from study_common import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def _top_p_probabilities(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1]")
    if top_p >= 1.0:
        return logits.softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probabilities = sorted_logits.softmax(dim=-1)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative > float(top_p)
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    return torch.zeros_like(sorted_logits).scatter(
        1, sorted_indices, sorted_logits.softmax(dim=-1)
    )


@torch.inference_mode()
def generate_seeded_sample_pool(
    model,
    condition: torch.Tensor,
    *,
    maximum_steps: int,
    draws: int,
    temperature: float,
    top_p: float,
    seeds: Sequence[int],
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched samples with an independent, immutable RNG stream per molecule."""
    if len(seeds) != len(condition):
        raise ValueError("One sample seed is required per conditioning vector")
    if draws < 1 or temperature <= 0.0:
        raise ValueError("Invalid stochastic generation parameters")
    if maximum_steps + 1 > model.maximum_positions:
        raise ValueError("Generation ceiling exceeds positional capacity")
    batch = len(condition)
    device = condition.device
    expanded = batch * int(draws)
    tokens = torch.full(
        (expanded, maximum_steps + 1), PAD_TOKEN, dtype=torch.long, device=device
    )
    tokens[:, 0] = BOS_TOKEN
    finished = torch.zeros(expanded, dtype=torch.bool, device=device)
    scores = torch.zeros(expanded, dtype=torch.float32, device=device)
    memory, bias = model.encode_condition(condition)
    memory = memory.repeat_interleave(draws, dim=0)
    bias = bias.repeat_interleave(draws, dim=0)
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) % (2**63 - 1))
        generators.append(generator)
    for step in range(maximum_steps):
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            logits = model.decode_prefix(tokens[:, : step + 1], memory, bias)[:, -1]
        logits = logits.float()
        base_log_probabilities = logits.log_softmax(dim=-1)
        sampling_logits = logits / float(temperature)
        sampling_logits[:, PAD_TOKEN] = -torch.inf
        sampling_logits[:, BOS_TOKEN] = -torch.inf
        probabilities = _top_p_probabilities(sampling_logits, float(top_p))
        pieces = []
        for position, generator in enumerate(generators):
            start = position * draws
            stop = start + draws
            pieces.append(
                torch.multinomial(
                    probabilities[start:stop],
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).squeeze(1)
            )
        sampled = torch.cat(pieces)
        selected_scores = base_log_probabilities.gather(
            1, sampled.unsqueeze(1)
        ).squeeze(1)
        sampled = torch.where(finished, torch.full_like(sampled, PAD_TOKEN), sampled)
        selected_scores = torch.where(
            finished, torch.zeros_like(selected_scores), selected_scores
        )
        tokens[:, step + 1] = sampled
        scores += selected_scores
        finished |= sampled.eq(EOS_TOKEN)
        if bool(finished.all()):
            break
    generated = tokens[:, 1:].view(batch, draws, maximum_steps)
    eos = generated.eq(EOS_TOKEN)
    positions = torch.arange(1, maximum_steps + 1, device=device).view(1, 1, -1)
    lengths = torch.where(
        eos, positions, torch.full_like(positions, maximum_steps + 1)
    ).amin(dim=-1).clamp_max(maximum_steps)
    return generated, scores.view(batch, draws), lengths


def beam_order(
    scores: torch.Tensor, lengths: torch.Tensor, length_penalty: float
) -> torch.Tensor:
    normalizer = torch.pow(
        (5.0 + lengths.float()) / 6.0, float(length_penalty)
    )
    adjusted = scores.float() / normalizer
    return torch.argsort(adjusted, dim=1, descending=True, stable=True)


def proportional_merge(
    beam: list[tuple[str, int]], sample: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """Deterministically interleave equal registered source quotas."""
    if len(beam) != len(sample):
        raise ValueError("Balanced merge requires equal source quotas")
    result: list[tuple[str, int]] = []
    for first, second in zip(beam, sample):
        result.extend((first, second))
    return result


__all__ = [
    "beam_order",
    "generate_beam_pool",
    "generate_seeded_sample_pool",
    "proportional_merge",
]
