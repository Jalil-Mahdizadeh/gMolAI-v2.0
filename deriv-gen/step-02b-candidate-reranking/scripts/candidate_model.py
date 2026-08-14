"""Inference-only candidate generators for the frozen Step-2 decoder."""

from __future__ import annotations

import torch

from common import ConditionalSmilesTransformer

from study_common import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


@torch.inference_mode()
def generate_beam_pool(
    model: ConditionalSmilesTransformer,
    condition: torch.Tensor,
    *,
    maximum_steps: int,
    beam_width: int,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all final beam hypotheses, cumulative scores, and token lengths."""
    if beam_width < 2:
        raise ValueError("beam_width must be at least two")
    if maximum_steps + 1 > model.maximum_positions:
        raise ValueError("Generation ceiling exceeds positional capacity")
    batch = len(condition)
    device = condition.device
    tokens = torch.full(
        (batch, beam_width, maximum_steps + 1),
        PAD_TOKEN,
        dtype=torch.long,
        device=device,
    )
    tokens[:, :, 0] = BOS_TOKEN
    scores = torch.full(
        (batch, beam_width), -torch.inf, dtype=torch.float32, device=device
    )
    scores[:, 0] = 0.0
    finished = torch.zeros(
        (batch, beam_width), dtype=torch.bool, device=device
    )
    memory, bias = model.encode_condition(condition)
    memory = (
        memory.unsqueeze(1)
        .expand(-1, beam_width, -1, -1)
        .reshape(batch * beam_width, model.condition_memory_tokens, model.d_model)
    )
    bias = (
        bias.unsqueeze(1)
        .expand(-1, beam_width, -1)
        .reshape(batch * beam_width, model.d_model)
    )
    for step in range(maximum_steps):
        prefix = tokens[:, :, : step + 1].reshape(batch * beam_width, step + 1)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            logits = model.decode_prefix(prefix, memory, bias)[:, -1]
        log_probabilities = logits.float().log_softmax(dim=-1).view(
            batch, beam_width, model.vocab_size
        )
        log_probabilities[:, :, PAD_TOKEN] = -torch.inf
        log_probabilities[:, :, BOS_TOKEN] = -torch.inf
        if bool(finished.any()):
            log_probabilities = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(log_probabilities, -torch.inf),
                log_probabilities,
            )
            ended = finished.nonzero(as_tuple=False)
            log_probabilities[ended[:, 0], ended[:, 1], PAD_TOKEN] = 0.0
        candidates = scores.unsqueeze(-1) + log_probabilities
        next_scores, flat_indices = torch.topk(
            candidates.view(batch, -1),
            k=beam_width,
            dim=1,
            largest=True,
            sorted=True,
        )
        parents = torch.div(
            flat_indices, model.vocab_size, rounding_mode="floor"
        )
        next_tokens = flat_indices.remainder(model.vocab_size)
        gather = parents.unsqueeze(-1).expand(-1, -1, maximum_steps + 1)
        tokens = tokens.gather(1, gather)
        finished = finished.gather(1, parents)
        tokens[:, :, step + 1] = next_tokens
        finished |= next_tokens.eq(EOS_TOKEN)
        scores = next_scores
        if bool(finished.all()):
            break
    generated = tokens[:, :, 1:]
    eos = generated.eq(EOS_TOKEN)
    positions = torch.arange(1, maximum_steps + 1, device=device).view(1, 1, -1)
    lengths = torch.where(
        eos,
        positions,
        torch.full_like(positions, maximum_steps + 1),
    ).amin(dim=-1).clamp_max(maximum_steps)
    return generated, scores, lengths


def _top_p_probabilities(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1]")
    if top_p >= 1.0:
        return logits.softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probabilities = sorted_logits.softmax(dim=-1)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    probabilities = torch.zeros_like(sorted_logits).scatter(
        1, sorted_indices, sorted_logits.softmax(dim=-1)
    )
    return probabilities


@torch.inference_mode()
def generate_sample_pool(
    model: ConditionalSmilesTransformer,
    condition: torch.Tensor,
    *,
    maximum_steps: int,
    draws: int,
    temperature: float,
    top_p: float,
    seed: int,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed-seed samples and their untempered decoder scores."""
    if draws < 1:
        raise ValueError("draws must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if maximum_steps + 1 > model.maximum_positions:
        raise ValueError("Generation ceiling exceeds positional capacity")
    batch = len(condition)
    device = condition.device
    expanded = batch * draws
    tokens = torch.full(
        (expanded, maximum_steps + 1),
        PAD_TOKEN,
        dtype=torch.long,
        device=device,
    )
    tokens[:, 0] = BOS_TOKEN
    finished = torch.zeros(expanded, dtype=torch.bool, device=device)
    scores = torch.zeros(expanded, dtype=torch.float32, device=device)
    memory, bias = model.encode_condition(condition)
    memory = memory.repeat_interleave(draws, dim=0)
    bias = bias.repeat_interleave(draws, dim=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) % (2**63 - 1))
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
        sampled = torch.multinomial(
            probabilities, num_samples=1, replacement=True, generator=generator
        ).squeeze(1)
        selected_scores = base_log_probabilities.gather(
            1, sampled.unsqueeze(1)
        ).squeeze(1)
        sampled = torch.where(
            finished, torch.full_like(sampled, PAD_TOKEN), sampled
        )
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
        eos,
        positions,
        torch.full_like(positions, maximum_steps + 1),
    ).amin(dim=-1).clamp_max(maximum_steps)
    return generated, scores.view(batch, draws), lengths
