#!/usr/bin/env python3
"""Fast component checks for the conditional byte-SMILES decoder."""

from __future__ import annotations

import json

import numpy as np
import torch

from decoder_model import ConditionalSmilesTransformer
from study_common import decode_tokens, token_matrix


def main() -> None:
    values = [
        "CCO",
        "N[C@@H](C)C(=O)O",
        "F/C=C\\F",
        "[NH3+]CC(=O)[O-]",
        "c1ccccc1Br",
    ]
    tokens = token_matrix(values, 128)
    decoded = [decode_tokens(row[1:]) for row in tokens]
    if [value for value, error in decoded] != values or any(
        error for value, error in decoded
    ):
        raise RuntimeError("Lossless byte-SMILES round trip failed")

    config = {
        "vocab_size": 131,
        "d_model": 32,
        "maximum_positions": 130,
        "condition_memory_tokens": 2,
        "condition_dimensions": 384,
        "attention_heads": 4,
        "feedforward_dimensions": 64,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_first": True,
        "decoder_layers": 2,
    }
    torch.manual_seed(20260814)
    model = ConditionalSmilesTransformer(config).eval()
    prefix = torch.from_numpy(tokens[:2, :-1]).long()
    first = model(prefix, torch.zeros((2, 384)))
    second = model(prefix, torch.ones((2, 384)))
    if first.shape != (2, prefix.shape[1], 131):
        raise RuntimeError("Decoder output shape is invalid")
    if np.allclose(
        first.detach().numpy(), second.detach().numpy(), rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("Decoder logits do not respond to conditioning")
    beam = model.generate_beam(
        torch.zeros((2, 384)),
        maximum_steps=8,
        beam_width=3,
        length_penalty=0.6,
        autocast_dtype=torch.float32,
    )
    if beam.shape != (2, 8):
        raise RuntimeError("Batched beam output shape is invalid")
    print(
        json.dumps(
            {
                "status": "passed",
                "lossless_examples": len(values),
                "stereochemical_examples": 2,
                "condition_changes_logits": True,
                "batched_beam_search": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
