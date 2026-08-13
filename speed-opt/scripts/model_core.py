#!/usr/bin/env python3
"""Frozen gMolAI loading and equivalent inference cores for speed tuning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gmolai_retrain.chem import featurize_molecule
from gmolai_retrain.config import apply_training_plan, load_config
from gmolai_retrain.representations import (
    _calibrator_expected_identity,
    _load_embedding_calibrator,
    load_saved_model,
)

from fast_graph import PackedBatch


CHECKPOINT_SHA256 = "02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e"
CALIBRATOR_SHA256 = "5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class FrozenBundle:
    model: nn.Module
    mean_cpu: np.ndarray
    scale_cpu: np.ndarray
    mean_gpu: torch.Tensor
    scale_gpu: torch.Tensor
    graph_dimensions: int
    device: torch.device
    metadata: dict[str, Any]


def load_frozen_bundle() -> FrozenBundle:
    config_path = REPOSITORY_ROOT / "configs/retrain.yaml"
    plan_path = REPOSITORY_ROOT / "configs/representation-pilot-mean-node-contrastive-001-desc050.yaml"
    run_dir = REPOSITORY_ROOT / "runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050"
    checkpoint_name = "checkpoints/step-000010000.pt"
    checkpoint_path = run_dir / checkpoint_name
    calibrator_path = run_dir / "representation-calibrator.pt"
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Promoted checkpoint SHA-256 changed")
    if sha256_file(calibrator_path) != CALIBRATOR_SHA256:
        raise RuntimeError("Promoted calibrator SHA-256 changed")

    cfg = load_config(config_path)
    apply_training_plan(cfg, plan_path)
    cfg["paths"]["run_dir"] = str(run_dir)
    cfg["experiment_name"] = run_dir.name
    if cfg["features"] != {
        "include_atom_chirality": True,
        "canonical_position_encoding_dim": 0,
    }:
        raise RuntimeError("Promoted feature configuration changed")

    device = torch.device("cuda:0")
    cfg, manifest, _, model, checkpoint = load_saved_model(
        cfg, checkpoint_name, device
    )
    model.eval().requires_grad_(False)
    if int(checkpoint["global_step"]) != 10_000:
        raise RuntimeError("Loaded checkpoint is not seed-42 step 10,000")
    mean, scale, calibration_metadata, observed_hash = _load_embedding_calibrator(
        calibrator_path,
        expected=_calibrator_expected_identity(cfg, manifest, checkpoint_path, checkpoint),
        dimensions=384,
    )
    if observed_hash != CALIBRATOR_SHA256:
        raise RuntimeError("Loaded calibrator identity changed")
    mean_cpu = mean.detach().cpu().numpy().astype(np.float32, copy=False)
    scale_cpu = scale.detach().cpu().numpy().astype(np.float32, copy=False)
    return FrozenBundle(
        model=model,
        mean_cpu=mean_cpu,
        scale_cpu=scale_cpu,
        mean_gpu=mean.detach().float().to(device),
        scale_gpu=scale.detach().float().to(device),
        graph_dimensions=int(model.graph_latent_dim),
        device=device,
        metadata={
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "calibrator_sha256": CALIBRATOR_SHA256,
            "checkpoint_step": int(checkpoint["global_step"]),
            "calibration_graphs": int(calibration_metadata["graphs"]),
        },
    )


def calibrate_cpu(raw: np.ndarray, bundle: FrozenBundle) -> np.ndarray:
    selected = (raw - bundle.mean_cpu) / bundle.scale_cpu
    selected[:, bundle.graph_dimensions :] *= np.float32(3.0)
    return np.asarray(selected, dtype=np.float32)


@torch.inference_mode()
def reference_encode(values: list[str], bundle: FrozenBundle) -> np.ndarray:
    graphs = []
    for value in values:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"gMolAI could not parse {value!r}")
        x, edge_index, edge_attr = featurize_molecule(
            molecule, include_chirality=True, position_dim=0
        )
        graphs.append(
            Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
            )
        )
    batch = Batch.from_data_list(graphs).to(bundle.device)
    node_z, graph_z = bundle.model.encode(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch
    )
    mean_node_z = global_mean_pool(node_z, batch.batch)
    raw = np.concatenate(
        (
            graph_z.detach().float().cpu().numpy(),
            mean_node_z.detach().float().cpu().numpy(),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return calibrate_cpu(raw, bundle)


class ReferenceRawCore(nn.Module):
    """Authoritative GPU computation, returning uncalibrated hybrid vectors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        node_z, graph_z = self.model.encode(x, edge_index, edge_attr, batch)
        return torch.cat((graph_z, global_mean_pool(node_z, batch)), dim=1)


class ReusedPoolRawCore(nn.Module):
    """Equivalent core that computes the node sum only once and reuses its mean."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.graph_readout = model.graph_readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        node_z = self.encoder(x, edge_index, edge_attr)
        total = global_add_pool(node_z, batch)
        maximum = global_max_pool(node_z, batch, size=total.shape[0])
        count = torch.bincount(batch, minlength=total.shape[0]).to(node_z.dtype).clamp_min(1)
        mean = total / count.unsqueeze(-1)
        normalized_total = total / count.sqrt().unsqueeze(-1)
        size = count.log1p().unsqueeze(-1)
        graph_z = self.graph_readout(
            torch.cat((mean, normalized_total, maximum, size), dim=-1)
        )
        return torch.cat((graph_z, mean), dim=1)



class ManualScatterRawCore(nn.Module):
    """Inline GINE message passing to expose the frozen encoder to compilation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.input_projection = model.encoder.input_projection
        self.convolutions = model.encoder.convolutions
        self.normalizations = model.encoder.normalizations
        self.output_projection = model.encoder.output_projection
        self.output_normalization = model.encoder.output_normalization
        self.graph_readout = model.graph_readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(x)
        source = edge_index[0]
        destination = edge_index[1]
        for convolution, normalization in zip(
            self.convolutions, self.normalizations, strict=True
        ):
            edge_hidden = convolution.lin(edge_attr)
            messages = (hidden.index_select(0, source) + edge_hidden).relu()
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, destination, messages)
            update = convolution.nn(
                aggregate + (1.0 + convolution.eps) * hidden
            )
            hidden = F.silu(normalization(hidden + update))
        node_z = self.output_normalization(self.output_projection(hidden))

        total = global_add_pool(node_z, batch)
        maximum = global_max_pool(node_z, batch, size=total.shape[0])
        count = torch.bincount(batch, minlength=total.shape[0]).to(node_z.dtype).clamp_min(1)
        mean = total / count.unsqueeze(-1)
        normalized_total = total / count.sqrt().unsqueeze(-1)
        size = count.log1p().unsqueeze(-1)
        graph_z = self.graph_readout(
            torch.cat((mean, normalized_total, maximum, size), dim=-1)
        )
        return torch.cat((graph_z, mean), dim=1)

def packed_to_device(
    packed: PackedBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (
        torch.from_numpy(packed.x),
        torch.from_numpy(packed.edge_index),
        torch.from_numpy(packed.edge_attr),
        torch.from_numpy(packed.batch),
    )
    if pin_memory:
        tensors = tuple(value.pin_memory() for value in tensors)
    return tuple(value.to(device, non_blocking=non_blocking) for value in tensors)


def raw_to_embedding(raw: torch.Tensor, bundle: FrozenBundle) -> np.ndarray:
    return calibrate_cpu(raw.detach().float().cpu().numpy(), bundle)

