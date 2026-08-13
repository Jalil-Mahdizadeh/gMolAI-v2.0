#!/usr/bin/env python3
"""Exact gMolAI graph features with lower Python and PyG collation overhead."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures


# These ordered categories are the immutable promoted gMolAI feature schema.
ATOM_TYPES = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "B", "Si")
FORMAL_CHARGES = (-3, -2, -1, 0, 1, 2, 3)
HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED")
DEGREES = (0, 1, 2, 3, 4, 5, 6)
TOTAL_HYDROGENS = (0, 1, 2, 3, 4)
CHIRAL_TAGS = ("CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW")
BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE")
BOND_STEREO = (
    "STEREONONE",
    "STEREOANY",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
)

NODE_DIM = 48
EDGE_DIM = 15

_ATOM_INDEX = {value: index for index, value in enumerate(ATOM_TYPES)}
_CHARGE_INDEX = {value: index for index, value in enumerate(FORMAL_CHARGES)}
_HYBRID_INDEX = {value: index for index, value in enumerate(HYBRIDIZATIONS)}
_DEGREE_INDEX = {value: index for index, value in enumerate(DEGREES)}
_HYDROGEN_INDEX = {value: index for index, value in enumerate(TOTAL_HYDROGENS)}
_CHIRAL_INDEX = {value: index for index, value in enumerate(CHIRAL_TAGS)}
_BOND_INDEX = {value: index for index, value in enumerate(BOND_TYPES)}
_STEREO_INDEX = {value: index for index, value in enumerate(BOND_STEREO)}


# BaseFeatures.fdef donor/acceptor definitions copied verbatim in meaning from
# RDKit 2025.09.3. Building a factory with only the two families avoids matching
# unrelated aromatic, ionizable, zinc-binding, and hydrophobic features.
_HBOND_FDEF = r"""
AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]
AtomType AmideN [$(N-C(=O))]
AtomType SulfonamideN [$([N;H0]S(=O)(=O))]
AtomType NDonor [$([Nv3](-C)(-C)-C)]
AtomType NDonor [$(n[n;H1]),$(nc[n;H1])]
AtomType ChalcDonor [O,S;H1;+0]
DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor}]
  Family Donor
  Weights 1
EndFeature
AtomType NAcceptor [n;+0;!X3;!$([n;H1](cc)cc)]
AtomType NAcceptor [$([N;H0]#[C&v4])]
AtomType NAcceptor [N&v3;H0;$(Nc)]
AtomType ChalcAcceptor [O;H0;v2;!$(O=N-*)]
Atomtype ChalcAcceptor [O;-;!$(*-N=O)]
Atomtype ChalcAcceptor [o;+0]
AtomType Hydroxyl [O;H1;v2]
AtomType HalogenAcceptor [F;$(F-[#6]);!$(FC[F,Cl,Br,I])]
DefineFeature SingleAtomAcceptor [{Hydroxyl},{ChalcAcceptor},{NAcceptor},{HalogenAcceptor}]
  Family Acceptor
  Weights 1
EndFeature
"""

_HBOND_FACTORY = None


def _hbond_factory():
    global _HBOND_FACTORY
    if _HBOND_FACTORY is None:
        _HBOND_FACTORY = ChemicalFeatures.BuildFeatureFactoryFromString(_HBOND_FDEF)
    return _HBOND_FACTORY


def _category(index: dict, value, other: int, label: str) -> int:
    try:
        return index[value]
    except KeyError:
        if other < 0:
            raise ValueError(f"Unsupported {label} value {value!r}") from None
        return other


def fast_featurize_molecule(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays exactly matching ``chem.featurize_molecule(..., True, 0)``."""

    atom_count = mol.GetNumAtoms()
    x = np.zeros((atom_count, NODE_DIM), dtype=np.float32)

    for feature in _hbond_factory().GetFeaturesForMol(mol):
        column = 27 if feature.GetFamily() == "Donor" else 28
        for atom_index in feature.GetAtomIds():
            x[int(atom_index), column] = 1.0

    for atom in mol.GetAtoms():
        row = atom.GetIdx()
        x[row, _category(_ATOM_INDEX, atom.GetSymbol(), -1, "atom type")] = 1.0
        x[row, 12 + _category(_CHARGE_INDEX, atom.GetFormalCharge(), 7, "charge")] = 1.0
        x[row, 20 + _category(
            _HYBRID_INDEX, str(atom.GetHybridization()), 6, "hybridization"
        )] = 1.0
        x[row, 29] = float(atom.GetIsAromatic())
        x[row, 30 + _category(_DEGREE_INDEX, atom.GetTotalDegree(), 7, "degree")] = 1.0
        x[row, 38 + _category(
            _HYDROGEN_INDEX, atom.GetTotalNumHs(), 5, "total hydrogens"
        )] = 1.0
        x[row, 44 + _category(
            _CHIRAL_INDEX, str(atom.GetChiralTag()), 3, "chirality"
        )] = 1.0

    bond_count = mol.GetNumBonds()
    edge_count = 2 * bond_count
    edge_index = np.empty((2, edge_count), dtype=np.int64)
    edge_attr = np.zeros((edge_count, EDGE_DIM), dtype=np.float32)
    for bond_number, bond in enumerate(mol.GetBonds()):
        first = 2 * bond_number
        second = first + 1
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index[:, first] = (begin, end)
        edge_index[:, second] = (end, begin)
        bond_column = _category(_BOND_INDEX, str(bond.GetBondType()), 5, "bond type")
        stereo_column = _category(_STEREO_INDEX, str(bond.GetStereo()), 6, "bond stereo")
        edge_attr[first, bond_column] = 1.0
        edge_attr[first, 6] = float(bond.IsInRing())
        edge_attr[first, 7] = float(bond.GetIsConjugated())
        edge_attr[first, 8 + stereo_column] = 1.0
        edge_attr[second] = edge_attr[first]
    return x, edge_index, edge_attr


@dataclass(slots=True)
class PackedBatch:
    start: int
    graph_count: int
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    batch: np.ndarray
    node_counts: np.ndarray


def pack_smiles_task(task: tuple[int, list[str]]) -> PackedBatch:
    """Parse, featurize, and directly pack one ordered SMILES batch."""

    start, smiles = task
    features: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"gMolAI could not parse {value!r}")
        features.append(fast_featurize_molecule(molecule))

    node_counts = np.asarray([len(item[0]) for item in features], dtype=np.int64)
    edge_counts = np.asarray([item[1].shape[1] for item in features], dtype=np.int64)
    total_nodes = int(node_counts.sum())
    total_edges = int(edge_counts.sum())
    x = np.empty((total_nodes, NODE_DIM), dtype=np.float32)
    edge_index = np.empty((2, total_edges), dtype=np.int64)
    edge_attr = np.empty((total_edges, EDGE_DIM), dtype=np.float32)
    batch = np.repeat(np.arange(len(features), dtype=np.int64), node_counts)

    node_offset = 0
    edge_offset = 0
    for graph_x, graph_edge_index, graph_edge_attr in features:
        next_node = node_offset + len(graph_x)
        next_edge = edge_offset + graph_edge_index.shape[1]
        x[node_offset:next_node] = graph_x
        edge_index[:, edge_offset:next_edge] = graph_edge_index + node_offset
        edge_attr[edge_offset:next_edge] = graph_edge_attr
        node_offset = next_node
        edge_offset = next_edge

    return PackedBatch(
        start=start,
        graph_count=len(features),
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        node_counts=node_counts,
    )


def initialize_worker() -> None:
    """Warm immutable worker state and prevent nested CPU oversubscription."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _hbond_factory()


def tasks(values: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]

