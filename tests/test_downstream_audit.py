import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from gmolai_retrain.cli import build_parser
from gmolai_retrain.downstream import PreparedMoleculeNetDataset
from gmolai_retrain.downstream_audit import (
    _descriptor_matrix,
    _join_pretraining_rows,
)
from gmolai_retrain.exposure import _rank_exposure
from gmolai_retrain.util import atomic_write_csv


DESCRIPTOR_NAMES = [
    "qed",
    "MolWt",
    "NumValenceElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "BalabanJ",
    "LabuteASA",
    "TPSA",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "MolLogP",
    "MolMR",
]


def _hash(smiles):
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def _cfg():
    return {
        "data": {
            "hash_buckets": 1,
            "descriptor_columns": [str(index) for index in range(13)],
        },
        "_descriptors": {
            "features": [{"name": name} for name in DESCRIPTOR_NAMES]
        },
    }


def test_bucket_join_returns_exact_corpus_split_and_frozen_descriptors(tmp_path):
    first_smiles, second_smiles = "CC", "CCC"
    first_hash, second_hash = _hash(first_smiles), _hash(second_smiles)
    columns = {
        "molecule_hash": [first_hash],
        "canonical_smiles": [first_smiles],
        "split": ["train"],
    }
    for index in range(13):
        columns[f"d{index:02d}"] = [float(index + 1)]
    parquet = tmp_path / "bucket-0000.parquet"
    pq.write_table(pa.table(columns), parquet)
    prepared = PreparedMoleculeNetDataset(
        molecules=[Chem.MolFromSmiles(first_smiles), Chem.MolFromSmiles(second_smiles)],
        targets=np.asarray([0.0, 1.0]),
        scaffold_groups=np.asarray(["a", "b"], dtype=object),
        canonical_smiles=(first_smiles, second_smiles),
        molecule_hashes=(first_hash, second_hash),
        source_buckets=np.asarray([0, 0], dtype=np.int16),
        preparation={"molecules": 2, "scaffold_groups": 2},
    )
    result = _join_pretraining_rows(
        _cfg(),
        {"parquet_files": [str(parquet)]},
        {"fixture": prepared},
        include_descriptors=True,
    )["fixture"]

    assert result[0]["split"] == "train"
    assert result[0]["descriptors"] == [float(index + 1) for index in range(13)]
    assert result[1] is None


def test_descriptor_matrix_uses_the_frozen_13_feature_order():
    names, values = _descriptor_matrix(
        _cfg(), [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    )
    assert names == DESCRIPTOR_NAMES
    assert values.shape == (2, 13)
    assert np.isfinite(values).all()


def test_rank_exposure_counts_cycles_presentations_and_unique_graphs():
    shards = [{"graphs": 10, "path": "shard.pt", "split": "train"}]
    first_cycle = _rank_exposure(
        shards,
        seed=42,
        rank=0,
        world_size=1,
        cursor={"cycle": 0, "shard_position": 0, "graph_position": 7},
    )
    second_cycle = _rank_exposure(
        shards,
        seed=42,
        rank=0,
        world_size=1,
        cursor={"cycle": 1, "shard_position": 0, "graph_position": 3},
    )
    assert first_cycle["total_presentations"] == 7
    assert first_cycle["unique_graphs_presented"] == 7
    assert second_cycle["total_presentations"] == 13
    assert second_cycle["unique_graphs_presented"] == 10


def test_atomic_csv_artifacts_use_lf_line_endings(tmp_path):
    output = tmp_path / "audit.csv"
    atomic_write_csv(output, [{"name": "example", "value": 1}])

    assert output.read_bytes() == b"name,value\nexample,1\n"


def test_no_training_audit_commands_are_registered():
    overlap = build_parser().parse_args(
        [
            "audit-downstream-overlap",
            "--datasets-dir",
            "datasets",
            "--output",
            "overlap.json",
            "--summary-csv",
            "overlap.csv",
        ]
    )
    descriptor = build_parser().parse_args(
        [
            "benchmark-descriptor-control",
            "--datasets-dir",
            "datasets",
            "--reference-benchmark",
            "reference.json",
            "--output",
            "descriptor.json",
            "--summary-csv",
            "descriptor.csv",
        ]
    )
    exposure = build_parser().parse_args(
        [
            "audit-training-exposure",
            "--run-dir",
            "run",
            "--checkpoint",
            "checkpoints/step-000010000.pt",
            "--output",
            "exposure.json",
            "--summary-csv",
            "exposure.csv",
        ]
    )
    assert overlap.command == "audit-downstream-overlap"
    assert descriptor.command == "benchmark-descriptor-control"
    assert exposure.command == "audit-training-exposure"
    assert exposure.checkpoints == ["checkpoints/step-000010000.pt"]
